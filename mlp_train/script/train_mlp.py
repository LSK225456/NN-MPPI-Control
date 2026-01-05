#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
import torch.jit
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import sys
import rospy

# ==========================================
# 1. 配置管理模块 (Configuration)
# ==========================================
class ConfigHandler:
    """
    负责从ROS参数服务器加载配置，并验证环境路径。
    """
    def __init__(self):
        # 初始化节点
        rospy.init_node('mlp_trainer_node', anonymous=False)
        
        # // 12.02: 路径配置，兼容 launch 文件传入的参数
        self.data_dir = rospy.get_param("~dataset/input_dir", "/home/lsk1804/lsk_graduate/lsk_ws/src/mlp_train/data/train_data/")
        self.output_dir = rospy.get_param("~dataset/output_dir", "/home/lsk1804/lsk_graduate/lsk_ws/src/mlp_train/data/model")
        self.model_name = rospy.get_param("~dataset/model_name", "best_mlp_model")
        
        # 数据划分与训练配置
        self.split_ratio = rospy.get_param("~split/ratio", [0.8, 0.1, 0.1]) # Train, Val, Test
        self.batch_size = rospy.get_param("~training/batch_size", 256)
        self.lr = rospy.get_param("~training/learning_rate", 1e-3)
        self.epochs = rospy.get_param("~training/epochs", 500)
        self.patience = rospy.get_param("~training/early_stop_patience", 30)
        
        # 设备选择
        self.device = torch.device("cpu") # 默认使用CPU，确保导出的TorchScript通用性
        
        self._validate_paths()

    def _validate_paths(self):
        """
        验证输入输出路径的有效性。
        """
        if not os.path.exists(self.data_dir):
            rospy.logerr(f"数据集目录不存在: {self.data_dir}")
            sys.exit(1)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            rospy.loginfo(f"已创建输出目录: {self.output_dir}")



# ==========================================
# 2. 数据集管理模块 (Dataset Management)
# ==========================================
class DatasetManager:
    """
    负责数据的加载、清洗、归一化及DataLoader的构建。
    针对端到端训练：Input [N, 5], Output [N, 2]
    """
    def __init__(self, config):
        self.cfg = config
        # 统计量初始化
        self.x_mean = None
        self.x_std = None
        self.y_mean = None
        self.y_std = None

    def load_and_process(self):
        """
        加载所有 .pt 数据文件，进行标准化处理并划分为训练/验证/测试集。
        输出: train_loader, val_loader, test_data(原始未归一化Y)
        """
        # 1. 文件搜索
        search_path = os.path.join(self.cfg.data_dir, "*.pt")
        files = glob.glob(search_path)
        if not files:
            rospy.logerr(f"未找到数据文件: {search_path}")
            sys.exit(1)
        
        rospy.loginfo(f"共发现 {len(files)} 个数据文件，开始合并处理...")

        # 2. 数据合并
        x_list, y_list = [], []
        for f in files:
            try:
                # map_location确保在CPU上加载
                data = torch.load(f, map_location='cpu')
                # // 12.02: 假设数据集中 'x' 为 [v, w, cmd_v, cmd_w, dt], 'y' 为 [next_v, next_w]
                x_list.append(data['x'])
                y_list.append(data['y'])
            except Exception as e:
                rospy.logwarn(f"无法读取文件 {f}: {e}")
        
        if not x_list:
            rospy.logerr("有效数据为空，程序退出。")
            sys.exit(1)

        X_all = torch.cat(x_list, dim=0).float()
        Y_all = torch.cat(y_list, dim=0).float()

        #  维度安全检查,确保输入是 [v, w, cmd_v, cmd_w, dt] (5维),确保输出是 [next_v, next_w] (2维)
        if X_all.shape[1] != 7:
            rospy.logerr(f"输入数据维度错误！期望 5, 实际 {X_all.shape[1]}")
            sys.exit(1)
        if Y_all.shape[1] != 2:
            rospy.logerr(f"输出数据维度错误！期望 2 (端到端), 实际 {Y_all.shape[1]}")
            sys.exit(1)
        
        total_samples = X_all.shape[0]
        rospy.loginfo(f"数据加载完成 | 样本数: {total_samples} | 输入维度: 7 | 输出维度: 2")
        rospy.loginfo(f"输入维度: {X_all.shape[1]}, 输出维度: {Y_all.shape[1]}")

        # 3. 数据集划分 (Shuffle & Split)
        indices = torch.randperm(total_samples)
        
        r_train, r_val, r_test = self.cfg.split_ratio
        n_train = int(total_samples * r_train)
        n_val = int(total_samples * r_val)
        
        idx_train = indices[:n_train]
        idx_val = indices[n_train : n_train + n_val]
        idx_test = indices[n_train + n_val :]

        # 4. Z-Score 标准化 (仅使用训练集统计量)
        # // 12.02: 即使是端到端输出，为了训练稳定性，建议对 Y 也进行归一化
        rospy.loginfo("正在计算统计特性并执行标准化...")
        self.x_mean = X_all[idx_train].mean(dim=0)
        self.x_std = X_all[idx_train].std(dim=0) + 1e-6 # 避免除零
        
        self.y_mean = Y_all[idx_train].mean(dim=0)
        self.y_std = Y_all[idx_train].std(dim=0) + 1e-6

        # 应用标准化
        X_norm = (X_all - self.x_mean) / self.x_std
        Y_norm = (Y_all - self.y_mean) / self.y_std

        # 5. 构建 DataLoader
        train_loader = self._create_loader(X_norm[idx_train], Y_norm[idx_train], shuffle=True)
        val_loader = self._create_loader(X_norm[idx_val], Y_norm[idx_val], shuffle=False)
        
        # 测试集：保留归一化的X用于推理，保留原始Y用于真实误差评估
        test_data = (X_norm[idx_test], Y_all[idx_test]) 

        return train_loader, val_loader, test_data

    def _create_loader(self, x, y, shuffle):
        """
        辅助函数：创建 PyTorch DataLoader
        """
        ds = TensorDataset(x, y)
        return DataLoader(ds, batch_size=self.cfg.batch_size, shuffle=shuffle)

    def save_stats(self, path):
        """
        保存标准化参数，供C++推理端加载使用。
        """
        stats = {
            'x_mean': self.x_mean, 'x_std': self.x_std,
            'y_mean': self.y_mean, 'y_std': self.y_std
        }
        torch.save(stats, path)
        rospy.loginfo(f"标准化统计参数已保存: {path}")

# ==========================================
# 3. 神经网络模型 (Model Architecture)
# ==========================================
class EndToEndDynamicsModel(nn.Module):
    """
    端到端车辆动力学模型
    输入: [v, w, cmd_v, cmd_w, dt] (5维)
    输出: [next_v, next_w] (2维)
    """
    def __init__(self, input_dim=7, output_dim=2):
        super(EndToEndDynamicsModel, self).__init__()
        
        # // 12.02: 设计深层MLP结构，适配端到端映射的非线性复杂度
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Tanh(),
            nn.Dropout(0.1),
            
            nn.Linear(128, 256),
            nn.Tanh(),
            nn.Dropout(0.1),
            
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Dropout(0.1),
            
            nn.Linear(128, 64),
            nn.Tanh(),
            # 最后一层直接输出
            nn.Linear(64, output_dim)
        )
        self._init_weights()
        

    def _init_weights(self):
        """
        Kaiming 初始化，有助于深层网络收敛
        """
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.net(x)



# ==========================================
# 4. 训练引擎 (Training Engine)
# ==========================================
class ModelTrainer:
    """
    负责模型的训练、验证、早停及模型导出。
    """
    def __init__(self, model, config):
        self.model = model.to(config.device)
        self.cfg = config
        
        # 优化器: AdamW (带权重衰减，泛化性更好)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=config.lr, weight_decay=1e-3)
        
        # 损失函数: MSE (均方误差)
        self.criterion = nn.MSELoss()
        
        # 学习率调度: 余弦退火
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2
        )
        
        self.history = {'train_loss': [], 'val_loss': []}

    def train(self, train_loader, val_loader):
        """
        执行完整训练循环，包含早停机制。
        """
        best_val_loss = float('inf')
        patience_count = 0

        rospy.loginfo("开始训练流程...")
        
        for epoch in range(self.cfg.epochs):
            # --- Training ---
            self.model.train()
            train_loss_acc = 0.0
            for x, y in train_loader:
                x, y = x.to(self.cfg.device), y.to(self.cfg.device)
                
                self.optimizer.zero_grad()
                pred = self.model(x)
                loss = self.criterion(pred, y)
                loss.backward()
                self.optimizer.step()
                
                train_loss_acc += loss.item()
            
            avg_train = train_loss_acc / len(train_loader)
            self.history['train_loss'].append(avg_train)

            # --- Validation ---
            self.model.eval()
            val_loss_acc = 0.0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(self.cfg.device), y.to(self.cfg.device)
                    pred = self.model(x)
                    loss = self.criterion(pred, y)
                    val_loss_acc += loss.item()
            
            avg_val = val_loss_acc / len(val_loader)
            self.history['val_loss'].append(avg_val)

            # 更新学习率
            self.scheduler.step()

            # 日志输出
            if (epoch + 1) % 10 == 0:
                rospy.loginfo(f"Epoch [{epoch+1}/{self.cfg.epochs}] | Train Loss: {avg_train:.6f} | Val Loss: {avg_val:.6f}")

            # --- Early Stopping Check ---
            if avg_val < best_val_loss:
                best_val_loss = avg_val
                patience_count = 0
                # 保存最佳权重临时文件
                torch.save(self.model.state_dict(), os.path.join(self.cfg.output_dir, "best_checkpoint.pth"))
            else:
                patience_count += 1
                if patience_count >= self.cfg.patience:
                    rospy.logwarn(f"验证集Loss不再下降，触发早停。停止于 Epoch {epoch+1}")
                    break
        
        # 恢复最佳模型权重
        self.model.load_state_dict(torch.load(os.path.join(self.cfg.output_dir, "best_checkpoint.pth")))
        rospy.loginfo("训练完成，已加载最佳模型参数。")

    def export_script_model(self, sample_input):
        """
        将模型导出为 TorchScript (.pt) 格式，供 C++ LibTorch 调用。
        """
        self.model.eval()
        try:
            # // 12.02: Tracing 必须在 CPU 上进行以保证兼容性
            example = sample_input[0:1].to('cpu') 
            self.model.to('cpu')
            
            traced_module = torch.jit.trace(self.model, example)
            save_path = os.path.join(self.cfg.output_dir, f"{self.cfg.model_name}.pt")
            traced_module.save(save_path)
            
            rospy.loginfo(f"模型导出成功: {save_path}")
        except Exception as e:
            rospy.logerr(f"模型导出失败: {e}")




# ==========================================
# 5. 结果可视化与评估 (Visualization)
# ==========================================
class ResultVisualizer:
    """
    负责绘制训练曲线及测试集评估报告。
    """
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def plot_loss(self, history):
        plt.figure(figsize=(10, 5))
        plt.plot(history['train_loss'], label='Train')
        plt.plot(history['val_loss'], label='Val')
        plt.title('Loss Curve')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, "loss_curve.png"))
        plt.close()

    def evaluate(self, model, test_data, stats):
        """
        对比 神经网络(NN) 与 理想运动学模型(Kinematics Baseline) 的性能
        """
        x_norm, y_raw_truth = test_data
        model.eval()
        model.to('cpu')
        
        # 1. 神经网络推理
        with torch.no_grad():
            pred_norm = model(x_norm)
        
        # 2. 反归一化 NN 预测值
        y_mean = stats['y_mean']
        y_std = stats['y_std']
        pred_raw_nn = pred_norm * y_std + y_mean
        
        # 3. 获取运动学基准 (Ideal Kinematics)
        # 逻辑：下一时刻速度 = 当前时刻的控制指令 (Ideal Response)
        # 需要先反归一化输入 x 得到原始控制指令
        x_mean = stats['x_mean']
        x_std = stats['x_std']
        
        # X_raw = X_norm * std + mean
        # 输入维度定义: [v, w, cmd_v, cmd_w, dt]
        x_raw = x_norm * x_std + x_mean
        
        # 提取控制指令 [cmd_v, cmd_w] 作为基准预测
        # cmd_v 是 idx 2, cmd_w 是 idx 3
        pred_raw_baseline = x_raw[:, 2:4] 
        
        # 4. 计算误差 (MAE)
        # NN Error
        mae_nn = torch.mean(torch.abs(y_raw_truth - pred_raw_nn), dim=0)
        # Baseline Error
        mae_base = torch.mean(torch.abs(y_raw_truth - pred_raw_baseline), dim=0)
        
        # 5. 计算提升百分比
        # (Base - NN) / Base * 100
        improvement = (mae_base - mae_nn) / (mae_base + 1e-9) * 100

        # 6. 打印对比表格
        rospy.loginfo("\n" + "="*90)
        rospy.loginfo(f"{'Performance Comparison: NN Model vs Kinematic Baseline (Test Set)':^90}")
        rospy.loginfo("="*90)
        rospy.loginfo(f"{'Dimension':<20} | {'Baseline MAE':<15} | {'NN Model MAE':<15} | {'Improvement':<12}")
        rospy.loginfo("-" * 90)
        
        labels = ['Velocity (m/s)', 'Omega (rad/s)']
        
        for i in range(2):
            rospy.loginfo(f"{labels[i]:<20} | {mae_base[i]:.6f}        | {mae_nn[i]:.6f}        | {improvement[i]:.2f}%")
            
        rospy.loginfo("-" * 90)
        rospy.loginfo("注: Baseline MAE 基于假设 'Next_State = Control_Command' (理想无延迟响应)")
        rospy.loginfo("="*90 + "\n")

        # 7. 绘图对比
        self._plot_compare(y_raw_truth, pred_raw_nn, pred_raw_baseline)

    def _plot_compare(self, y_true, y_nn, y_base):
        samples = min(200, y_true.shape[0])
        fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
        labels = ['Velocity (m/s)', 'Omega (rad/s)']
        
        for i in range(2):
            # 真实值 (黑色实线)
            axes[i].plot(y_true[:samples, i].numpy(), 'k-', linewidth=1.5, alpha=0.7, label='Ground Truth')
            
            # 基准模型 (蓝色点线) - 代表控制指令
            axes[i].plot(y_base[:samples, i].numpy(), 'b:', linewidth=1.5, alpha=0.8, label='Kinematic Baseline (Cmd)')
            
            # 神经网络 (红色虚线)
            axes[i].plot(y_nn[:samples, i].numpy(), 'r--', linewidth=1.5, alpha=0.9, label='NN Prediction')
            
            axes[i].set_ylabel(labels[i])
            axes[i].legend(loc='upper right')
            axes[i].grid(True, linestyle='--', alpha=0.5)
            axes[i].set_title(f"{labels[i]} Tracking Performance")
        
        axes[1].set_xlabel('Sample Index')
        plt.suptitle('Prediction Comparison: Truth vs NN vs Baseline')
        
        save_path = os.path.join(self.output_dir, "test_prediction_comparison.png")
        plt.savefig(save_path)
        plt.close()
        rospy.loginfo(f"对比曲线图已保存: {save_path}")

# ==========================================
# 主程序入口
# ==========================================
def main():
    # 1. 配置加载
    cfg = ConfigHandler()
    
    # 2. 数据准备
    dm = DatasetManager(cfg)
    train_loader, val_loader, test_data = dm.load_and_process()
    
    # 保存统计参数至 output_dir，务必确保此文件被妥善保存，推理需要
    dm.save_stats(os.path.join(cfg.output_dir, "norm_stats.pt"))

    # 3. 模型初始化
    model = EndToEndDynamicsModel(input_dim=7, output_dim=2)
    
    # 4. 模型训练
    trainer = ModelTrainer(model, cfg)
    trainer.train(train_loader, val_loader)
    
    # 5. 导出部署模型
    trainer.export_script_model(test_data[0])

    # 6. 评估与可视化
    viz = ResultVisualizer(cfg.output_dir)
    viz.plot_loss(trainer.history)

    stats = {
        'x_mean': dm.x_mean, 'x_std': dm.x_std,
        'y_mean': dm.y_mean, 'y_std': dm.y_std
    }
    viz.evaluate(model, test_data, stats)




if __name__ == "__main__":
    main()