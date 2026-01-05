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
import argparse
import rospy

# ==========================================
# 1. 配置管理模块
# ==========================================
class ConfigHandler:
    def __init__(self):
        rospy.init_node('mlp_trainer_node', anonymous=False)
        
        # 路径配置
        self.data_dir = rospy.get_param("~dataset/input_dir", "/home/lsk1804/lsk_graduate/lsk_ws/src/mlp_train/data/train_data/")
        self.output_dir = rospy.get_param("~dataset/output_dir", "/home/lsk1804/lsk_graduate/lsk_ws/src/mlp_train/data/model")
        self.model_name = rospy.get_param("~dataset/model_name", "deep_dynamics_model")
        
        # 数据划分配置
        self.split_ratio = rospy.get_param("~split/ratio", [0.7, 0.2, 0.1]) # Train, Val, Test
        self.batch_size = rospy.get_param("~training/batch_size", 256)
        
        # 训练超参数
        self.lr = rospy.get_param("~training/learning_rate", 1e-3)
        self.epochs = rospy.get_param("~training/epochs", 500)
        self.patience = rospy.get_param("~training/early_stop_patience", 20)
        self.device = torch.device("cpu")                   # 明确指定使用 CPU (符合你的环境)

        self._validate_paths()

    def _validate_paths(self):
        if not os.path.exists(self.data_dir):
            print("self.data_dir:",self.data_dir)
            rospy.logerr(f"数据集目录不存在: {self.data_dir}")
            sys.exit(1)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)



# ==========================================
# 2. 数据集管理模块
# ==========================================
class DatasetManager:
    def __init__(self, config):
        self.cfg = config
        self.x_mean = None
        self.x_std = None
        self.y_mean = None
        self.y_std = None

    def load_and_process(self):
        """
        功能：遍历文件夹，加载所有PT文件，合并，归一化，划分
        """
        # 1. 寻找所有 .pt 文件
        search_path = os.path.join(self.cfg.data_dir, "*.pt")
        files = glob.glob(search_path)
        if not files:
            rospy.logerr(f"在 {self.cfg.data_dir} 未找到 .pt 文件")
            sys.exit(1)
        
        rospy.loginfo(f"发现 {len(files)} 个数据文件，正在合并...")

        # 2. 合并数据
        x_list, y_list = [], []
        for f in files:
            try:
                data = torch.load(f, map_location='cpu')
                x_list.append(data['x'])
                y_list.append(data['y'])
            except Exception as e:
                rospy.logwarn(f"跳过损坏文件 {f}: {e}")
        
        X_all = torch.cat(x_list, dim=0)
        Y_all = torch.cat(y_list, dim=0)
        
        total_samples = X_all.shape[0]
        rospy.loginfo(f"总样本数: {total_samples}")

        # 3. 随机打乱索引
        indices = torch.randperm(total_samples)
        
        # 4. 计算划分点
        r_train, r_val, r_test = self.cfg.split_ratio
        n_train = int(total_samples * r_train)
        n_val = int(total_samples * r_val)
        
        idx_train = indices[:n_train]
        idx_val = indices[n_train : n_train + n_val]
        idx_test = indices[n_train + n_val :]

        # 5. 标准化 (仅基于训练集计算 Mean/Std，防止数据泄露)
        rospy.loginfo("正在计算统计量并执行 Z-Score 标准化...")
        self.x_mean = X_all[idx_train].mean(dim=0)
        self.x_std = X_all[idx_train].std(dim=0) + 1e-6 # 防止除0
        
        self.y_mean = Y_all[idx_train].mean(dim=0)
        self.y_std = Y_all[idx_train].std(dim=0) + 1e-6

        # 应用标准化
        X_norm = (X_all - self.x_mean) / self.x_std
        Y_norm = (Y_all - self.y_mean) / self.y_std

        # 6. 封装为 DataLoader
        train_loader = self._create_loader(X_norm[idx_train], Y_norm[idx_train], shuffle=True)
        val_loader = self._create_loader(X_norm[idx_val], Y_norm[idx_val], shuffle=False)
        test_data = (X_norm[idx_test], Y_all[idx_test]) # 测试集保留 Y 的原始值用于评估真实误差

        return train_loader, val_loader, test_data

    def _create_loader(self, x, y, shuffle):
        ds = TensorDataset(x, y)
        return DataLoader(ds, batch_size=self.cfg.batch_size, shuffle=shuffle)

    def save_stats(self, path):
        """保存标准化参数，推理时必须使用"""
        stats = {
            'x_mean': self.x_mean, 'x_std': self.x_std,
            'y_mean': self.y_mean, 'y_std': self.y_std
        }
        torch.save(stats, path)
        rospy.loginfo(f"统计参数已保存至: {path}")

# ==========================================
# 3. 网络架构模块 (高性能深层版)
# ==========================================
class DeepDynamicsModel(nn.Module):
    def __init__(self, input_dim=5, output_dim=5):
        super(DeepDynamicsModel, self).__init__()
        # 架构: [5 -> 64 -> 64 -> 5]
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Dropout(0.2),  # [新增] 丢弃 10% 的连接
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Dropout(0.1),  # [新增]
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, output_dim)
        )
        
        # 权重初始化
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.net(x)

# ==========================================
# 4. 模型训练模块
# ==========================================
class ModelTrainer:
    def __init__(self, model, config):
        self.model = model.to(config.device)
        self.cfg = config
        
        # 优化器: AdamW
        self.optimizer = optim.AdamW(self.model.parameters(), lr=config.lr, weight_decay=1e-2)
        
        # 损失函数: MSE (因为是清洗过的数据，追求高性能)
        self.criterion = nn.MSELoss()
        
        # 学习率调度: 余弦退火
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2
        )
        
        self.history = {'train_loss': [], 'val_loss': []}

    def train(self, train_loader, val_loader):
        best_val_loss = float('inf')
        patience_counter = 0

        rospy.loginfo("开始训练...")
        
        for epoch in range(self.cfg.epochs):
            # --- 训练阶段 ---
            self.model.train()
            train_loss_sum = 0
            for x, y in train_loader:
                x, y = x.to(self.cfg.device), y.to(self.cfg.device)
                
                self.optimizer.zero_grad()
                pred = self.model(x)
                loss = self.criterion(pred, y)
                loss.backward()
                self.optimizer.step()
                
                train_loss_sum += loss.item()
            
            avg_train_loss = train_loss_sum / len(train_loader)
            self.history['train_loss'].append(avg_train_loss)

            # --- 验证阶段 ---
            self.model.eval()
            val_loss_sum = 0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(self.cfg.device), y.to(self.cfg.device)
                    pred = self.model(x)
                    loss = self.criterion(pred, y)
                    val_loss_sum += loss.item()
            
            avg_val_loss = val_loss_sum / len(val_loader)
            self.history['val_loss'].append(avg_val_loss)

            # --- 调度器更新 ---
            self.scheduler.step()

            # --- 日志与早停 ---
            if (epoch + 1) % 10 == 0:
                rospy.loginfo(f"Epoch [{epoch+1}/{self.cfg.epochs}] Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # 保存最佳权重的临时状态
                torch.save(self.model.state_dict(), os.path.join(self.cfg.output_dir, "best_checkpoint.pth"))
            else:
                patience_counter += 1
                if patience_counter >= self.cfg.patience:
                    rospy.logwarn(f"早停触发! 在 Epoch {epoch+1} 停止.")
                    break
        
        # 加载最佳模型
        self.model.load_state_dict(torch.load(os.path.join(self.cfg.output_dir, "best_checkpoint.pth")))
        rospy.loginfo("训练结束，已加载最佳模型权重。")

    def export_model(self, sample_input):
        """
        核心功能：导出 TorchScript (.pt) 模型
        这才是 MPPI C++ 接口真正需要的格式
        """
        self.model.eval()
        # 创建 JIT Trace
        # 注意：这里需要在 CPU 上 trace
        example_input = sample_input[0:1].to(self.cfg.device)
        
        try:
            traced_script_module = torch.jit.trace(self.model, example_input)
            
            save_path = os.path.join(self.cfg.output_dir, f"{self.cfg.model_name}.pt")
            traced_script_module.save(save_path)
            
            rospy.loginfo(f"SUCCESS: 模型已编译并保存为 TorchScript 格式: {save_path}")
            rospy.loginfo("该文件可直接用于 C++ LibTorch 加载或 Python 部署。")
        except Exception as e:
            rospy.logerr(f"模型导出失败: {e}")

# ==========================================
# 5. 结果可视化模块
# ==========================================
class ResultVisualizer:
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def plot_loss(self, history):
        plt.figure(figsize=(10, 5))
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Val Loss')
        plt.title('Training Dynamics')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, "loss_curve.png"))
        plt.close()

    def evaluate_test_set(self, model, test_data, stats):
        """
        评估并在测试集上绘图
        test_data: (X_norm, Y_raw)
        stats: 用于反归一化预测值
        """
        x_norm, y_raw = test_data
        model.eval()
        
        with torch.no_grad():
            # 预测 (输出的是归一化的残差)
            pred_norm = model(x_norm)
        
        # 反归一化预测值
        y_mean = stats['y_mean']
        y_std = stats['y_std']
        pred_raw = pred_norm * y_std + y_mean
        
        # Baseline Error (纯运动学误差): 即真实残差本身的绝对值均值 (因为运动学模型默认预测残差为0)
        mae_baseline = torch.mean(torch.abs(y_raw), dim=0)
        
        # Model Error (预测修正后误差): 真实残差 - 预测残差 = 剩余误差
        mae_model = torch.mean(torch.abs(y_raw - pred_raw), dim=0)
        
        # 计算提升百分比: (Before - After) / Before * 100
        improvement = (mae_baseline - mae_model) / (mae_baseline + 1e-9) * 100

        # // 11.26: 打印对比表格
        rospy.loginfo("\n" + "="*80)
        rospy.loginfo(f"{'Metric Analysis (Test Set)':^80}")
        rospy.loginfo("="*80)
        rospy.loginfo(f"{'Dimension':<15} | {'Baseline MAE':<15} | {'Model MAE':<15} | {'Improvement':<10}")
        rospy.loginfo("-" * 80)
        
        labels = ['Longitudinal', 'Lateral', 'Theta', 'Velocity', 'Omega']
        units  = ['m', 'm', 'rad', 'm/s', 'rad/s']

        for i in range(5):
            dim_str = f"{labels[i]} ({units[i]})"
            rospy.loginfo(f"{dim_str:<15} | {mae_baseline[i]:.4f}          | {mae_model[i]:.4f}          | {improvement[i]:.2f}%")
        
        rospy.loginfo("="*80 + "\n")

        # 绘图：取前 200 个样本对比
        samples = 200
        if y_raw.shape[0] < samples: samples = y_raw.shape[0]
        
        fig, axes = plt.subplots(5, 1, figsize=(10, 15), sharex=True)
        labels = ['Longitudinal', 'Lateral', 'Theta', 'Velocity', 'Omega']
        
        for i in range(5):
            axes[i].plot(y_raw[:samples, i].numpy(), 'k-', alpha=0.6, label='Ground Truth (Kinematic Error)')
            axes[i].plot(pred_raw[:samples, i].numpy(), 'r--', alpha=0.8, label='NN Prediction')
            
            # // 11.26: 添加修正后的剩余误差曲线 (绿色)，让效果更直观
            residual_error = y_raw[:samples, i] - pred_raw[:samples, i]
            axes[i].plot(residual_error.numpy(), 'g:', alpha=0.5, label='Residual After Correction')

            axes[i].set_ylabel(labels[i])
            axes[i].legend(loc='upper right', fontsize='small')
            axes[i].grid(True)
            
        plt.xlabel('Test Sample Index')
        plt.suptitle('Prediction vs Ground Truth (Unseen Data)')
        plt.savefig(os.path.join(self.output_dir, "test_prediction.png"))
        plt.close()


# ==========================================
# 主流程
# ==========================================
def main():
    # 1. 配置
    cfg = ConfigHandler()
    
    # 2. 数据
    dm = DatasetManager(cfg)
    train_loader, val_loader, test_data = dm.load_and_process()
    
    # 保存统计参数 (非常重要，推理需要)
    dm.save_stats(os.path.join(cfg.output_dir, "norm_stats.pt"))

    # 3. 模型
    model = DeepDynamicsModel()
    
    # 4. 训练
    trainer = ModelTrainer(model, cfg)
    trainer.train(train_loader, val_loader)
    
    # 5. 导出
    # 使用测试集的一个样本作为 trace 的输入示例
    trainer.export_model(test_data[0])

    # 6. 可视化与评估
    viz = ResultVisualizer(cfg.output_dir)
    viz.plot_loss(trainer.history)
    # 传入统计数据用于反归一化
    stats = {'y_mean': dm.y_mean, 'y_std': dm.y_std}
    viz.evaluate_test_set(trainer.model, test_data, stats)

if __name__ == "__main__":
    main()