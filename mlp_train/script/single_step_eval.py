#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
功能：单步预测性能评估 (Single-Step Prediction Evaluation)
对比对象：[Ground Truth 真值] vs [NN Model 神经网络] vs [Kinematics 运动学基准]
评估维度：
1. 速度层：v, w 的 MAE/RMSE
2. 位置层（积分）：dx, dy, dtheta 的单步积分误差
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import math

# ==========================================
# 1. 配置区域
# ==========================================
CONFIG = {
    # 数据集路径 (请修改为您生成的包含加速度的测试集或验证集)
    'dataset_path':     '/home/lsk1804/lsk_graduate/lsk_ws/src/mlp_train/data/experiment_1.pt',
    
    # 模型路径 (您当前的7维输入模型)
    'model_path':       '/home/lsk1804/lsk_graduate/lsk_ws/src/mlp_train/data/models/version_1202/best_mlp_model.pt',
    
    # 统计量路径 (用于反归一化)
    'stats_path':       '/home/lsk1804/lsk_graduate/lsk_ws/src/mlp_train/data/models/version_1202/norm_stats.pt',
    
    'output_dir':       '/home/lsk1804/lsk_graduate/lsk_ws/src/mlp_train/data/models/version_1202/results',
    'device':           'cpu',
    'plot_samples':     500  # 绘图时采样的点数，避免图太乱
}

class SingleStepEvaluator:
    def __init__(self, config):
        self.cfg = config
        self.device = torch.device(config['device'])
        
        self._check_paths()
        self.model = self._load_model()
        self.stats = self._load_stats()
        self.data_x, self.data_y = self._load_data()
        
    def _check_paths(self):
        if not os.path.exists(self.cfg['output_dir']):
            os.makedirs(self.cfg['output_dir'])
        
    def _load_model(self):
        try:
            model = torch.jit.load(self.cfg['model_path'], map_location=self.device)
            model.eval()
            print(f"[Info] 模型加载成功: {self.cfg['model_path']}")
            return model
        except Exception as e:
            print(f"[Error] 模型加载失败: {e}")
            sys.exit(1)

    def _load_stats(self):
        return torch.load(self.cfg['stats_path'], map_location=self.device)

    def _load_data(self):
        data = torch.load(self.cfg['dataset_path'], map_location=self.device)
        # 假设 x格式: [v_curr, w_curr, v_cmd, w_cmd, dt, acc_v, acc_w] (7维)
        # 假设 y格式: [v_next, w_next] (2维)
        return data['x'], data['y']

    def _integrate_step(self, v, w, dt):
        """
        简单的单步积分，计算位移增量
        """
        # 假设局部坐标系下，x轴朝前
        # dx = v * cos(0) * dt = v * dt
        # dy = v * sin(0) * dt = 0 (瞬时)
        # 但为了更精确对比，我们通常假设这一步内做圆弧运动或直线运动
        # 这里采用最简单的欧拉积分: delta_pos_local = [v*dt, 0]
        # 但为了对比位置误差，我们需要统一标准。
        # 更加通用的方式：计算这一步长内的位移模长和角度变化
        
        d_dist = v * dt
        d_theta = w * dt
        return d_dist, d_theta

    def evaluate(self):
        print(f"[*] 开始评估，数据总量: {len(self.data_x)}")
        
        # 容器
        results = {
            'gt':  {'v': [], 'w': [], 'dist': [], 'dtheta': []},
            'nn':  {'v': [], 'w': [], 'dist': [], 'dtheta': []},
            'kin': {'v': [], 'w': [], 'dist': [], 'dtheta': []}
        }

        with torch.no_grad():
            # 批量推理以提高速度
            # 1. 归一化输入
            x_norm = (self.data_x - self.stats['x_mean']) / self.stats['x_std']
            
            # 2. 模型预测 (输出也是归一化的)
            pred_norm = self.model(x_norm)
            
            # 3. 反归一化
            pred_y = pred_norm * self.stats['y_std'] + self.stats['y_mean']
            
        # 转换为 Numpy 进行分析
        X_np = self.data_x.cpu().numpy()
        Y_gt_np = self.data_y.cpu().numpy() # [v_next, w_next]
        Y_nn_np = pred_y.cpu().numpy()      # [v_next, w_next]
        
        # 提取关键列
        # X: [v(0), w(1), cmd_v(2), cmd_w(3), dt(4), acc_v(5), acc_w(6)]
        dt_seq = X_np[:, 4]
        cmd_v_seq = X_np[:, 2]
        cmd_w_seq = X_np[:, 3]
        
        # 遍历所有样本计算指标
        for i in range(len(X_np)):
            dt = dt_seq[i]
            
            # --- 1. Ground Truth (真值) ---
            v_gt = Y_gt_np[i, 0]
            w_gt = Y_gt_np[i, 1]
            d_dist_gt, d_theta_gt = self._integrate_step(v_gt, w_gt, dt)
            
            # --- 2. Neural Network (模型预测) ---
            v_nn = Y_nn_np[i, 0]
            w_nn = Y_nn_np[i, 1]
            d_dist_nn, d_theta_nn = self._integrate_step(v_nn, w_nn, dt)
            
            # --- 3. Kinematics (运动学基准) ---
            # 假设完全响应：v_next = v_cmd
            v_kin = cmd_v_seq[i]
            w_kin = cmd_w_seq[i]
            d_dist_kin, d_theta_kin = self._integrate_step(v_kin, w_kin, dt)
            
            # 存入列表
            results['gt']['v'].append(v_gt); results['gt']['w'].append(w_gt)
            results['gt']['dist'].append(d_dist_gt); results['gt']['dtheta'].append(d_theta_gt)
            
            results['nn']['v'].append(v_nn); results['nn']['w'].append(w_nn)
            results['nn']['dist'].append(d_dist_nn); results['nn']['dtheta'].append(d_theta_nn)
            
            results['kin']['v'].append(v_kin); results['kin']['w'].append(w_kin)
            results['kin']['dist'].append(d_dist_kin); results['kin']['dtheta'].append(d_theta_kin)

        # 转换为 Array
        for key in results:
            for subkey in results[key]:
                results[key][subkey] = np.array(results[key][subkey])
                
        self._print_metrics(results)
        self._plot_comparisons(results)

    def _print_metrics(self, r):
        print("\n" + "="*80)
        print(f"{'单步预测误差分析 (Single-Step Prediction Error)':^80}")
        print("="*80)
        
        # 计算 MAE
        def calc_mae(pred, gt):
            return np.mean(np.abs(pred - gt))
            
        mae_v_nn = calc_mae(r['nn']['v'], r['gt']['v'])
        mae_v_kin = calc_mae(r['kin']['v'], r['gt']['v'])
        
        mae_w_nn = calc_mae(r['nn']['w'], r['gt']['w'])
        mae_w_kin = calc_mae(r['kin']['w'], r['gt']['w'])
        
        mae_dist_nn = calc_mae(r['nn']['dist'], r['gt']['dist'])
        mae_dist_kin = calc_mae(r['kin']['dist'], r['gt']['dist'])
        
        # 打印表格
        print(f"{'Metric':<25} | {'Kinematics (Baseline)':<20} | {'Neural Network':<20} | {'Improvement':<10}")
        print("-" * 80)
        
        def print_row(name, err_kin, err_nn):
            imp = (err_kin - err_nn) / (err_kin + 1e-9) * 100
            print(f"{name:<25} | {err_kin:.5f}              | {err_nn:.5f}             | {imp:.1f}%")
            
        print_row("Velocity MAE (m/s)", mae_v_kin, mae_v_nn)
        print_row("Omega MAE (rad/s)", mae_w_kin, mae_w_nn)
        print("-" * 80)
        print_row("Integ. Dist MAE (m)", mae_dist_kin, mae_dist_nn)
        print_row("Integ. Angle MAE (rad)", calc_mae(r['kin']['dtheta'], r['gt']['dtheta']), calc_mae(r['nn']['dtheta'], r['gt']['dtheta']))
        print("="*80)
        print("注: 'Integ. Dist' 代表单步时间(dt)内积分出的位移误差。")

    def _plot_comparisons(self, r):
        # 1. 速度散点图对比
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # 降采样绘图，避免太密集
        idx = np.random.choice(len(r['gt']['v']), size=min(len(r['gt']['v']), self.cfg['plot_samples']), replace=False)
        
        # Linear Velocity
        ax = axes[0]
        ax.scatter(r['gt']['v'][idx], r['kin']['v'][idx], c='blue', alpha=0.5, label='Kinematics (Cmd)', s=15)
        ax.scatter(r['gt']['v'][idx], r['nn']['v'][idx], c='red', alpha=0.5, label='NN Prediction', s=15)
        ax.plot([min(r['gt']['v']), max(r['gt']['v'])], [min(r['gt']['v']), max(r['gt']['v'])], 'k--', lw=2)
        ax.set_xlabel('Ground Truth Velocity (m/s)')
        ax.set_ylabel('Predicted Velocity (m/s)')
        ax.set_title('Linear Velocity: Pred vs GT')
        ax.legend()
        ax.grid(True)
        
        # Angular Velocity
        ax = axes[1]
        ax.scatter(r['gt']['w'][idx], r['kin']['w'][idx], c='blue', alpha=0.5, label='Kinematics (Cmd)', s=15)
        ax.scatter(r['gt']['w'][idx], r['nn']['w'][idx], c='red', alpha=0.5, label='NN Prediction', s=15)
        ax.plot([min(r['gt']['w']), max(r['gt']['w'])], [min(r['gt']['w']), max(r['gt']['w'])], 'k--', lw=2)
        ax.set_xlabel('Ground Truth Omega (rad/s)')
        ax.set_ylabel('Predicted Omega (rad/s)')
        ax.set_title('Angular Velocity: Pred vs GT')
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.cfg['output_dir'], 'single_step_velocity_scatter.png'))
        print(f"[Plot] 散点图已保存至: {os.path.join(self.cfg['output_dir'], 'single_step_velocity_scatter.png')}")
        
        # 2. 位置误差直方图
        plt.figure(figsize=(10, 6))
        err_dist_kin = np.abs(r['kin']['dist'] - r['gt']['dist'])
        err_dist_nn = np.abs(r['nn']['dist'] - r['gt']['dist'])
        
        plt.hist(err_dist_kin, bins=50, alpha=0.5, label='Kinematics Error', color='blue', range=(0, 0.05))
        plt.hist(err_dist_nn, bins=50, alpha=0.5, label='NN Error', color='red', range=(0, 0.05))
        plt.xlabel('Single Step Position Error (m)')
        plt.ylabel('Count')
        plt.title('Distribution of Single Step Integration Error')
        plt.legend()
        plt.grid(True)
        
        plt.savefig(os.path.join(self.cfg['output_dir'], 'single_step_pos_error_hist.png'))
        print(f"[Plot] 误差直方图已保存至: {os.path.join(self.cfg['output_dir'], 'single_step_pos_error_hist.png')}")

if __name__ == "__main__":
    evaluator = SingleStepEvaluator(CONFIG)
    evaluator.evaluate()