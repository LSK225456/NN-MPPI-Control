#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
功能：对比 [真实轨迹] vs [运动学基准] vs [神经网络端到端预测]
适配模型：输入 [v, w, cmd_v, cmd_w, dt] -> 输出 [next_v, next_w]
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

# ==========================================
# 1. 全局配置参数
# ==========================================
CONFIG = {
    # 路径配置 (请根据实际情况修改)
    'dataset_path':     '/home/lsk1804/lsk_graduate/lsk_ws/src/mlp_train/data/experiment_1.pt',
    'model_path':       '/home/lsk1804/lsk_graduate/lsk_ws/src/mlp_train/data/models/version_1202/best_mlp_model.pt',
    'stats_path':       '/home/lsk1804/lsk_graduate/lsk_ws/src/mlp_train/data/models/version_1202/norm_stats.pt',
    'output_dir':       '/home/lsk1804/lsk_graduate/lsk_ws/src/mlp_train/data/models/version_1202/results/',
    
    # // 12.02: 移除 'dt' 参数，防止误用。现在 dt 均从数据集中动态读取。
    
    'test_start_idx':   1500,    # 从数据集的第几帧开始预测
    'duration_sec':     3.5,    # 预测总时长 (秒)
    'device':           'cpu', 
}

class OpenLoopTester:
    def __init__(self, config):
        self.cfg = config
        self.device = torch.device(config['device'])
        self._check_paths()
        
        self.model = self._load_model()
        self.stats = self._load_stats()
        self.data_x, self.data_y = self._load_data()
        
        sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

    def _check_paths(self):
        if not os.path.exists(self.cfg['output_dir']):
            os.makedirs(self.cfg['output_dir'])
            print(f"[Info] 创建输出目录: {self.cfg['output_dir']}")
        for k in ['dataset_path', 'model_path', 'stats_path']:
            if not os.path.exists(self.cfg[k]):
                print(f"[Error] 文件不存在: {self.cfg[k]}")
                sys.exit(1)

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
        try:
            stats = torch.load(self.cfg['stats_path'], map_location=self.device)
            return stats
        except Exception as e:
            print(f"[Error] 统计量加载失败: {e}")
            sys.exit(1)

    def _load_data(self):
        # 加载数据 (假设是 DatasetManager 生成的字典)
        data = torch.load(self.cfg['dataset_path'], map_location=self.device)
        # x: [N, 5], y: [N, 2]
        return data['x'], data['y']

    def _kinematics_step(self, state, vel, dt):
        """
        运动学积分: 根据速度和时间步长更新位姿
        state: [x, y, theta]
        vel:   [v, w]
        dt:    当前帧的真实时间间隔
        """
        x, y, theta = state
        v, w = vel
        
        # 离散积分 (Runge-Kutta 也可以，这里用欧拉积分保持与MPPI一致)
        next_x = x + v * math.cos(theta) * dt
        next_y = y + v * math.sin(theta) * dt
        next_theta = theta + w * dt
        
        # 角度归一化 (-pi ~ pi)
        next_theta = (next_theta + math.pi) % (2 * math.pi) - math.pi
        
        return np.array([next_x, next_y, next_theta])

    def run_prediction(self):
        start_idx = self.cfg['test_start_idx']
        duration = self.cfg['duration_sec']
        
        # 轨迹容器
        traj_gt = []    # Ground Truth
        traj_kin = []   # Kinematic Baseline
        traj_nn = []    # Neural Network
        
        vel_gt_log = [] 
        vel_nn_log = []
        
        # 初始状态统一设为 0
        curr_gt_state = np.zeros(3) 
        curr_kin_state = np.zeros(3)
        curr_nn_state = np.zeros(3)
        
        # 获取初始速度用于网络闭环递归
        init_feat = self.data_x[start_idx]
        curr_nn_vel = np.array([init_feat[0].item(), init_feat[1].item()])

        curr_nn_acc = np.array([init_feat[5].item(), init_feat[6].item()])      # 初始化起始加速度 (从数据集直接读取真实值)

        print(f"[*] 开始动态时间步长预测: Start Frame {start_idx}, Duration {duration}s")

        current_time = 0.0
        idx = start_idx
        step_count = 0

        # // 12.02: 改为基于时间的循环，不再假设固定步数
        while current_time < duration and idx < len(self.data_x):
            # 1. 获取该帧数据
            raw_feat = self.data_x[idx]
            
            # // 12.02: 动态获取该帧记录的真实 dt
            # 输入特征格式: [v, w, cmd_v, cmd_w, dt] -> dt是索引4
            real_dt = raw_feat[4].item()
            
            # 如果 dt 异常(例如数据处理错误导致0或负数)，做个保护
            if real_dt <= 0.001: 
                real_dt = 0.001

            # 累加时间
            current_time += real_dt
            
            # 读取控制指令
            cmd_v = raw_feat[2].item()
            cmd_w = raw_feat[3].item()
            cmd = np.array([cmd_v, cmd_w])
            
            # 读取真实下一时刻速度 (用于 GT 积分)
            gt_next_v = self.data_y[idx][0].item()
            gt_next_w = self.data_y[idx][1].item()
            gt_vel = np.array([gt_next_v, gt_next_w])

            # ------------------------------------------------------
            # 2. 轨迹积分 (使用 real_dt)
            # ------------------------------------------------------
            
            # A. Ground Truth
            traj_gt.append(curr_gt_state.copy())
            curr_gt_state = self._kinematics_step(curr_gt_state, gt_vel, real_dt)
            vel_gt_log.append(gt_vel)

            # B. Kinematic Baseline (假设 v_real = v_cmd)
            traj_kin.append(curr_kin_state.copy())
            curr_kin_state = self._kinematics_step(curr_kin_state, cmd, real_dt)

            # C. NN Model
            traj_nn.append(curr_nn_state.copy())

            # // 12.02: 构造输入 [v, w, cmd_v, cmd_w, dt, acc_v, acc_w]
            nn_input = torch.tensor([
                curr_nn_vel[0], curr_nn_vel[1], 
                cmd_v, cmd_w, real_dt,
                curr_nn_acc[0], curr_nn_acc[1]  # 加入加速度特征
            ], dtype=torch.float32).to(self.device)

            # 归一化 & 推理
            nn_input_norm = (nn_input - self.stats['x_mean']) / self.stats['x_std']
            with torch.no_grad():
                pred_norm = self.model(nn_input_norm.unsqueeze(0)).squeeze(0)
            
            # 反归一化
            pred_vel_raw = pred_norm * self.stats['y_std'] + self.stats['y_mean']
            pred_v = pred_vel_raw[0].item()
            pred_w = pred_vel_raw[1].item()
            pred_vel = np.array([pred_v, pred_w])
            
            pred_acc_v = (pred_v - curr_nn_vel[0]) / real_dt
            pred_acc_w = (pred_w - curr_nn_vel[1]) / real_dt
            pred_acc = np.array([pred_acc_v, pred_acc_w])

            # 状态更新 (使用 real_dt)
            curr_nn_state = self._kinematics_step(curr_nn_state, pred_vel, real_dt)
            
            # 闭环递归: 更新网络内部速度状态
            curr_nn_vel = pred_vel
            curr_nn_acc = pred_acc
            vel_nn_log.append(pred_vel)

            # ------------------------------------------------------
            # 3. 诊断信息
            # ------------------------------------------------------
            if step_count < 3: # 仅打印前3帧调试
                print(f"Step {step_count} | DT={real_dt:.4f}s | CMD_V={cmd_v:.2f} | NN_V={pred_v:.2f} | GT_V={gt_next_v:.2f}")

            # 索引递增
            idx += 1
            step_count += 1

        # 转换为 Numpy
        self.traj_gt = np.array(traj_gt)
        self.traj_kin = np.array(traj_kin)
        self.traj_nn = np.array(traj_nn)
        self.vel_gt = np.array(vel_gt_log)
        self.vel_nn = np.array(vel_nn_log)
        
        # 生成时间轴数组 (用于绘图x轴)
        # 注意：这里需要重新从数据集中提取累积时间，或者简单使用 0..N
        # 为了简单直观，绘图时直接用索引 (Frame) 或 估算时间均可，这里暂用索引
        self._plot_results()

    def _plot_results(self):
        # 1. 轨迹图
        plt.figure(figsize=(12, 10))
        plt.plot(self.traj_gt[:, 0], self.traj_gt[:, 1], 'k-', linewidth=2, label='Ground Truth')
        plt.plot(self.traj_kin[:, 0], self.traj_kin[:, 1], 'b:', linewidth=2, label='Kinematic (Ideal)')
        plt.plot(self.traj_nn[:, 0], self.traj_nn[:, 1], 'r--', linewidth=2, label='NN Prediction')
        
        plt.title(f"Trajectory Prediction (Duration: {self.cfg['duration_sec']}s)")
        plt.xlabel("X [m]")
        plt.ylabel("Y [m]")
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        
        traj_path = os.path.join(self.cfg['output_dir'], 'traj_comparison_variable_dt.png')
        plt.savefig(traj_path)
        print(f"[Result] 轨迹对比图已保存: {traj_path}")
        
        # 2. 速度响应图
        plt.figure(figsize=(12, 6))
        
        plt.subplot(2, 1, 1)
        plt.plot(self.vel_gt[:, 0], 'k-', alpha=0.6, label='GT Vel')
        plt.plot(self.vel_nn[:, 0], 'r--', label='NN Vel')
        plt.ylabel('v [m/s]')
        plt.title('Velocity Response')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 1, 2)
        plt.plot(self.vel_gt[:, 1], 'k-', alpha=0.6, label='GT Omega')
        plt.plot(self.vel_nn[:, 1], 'r--', label='NN Omega')
        plt.ylabel('w [rad/s]')
        plt.xlabel('Steps (Variable DT)')
        plt.legend()
        plt.grid(True)
        
        vel_path = os.path.join(self.cfg['output_dir'], 'velocity_response.png')
        plt.savefig(vel_path)
        print(f"[Result] 速度响应图已保存: {vel_path}")

def main():
    tester = OpenLoopTester(CONFIG)
    tester.run_prediction()

if __name__ == "__main__":
    main()