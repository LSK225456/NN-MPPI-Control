#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import sys

# 设置绘图风格 (使用经典风格，避免依赖 seaborn)
plt.style.use('bmh')

class ResidualAnalyzer:
    """
    残差分析与绘图器
    职责：加载 .pt 数据集，计算统计指标，绘制并保存残差曲线。
    """
    def __init__(self, pt_path):
        self.pt_path = pt_path
        self.output_dir = os.path.dirname(pt_path)
        self.filename_base = os.path.splitext(os.path.basename(pt_path))[0]
        
        self.data = None
        self.y_numpy = None
        self.labels = ['Longitudinal Error (m)', 'Lateral Error (m)', 
                       'Heading Error (rad)', 'Linear Vel Error (m/s)', 'Angular Vel Error (rad/s)']
        self.tags = ['err_long', 'err_lat', 'err_theta', 'err_v', 'err_w']

    def load_data(self):
        """加载数据并转换为 Numpy 格式"""
        if not os.path.exists(self.pt_path):
            print(f"[Error] 文件不存在: {self.pt_path}")
            sys.exit(1)

        print(f"[*] 正在加载数据: {self.pt_path}")
        try:
            # map_location='cpu' 确保即使没显卡也能运行
            self.data = torch.load(self.pt_path, map_location='cpu')
            
            # 提取 Y (Labels) 并转为 numpy
            # Y shape: [N, 5] -> [err_long, err_lat, err_theta, err_v, err_w]
            self.y_numpy = self.data['y'].detach().numpy()
            
            # 简单校验
            if self.y_numpy.shape[1] != 5:
                print(f"[Error] 数据维度不对! 期望 5 列，实际 {self.y_numpy.shape[1]} 列")
                sys.exit(1)
                
            print(f"[*] 数据加载成功，样本数: {self.y_numpy.shape[0]}")

        except Exception as e:
            print(f"[Error] 加载失败: {e}")
            sys.exit(1)

    def print_statistics(self):
        """打印统计摘要"""
        print("\n" + "="*50)
        print(" 数据统计摘要 (Statistics Summary)")
        print("="*50)
        print(f"{'Dimension':<25} | {'Mean':<10} | {'Mean Abs':<10} | {'Std Dev':<10} | {'Max Abs':<10}")
        print("-" * 65)

        for i, tag in enumerate(self.tags):
            col_data = self.y_numpy[:, i]
            mean_val = np.mean(col_data)
            mean_abs_val = np.mean(np.abs(col_data))
            std_val = np.std(col_data)
            max_val = np.max(np.abs(col_data))
            
            print(f"{tag:<25} | {mean_val:6.4f}     | {mean_abs_val:6.4f}     | {std_val:6.4f}     | {max_val:6.4f}")
        print("="*50 + "\n")

    def _plot_subplot(self, ax, data_col, title, ylabel, color):
        """内部辅助绘图函数"""
        frames = np.arange(len(data_col))
        ax.plot(frames, data_col, color=color, linewidth=1.0, alpha=0.8, label='Residual')
        
        # 绘制 0 刻度线 (基准)
        ax.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
        
        # 绘制均值线
        mean_val = np.mean(data_col)
        ax.axhline(mean_val, color='red', linestyle=':', linewidth=1.0, label=f'Mean: {mean_val:.4f}')

        ax.set_title(title, fontsize=10)
        ax.set_xlabel('Frame Index', fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, which='both', linestyle='--', alpha=0.7)

    def plot_all(self):
        """执行所有绘图任务"""
        if self.y_numpy is None:
            return

        # 图 1: 位置误差 (Longitudinal & Lateral)
        fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        self._plot_subplot(ax1, self.y_numpy[:, 0], 'Longitudinal Error (Body X)', 'Error (m)', '#1f77b4') # Blue
        self._plot_subplot(ax2, self.y_numpy[:, 1], 'Lateral Error (Body Y)', 'Error (m)', '#ff7f0e')      # Orange
        fig1.suptitle(f'Position Residuals - {self.filename_base}', fontsize=12)
        save_path1 = os.path.join(self.output_dir, f'{self.filename_base}_position.png')
        fig1.savefig(save_path1, dpi=300)
        print(f"[+] 保存图像: {save_path1}")
        plt.close(fig1)

        # 图 2: 角度相关误差 (Theta & Omega)
        fig2, (ax3, ax4) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        self._plot_subplot(ax3, self.y_numpy[:, 2], 'Heading Error (Theta)', 'Error (rad)', '#2ca02c')     # Green
        self._plot_subplot(ax4, self.y_numpy[:, 4], 'Angular Velocity Error (Omega)', 'Error (rad/s)', '#d62728') # Red
        fig2.suptitle(f'Heading & Rotation Residuals - {self.filename_base}', fontsize=12)
        save_path2 = os.path.join(self.output_dir, f'{self.filename_base}_rotation.png')
        fig2.savefig(save_path2, dpi=300)
        print(f"[+] 保存图像: {save_path2}")
        plt.close(fig2)

        # 图 3: 线速度误差 (Linear Velocity)
        fig3, ax5 = plt.subplots(1, 1, figsize=(10, 4))
        self._plot_subplot(ax5, self.y_numpy[:, 3], 'Linear Velocity Error', 'Error (m/s)', '#9467bd')     # Purple
        fig3.suptitle(f'Velocity Residuals - {self.filename_base}', fontsize=12)
        save_path3 = os.path.join(self.output_dir, f'{self.filename_base}_velocity.png')
        fig3.savefig(save_path3, dpi=300)
        print(f"[+] 保存图像: {save_path3}")
        plt.close(fig3)

def main():
    parser = argparse.ArgumentParser(description="Visualize Dataset Residuals")
    parser.add_argument('file', help="Path to the .pt file")
    args = parser.parse_args()

    visualizer = ResidualAnalyzer(args.file)
    visualizer.load_data()
    visualizer.print_statistics()
    visualizer.plot_all()

if __name__ == "__main__":
    main()