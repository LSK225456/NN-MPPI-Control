#!/usr/bin/env python3
# -*- coding: utf-8 -*-



"""
Vehicle Dynamics Prediction Node (Offline) - Batch Processing Version
功能：读取ROS Bag，基于残差神经网络(LSTM/Recurrent)与运动学模型进行批量多步开环轨迹预测，并自动保存对比结果图。

修改记录：
- 2025-12-08: 
  1. 增加图片输出路径配置，移除交互式弹窗。
  2. 增加 N_TEST(测试次数) 和 TEST_STEP(步长间隔) 参数。
  3. 优化内存管理，批量绘图不卡顿。
"""

import os
import sys
import numpy as np
import pandas as pd
import rosbag
import joblib
import matplotlib.pyplot as plt
from collections import deque
from scipy.interpolate import interp1d
from tensorflow import keras
from scipy.spatial.transform import Rotation as R
import tensorflow as tf
import tensorflow.keras.backend as K



# [新增] 自定义损失函数 (必须与训练代码完全一致)

def physics_weighted_mse(y_true, y_pred):
    """
    自定义物理加权损失函数，用于模型加载时的映射
    """
    # 1. 计算标准的平方差
    squared_diff = tf.square(y_true - y_pred)
    
    # 2. 定义权重向量 (必须与训练时一致，例如 [1.0, 10.0])
    # 注意：如果训练时改了权重，这里也要改，否则评估出的 loss 值会不对（但不影响预测结果）
    weights = tf.constant([1.0, 1.0060286442927682])
    
    # 3. 加权
    weighted_squared_diff = squared_diff * weights
    
    # 4. 返回平均值
    return K.mean(weighted_squared_diff, axis=-1)

# ==============================================================================
# 1. Configuration (参数配置区)
# ==============================================================================
class Config:
    # --- 路径设置 (请修改此处) ---
    BAG_PATH = '/home/lsk1804/lsk_graduate/lsk_ws/bag/7801_ndt_teb_4l_test_1207.bag'
    MODEL_PATH = '/home/lsk1804/lsk_graduate/lsk_ws/src/NeuralNetwork_for_VehicleDynamicsModeling/outputs/2025_12_18/20_24_41/keras_model_recurrent.h5'
    SCALER_X_PATH = '/home/lsk1804/lsk_graduate/lsk_ws/src/NeuralNetwork_for_VehicleDynamicsModeling/outputs/2025_12_18/20_24_41/scaler.plk'
    SCALER_Y_PATH = '/home/lsk1804/lsk_graduate/lsk_ws/src/NeuralNetwork_for_VehicleDynamicsModeling/outputs/2025_12_18/20_24_41/scaler_y.joblib'
    
    # [新增] 图片保存文件夹路径
    OUTPUT_IMG_DIR = '/home/lsk1804/lsk_graduate/lsk_ws/src/NeuralNetwork_for_VehicleDynamicsModeling/outputs/2025_12_18/20_24_41/predict/'

    # --- 话题设置 ---
    TOPIC_WHEEL = '/wheel_odom'   # ~50Hz
    TOPIC_CMD = '/cmd_vel'        # ~15Hz
    TOPIC_POSE = '/odom'          # ~10Hz

    # --- 数据处理参数 (严格对齐 data_process.py) ---
    TARGET_DT = 0.05              # 锁定推理步长 0.05s (20Hz)
    SMOOTH_WINDOW = 8             # 平滑窗口
    STATIC_VEL_THRESHOLD = 0.05   # 静态滤除阈值
    MAX_DT_THRESHOLD = 0.3        # Cmd匹配容差

    # --- 模型与推理参数 ---
    INPUT_TIMESTEPS = 25         # LSTM历史窗口长度 (必须与训练设置一致!)
    INPUT_FEATURES = 10            # 输入特征维度 [v, w, dt, cmd_v, cmd_w, a_v, a_w, err_v, err_w, v_x_w]
    PREDICT_HORIZON_SEC = 3.0     # 预测未来时长 (秒)
    KINEMATIC_TAU = 0.2           # 运动学模型一阶滞后时间常数
    
    # --- [新增] 批量测试控制参数 ---
    N_TEST_SAMPLES = 30           # 总共进行多少次推理实验 (生成多少张图)
    TEST_STEP_STRIDE = 100        # 每次推理起点的间隔步数 (例如: 第100帧测一次, 第200帧测一次...)

    # 自动计算预测步数
    @property
    def PREDICT_STEPS(self):
        return int(self.PREDICT_HORIZON_SEC / self.TARGET_DT)

# ==============================================================================
# 2. Data Processor (数据处理模块)
# ==============================================================================
class BagProcessor:
    """负责从Bag读取数据，并执行平滑、对齐、重采样，生成标准DataFrame"""
    
    def __init__(self, config):
        self.cfg = config

    def _read_bag(self):
        wheel_data, cmd_data, pose_data = [], [], []
        
        print(f"[Info] Loading Bag: {self.cfg.BAG_PATH} ...")
        try:
            with rosbag.Bag(self.cfg.BAG_PATH, 'r') as bag:
                # 1. Wheel Odom
                for topic, msg, t in bag.read_messages(topics=[self.cfg.TOPIC_WHEEL]):
                    ts = msg.header.stamp.to_sec() if hasattr(msg, 'header') else t.to_sec()
                    wheel_data.append({'timestamp': ts, 'v': msg.twist.twist.linear.x, 'w': msg.twist.twist.angular.z})

                # 2. Cmd Vel
                for topic, msg, t in bag.read_messages(topics=[self.cfg.TOPIC_CMD]):
                    ts = t.to_sec()
                    cmd_data.append({'timestamp': ts, 'cmd_v': msg.linear.x, 'cmd_w': msg.angular.z})
                
                # 3. Pose
                for topic, msg, t in bag.read_messages(topics=[self.cfg.TOPIC_POSE]):
                    ts = msg.header.stamp.to_sec() if hasattr(msg, 'header') else t.to_sec()
                    px = msg.pose.pose.position.x
                    py = msg.pose.pose.position.y
                    q_list = [msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, 
                              msg.pose.pose.orientation.z, msg.pose.pose.orientation.w]
                    yaw = R.from_quat(q_list).as_euler('xyz')[2]
                    pose_data.append({'timestamp': ts, 'x_gt': px, 'y_gt': py, 'theta_gt': yaw})

        except Exception as e:
            print(f"[Error] Failed to read bag: {e}")
            sys.exit(1)

        return pd.DataFrame(wheel_data), pd.DataFrame(cmd_data), pd.DataFrame(pose_data)

    def process_data(self):
        df_wheel, df_cmd, df_pose = self._read_bag()

        if df_wheel.empty or df_cmd.empty:
            raise ValueError("Wheel or Cmd data is empty!")

        # 1. 预处理
        df_wheel = df_wheel.sort_values('timestamp').drop_duplicates('timestamp')
        df_cmd = df_cmd.sort_values('timestamp').drop_duplicates('timestamp')
        if not df_pose.empty:
            df_pose = df_pose.sort_values('timestamp').drop_duplicates('timestamp')

        # 2. 平滑滤波
        df_wheel['v'] = df_wheel['v'].rolling(window=self.cfg.SMOOTH_WINDOW, center=True, min_periods=1).mean()
        df_wheel['w'] = df_wheel['w'].rolling(window=self.cfg.SMOOTH_WINDOW, center=True, min_periods=1).mean()

        # 3. 时间网格
        t_start = max(df_wheel['timestamp'].min(), df_cmd['timestamp'].min())
        t_end = min(df_wheel['timestamp'].max(), df_cmd['timestamp'].max())
        if not df_pose.empty:
            t_start = max(t_start, df_pose['timestamp'].min())
            t_end = min(t_end, df_pose['timestamp'].max())

        t_grid = np.arange(t_start, t_end, self.cfg.TARGET_DT)
        if len(t_grid) < self.cfg.INPUT_TIMESTEPS + 10:
            raise ValueError("Data duration is too short for the required history window!")

        df_sync = pd.DataFrame({'timestamp': t_grid})

        # 4. 插值 Wheel
        f_v = interp1d(df_wheel['timestamp'], df_wheel['v'], kind='linear', fill_value="extrapolate")
        f_w = interp1d(df_wheel['timestamp'], df_wheel['w'], kind='linear', fill_value="extrapolate")
        df_sync['v'] = f_v(t_grid)
        df_sync['w'] = f_w(t_grid)

        # 5. 插值 Pose
        if not df_pose.empty:
            f_x = interp1d(df_pose['timestamp'], df_pose['x_gt'], kind='linear', fill_value="extrapolate")
            f_y = interp1d(df_pose['timestamp'], df_pose['y_gt'], kind='linear', fill_value="extrapolate")
            theta_unwrapped = np.unwrap(df_pose['theta_gt'])
            f_theta = interp1d(df_pose['timestamp'], theta_unwrapped, kind='linear', fill_value="extrapolate")
            df_sync['x_gt'] = f_x(t_grid)
            df_sync['y_gt'] = f_y(t_grid)
            df_sync['theta_gt'] = f_theta(t_grid)
        else:
            df_sync['x_gt'] = 0.0; df_sync['y_gt'] = 0.0; df_sync['theta_gt'] = 0.0

        # 6. 匹配 Cmd (Backward)
        df_sync = pd.merge_asof(
            df_sync, df_cmd, on='timestamp', direction='backward', tolerance=self.cfg.MAX_DT_THRESHOLD
        )
        df_sync.dropna(subset=['cmd_v', 'cmd_w'], inplace=True)

        # 7. 静态滤除
        df_final = df_sync[abs(df_sync['v']) > self.cfg.STATIC_VEL_THRESHOLD].reset_index(drop=True)

        # 8. 锁定 DT
        df_final['dt'] = self.cfg.TARGET_DT

        # ------------------------------------------------------------------
        # 1212：特征工程 - 预计算推理所需的物理特征 (与 data_process.py 保持一致)
        # ------------------------------------------------------------------
        # A. 计算差分 (用于计算加速度)
        # 填充 fillna(0) 防止第一帧 NaN 导致报错
        df_final['diff_v'] = df_final['v'].diff().fillna(0.0)
        df_final['diff_w'] = df_final['w'].diff().fillna(0.0)

        # B. 计算历史加速度 (Input: a_v, a_w)
        df_final['a_v'] = df_final['diff_v'] / df_final['dt']
        df_final['a_w'] = df_final['diff_w'] / df_final['dt']

        # C. 计算控制误差 (Input: err_v, err_w)
        df_final['err_v'] = df_final['cmd_v'] - df_final['v']
        df_final['err_w'] = df_final['cmd_w'] - df_final['w']

        # D. 计算动力学耦合项 (Input: v_x_w)
        df_final['v_x_w'] = df_final['v'] * df_final['w']
        # ------------------------------------------------------------------

        print(f"[Info] Preprocessing Done. Valid Frames: {len(df_final)}")
        return df_final

# ==============================================================================
# 3. Predictor (推理引擎)
# ==============================================================================
class DynamicsPredictor:
    def __init__(self, config):
        self.cfg = config
        self._load_resources()

    def _load_resources(self):
        print(f"[Info] Loading Model: {self.cfg.MODEL_PATH}")
        # self.model = keras.models.load_model(self.cfg.MODEL_PATH)
        # 自定义loss函数
        self.model = keras.models.load_model(
            self.cfg.MODEL_PATH,
            custom_objects={'physics_weighted_mse': physics_weighted_mse,
                            'physics_weighted_mse_dynamic': physics_weighted_mse    # 训练时使用的是动态loss函数
                            }
        )
        self.scaler_X = joblib.load(self.cfg.SCALER_X_PATH)
        self.scaler_y = joblib.load(self.cfg.SCALER_Y_PATH)

    def predict_trajectory(self, start_idx, df_data):
        """
        支持 LSTM 多帧历史输入的开环预测
        """
        steps = self.cfg.PREDICT_STEPS
        dt = self.cfg.TARGET_DT
        history_len = self.cfg.INPUT_TIMESTEPS
        
        # 1. 边界检查
        if start_idx < history_len or start_idx + steps >= len(df_data):
            return None, None, None

        # 2. 初始化历史窗口 (预热)
        history_buffer = deque(maxlen=history_len)
        for i in range(history_len):
            row = df_data.iloc[start_idx - history_len + i]
            # feat = [row['v'], row['w'], dt, row['cmd_v'], row['cmd_w']]
            feat = [
                row['v'], row['w'], dt, row['cmd_v'], row['cmd_w'],
                row['a_v'], row['a_w'], row['err_v'], row['err_w'], row['v_x_w']
            ]
            history_buffer.append(feat)

        # 3. 初始化当前状态
        initial_state = df_data.iloc[start_idx]
        nn_state = {
            'v': initial_state['v'], 'w': initial_state['w'],
            'x': initial_state['x_gt'], 'y': initial_state['y_gt'], 'theta': initial_state['theta_gt']
        }
        kin_state = nn_state.copy()

        traj_nn = [nn_state.copy()]
        traj_kin = [kin_state.copy()]
        
        # GT 轨迹
        gt_slice = df_data.iloc[start_idx : start_idx + steps + 1]
        traj_gt = gt_slice[['timestamp', 'x_gt', 'y_gt', 'v', 'w', 'theta_gt']].to_dict('records')

        # 4. 循环推理
        for k in range(steps):
            # A. 神经网络分支
            input_seq = np.array(history_buffer)
            input_scaled = self.scaler_X.transform(input_seq)
            model_input = input_scaled.reshape(1, history_len, self.cfg.INPUT_FEATURES)
            
            pred_res_scaled = self.model.predict(model_input, verbose=0)
            pred_res = self.scaler_y.inverse_transform(pred_res_scaled)
            diff_v, diff_w = pred_res[0][0], pred_res[0][1]
            
            nn_state['v'] += diff_v
            nn_state['w'] += diff_w
            
            nn_state['x'] += nn_state['v'] * np.cos(nn_state['theta']) * dt
            nn_state['y'] += nn_state['v'] * np.sin(nn_state['theta']) * dt
            nn_state['theta'] += nn_state['w'] * dt
            
            traj_nn.append(nn_state.copy())

            # 自回归更新 Buffer
            if start_idx + k + 1 < len(df_data):
                next_cmd_row = df_data.iloc[start_idx + k + 1] 
                next_cmd_v, next_cmd_w = next_cmd_row['cmd_v'], next_cmd_row['cmd_w']
            else:
                next_cmd_v, next_cmd_w = history_buffer[-1][3], history_buffer[-1][4]

            # 2. 计算派生特征 (Accel, Error, Interaction)
            # 加速度 a = delta_v / dt (注意：模型预测的正是 delta_v)
            next_a_v = diff_v / dt
            next_a_w = diff_w / dt
            
            # 误差 err = cmd - v (注意：使用更新后的 v)
            next_err_v = next_cmd_v - nn_state['v']
            next_err_w = next_cmd_w - nn_state['w']
            
            # 耦合项
            next_v_x_w = nn_state['v'] * nn_state['w']

            next_feat = [
                nn_state['v'], nn_state['w'], dt, next_cmd_v, next_cmd_w,  # 基础 5 维
                next_a_v, next_a_w, next_err_v, next_err_w, next_v_x_w     # 新增 5 维
            ]
            history_buffer.append(next_feat)



            #####  B. 运动学模型分支
            curr_cmd_row = df_data.iloc[start_idx + k]
            curr_cmd_v, curr_cmd_w = curr_cmd_row['cmd_v'], curr_cmd_row['cmd_w']
            
            tau = self.cfg.KINEMATIC_TAU
            kin_state['v'] += (1.0/tau) * (curr_cmd_v - kin_state['v']) * dt
            kin_state['w'] += (1.0/tau) * (curr_cmd_w - kin_state['w']) * dt
            kin_state['x'] += kin_state['v'] * np.cos(kin_state['theta']) * dt
            kin_state['y'] += kin_state['v'] * np.sin(kin_state['theta']) * dt
            kin_state['theta'] += kin_state['w'] * dt
            traj_kin.append(kin_state.copy())

        return traj_nn, traj_kin, traj_gt

# ==============================================================================
# 4. Visualizer (绘图与保存模块)
# ==============================================================================
class Visualizer:
    @staticmethod
    def save_comparison(traj_nn, traj_kin, traj_gt, dt, output_dir, file_id):
        """绘制并保存图片，不显示"""
        df_nn = pd.DataFrame(traj_nn)
        df_kin = pd.DataFrame(traj_kin)
        df_gt = pd.DataFrame(traj_gt)
        t_axis = np.arange(len(df_nn)) * dt

        # 创建 Figure
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"Prediction Horizon: {t_axis[-1]:.1f}s | Sample ID: {file_id}", fontsize=14)

        # 1. Trajectory
        ax = axes[0, 0]
        ax.plot(df_gt['x_gt'], df_gt['y_gt'], 'k-', lw=2, label='GT')
        ax.plot(df_nn['x'], df_nn['y'], 'r--', lw=2, label='NN')
        ax.plot(df_kin['x'], df_kin['y'], 'b:', lw=2, label='Kinematic')
        ax.set_title("2D Trajectory")
        ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]")
        ax.axis('equal'); ax.legend(); ax.grid(True, alpha=0.5)

        # 2. Linear Vel
        ax = axes[0, 1]
        ax.plot(t_axis, df_gt['v'], 'k-', label='GT')
        ax.plot(t_axis, df_nn['v'], 'r--', label='NN')
        ax.plot(t_axis, df_kin['v'], 'b:', label='Kinematic')
        ax.set_title("Linear Velocity"); ax.grid(True)

        # 3. Angular Vel
        ax = axes[1, 0]
        ax.plot(t_axis, df_gt['w'], 'k-', label='GT')
        ax.plot(t_axis, df_nn['w'], 'r--', label='NN')
        ax.plot(t_axis, df_kin['w'], 'b:', label='Kinematic')
        ax.set_title("Angular Velocity"); ax.grid(True)

        # 4. Heading
        ax = axes[1, 1]
        ax.plot(t_axis, df_gt['theta_gt'], 'k-', label='GT')
        ax.plot(t_axis, df_nn['theta'], 'r--', label='NN')
        ax.plot(t_axis, df_kin['theta'], 'b:', label='Kinematic')
        ax.set_title("Heading (Theta)"); ax.grid(True)

        plt.tight_layout()
        
        # 保存图片
        filename = f"pred_sample_{file_id}.png"
        save_path = os.path.join(output_dir, filename)
        plt.savefig(save_path)
        print(f"   [Saved] {save_path}")
        
        # 关键：关闭图形以释放内存
        plt.close(fig)


class Evaluator:
    def __init__(self):
        self.metrics_list = []

    def calculate_single_metrics(self, traj_pred, traj_gt, label="Model"):
        """计算单次推理的各项指标"""
        df_pred = pd.DataFrame(traj_pred)
        df_gt = pd.DataFrame(traj_gt)
        
        # 1. 轨迹位置误差
        # 计算每一时刻的欧氏距离
        dist_errors = np.sqrt((df_pred['x'] - df_gt['x_gt'])**2 + (df_pred['y'] - df_gt['y_gt'])**2)
        ade = np.mean(dist_errors) # 平均位移误差 (Average Displacement Error)
        fde = dist_errors.iloc[-1] # 终点位移误差 (Final Displacement Error)
        max_dist_err = np.max(dist_errors)

        # 2. 状态误差 (MAE)
        mae_v = np.mean(np.abs(df_pred['v'] - df_gt['v']))
        mae_w = np.mean(np.abs(df_pred['w'] - df_gt['w']))
        
        # 3. 航向角误差 (需处理周期性)
        # 误差 = 预测 - 真值，然后归一化到 [-pi, pi]
        theta_err = df_pred['theta'] - df_gt['theta_gt']
        theta_err = np.arctan2(np.sin(theta_err), np.cos(theta_err)) # Wrap to [-pi, pi]
        mae_theta = np.mean(np.abs(theta_err))

        return {
            'ADE': ade, 'FDE': fde, 'MaxPos': max_dist_err,
            'MAE_v': mae_v, 'MAE_w': mae_w, 'MAE_theta': mae_theta
        }

    def record_sample(self, sample_id, traj_nn, traj_kin, traj_gt):
        """记录这一轮的对比结果"""
        m_nn = self.calculate_single_metrics(traj_nn, traj_gt, "NN")
        m_kin = self.calculate_single_metrics(traj_kin, traj_gt, "Kinematic")
        
        # 计算提升率 (Improvement %)
        # Formula: (Base - Method) / Base * 100
        imp = {}
        for key in m_nn.keys():
            base = m_kin[key]
            curr = m_nn[key]
            # 避免除以0
            if abs(base) < 1e-6: 
                percent = 0.0 
            else:
                percent = (base - curr) / base * 100
            imp[f"Imp_{key}"] = percent

        # 合并所有数据
        record = {'SampleID': sample_id}
        # 添加 NN 指标 (前缀 NN_)
        record.update({f"NN_{k}": v for k, v in m_nn.items()})
        # 添加 Kin 指标 (前缀 Kin_)
        record.update({f"Kin_{k}": v for k, v in m_kin.items()})
        # 添加 提升率 (前缀 Imp_)
        record.update(imp)
        
        self.metrics_list.append(record)

    def save_report(self, output_dir):
        """生成并保存 TXT 报告"""
        if not self.metrics_list:
            print("[Warn] No metrics recorded.")
            return

        df = pd.DataFrame(self.metrics_list)
        save_path = os.path.join(output_dir, 'evaluation_report.txt')
        
        with open(save_path, 'w') as f:
            f.write("================================================================\n")
            f.write("                VEHICLE DYNAMICS PREDICTION REPORT              \n")
            f.write("================================================================\n\n")
            
            # --- Part 1: Overall Summary ---
            f.write("1. OVERALL PERFORMANCE SUMMARY (Average over all samples)\n")
            f.write("-" * 75 + "\n")
            f.write(f"{'Metric':<15} | {'Kinematic':<12} | {'NeuralNet':<12} | {'Improvement':<10}\n")
            f.write("-" * 75 + "\n")
            
            # 定义要展示的核心指标 keys
            core_metrics = ['ADE', 'FDE', 'MAE_v', 'MAE_w', 'MAE_theta']
            
            for m in core_metrics:
                mean_kin = df[f"Kin_{m}"].mean()
                mean_nn = df[f"NN_{m}"].mean()
                if abs(mean_kin) < 1e-6:
                    mean_imp = 0.0
                else:
                    mean_imp = (mean_kin - mean_nn) / mean_kin * 100
                
                f.write(f"{m:<15} | {mean_kin:.4f}       | {mean_nn:.4f}       | {mean_imp:+.2f}%\n")
            
            f.write("-" * 75 + "\n\n")
            f.write("Note: \n")
            f.write("  ADE = 平均位移误差,计算预测轨迹上这几十个点与真值点的平均欧氏距离。\n")
            f.write("  FDE = 终点位移误差,最后一步的位置误差\n")
            f.write("  MAE = 平均绝对误差\n\n")

            # --- Part 2: Detailed Logs (Optional, CSV style inside TXT) ---
            f.write("2. DETAILED SAMPLE LOGS\n")
            f.write("-" * 160 + "\n")
            # 构建表头
            cols = [
                'SampleID', 
                'NN_ADE', 'Kin_ADE', 'Imp_ADE', 
                'NN_FDE', 'Kin_FDE', 'Imp_FDE',
                'NN_MAE_v', 'Kin_MAE_v', 'Imp_MAE_v', 
                'NN_MAE_w', 'Kin_MAE_w', 'Imp_MAE_w'  
            ]
            header = "".join([f"{c:<12}" for c in cols])
            f.write(header + "\n")
            f.write("-" * 160 + "\n")
            
            for _, row in df.iterrows():
                line = ""
                for c in cols:
                    val = row[c]
                    if isinstance(val, float):
                        line += f"{val:<12.4f}"
                    else:
                        line += f"{val:<12}"
                f.write(line + "\n")
            
            f.write("-" * 160 + "\n")

        print(f"[Success] Evaluation report saved to: {save_path}")
        
        # 顺便存一个 csv 方便后续画图分析
        csv_path = os.path.join(output_dir, 'evaluation_metrics.csv')
        df.to_csv(csv_path, index=False)
        print(f"[Success] Raw metrics CSV saved to: {csv_path}")



# ==============================================================================
# 5. Main (主入口)
# ==============================================================================
def main():
    cfg = Config()
    
    # 0. 检查输出目录
    if not os.path.exists(cfg.OUTPUT_IMG_DIR):
        os.makedirs(cfg.OUTPUT_IMG_DIR)
        print(f"[Info] Created output directory: {cfg.OUTPUT_IMG_DIR}")

    # 1. 数据处理
    processor = BagProcessor(cfg)
    try:
        df_processed = processor.process_data()
    except Exception as e:
        print(f"[Fatal] {e}")
        return
    

    # 2. 模型加载
    predictor = DynamicsPredictor(cfg)
    evaluator = Evaluator() # 实例化评估器


    # 3. 生成测试索引序列
    # 起点: INPUT_TIMESTEPS + 10 (保证有足够的预热数据)
    # 终点: 数据末尾 - 预测时长
    # 步长: TEST_STEP_STRIDE
    start_offset = cfg.INPUT_TIMESTEPS + 10
    max_idx = len(df_processed) - cfg.PREDICT_STEPS
    
    # 生成所有可能的候选索引
    candidate_indices = list(range(start_offset, max_idx, cfg.TEST_STEP_STRIDE))
    
    # 截取前 N_TEST_SAMPLES 个
    test_indices = candidate_indices[:cfg.N_TEST_SAMPLES]
    
    print(f"\n[Info] Ready to run batch prediction.")
    print(f"       Total Samples: {len(test_indices)}")
    print(f"       Step Stride:   {cfg.TEST_STEP_STRIDE}")
    print(f"       Output Dir:    {cfg.OUTPUT_IMG_DIR}")

    # 4. 批量推理循环
    for i, idx in enumerate(test_indices):
        print(f"--- Processing ({i+1}/{len(test_indices)}) @ index {idx} (Time: {df_processed.iloc[idx]['timestamp']:.2f}) ---")
        
        traj_nn, traj_kin, traj_gt = predictor.predict_trajectory(idx, df_processed)
        
        if traj_nn is None:
            print("[Warn] Trajectory prediction failed (likely end of data). Stopping.")
            break
            
        # 绘图并保存 (传入当前的索引作为ID)
        Visualizer.save_comparison(traj_nn, traj_kin, traj_gt, cfg.TARGET_DT, cfg.OUTPUT_IMG_DIR, idx)

        # 计算并记录指标
        evaluator.record_sample(idx, traj_nn, traj_kin, traj_gt)

    print("\n[Done] Batch prediction finished.")

    evaluator.save_report(cfg.OUTPUT_IMG_DIR)

if __name__ == "__main__":
    main()