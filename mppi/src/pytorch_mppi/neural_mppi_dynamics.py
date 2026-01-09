#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Neural Dynamics Model Wrapper for MPPI Integration
神经网络动力学模型封装类 - 用于集成到MPPI控制器

功能：
1. 加载训练好的 GRU 模型和数据归一化器
2. 维护真实历史数据窗口（25帧）
3. 维护MPPI粒子推演历史 [K, 25, 10]
4. 提供向量化批量推理接口

"""

import os
import numpy as np
import joblib
from collections import deque
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K


# ==============================================================================
# 配置区
# ==============================================================================
class NeuralDynamicsConfig:
    """神经网络动力学模型配置类"""
    
    # --- 模型文件路径（必须修改为实际路径）---
    MODEL_PATH = '/path/to/your/keras_model_recurrent.h5'       # .h5 模型文件路径
    SCALER_X_PATH = '/path/to/your/scaler.plk'                  # 输入特征标准化器路径
    SCALER_Y_PATH = '/path/to/your/scaler_y.joblib'             # 输出标签标准化器路径
    
    # --- 模型结构参数（必须与训练时一致）---
    HISTORY_WINDOW_SIZE = 25        # GRU 历史窗口长度（训练时使用的 timesteps）
    INPUT_FEATURES = 10             # 输入特征维度 [v, w, dt, cmd_v, cmd_w, a_v, a_w, err_v, err_w, v_x_w]
    OUTPUT_FEATURES = 2             # 输出维度 [diff_v, diff_w]
    
    # --- 推理参数 ---
    FIXED_DT = 0.05                 # 固定时间步长（秒），必须与训练时一致，不可修改！
    
    # --- 初始化参数 ---
    STATIC_INIT_V = 0.0             # 静止状态初始化：线速度
    STATIC_INIT_W = 0.0             # 静止状态初始化：角速度
    STATIC_INIT_CMD_V = 0.0         # 静止状态初始化：指令线速度
    STATIC_INIT_CMD_W = 0.0         # 静止状态初始化：指令角速度
    
    # --- 性能优化参数 ---
    USE_MANUAL_SCALING = True       # 是否使用手动标准化（True=更快，False=使用sklearn scaler）
    ENABLE_WARMUP = True            # 是否在初始化时预热模型（避免首次推理卡顿）
    WARMUP_BATCH_SIZE = 100         # 预热时使用的批量大小
    
    # --- Debug 参数 ---
    VERBOSE = True                  # 是否打印详细日志


# ==============================================================================
# 自定义损失函数（必须与训练代码完全一致）
# ==============================================================================
def physics_weighted_mse(y_true, y_pred):
    """
    物理加权均方误差损失函数
    注意：权重必须与训练时完全一致，否则模型加载会失败
    """
    squared_diff = tf.square(y_true - y_pred)
    # 权重：[diff_v的权重, diff_w的权重]
    weights = tf.constant([1.0, 1.0060286442927682])
    weighted_squared_diff = squared_diff * weights
    return K.mean(weighted_squared_diff, axis=-1)


# ==============================================================================
# 神经网络动力学模型封装类
# ==============================================================================
class NeuralDynamicsModel:
    """
    神经网络动力学模型封装类
    
    主要功能：
    1. 加载 GRU 模型并进行预热
    2. 维护真实历史数据窗口（用于初始化MPPI推演）
    3. 维护粒子推演历史 [K, 25, 10]（用于多步预测）
    4. 提供批量推理接口 step_dynamics(states, actions)
    """
    
    def __init__(self, config=None, model_path=None, scaler_x_path=None, scaler_y_path=None, dt=None):
        """
        初始化神经网络动力学模型
        
        Args:
            config: NeuralDynamicsConfig 对象（可选，优先级最低）
            model_path: 模型文件路径（可选，优先级高于config）
            scaler_x_path: 输入特征标准化器路径（可选）
            scaler_y_path: 输出标签标准化器路径（可选）
            dt: 时间步长（可选，默认使用config中的值）
        """
        # 配置优先级：函数参数 > config对象 > 默认配置
        self.cfg = config if config is not None else NeuralDynamicsConfig()
        
        # 允许通过参数覆盖配置
        if model_path is not None:
            self.cfg.MODEL_PATH = model_path
        if scaler_x_path is not None:
            self.cfg.SCALER_X_PATH = scaler_x_path
        if scaler_y_path is not None:
            self.cfg.SCALER_Y_PATH = scaler_y_path
        if dt is not None:
            if abs(dt - self.cfg.FIXED_DT) > 1e-6:
                print(f"[Warning] 输入的 dt={dt} 与训练时的 FIXED_DT={self.cfg.FIXED_DT} 不一致！")
                print(f"[Warning] 强制使用 dt={self.cfg.FIXED_DT}，模型可能会预测不准确！")
        
        # 加载模型和标准化器
        self._load_model()
        self._load_scalers()
        
        # 初始化历史数据窗口（真实数据）
        self._init_real_history()
        
        # 初始化推演缓冲区（MPPI粒子历史）
        self.rollout_buffer = None  # 形状: [K, 25, 10]，会在 reset_particles 时初始化
        
        # 预热模型（避免首次推理卡顿）
        if self.cfg.ENABLE_WARMUP:
            self._warmup_model()
        
        if self.cfg.VERBOSE:
            print("[NeuralDynamicsModel] 初始化完成")
            print(f"  - 模型路径: {self.cfg.MODEL_PATH}")
            print(f"  - 历史窗口: {self.cfg.HISTORY_WINDOW_SIZE} 帧")
            print(f"  - 固定时间步: {self.cfg.FIXED_DT} 秒")
    
    # --------------------------------------------------------------------------
    # 私有方法：模型加载与初始化
    # --------------------------------------------------------------------------
    
    def _load_model(self):
        """加载 Keras 模型（注册自定义损失函数）"""
        if not os.path.exists(self.cfg.MODEL_PATH):
            raise FileNotFoundError(f"模型文件不存在: {self.cfg.MODEL_PATH}")
        
        if self.cfg.VERBOSE:
            print(f"[NeuralDynamicsModel] 正在加载模型: {self.cfg.MODEL_PATH}")
        
        # 注册自定义损失函数（必须与训练时一致）
        custom_objects = {
            'physics_weighted_mse': physics_weighted_mse,
            'physics_weighted_mse_dynamic': physics_weighted_mse  # 训练时可能用的别名
        }
        
        self.model = keras.models.load_model(
            self.cfg.MODEL_PATH,
            custom_objects=custom_objects
        )
        
        if self.cfg.VERBOSE:
            print(f"[NeuralDynamicsModel] 模型加载成功")
    
    def _load_scalers(self):
        """加载输入输出标准化器"""
        if not os.path.exists(self.cfg.SCALER_X_PATH):
            raise FileNotFoundError(f"输入标准化器不存在: {self.cfg.SCALER_X_PATH}")
        if not os.path.exists(self.cfg.SCALER_Y_PATH):
            raise FileNotFoundError(f"输出标准化器不存在: {self.cfg.SCALER_Y_PATH}")
        
        self.scaler_X = joblib.load(self.cfg.SCALER_X_PATH)
        self.scaler_y = joblib.load(self.cfg.SCALER_Y_PATH)
        
        # 如果使用手动标准化，预缓存 mean 和 scale（加速推理）
        if self.cfg.USE_MANUAL_SCALING:
            self.scaler_X_mean = self.scaler_X.mean_
            self.scaler_X_scale = self.scaler_X.scale_
            self.scaler_y_mean = self.scaler_y.mean_
            self.scaler_y_scale = self.scaler_y.scale_
            
            if self.cfg.VERBOSE:
                print("[NeuralDynamicsModel] 已启用手动标准化加速")
        
        if self.cfg.VERBOSE:
            print("[NeuralDynamicsModel] 标准化器加载成功")
    
    def _init_real_history(self):
        """
        初始化真实历史数据窗口（使用静止状态填充）
        
        真实历史用于记录机器人的实际运行状态，在每次MPPI规划前
        用于初始化所有粒子的历史窗口
        """
        self.real_history_deque = deque(maxlen=self.cfg.HISTORY_WINDOW_SIZE)
        
        # 构造静止状态特征向量 [v, w, dt, cmd_v, cmd_w, a_v, a_w, err_v, err_w, v_x_w]
        static_feature = [
            self.cfg.STATIC_INIT_V,       # v = 0
            self.cfg.STATIC_INIT_W,       # w = 0
            self.cfg.FIXED_DT,            # dt = 0.05
            self.cfg.STATIC_INIT_CMD_V,   # cmd_v = 0
            self.cfg.STATIC_INIT_CMD_W,   # cmd_w = 0
            0.0,                          # a_v = 0（加速度）
            0.0,                          # a_w = 0（角加速度）
            0.0,                          # err_v = 0（速度误差）
            0.0,                          # err_w = 0（角速度误差）
            0.0                           # v_x_w = 0（耦合项）
        ]
        
        # 填充25帧静止状态
        for _ in range(self.cfg.HISTORY_WINDOW_SIZE):
            self.real_history_deque.append(static_feature.copy())
        
        if self.cfg.VERBOSE:
            print(f"[NeuralDynamicsModel] 真实历史窗口已初始化（{self.cfg.HISTORY_WINDOW_SIZE}帧静止状态）")
    
    def _warmup_model(self):
        """
        模型预热：用随机数据进行一次推理，避免首次调用卡顿
        
        TensorFlow 在首次推理时会进行图优化和内存分配，导致延迟较高
        预热可以提前完成这些操作
        """
        if self.cfg.VERBOSE:
            print(f"[NeuralDynamicsModel] 正在预热模型（批量大小={self.cfg.WARMUP_BATCH_SIZE}）...")
        
        # 构造随机输入 [K, 25, 10]
        dummy_input = np.random.randn(
            self.cfg.WARMUP_BATCH_SIZE,
            self.cfg.HISTORY_WINDOW_SIZE,
            self.cfg.INPUT_FEATURES
        ).astype(np.float32)
        
        # 执行一次推理（不使用结果）
        _ = self.model.predict(dummy_input, verbose=0)
        
        if self.cfg.VERBOSE:
            print("[NeuralDynamicsModel] 模型预热完成")
    
    # --------------------------------------------------------------------------
    # 公有方法：数据更新接口
    # --------------------------------------------------------------------------
    
    def update_real_history(self, v, w, cmd_v, cmd_w):
        """
        更新真实历史数据窗口（在ROS回调中调用）
        
        每次发布控制指令后，调用此函数更新真实历史数据
        
        Args:
            v: 当前实际线速度（m/s）
            w: 当前实际角速度（rad/s）
            cmd_v: 当前控制指令线速度（m/s）
            cmd_w: 当前控制指令角速度（rad/s）
        """
        # 1. 计算加速度（需要历史数据）
        if len(self.real_history_deque) > 0:
            last_feature = self.real_history_deque[-1]
            last_v = last_feature[0]
            last_w = last_feature[1]
            a_v = (v - last_v) / self.cfg.FIXED_DT
            a_w = (w - last_w) / self.cfg.FIXED_DT
        else:
            a_v = 0.0
            a_w = 0.0
        
        # 2. 计算控制误差
        err_v = cmd_v - v
        err_w = cmd_w - w
        
        # 3. 计算动力学耦合项
        v_x_w = v * w
        
        # 4. 构造10维特征向量
        feature_vector = [
            v, w, self.cfg.FIXED_DT, cmd_v, cmd_w,
            a_v, a_w, err_v, err_w, v_x_w
        ]
        
        # 5. 加入历史窗口（自动丢弃最旧的一帧）
        self.real_history_deque.append(feature_vector)
    
    def reset_particles(self, batch_size):
        """
        重置粒子推演历史缓冲区（在每次MPPI规划前调用）
        
        将所有 K 个粒子的历史窗口重置为当前的真实历史
        
        Args:
            batch_size: MPPI 采样数量（K）
        """
        # 将 deque 转换为 numpy 数组 [25, 10]
        real_history_array = np.array(self.real_history_deque, dtype=np.float32)
        
        # 检查历史窗口是否已满
        if len(self.real_history_deque) < self.cfg.HISTORY_WINDOW_SIZE:
            print(f"[Warning] 真实历史窗口未满（{len(self.real_history_deque)}/{self.cfg.HISTORY_WINDOW_SIZE}）")
        
        # 复制 K 份，构建 [K, 25, 10] 张量
        self.rollout_buffer = np.tile(
            real_history_array[np.newaxis, :, :],  # [1, 25, 10]
            (batch_size, 1, 1)                     # 复制 K 次
        )
        
        if self.cfg.VERBOSE:
            print(f"[NeuralDynamicsModel] 粒子历史已重置（K={batch_size}）")
    
    # --------------------------------------------------------------------------
    # 核心推理接口
    # --------------------------------------------------------------------------
    
    def step_dynamics(self, states, actions):
        """
        神经网络动力学模型推理（批量向量化版本）
        
        这是替代 MPPI 中 dynamics_model 函数的核心接口
        
        Args:
            states: 当前状态张量 [K, 5]，格式为 [x, y, θ, v, w]
            actions: 控制输入张量 [K, 2]，格式为 [cmd_v, cmd_w]
        
        Returns:
            next_states: 下一步状态张量 [K, 5]，格式为 [x, y, θ, v, w]
        """
        # ----------------------------------------------------------------
        # Step 1: 输入检查与类型转换
        # ----------------------------------------------------------------
        if states.shape[1] != 5:
            raise ValueError(f"状态向量维度错误：期望 [K, 5]，实际 {states.shape}")
        if actions.shape[1] != 2:
            raise ValueError(f"动作向量维度错误：期望 [K, 2]，实际 {actions.shape}")
        
        K = states.shape[0]  # 粒子数量
        
        # 转换为 numpy（如果输入是 torch.Tensor）
        if hasattr(states, 'cpu'):  # PyTorch Tensor
            states_np = states.cpu().numpy()
            actions_np = actions.cpu().numpy()
        else:
            states_np = states
            actions_np = actions
        
        # ----------------------------------------------------------------
        # Step 2: 提取当前状态
        # ----------------------------------------------------------------
        x_curr = states_np[:, 0]      # [K]
        y_curr = states_np[:, 1]      # [K]
        theta_curr = states_np[:, 2]  # [K]
        v_curr = states_np[:, 3]      # [K]
        w_curr = states_np[:, 4]      # [K]
        
        cmd_v = actions_np[:, 0]      # [K]
        cmd_w = actions_np[:, 1]      # [K]
        
        # ----------------------------------------------------------------
        # Step 3: 构造10维特征向量（向量化计算）
        # ----------------------------------------------------------------
        # 从 rollout_buffer 获取历史加速度（上一帧的特征）
        last_features = self.rollout_buffer[:, -1, :]  # [K, 10]
        last_v = last_features[:, 0]  # [K]
        last_w = last_features[:, 1]  # [K]
        
        # 计算加速度
        a_v = (v_curr - last_v) / self.cfg.FIXED_DT  # [K]
        a_w = (w_curr - last_w) / self.cfg.FIXED_DT  # [K]
        
        # 计算控制误差
        err_v = cmd_v - v_curr  # [K]
        err_w = cmd_w - w_curr  # [K]
        
        # 计算耦合项
        v_x_w = v_curr * w_curr  # [K]
        
        # 构造当前时刻的10维特征 [K, 10]
        current_features = np.stack([
            v_curr, w_curr,
            np.full(K, self.cfg.FIXED_DT),  # dt 固定为 0.05
            cmd_v, cmd_w,
            a_v, a_w,
            err_v, err_w,
            v_x_w
        ], axis=1)  # [K, 10]
        
        # ----------------------------------------------------------------
        # Step 4: 标准化输入（向量化）
        # ----------------------------------------------------------------
        # 将 [K, 25, 10] 的 rollout_buffer reshape 成 [K*25, 10]
        input_features = self.rollout_buffer.reshape(-1, self.cfg.INPUT_FEATURES)  # [K*25, 10]
        
        if self.cfg.USE_MANUAL_SCALING:
            # 手动标准化（更快）
            input_scaled = (input_features - self.scaler_X_mean) / self.scaler_X_scale
        else:
            # 使用 sklearn scaler（较慢）
            input_scaled = self.scaler_X.transform(input_features)
        
        # Reshape 回 [K, 25, 10]
        input_scaled = input_scaled.reshape(K, self.cfg.HISTORY_WINDOW_SIZE, self.cfg.INPUT_FEATURES)
        
        # ----------------------------------------------------------------
        # Step 5: GRU 模型推理（批量推理，一次处理 K 条轨迹）
        # ----------------------------------------------------------------
        pred_scaled = self.model.predict(input_scaled, verbose=0)  # [K, 2]
        
        # ----------------------------------------------------------------
        # Step 6: 反标准化输出
        # ----------------------------------------------------------------
        if self.cfg.USE_MANUAL_SCALING:
            pred_diff = pred_scaled * self.scaler_y_scale + self.scaler_y_mean  # [K, 2]
        else:
            pred_diff = self.scaler_y.inverse_transform(pred_scaled)  # [K, 2]
        
        diff_v = pred_diff[:, 0]  # [K]
        diff_w = pred_diff[:, 1]  # [K]
        
        # ----------------------------------------------------------------
        # Step 7: 更新速度状态（残差网络）
        # ----------------------------------------------------------------
        v_next = v_curr + diff_v  # [K]
        w_next = w_curr + diff_w  # [K]
        
        # ----------------------------------------------------------------
        # Step 8: 运动学更新位置状态
        # ----------------------------------------------------------------
        x_next = x_curr + v_next * np.cos(theta_curr) * self.cfg.FIXED_DT  # [K]
        y_next = y_curr + v_next * np.sin(theta_curr) * self.cfg.FIXED_DT  # [K]
        theta_next = theta_curr + w_next * self.cfg.FIXED_DT  # [K]
        
        # 归一化角度到 [-π, π]
        theta_next = np.arctan2(np.sin(theta_next), np.cos(theta_next))  # [K]
        
        # ----------------------------------------------------------------
        # Step 9: 更新 rollout_buffer（滚动窗口）
        # ----------------------------------------------------------------
        # 方法：抛弃第一帧，加入新的一帧
        # rollout_buffer: [K, 25, 10] -> [K, 24, 10] + [K, 1, 10] -> [K, 25, 10]
        
        # 标准化 current_features（用于加入buffer）
        if self.cfg.USE_MANUAL_SCALING:
            current_features_scaled = (current_features - self.scaler_X_mean) / self.scaler_X_scale
        else:
            current_features_scaled = self.scaler_X.transform(current_features)
        
        # 滚动更新：丢弃第一帧，加入新的一帧
        self.rollout_buffer = np.concatenate([
            self.rollout_buffer[:, 1:, :],              # [K, 24, 10]（丢弃第一帧）
            current_features_scaled[:, np.newaxis, :]   # [K, 1, 10]（加入新帧）
        ], axis=1)  # [K, 25, 10]
        
        # ----------------------------------------------------------------
        # Step 10: 返回新状态
        # ----------------------------------------------------------------
        next_states = np.stack([x_next, y_next, theta_next, v_next, w_next], axis=1)  # [K, 5]
        
        return next_states


# ==============================================================================
# 测试代码（可选）
# ==============================================================================
if __name__ == "__main__":
    """
    独立测试脚本：验证模型能否正常加载和推理
    """
    print("=" * 70)
    print("神经网络动力学模型测试")
    print("=" * 70)
    
    # 1. 配置路径（需要修改为实际路径）
    config = NeuralDynamicsConfig()
    config.MODEL_PATH = '/home/lsk1804/lsk_graduate/lsk_ws/src/NeuralNetwork_for_VehicleDynamicsModeling/models/1216-optuna超参数调参-0.32/keras_model_recurrent.h5'
    config.SCALER_X_PATH = '/home/lsk1804/lsk_graduate/lsk_ws/src/NeuralNetwork_for_VehicleDynamicsModeling/models/1216-optuna超参数调参-0.32/scaler.plk'
    config.SCALER_Y_PATH = '/home/lsk1804/lsk_graduate/lsk_ws/src/NeuralNetwork_for_VehicleDynamicsModeling/models/1216-optuna超参数调参-0.32/scaler_y.joblib'
    config.VERBOSE = True
    
    # 2. 初始化模型
    try:
        nn_model = NeuralDynamicsModel(config=config)
        print("\n[SUCCESS] 模型加载成功！")
    except Exception as e:
        print(f"\n[ERROR] 模型加载失败: {e}")
        exit(1)
    
    # 3. 模拟更新真实历史（模拟机器人运行10步）
    print("\n" + "=" * 70)
    print("测试1：更新真实历史数据")
    print("=" * 70)
    for i in range(10):
        nn_model.update_real_history(
            v=0.5 + 0.1 * i,      # 线速度逐渐增加
            w=0.1,                # 角速度固定
            cmd_v=0.6,            # 指令线速度
            cmd_w=0.1             # 指令角速度
        )
    print(f"[SUCCESS] 真实历史更新完成（当前窗口长度: {len(nn_model.real_history_deque)}）")
    
    # 4. 测试粒子重置
    print("\n" + "=" * 70)
    print("测试2：重置粒子历史")
    print("=" * 70)
    K = 100  # MPPI 采样数
    nn_model.reset_particles(batch_size=K)
    print(f"[SUCCESS] 粒子历史重置完成（形状: {nn_model.rollout_buffer.shape}）")
    
    # 5. 测试批量推理
    print("\n" + "=" * 70)
    print("测试3：批量推理")
    print("=" * 70)
    
    # 构造随机状态和动作
    states = np.random.randn(K, 5).astype(np.float32)  # [K, 5]: [x, y, θ, v, w]
    states[:, 3] = 0.5  # 设置 v = 0.5
    states[:, 4] = 0.1  # 设置 w = 0.1
    
    actions = np.random.randn(K, 2).astype(np.float32)  # [K, 2]: [cmd_v, cmd_w]
    actions[:, 0] = 0.6  # cmd_v = 0.6
    actions[:, 1] = 0.1  # cmd_w = 0.1
    
    import time
    t_start = time.time()
    next_states = nn_model.step_dynamics(states, actions)
    t_end = time.time()
    
    print(f"[SUCCESS] 推理完成")
    print(f"  - 输入状态形状: {states.shape}")
    print(f"  - 输出状态形状: {next_states.shape}")
    print(f"  - 推理时间: {(t_end - t_start) * 1000:.2f} ms")
    print(f"  - 单个粒子平均时间: {(t_end - t_start) / K * 1000:.2f} ms")
    
    # 6. 测试多步推理性能
    print("\n" + "=" * 70)
    print("测试4：多步推理性能（模拟MPPI场景）")
    print("=" * 70)
    
    T = 30  # MPPI 预测步长
    total_time = 0.0
    
    for t in range(T):
        t_start = time.time()
        next_states = nn_model.step_dynamics(states, actions)
        t_end = time.time()
        total_time += (t_end - t_start)
        states = next_states  # 更新状态
    
    print(f"[SUCCESS] {T} 步推理完成")
    print(f"  - 总时间: {total_time * 1000:.2f} ms")
    print(f"  - 平均每步: {total_time / T * 1000:.2f} ms")
    print(f"  - 是否满足实时性（<50ms）: {'是' if total_time < 0.05 else '否（需要优化）'}")
    
    print("\n" + "=" * 70)
    print("所有测试完成！")
    print("=" * 70)