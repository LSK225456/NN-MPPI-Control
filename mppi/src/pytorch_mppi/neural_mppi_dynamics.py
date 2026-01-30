#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
神经网络动力学模型封装模块（重构版 - 支持批量 Rollout）
用于 MPPI 控制器中的车辆动力学预测

支持三种后端:
    1. Keras/TensorFlow 后端 (NeuralDynamics) - 原始实现，兼容性好
    2. PyTorch 后端 (NeuralDynamicsPyTorch) - 高性能实现
    3. PyTorch 批量 Rollout 后端 (NeuralDynamicsBatchRollout) - 终极优化

特征顺序（10维，与训练时严格一致）:
[v, w, dt, cmd_v, cmd_w, a_v, a_w, err_v, err_w, v_x_w]

输出（2维）:
[diff_v, diff_w] - 速度变化量（残差预测）
"""

import numpy as np
import joblib
import os
import sys

# 延迟导入 TensorFlow，避免启动时卡住
_tf_model = None
_tf_loaded = False


def _load_tensorflow():
    """延迟加载 TensorFlow"""
    global _tf_loaded
    if not _tf_loaded:
        import tensorflow as tf
        tf.config.threading.set_intra_op_parallelism_threads(4)
        tf.config.threading.set_inter_op_parallelism_threads(2)
        _tf_loaded = True
    return __import__('tensorflow')


def physics_weighted_mse(y_true, y_pred):
    """自定义物理加权损失函数"""
    tf = _load_tensorflow()
    squared_diff = tf.square(y_true - y_pred)
    weights = tf.constant([1.0, 1.0060286442927682])
    weighted_squared_diff = squared_diff * weights
    return tf.keras.backend.mean(weighted_squared_diff, axis=-1)


# 导入抽象基类
try:
    from dynamics_interface import DynamicsModel
except ImportError:
    from pytorch_mppi.dynamics_interface import DynamicsModel


# ============================================================================
# Keras 后端（保持不变）
# ============================================================================
class NeuralDynamics(DynamicsModel):
    """GRU 神经网络动力学模型 (Keras/TensorFlow 后端)"""
    
    # 固定参数（与训练时一致）
    SEQUENCE_LENGTH = 25  # 历史窗口长度
    NUM_FEATURES = 10     # 输入特征维度
    NUM_OUTPUTS = 2       # 输出维度 [diff_v, diff_w]
    
    def __init__(
        self,
        model_path: str,
        scaler_x_path: str,
        scaler_y_path: str,
        num_particles: int = 500
    ):
        self.num_particles = num_particles
        self._load_model(model_path)
        self._load_scalers(scaler_x_path, scaler_y_path)
        
        self.real_history = np.zeros(
            (self.SEQUENCE_LENGTH, self.NUM_FEATURES), 
            dtype=np.float32
        )
        self.rollout_buffer = np.zeros(
            (num_particles, self.SEQUENCE_LENGTH, self.NUM_FEATURES),
            dtype=np.float32
        )
        self.particle_poses = np.zeros((num_particles, 5), dtype=np.float32)
        
        print(f"[NeuralDynamics] 初始化完成 (Keras 后端)")
        print(f"  - 粒子数: {num_particles}")
        print(f"  - 历史窗口: {self.SEQUENCE_LENGTH}")
        print(f"  - 特征维度: {self.NUM_FEATURES}")
        print(f"  - 时间步长: {self.DT}s")
    
    def _load_model(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        tf = _load_tensorflow()
        custom_objects = {
            'physics_weighted_mse': physics_weighted_mse,
            'physics_weighted_mse_dynamic': physics_weighted_mse
        }
        self.model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        print(f"[NeuralDynamics] 模型加载成功: {model_path}")
    
    def _load_scalers(self, scaler_x_path: str, scaler_y_path: str):
        if not os.path.exists(scaler_x_path):
            raise FileNotFoundError(f"X 标准化器不存在: {scaler_x_path}")
        if not os.path.exists(scaler_y_path):
            raise FileNotFoundError(f"Y 标准化器不存在: {scaler_y_path}")
        self.scaler_x = joblib.load(scaler_x_path)
        self.scaler_y = joblib.load(scaler_y_path)
        print(f"[NeuralDynamics] 标准化器加载成功")
    
    def reset(self, current_state: np.ndarray):
        self.rollout_buffer[:] = self.real_history[np.newaxis, :, :]
        self.particle_poses[:] = current_state[np.newaxis, :]
        self.rollout_buffer[:, -1, 0] = current_state[3]
        self.rollout_buffer[:, -1, 1] = current_state[4]
    
    def step(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        K = self.num_particles
        v_curr = self.rollout_buffer[:, -1, 0].copy()
        w_curr = self.rollout_buffer[:, -1, 1].copy()
        cmd_v = action[:, 0]
        cmd_w = action[:, 1]
        
        v_prev = self.rollout_buffer[:, -2, 0]
        w_prev = self.rollout_buffer[:, -2, 1]
        a_v = np.clip((v_curr - v_prev) / self.DT, -20.0, 20.0)
        a_w = np.clip((w_curr - w_prev) / self.DT, -20.0, 20.0)
        err_v = cmd_v - v_curr
        err_w = cmd_w - w_curr
        v_x_w = v_curr * w_curr
        
        new_frame = np.stack([
            v_curr, w_curr, np.full(K, self.DT, dtype=np.float32),
            cmd_v, cmd_w, a_v, a_w, err_v, err_w, v_x_w
        ], axis=1).astype(np.float32)
        
        self.rollout_buffer = np.roll(self.rollout_buffer, -1, axis=1)
        self.rollout_buffer[:, -1, :] = new_frame
        
        buffer_flat = self.rollout_buffer.reshape(-1, self.NUM_FEATURES)
        buffer_scaled = self.scaler_x.transform(buffer_flat)
        buffer_scaled = buffer_scaled.reshape(K, self.SEQUENCE_LENGTH, self.NUM_FEATURES)
        
        pred_scaled = self.model.predict(buffer_scaled, verbose=0)
        delta = self.scaler_y.inverse_transform(pred_scaled)
        diff_v = delta[:, 0].astype(np.float32)
        diff_w = delta[:, 1].astype(np.float32)
        
        v_next = v_curr + diff_v
        w_next = w_curr + diff_w
        
        x = self.particle_poses[:, 0]
        y = self.particle_poses[:, 1]
        theta = self.particle_poses[:, 2]
        
        x_next = x + v_next * np.cos(theta) * self.DT
        y_next = y + v_next * np.sin(theta) * self.DT
        theta_next = np.arctan2(np.sin(theta + w_next * self.DT), np.cos(theta + w_next * self.DT))
        
        self.particle_poses[:, 0] = x_next
        self.particle_poses[:, 1] = y_next
        self.particle_poses[:, 2] = theta_next
        self.particle_poses[:, 3] = v_next
        self.particle_poses[:, 4] = w_next
        
        self.rollout_buffer[:, -1, 0] = v_next
        self.rollout_buffer[:, -1, 1] = w_next
        
        return np.stack([x_next, y_next, theta_next, v_next, w_next], axis=1).astype(np.float32)
    
    def update_real_history(self, v, w, cmd_v, cmd_w, a_v, a_w, err_v, err_w, v_x_w):
        new_frame = np.array([v, w, self.DT, cmd_v, cmd_w, a_v, a_w, err_v, err_w, v_x_w], dtype=np.float32)
        self.real_history = np.roll(self.real_history, -1, axis=0)
        self.real_history[-1] = new_frame


# ============================================================================
# PyTorch 后端（优化版本）
# ============================================================================
class NeuralDynamicsPyTorch(DynamicsModel):
    """
    GRU 神经网络动力学模型 (PyTorch 后端) - 优化版本
    
    优化点：
        1. 预分配所有缓冲区，避免运行时内存分配
        2. 使用索引操作代替 np.roll
        3. 向量化标准化操作
        4. 可选 torch.compile() JIT 优化
    """
    
    SEQUENCE_LENGTH = 25
    NUM_FEATURES = 10
    NUM_OUTPUTS = 2
    
    def __init__(
        self,
        model_path: str,
        scaler_x_path: str,
        scaler_y_path: str,
        num_particles: int = 500,
        device: str = "cpu",
        use_compile: bool = False  # 是否使用 torch.compile 优化
    ):
        import torch
        
        self.num_particles = num_particles
        self.device = device
        self.torch = torch
        self.use_compile = use_compile
        
        self._load_model(model_path)
        self._load_scalers(scaler_x_path, scaler_y_path)
        
        # ================================================================
        # 预分配所有缓冲区（关键优化）
        # ================================================================
        self.real_history = np.zeros(
            (self.SEQUENCE_LENGTH, self.NUM_FEATURES), 
            dtype=np.float32
        )
        
        # 使用环形缓冲区索引代替 np.roll
        self.rollout_buffer = np.zeros(
            (num_particles, self.SEQUENCE_LENGTH, self.NUM_FEATURES),
            dtype=np.float32
        )
        
        self.particle_poses = np.zeros((num_particles, 5), dtype=np.float32)
        
        # 预分配中间缓冲区
        self._new_frame = np.zeros((num_particles, self.NUM_FEATURES), dtype=np.float32)
        self._buffer_scaled = np.zeros(
            (num_particles, self.SEQUENCE_LENGTH, self.NUM_FEATURES), 
            dtype=np.float32
        )
        self._delta = np.zeros((num_particles, 2), dtype=np.float32)
        
        # 预分配 PyTorch Tensor（复用，避免重复创建）
        self._input_tensor = torch.zeros(
            (num_particles, self.SEQUENCE_LENGTH, self.NUM_FEATURES),
            dtype=torch.float32, device=device
        )
        
        print(f"[NeuralDynamicsPyTorch] 初始化完成")
        print(f"  - 后端: PyTorch (优化版)")
        print(f"  - 设备: {device}")
        print(f"  - 粒子数: {num_particles}")
        print(f"  - JIT编译: {'启用' if use_compile else '禁用'}")
    
    def _load_model(self, model_path: str):
        import torch
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        model_def_path = os.path.join(os.path.dirname(__file__), '../../tests')
        if model_def_path not in sys.path:
            sys.path.insert(0, model_def_path)
        
        from pytorch_gru_model import PyTorchGRUModel
        
        self.model = PyTorchGRUModel()
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device, weights_only=True)
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # 可选：使用 torch.compile 进行 JIT 优化 (PyTorch 2.x)
        if self.use_compile:
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
                print(f"[NeuralDynamicsPyTorch] torch.compile 优化已启用")
            except Exception as e:
                print(f"[NeuralDynamicsPyTorch] torch.compile 失败: {e}")
        
        print(f"[NeuralDynamicsPyTorch] 模型加载成功: {model_path}")
    
    def _load_scalers(self, scaler_x_path: str, scaler_y_path: str):
        if not os.path.exists(scaler_x_path):
            raise FileNotFoundError(f"X 标准化器不存在: {scaler_x_path}")
        if not os.path.exists(scaler_y_path):
            raise FileNotFoundError(f"Y 标准化器不存在: {scaler_y_path}")
        
        scaler_x = joblib.load(scaler_x_path)
        scaler_y = joblib.load(scaler_y_path)
        
        self.scaler_x_mean = scaler_x.mean_.astype(np.float32)
        self.scaler_x_scale = scaler_x.scale_.astype(np.float32)
        self.scaler_y_mean = scaler_y.mean_.astype(np.float32)
        self.scaler_y_scale = scaler_y.scale_.astype(np.float32)
        
        print(f"[NeuralDynamicsPyTorch] 标准化器加载成功")
    
    def reset(self, current_state: np.ndarray):
        """重置粒子缓冲区"""
        self.rollout_buffer[:] = self.real_history[np.newaxis, :, :]
        self.particle_poses[:] = current_state[np.newaxis, :]
        self.rollout_buffer[:, -1, 0] = current_state[3]
        self.rollout_buffer[:, -1, 1] = current_state[4]
    
    def step(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """
        优化的单步推理
        
        优化点：
            1. 使用预分配缓冲区
            2. 原地操作减少内存分配
            3. 复用 PyTorch Tensor
        """
        K = self.num_particles
        
        # ================================================================
        # 1. 获取当前状态（避免 .copy()）
        # ================================================================
        v_curr = self.rollout_buffer[:, -1, 0]
        w_curr = self.rollout_buffer[:, -1, 1]
        cmd_v = action[:, 0]
        cmd_w = action[:, 1]
        
        # ================================================================
        # 2. 计算派生特征（原地操作）
        # ================================================================
        v_prev = self.rollout_buffer[:, -2, 0]
        w_prev = self.rollout_buffer[:, -2, 1]
        
        a_v = np.clip((v_curr - v_prev) / self.DT, -20.0, 20.0)
        a_w = np.clip((w_curr - w_prev) / self.DT, -20.0, 20.0)
        err_v = cmd_v - v_curr
        err_w = cmd_w - w_curr
        v_x_w = v_curr * w_curr
        
        # ================================================================
        # 3. 构建新帧（使用预分配缓冲区）
        # ================================================================
        self._new_frame[:, 0] = v_curr
        self._new_frame[:, 1] = w_curr
        self._new_frame[:, 2] = self.DT
        self._new_frame[:, 3] = cmd_v
        self._new_frame[:, 4] = cmd_w
        self._new_frame[:, 5] = a_v
        self._new_frame[:, 6] = a_w
        self._new_frame[:, 7] = err_v
        self._new_frame[:, 8] = err_w
        self._new_frame[:, 9] = v_x_w
        
        # ================================================================
        # 4. 滚动缓冲区（使用切片赋值代替 np.roll）
        # ================================================================
        self.rollout_buffer[:, :-1, :] = self.rollout_buffer[:, 1:, :]
        self.rollout_buffer[:, -1, :] = self._new_frame
        
        # ================================================================
        # 5. 标准化（使用预分配缓冲区，原地操作）
        # ================================================================
        np.subtract(self.rollout_buffer, self.scaler_x_mean, out=self._buffer_scaled)
        np.divide(self._buffer_scaled, self.scaler_x_scale, out=self._buffer_scaled)
        
        # ================================================================
        # 6. PyTorch 推理（复用 Tensor）
        # ================================================================
        # 直接复制数据到预分配的 Tensor
        self._input_tensor.copy_(self.torch.from_numpy(self._buffer_scaled))
        
        with self.torch.no_grad():
            pred_scaled = self.model.predict(self._input_tensor)
        
        # ================================================================
        # 7. 反标准化（使用预分配缓冲区）
        # ================================================================
        pred_np = pred_scaled.numpy()  # CPU 上不需要 .cpu()
        np.multiply(pred_np, self.scaler_y_scale, out=self._delta)
        np.add(self._delta, self.scaler_y_mean, out=self._delta)
        
        diff_v = self._delta[:, 0]
        diff_w = self._delta[:, 1]
        
        # ================================================================
        # 8. 更新速度和位置
        # ================================================================
        v_next = v_curr + diff_v
        w_next = w_curr + diff_w
        
        x = self.particle_poses[:, 0]
        y = self.particle_poses[:, 1]
        theta = self.particle_poses[:, 2]
        
        x_next = x + v_next * np.cos(theta) * self.DT
        y_next = y + v_next * np.sin(theta) * self.DT
        theta_next = theta + w_next * self.DT
        theta_next = np.arctan2(np.sin(theta_next), np.cos(theta_next))
        
        # 更新粒子状态
        self.particle_poses[:, 0] = x_next
        self.particle_poses[:, 1] = y_next
        self.particle_poses[:, 2] = theta_next
        self.particle_poses[:, 3] = v_next
        self.particle_poses[:, 4] = w_next
        
        # 同步缓冲区中的速度
        self.rollout_buffer[:, -1, 0] = v_next
        self.rollout_buffer[:, -1, 1] = w_next
        
        return np.stack([x_next, y_next, theta_next, v_next, w_next], axis=1).astype(np.float32)
    
    def update_real_history(self, v, w, cmd_v, cmd_w, a_v, a_w, err_v, err_w, v_x_w):
        """更新真实历史（使用切片代替 np.roll）"""
        self.real_history[:-1, :] = self.real_history[1:, :]
        self.real_history[-1] = [v, w, self.DT, cmd_v, cmd_w, a_v, a_w, err_v, err_w, v_x_w]


# ============================================================================
# 移除无效的 NeuralDynamicsBatchRollout 类
# 原因：GRU 的顺序依赖特性使其无法真正实现批量 rollout
# ============================================================================

# ============================================================================
# 向后兼容的工厂函数接口
# ============================================================================
def create_neural_dynamics_function(nn_model):
    """创建与 MPPI 兼容的动力学函数包装器"""
    import torch
    
    def dynamics_func(state: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
        control_np = control.cpu().numpy()
        state_np = state.cpu().numpy()
        next_states_np = nn_model.step(state_np, control_np)
        return torch.tensor(next_states_np, dtype=state.dtype, device=state.device)
    
    return dynamics_func