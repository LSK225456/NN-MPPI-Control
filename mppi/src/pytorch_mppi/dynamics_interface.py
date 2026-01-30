#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
动力学模型接口定义
设计模式：策略模式，支持运动学模型、神经网络模型、混合模型的灵活切换

支持的后端:
    - kinematic: 简化运动学模型（默认）
    - keras: Keras/TensorFlow 神经网络后端
    - pytorch: PyTorch 神经网络后端（推荐，高性能）
"""

from abc import ABC, abstractmethod
import numpy as np


class DynamicsModel(ABC):
    """动力学模型抽象基类"""
    
    DT = 0.05  # 固定时间步长 (与神经网络训练时一致)
    
    @abstractmethod
    def reset(self, current_state: np.ndarray):
        """
        每个控制周期开始时重置状态
        
        Args:
            current_state: [5] -> [x, y, θ, v, w]
        """
        pass
    
    @abstractmethod
    def step(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """
        单步推理
        
        Args:
            state: [K, 5] -> [x, y, θ, v, w]
            action: [K, 2] -> [cmd_v, cmd_w]
            
        Returns:
            next_state: [K, 5]
        """
        pass
    
    def update_real_history(self, *args, **kwargs):
        """更新真实历史缓冲区（仅神经网络模型需要实现）"""
        pass


class KinematicDynamics(DynamicsModel):
    """一阶滞后运动学模型"""
    
    def __init__(self, tau_v=0.1, tau_w=0.05):
        """
        Args:
            tau_v: 线速度时间常数
            tau_w: 角速度时间常数
        """
        self.tau_v = tau_v
        self.tau_w = tau_w
    
    def reset(self, current_state: np.ndarray):
        """运动学模型无状态，不需要重置"""
        pass
    
    def step(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """
        一阶滞后运动学模型单步推理
        
        Args:
            state: [K, 5] -> [x, y, θ, v, w]
            action: [K, 2] -> [cmd_v, cmd_w]
            
        Returns:
            next_state: [K, 5]
        """
        # 状态提取
        x = state[:, 0]
        y = state[:, 1]
        theta = state[:, 2]
        v = state[:, 3]
        w = state[:, 4]
        
        # 控制量提取
        cmd_v = action[:, 0]
        cmd_w = action[:, 1]
        
        # 一阶滞后速度更新
        alpha_v = self.DT / (self.tau_v + self.DT)
        alpha_w = self.DT / (self.tau_w + self.DT)
        v_next = v + alpha_v * (cmd_v - v)
        w_next = w + alpha_w * (cmd_w - w)
        
        # 运动学位置更新
        x_next = x + v_next * np.cos(theta) * self.DT
        y_next = y + v_next * np.sin(theta) * self.DT
        theta_next = theta + w_next * self.DT
        
        # 角度归一化到 [-π, π]
        theta_next = np.arctan2(np.sin(theta_next), np.cos(theta_next))
        
        return np.stack([x_next, y_next, theta_next, v_next, w_next], axis=1).astype(np.float32)


def create_dynamics_model(config: dict, num_particles: int, horizon: int = 20) -> DynamicsModel:
    """
    工厂函数：根据配置创建动力学模型
    
    Args:
        config: 配置字典
        num_particles: MPPI 粒子数量
        horizon: 预测时域步数（保留参数，但当前未使用）
    
    Returns:
        DynamicsModel 实例
    """
    if not config.get('use_neural', False):
        print("[DynamicsFactory] 使用简化运动学模型")
        return KinematicDynamics()
    
    backend = config.get('backend', 'pytorch').lower()
    device = config.get('device', 'cpu')
    use_compile = config.get('use_compile', False)
    
    print(f"[DynamicsFactory] 请求的后端: {backend}")
    
    # ================================================================
    # PyTorch 后端（推荐）
    # ================================================================
    if backend in ['pytorch', 'pytorch_batch']:
        # pytorch_batch 已弃用，回退到 pytorch
        if backend == 'pytorch_batch':
            print("[DynamicsFactory] 警告: pytorch_batch 已弃用（GRU无法批量rollout），使用 pytorch")
        
        try:
            from neural_mppi_dynamics import NeuralDynamicsPyTorch
            
            pytorch_model_path = config.get('pytorch_model_path', '')
            if not pytorch_model_path:
                raise ValueError("未配置 pytorch_model_path")
            
            model = NeuralDynamicsPyTorch(
                model_path=pytorch_model_path,
                scaler_x_path=config['scaler_x_path'],
                scaler_y_path=config['scaler_y_path'],
                num_particles=num_particles,
                device=device,
                use_compile=use_compile
            )
            print("[DynamicsFactory] 使用 PyTorch 神经网络动力学模型")
            return model
            
        except Exception as e:
            print(f"[DynamicsFactory] PyTorch 后端加载失败: {e}")
            import traceback
            traceback.print_exc()
            backend = 'keras'
    
    # ================================================================
    # Keras 后端
    # ================================================================
    if backend == 'keras':
        try:
            from neural_mppi_dynamics import NeuralDynamics
            
            model = NeuralDynamics(
                model_path=config['model_path'],
                scaler_x_path=config['scaler_x_path'],
                scaler_y_path=config['scaler_y_path'],
                num_particles=num_particles
            )
            print("[DynamicsFactory] 使用 Keras 神经网络动力学模型")
            return model
            
        except Exception as e:
            print(f"[DynamicsFactory] Keras 后端加载失败: {e}")
            return KinematicDynamics()
    
    print(f"[DynamicsFactory] 未知后端 '{backend}'，回退到运动学模型")
    return KinematicDynamics()
