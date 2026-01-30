#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch GRU 动力学模型定义

本模块定义了与 Keras GRU 模型完全等价的 PyTorch 实现。
用于 MPPI 控制器中的车辆动力学预测，支持 GPU 加速和 JIT 编译优化。

模型架构（与 Keras 训练模型严格一致）:
    - GRU Layer 1: input_size=10, hidden_size=96, return_sequences=True
    - Dropout 1: p=0.2 (推理时关闭)
    - GRU Layer 2: input_size=96, hidden_size=96, return_sequences=False
    - Dropout 2: p=0.2 (推理时关闭)
    - Dense Layer 1: 96 -> 96, activation=tanh
    - Dense Layer 2: 96 -> 2, activation=linear

输入: [batch, seq_len=25, features=10]
输出: [batch, 2] -> [diff_v, diff_w]

Author: MPPI Project
Date: 2024
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class PyTorchGRUModel(nn.Module):
    """
    PyTorch GRU 动力学预测模型
    
    该模型用于预测车辆速度变化量 [diff_v, diff_w]，
    架构与 Keras 训练模型完全一致，确保权重转换后输出相同。
    
    Attributes:
        gru1 (nn.GRU): 第一层 GRU，输出完整序列
        dropout1 (nn.Dropout): 第一层 Dropout（推理时关闭）
        gru2 (nn.GRU): 第二层 GRU，只取最后时间步输出
        dropout2 (nn.Dropout): 第二层 Dropout（推理时关闭）
        dense1 (nn.Linear): 第一层全连接，带 tanh 激活
        dense2 (nn.Linear): 第二层全连接，线性输出
    
    Example:
        >>> model = PyTorchGRUModel()
        >>> model.eval()  # 推理模式，关闭 Dropout
        >>> x = torch.randn(500, 25, 10)  # [batch, seq, features]
        >>> output = model(x)  # [500, 2]
    """
    
    # 模型超参数（与 Keras 训练配置一致）
    INPUT_SIZE = 10      # 输入特征维度
    HIDDEN_SIZE = 96     # GRU 隐藏层维度
    OUTPUT_SIZE = 2      # 输出维度 [diff_v, diff_w]
    DROPOUT_RATE = 0.2   # Dropout 比率（仅训练时使用）
    
    def __init__(
        self,
        input_size: int = INPUT_SIZE,
        hidden_size: int = HIDDEN_SIZE,
        output_size: int = OUTPUT_SIZE,
        dropout_rate: float = DROPOUT_RATE
    ):
        """
        初始化 PyTorch GRU 模型
        
        Args:
            input_size: 输入特征维度，默认 10
            hidden_size: GRU 隐藏层维度，默认 96
            output_size: 输出维度，默认 2
            dropout_rate: Dropout 比率，默认 0.2
        """
        super(PyTorchGRUModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # ============================================================
        # Layer 0: GRU Layer 1
        # Keras: GRU(96, return_sequences=True)
        # PyTorch: batch_first=True 使输入格式为 [batch, seq, features]
        # ============================================================
        self.gru1 = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,  # 输入格式 [batch, seq, features]
            bidirectional=False
        )
        
        # ============================================================
        # Layer 1: Dropout 1 (仅训练时生效)
        # ============================================================
        self.dropout1 = nn.Dropout(p=dropout_rate)
        
        # ============================================================
        # Layer 2: GRU Layer 2
        # Keras: GRU(96, return_sequences=False)
        # 只使用最后一个时间步的输出
        # ============================================================
        self.gru2 = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )
        
        # ============================================================
        # Layer 3: Dropout 2 (仅训练时生效)
        # ============================================================
        self.dropout2 = nn.Dropout(p=dropout_rate)
        
        # ============================================================
        # Layer 4: Dense Layer 1 (96 -> 96, activation=tanh)
        # 注意：Keras Dense 权重格式为 [in, out]
        #       PyTorch Linear 权重格式为 [out, in]，需要转置
        # ============================================================
        self.dense1 = nn.Linear(hidden_size, hidden_size)
        
        # ============================================================
        # Layer 5: Dense Layer 2 (96 -> 2, activation=linear)
        # ============================================================
        self.dense2 = nn.Linear(hidden_size, output_size)
    
    def forward(
        self,
        x: torch.Tensor,
        h1: Optional[torch.Tensor] = None,
        h2: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入张量，形状 [batch, seq_len, input_size]
               seq_len 通常为 25（历史窗口长度）
            h1: GRU1 的初始隐藏状态，可选
            h2: GRU2 的初始隐藏状态，可选
        
        Returns:
            output: 预测输出 [batch, output_size]，即 [diff_v, diff_w]
            h1_out: GRU1 最终隐藏状态，用于有状态推理
            h2_out: GRU2 最终隐藏状态，用于有状态推理
        
        Note:
            - 推理时务必调用 model.eval() 关闭 Dropout
            - 训练时调用 model.train() 启用 Dropout
        """
        batch_size = x.size(0)
        
        # GRU1: 输出完整序列
        # out1: [batch, seq_len, hidden_size]
        # h1_out: [1, batch, hidden_size]
        if h1 is None:
            out1, h1_out = self.gru1(x)
        else:
            out1, h1_out = self.gru1(x, h1)
        
        # Dropout1 (仅训练时生效)
        out1 = self.dropout1(out1)
        
        # GRU2: 只取最后时间步
        # out2: [batch, seq_len, hidden_size]
        # h2_out: [1, batch, hidden_size]
        if h2 is None:
            out2, h2_out = self.gru2(out1)
        else:
            out2, h2_out = self.gru2(out1, h2)
        
        # 只取最后一个时间步的输出
        # last_output: [batch, hidden_size]
        last_output = out2[:, -1, :]
        
        # Dropout2 (仅训练时生效)
        last_output = self.dropout2(last_output)
        
        # Dense1 + tanh 激活
        # Keras 的 Dense(activation='tanh') 等价于 Linear + tanh
        dense1_out = torch.tanh(self.dense1(last_output))
        
        # Dense2 (线性输出，无激活函数)
        output = self.dense2(dense1_out)
        
        return output, h1_out, h2_out
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        简化的预测接口（仅返回输出，忽略隐藏状态）
        
        Args:
            x: 输入张量，形状 [batch, seq_len, input_size]
        
        Returns:
            output: 预测输出 [batch, output_size]
        
        Note:
            此方法会自动设置 eval 模式并禁用梯度计算
        """
        self.eval()
        with torch.no_grad():
            output, _, _ = self.forward(x)
        return output


def load_pytorch_model(
    model_path: str,
    device: str = "cpu"
) -> PyTorchGRUModel:
    """
    加载预训练的 PyTorch GRU 模型
    
    Args:
        model_path: .pt 模型文件路径
        device: 运行设备 ("cpu" 或 "cuda")
    
    Returns:
        model: 加载权重后的 PyTorchGRUModel 实例，已设置为 eval 模式
    
    Raises:
        FileNotFoundError: 模型文件不存在
        RuntimeError: 权重加载失败
    
    Example:
        >>> model = load_pytorch_model("pytorch_model.pt", device="cuda")
        >>> model.eval()
        >>> output = model.predict(input_tensor)
    """
    import os
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    # 创建模型实例
    model = PyTorchGRUModel()
    
    # 加载权重（添加 weights_only=True 消除警告）
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    
    # 移动到目标设备
    model = model.to(device)
    
    # 设置为评估模式（关闭 Dropout）
    model.eval()
    
    print(f"[PyTorchGRUModel] 模型加载成功: {model_path}")
    print(f"[PyTorchGRUModel] 运行设备: {device}")
    
    return model


if __name__ == "__main__":
    """模块测试入口"""
    print("=" * 60)
    print("PyTorch GRU 模型架构测试")
    print("=" * 60)
    
    # 创建模型
    model = PyTorchGRUModel()
    model.eval()
    
    # 打印模型结构
    print("\n模型结构:")
    print(model)
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    
    # 测试前向传播
    print("\n前向传播测试:")
    batch_size = 500
    seq_len = 25
    input_size = 10
    
    x = torch.randn(batch_size, seq_len, input_size)
    print(f"输入形状: {x.shape}")
    
    with torch.no_grad():
        output, h1, h2 = model(x)
    
    print(f"输出形状: {output.shape}")
    print(f"GRU1 隐藏状态形状: {h1.shape}")
    print(f"GRU2 隐藏状态形状: {h2.shape}")
    
    # 测试简化预测接口
    print("\n简化预测接口测试:")
    output = model.predict(x)
    print(f"预测输出形状: {output.shape}")
    print(f"预测输出范例 (前5个): {output[:5]}")
    
    print("\n" + "=" * 60)
    print("测试通过！")
    print("=" * 60)
