#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras GRU 模型权重转换到 PyTorch 格式

本模块实现了从 Keras (.h5) 模型到 PyTorch (.pt) 模型的权重转换。
核心难点是处理 GRU 门顺序的差异：
    - Keras GRU 门顺序: [z, r, h] (update, reset, new)
    - PyTorch GRU 门顺序: [r, z, n] (reset, update, new)

此外还需要处理：
    - 权重矩阵转置（Keras [in, out] -> PyTorch [out, in]）
    - Bias 的分离（Keras 分开存储 input_bias 和 recurrent_bias）

使用方法:
    python convert_keras_to_pytorch.py \
        --keras_model /path/to/keras_model.h5 \
        --output /path/to/pytorch_model.pt

Author: MPPI Project
Date: 2024
"""

import os
import sys
import argparse
import numpy as np
from typing import Dict, Tuple, List

# 添加 PyTorch 模型路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 
    '../../mppi/tests'))


def convert_gru_weights(
    keras_weights: List[np.ndarray],
    hidden_size: int = 96
) -> Dict[str, np.ndarray]:
    """
    转换单个 Keras GRU 层的权重到 PyTorch 格式
    
    Keras GRU 权重结构:
        - kernel (input weights): shape [input_size, 3 * hidden_size]
        - recurrent_kernel: shape [hidden_size, 3 * hidden_size]
        - bias: shape [2, 3 * hidden_size]
            - bias[0]: input_bias
            - bias[1]: recurrent_bias
    
    门顺序转换:
        - Keras: [z, r, h] -> update gate, reset gate, new gate
        - PyTorch: [r, z, n] -> reset gate, update gate, new gate
    
    PyTorch 权重命名:
        - weight_ih: input-hidden weights [3*hidden, input_size]
        - weight_hh: hidden-hidden weights [3*hidden, hidden_size]
        - bias_ih: input-hidden bias [3*hidden]
        - bias_hh: hidden-hidden bias [3*hidden]
    
    Args:
        keras_weights: Keras GRU 层的权重列表 [kernel, recurrent_kernel, bias]
        hidden_size: 隐藏层维度，默认 96
    
    Returns:
        pytorch_weights: 字典，包含 PyTorch GRU 层所需的 4 个权重张量
    
    Example:
        >>> keras_weights = [kernel, recurrent_kernel, bias]
        >>> pytorch_weights = convert_gru_weights(keras_weights, hidden_size=96)
        >>> # pytorch_weights = {
        >>> #     'weight_ih_l0': ...,
        >>> #     'weight_hh_l0': ...,
        >>> #     'bias_ih_l0': ...,
        >>> #     'bias_hh_l0': ...
        >>> # }
    """
    kernel = keras_weights[0]           # [input_size, 3*hidden_size]
    recurrent_kernel = keras_weights[1] # [hidden_size, 3*hidden_size]
    bias = keras_weights[2]             # [2, 3*hidden_size]
    
    input_size = kernel.shape[0]
    
    print(f"    Keras kernel shape: {kernel.shape}")
    print(f"    Keras recurrent_kernel shape: {recurrent_kernel.shape}")
    print(f"    Keras bias shape: {bias.shape}")
    
    # ================================================================
    # 1. 拆分 Keras 权重为各个门 [z, r, h]
    # ================================================================
    # kernel: [input_size, 3*hidden] -> 拆分为 3 个 [input_size, hidden]
    kernel_z = kernel[:, :hidden_size]          # update gate
    kernel_r = kernel[:, hidden_size:2*hidden_size]  # reset gate
    kernel_h = kernel[:, 2*hidden_size:]        # new gate
    
    # recurrent_kernel: [hidden, 3*hidden] -> 拆分为 3 个 [hidden, hidden]
    rec_z = recurrent_kernel[:, :hidden_size]
    rec_r = recurrent_kernel[:, hidden_size:2*hidden_size]
    rec_h = recurrent_kernel[:, 2*hidden_size:]
    
    # bias: [2, 3*hidden] -> input_bias 和 recurrent_bias
    input_bias = bias[0]      # [3*hidden]
    recurrent_bias = bias[1]  # [3*hidden]
    
    # 拆分 bias
    bias_z_i = input_bias[:hidden_size]
    bias_r_i = input_bias[hidden_size:2*hidden_size]
    bias_h_i = input_bias[2*hidden_size:]
    
    bias_z_h = recurrent_bias[:hidden_size]
    bias_r_h = recurrent_bias[hidden_size:2*hidden_size]
    bias_h_h = recurrent_bias[2*hidden_size:]
    
    # ================================================================
    # 2. 重组为 PyTorch 门顺序 [r, z, n]
    # ================================================================
    # weight_ih: [3*hidden, input_size] (需要转置)
    # PyTorch 顺序: [r, z, n]
    weight_ih = np.concatenate([
        kernel_r.T,  # reset gate
        kernel_z.T,  # update gate  
        kernel_h.T   # new gate
    ], axis=0)
    
    # weight_hh: [3*hidden, hidden_size] (需要转置)
    weight_hh = np.concatenate([
        rec_r.T,  # reset gate
        rec_z.T,  # update gate
        rec_h.T   # new gate
    ], axis=0)
    
    # bias_ih: [3*hidden]
    bias_ih = np.concatenate([
        bias_r_i,  # reset gate
        bias_z_i,  # update gate
        bias_h_i   # new gate
    ], axis=0)
    
    # bias_hh: [3*hidden]
    bias_hh = np.concatenate([
        bias_r_h,  # reset gate
        bias_z_h,  # update gate
        bias_h_h   # new gate
    ], axis=0)
    
    print(f"    PyTorch weight_ih shape: {weight_ih.shape}")
    print(f"    PyTorch weight_hh shape: {weight_hh.shape}")
    print(f"    PyTorch bias_ih shape: {bias_ih.shape}")
    print(f"    PyTorch bias_hh shape: {bias_hh.shape}")
    
    return {
        'weight_ih_l0': weight_ih.astype(np.float32),
        'weight_hh_l0': weight_hh.astype(np.float32),
        'bias_ih_l0': bias_ih.astype(np.float32),
        'bias_hh_l0': bias_hh.astype(np.float32)
    }


def convert_dense_weights(
    keras_weights: List[np.ndarray]
) -> Dict[str, np.ndarray]:
    """
    转换 Keras Dense 层权重到 PyTorch Linear 格式
    
    Keras Dense 权重格式:
        - kernel: [input_size, output_size]
        - bias: [output_size]
    
    PyTorch Linear 权重格式:
        - weight: [output_size, input_size] (需要转置)
        - bias: [output_size]
    
    Args:
        keras_weights: Keras Dense 层的权重列表 [kernel, bias]
    
    Returns:
        pytorch_weights: 字典，包含 weight 和 bias
    """
    kernel = keras_weights[0]  # [input, output]
    bias = keras_weights[1]    # [output]
    
    print(f"    Keras Dense kernel shape: {kernel.shape}")
    print(f"    Keras Dense bias shape: {bias.shape}")
    
    # 转置权重矩阵
    weight = kernel.T  # [output, input]
    
    print(f"    PyTorch Linear weight shape: {weight.shape}")
    
    return {
        'weight': weight.astype(np.float32),
        'bias': bias.astype(np.float32)
    }


def load_keras_model_weights(model_path: str) -> Dict[str, List[np.ndarray]]:
    """
    加载 Keras 模型并提取各层权重
    
    模型层结构（索引从0开始）:
        - Layer 0: GRU 1 (return_sequences=True)
        - Layer 1: Dropout 1 (无权重)
        - Layer 2: GRU 2 (return_sequences=False)
        - Layer 3: Dropout 2 (无权重)
        - Layer 4: Dense 1 (96 -> 96, tanh)
        - Layer 5: Dense 2 (96 -> 2, linear)
    
    Args:
        model_path: Keras .h5 模型文件路径
    
    Returns:
        layer_weights: 字典，键为层名称，值为该层的权重列表
    """
    # 延迟导入 TensorFlow
    import tensorflow as tf
    
    # 定义自定义损失函数（用于加载模型）
    def physics_weighted_mse(y_true, y_pred):
        squared_diff = tf.square(y_true - y_pred)
        weights = tf.constant([1.0, 1.0060286442927682])
        weighted_squared_diff = squared_diff * weights
        return tf.keras.backend.mean(weighted_squared_diff, axis=-1)
    
    custom_objects = {
        'physics_weighted_mse': physics_weighted_mse,
        'physics_weighted_mse_dynamic': physics_weighted_mse
    }
    
    print(f"加载 Keras 模型: {model_path}")
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    
    print(f"\n模型层数: {len(model.layers)}")
    print("\n各层信息:")
    for i, layer in enumerate(model.layers):
        weights = layer.get_weights()
        weight_shapes = [w.shape for w in weights] if weights else "无权重"
        print(f"  Layer {i}: {layer.name} ({layer.__class__.__name__})")
        print(f"           权重形状: {weight_shapes}")
    
    # 提取各层权重
    layer_weights = {
        'gru1': model.layers[0].get_weights(),   # GRU 1
        'gru2': model.layers[2].get_weights(),   # GRU 2
        'dense1': model.layers[4].get_weights(), # Dense 1
        'dense2': model.layers[5].get_weights()  # Dense 2
    }
    
    return layer_weights


def convert_keras_to_pytorch(
    keras_model_path: str,
    pytorch_model_path: str,
    hidden_size: int = 96
) -> None:
    """
    完整转换 Keras 模型到 PyTorch 格式
    
    Args:
        keras_model_path: 输入的 Keras .h5 模型路径
        pytorch_model_path: 输出的 PyTorch .pt 模型路径
        hidden_size: GRU 隐藏层维度
    """
    import torch
    from pytorch_gru_model import PyTorchGRUModel
    
    print("=" * 60)
    print("Keras -> PyTorch 模型转换")
    print("=" * 60)
    
    # 1. 加载 Keras 权重
    keras_weights = load_keras_model_weights(keras_model_path)
    
    # 2. 创建 PyTorch 模型
    print("\n创建 PyTorch 模型...")
    pytorch_model = PyTorchGRUModel(
        input_size=10,
        hidden_size=hidden_size,
        output_size=2
    )
    
    # 3. 转换各层权重
    print("\n" + "-" * 40)
    print("转换 GRU Layer 1 权重...")
    gru1_weights = convert_gru_weights(keras_weights['gru1'], hidden_size)
    
    print("\n" + "-" * 40)
    print("转换 GRU Layer 2 权重...")
    gru2_weights = convert_gru_weights(keras_weights['gru2'], hidden_size)
    
    print("\n" + "-" * 40)
    print("转换 Dense Layer 1 权重...")
    dense1_weights = convert_dense_weights(keras_weights['dense1'])
    
    print("\n" + "-" * 40)
    print("转换 Dense Layer 2 权重...")
    dense2_weights = convert_dense_weights(keras_weights['dense2'])
    
    # 4. 加载权重到 PyTorch 模型
    print("\n" + "-" * 40)
    print("加载权重到 PyTorch 模型...")
    
    state_dict = pytorch_model.state_dict()
    
    # GRU1 权重
    state_dict['gru1.weight_ih_l0'] = torch.from_numpy(gru1_weights['weight_ih_l0'])
    state_dict['gru1.weight_hh_l0'] = torch.from_numpy(gru1_weights['weight_hh_l0'])
    state_dict['gru1.bias_ih_l0'] = torch.from_numpy(gru1_weights['bias_ih_l0'])
    state_dict['gru1.bias_hh_l0'] = torch.from_numpy(gru1_weights['bias_hh_l0'])
    
    # GRU2 权重
    state_dict['gru2.weight_ih_l0'] = torch.from_numpy(gru2_weights['weight_ih_l0'])
    state_dict['gru2.weight_hh_l0'] = torch.from_numpy(gru2_weights['weight_hh_l0'])
    state_dict['gru2.bias_ih_l0'] = torch.from_numpy(gru2_weights['bias_ih_l0'])
    state_dict['gru2.bias_hh_l0'] = torch.from_numpy(gru2_weights['bias_hh_l0'])
    
    # Dense1 权重
    state_dict['dense1.weight'] = torch.from_numpy(dense1_weights['weight'])
    state_dict['dense1.bias'] = torch.from_numpy(dense1_weights['bias'])
    
    # Dense2 权重
    state_dict['dense2.weight'] = torch.from_numpy(dense2_weights['weight'])
    state_dict['dense2.bias'] = torch.from_numpy(dense2_weights['bias'])
    
    # 加载权重
    pytorch_model.load_state_dict(state_dict)
    
    # 5. 保存 PyTorch 模型
    print(f"\n保存 PyTorch 模型: {pytorch_model_path}")
    torch.save(pytorch_model.state_dict(), pytorch_model_path)
    
    # 6. 验证保存的模型可以正确加载
    print("\n验证模型加载...")
    test_model = PyTorchGRUModel()
    test_model.load_state_dict(torch.load(pytorch_model_path))
    test_model.eval()
    
    # 简单测试
    test_input = torch.randn(1, 25, 10)
    with torch.no_grad():
        test_output = test_model.predict(test_input)
    print(f"测试输出形状: {test_output.shape}")
    
    print("\n" + "=" * 60)
    print("转换完成！")
    print("=" * 60)
    print(f"输入模型: {keras_model_path}")
    print(f"输出模型: {pytorch_model_path}")


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(
        description='将 Keras GRU 模型转换为 PyTorch 格式'
    )
    parser.add_argument(
        '--keras_model', '-k',
        type=str,
        required=True,
        help='Keras .h5 模型文件路径'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='输出的 PyTorch .pt 模型路径（默认与输入同目录）'
    )
    parser.add_argument(
        '--hidden_size',
        type=int,
        default=96,
        help='GRU 隐藏层维度（默认 96）'
    )
    
    args = parser.parse_args()
    
    # 设置输出路径
    if args.output is None:
        output_dir = os.path.dirname(args.keras_model)
        output_path = os.path.join(output_dir, 'pytorch_model.pt')
    else:
        output_path = args.output
    
    # 执行转换
    convert_keras_to_pytorch(
        keras_model_path=args.keras_model,
        pytorch_model_path=output_path,
        hidden_size=args.hidden_size
    )


if __name__ == '__main__':
    main()
