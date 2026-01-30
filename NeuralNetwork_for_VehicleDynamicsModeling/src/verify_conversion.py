#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 到 PyTorch 模型转换验证脚本

本脚本用于验证转换后的 PyTorch 模型与原始 Keras 模型输出一致性。

验证内容:
    1. 随机输入测试：100 组随机输入，对比输出误差
    2. 端到端测试：使用真实 scaler 数据验证完整推理流程
    3. 批量推理测试：验证批量处理的正确性

通过条件:
    - 最大绝对误差 < 1e-4
    - 平均绝对误差 < 1e-5

使用方法（示例）:
    cd ~/lsk_graduate/lsk_ws/src/NeuralNetwork_for_VehicleDynamicsModeling/src
    
    python verify_conversion.py \
        --keras_model ../models/0129-1216参数-0.32-mppi环境下训练/keras_model_recurrent.h5 \
        --pytorch_model ../models/0129-1216参数-0.32-mppi环境下训练/pytorch_model.pt \
        --scaler_x ../models/0129-1216参数-0.32-mppi环境下训练/scaler.plk \
        --scaler_y ../models/0129-1216参数-0.32-mppi环境下训练/scaler_y.joblib

Author: MPPI Project
Date: 2024
"""

import os
import sys
import argparse
import numpy as np
import joblib
from typing import Tuple, Optional

# 添加 PyTorch 模型路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 
    '../../mppi/tests'))


def load_keras_model(model_path: str):
    """加载 Keras 模型"""
    import tensorflow as tf
    
    # 自定义损失函数
    def physics_weighted_mse(y_true, y_pred):
        squared_diff = tf.square(y_true - y_pred)
        weights = tf.constant([1.0, 1.0060286442927682])
        weighted_squared_diff = squared_diff * weights
        return tf.keras.backend.mean(weighted_squared_diff, axis=-1)
    
    custom_objects = {
        'physics_weighted_mse': physics_weighted_mse,
        'physics_weighted_mse_dynamic': physics_weighted_mse
    }
    
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    return model


def load_pytorch_model(model_path: str, device: str = "cpu"):
    """加载 PyTorch 模型"""
    import torch
    from pytorch_gru_model import PyTorchGRUModel
    
    model = PyTorchGRUModel()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device)
    return model


def compare_outputs(
    keras_output: np.ndarray,
    pytorch_output: np.ndarray,
    test_name: str = "Test"
) -> Tuple[float, float, bool]:
    """
    对比两个模型的输出
    
    Args:
        keras_output: Keras 模型输出
        pytorch_output: PyTorch 模型输出
        test_name: 测试名称，用于打印
    
    Returns:
        max_error: 最大绝对误差
        mean_error: 平均绝对误差
        passed: 是否通过测试（max_error < 1e-4）
    """
    diff = np.abs(keras_output - pytorch_output)
    max_error = np.max(diff)
    mean_error = np.mean(diff)
    
    threshold = 1e-4
    passed = max_error < threshold
    
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"\n{test_name}:")
    print(f"  最大绝对误差: {max_error:.2e}")
    print(f"  平均绝对误差: {mean_error:.2e}")
    print(f"  阈值: {threshold:.2e}")
    print(f"  结果: {status}")
    
    return max_error, mean_error, passed


def test_random_inputs(
    keras_model,
    pytorch_model,
    num_tests: int = 100,
    device: str = "cpu"
) -> bool:
    """
    随机输入测试
    
    Args:
        keras_model: Keras 模型
        pytorch_model: PyTorch 模型
        num_tests: 测试数量
        device: PyTorch 设备
    
    Returns:
        passed: 所有测试是否通过
    """
    import torch
    
    print("=" * 60)
    print(f"随机输入测试 ({num_tests} 组)")
    print("=" * 60)
    
    all_max_errors = []
    all_mean_errors = []
    
    for i in range(num_tests):
        # 生成随机输入
        random_input = np.random.randn(1, 25, 10).astype(np.float32)
        
        # Keras 推理
        keras_output = keras_model.predict(random_input, verbose=0)
        
        # PyTorch 推理
        pytorch_input = torch.from_numpy(random_input).to(device)
        with torch.no_grad():
            pytorch_output = pytorch_model.predict(pytorch_input).cpu().numpy()
        
        # 计算误差
        diff = np.abs(keras_output - pytorch_output)
        all_max_errors.append(np.max(diff))
        all_mean_errors.append(np.mean(diff))
    
    # 统计结果
    overall_max = max(all_max_errors)
    overall_mean = np.mean(all_mean_errors)
    threshold = 1e-4
    passed = overall_max < threshold
    
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"\n随机输入测试汇总:")
    print(f"  测试数量: {num_tests}")
    print(f"  最大误差 (所有测试): {overall_max:.2e}")
    print(f"  平均误差 (所有测试): {overall_mean:.2e}")
    print(f"  阈值: {threshold:.2e}")
    print(f"  结果: {status}")
    
    return passed


def test_batch_inference(
    keras_model,
    pytorch_model,
    batch_sizes: list = [1, 32, 100, 500],
    device: str = "cpu"
) -> bool:
    """
    批量推理测试
    
    Args:
        keras_model: Keras 模型
        pytorch_model: PyTorch 模型
        batch_sizes: 测试的批量大小列表
        device: PyTorch 设备
    
    Returns:
        passed: 所有测试是否通过
    """
    import torch
    
    print("\n" + "=" * 60)
    print("批量推理测试")
    print("=" * 60)
    
    all_passed = True
    
    for batch_size in batch_sizes:
        # 生成批量输入
        batch_input = np.random.randn(batch_size, 25, 10).astype(np.float32)
        
        # Keras 推理
        keras_output = keras_model.predict(batch_input, verbose=0)
        
        # PyTorch 推理
        pytorch_input = torch.from_numpy(batch_input).to(device)
        with torch.no_grad():
            pytorch_output = pytorch_model.predict(pytorch_input).cpu().numpy()
        
        # 对比
        _, _, passed = compare_outputs(
            keras_output, 
            pytorch_output,
            f"批量大小 = {batch_size}"
        )
        all_passed = all_passed and passed
    
    return all_passed


def test_with_scaler(
    keras_model,
    pytorch_model,
    scaler_x_path: str,
    scaler_y_path: str,
    device: str = "cpu"
) -> bool:
    """
    使用真实 scaler 的端到端测试
    
    Args:
        keras_model: Keras 模型
        pytorch_model: PyTorch 模型
        scaler_x_path: 输入标准化器路径
        scaler_y_path: 输出标准化器路径
        device: PyTorch 设备
    
    Returns:
        passed: 测试是否通过
    """
    import torch
    
    print("\n" + "=" * 60)
    print("端到端测试（使用真实 Scaler）")
    print("=" * 60)
    
    # 加载 scaler
    scaler_x = joblib.load(scaler_x_path)
    scaler_y = joblib.load(scaler_y_path)
    
    print(f"Scaler X - mean shape: {scaler_x.mean_.shape}")
    print(f"Scaler X - scale shape: {scaler_x.scale_.shape}")
    print(f"Scaler Y - mean shape: {scaler_y.mean_.shape}")
    print(f"Scaler Y - scale shape: {scaler_y.scale_.shape}")
    
    # 生成模拟真实数据范围的输入
    # 特征顺序: [v, w, dt, cmd_v, cmd_w, a_v, a_w, err_v, err_w, v_x_w]
    num_tests = 50
    all_passed = True
    all_max_errors = []
    
    for i in range(num_tests):
        # 生成合理范围的原始特征
        v = np.random.uniform(-0.5, 0.5, (25,))
        w = np.random.uniform(-0.5, 0.5, (25,))
        dt = np.full((25,), 0.05)
        cmd_v = np.random.uniform(-0.4, 0.5, (25,))
        cmd_w = np.random.uniform(-0.4, 0.4, (25,))
        a_v = np.random.uniform(-2, 2, (25,))
        a_w = np.random.uniform(-2, 2, (25,))
        err_v = cmd_v - v
        err_w = cmd_w - w
        v_x_w = v * w
        
        # 组合特征
        raw_features = np.stack([
            v, w, dt, cmd_v, cmd_w, a_v, a_w, err_v, err_w, v_x_w
        ], axis=-1).astype(np.float32)  # [25, 10]
        
        # 标准化
        scaled_features = scaler_x.transform(raw_features)  # [25, 10]
        batch_input = scaled_features.reshape(1, 25, 10).astype(np.float32)
        
        # Keras 推理
        keras_pred_scaled = keras_model.predict(batch_input, verbose=0)
        keras_output = scaler_y.inverse_transform(keras_pred_scaled)
        
        # PyTorch 推理
        pytorch_input = torch.from_numpy(batch_input).to(device)
        with torch.no_grad():
            pytorch_pred_scaled = pytorch_model.predict(pytorch_input).cpu().numpy()
        pytorch_output = scaler_y.inverse_transform(pytorch_pred_scaled)
        
        # 计算误差
        diff = np.abs(keras_output - pytorch_output)
        max_error = np.max(diff)
        all_max_errors.append(max_error)
    
    # 统计结果
    overall_max = max(all_max_errors)
    overall_mean = np.mean(all_max_errors)
    threshold = 1e-4
    passed = overall_max < threshold
    
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"\n端到端测试汇总:")
    print(f"  测试数量: {num_tests}")
    print(f"  最大误差 (反标准化后): {overall_max:.2e}")
    print(f"  平均误差: {overall_mean:.2e}")
    print(f"  阈值: {threshold:.2e}")
    print(f"  结果: {status}")
    
    # 打印一个具体样例
    print("\n样例对比 (最后一次测试):")
    print(f"  Keras 输出:   [diff_v={keras_output[0,0]:.6f}, diff_w={keras_output[0,1]:.6f}]")
    print(f"  PyTorch 输出: [diff_v={pytorch_output[0,0]:.6f}, diff_w={pytorch_output[0,1]:.6f}]")
    print(f"  绝对误差:     [diff_v={diff[0,0]:.2e}, diff_w={diff[0,1]:.2e}]")
    
    return passed


def test_inference_speed(
    keras_model,
    pytorch_model,
    batch_size: int = 500,
    num_iterations: int = 100,
    device: str = "cpu"
) -> None:
    """
    推理速度对比测试
    
    Args:
        keras_model: Keras 模型
        pytorch_model: PyTorch 模型
        batch_size: 批量大小
        num_iterations: 迭代次数
        device: PyTorch 设备
    """
    import torch
    import time
    
    print("\n" + "=" * 60)
    print(f"推理速度对比测试 (batch={batch_size}, iterations={num_iterations})")
    print("=" * 60)
    
    # 准备输入
    input_np = np.random.randn(batch_size, 25, 10).astype(np.float32)
    input_torch = torch.from_numpy(input_np).to(device)
    
    # Keras 预热
    for _ in range(10):
        keras_model.predict(input_np, verbose=0)
    
    # Keras 计时
    start = time.perf_counter()
    for _ in range(num_iterations):
        keras_model.predict(input_np, verbose=0)
    keras_time = (time.perf_counter() - start) * 1000 / num_iterations
    
    # PyTorch 预热
    with torch.no_grad():
        for _ in range(10):
            pytorch_model.predict(input_torch)
    
    # PyTorch 计时
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_iterations):
            pytorch_model.predict(input_torch)
    pytorch_time = (time.perf_counter() - start) * 1000 / num_iterations
    
    speedup = keras_time / pytorch_time
    
    print(f"\n推理速度对比:")
    print(f"  Keras 平均耗时:   {keras_time:.2f} ms")
    print(f"  PyTorch 平均耗时: {pytorch_time:.2f} ms")
    print(f"  加速比: {speedup:.2f}x")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='验证 Keras 到 PyTorch 模型转换的正确性',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  cd ~/lsk_graduate/lsk_ws/src/NeuralNetwork_for_VehicleDynamicsModeling/src
  
  python verify_conversion.py \\
      --keras_model ../models/0129-1216参数-0.32-mppi环境下训练/keras_model_recurrent.h5 \\
      --pytorch_model ../models/0129-1216参数-0.32-mppi环境下训练/pytorch_model.pt \\
      --scaler_x ../models/0129-1216参数-0.32-mppi环境下训练/scaler.plk \\
      --scaler_y ../models/0129-1216参数-0.32-mppi环境下训练/scaler_y.joblib
        """
    )
    parser.add_argument(
        '--keras_model', '-k',
        type=str,
        required=True,
        help='Keras .h5 模型文件路径'
    )
    parser.add_argument(
        '--pytorch_model', '-p',
        type=str,
        required=True,
        help='PyTorch .pt 模型文件路径'
    )
    parser.add_argument(
        '--scaler_x', '-sx',
        type=str,
        default=None,
        help='输入标准化器路径（可选，用于端到端测试）'
    )
    parser.add_argument(
        '--scaler_y', '-sy',
        type=str,
        default=None,
        help='输出标准化器路径（可选，用于端到端测试）'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
        help='PyTorch 运行设备'
    )
    
    args = parser.parse_args()
    
    # ====== 添加路径验证 ======
    print("=" * 60)
    print("Keras -> PyTorch 模型转换验证")
    print("=" * 60)
    print(f"Keras 模型: {args.keras_model}")
    print(f"PyTorch 模型: {args.pytorch_model}")
    print(f"运行设备: {args.device}")
    
    # 检查文件是否存在
    if not os.path.exists(args.keras_model):
        print(f"\n错误: Keras 模型文件不存在: {args.keras_model}")
        print("\n请确认路径是否正确，或参考以下示例:")
        print("python verify_conversion.py \\")
        print("    --keras_model ../models/0129-1216参数-0.32-mppi环境下训练/keras_model_recurrent.h5 \\")
        print("    --pytorch_model ../models/0129-1216参数-0.32-mppi环境下训练/pytorch_model.pt \\")
        print("    --scaler_x ../models/0129-1216参数-0.32-mppi环境下训练/scaler.plk \\")
        print("    --scaler_y ../models/0129-1216参数-0.32-mppi环境下训练/scaler_y.joblib")
        return 1
    
    if not os.path.exists(args.pytorch_model):
        print(f"\n错误: PyTorch 模型文件不存在: {args.pytorch_model}")
        print("请先运行 convert_keras_to_pytorch.py 生成 .pt 文件")
        return 1
    
    if args.scaler_x and not os.path.exists(args.scaler_x):
        print(f"\n错误: Scaler X 文件不存在: {args.scaler_x}")
        return 1
    
    if args.scaler_y and not os.path.exists(args.scaler_y):
        print(f"\n错误: Scaler Y 文件不存在: {args.scaler_y}")
        return 1
    
    # 加载模型
    print("\n加载模型...")
    keras_model = load_keras_model(args.keras_model)
    pytorch_model = load_pytorch_model(args.pytorch_model, args.device)
    print("模型加载完成")
    
    # 执行测试
    results = []
    
    # 1. 随机输入测试
    results.append(("随机输入测试", test_random_inputs(
        keras_model, pytorch_model, num_tests=100, device=args.device
    )))
    
    # 2. 批量推理测试
    results.append(("批量推理测试", test_batch_inference(
        keras_model, pytorch_model, device=args.device
    )))
    
    # 3. 端到端测试（如果提供了 scaler）
    if args.scaler_x and args.scaler_y:
        results.append(("端到端测试", test_with_scaler(
            keras_model, pytorch_model,
            args.scaler_x, args.scaler_y,
            device=args.device
        )))
    
    # 4. 速度测试
    test_inference_speed(
        keras_model, pytorch_model,
        batch_size=500, num_iterations=100,
        device=args.device
    )
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("验证结果汇总")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")
        all_passed = all_passed and passed
    
    print("\n" + "-" * 60)
    if all_passed:
        print("✓ 所有验证通过！PyTorch 模型可以安全使用。")
    else:
        print("✗ 部分验证失败！请检查转换逻辑。")
    print("-" * 60)
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
