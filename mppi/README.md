# MPPI 模型预测路径积分控制器

基于 PyTorch 实现的 MPPI（Model Predictive Path Integral）控制器，用于移动机器人路径跟踪。支持运动学模型和神经网络动力学模型。

---

## 📂 代码文件结构与功能说明

### 🎯 核心功能模块

| 文件路径 | 功能说明 |
|---------|---------|
| **src/pytorch_mppi/mppi.py** | **MPPI 核心算法实现**，采样轨迹、计算代价、更新控制序列（算法主循环） |
| **src/pytorch_mppi/mppi_ros_test.py** | **ROS 节点入口**，订阅定位/路径话题，调用 MPPI 算法，发布速度指令 |
| **src/pytorch_mppi/dynamics_interface.py** | **动力学模型接口**，定义抽象基类，实现运动学模型，提供工厂函数创建不同后端 |
| **src/pytorch_mppi/neural_mppi_dynamics.py** | **神经网络动力学封装**，包含 Keras 和 PyTorch 两种后端实现，管理历史缓冲区和粒子状态 |
| **src/pytorch_mppi/mppi_viz.py** | **RViz 可视化工具**，发布候选轨迹、最优轨迹、参考路径等可视化话题 |

---

### 🔧 PyTorch GRU 模型定义（Phase 3 新增）

| 文件路径 | 功能说明 |
|---------|---------|
| **tests/pytorch_gru_model.py** | **PyTorch GRU 模型定义**，定义与 Keras 模型完全等价的网络结构，用于加载 `.pt` 权重 |

---

### ⚙️ 配置文件

| 文件路径 | 功能说明 |
|---------|---------|
| **config/mppi_params.yaml** | **MPPI 参数配置**，包含粒子数、horizon、代价权重、动力学模型路径等所有可调参数 |
| **launch/mppi_controller.launch** | **ROS 启动文件**，加载 YAML 配置并启动 MPPI 节点 |

---

### 🧪 性能诊断工具（Phase 3/4 新增）

| 文件路径 | 功能说明 |
|---------|---------|
| **tests/benchmark_dynamics.py** | **性能诊断工具**，测试纯模型推理、数据转换、step 函数各部分耗时，对比不同后端性能 |

---

## 🚀 快速使用

### 启动 MPPI 控制器（使用神经网络）
```bash
roslaunch mppi mppi_controller.launch use_neural:=true
```

### 启动 MPPI 控制器（使用运动学模型）
```bash
roslaunch mppi mppi_controller.launch use_neural:=false
```

### 性能诊断
```bash
cd tests
python benchmark_dynamics.py
```

---

## 📊 工作流程

```
1. ROS 节点启动 (mppi_ros_test.py)
   ↓
2. 加载配置 (mppi_params.yaml)
   ↓
3. 创建动力学模型 (dynamics_interface.py)
   ├─ 运动学模型 (KinematicDynamics)
   └─ 神经网络模型 (NeuralDynamicsPyTorch)
   ↓
4. 初始化 MPPI 算法 (mppi.py)
   ↓
5. 订阅话题 (/odom, /global_plan)
   ↓
6. 控制主循环 (20Hz)
   ├─ 更新局部参考路径
   ├─ 重置动力学模型状态
   ├─ 采样 K 条轨迹 (mppi.command)
   ├─ 推演动力学并计算代价
   ├─ 加权更新控制序列
   └─ 发布速度指令 (/cmd_vel)
   ↓
7. 可视化 (mppi_viz.py)
   └─ 发布 RViz marker
```

---

## ⚙️ 关键参数说明

### MPPI 算法参数（mppi_params.yaml）

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `horizon` | 20 | 预测步长，越大越能提前规划，但计算慢 |
| `num_samples` | 300 | 采样轨迹数，越多越接近最优，但计算慢 |
| `lambda` | 1.0 | 温度系数，越大越随机探索 |
| `noise_sigma` | [0.5, 0.6] | 控制噪声标准差 [线速度, 角速度] |
| `control_freq` | 20.0 | 控制频率 (Hz) |

### 代价权重

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `w_cte` | 100.0 | 横向误差权重（最高优先级） |
| `w_yaw` | 1.0 | 航向误差权重 |
| `w_vel` | 3.0 | 速度跟踪权重 |
| `target_velocity` | 0.5 | 目标速度 (m/s) |

### 动力学模型配置

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `use_neural` | true | 是否使用神经网络模型 |
| `backend` | "pytorch" | 后端选择 (pytorch/keras) |
| `device` | "cpu" | 运行设备 (cpu/cuda) |
| `use_compile` | false | 是否启用 torch.compile JIT 优化 |

---

## 🔍 性能优化历程

### Phase 1: Keras 后端
- 推理时间：~1100 ms (100 粒子 × 20 步)
- 瓶颈：TensorFlow 开销 + NumPy 转换

### Phase 2: PyTorch 后端
- 推理时间：~700 ms
- 加速比：1.6x
- 优化：减少数据转换 + 向量化标准化

### Phase 3: 优化 PyTorch 后端
- 推理时间：~700 ms（与 Phase 2 相同）
- 瓶颈分析：模型推理占 97.4%，数据转换仅 0.2%
- 结论：CPU 算力瓶颈，GRU 顺序依赖无法批量 rollout

### 实际加速比
- 验证脚本（纯模型推理）：2.57x
- MPPI 实际运行：1.6x（受 Python 循环和代价函数开销限制）

---

## 🛠️ 调试建议

### 性能慢？
1. 降低 `num_samples` (如 100)
2. 降低 `horizon` (如 10-15)
3. 使用运动学模型 (`use_neural: false`)
4. 检查 CPU 占用率（`top` 命令）

### 路径跟踪不准？
1. 增大 `w_cte` 权重 (如 200-500)
2. 增大 `num_samples` (如 500-1000)
3. 调整 `ref_path_length` (如 3-8 米)

### 运动抖动？
1. 减小 `noise_sigma` (如 [0.3, 0.4])
2. 增大 `lambda` (如 2.0-5.0)

---

## 📝 注意事项

1. **时间步长一致性**：神经网络训练时的 dt=0.05s，控制频率应为 20Hz
2. **粒子数限制**：CPU 环境建议 num_samples < 500，否则超过 1s
3. **模型格式**：PyTorch 后端需要 `.pt` 格式，使用 `convert_keras_to_pytorch.py` 转换
4. **路径要求**：全局路径至少 2 个点，否则会回退到运动学模型

---

## 📚 参考资料

- MPPI 论文：Williams et al., "Information Theoretic MPC for Model-Based RL" (2017)
- 开源实现：[pytorch_mppi](https://github.com/UM-ARM-Lab/pytorch_mppi)
