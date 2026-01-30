# 神经网络车辆动力学建模功能包

本功能包用于训练 GRU 神经网络模型来预测车辆动力学（速度变化量），并将训练好的模型用于 MPPI 控制器。

---

## 📂 代码文件结构与功能说明

### 🎯 核心入口与配置（Tier 1）

| 文件路径 | 功能说明 |
|---------|---------|
| `main_NN_Vehicle_dynamics.py` | **程序唯一入口**，根据配置决定是训练还是推理，协调所有模块工作 |
| `params/parameters.toml` | **总配置文件**，定义训练超参数、数据路径、模型架构等所有参数 |
| `helper_funcs_NN/src/manage_paths.py` | **路径管理器**，将所有文件路径转为绝对路径，避免路径错误 |
| `helper_funcs_NN/src/handle_params.py` | **参数读取器**，解析 `parameters.toml` 并加载到程序中 |

---

### 🔧 核心功能实现（Tier 2 - 用户常需修改）

| 文件路径 | 功能说明 |
|---------|---------|
| `src/train_neuralnetwork.py` | **训练流程总指挥**，负责模型训练、验证、保存模型和 scaler |
| `src/run_neuralnetwork.py` | **推理流程总指挥**，负责加载模型、预测测试集、计算误差指标 |
| `src/load_data_for_nn.py` | **数据加载器**，读取 CSV 文件并转换为 NumPy 数组 |
| `src/prepare_data.py` | **数据预处理核心**，执行标准化、时间窗口切片（t-24 到 t）、划分训练集/测试集 |
| `src/neural_network_fcn.py` | **模型定义文件**，定义 GRU/LSTM 网络层结构和损失函数 |
| `visualization/plot_results.py` | **可视化工具**，绘制预测曲线对比图，计算 MSE/MAE 指标 |

---

### 🛠️ 辅助工具（Tier 3）

| 文件路径 | 功能说明 |
|---------|---------|
| `helper_funcs_NN/src/select_optimizer.py` | **优化器选择器**，根据配置字符串（如 'Adam'）返回 Keras 优化器对象 |

---

### 🔄 模型格式转换（Phase 3 新增）

| 文件路径 | 功能说明 |
|---------|---------|
| `src/convert_keras_to_pytorch.py` | **Keras→PyTorch 转换器**，将训练好的 `.h5` 模型转换为 `.pt` 格式，处理 GRU 门顺序重排 |
| `src/verify_conversion.py` | **转换验证工具**，对比 Keras 和 PyTorch 模型输出，确保转换精度（误差 < 1e-4） |

---

## 🚀 快速使用

### 训练模型
```bash
# 1. 修改 parameters.toml 中的配置
# 2. 运行训练
python main_NN_Vehicle_dynamics.py
```

### 推理测试
```bash
# 确保 parameters.toml 中 mode = "infer"
python main_NN_Vehicle_dynamics.py
```

### 转换为 PyTorch 格式
```bash
cd src
python convert_keras_to_pytorch.py \
    --keras_model ../models/xxx/keras_model_recurrent.h5 \
    --output ../models/xxx/pytorch_model.pt
```

### 验证转换正确性
```bash
python verify_conversion.py \
    --keras_model ../models/xxx/keras_model_recurrent.h5 \
    --pytorch_model ../models/xxx/pytorch_model.pt \
    --scaler_x ../models/xxx/scaler.plk \
    --scaler_y ../models/xxx/scaler_y.joblib
```

---

## 📊 典型工作流

1. **准备数据** → CSV 文件放入 `data/` 目录
2. **配置训练** → 修改 `parameters.toml`
3. **训练模型** → 运行 `main_NN_Vehicle_dynamics.py`
4. **转换格式** → 运行 `convert_keras_to_pytorch.py`
5. **验证转换** → 运行 `verify_conversion.py`
6. **集成到 MPPI** → 将 `.pt` 和 scaler 路径配置到 MPPI 的 YAML 中

---

## ⚙️ 关键配置说明

- **时间窗口长度**：`parameters.toml` 中 `sequence_length = 25` 表示使用过去 25 步预测下一步
- **特征顺序**：`[v, w, dt, cmd_v, cmd_w, a_v, a_w, err_v, err_w, v_x_w]` （10 维）
- **输出**：`[diff_v, diff_w]` （速度残差，2 维）
- **模型架构**：2 层 GRU (96 单元) + 2 层 Dense



