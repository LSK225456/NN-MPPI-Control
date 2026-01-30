import numpy as np
import sys
import random

# custom modules
import helper_funcs_NN
import src
import visualization

"""
Created by: Leonhard Hermansdorfer, Rainer Trauth
Created on: 01.04.2020

Documentation
main script to run neural network training
"""

# 设置随机种子，保证结果可复现,神经网络初始化权重通常是随机的。设置 seed(7) 后，每次重新运行代码，初始权重和数据打乱顺序都完全一样。
random.seed(7)
np.random.seed(7)

# 自动获取当前项目的根目录，并生成一个包含所有关键路径的字典 `path_dict`。
path_dict = helper_funcs_NN.src.manage_paths.manage_paths()


# 读取 `params/parameters.toml` 文件，并将其解析为 Python 字典 `params_dict`,之后所有对超参数的访问都通过这个字典进行
params_dict = helper_funcs_NN.src.handle_params.handle_params(path_dict=path_dict)


# ----------------------------------------------------------------------------------------------------------------------
# Training of the Neural Network ---------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

# model_mode:0 --> 不执行，1 --> 使用前馈模型执行，2 --> 使用循环模型执行
if params_dict['NeuralNetwork_Settings']['model_mode'] == 1:
    src.train_neuralnetwork.train_neuralnetwork(path_dict=path_dict,
                                                params_dict=params_dict,
                                                nn_mode='feedforward')



if params_dict['NeuralNetwork_Settings']['model_mode'] == 2:
    src.train_neuralnetwork.train_neuralnetwork(path_dict=path_dict,        # train_neuralnetwork函数内部会完成：加载数据 -> 归一化 -> 建立模型 -> model.fit() -> 保存模型。
                                                params_dict=params_dict,
                                                nn_mode="recurrent")




# 检查 params.toml 中的 run_file_mode 是否为 0。0 代表“只训练，不测试”。如果为 0，程序直接在这里退出。
if params_dict['NeuralNetwork_Settings']['run_file_mode'] == 0:
    sys.exit('SYSTEM EXIT: exit due to run_file_mode is set to zero to avoid testing the neural network against '
             + 'vehicle sensor data')

for i_count in range(0, params_dict['Test']['n_test']):

    # 计算本次测试在测试集 CSV 中的起始行号。
    # run_timestart 是初始偏移量，iteration_step 是步长。
    # 例如：第一次测第 0 行开始，第二次测第 1250 行开始... 从而覆盖各种路况。
    idx_start = params_dict['Test']['run_timestart'] + i_count * params_dict['Test']['iteration_step']

    if params_dict['NeuralNetwork_Settings']['run_file_mode'] == 1:
        print('STARTING RUN FEEDFORWARD NETWORK')

        src.run_neuralnetwork.run_nn(path_dict=path_dict,
                                     params_dict=params_dict,
                                     startpoint=idx_start,
                                     counter=i_count,
                                     nn_mode="feedforward")

    # 如果配置为测试 Recurrent 网络 (mode=2，当前情况)
    if params_dict['NeuralNetwork_Settings']['run_file_mode'] == 2:
        print('STARTING RUN RECURRENT NETWORK')
        
        # 核心调用：进入 src/run_neuralnetwork.py 进行开环多步推理。
        # 这是一个非常耗时的步骤，它会模拟“如果完全信任模型预测的下一刻状态并以此继续预测”会发生什么。
        # 结果会保存为 .csv 文件在 outputs 文件夹中。
        src.run_neuralnetwork.run_nn(path_dict=path_dict,
                                     params_dict=params_dict,
                                     startpoint=idx_start,
                                     counter=i_count,
                                     nn_mode="recurrent")

    # 结果可视化与指标计算。
    # 调用 visualization/plot_results.py。
    # 作用：
    # 1. 读取 run_nn 生成的预测结果 csv。
    # 2. 读取真实的测试集 Ground Truth。
    # 3. 计算 MSE/MAE 并打印到控制台。
    visualization.plot_results.plot_run_lsk(path_dict=path_dict,
                                        params_dict=params_dict,
                                        counter=i_count,
                                        start=idx_start)
