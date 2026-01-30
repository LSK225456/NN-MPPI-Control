import numpy as np
import sys
from tqdm import tqdm
import os.path
from tensorflow import keras
from joblib import load

# custom modules
import src

"""
Created by: Rainer Trauth
Created on: 01.04.2020
"""

# SET FLOATING POINT PRECISION
np.set_printoptions(formatter={'float': lambda x: "{0:0.16f}".format(x)})


def run_nn(path_dict: dict,
           params_dict: dict,
           startpoint: float,
           counter: int,
           nn_mode: str):
    """Runs the recurrent neural network to test its predictions against actual vehicle data.

    :param path_dict:           dictionary which contains paths to all relevant folders and files of this module
    :type path_dict: dict
    :param params_dict:         dictionary which contains all parameters necessary to run this module
    :type params_dict: dict
    :param startpoint:          row index where to start using provided test data
    :type startpoint: float
    :param counter:             number of current testing loop (only used for naming of output files)
    :type counter: int
    """

    if not nn_mode == "feedforward" or not nn_mode == "recurrent":
        ValueError('unknown "neural network mode"; must be either "feedforard" or "recurrent"')

    # if no model was trained, load existing model in inputs folder /inputs/trained_models
    if params_dict['NeuralNetwork_Settings']['model_mode'] == 0:
        path2scaler = path_dict['filepath2scaler_load']

        if nn_mode == "feedforward":
            path2model = path_dict['filepath2inputs_trainedmodel_ff']
        elif nn_mode == "recurrent":
            path2model = path_dict['filepath2inputs_trainedmodel_recurr']

    else:
        path2scaler = path_dict['filepath2scaler_save']

        if nn_mode == "feedforward":
            path2model = path_dict['filepath2results_trainedmodel_ff']
        elif nn_mode == "recurrent":
            path2model = path_dict['filepath2results_trainedmodel_recurr']
    
    # todo :gemini is fucking dying
    if params_dict['General']['use_old_transformation']:
        path2scaler_main = path_dict['filepath2scaler_load']
    else:
        path2scaler_main = path_dict['filepath2scaler_save']
    
    scaler_main = load(path2scaler_main)


    # 读取用于测试的 CSV 文件 (data_to_run.csv),这里读取的是还没有归一化的原始物理数据。
    with open(path_dict['filepath2inputs_testdata'] + '.csv', 'r') as fh:
        data = np.loadtxt(fh, delimiter=',')

    # 安全检查：确保您指定的开始时间点 (startpoint) 加上测试时长 (run_timespan) 没有超出文件总行数
    if startpoint + params_dict['Test']['run_timespan'] > data.shape[0]:
        sys.exit("test dataset fully covered -> exit main script")

    input_shape = params_dict['NeuralNetwork_Settings']['input_shape']
    output_shape = params_dict['NeuralNetwork_Settings']['output_shape']
    input_timesteps = params_dict['NeuralNetwork_Settings']['input_timesteps']

    # scale dataset the vanish effects of different input data quantities必须使用训练时生成的那个 scaler对整个测试数据集进行归一化。否则模型看不懂数据。
    data = src.prepare_data.scaler_run(path2scaler=path2scaler,
                                       params_dict=params_dict,
                                       dataset=data)

    #todolsk:硬编码
    # initial, steeringangle_rad, torqueRL_Nm, torqueRR_Nm, brakepresF_bar, brakepresR_bar = \
    #     src.prepare_data.create_dataset_separation_run(data, params_dict, startpoint,
    #                                                    params_dict['Test']['run_timespan'], nn_mode)
    initial, future_controls = \
        src.prepare_data.create_dataset_separation_run(data, params_dict, startpoint,
                                                       params_dict['Test']['run_timespan'], nn_mode) # 推理硬编码修改


  
    # load neural network model
    model = keras.models.load_model(path2model)

    # results = np.zeros((len(torqueRR_Nm) + input_timesteps, input_shape))
    results = np.zeros((len(future_controls) + input_timesteps, input_shape))

    if nn_mode == "feedforward":
        new_input = np.zeros((1, input_shape * input_timesteps))

        for m in range(0, input_timesteps):
            results[m, 0:output_shape] = initial[:, m * input_shape:m * input_shape + output_shape]

    elif nn_mode == "recurrent":
        new_input = np.zeros((1, input_timesteps, input_shape))

        results[0:input_timesteps, :] = initial[0, :, :]

    # 循环遍历每一个时间步
    # 注意：这里的 len(torqueRR_Nm) 其实就是 run_timespan (测试时长)
    # for i_count in tqdm(range(0, len(torqueRR_Nm))):
    for i_count in tqdm(range(0, len(future_controls))):
        # 第一步：准备输入数据,如果是第一步，直接用从文件读取的真实 20 帧 (initial)
        if i_count == 0:
            data_convert = initial

        else:
            # 如果不是第一步，用上一步构造出来的 new_input
            # 关键点：这个 new_input 里包含的是模型自己预测出的历史，这就是“开环”！
            data_convert = new_input

        # 第二步：模型预测
        # 输入 (1, 20, 5)，输出 (1, 2) -> [v_pred, w_pred]
        pred_raw  = model.predict(data_convert)

    
            # 保存预测结果到 results 矩阵
            # results[i_count + input_timesteps, 0:output_shape] = result_process
        final_prediction_scaled = pred_raw
        
        results[i_count + input_timesteps, 0:output_shape] = final_prediction_scaled

        # convert test data
        if nn_mode == "feedforward":
            temp = np.zeros((1, input_shape * input_timesteps))
            temp[:, 0:input_shape * (input_timesteps - 1)] = data_convert[0, input_shape:input_shape * input_timesteps]

            temp[:, input_shape * (input_timesteps - 1):input_shape * (input_timesteps - 1) + output_shape] \
                = final_prediction_scaled

            # temp[:, input_shape * (input_timesteps - 1) + output_shape] = steeringangle_rad[i_count]
            # temp[:, input_shape * (input_timesteps - 1) + output_shape + 1] = torqueRL_Nm[i_count]
            # temp[:, input_shape * (input_timesteps - 1) + output_shape + 2] = torqueRR_Nm[i_count]
            # temp[:, input_shape * (input_timesteps - 1) + output_shape + 3] = brakepresF_bar[i_count]
            # temp[:, input_shape * (input_timesteps - 1) + output_shape + 4] = brakepresR_bar[i_count]
            idx_last_step_control_start = input_shape * (input_timesteps - 1) + output_shape
            temp[:, idx_last_step_control_start:] = future_controls[i_count] # 推理硬编码修改

        # 第三步：构造下一步的输入
        elif nn_mode == "recurrent":
            # 创建一个新的临时容器 temp，形状 (1, 20, 5)
            temp = np.zeros((1, input_timesteps, input_shape))
            # 历史平移  把当前窗口的第 1~19 帧，复制到新窗口的 0~18 帧。相当于扔掉最老的一帧，腾出位置给最新的一帧。
            temp[0, 0:input_timesteps - 1, :] = data_convert[0, 1:input_timesteps, :]

            # 填入最新的预测状态 (v, w)   把模型刚预测出的 v, w 填入新窗口的第 19 帧 (最新帧) 的前两列。这是“自回归”的关键：用自己的预测作为下一步的输入。
            temp[0, input_timesteps - 1, 0:output_shape] = final_prediction_scaled

            # todolsk
            # temp[0, input_timesteps - 1, output_shape] = steeringangle_rad[i_count]
            # temp[0, input_timesteps - 1, output_shape + 1] = torqueRL_Nm[i_count]
            # temp[0, input_timesteps - 1, output_shape + 2] = torqueRR_Nm[i_count]
            # temp[0, input_timesteps - 1, output_shape + 3] = brakepresF_bar[i_count]
            # temp[0, input_timesteps - 1, output_shape + 4] = brakepresR_bar[i_count]
            # 自适应填入控制量 (dt, cmd_v, cmd_w)   逻辑：当前时间窗的最后一帧 (input_timesteps - 1)，从 output_shape 列开始，填入 future_controls 的当前值
            temp[0, input_timesteps - 1, output_shape:] = future_controls[i_count] # 推理硬编码修改

        new_input = temp

    # 循环结束后，把剩下的真实数据（Input部分）填回 results 矩阵的后几列
    # 这样 results 矩阵就完整了：前2列是预测的，后3列是真实的 dt, cmd。
    # results[:, output_shape:input_shape] = data[startpoint:startpoint + len(steeringangle_rad) + input_timesteps,
    #                                             output_shape:input_shape]
    results[:, output_shape:input_shape] = data[startpoint:startpoint + len(future_controls) + input_timesteps,
                                                output_shape:input_shape] # 推理硬编码修改

    # 反归一化：把 [-1, 1] 的数值变回 m/s, rad/s
    # results = src.prepare_data.scaler_reverse(path2scaler=path2scaler,
    #                                           params_dict=params_dict,
    #                                           dataset=results)
    results = scaler_main.inverse_transform(results)

    np.savetxt(os.path.join(path_dict['path2results_matfiles'], 'prediction_result_' + nn_mode + str(counter) + '.csv'),
               results)
