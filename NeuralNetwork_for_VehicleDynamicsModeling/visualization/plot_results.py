import numpy as np
import matplotlib.pyplot as plt
import os.path
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler



def plot_and_save_lsk(params_dict: dict,
                      inp_1,
                      inp_2,
                      inp_3,
                      value,
                      savename):
    """
    [LSK 重写] 通用绘图函数，移除了 '8 ms' 等硬编码。
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8)) # 稍微调大一点尺寸

    # 上图：结果对比
    ax1.plot(inp_1, label='Result (Pred)', color='tab:orange')
    ax1.plot(inp_2, label='Label (GT)', color='tab:blue', alpha=0.7) # 加一点透明度看重叠




    ax1.set_ylabel(value)
    ax1.set_xlabel('Time steps') # 移除硬编码的 '8 ms'
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.5) # 加网格方便看

    # 下图：误差曲线
    ax2.plot(inp_3, label='Error (Label - Result)', color='tab:red', linewidth=1.0)
    ax2.set_ylabel('Difference')
    ax2.set_xlabel('Time steps')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.5)

    if params_dict['General']['plot_result']:
        plt.show()

    if params_dict['General']['save_figures']:
        # 自动创建父目录（如果不存在），防止报错
        os.makedirs(os.path.dirname(savename), exist_ok=True)
        fig.savefig(savename, format='png', dpi=150) # 提高一点分辨率
        plt.close(fig)


def plot_run_lsk(path_dict: dict,
                 params_dict: dict,
                 counter,
                 start):
    """
    [LSK 重写] 完全自适应的评估函数。
    自动根据 input_shape 遍历所有维度，计算指标并绘图。
    """
    
    # 1. 确定文件名 (保持原逻辑)
    if params_dict['NeuralNetwork_Settings']['run_file_mode'] == 1:
        filename_model = 'prediction_result_feedforward'
    elif params_dict['NeuralNetwork_Settings']['run_file_mode'] == 2:
        filename_model = 'prediction_result_recurrent'
    else:
        return

    # 2. 加载数据
    filepath2results = os.path.join(path_dict['path2results_matfiles'], filename_model + str(counter) + '.csv')
    
    # 加载预测结果
    with open(filepath2results, 'r') as fh:
        results = np.loadtxt(fh)

    # 加载真实标签 (原始测试集)
    with open(path_dict['filepath2inputs_testdata'] + '.csv', 'r') as fh:
        labels_full = np.loadtxt(fh, delimiter=',')

    # 3. 截取对应的 Ground Truth 片段
    # 注意：results 的长度通常是 run_timespan + input_timesteps (因为有预热)
    # 这里的截取逻辑需要和 run_neuralnetwork.py 里的回填逻辑对齐
    run_duration = params_dict['Test']['run_timespan']
    input_timesteps = params_dict['NeuralNetwork_Settings']['input_timesteps']
    
    # 安全检查：确保索引不越界
    end_idx = start + run_duration + input_timesteps
    if end_idx > labels_full.shape[0]:
        end_idx = labels_full.shape[0]
        
    labels = labels_full[start:end_idx, :]

    # 4. 获取维度信息
    input_shape = params_dict['NeuralNetwork_Settings']['input_shape']
    output_shape = params_dict['NeuralNetwork_Settings']['output_shape']

    # 确保 results 和 labels 长度一致 (如果不一致，取交集长度)
    min_len = min(results.shape[0], labels.shape[0])
    results = results[:min_len, :]
    labels = labels[:min_len, :]

    # 5. 准备归一化数据 (用于计算 Scaled Metrics)
    scaler_results = MinMaxScaler(feature_range=(0, 1))
    
    # 为了保证 min/max 统计正确，将 result 和 label 拼在一起 fit
    combined_data = np.concatenate((results, labels), axis=0)
    scaler_results.fit(combined_data)
    
    results_scaled = scaler_results.transform(results)
    labels_scaled = scaler_results.transform(labels)

    # 6. 循环计算指标 (MSE/MAE)
    mse_list = []
    mae_list = []
    mse_scaled_list = []
    mae_scaled_list = []
    
    # 列名列表 (用于打印)
    col_names = []

    for dim in range(input_shape):

     


        # 区分预测变量和控制变量
        if dim < output_shape:
            type_str = "(Pred)" # 比如 v, w
        else:
            type_str = "(Ctrl)" # 比如 dt, cmd

        col_name = f"Dim {dim} {type_str}"
        col_names.append(col_name)

        # 计算原始指标
        mse = mean_squared_error(labels[:, dim], results[:, dim])
        mae = mean_absolute_error(labels[:, dim], results[:, dim])
        mse_list.append(mse)
        mae_list.append(mae)

        # 计算归一化指标
        mse_sc = mean_squared_error(labels_scaled[:, dim], results_scaled[:, dim])
        mae_sc = mean_absolute_error(labels_scaled[:, dim], results_scaled[:, dim])
        mse_scaled_list.append(mse_sc)
        mae_scaled_list.append(mae_sc)

    # 7. 打印表格 (Unscaled)
    round_digits = 5
    print('\n' + '='*80)
    print(f'METRICS (UNSCALED) - Test Run {counter}')
    print('-'*80)
    
    # 动态表头格式
    # 每个格子宽 15 字符
    header_fmt = "{:>15}" * (len(col_names) + 1)
    print(header_fmt.format("Metric", *col_names))
    
    # 打印数据行
    print(header_fmt.format("MSE", *[round(x, round_digits) for x in mse_list]))
    print(header_fmt.format("MAE", *[round(x, round_digits) for x in mae_list]))
    print('='*80)

    # 8. 打印表格 (Scaled)
    print(f'METRICS (SCALED [0,1]) - Test Run {counter}')
    print('-'*80)
    print(header_fmt.format("Metric", *col_names))
    print(header_fmt.format("MSE", *[round(x, round_digits) for x in mse_scaled_list]))
    print(header_fmt.format("MAE", *[round(x, round_digits) for x in mae_scaled_list]))
    print('='*80 + '\n')

    # 9. 循环绘图
    for dim in range(input_shape):
        # 准备数据
        val_res = results[:, dim]
        val_lbl = labels[:, dim]
        val_diff = val_lbl - val_res
        
        # 动态生成标签和文件名
        # Dim 0 -> Dim_0_Run_0.png
        ylabel = col_names[dim] # e.g., "Dim 0 (Pred)"
        filename = f"Run_{counter}_Dim_{dim}.png"
        save_path = os.path.join(path_dict['path2results_figures'], filename)
        
        plot_and_save_lsk(params_dict, val_res, val_lbl, val_diff, ylabel, save_path)




def plot_run(path_dict: dict,
             params_dict: dict,
             counter,
             start):
    """Plots test results of comparison between neural network and provided vehicle data.

    :param path_dict:       dictionary which contains paths to all relevant folders and files of this module
    :type path_dict: dict
    :param params_dict:    dictionary which contains all parameters necessary to run this module
    :type params_dict: dict
    :param counter: [description]
    :type counter: [type]
    :param start: [description]
    :type start: [type]
    """

    if params_dict['NeuralNetwork_Settings']['run_file_mode'] == 1:
        filename_model = 'prediction_result_feedforward'

    if params_dict['NeuralNetwork_Settings']['run_file_mode'] == 2:
        filename_model = 'prediction_result_recurrent'

    filepath2results = os.path.join(path_dict['path2results_matfiles'], filename_model + str(counter) + '.csv')

    # load results
    with open(filepath2results, 'r') as fh:
        results = np.loadtxt(fh)

    # load label data
    with open(path_dict['filepath2inputs_testdata'] + '.csv', 'r') as fh:
        labels = np.loadtxt(fh, delimiter=',')

    vx_result = results[:, 0][:, np.newaxis]
    vy_result = results[:, 1][:, np.newaxis]
    yaw_result = results[:, 2][:, np.newaxis]
    ax_result = results[:, 3][:, np.newaxis]
    ay_result = results[:, 4][:, np.newaxis]

    vx_label = labels[start:params_dict['Test']['run_timespan'] + start, 0][:, np.newaxis]
    vy_label = labels[start:params_dict['Test']['run_timespan'] + start, 1][:, np.newaxis]
    yaw_label = labels[start:params_dict['Test']['run_timespan'] + start, 2][:, np.newaxis]
    ax_label = labels[start:params_dict['Test']['run_timespan'] + start, 3][:, np.newaxis]
    ay_label = labels[start:params_dict['Test']['run_timespan'] + start, 4][:, np.newaxis]

    yaw_diff = yaw_label - yaw_result
    vy_diff = vy_label - vy_result
    vx_diff = vx_label - vx_result
    ay_diff = ay_label - ay_result
    ax_diff = ax_label - ax_result

    # calculate scaled results
    scaler_results = MinMaxScaler(feature_range=(0, 1))

    scaler_temp_result = np.concatenate((vx_result, vy_result, yaw_result, ax_result, ay_result), axis=1)
    scaler_temp_label = np.concatenate((vx_label, vy_label, yaw_label, ax_label, ay_label), axis=1)
    scaler_temp = np.concatenate((scaler_temp_result, scaler_temp_label), axis=0)

    scaler_results = scaler_results.fit(scaler_temp)
    scaler_temp_result = scaler_results.transform(scaler_temp_result)
    scaler_temp_label = scaler_results.transform(scaler_temp_label)

    vx_result_scaled = scaler_temp_result[:, 0]
    vy_result_scaled = scaler_temp_result[:, 1]
    yaw_result_scaled = scaler_temp_result[:, 2]
    ax_result_scaled = scaler_temp_result[:, 3]
    ay_result_scaled = scaler_temp_result[:, 4]

    vx_label_scaled = scaler_temp_label[:, 0]
    vy_label_scaled = scaler_temp_label[:, 1]
    yaw_label_scaled = scaler_temp_label[:, 2]
    ax_label_scaled = scaler_temp_label[:, 3]
    ay_label_scaled = scaler_temp_label[:, 4]

    # print deviation from label

    round_digits = 5

    print('\n')
    print('MSE AND MAE OF UNSCALED VALUES: ' + 'Test No. ' + str(counter))

    data = np.asarray([mean_squared_error(yaw_label, yaw_result),
                       mean_squared_error(vx_label, vx_result),
                       mean_squared_error(vy_label, vy_result),
                       mean_squared_error(ax_label, ax_result),
                       mean_squared_error(ay_label, ay_result),
                       mean_absolute_error(yaw_label, yaw_result),
                       mean_absolute_error(vx_label, vx_result),
                       mean_absolute_error(vy_label, vy_result),
                       mean_absolute_error(ax_label, ax_result),
                       mean_absolute_error(ay_label, ay_result)]).reshape(2, 5).round(round_digits)

    column_header = ['yaw rate', 'long. vel. vx', 'lat. vel. vy', 'long. acc. ax', 'lat. vel. ay']
    row_header = ['MSE', 'MAE']

    row_format = "{:>15}" * (len(column_header) + 1)
    print(row_format.format("", *column_header))
    for row_head, row_data in zip(row_header, data):
        print(row_format.format(row_head, *row_data))

    print('MSE AND MAE OF SCALED VALUES: ' + 'Test No. ' + str(counter))

    data = np.asarray([mean_squared_error(yaw_label_scaled, yaw_result_scaled),
                       mean_squared_error(vx_label_scaled, vx_result_scaled),
                       mean_squared_error(vy_label_scaled, vy_result_scaled),
                       mean_squared_error(ax_label_scaled, ax_result_scaled),
                       mean_squared_error(ay_label_scaled, ay_result_scaled),
                       mean_absolute_error(yaw_label_scaled, yaw_result_scaled),
                       mean_absolute_error(vx_label_scaled, vx_result_scaled),
                       mean_absolute_error(vy_label_scaled, vy_result_scaled),
                       mean_absolute_error(ax_label_scaled, ax_result_scaled),
                       mean_absolute_error(ay_label_scaled, ay_result_scaled)]).reshape(2, 5).round(round_digits)

    for row_head, row_data in zip(row_header, data):
        print(row_format.format(row_head, *row_data))

    print('\n')

    # plot and save comparsion between NN predicted and actual vehicle state
    plot_and_save(params_dict, yaw_result, yaw_label, yaw_diff, 'Yaw rate in rad/s',
                  os.path.join(path_dict['path2results_figures'], 'yaw' + str(counter) + '.png'))
    plot_and_save(params_dict, vy_result, vy_label, vy_diff, 'Lat. vel. vy in m/s',
                  os.path.join(path_dict['path2results_figures'], 'vy' + str(counter) + '.png'))
    plot_and_save(params_dict, vx_result, vx_label, vx_diff, 'Long. vel. vx in m/s',
                  os.path.join(path_dict['path2results_figures'], 'vx' + str(counter) + '.png'))
    plot_and_save(params_dict, ay_result, ay_label, ay_diff, 'Lat. acc. ay in m/s2',
                  os.path.join(path_dict['path2results_figures'], 'ay' + str(counter) + '.png'))
    plot_and_save(params_dict, ax_result, ax_label, ax_diff, 'Long. acc. ax in m/s2',
                  os.path.join(path_dict['path2results_figures'], 'ax' + str(counter) + '.png'))


# ----------------------------------------------------------------------------------------------------------------------

def plot_and_save(params_dict: dict,
                  inp_1,
                  inp_2,
                  inp_3,
                  value,
                  savename):
    """Plots and saves comparison of NN predicted and actual vehicle states values.

    :param params_dict:     dictionary which contains paths to all relevant folders and files of this module
    :type params_dict: dict
    :param inp_1:           NN predicted vehicle state value
    :type inp_1: [type]
    :param inp_2:           actual vehicle state value from test data
    :type inp_2: [type]
    :param inp_3:           calculated difference between predicted and actual vehicle state
    :type inp_3: [type]
    :param value:           name of compared vehicle state value
    :type value: [type]
    :param savename:        filename where to save plot
    :type savename: [type]
    """

    fig, (ax1, ax2) = plt.subplots(2, 1)

    ax1.plot(inp_1, label='Result', color='tab:orange')
    ax1.plot(inp_2, label='Label', color='tab:blue')
    ax2.plot(inp_3, label='Difference', color='tab:blue', linewidth=1.0)

    ax1.set_ylabel(value)
    ax2.set_ylabel('Difference label - result')
    ax1.set_xlabel('Time steps (8 ms)')
    ax2.set_xlabel('Time steps (8 ms)')
    ax1.legend()
    ax2.legend()

    if params_dict['General']['plot_result']:
        plt.show()

    if params_dict['General']['save_figures']:
        fig.savefig(savename, format='png')
        plt.close(fig)


# ----------------------------------------------------------------------------------------------------------------------

def plot_mse(path_dict: dict,
             params_dict: dict,
             histories):
    """Plots the MSE of comparion between the neural network's vehicle state output and the real vehicle state.

    :param path_dict:       dictionary which contains paths to all relevant folders and files of this module
    :type path_dict: dict
    :param params_dict:    dictionary which contains all parameters necessary to run this module
    :type params_dict: dict
    :param histories: [description]
    :type histories: [type]
    """

    # Plot training & validation accuracy values
    fig = plt.figure()

    plt.plot(histories.history[params_dict['NeuralNetwork_Settings']['Optimizer']['loss_function']])
    plt.plot(histories.history['val_' + params_dict['NeuralNetwork_Settings']['Optimizer']['loss_function']])

    plt.axis([0, params_dict['NeuralNetwork_Settings']['epochs'],
              params_dict['General']['min_scale_plot'], params_dict['General']['max_scale_plot']])

    plt.xlabel('Epoche')
    plt.ylabel(params_dict['NeuralNetwork_Settings']['Optimizer']['loss_function'])

    plt.title('Model ' + params_dict['NeuralNetwork_Settings']['Optimizer']['loss_function'])
    plt.legend(['Training loss', 'Validation loss'], loc='upper left')
    plt.show()

    fig.savefig(os.path.join(path_dict['path2results_figures'], 'loss_function.png'), format='png')
