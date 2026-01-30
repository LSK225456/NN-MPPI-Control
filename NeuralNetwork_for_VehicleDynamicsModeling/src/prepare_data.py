import numpy as np
import os.path
import pickle
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from joblib import dump, load

"""
Created by: Rainer Trauth
Created on: 01.04.2020
"""


def scaler(path_dict: dict,
           params_dict: dict,
           dataset: np.array) -> np.array:
    """Scales dataset during preprocessing.

    :param path_dict:           dictionary which contains paths to all relevant folders and files of this module
    :type path_dict: dict
    :param params_dict:         dictionary which contains all parameters necessary to run this module
    :type params_dict: dict
    :param dataset:             dataset which should get scaled
    :type dataset: np.array
    :return:                    scaled dataset
    :rtype: np.array
    """

    if params_dict['General']['use_old_transformation']:

        if params_dict['General']['scaler_mode'] == 2:

            with open(os.path.join(path_dict['path2results'], 'scaler_tanh'), 'rb') as f:
                m, std = pickle.load(f)
                dataset_out = 0.5 * (np.tanh(0.01 * ((dataset - m) / std)) + 1)

        else:
            print('USE OLD TRANSFORMATION')
            scalers = load(path_dict['filepath2scaler_load'])
            dataset_out = scalers.transform(dataset)

    else:

        if params_dict['General']['scaler_mode'] == 0:
            print('USE STANDARD SCALER')
            scalers = StandardScaler()  # with_mean=True, with_std=True
            scalers = scalers.fit(dataset)
            dataset_out = scalers.transform(dataset)

        if params_dict['General']['scaler_mode'] == 1:
            print('USE MINMAX SCALER')
            scalers = MinMaxScaler(feature_range=(-1, 1))
            scalers = scalers.fit(dataset)
            dataset_out = scalers.transform(dataset)

        if params_dict['General']['scaler_mode'] == 2:
            m = np.mean(dataset, axis=0)
            std = np.std(dataset, axis=0)
            dataset_out = 0.5 * (np.tanh(0.01 * ((dataset - m) / std)) + 1)

        # save scaling information
        if params_dict['General']['scaler_mode'] == 2:

            with open(os.path.join(path_dict['path2results'], 'scaler_tanh'), 'wb') as f:
                pickle.dump([m, std], f)

        else:
            dump(scalers, path_dict['filepath2scaler_save'])

    return dataset_out


# ----------------------------------------------------------------------------------------------------------------------

def scaler_run(path2scaler: str,
               params_dict: dict,
               dataset: np.array):
    """Scales dataset for runing NN test.

    :param path_dict:           dictionary which contains paths to all relevant folders and files of this module
    :type path_dict: dict
    :param params_dict:         dictionary which contains all parameters necessary to run this module
    :type params_dict: dict
    :param dataset:             dataset which should get scaled
    :type dataset: np.array
    :return:                    scaled dataset
    :rtype: np.array
    """

    if params_dict['General']['scaler_mode'] == 2:

        with open('outputs/scaler_tanh', 'rb') as f:
            m, std = pickle.load(f)
            dataset_out = 0.5 * (np.tanh(0.01 * ((dataset - m) / std)) + 1)

    else:
        scalers = load(path2scaler)
        dataset_out = scalers.transform(dataset)

    return dataset_out


# ----------------------------------------------------------------------------------------------------------------------

def scaler_reverse(path2scaler: str,
                   params_dict: dict,
                   dataset: np.array) -> np.array:
    """Rescaled dataset to physical quantities.

    :param path_dict:           dictionary which contains paths to all relevant folders and files of this module
    :type path_dict: dict
    :param params_dict:         dictionary which contains all parameters necessary to run this module
    :type params_dict: dict
    :param dataset:             dataset which should get rescaled
    :type dataset: np.array
    :return:                    rescaled dataset
    :rtype: np.array
    """

    print('TRANSFORM RESULT WITH SCALER TO PHYSICAL QUANTITIES')

    if params_dict['General']['scaler_mode'] == 2:

        with open('outputs/scaler_tanh', 'rb') as f:
            m, std = pickle.load(f)
            dataset_std_rev = m + 100 * std * np.arctanh(2 * dataset - 1)

    else:
        scalers = load(path2scaler)
        dataset_std_rev = scalers.inverse_transform(dataset)

    return dataset_std_rev


# ----------------------------------------------------------------------------------------------------------------------

def create_dataset_separation_run(data_test: np.array,
                                  params_dict: dict,
                                  start: int,
                                  duration: int,
                                  nn_mode: str) -> tuple:
    """Creates a dataset to test the neural network against actual vehicle data.

    :param data_test:           vehicle input data of test data file
    :type data_test: np.array
    :param params_dict:         dictionary which contains all parameters necessary to run this module
    :type params_dict: dict
    :param start:               start index (row) of test data file
    :type start: int
    :param duration:            duration of one test setion (as timesteps)
    :type duration: int
    :param nn_mode:             Neural network mode which defines type of NN (feedforward or recurrent)
    :type nn_mode: str
    :return:                    vehicle input data of test data file
    :rtype: tuple
    """

    if nn_mode != "feedforward" and nn_mode != "recurrent":
        ValueError('unknown "neural network mode"; must be either "feedforard" or "recurrent"')

    input_shape = params_dict['NeuralNetwork_Settings']['input_shape']
    output_shape = params_dict['NeuralNetwork_Settings']['output_shape']
    input_timesteps = params_dict['NeuralNetwork_Settings']['input_timesteps']

    initials = data_test[start:start + input_timesteps, :]

    if nn_mode == "feedforward":
        initials = np.reshape(initials, (1, input_timesteps * input_shape))

    elif nn_mode == "recurrent":
        initials = np.reshape(initials, (1, input_timesteps, input_shape))

    ###### 原来的硬编码提取部分
    # steeringangle_rad = data_test[start + input_timesteps:start + duration, output_shape]

    # torqueRL_Nm = data_test[start + input_timesteps:start + duration, output_shape + 1]
    # torqueRR_Nm = data_test[start + input_timesteps:start + duration, output_shape + 2]

    # brakepresF_bar = data_test[start + input_timesteps:start + duration, output_shape + 3]
    # brakepresR_bar = data_test[start + input_timesteps:start + duration, output_shape + 4]

    # return initials, steeringangle_rad, torqueRL_Nm, torqueRR_Nm, brakepresF_bar, brakepresR_bar

    # [新增] 自适应提取剩余所有列作为未来输入
    # 逻辑：从 output_shape (即第2列) 开始一直取到最后，无论后面有几列，统统算作控制输入
    future_inputs = data_test[start + input_timesteps:start + duration, output_shape:] # 推理硬编码修改

    # [修改] 返回值改为通用的 future_inputs
    return initials, future_inputs # 推理硬编码修改


# ----------------------------------------------------------------------------------------------------------------------

def extract_part(datax,
                 params_dict: dict,
                 data_infox,
                 z):

    summ = np.sum(data_infox[0:1 + z, :])
    data_part = datax[summ - data_infox[z, 0]:summ, :]
    labels_part = data_part[2 * params_dict['NeuralNetwork_Settings']['input_timesteps']
                            - 1::params_dict['NeuralNetwork_Settings']['input_timesteps'],
                            0:params_dict['NeuralNetwork_Settings']['output_shape']]

    data_part = data_part[0:len(data_part) - params_dict['NeuralNetwork_Settings']['input_timesteps'], :]

    return np.array(data_part), np.array(labels_part)


# ----------------------------------------------------------------------------------------------------------------------

def create_dataset_separation(path_dict: dict,
                              params_dict: dict,
                              data: dict,
                              nn_mode: str) -> tuple:
    """Creates a training and validation dataset out of the provided data.

    :param path_dict:           dictionary which contains paths to all relevant folders and files of this module
    :type path_dict: dict
    :param params_dict:         dictionary which contains all parameters necessary to run this module
    :type params_dict: dict
    :param data:                complete dataset which should be used for NN training
    :type data: dict
    :param nn_mode:             Neural network mode which defines type of NN (feedforward or recurrent)
    :type nn_mode: str
    :return:                    training and validation datasets
    :rtype: tuple
    """

    if nn_mode != "feedforward" and nn_mode != "recurrent":
        ValueError('unknown "neural network mode"; must be either "feedforard" or "recurrent"')

    input_shape = params_dict['NeuralNetwork_Settings']['input_shape']
    output_shape = params_dict['NeuralNetwork_Settings']['output_shape']
    input_timesteps = params_dict['NeuralNetwork_Settings']['input_timesteps']

    # 引入残差网络
    pred_target = params_dict['NeuralNetwork_Settings'].get('prediction_target', 'state')

    # 引入多步损失
    enable_seq_train = params_dict.get('Training_Hyperparameters', {}).get('enable_sequence_training', False)
    seq_len = params_dict.get('Training_Hyperparameters', {}).get('sequence_length', 1)
    
    if not enable_seq_train:
        seq_len = 1

    # count training data files
    file_counting = 0
    filepath = path_dict['path2inputs_trainingdata']

    if os.path.exists(filepath):
        for file in os.listdir(filepath):
            if file.startswith('data_to_train'):
                file_counting += 1

    # generate datasets预计算总数据量，以便分配内存
    lengthsum = 0
    lengthsumtwo = 0
    lengthsumtwolabels = 0

    # 遍历每个文件，计算有效样本数
    # 一个长度为 L 的文件，如果是 input_timesteps=20
    # 能切出的样本数是 L - 20 (因为前 20 帧无法作为 Label，它们是历史)
    for m in range(0, file_counting):
        if enable_seq_train:
            # 比如 seq_len=5, 我们需要保证 idx+seq_len 不越界，所以有效长度减少 (seq_len - 1)
            valid_len = len(data[m]) - input_timesteps - (seq_len - 1)
        else:
            valid_len = len(data[m]) - input_timesteps
            
        # 引入多步损失：防止数据太短导致的负数
        if valid_len > 0:
            lengthsum += valid_len

    data_train = np.zeros((lengthsum * input_timesteps, input_shape))       # data_train: 存放输入 X。注意这里的形状是平铺的：(总样本数 * 20, 5)
    # 引入多步损失：根据是否开启多步，初始化不同形状的 data_labels
    if enable_seq_train:
        # 形状变为 (N, sequence_length, output_shape)
        data_labels = np.zeros((lengthsum, seq_len, output_shape))
    else:
        # 原有逻辑：形状为 (N, output_shape)
        data_labels = np.zeros((lengthsum, output_shape))

    # generate the training dataset which contains 'input_timesteps' number of previous data points
    for u in range(0, file_counting):

        # --- 填充 Label (Y) ---
        # 逻辑：取从第 20 行开始的数据，取前 output_shape (2) 列
        # 对应数据列：[v, w]
        
        # 引入多步损失：计算当前文件的有效样本数
        if enable_seq_train:
            current_valid_len = len(data[u]) - input_timesteps - (seq_len - 1)
        else:
            current_valid_len = len(data[u]) - input_timesteps

        if current_valid_len <= 0: continue

        # 引入多步损失：填充 Labels 数据 // 11.15
        if enable_seq_train:
            # 使用循环构建多步标签 (虽然比向量化慢一点点，但逻辑最清晰且稳健)
            # data_labels 的形状是 [总样本数, 预测步数, 输出维度]
            for i in range(current_valid_len):
                # 截取从当前时刻往后 seq_len 步的数据作为 Label
                # 这里的 input_timesteps 是输入的结束点，也是预测的开始点
                start_idx = input_timesteps + i
                segment = data[u][start_idx : start_idx + seq_len, 0:output_shape]
                data_labels[lengthsumtwolabels + i, :, :] = segment
        else:
            target_data = (data[u])[input_timesteps:, 0:output_shape]


        # label calculate
        current_labels = target_data

        if enable_seq_train:
            return      #todo
        else:
            data_labels[lengthsumtwolabels :    lengthsumtwolabels + current_valid_len, :] = current_labels
    
        # [修改] 提取 Input 数据 # 残差网络
        # 现在的 CSV 结构是 [Diff_v, Diff_w, v, w, dt, cmd_v, cmd_w]
        # Label 占前 output_shape 列 (0:2)
        # Input 占后 input_shape 列 (2:7)
        start_col = output_shape
        end_col = output_shape + input_shape

        for pp in range(0, current_valid_len):
            idx = lengthsumtwo + pp * input_timesteps
            # data_train[idx:idx + input_timesteps, :] = (data[u])[pp:pp + input_timesteps, :]
            # [修改] 显式切片 start_col:end_col，只读取后5列物理状态作为输入
            data_train[idx:idx + input_timesteps, :] = (data[u])[pp:pp + input_timesteps, start_col:end_col] # 残差网络

        lengthsumtwolabels += current_valid_len
        lengthsumtwo += (current_valid_len * input_timesteps)

    # reshape training dataset
    if nn_mode == "feedforward":
        data_train = np.reshape(data_train, (len(data_labels), input_timesteps * input_shape))

    elif nn_mode == "recurrent":    # 将平铺的 X 变为三维张量: (样本数, 20, 5)
        data_train = np.reshape(data_train, (len(data_train) // input_timesteps, input_timesteps, input_shape))

    # 1218：数据集分割 - 读取参数
    num_samples = len(data_train)
    val_split = params_dict['NeuralNetwork_Settings']['val_split']
    use_block_split = params_dict['NeuralNetwork_Settings'].get('use_block_split', False)  # 1218：数据集分割 - 读取分割方式开关
    
    if use_block_split:
        # 1218：交错块状分割数据集 - 读取块分割相关参数
        block_size = params_dict['NeuralNetwork_Settings'].get('block_size', 500)
        block_seed = params_dict['NeuralNetwork_Settings'].get('block_shuffle_seed', 42)
        gap_size = input_timesteps  # gap 等于输入窗口大小，确保无重叠
    
        # ========== 1218：交错块状分割模式 ==========
        # 1218：交错块状分割数据集 - 计算块的边界索引
        # 每个"有效块"包含 block_size 个样本，块与块之间跳过 gap_size 个样本
        block_start_indices = []  # 1218：交错块状分割数据集 - 存储每个块的起始索引
        current_idx = 0  # 1218：交错块状分割数据集
        while current_idx + block_size <= num_samples:  # 1218：交错块状分割数据集
            block_start_indices.append(current_idx)  # 1218：交错块状分割数据集
            current_idx += block_size + gap_size  # 1218：交错块状分割数据集 - 跳过 gap
        
        num_blocks = len(block_start_indices)  # 1218：交错块状分割数据集
        print(f"[1218] 交错块状分割: 总样本数={num_samples}, 块大小={block_size}, gap={gap_size}, 块数={num_blocks}")  # 1218：交错块状分割数据集
        
        # 1218：交错块状分割数据集 - 随机分配块到训练集/验证集
        block_indices = np.arange(num_blocks)  # 1218：交错块状分割数据集
        np.random.RandomState(block_seed).shuffle(block_indices)  # 1218：交错块状分割数据集 - 固定种子保证可复现
        
        num_val_blocks = max(1, int(num_blocks * val_split))  # 1218：交错块状分割数据集 - 至少1个验证块
        num_train_blocks = num_blocks - num_val_blocks  # 1218：交错块状分割数据集
        
        val_block_indices = block_indices[:num_val_blocks]  # 1218：交错块状分割数据集 - 前 val_split 的块作为验证集
        train_block_indices = block_indices[num_val_blocks:]  # 1218：交错块状分割数据集 - 剩余作为训练集
        
        print(f"[1218] 训练块数={num_train_blocks}, 验证块数={num_val_blocks}")  # 1218：交错块状分割数据集
        
        # 1218：交错块状分割数据集 - 提取训练集样本索引
        train_sample_indices = []  # 1218：交错块状分割数据集
        for blk_idx in train_block_indices:  # 1218：交错块状分割数据集
            start = block_start_indices[blk_idx]  # 1218：交错块状分割数据集
            end = start + block_size  # 1218：交错块状分割数据集
            train_sample_indices.extend(range(start, end))  # 1218：交错块状分割数据集
        
        # 1218：交错块状分割数据集 - 提取验证集样本索引
        val_sample_indices = []  # 1218：交错块状分割数据集
        for blk_idx in val_block_indices:  # 1218：交错块状分割数据集
            start = block_start_indices[blk_idx]  # 1218：交错块状分割数据集
            end = start + block_size  # 1218：交错块状分割数据集
            val_sample_indices.extend(range(start, end))  # 1218：交错块状分割数据集
        
        train_sample_indices = np.array(train_sample_indices)  # 1218：交错块状分割数据集
        val_sample_indices = np.array(val_sample_indices)  # 1218：交错块状分割数据集
        
        # 1218：交错块状分割数据集 - 分割数据
        train_x_raw = data_train[train_sample_indices]  # 1218：交错块状分割数据集
        train_y_raw = data_labels[train_sample_indices]  # 1218：交错块状分割数据集
        val_x_raw = data_train[val_sample_indices]  # 1218：交错块状分割数据集
        val_y_raw = data_labels[val_sample_indices]  # 1218：交错块状分割数据集
        
        print(f"[1218] 训练样本数={len(train_x_raw)}, 验证样本数={len(val_x_raw)}")  # 1218：交错块状分割数据集
    
    else:
        # ========== 1218：时间顺序分割模式（传统方式）==========
        split_idx = int(num_samples * (1 - val_split))  # 1218：时间顺序分割 - 按比例计算分割点
        
        train_x_raw = data_train[:split_idx]  # 1218：时间顺序分割 - 前部分作为训练集
        train_y_raw = data_labels[:split_idx]  # 1218：时间顺序分割
        val_x_raw = data_train[split_idx:]  # 1218：时间顺序分割 - 后部分作为验证集
        val_y_raw = data_labels[split_idx:]  # 1218：时间顺序分割
        
        print(f"[1218] 时间顺序分割: 总样本数={num_samples}, 训练样本数={len(train_x_raw)}, 验证样本数={len(val_x_raw)}")  # 1218：时间顺序分割

    # 1218：数据集分割 - 打乱训练集内部顺序（两种分割方式共用）
    indices_train = np.arange(len(train_x_raw))  # 1218：数据集分割

    if params_dict['General']['shuffle_mode']:
        np.random.RandomState(params_dict['General']['shuffle_number']).shuffle(indices_train)
    else:
        np.random.shuffle(indices_train)

    # 应用打乱到训练集
    train_x = train_x_raw[indices_train]  # 1218：数据集分割
    train_y = train_y_raw[indices_train]  # 1218：数据集分割

    # 验证集保持原序，不打乱
    val_x = val_x_raw  # 1218：数据集分割
    val_y = val_y_raw  # 1218：数据集分割


    train_x_for_scaler = np.reshape(train_x, (-1, input_shape)) 
    val_x_for_scaler = np.reshape(val_x, (-1, input_shape))


    # 对训练集：Fit + Transform
    # 注意：scaler 函数内部会保存 scaler 文件
    train_x_scaled_flat = scaler(path_dict=path_dict,
                                 params_dict=params_dict,
                                 dataset=train_x_for_scaler)

    # 对验证集：Transform Only (使用刚才保存的 scaler)
    val_x_scaled_flat = scaler_run(path2scaler=path_dict['filepath2scaler_save'],
                                   params_dict=params_dict,
                                   dataset=val_x_for_scaler)

    # ----------------------------------------------------------------------------------------------
    # 步骤5：恢复维度 (Reshape back)
    # ----------------------------------------------------------------------------------------------
    
    if nn_mode == "recurrent":
        # 从 (N*20, 5) 恢复回 (N, 20, 5)
        train_x = np.reshape(train_x_scaled_flat, (len(train_y), input_timesteps, input_shape))
        val_x = np.reshape(val_x_scaled_flat, (len(val_y), input_timesteps, input_shape))
    else:
        train_x = np.reshape(train_x_scaled_flat, (len(train_y), input_timesteps * input_shape))
        val_x = np.reshape(val_x_scaled_flat, (len(val_y), input_timesteps * input_shape))

    # ----------------------------------------------------------------------------------------------
    # 步骤6：标签归一化 (Label Scaling)
    # ----------------------------------------------------------------------------------------------
    path2scaler_y = os.path.join(path_dict['path2results'], 'scaler_y.joblib')

    if enable_seq_train:
        return # 暂不处理多步 Label
    else:
        # 训练集标签归一化
        scaler_y = StandardScaler() # 或者 MinMaxScaler(feature_range=(-1, 1))
        scaler_y.fit(train_y)
        # 2. 保存 Label Scaler (供推理时反归一化使用)
        dump(scaler_y, path2scaler_y)
        print(f"Label Scaler saved to: {path2scaler_y}")
        # 3. 转换训练集和验证集
        train_y = scaler_y.transform(train_y)
        val_y = scaler_y.transform(val_y)

    # # shuffle training dataset打乱数据
    # indices = np.arange(data_train.shape[0])

    # if params_dict['General']['shuffle_mode']:
    #     np.random.RandomState(params_dict['General']['shuffle_number']).shuffle(indices)

    # else:
    #     np.random.shuffle(indices)

    # data_labels = data_labels[indices]
    # if nn_mode == "recurrent":
    #      # 如果是 recurrent，data_train 此时是 (N, T, F)，indices 也是针对 N 的
    #      data_train_shuffled = data_train[indices]
    #      # 压扁以适配 scaler 接口
    #      data_train = np.reshape(data_train_shuffled, (len(data_labels) * input_timesteps, input_shape))
    # else:
    #      data_train = data_train[indices]
    #      data_train = np.reshape(data_train, (len(data_labels) * input_timesteps, input_shape))


    # # split provided dataset into training and validation datasets
    # p = int(len(data_train) * (1 - params_dict['NeuralNetwork_Settings']['val_split']))

    # mod = p % input_timesteps
    # p = p - mod

    # train_x = scaler(path_dict=path_dict,
    #                  params_dict=params_dict,
    #                  dataset=data_train[0:p, :])
    

    # # if pred_target == 'residual':

    # #     # 引入多步损失：处理 data_labels (train_y) 的归一化 // 11.15
    # #     if enable_seq_train:
    # #         # 多步模式下，data_labels 是 (N, Seq, Output)。Scaler 需要 (N, Input)。
    # #         # 1. 压扁成 (N * Seq, Output)
    # #         N, S, O = data_labels.shape
    # #         flat_labels = np.reshape(data_labels, (N * S, O))

    # #     else:
    # #         flat_labels = data_labels

    # #     num_train_samples = p // input_timesteps
    # #     train_rows = num_train_samples * (seq_len if enable_seq_train else 1)
    # #     train_labels_raw = flat_labels[0 : train_rows,:]

    # #     scaler_y = StandardScaler()
    # #     scaler_y.fit(train_labels_raw)

    # #     dump(scaler_y, os.path.join(path_dict['path2results'],  'scaler_y_residual.joblib'))
    # #     flat_labels_scaled =     scaler_y.transform(flat_labels)
    # #     if enable_seq_train:
    # #         return
    # #     else:
    # #         data_labels = flat_labels_scaled

  
    # if enable_seq_train:
    #     return
    # else:
    #     temp = np.zeros((len(data_labels), input_shape))
    #     temp[:, 0:output_shape] = data_labels

    #     temp = scaler_run(path2scaler=path_dict['filepath2scaler_save'],
    #                     params_dict=params_dict,
    #                     dataset=temp)

    #     data_labels = temp[:, 0:output_shape]


    # # prepare training data
    # if nn_mode == "feedforward":
    #     train_x = np.reshape(train_x, (p // input_timesteps, input_timesteps * input_shape))

    # elif nn_mode == "recurrent":
    #     train_x = np.reshape(train_x, (p // input_timesteps, input_timesteps, input_shape))


    # num_train_samples = p // input_timesteps
    # train_y = data_labels[0:num_train_samples, :]

    # # prepare validation data
    # val_x = data_train[p:len(data_train), :]
    # val_x = scaler_run(path2scaler=path_dict['filepath2scaler_save'],
    #                    params_dict=params_dict,
    #                    dataset=val_x)

    # if nn_mode == "feedforward":
    #     val_x = np.reshape(val_x, ((len(data_train) - p * input_timesteps), input_timesteps * input_shape))

    # elif nn_mode == "recurrent":
    #     val_x = np.reshape(val_x, ((len(data_train) - p) // input_timesteps, input_timesteps, input_shape))

    # val_y = data_labels[num_train_samples:len(data_labels), :]

    return (train_x, train_y), (val_x, val_y)
