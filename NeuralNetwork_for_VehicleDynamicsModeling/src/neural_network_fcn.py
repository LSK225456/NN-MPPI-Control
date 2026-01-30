from tensorflow import keras

# custom modules
import helper_funcs_NN
import tensorflow as tf
import tensorflow.keras.backend as K

"""
Created by: Rainer Trauth
Created on: 01.04.2020
"""

def physics_weighted_mse(y_true, y_pred):
    """
    自定义物理加权损失函数 (Physics-Weighted Loss)
    
    背景：
    在车辆动力学中，角速度(w)的预测误差虽然数值小，但积分后的累积影响巨大(导致轨迹发散)。
    标准的 MSE 会因为线速度(v)数值较大而忽略角速度的优化。
    
    逻辑：
    此函数人为放大了角速度残差的权重，强迫模型“更在乎”角速度的准确性。
    假设输出层维度顺序为: [diff_v, diff_w]
    """
    # 1. 计算标准的平方差: (true - pred)^2
    squared_diff = tf.square(y_true - y_pred)
    
    # 2. 定义权重向量 (超参数)
    # 第1个数 1.0  -> 对应 diff_v (线速度残差)，保持原样
    # 第2个数 10.0 -> 对应 diff_w (角速度残差)，惩罚放大 10 倍
    weights = tf.constant([1.0, 1.0060286442927682])
    
    # 3. 对平方差进行加权: squared_diff * weights
    # 这样，角速度的微小误差在 Loss 中占比会显著增加
    weighted_squared_diff = squared_diff * weights
    
    # 4. 返回加权后的平均误差 (Mean)，作为最终用于梯度下降的标量 Loss
    return K.mean(weighted_squared_diff, axis=-1)

def create_nnmodel(path_dict: dict,
                   params_dict: dict,
                   nn_mode: str):
    """Creates a new neural network model or loads an existing one.

    :param path_dict:           dictionary which contains paths to all relevant folders and files of this module
    :type path_dict: dict
    :param params_dict:         dictionary which contains all parameters necessary to run this module
    :type params_dict: dict
    :param nn_mode:             Neural network mode which defines type of NN (feedforward or recurrent)
    :type nn_mode: str
    :return: [description]
    :rtype: [type]
    """

    if not nn_mode == "feedforward" or not nn_mode == "recurrent":
        ValueError('unknown "neural network mode"; must be either "feedforard" or "recurrent"')

    if nn_mode == "feedforward":
        filepath2inputs_trainedmodel = path_dict['filepath2inputs_trainedmodel_ff']

    elif nn_mode == "recurrent":
        filepath2inputs_trainedmodel = path_dict['filepath2inputs_trainedmodel_recurr']

    if params_dict['General']['bool_load_existingmodel']:
        print('LOAD ALREADY CREATED MODEL FOR FURTHER TRAINING')
        model_create = keras.models.load_model(filepath2inputs_trainedmodel)

    else:
        print('CREATE NEW MODEL')

        if nn_mode == "feedforward":
            model_create = create_model_feedforward(path_dict=path_dict,
                                                    params_dict=params_dict)

        elif nn_mode == "recurrent":
            model_create = create_model_recurrent(path_dict=path_dict,
                                                  params_dict=params_dict)

       # 1212: 自动超参数优化 - 从参数字典读取权重 (默认10.0)
        w_omega_val = params_dict['NeuralNetwork_Settings'].get('loss_weight_omega', 1.0060286442927682)

        # 1212: 自动超参数优化 - 定义动态权重的 Loss 函数 (闭包)
        def physics_weighted_mse_dynamic(y_true, y_pred):
            squared_diff = tf.square(y_true - y_pred)
            # 使用动态读取的 w_omega_val
            weights = tf.constant([1.0, w_omega_val], dtype=tf.float32) 
            weighted_squared_diff = squared_diff * weights
            return K.mean(weighted_squared_diff, axis=-1)

        optimizer = helper_funcs_NN.src.select_optimizer.select_optimizer(          # 根据配置字符串（如 'Adam'）返回 Keras 的优化器对象。
            optimizer=params_dict['NeuralNetwork_Settings']['Optimizer']['optimizer_set'],
            learning_rate=params_dict['NeuralNetwork_Settings']['learning_rate'],
            clipnorm=params_dict['NeuralNetwork_Settings']['Optimizer']['clipnorm'])


        # model_create.compile(optimizer=optimizer,
        #                      loss=params_dict['NeuralNetwork_Settings']['Optimizer']['loss_function'],
        #                      metrics=[keras.metrics.mae, keras.metrics.mse])
        model_create.compile(
            optimizer=optimizer,
            #  指定 Loss 为我们刚才定义的物理加权函数
            loss=physics_weighted_mse_dynamic, 
            # 2. 显式指定评估指标 (Metrics)
            # 非常重要：因为现在的主 loss 是加权过的，数值会变大。
            # 我们需要显式保留 'mse' 和 'mae' 字符串指标，
            # 这样 train_neuralnetwork.py 里的 EarlyStopping 才能正确监控到 'val_mse' 或 'val_loss'。
            metrics=['mse', 'mae']
        )

        model_create.summary()

    return model_create


# ----------------------------------------------------------------------------------------------------------------------

def create_model_feedforward(path_dict: dict,
                             params_dict: dict):
    """Set up a new feedforward NN model

    :param path_dict:           dictionary which contains paths to all relevant folders and files of this module
    :type path_dict: dict
    :param params_dict:         dictionary which contains all parameters necessary to run this module
    :type params_dict: dict
    :return:                    neural network model
    :rtype: [type]
    """

    print('CREATE FEEDFORWARD NEURAL NETWORK')

    model_create = keras.models.Sequential()

    if params_dict['NeuralNetwork_Settings']['Initializer'] == "he":
        kernel_init = keras.initializers.he_uniform(seed=True)

    elif params_dict['NeuralNetwork_Settings']['Initializer'] == "glorot":
        kernel_init = keras.initializers.GlorotUniform(seed=True)

    reg_dense = keras.regularizers.l1_l2(params_dict['NeuralNetwork_Settings']['l1regularization'],
                                         params_dict['NeuralNetwork_Settings']['l2regularization'])

    input_shape = params_dict['NeuralNetwork_Settings']['input_shape'] \
        * params_dict['NeuralNetwork_Settings']['input_timesteps']

    model_create.add(
        keras.layers.Dense(input_shape=(input_shape,),
                           units=params_dict['NeuralNetwork_Settings']['Feedforward']['neurons_first_layer'],
                           use_bias=True,
                           bias_initializer='zeros',
                           activation=params_dict['NeuralNetwork_Settings']['Feedforward']['activation_1'],
                           kernel_initializer=kernel_init,
                           kernel_regularizer=reg_dense))

    if params_dict['NeuralNetwork_Settings']['Feedforward']['leakyrelu'] == 1:
        model_create.add(keras.layers.LeakyReLU(alpha=0.2))

    if params_dict['NeuralNetwork_Settings']['bool_use_dropout']:
        model_create.add(keras.layers.Dropout(params_dict['NeuralNetwork_Settings']['drop_1']))

    model_create.add(
        keras.layers.Dense(units=params_dict['NeuralNetwork_Settings']['Feedforward']['neurons_second_layer'],
                           bias_initializer='zeros',
                           use_bias=True,
                           kernel_initializer=kernel_init,
                           activation=params_dict['NeuralNetwork_Settings']['Feedforward']['activation_2'],
                           kernel_regularizer=reg_dense))

    if params_dict['NeuralNetwork_Settings']['Feedforward']['leakyrelu'] == 1:
        model_create.add(keras.layers.LeakyReLU(alpha=0.2))

    if params_dict['NeuralNetwork_Settings']['bool_use_dropout']:
        model_create.add(keras.layers.Dropout(params_dict['NeuralNetwork_Settings']['drop_2']))

    model_create.add(
        keras.layers.Dense(units=params_dict['NeuralNetwork_Settings']['output_shape'], activation='linear'))

    return model_create


# ----------------------------------------------------------------------------------------------------------------------
def create_model_recurrent(path_dict: dict,
                           params_dict: dict):
    """Set up a new recurrent NN model

    :param path_dict:           dictionary which contains paths to all relevant folders and files of this module
    :type path_dict: dict
    :param params_dict:         dictionary which contains all parameters necessary to run this module
    :type params_dict: dict
    :return:                    neural network model
    :rtype: [type]
    """

    print('CREATE RECURRENT NEURAL NETWORK')

    model_create = keras.models.Sequential()

    if params_dict['NeuralNetwork_Settings']['Initializer'] == "he":
        kernel_init = keras.initializers.he_uniform(seed=True)

    elif params_dict['NeuralNetwork_Settings']['Initializer'] == "glorot":
        kernel_init = keras.initializers.GlorotUniform(seed=True)

    reg_layer = keras.regularizers.l1_l2(params_dict['NeuralNetwork_Settings']['l1regularization'],
                                         params_dict['NeuralNetwork_Settings']['l2regularization'])

    # load specified recurrent layer type (from parameter file)
    if params_dict['NeuralNetwork_Settings']['Recurrent']['recurrent_mode'] == 'GRU':
        recurrent_mode = keras.layers.GRU

    elif params_dict['NeuralNetwork_Settings']['Recurrent']['recurrent_mode'] == 'LSTM':
        recurrent_mode = keras.layers.LSTM

    elif params_dict['NeuralNetwork_Settings']['Recurrent']['recurrent_mode'] == 'SimpleRNN':
        recurrent_mode = keras.layers.SimpleRNN

    elif params_dict['NeuralNetwork_Settings']['Recurrent']['recurrent_mode'] == 'ConvLSTM2D':
        recurrent_mode = keras.layers.ConvLSTM2D

    elif params_dict['NeuralNetwork_Settings']['Recurrent']['recurrent_mode'] == 'RNN':
        recurrent_mode = keras.layers.RNN

    model_create.add(
        recurrent_mode(input_shape=(params_dict['NeuralNetwork_Settings']['input_timesteps'],
                                    params_dict['NeuralNetwork_Settings']['input_shape']),
                       units=params_dict['NeuralNetwork_Settings']['Recurrent']['neurons_first_layer_recurrent'],
                       return_sequences=True,           # 1212: 解决欠拟合 - 关键！让第一层输出完整序列
                       use_bias=True,
                       bias_initializer='zeros',
                    #    kernel_initializer=kernel_init,        #todo:0129移除 kernel_initializer 参数
                       kernel_regularizer=reg_layer,
                       activation=params_dict['NeuralNetwork_Settings']['Recurrent']['activation_1_recurrent']))

    if params_dict['NeuralNetwork_Settings']['bool_use_dropout']:
        model_create.add(keras.layers.Dropout(params_dict['NeuralNetwork_Settings']['drop_1']))

    # 新增：第二层循环层：return_sequences=False，提取最终状态
    # 这里我们复用 neurons_first_layer_recurrent 的大小，保证主干通道宽阔
    model_create.add(
        recurrent_mode(units=params_dict['NeuralNetwork_Settings']['Recurrent']['neurons_first_layer_recurrent'], # 1212: 解决欠拟合 - 第二层保持宽度
                       return_sequences=False, # 1212: 解决欠拟合 - 最后一层循环层压缩时间维
                       use_bias=True,
                       bias_initializer='zeros',
                    #    kernel_initializer=kernel_init,    #todo:0129移除 kernel_initializer 参数
                       kernel_regularizer=reg_layer,
                       activation=params_dict['NeuralNetwork_Settings']['Recurrent']['activation_1_recurrent']))
    
    if params_dict['NeuralNetwork_Settings']['bool_use_dropout']:
        model_create.add(keras.layers.Dropout(params_dict['NeuralNetwork_Settings']['drop_2']))

    model_create.add(
        keras.layers.Dense(units=params_dict['NeuralNetwork_Settings']['Recurrent']['neurons_second_layer_recurrent'],
                           use_bias=True,
                           bias_initializer='zeros',
                        #    kernel_initializer=kernel_init,     #todo:0129移除 kernel_initializer 参数
                           activation=params_dict['NeuralNetwork_Settings']['Recurrent']['activation_dense_recurrent'],
                           kernel_regularizer=reg_layer))

    # if params_dict['NeuralNetwork_Settings']['bool_use_dropout']:
    #     model_create.add(keras.layers.Dropout(params_dict['NeuralNetwork_Settings']['drop_2']))

    model_create.add(
        keras.layers.Dense(units=params_dict['NeuralNetwork_Settings']['output_shape'], activation='linear'))

    return model_create




# def create_model_recurrent(path_dict: dict,
#                            params_dict: dict):
#     """Set up a new recurrent NN model

#     :param path_dict:           dictionary which contains paths to all relevant folders and files of this module
#     :type path_dict: dict
#     :param params_dict:         dictionary which contains all parameters necessary to run this module
#     :type params_dict: dict
#     :return:                    neural network model
#     :rtype: [type]
#     """

#     print('CREATE RECURRENT NEURAL NETWORK')

#     model_create = keras.models.Sequential()

#     if params_dict['NeuralNetwork_Settings']['Initializer'] == "he":
#         kernel_init = keras.initializers.he_uniform(seed=True)

#     elif params_dict['NeuralNetwork_Settings']['Initializer'] == "glorot":
#         kernel_init = keras.initializers.GlorotUniform(seed=True)

#     reg_layer = keras.regularizers.l1_l2(params_dict['NeuralNetwork_Settings']['l1regularization'],
#                                          params_dict['NeuralNetwork_Settings']['l2regularization'])

#     # load specified recurrent layer type (from parameter file)
#     if params_dict['NeuralNetwork_Settings']['Recurrent']['recurrent_mode'] == 'GRU':
#         recurrent_mode = keras.layers.GRU

#     elif params_dict['NeuralNetwork_Settings']['Recurrent']['recurrent_mode'] == 'LSTM':
#         recurrent_mode = keras.layers.LSTM

#     elif params_dict['NeuralNetwork_Settings']['Recurrent']['recurrent_mode'] == 'SimpleRNN':
#         recurrent_mode = keras.layers.SimpleRNN

#     elif params_dict['NeuralNetwork_Settings']['Recurrent']['recurrent_mode'] == 'ConvLSTM2D':
#         recurrent_mode = keras.layers.ConvLSTM2D

#     elif params_dict['NeuralNetwork_Settings']['Recurrent']['recurrent_mode'] == 'RNN':
#         recurrent_mode = keras.layers.RNN

#     model_create.add(
#         recurrent_mode(input_shape=(params_dict['NeuralNetwork_Settings']['input_timesteps'],
#                                     params_dict['NeuralNetwork_Settings']['input_shape']),
#                        units=params_dict['NeuralNetwork_Settings']['Recurrent']['neurons_first_layer_recurrent'],
#                        return_sequences=False,
#                        use_bias=True,
#                        bias_initializer='zeros',
#                        kernel_initializer=kernel_init,
#                        kernel_regularizer=reg_layer,
#                        activation=params_dict['NeuralNetwork_Settings']['Recurrent']['activation_1_recurrent']))

#     if params_dict['NeuralNetwork_Settings']['bool_use_dropout']:
#         model_create.add(keras.layers.Dropout(params_dict['NeuralNetwork_Settings']['drop_1']))

#     model_create.add(
#         keras.layers.Dense(units=params_dict['NeuralNetwork_Settings']['Recurrent']['neurons_second_layer_recurrent'],
#                            use_bias=True,
#                            bias_initializer='zeros',
#                            kernel_initializer=kernel_init,
#                            activation=params_dict['NeuralNetwork_Settings']['Recurrent']['activation_dense_recurrent'],
#                            kernel_regularizer=reg_layer))

#     if params_dict['NeuralNetwork_Settings']['bool_use_dropout']:
#         model_create.add(keras.layers.Dropout(params_dict['NeuralNetwork_Settings']['drop_2']))

#     model_create.add(
#         keras.layers.Dense(units=params_dict['NeuralNetwork_Settings']['output_shape'], activation='linear'))

#     return model_create
