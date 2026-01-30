import os.path
import shutil
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TerminateOnNaN, Callback 
import optuna
import src
import visualization



# 1213: 手动实现剪枝回调，解决 optuna-integration 缺失或版本不兼容的问题
class CustomPruningCallback(Callback):
    """
    Optuna 剪枝回调的手动实现版本。
    在每个 Epoch 结束时汇报 Loss，如果 Optuna 认为该实验没有前途，则抛出异常停止训练。
    """
    def __init__(self, trial, monitor):
        super(CustomPruningCallback, self).__init__()
        self.trial = trial
        self.monitor = monitor

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        # 获取当前的验证集指标
        current_score = logs.get(self.monitor)
        
        if current_score is None:
            return

        # 1. 向 Optuna 汇报当前分数
        self.trial.report(current_score, step=epoch)

        # 2. 检查是否需要剪枝
        if self.trial.should_prune():
            message = "Trial was pruned at epoch {}.".format(epoch)
            # 停止 Keras 训练
            self.model.stop_training = True
            # 抛出剪枝异常，这会被 auto_tuner.py 捕获
            raise optuna.exceptions.TrialPruned(message)


# def train_neuralnetwork(path_dict: dict,
#                         params_dict: dict,
#                         nn_mode: str) -> None:
def train_neuralnetwork(path_dict: dict,
                        params_dict: dict,
                        nn_mode: str,
                        trial=None) -> float:     # 1212: 自动超参数优化 - 修改返回类型,# 1213: 新增 trial 参数用于接收剪枝上下文
    """Manages the training process of the neural network.

    :param path_dict:           dictionary which contains paths to all relevant folders and files of this module
    :type path_dict: dict
    :param params_dict:         dictionary which contains all parameters necessary to run this module
    :type params_dict: dict
    :param nn_mode:             Neural network mode which defines type of NN (feedforward or recurrent)
    :type nn_mode: str
    """

    if not nn_mode == "feedforward" or not nn_mode == "recurrent":
        ValueError('unknown "neural network mode"; must be either "feedforard" or "recurrent"')

    print('SAVE SETTINGS')

    # 如果不是加载旧模型继续训练（即是从头训练），则备份当前的 parameters.toml.把本次训练用的超参数文件直接复制到了结果文件夹（outputs/results/）下。
    if not params_dict['General']['bool_load_existingmodel'] and trial is None:
        shutil.copyfile('params/parameters.toml',
                        os.path.join(path_dict['path2results'], 'settings_' + nn_mode + '.toml'))

    print('LOAD AND SCALE DATA')

    # 调用 load_data_for_nn 模块，读取所有的 .csv 文件
    # path_dict['path2inputs_trainingdata'] 指向 inputs/trainingdata/
    # path_dict['filename_trainingdata'] 是前缀 'data_to_train'
    data = src.load_data_for_nn.load_data(path2inputs_trainingdata=path_dict['path2inputs_trainingdata'],
                                          filename_trainingdata=path_dict['filename_trainingdata'])
    # data 是一个列表，每个元素是一个 (N, 5) 的 numpy 数组（N是该bag的行数，5是您修改后的维度）。

    # 这一行是整个训练准备阶段的核心
    # train_data: 一个元组 (train_x, train_y)。
    # val_data: 一个元组 (val_x, val_y)
    # 并完成了 归一化 (Scaler)。
    train_data, val_data = src.prepare_data.create_dataset_separation(path_dict=path_dict,
                                                                      params_dict=params_dict,
                                                                      data=data,
                                                                      nn_mode=nn_mode)

    if nn_mode == "feedforward":

        monitor = params_dict['NeuralNetwork_Settings']['Optimizer']['loss_function']

        filepath2results_trainedmodel = path_dict['filepath2results_trainedmodel_ff']

        min_delta = 0.000005

    elif nn_mode == "recurrent":

        # monitor = 'val_' + params_dict['NeuralNetwork_Settings']['Optimizer']['loss_function']
        monitor = 'val_mse'     # 强制监控验证集的 MSE
        
        filepath2results_trainedmodel = path_dict['filepath2results_trainedmodel_recurr']

        min_delta = 0.000001

    # 早停,如果 monitor (验证集Loss) 在 patience (例如60个epoch) 内没有下降，就提前停止训练。
    es = EarlyStopping(monitor=monitor,
                       mode='min',
                       verbose=1,
                       patience=params_dict['NeuralNetwork_Settings']['earlystopping_patience'])

    # 模型检查点 (ModelCheckpoint) - 自动保存最优模型
    # save_best_only=True: 只有当验证集 Loss 创下新低时，才覆盖保存模型文件。
    mc = ModelCheckpoint(filepath=filepath2results_trainedmodel,
                         monitor=monitor,
                         mode='min',
                         verbose=1,
                         save_best_only=True)

    # 学习率自动衰减 ,当 Loss 不再下降时，把学习率乘以 factor (例如 0.8)，试图让模型跳出局部最优或进行微调。
    reduce_lr_loss = ReduceLROnPlateau(monitor=monitor,
                                       factor=params_dict['NeuralNetwork_Settings']['reduceLR_factor'],
                                       patience=params_dict['NeuralNetwork_Settings']['patience_LR'],
                                       verbose=1,
                                       mode='min',
                                       min_delta=min_delta)


    # 如果 Loss 变成 NaN (梯度爆炸)，立即停止，防止浪费时间。
    Nan = TerminateOnNaN()

    # 调用 src.neural_network_fcn 构建网络结构 (Layers, Neurons, Activations)
    # 并且完成了 model.compile (设置优化器和 Loss)
    model = src.neural_network_fcn.create_nnmodel(path_dict=path_dict,
                                                  params_dict=params_dict,
                                                  nn_mode=nn_mode)

    # 1213: 剪枝核心逻辑 - 如果是由 auto_tuner 调用的，添加剪枝回调
    callbacks_list = [reduce_lr_loss, es, mc, Nan]
    if trial is not None:
        # TFKerasPruningCallback 会在每个 epoch 结束时向 Optuna 汇报 monitor 的值
        # 如果 Optuna 判定该值不合格，会抛出 TrialPruned 异常停止训练
        # pruning_callback = optuna.integration.TFKerasPruningCallback(trial, monitor)
        pruning_callback = CustomPruningCallback(trial, monitor)    # 使用我们刚才手动定义的类，不再依赖 optuna.integration
        callbacks_list.append(pruning_callback)

    # 开始训练
    history_mod = model.fit(
                            # 这是一个 numpy 数组，形状为 (样本数, 20, 5)。包含了历史 20 帧的 v, w, dt, cmd_v, cmd_w。
                            x=train_data[0],                  
                            #  这是一个 numpy 数组，形状为 (样本数, 2)。包含了下一帧真实的 v, w。模型的任务就是让输出尽可能接近这个 y。
                            y=train_data[1],      
                            # 模型不会一次看完几万条数据才更新一次权重，也不会看一条更新一次。而是每次随机抽取batch_size条数据，算一个平均误差，然后更新一次权重。这叫“小批量随机梯度下降”。                
                            batch_size=params_dict['NeuralNetwork_Settings']['batch_size'],
                            # 在每个 epoch 结束时，模型会暂停训练，用这部分数据做一次“模拟考试”。考试成绩（val_loss）只用来评估和触发早停，**绝对不会用来更新权重**（不参与训练）。
                            validation_data=(val_data[0], val_data[1]),
                            # 训练轮数
                            epochs=params_dict['NeuralNetwork_Settings']['epochs'],
                            # 日志显示模式
                            verbose=1,
                            # shuffle=True: 数据洗牌,在每个 epoch 开始前，把训练数据的顺序彻底打乱。防止模型记住数据的顺序会影响梯度方向。
                            shuffle=True,
                            # 这是训练过程的“监管者”:reduce_lr_loss: 发现学不动了（Loss不降），就减小学习率，让步子迈小点，精细调整。es : 发现再学也没长进了（验证集Loss不降反升），直接终止训练，防止过拟合。
                            # mc : 每次考出历史最好成绩，就把模型保存下来。Nan: 如果算出 NaN（梯度爆炸），立刻报错停止。
                            callbacks=callbacks_list,
                            use_multiprocessing=True)

    print(history_mod.history.keys())

    if params_dict['General']['plot_mse']:
        print('PLOT MSE CURVE')

        visualization.plot_results.plot_mse(path_dict=path_dict,
                                            params_dict=params_dict,
                                            histories=history_mod)

    # 1212: 自动超参数优化 - 获取验证集最优MSE并返回
    # 注意：monitor 变量在前面定义过，通常是 'val_mse' 或 'val_loss'
    # 如果 monitor 包含 'val_' 前缀，直接使用它从 history 中提取
    val_metric_name = 'val_mse' if 'val_mse' in history_mod.history else 'val_loss'
    best_val_score = min(history_mod.history[val_metric_name])

    return best_val_score  # 1212: 自动超参数优化 - 返回给 Optuna