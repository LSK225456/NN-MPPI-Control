import os
import sys
import copy
import toml
import optuna
import logging
import traceback
from datetime import datetime
import tensorflow.keras.backend as K
import gc # 垃圾回收

# ==============================================================================
# 1. 全局配置区 (CONFIGURATION)
# ==============================================================================

# 实验名称与存储路径
STUDY_NAME = "2025_1213_v6"
STORAGE_DB = "sqlite:///optuna_vehicle_dynamics.db"  # 使用SQLite数据库

# 最优参数保存路径
BEST_PARAMS_OUTPUT_PATH = 'params/best_params_optuna.toml'

# 优化设置
N_TRIALS = 500         # 尝试的总次数
N_JOBS = 1             # 并行线程数
TIMEOUT = None         # 超时时间 (秒)

# 确保能导入项目模块
sys.path.append(os.getcwd())

# 导入模块 (根据你的项目结构)
try:
    # 根据之前的上下文，这些文件位于 helper_funcs_NN/src/ 和 src/ 下
    from helper_funcs_NN.src import manage_paths, handle_params
    from src import train_neuralnetwork
except ImportError as e:
    print("Error: 无法导入项目模块，请确保脚本位于项目根目录，且 helper_funcs_NN 和 src 文件夹存在。")
    raise e

# 配置日志
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


# ==============================================================================
# 2. 功能模块区 (FUNCTIONS)
# ==============================================================================

def load_base_config():
    """
    加载基础配置和路径字典
    修正点：使用正确的函数名称 manage_paths() 和 handle_params()
    """
    # 1. 获取路径字典
    # [修正] 原 check_params -> manage_paths.manage_paths()
    path_dict = manage_paths.manage_paths() 

    # 2. 读取原始参数
    # [修正] 原 check_params -> handle_params.handle_params(path_dict)
    # 注意：handle_params 内部会自动寻找 path_dict['filepath2params']
    params_dict = handle_params.handle_params(path_dict=path_dict)
    
    return path_dict, params_dict


def suggest_hyperparameters(trial, params):
    """
    定义超参数搜索空间
    """
    # === 1. 数据处理参数 ===
    # 历史步长：决定看过去多少帧数据 (影响最大，需重新切分数据)
    params['NeuralNetwork_Settings']['input_timesteps'] = trial.suggest_int('input_timesteps', 10, 60, step=5)
    
    # === 2. 物理 Loss 权重 (这是我们在 neural_network_fcn.py 中新加的参数) ===
    # 建议范围大一点，观察模型对权重的敏感度
    params['NeuralNetwork_Settings']['loss_weight_omega'] = trial.suggest_float('loss_weight_omega', 1.0, 50.0)

    # === 3. 网络结构参数 ===
    # 第一层神经元数量
    params['NeuralNetwork_Settings']['Recurrent']['neurons_first_layer_recurrent'] = trial.suggest_int('n_layer_1', 64, 256, step=32)
    # 第二层神经元数量
    params['NeuralNetwork_Settings']['Recurrent']['neurons_second_layer_recurrent'] = trial.suggest_int('n_layer_2', 32, 256, step=32)
    
    # === 4. 正则化参数 ===
    # Dropout 比例
    params['NeuralNetwork_Settings']['drop_1'] = trial.suggest_float('dropout_1', 0.2, 0.6)
    params['NeuralNetwork_Settings']['drop_2'] = trial.suggest_float('dropout_2', 0.2, 0.6)
    
    # L2 正则化
    params['NeuralNetwork_Settings']['l2regularization'] = trial.suggest_float('l2_reg', 1e-6, 1e-2, log=True)

    # === 5. 训练参数 ===
    # 学习率
    params['NeuralNetwork_Settings']['learning_rate'] = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    # Batch Size
    params['NeuralNetwork_Settings']['batch_size'] = trial.suggest_categorical('batch_size', [64, 128, 256])

    return params


def run_training_task(trial, path_dict, params_dict):
    """
    执行单次训练任务
    """
    nn_mode = 'recurrent'
    
    try:
        # 1213: 将 trial 对象传递给训练函数，激活剪枝机制
        best_val_score = train_neuralnetwork.train_neuralnetwork(
            path_dict=path_dict,
            params_dict=params_dict,
            nn_mode=nn_mode,
            trial=trial  # <--- 关键修改：传入 trial
        )
        return best_val_score
    
    # 1213: 专门捕获剪枝异常，标记为 "PRUNED" 而不是 "FAIL"
    except optuna.exceptions.TrialPruned:
        logger.info(f"Trial {trial.number} pruned.")
        raise optuna.exceptions.TrialPruned()

    except Exception as e:
        # 捕获异常 (如 OOM, Loss NaN)
        logger.error(f"Trial {trial.number} failed with error: {e}")
        logger.error(traceback.format_exc())
        return float('inf')
    
    finally:
        # 每次Trial结束后必须清理显存
        K.clear_session()
        gc.collect()


def save_best_params(study, original_params):
    """
    将最优参数写回 toml 文件
    """
    logger.info("Saving best parameters to file...")
    
    best_config = copy.deepcopy(original_params)
    best_trial = study.best_trial.params
    
    # 映射 Optuna 参数回字典结构
    if 'input_timesteps' in best_trial:
        best_config['NeuralNetwork_Settings']['input_timesteps'] = best_trial['input_timesteps']
    
    if 'loss_weight_omega' in best_trial:
        best_config['NeuralNetwork_Settings']['loss_weight_omega'] = best_trial['loss_weight_omega']

    if 'n_layer_1' in best_trial:
        best_config['NeuralNetwork_Settings']['Recurrent']['neurons_first_layer_recurrent'] = best_trial['n_layer_1']
    if 'n_layer_2' in best_trial:
        best_config['NeuralNetwork_Settings']['Recurrent']['neurons_second_layer_recurrent'] = best_trial['n_layer_2']
        
    if 'dropout_1' in best_trial:
        best_config['NeuralNetwork_Settings']['drop_1'] = best_trial['dropout_1']
    if 'dropout_2' in best_trial:
        best_config['NeuralNetwork_Settings']['drop_2'] = best_trial['dropout_2']
    if 'l2_reg' in best_trial:
        best_config['NeuralNetwork_Settings']['l2regularization'] = best_trial['l2_reg']
        
    if 'learning_rate' in best_trial:
        best_config['NeuralNetwork_Settings']['learning_rate'] = best_trial['learning_rate']
    if 'batch_size' in best_trial:
        best_config['NeuralNetwork_Settings']['batch_size'] = best_trial['batch_size']

    # 写入文件
    with open(BEST_PARAMS_OUTPUT_PATH, 'w') as f:
        toml.dump(best_config, f)
    
    logger.info(f"Best parameters saved to: {os.path.abspath(BEST_PARAMS_OUTPUT_PATH)}")


# ==============================================================================
# 3. 主逻辑区 (MAIN)
# ==============================================================================

class Objective:
    def __init__(self, path_dict, base_params):
        self.path_dict = path_dict
        self.base_params = base_params

    def __call__(self, trial):
        # 每次实验深拷贝配置
        params_current = copy.deepcopy(self.base_params)
        # 采样参数
        params_current = suggest_hyperparameters(trial, params_current)
        # 运行训练
        score = run_training_task(trial, self.path_dict, params_current)
        return score


if __name__ == "__main__":
    print("========================================================")
    print("   Vehicle Dynamics Neural Network - Auto Tuner (Optuna)")
    print("========================================================")

    # 1. 准备数据
    logger.info("Loading base configuration...")
    # [修正] 调用正确的加载函数
    path_dict, base_params = load_base_config()

    pruner_config = optuna.pruners.MedianPruner(
        n_startup_trials=1,    # n_startup_trials=10: 前1次实验完全不剪枝，先收集数据建立基准（防止初期误判）
        n_warmup_steps=40,      # n_warmup_steps=50:  每个实验的前50个Epoch绝不剪枝（防止大器晚成被误杀）
        interval_steps=1            # interval_steps=1:   过了50轮后，每个Epoch都检查一次
    )

    # 2. 创建/加载 Study
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=STORAGE_DB,
        load_if_exists=True,
        direction='minimize',
        pruner=pruner_config  # <--- 关键修改：启用剪枝器
    )

    # ==============================================================================
    # [新增] 热启动：将 parameters.toml 中的参数作为第一个 Trial 运行
    # 这样可以建立一个强Baseline，避免一开始随机搜索浪费时间
    # ==============================================================================
    initial_params = {
        # 数据参数
        "input_timesteps": base_params['NeuralNetwork_Settings']['input_timesteps'],
        
        # 物理权重 (如果toml里没有这个键，手动指定你觉得好的值，比如 10.0)
        "loss_weight_omega": base_params['NeuralNetwork_Settings'].get('loss_weight_omega', 10.0),
        
        # 网络结构
        "n_layer_1": base_params['NeuralNetwork_Settings']['Recurrent']['neurons_first_layer_recurrent'],
        "n_layer_2": base_params['NeuralNetwork_Settings']['Recurrent']['neurons_second_layer_recurrent'],
        
        # 正则化
        "dropout_1": base_params['NeuralNetwork_Settings']['drop_1'],
        "dropout_2": base_params['NeuralNetwork_Settings']['drop_2'],
        "l2_reg": base_params['NeuralNetwork_Settings']['l2regularization'],
        
        # 训练参数
        "learning_rate": base_params['NeuralNetwork_Settings']['learning_rate'],
        "batch_size": base_params['NeuralNetwork_Settings']['batch_size']
    }
    # 将这组参数加入队列
    study.enqueue_trial(initial_params)

    # 3. 开始优化
    logger.info(f"Starting optimization for {N_TRIALS} trials...")
    
    objective_func = Objective(path_dict, base_params)
    
    try:
        study.optimize(
            objective_func, 
            n_trials=N_TRIALS, 
            n_jobs=N_JOBS, 
            timeout=TIMEOUT
        )
    except KeyboardInterrupt:
        logger.warning("Optimization interrupted by user!")

    # 4. 输出结果
    print("\n========================================================")
    print("Optimization Finished!")
    if len(study.trials) > 0:
        print(f"Number of finished trials: {len(study.trials)}")
        print(f"Best trial value (Val MSE): {study.best_value}")
        print("Best parameters:")
        for key, value in study.best_params.items():
            print(f"    {key}: {value}")
        
        # 5. 保存最优参数
        save_best_params(study, base_params)
    else:
        print("No trials completed.")
    print("========================================================")