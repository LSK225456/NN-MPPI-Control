import rosbag
import pandas as pd
import numpy as np
import os
import glob
from nav_msgs.msg import Odometry
from scipy.interpolate import interp1d

# 配置区域：请在此处修改路径、话题和阈值参数

# 1. 路径设置
# 存放原始ROS Bag文件的文件夹路径
INPUT_BAG_DIR = '/home/lsk1804/lsk_graduate/lsk_ws/bag'    # 存放原始ROS Bag文件的文件夹路径
OUTPUT_CSV_DIR = '/home/lsk1804/lsk_graduate/lsk_ws/src/NeuralNetwork_for_VehicleDynamicsModeling/inputs/trainingdata'      # 输出CSV训练数据集的文件夹路径 (对应 TUMFTM 项目 inputs/trainingdata)

# 2. 话题设置
WHEEL_TOPIC = '/wheel_odom'     # 车辆实际速度来源话题 
CMD_TOPIC = '/cmd_vel'  # 车辆控制指令话题

# 3. 过滤阈值
STATIC_VEL_THRESHOLD = 0.05     # 静态数据滤除阈值 (m/s)，绝对速度小于此值的数据将被丢弃
MAX_DT_THRESHOLD = 0.3              # 最大时间步长阈值 (s)，用于剔除丢包或停顿产生的异常dt

# 4. 数据结构设置
INPUT_SHAPE = 10         # 目标CSV的总列数 (注: 这里的Input指的是物理输入维度，实际CSV列数会变多)

# 1218：平滑方案EMA - 平滑算法配置
USE_SMOOTHING = True     # 是否启用平滑算法 (True=启用, False=不平滑，直接使用原始数据)
SMOOTHING_METHOD = 'ROLLING' # 平滑方法: 'EMA'=指数移动平均(推荐), 'ROLLING'=因果滚动平均
EMA_ALPHA = 0.2          # EMA 平滑系数 (0.1~0.3)，越大越接近原始数据，越小越平滑

# 输出文件的列名列表 
CSV_HEADER = [
    '# diff_v',     # Label: 速度残差 (v_t - v_{t-1}) # 残差网络
    'diff_w',     #  Label: 角速度残差 (w_t - w_{t-1}) # 残差网络
    '# v_mps',      # Input: 纵向速度
    'w_radps',    # Input: 角速度
    'dt_s',       # Input: 时间步长
    'cmd_v',      # Input: 指令线速度
    'cmd_w',      # Input: 指令角速度
    'a_v',        # 1212：特征工程 - Input: 纵向加速度 (v_t - v_{t-1})/dt
    'a_w',        # 1212：特征工程 - Input: 角加速度 (w_t - w_{t-1})/dt
    'err_v',      # 1212：特征工程 - Input: 速度误差 (cmd_v - v)
    'err_w',      # 1212：特征工程 - Input: 角速度误差 (cmd_w - w)
    'v_x_w',      # 1212：特征工程 - Input: 侧向动力学耦合项 (v * w)
]

TARGET_DT = 0.05         # 目标固定时间步长 (s), 建议设为 0.05 (20Hz) 或 0.1 (10Hz)
SMOOTH_WINDOW = 8       # 平滑滤波窗口大小，用于因果滚动平均 (仅当 SMOOTHING_METHOD='ROLLING' 时有效)


# 1218：平滑方案EMA - 指数移动平均函数实现
def apply_ema_smoothing(series, alpha=0.2):
    """
    应用指数移动平均 (EMA) 平滑，因果滤波器（只使用历史数据）
    
    公式：v_smooth[t] = alpha * v_raw[t] + (1-alpha) * v_smooth[t-1]
    
    :param series: pandas Series 待平滑的数据序列
    :param alpha: float 平滑系数 (0~1)，越大越接近原始数据
    :return: numpy array 平滑后的数据
    """
    result = np.zeros(len(series))
    result[0] = series.iloc[0]  # 初始值：使用第一帧原始值
    
    for i in range(1, len(series)):
        result[i] = alpha * series.iloc[i] + (1.0 - alpha) * result[i-1]
    
    return result


def read_bag_to_dataframe(bag_path, wheel_topic, cmd_topic):
    """
    读取单个Bag文件，提取里程计和控制指令数据，并返回原始DataFrame。

    :param bag_path: Bag文件的完整路径 (str)
    :param wheel_topic: 里程计话题名称 (str)
    :param cmd_topic: 控制指令话题名称 (str)
    :return: 包含 odom 和 cmd 数据的两个 DataFrame (tuple)
    """
    odom_list = []
    cmd_list = []

    try:
        with rosbag.Bag(bag_path, 'r') as bag:
            # 读取车辆状态数据 (Odometry)
            # topic: 话题名, msg: 消息体, t: 记录时间
            for topic, msg, t in bag.read_messages(topics=[wheel_topic]):
               # 检查 msg 是否有 header (Odometry 肯定有，TwistStamped 有，但 Twist 没有)
                if hasattr(msg, 'header'):
                    timestamp = msg.header.stamp.to_sec()
                else:
                    timestamp = t.to_sec() # 如果没有 header 才用录制时间(如纯Twist)

                odom_list.append({
                    'timestamp': timestamp, 
                    'v': msg.twist.twist.linear.x,
                    'w': msg.twist.twist.angular.z
                })
            
            # 2. 读取控制指令数据 (Twist) -> 只能使用 录制时间 t
            for topic, msg, t in bag.read_messages(topics=[cmd_topic]):
                # Twist 消息没有 header，直接使用 bag 录制时间 t
                timestamp = t.to_sec()
                
                cmd_list.append({
                    'timestamp': timestamp,  # [兼容] 使用录制时间
                    'cmd_v': msg.linear.x,
                    'cmd_w': msg.angular.z
                })


    except Exception as e:
        print(f"[Error] 读取Bag文件失败: {bag_path}, 错误信息: {e}")
        return pd.DataFrame(), pd.DataFrame()

    df_odom = pd.DataFrame(odom_list)
    df_cmd = pd.DataFrame(cmd_list)
    
    return df_odom, df_cmd

def sync_and_process_data(df_odom, df_cmd):
    """
    对 odom 和 cmd 数据进行时间同步、静态滤除、dt计算及填充处理。

    :param df_odom: 原始里程计数据 (pd.DataFrame)
    :param df_cmd: 原始控制指令数据 (pd.DataFrame)
    :return: 处理完成可直接保存的 DataFrame (pd.DataFrame)

    # 1203：数据清洗 - 增加重采样和平滑滤波逻
    """
    if df_odom.empty or df_cmd.empty:
        return pd.DataFrame()

    # 1. 预处理：排序与去重
    df_odom = df_odom.sort_values('timestamp').drop_duplicates(subset=['timestamp'], keep='first')
    df_cmd = df_cmd.sort_values('timestamp').drop_duplicates(subset=['timestamp'], keep='first') # 1203：数据清洗 - cmd也去重更安全

    # 2. 1218：平滑方案EMA - 原始数据平滑滤波 (仅针对 Odom 的物理状态 v, w)
    if USE_SMOOTHING:  # 1218：平滑方案EMA - 根据配置决定是否平滑
        if SMOOTHING_METHOD == 'EMA':
            # 1218：平滑方案EMA - 使用指数移动平均（因果滤波器，无相位滞后）
            df_odom['v'] = apply_ema_smoothing(df_odom['v'], alpha=EMA_ALPHA)
            df_odom['w'] = apply_ema_smoothing(df_odom['w'], alpha=EMA_ALPHA)
        elif SMOOTHING_METHOD == 'ROLLING':
            # 1218：平滑方案EMA - 使用因果滚动平均（center=False，只用历史数据）
            df_odom['v'] = df_odom['v'].rolling(window=SMOOTH_WINDOW, center=True, min_periods=1).mean()
            df_odom['w'] = df_odom['w'].rolling(window=SMOOTH_WINDOW, center=True, min_periods=1).mean()
        else:
            print(f"[Warning] Unknown SMOOTHING_METHOD: {SMOOTHING_METHOD}, skipping smoothing.")
    # 1218：平滑方案EMA - 如果 USE_SMOOTHING=False，则不做任何平滑处理，直接使用原始数据

    # 3. 1203：数据清洗 - 构建固定频率的时间网格 (Resampling Grid)
    # 取 odom 和 cmd 时间的交集范围，确保插值有效
    t_start = max(df_odom['timestamp'].min(), df_cmd['timestamp'].min())
    t_end = min(df_odom['timestamp'].max(), df_cmd['timestamp'].max())
    
    # 生成固定的时间序列
    t_grid = np.arange(t_start, t_end, TARGET_DT)

    if len(t_grid) < 2:
        return pd.DataFrame()
    
    # 4. 1203：数据清洗 - 对 Odom 进行线性插值 (物理量是连续的)
    # interp1d 创建插值函数
    f_v = interp1d(df_odom['timestamp'], df_odom['v'], kind='linear', fill_value="extrapolate")
    f_w = interp1d(df_odom['timestamp'], df_odom['w'], kind='linear', fill_value="extrapolate")
    
    # 5. 1203：数据清洗 - 创建同步后的 DataFrame
    df_sync = pd.DataFrame()
    df_sync['timestamp'] = t_grid
    df_sync['v'] = f_v(t_grid)
    df_sync['w'] = f_w(t_grid)
    
    # 6. 1203：数据清洗 - 对 Cmd 进行匹配 (使用 merge_asof 实现零阶保持 Zero-Order Hold)
    # 因为控制指令是离散下发的，不应该插值，而应该取“最近的一个历史指令”
    df_sync = pd.merge_asof(
        df_sync, 
        df_cmd, 
        on='timestamp', 
        direction='backward',
        tolerance=MAX_DT_THRESHOLD # 使用最大阈值防止匹配太久以前的指令
    )

    # 剔除匹配不到cmd的行
    df_sync.dropna(subset=['cmd_v', 'cmd_w'], inplace=True)

    # 7. 静态数据滤除 (使用重采样后的数据)
    df_filtered = df_sync[abs(df_sync['v']) > STATIC_VEL_THRESHOLD].copy()
    
    if len(df_filtered) < 2:
        return pd.DataFrame()
    
    # 8. 1203：数据清洗 - dt 现在是强制固定的常数
    # 直接赋值，不再通过 np.diff 计算，彻底消除 dt 抖动和第一帧异常
    df_filtered['dt'] = TARGET_DT 

    # 9. 过滤逻辑 (仅保留静态过滤，dt过滤已不再需要因为是固定的)
    # mask_min_dt / mask_max_dt 逻辑已由重采样过程隐式保证
    df_final = df_filtered.copy()

    df_final['diff_v'] = df_final['v'].diff() # 残差网络
    df_final['diff_w'] = df_final['w'].diff() # 残差网络
    # 由于 diff() 第一行会产生 NaN，必须剔除
    df_final.dropna(subset=['diff_v', 'diff_w'], inplace=True)

    # 1212：特征工程 - 计算新增物理特征
    # 1. 历史加速度 a = (v_t - v_{t-1}) / dt
    # 注意：这里的 diff_v 已经是 v_t - v_{t-1}，且 dt 已经是固定的 TARGET_DT
    df_final['a_v'] = df_final['diff_v'] / df_final['dt']
    df_final['a_w'] = df_final['diff_w'] / df_final['dt']

    # 2. 控制误差 err = cmd - state
    df_final['err_v'] = df_final['cmd_v'] - df_final['v']
    df_final['err_w'] = df_final['cmd_w'] - df_final['w']

    # 3. 动力学耦合项 (侧向加速度近似，解决转弯预测不准)
    df_final['v_x_w'] = df_final['v'] * df_final['w']


    output_data = pd.DataFrame()
   # --- Label 部分 (前2列) ---
    output_data['diff_v'] = df_final['diff_v'] # 残差网络
    output_data['diff_w'] = df_final['diff_w'] # 残差网络
    
    output_data['v_mps'] = df_final['v']
    output_data['w_radps'] = df_final['w']
    output_data['dt_s'] = df_final['dt']
    output_data['cmd_v'] = df_final['cmd_v']
    output_data['cmd_w'] = df_final['cmd_w']

    # 1212：特征工程 - 保存新增特征列
    output_data['a_v'] = df_final['a_v']
    output_data['a_w'] = df_final['a_w']
    output_data['err_v'] = df_final['err_v']
    output_data['err_w'] = df_final['err_w']
    output_data['v_x_w'] = df_final['v_x_w']

    return output_data

# ==============================================================================
# 主函数
# ==============================================================================

def main():
    """程序入口：遍历文件夹，批处理Bag文件并保存CSV。"""
    
    if not os.path.exists(OUTPUT_CSV_DIR):
        os.makedirs(OUTPUT_CSV_DIR)
        print(f"[Info] 创建输出目录: {OUTPUT_CSV_DIR}")

    # 获取所有 .bag 文件路径
    bag_files = glob.glob(os.path.join(INPUT_BAG_DIR, "*.bag"))
    print(f"[Info] 发现 {len(bag_files)} 个Bag文件，开始处理...")

    for i, bag_file in enumerate(bag_files):
        print(f"-> 正在处理 ({i+1}/{len(bag_files)}): {os.path.basename(bag_file)}")
        
        # 1. 读取数据
        df_odom, df_cmd = read_bag_to_dataframe(bag_file, WHEEL_TOPIC, CMD_TOPIC)
        
        # 2. 处理数据
        df_result = sync_and_process_data(df_odom, df_cmd)
        
        if df_result.empty:
            print(f"   [Warn] 文件 {os.path.basename(bag_file)} 处理后数据为空，已跳过。")
            continue

        # 3. 保存 CSV
        # 文件名格式: data_to_train_X.csv (符合开源项目要求)
        save_name = f"data_to_train_{i}.csv"
        save_path = os.path.join(OUTPUT_CSV_DIR, save_name)
        
        # header=True (TUMFTM项目读取时通常跳过首行或按列索引读取，带header更安全)
        # index=False (不保存索引列)
        df_result.to_csv(save_path, header=CSV_HEADER, index=False)
        print(f"   [Success] 已保存: {save_path} (行数: {len(df_result)})")

    print("\n[Done] 所有文件处理完毕。")

if __name__ == "__main__":
    main()
# //12.02（当天日期）