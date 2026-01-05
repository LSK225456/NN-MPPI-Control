#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rosbag
import numpy as np
import math
import argparse
import sys
import os
import glob
import rospy
import torch  
from collections import deque       # cmd修正对齐: 引入双端队列用于维护历史cmd


# cmd修正对齐: 可调参数 - 提前的帧数 (0=使用最新, 1=倒退1帧, 3=倒退3帧...)
CMD_LOOKBACK_FRAMES = 0

# 定义数据列名的常量，方便维护
COLUMNS_X = ['v_curr', 'w_curr', 'v_cmd', 'w_cmd', 'dt', 'acc_v', 'acc_w']
COLUMNS_Y = ['v_next', 'w_next']

class ConfigManager:
    """
    配置管理器：负责加载 ROS 参数服务器上的所有配置
    """
    def __init__(self):
        # 初始化节点以获取 launch 文件中的 param
        rospy.init_node('dataset_builder_node', anonymous=True)
        
        # 话题配置
        self.topics = {
            'odom':  rospy.get_param("~topics/odom", "/odom"),
            'wheel': rospy.get_param("~topics/wheel", "/wheel_odom"),
            'cmd':   rospy.get_param("~topics/cmd", "/cmd_vel")
        }

        # 阈值配置
        self.thresholds = {
            'filter_stationary': rospy.get_param("~thresholds/filter_stationary", True),
            'min_lin_vel': rospy.get_param("~thresholds/min_lin_vel", 0.01),
            'min_ang_vel': rospy.get_param("~thresholds/min_ang_vel", 0.01)
        }

        # 模型相关配置
        self.model = {
            'dt_tolerance': rospy.get_param("~model/dt_tolerance", 0.3)
        }

        # 批量处理模式的路径参数
        self.batch = {
            'input_dir': rospy.get_param("~batch/input_dir", ""),
            'output_dir': rospy.get_param("~batch/output_dir", "")
        }

        rospy.loginfo(f"配置加载完成")

class MathUtils:
    """
    数学工具类：提供静态数学辅助函数
    """
    @staticmethod
    def quaternion_to_yaw(orientation):
        """
        功能：四元数转 Yaw 角
        输入：geometry_msgs/Quaternion
        输出：float (rad)
        """
        x, y, z, w = orientation.x, orientation.y, orientation.z, orientation.w
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)

class DatasetBuilder:
    """
    数据集构建器：核心逻辑类，负责解析 Bag 并生成 Tensor 数据
    """
    def __init__(self, cfg_manager):
        self.cfg = cfg_manager
        self.reset_buffer()

    def reset_buffer(self):
        """
        功能：重置内部数据缓存，用于处理新的文件
        """
        self.inputs = []
        self.outputs = []
        self.last_vel = np.zeros(2) # [v, w] from wheel odom
        self.last_cmd = np.zeros(2) # [v, w] from cmd_vel
        self.prev_sample = None     # 上一帧的状态缓存

        # 初始化cmd历史队列，长度设为 (回退帧数 + 1) 以便能访问到倒数第N+1个元素
        self.cmd_buffer = deque(maxlen=CMD_LOOKBACK_FRAMES + 1)
        for _ in range(CMD_LOOKBACK_FRAMES + 1):
            self.cmd_buffer.append(np.zeros(2))


    def process_single_bag(self, bag_path):
        """
        功能：处理单个 Bag 文件
        输入：bag_path (str)
        输出：bool (是否成功)
        """
        print(f"[*] 正在处理 Bag: {bag_path}")
        self.reset_buffer() # 处理新文件前务必重置

        try:
            bag = rosbag.Bag(bag_path)
        except Exception as e:
            rospy.logerr(f"无法打开 Bag 文件: {e}")
            return False

        topics = [self.cfg.topics['odom'], self.cfg.topics['wheel'], self.cfg.topics['cmd']]
        valid_count = 0

        # 遍历消息，构建 (State_t, Action_t) -> State_{t+1} 对
        for topic, msg, t in bag.read_messages(topics=topics):
            timestamp = t.to_sec()

            if topic == self.cfg.topics['wheel']:
                self.last_vel[0] = msg.twist.twist.linear.x
                self.last_vel[1] = msg.twist.twist.angular.z

            elif topic == self.cfg.topics['cmd']:
                # self.last_cmd[0] = msg.linear.x
                # self.last_cmd[1] = msg.angular.z

                # cmd修正对齐: 获取当前指令并存入历史队列
                current_cmd = np.array([msg.linear.x, msg.angular.z])
                self.cmd_buffer.append(current_cmd)
                # 更新 last_cmd (为了兼容原有逻辑，虽然下面构建样本时主要用 buffer)
                self.last_cmd = current_cmd
                

            elif topic == self.cfg.topics['odom']:
                # 解析当前帧状态 (State_{t+1})
                # 注意：这里我们提取的是“真实观测到的速度”，作为 Label
                curr_v = self.last_vel[0]
                curr_w = self.last_vel[1]
                
                # // 12.02 (说明): curr_vec 包含 [x, y, theta, v, w]
                curr_vec = [
                    msg.pose.pose.position.x,
                    msg.pose.pose.position.y,
                    MathUtils.quaternion_to_yaw(msg.pose.pose.orientation),
                    curr_v,
                    curr_w
                ]

                # 初始化当前加速度，默认为0 (第一帧无法计算)
                curr_acc_v = 0.0
                curr_acc_w = 0.0
                
                # 如果有上一帧数据，则可以构建一个样本
                if self.prev_sample is not None:
                    dt = timestamp - self.prev_sample['time']
                    
                    # 检查 dt 有效性
                    if 0.0 < dt <= self.cfg.model['dt_tolerance']:
                        prev_vec = self.prev_sample['state']
                        cmd_vec = self.prev_sample['cmd']

                        # 12.02: 计算当前时刻相对于上一时刻的加速度 (用于存入下一次迭代的prev_sample)
                        # 注意：这里的加速度是 a_t = (v_t - v_{t-1}) / dt
                        curr_acc_v = (curr_v - prev_vec[3]) / dt
                        curr_acc_w = (curr_w - prev_vec[4]) / dt

                        # 静止剔除逻辑
                        is_still = (abs(prev_vec[3]) < self.cfg.thresholds['min_lin_vel'] and 
                                    abs(prev_vec[4]) < self.cfg.thresholds['min_ang_vel'])
                        
                        if not (self.cfg.thresholds['filter_stationary'] and is_still):
                            # 端到端输入构造
                            # Input (X): [v_curr, w_curr, v_cmd, w_cmd, dt, acc_v, acc_w]
                            # 这里的 prev_vec 是 t 时刻的状态
                            prev_acc = self.prev_sample.get('acc', [0.0, 0.0])
                            feature_row = [prev_vec[3], prev_vec[4], cmd_vec[0], cmd_vec[1], dt, prev_acc[0], prev_acc[1]]

                            #  端到端输出构造
                            # Output (Y): [v_next, w_next]
                            # 这里的 curr_vec 是 t+1 时刻的真实状态 (Ground Truth)
                            # 不再计算残差，直接学习 Next State
                            label_row = [curr_vec[3], curr_vec[4]]

                            self.inputs.append(feature_row)
                            self.outputs.append(label_row)
                            valid_count += 1
                    else:
                        # dt 超时，断开连续性，重置上一帧
                        self.prev_sample = None
                        continue

                # 更新上一帧缓存
                # 从历史队列中提取回退 N 帧后的指令
                # self.cmd_buffer[-1] 是最新指令，[- (1 + N)] 即为倒退 N 帧的指令
                delayed_cmd = self.cmd_buffer[-(1 + CMD_LOOKBACK_FRAMES)].copy()
                self.prev_sample = {
                    'time': timestamp, 
                    'state': curr_vec, 
                    'cmd': delayed_cmd,
                    'acc': [curr_acc_v, curr_acc_w] # 12.02: 保存加速度
                }

                # self.prev_sample = {'time': timestamp, 'state': curr_vec, 'cmd': self.last_cmd.copy()}

        bag.close()
        rospy.loginfo(f"-> 完成: {os.path.basename(bag_path)}, 提取样本数: {valid_count}")
        return True

    def save_to_pt(self, output_path):
        """
        功能：将内存中的数据保存为 .pt 文件
        """
        if not self.inputs:
            rospy.logwarn(f"数据为空，跳过保存: {output_path}")
            return

        # 转换为 Tensor
        x_tensor = torch.tensor(self.inputs, dtype=torch.float32)
        y_tensor = torch.tensor(self.outputs, dtype=torch.float32)

        #  更新 Meta 信息以匹配端到端模式
        data_dict = {
            'x': x_tensor,
            'y': y_tensor,
            'meta': {
                'columns_x': COLUMNS_X,
                'columns_y': COLUMNS_Y,
                'description': 'End-to-End Dynamics Data: State+Control -> Next_State'
            }
        }

        # 自动创建输出目录
        out_dir = os.path.dirname(output_path)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir)

        torch.save(data_dict, output_path)
        rospy.loginfo(f"保存成功: {output_path}")
        rospy.loginfo(f"Shape -> X: {x_tensor.shape}, Y: {y_tensor.shape}")



class BatchProcessor:
    """
    批量处理器：负责文件夹级别的调度与文件名映射管理
    """
    def __init__(self, builder):
        self.builder = builder

    def run_batch(self, input_dir, output_dir):
        """
        功能：执行批量处理逻辑，生成 experiment_x.pt 和 mapping.txt
        输入：input_dir (bag文件夹), output_dir (pt输出文件夹)
        """
        if not os.path.exists(input_dir):
            rospy.logerr(f"输入目录不存在: {input_dir}")
            return
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        #  使用 sorted 确保文件处理顺序确定
        bag_files = sorted(glob.glob(os.path.join(input_dir, "*.bag")))
        rospy.loginfo(f"找到 {len(bag_files)} 个 Bag 文件，准备开始处理...")

        mapping_records = [] # 用于存储 "文件名: 原bag名" 的映射关系

        for idx, bag_path in enumerate(bag_files):
            #  按序号生成标准化文件名 experiment_x.pt
            file_idx = idx + 1
            new_filename = f"experiment_{file_idx}.pt"
            output_pt_path = os.path.join(output_dir, new_filename)
            original_bag_name = os.path.basename(bag_path)

            rospy.loginfo(f"[{file_idx}/{len(bag_files)}] 处理: {original_bag_name} -> {new_filename}")

            # 调用 Builder 处理单个文件
            success = self.builder.process_single_bag(bag_path)
            
            if success:
                self.builder.save_to_pt(output_pt_path)
                # 记录成功处理的映射关系
                mapping_records.append(f"{new_filename}: {original_bag_name}")
            else:
                rospy.logwarn(f"跳过处理失败的文件: {original_bag_name}")

        # 将映射表写入 mapping.txt
        mapping_path = os.path.join(output_dir, "mapping.txt")
        try:
            with open(mapping_path, "w") as f:
                f.write("\n".join(mapping_records))
            rospy.loginfo(f"批量处理完成，映射表已保存至: {mapping_path}")
        except IOError as e:
            rospy.logerr(f"保存映射表失败: {e}")

def main():
    # // 12.02 (修改): 优化参数解析器以支持批量参数的显式传入
    parser = argparse.ArgumentParser(description="Vehicle Dynamics Dataset Builder")
    
    # // 12.02 (注释): 位置参数设置为可选，避免在批量模式下报错
    parser.add_argument('bag_path', nargs='?', help="Single input rosbag path")
    parser.add_argument('out_path', nargs='?', help="Single output .pt file path")
    
    # // 12.02 (修改): 添加批量处理的 Flag 参数
    parser.add_argument('--in_dir', type=str, default=None, help="Batch input directory")
    parser.add_argument('--out_dir', type=str, default=None, help="Batch output directory")

    # // 12.02 (注释): parse_known_args 用于过滤掉 ROS 自动注入的参数 (如 __name, __log)
    args, unknown = parser.parse_known_args()

    # 初始化配置
    cfg = ConfigManager()
    builder = DatasetBuilder(cfg)

    # // 12.02 (修改): 逻辑分流，优先检查批量参数是否被赋值
    # 判定逻辑：如果命令行传入了 in_dir 和 out_dir，则执行批量；否则尝试单文件
    if args.in_dir and args.out_dir:
        rospy.loginfo(f"检测到批量处理参数，模式: BATCH")
        rospy.loginfo(f"输入目录: {args.in_dir}")
        rospy.loginfo(f"输出目录: {args.out_dir}")
        
        batch_proc = BatchProcessor(builder)
        batch_proc.run_batch(args.in_dir, args.out_dir)
        
    elif args.bag_path and args.out_path:
        rospy.loginfo("检测到单文件参数，模式: SINGLE")
        if builder.process_single_bag(args.bag_path):
            builder.save_to_pt(args.out_path)
            
    else:
        # // 12.02 (修改): 均未提供时的错误提示
        rospy.logerr("参数缺失: 请通过 Launch 文件或命令行提供 (--in_dir, --out_dir) 或 (bag_path, out_path)")
        parser.print_help()




if __name__ == "__main__":
    main()