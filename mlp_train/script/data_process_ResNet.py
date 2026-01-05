#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rosbag
import numpy as np
import math
import argparse
import sys
import os
import rospy
import torch  

class ConfigManager:
    """
    加载配置
    """
    def __init__(self):
        # // 11.25: 初始化 ROS 节点以获取参数
        rospy.init_node('dataset_builder_node', anonymous=True)
        
        # 获取私有命名空间参数 (~param_name)
        # 格式：rospy.get_param("~param_name", default_value)
        self.topics = {
            'odom':  rospy.get_param("~topics/odom", "/odom"),
            'wheel': rospy.get_param("~topics/wheel", "/wheel_odom"),
            'cmd':   rospy.get_param("~topics/cmd", "/cmd_vel")
        }

        self.thresholds = {
            'filter_stationary': rospy.get_param("~thresholds/filter_stationary", True),
            'min_lin_vel': rospy.get_param("~thresholds/min_lin_vel", 0.01),
            'min_ang_vel': rospy.get_param("~thresholds/min_ang_vel", 0.01)
        }

        self.model = {
            'dt_tolerance': rospy.get_param("~model/dt_tolerance", 0.3)
        }

        rospy.loginfo(f"配置加载完成: 话题Odom={self.topics['odom']}, 丢帧阈值={self.model['dt_tolerance']}")



class MathUtils:
    """
    数学工具类
    """
    @staticmethod
    def quaternion_to_yaw(orientation):
        """
        功能：四元数转 Yaw
        输入：geometry_msgs/Quaternion
        输出：float (rad)
        """
        x, y, z, w = orientation.x, orientation.y, orientation.z, orientation.w
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)

    @staticmethod
    def normalize_angle(angle):
        """
        功能：角度归一化至 [-pi, pi]
        """
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    @staticmethod
    def global_to_body(dx_global, dy_global, yaw_body):
        """
        功能：全局误差投影至车身系
        输入：dx_global, dy_global, yaw_body
        输出：d_long, d_lat
        """
        cos_theta = math.cos(yaw_body)
        sin_theta = math.sin(yaw_body)
        d_long = cos_theta * dx_global + sin_theta * dy_global
        d_lat  = -sin_theta * dx_global + cos_theta * dy_global
        return d_long, d_lat



class KinematicsEngine:
    """
    运动学核心引擎
    """
    @staticmethod
    def predict_next_state(curr_state, cmd_vel, dt):
        """
        功能：Unicycle 模型推演
        输入：curr_state [x,y,th,v,w], cmd_vel [v,w], dt
        输出：[x_kin, y_kin, th_kin]
        """
        x, y, theta, _, _ = curr_state
        v_cmd, w_cmd = cmd_vel

        theta_next = theta + w_cmd * dt
        x_next = x + v_cmd * math.cos(theta) * dt
        y_next = y + v_cmd * math.sin(theta) * dt
        return [x_next, y_next, theta_next]

    @staticmethod
    def compute_residuals(curr_state, next_state_true, next_state_kin):
        """
        功能：计算车身系下的物理残差
        输入：当前状态、真实下一刻状态、运动学预测状态
        输出：[err_long, err_lat, err_theta, err_v, err_w]
        """
        current_yaw = curr_state[2]
        
        # 解包
        x_true, y_true, theta_true = next_state_true[0:3]
        v_true, w_true = next_state_true[3:5]
        x_kin, y_kin, theta_kin = next_state_kin[0:3]

        # 位置误差 (全局)
        dx_global = x_true - x_kin
        dy_global = y_true - y_kin

        # 投影至车身
        d_long, d_lat = MathUtils.global_to_body(dx_global, dy_global, current_yaw)

        # 状态误差
        d_theta = MathUtils.normalize_angle(theta_true - theta_kin)
        d_v = v_true - curr_state[3] # 记录速度变化量或误差，视具体需求
        d_w = w_true - curr_state[4]

        return [d_long, d_lat, d_theta, d_v, d_w]

class DatasetBuilder:
    """
    数据集构建器
    """
    def __init__(self, cfg_manager):
        self.cfg = cfg_manager
        self.inputs = []
        self.outputs = []
        
        # 零阶保持缓存
        self.last_vel = np.zeros(2)
        self.last_cmd = np.zeros(2)
        self.prev_sample = None 

    def process_bag(self, bag_path):
        print(f"[*] 开始处理 Bag: {bag_path}")
        try:
            bag = rosbag.Bag(bag_path)
        except Exception as e:
            rospy.logerr(f"无法打开文件: {e}")
            return False

        # // 11.25: 使用 Config 类中加载的参数
        topics = [self.cfg.topics['odom'], self.cfg.topics['wheel'], self.cfg.topics['cmd']]
        valid_count = 0

        for topic, msg, t in bag.read_messages(topics=topics):
            timestamp = t.to_sec()

            if topic == self.cfg.topics['wheel']:
                self.last_vel[0] = msg.twist.twist.linear.x
                self.last_vel[1] = msg.twist.twist.angular.z

            elif topic == self.cfg.topics['cmd']:
                self.last_cmd[0] = msg.linear.x
                self.last_cmd[1] = msg.angular.z

            elif topic == self.cfg.topics['odom']:
                # 解析当前状态
                curr_vec = [
                    msg.pose.pose.position.x,
                    msg.pose.pose.position.y,
                    MathUtils.quaternion_to_yaw(msg.pose.pose.orientation),
                    self.last_vel[0],
                    self.last_vel[1]
                ]
                
                if self.prev_sample is not None:
                    dt = timestamp - self.prev_sample['time']
                    
                    # 检查 dt 有效性
                    if 0.0 < dt <= self.cfg.model['dt_tolerance']:
                        prev_vec = self.prev_sample['state']
                        cmd_vec = self.prev_sample['cmd']

                        # 静止剔除检测
                        is_still = (abs(prev_vec[3]) < self.cfg.thresholds['min_lin_vel'] and 
                                    abs(prev_vec[4]) < self.cfg.thresholds['min_ang_vel'])
                        
                        if not (self.cfg.thresholds['filter_stationary'] and is_still):
                            # 构建 Input 特征
                            feature_row = [prev_vec[3], prev_vec[4], cmd_vec[0], cmd_vec[1], dt]

                            # 运动学推演
                            kin_next = KinematicsEngine.predict_next_state(prev_vec, cmd_vec, dt)
                            
                            # 残差计算
                            label_row = KinematicsEngine.compute_residuals(prev_vec, curr_vec, kin_next)

                            self.inputs.append(feature_row)
                            self.outputs.append(label_row)
                            valid_count += 1
                    else:
                        self.prev_sample = None
                        continue

                self.prev_sample = {'time': timestamp, 'state': curr_vec, 'cmd': self.last_cmd.copy()}

        bag.close()
        rospy.loginfo(f"处理完成，有效样本: {valid_count}")
        return True

    def save_pt(self, output_path):
        """
        // 11.25: 修正为保存 .pt 格式，使用 torch.save
        """
        if not self.inputs:
            rospy.logwarn("数据为空，未保存。")
            return

        # 转换为 Tensor
        x_tensor = torch.tensor(self.inputs, dtype=torch.float32)
        y_tensor = torch.tensor(self.outputs, dtype=torch.float32)

        # 构建字典
        data_dict = {
            'x': x_tensor,
            'y': y_tensor,
            'meta': {
                'columns_x': ['v', 'w', 'v_cmd', 'w_cmd', 'dt'],
                'columns_y': ['err_long', 'err_lat', 'err_th', 'err_v', 'err_w']
            }
        }

        torch.save(data_dict, output_path)
        rospy.loginfo(f"数据集已保存至: {output_path}")
        rospy.loginfo(f"Input Tensor: {x_tensor.shape}, Output Tensor: {y_tensor.shape}")

if __name__ == "__main__":
    # 这里的 argparse 仅处理文件路径，逻辑参数全部移交给 ROS Parameter Server
    parser = argparse.ArgumentParser()
    parser.add_argument('bag', help="Input rosbag path")
    parser.add_argument('output', help="Output .pt file path")
    
    # 处理 roslaunch 传入的额外参数干扰 (roslaunch 会传入 __name:=... 等参数)
    args, unknown = parser.parse_known_args()

    # 实例化配置 (会自动 init_node 并 load params)
    cfg = ConfigManager()
    
    builder = DatasetBuilder(cfg)
    if builder.process_bag(args.bag):
        builder.save_pt(args.output)