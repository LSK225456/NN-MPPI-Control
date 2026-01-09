#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rosbag
import numpy as np
import math
import sys
import os

# ==========================================
# 1. 用户配置区域 (请在此处修改)
# ==========================================

# 录制的 bag 文件路径 (请修改为你实际的路径)
BAG_FILE_PATH = '/home/pc/lsk/data/bag/2025-12-17-12-49-08.bag'

# 定位话题名称 (请修改为你实际使用的定位话题，如 /odom, /ndt_pose, /ground_truth/state)
POSE_TOPIC = '/odom'

# 采样设置
# 为了避免偶然跳变，我们会取起点的前N帧和终点的后N帧取平均值
SAMPLE_SIZE = 10 

# ==========================================
# 2. 核心计算逻辑
# ==========================================

def calculate_yaw_bias():
    print("========== 开始分析数据 ==========")
    print("读取 Bag 文件: {}".format(BAG_FILE_PATH))
    
    # 检查文件是否存在
    if not os.path.exists(BAG_FILE_PATH):
        print("错误: 找不到文件 {}".format(BAG_FILE_PATH))
        return

    points_x = []
    points_y = []

    try:
        bag = rosbag.Bag(BAG_FILE_PATH)
        
        # 读取消息
        for topic, msg, t in bag.read_messages(topics=[POSE_TOPIC]):
            # 兼容 Odometry 和 PoseStamped 两种常见格式
            if hasattr(msg, 'pose') and hasattr(msg.pose, 'pose'):
                # Nav_msgs/Odometry
                px = msg.pose.pose.position.x
                py = msg.pose.pose.position.y
            elif hasattr(msg, 'pose') and hasattr(msg.pose, 'position'):
                # Geometry_msgs/PoseStamped
                px = msg.pose.position.x
                py = msg.pose.position.y
            else:
                continue
                
            points_x.append(px)
            points_y.append(py)
            
        bag.close()
    except Exception as e:
        print("读取 Bag 失败: {}".format(e))
        return

    total_count = len(points_x)
    print("共读取到 {} 帧定位数据".format(total_count))

    if total_count < SAMPLE_SIZE * 2:
        print("数据量太少，无法计算。请录制更长的距离。")
        return

    # --- 关键步骤：提取起点和终点 ---
    # 我们不信任单帧数据，取前10帧和后10帧的平均值作为稳健的起点和终点
    
    # 1. 计算起点 (Start Point) - 取前 SAMPLE_SIZE 帧平均
    start_x = np.mean(points_x[:SAMPLE_SIZE])
    start_y = np.mean(points_y[:SAMPLE_SIZE])
    
    # 2. 计算终点 (End Point) - 取后 SAMPLE_SIZE 帧平均
    end_x = np.mean(points_x[-SAMPLE_SIZE:])
    end_y = np.mean(points_y[-SAMPLE_SIZE:])

    print("\n[轨迹数据]")
    print("起点坐标 (平均): x={:.4f}, y={:.4f}".format(start_x, start_y))
    print("终点坐标 (平均): x={:.4f}, y={:.4f}".format(end_x, end_y))

    # --- 核心计算 ---
    # 计算位移矢量 (Delta)
    dx = end_x - start_x
    dy = end_y - start_y
    distance = math.sqrt(dx**2 + dy**2)

    print("行驶总距离: {:.4f} 米".format(distance))

    if distance < 1.0:
        print("警告: 行驶距离过短 (<1米)，计算结果可能不准确！建议行驶 5-10 米以上。")

    # 计算角度偏差 (Yaw Bias)
    # math.atan2 返回的是弧度，范围 -pi 到 +pi
    # 这是软件轨迹相对于 X 轴的倾斜角
    yaw_bias_rad = math.atan2(dy, dx)
    yaw_bias_deg = math.degrees(yaw_bias_rad)

    print("\n========== 标定结果 ==========")
    print("计算出的 Baselink 角度偏差:")
    print("弧度 (rad): {:.6f}".format(yaw_bias_rad))
    print("角度 (deg): {:.4f}°".format(yaw_bias_deg))
    
    print("\n[结果分析]")
    if abs(yaw_bias_deg) < 0.5:
        print("偏差非常小 (<0.5°)，可能不需要修正，或者是偶然误差。")
    else:
        if yaw_bias_deg > 0:
            direction_str = "偏左 (逆时针)"
            action_str = "减去 (-)"
        else:
            direction_str = "偏右 (顺时针)"
            action_str = "加上 (+)" # 负负得正，实际上是数值增大

        print("轨迹倾向: {}".format(direction_str))
        print("诊断: 你的 Base_link 定义相对于物理车头 {} 歪了 {:.4f}°".format(direction_str, abs(yaw_bias_deg)))
        
        print("\n========== 修正建议 ==========")
        print("请在你的 TF 发布节点 (static_transform_publisher) 中修改 Yaw 参数。")
        print("如果你是 laser2base，修正公式为：")
        print("  New_Yaw = Old_Yaw - ({:.6f})".format(yaw_bias_rad))
        print("\n  如果当前 Yaw 是 0，你应该设置为: {:.6f}".format(-yaw_bias_rad))

if __name__ == '__main__':
    calculate_yaw_bias()