#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# // 2025.11.23 Created by Gemini for MPPI Visualization

import rospy
import torch
import numpy as np
from geometry_msgs.msg import PoseStamped, Point
from nav_msgs.msg import Path
from std_msgs.msg import Header, ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

class MPPIVisualizer:
    """
    MPPI 可视化处理器
    功能：将 MPPI 算法产生的状态张量转化为 ROS 可视化消息 (Rviz)
    设计原则：与控制逻辑解耦，仅负责单向数据流的可视化渲染
    """
    def __init__(self, frame_id="map"):
        """
        初始化可视化发布器
        输入参数：
            frame_id (str): 可视化消息的参考坐标系，通常为 'odom' 或 'map'
        """
        self.frame_id = frame_id
        
        # 核心层：最优轨迹与候选轨迹簇发布器
        self.pub_optimal_path = rospy.Publisher('/mppi_viz/optimal_path', Path, queue_size=1)
        self.pub_candidate_paths = rospy.Publisher('/mppi_viz/candidate_paths', MarkerArray, queue_size=1)
        
        # 逻辑层：局部目标点发布器
        self.pub_local_target = rospy.Publisher('/mppi_viz/local_target', Marker, queue_size=1)

    def visualize_optimal_trajectory(self, states):
        """
        发布 MPPI 预测的最优轨迹 (Nominal Trajectory)
        输入参数：
            states (Tensor): 形状为 [1, T, nx] 或 [T, nx] 的状态序列，包含 [x, y, theta]
        """
        if states is None:
            return

        # 数据降维与设备转移：Tensor(GPU) -> Numpy(CPU)
        states_np = states.detach().cpu().numpy()
        if states_np.ndim == 3:
            states_np = states_np[0]  # 去除 Batch 维度 -> [T, nx]

        path_msg = Path()
        path_msg.header.frame_id = self.frame_id
        path_msg.header.stamp = rospy.Time.now()

        # 构建路径点
        for i in range(states_np.shape[0]):
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = states_np[i, 0]
            pose.pose.position.y = states_np[i, 1]
            # 姿态四元数转换可选，此处仅显示位置轨迹
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)

        self.pub_optimal_path.publish(path_msg)

    def visualize_candidate_trajectories(self, states, costs, num_vis=20):
        """
        发布采样轨迹簇 (Top-K Candidates)
        输入参数：
            states (Tensor): 所有采样轨迹的状态张量，形状 [K, T, nx]
            costs (Tensor): 对应的轨迹代价张量，形状 [K]
            num_vis (int): 仅显示代价最低的前 K 条轨迹，避免 Rviz 过载
        """
        if states is None or costs is None:
            return
        

        # 1123可视化修改
        # 去除所有大小为1的维度，直到得到3维张量 [K, T, nx]
        original_shape = states.shape
        while len(states.shape) > 3 and states.shape[0] == 1:
            states = states.squeeze(0)
        
        # 如果仍然不是3维，尝试其他方法
        if len(states.shape) != 3:
            rospy.logwarn(f"Unexpected states shape: {original_shape}, attempting to reshape...")
            # 尝试找到 K, T, nx 维度
            if len(states.shape) == 4:
                # 假设形状为 [1, K, T, nx]
                states = states[0]
            elif len(states.shape) == 5:
                # 假设形状为 [1, 1, K, T, nx]  
                states = states[0, 0]
            else:
                rospy.logerr(f"Cannot handle states with shape: {original_shape}")
                return
        # 1123可视化修改
            
            
        K, T, _ = states.shape
        num_vis = min(K, num_vis)

        # 筛选 Top-K 最优轨迹索引
        # largest=False 表示选取代价最小的（最优的）
        topk_values, topk_indices = torch.topk(costs, k=num_vis, largest=False)
        
        indices_np = topk_indices.detach().cpu().numpy()
        states_np = states.detach().cpu().numpy()

        marker_array = MarkerArray()
        timestamp = rospy.Time.now()
        
        # 归一化代价用于颜色映射 (仅在 Top-K 内部归一化)
        min_c = topk_values.min().item()
        max_c = topk_values.max().item()
        cost_range = max_c - min_c + 1e-6

        for rank, idx in enumerate(indices_np):
            traj = states_np[idx] # [T, nx]
            
            marker = Marker()
            marker.header.frame_id = self.frame_id
            marker.header.stamp = timestamp
            marker.ns = "candidate_paths"
            marker.id = rank
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.scale.x = 0.02  # 线宽

            # 颜色映射：代价越低越蓝，代价越高透明度越低
            score = (topk_values[rank].item() - min_c) / cost_range
            marker.color.r = 0.0
            marker.color.g = 1.0 - score * 0.5 # 浅蓝偏青
            marker.color.b = 1.0
            marker.color.a = 0.6 - score * 0.4 # 越优秀的轨迹越不透明

            # 填充轨迹点
            for t in range(T):
                p = Point()
                p.x = traj[t, 0]
                p.y = traj[t, 1]
                p.z = 0.0
                marker.points.append(p)
            
            marker_array.markers.append(marker)

        self.pub_candidate_paths.publish(marker_array)

    def visualize_local_target(self, target_np):
        """
        发布局部目标点 (Local Target / Carrot)
        输入参数：
            target_np (np.array): 形状为 [2] 或 [3] 的坐标数组 [x, y, ...]
        """
        if target_np is None:
            return

        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = rospy.Time.now()
        marker.ns = "local_target"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        
        # 尺寸设置
        marker.scale.x = 0.3
        marker.scale.y = 0.3
        marker.scale.z = 0.3

        # 颜色设置：红色，高亮
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 0.9

        marker.pose.position.x = float(target_np[0])
        marker.pose.position.y = float(target_np[1])
        marker.pose.position.z = 0.1 # 略微抬起，避免被地图遮挡
        marker.pose.orientation.w = 1.0

        self.pub_local_target.publish(marker)