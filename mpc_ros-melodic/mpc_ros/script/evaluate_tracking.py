#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import tf
import math
import os
import time
import numpy as np
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped

# ==========================================
# 1. 用户参数配置区域 (Parameters Configuration)
# ==========================================

# --- 话题设置 ---
ODOM_TOPIC = "/odom_calib"                # 定位话题名称
GLOBAL_PATH_TOPIC = "/desired_path" # 全局参考路径话题 (请修改为你实际发送路径的话题名)

# --- 坐标系设置 ---
ODOM_FRAME_ID = "world"               # 全局坐标系名称
CAR_FRAME_ID = "base_link_calib"          # 车辆坐标系名称

# --- 静态坐标变换参数 (对应 C++ 代码) ---
ENABLE_STATIC_TRANSFORM = False      # 是否启用坐标变换 (laser -> base)
LASER2BASE_DIST = 0.38              # laser_link 在 base_link 前方的距离 (米)
YAW_OFFSET_DEG = 0                # 偏航角补偿 (度)

# --- 输出设置 ---
# 结果输出文件夹路径 (会自动创建)
OUTPUT_FOLDER_PATH = "/home/pc/lsk/data/experiment/track_eval/"
# 速度门限 (m/s): 只有速度大于此值时才计入评估，防止停车时的定位漂移干扰数据
MIN_SPEED_THRESHOLD = 0.2           

# ==========================================
# 2. 功能类定义 (Class Definition)
# ==========================================

class PathTrackingEvaluator:
    def __init__(self):
        self.init_node()
        self.init_variables()
        self.init_subscribers()
        
        rospy.loginfo("跟踪精度评估器已启动...")
        rospy.loginfo("等待接收全局路径话题: %s", GLOBAL_PATH_TOPIC)
        rospy.loginfo("等待接收定位话题: %s", ODOM_TOPIC)

    def init_node(self):
        """初始化ROS节点"""
        rospy.init_node('path_tracking_evaluator', anonymous=True)

    def init_variables(self):
        """初始化数据容器"""
        self.cte_list = []          # 横向误差列表
        self.heading_err_list = []  # 航向误差列表
        self.yaw_rate_list = []     # 角速度列表 (用于计算平滑度)
        self.data_count = 0
        self.global_path_msg = None # 存储接收到的路径消息
        self.is_path_received = False

    def init_subscribers(self):
        """初始化订阅者"""
        rospy.Subscriber(GLOBAL_PATH_TOPIC, Path, self.path_callback)
        rospy.Subscriber(ODOM_TOPIC, Odometry, self.odom_callback)

    def path_callback(self, msg):
        """
        [回调函数] 接收全局路径
        只需接收一次即可，或者随路径更新而更新
        """
        if not self.is_path_received:
            rospy.loginfo("成功接收到全局路径! 包含 %d 个路点。", len(msg.poses))
            self.is_path_received = True
        self.global_path_msg = msg

    def perform_coordinate_transform(self, raw_odom):
        """
        [核心功能] 执行坐标变换：从 laser_link 转换到 base_link
        完全复刻 C++ 代码逻辑
        """
        # 1. 提取四元数并转换为欧拉角 Yaw
        orientation_q = raw_odom.pose.pose.orientation
        quaternion_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        (_, _, raw_yaw) = tf.transformations.euler_from_quaternion(quaternion_list)

        # 2. 应用角度偏差修正 (Yaw Offset)
        # double yaw = tf::getYaw(...) + 3.4/180*M_PI;
        corrected_yaw = raw_yaw + (YAW_OFFSET_DEG * math.pi / 180.0)

        # 3. 归一化角度到 [-pi, pi]
        corrected_yaw = math.atan2(math.sin(corrected_yaw), math.cos(corrected_yaw))

        # 4. 如果不需要变换位置，直接返回修正后的角度
        if not ENABLE_STATIC_TRANSFORM:
            return raw_odom.pose.pose.position.x, raw_odom.pose.pose.position.y, corrected_yaw

        # 5. 应用位置变换 (Laser -> Base)
        # x -= _laser2base * cos(yaw);
        # y -= _laser2base * sin(yaw);
        trans_x = raw_odom.pose.pose.position.x - LASER2BASE_DIST * math.cos(corrected_yaw)
        trans_y = raw_odom.pose.pose.position.y - LASER2BASE_DIST * math.sin(corrected_yaw)

        return trans_x, trans_y, corrected_yaw

    def find_nearest_pose_on_path(self, car_x, car_y):
        """
        [核心算法] 在全局路径上找到离当前车辆位置最近的点（用于近似计算误差）
        注意：为了效率，这里遍历寻找欧氏距离最近的离散点。
        如果路径点很稀疏，建议优化为“点到线段的距离”。
        考虑到一般 path 话题点很密集，直接找最近点通常足够。
        """
        if self.global_path_msg is None:
            return None, None, None

        min_dist = float('inf')
        nearest_idx = -1

        # 简单的遍历搜索 (对于几百个点的路径，Python处理也很快)
        # 如果路径极长，可优化为 KD-Tree 或 局部搜索
        poses = self.global_path_msg.poses
        for i, pose_stamped in enumerate(poses):
            px = pose_stamped.pose.position.x
            py = pose_stamped.pose.position.y
            dist = math.sqrt((px - car_x)**2 + (py - car_y)**2)
            if dist < min_dist:
                min_dist = dist
                nearest_idx = i

        if nearest_idx == -1:
            return None, None, None

        # 获取最近点的坐标
        nearest_pose = poses[nearest_idx]
        target_x = nearest_pose.pose.position.x
        target_y = nearest_pose.pose.position.y

        # 获取最近点的朝向 (Yaw)
        q = nearest_pose.pose.orientation
        (_, _, target_yaw) = tf.transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])

        # 优化：计算点到“线段”的垂直距离 (CTE) 而不是点到点的欧氏距离
        # 找到最近点的前一个或后一个点，构成线段
        idx_next = nearest_idx + 1 if nearest_idx + 1 < len(poses) else nearest_idx - 1
        if idx_next >= 0 and idx_next < len(poses):
            p1 = poses[nearest_idx].pose.position
            p2 = poses[idx_next].pose.position
            # 向量运算求点(car)到直线(p1-p2)的距离
            # 面积法：CTE = |(x2-x1)(y1-y0) - (x1-x0)(y2-y1)| / sqrt((x2-x1)^2 + (y2-y1)^2)
            numerator = abs((p2.y - p1.y)*car_x - (p2.x - p1.x)*car_y + p2.x*p1.y - p2.y*p1.x)
            denominator = math.sqrt((p2.y - p1.y)**2 + (p2.x - p1.x)**2)
            if denominator > 0.0001:
                cte = numerator / denominator
            else:
                cte = min_dist # 如果两点重合，退化为欧氏距离
        else:
            cte = min_dist

        return cte, target_yaw, min_dist

    def calculate_metrics_step(self, cte, car_yaw, path_yaw, angular_vel):
        """
        [核心功能] 缓存单帧计算结果
        """
        # 1. 存储 CTE
        self.cte_list.append(cte)

        # 2. 计算航向误差 (Heading Error)
        heading_err = car_yaw - path_yaw
        # 归一化到 [-pi, pi]
        heading_err = math.atan2(math.sin(heading_err), math.cos(heading_err))
        self.heading_err_list.append(heading_err)

        # 3. 存储角速度 (用于平滑度)
        self.yaw_rate_list.append(angular_vel)
        self.data_count += 1

    def odom_callback(self, msg):
        """
        [回调函数] 定位数据处理主入口
        """
        # 0. 检查是否已收到路径
        if not self.is_path_received:
            return

        # 1. 速度过滤 (防止停车噪声)
        v_x = msg.twist.twist.linear.x
        v_y = msg.twist.twist.linear.y
        current_speed = math.sqrt(v_x**2 + v_y**2)

        if current_speed < MIN_SPEED_THRESHOLD:
            return

        # 2. 坐标变换
        car_x, car_y, car_yaw = self.perform_coordinate_transform(msg)
        
        # 3. 在路径上寻找匹配点并计算 CTE 和 目标航向
        cte, target_yaw, _ = self.find_nearest_pose_on_path(car_x, car_y)

        if cte is None:
            return

        # 4. 获取角速度
        angular_vel = msg.twist.twist.angular.z

        # 5. 记录数据
        self.calculate_metrics_step(cte, car_yaw, target_yaw, angular_vel)

    def calculate_final_statistics(self):
        """
        [核心功能] 计算最终统计指标
        """
        if self.data_count == 0:
            rospy.logwarn("评估结束：没有记录到有效数据 (可能是速度过低或话题未发布)")
            return None

        np_cte = np.array(self.cte_list)
        np_heading = np.array(self.heading_err_list)
        np_yaw_rate = np.array(self.yaw_rate_list)

        # 指标计算
        rmse_cte = np.sqrt(np.mean(np_cte**2))
        max_abs_cte = np.max(np.abs(np_cte))
        rmse_heading_deg = np.sqrt(np.mean(np_heading**2)) * 180.0 / math.pi
        smoothness_std = np.std(np_yaw_rate)

        return {
            "rmse_cte": rmse_cte,
            "max_cte": max_abs_cte,
            "rmse_heading": rmse_heading_deg,
            "smoothness": smoothness_std,
            "count": self.data_count
        }

    def save_metrics_to_txt(self, metrics):
        """
        [核心功能] 生成中文报告 TXT
        """
        if metrics is None:
            return

        # 1. 确保文件夹存在
        if not os.path.exists(OUTPUT_FOLDER_PATH):
            os.makedirs(OUTPUT_FOLDER_PATH)

        # 2. 生成文件名 (基于时间戳)
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        filename = "tracking_result_{}.txt".format(timestamp)
        full_path = os.path.join(OUTPUT_FOLDER_PATH, filename)

        # 3. 准备写入的内容
        content = []
        content.append("========== 自动驾驶跟踪精度评估报告 ==========")
        content.append("实验时间: {}".format(timestamp))
        content.append("参考路径话题: {}".format(GLOBAL_PATH_TOPIC))
        content.append("有效数据点数: {}".format(metrics["count"]))
        content.append("\n========== 指标定义说明 ==========")
        content.append("1. CTE 均方根误差 (RMSE of CTE):")
        content.append("   单位: 米 (m)。反映了车辆整体偏离路径的程度，数值越小，跟踪越精准。")
        content.append("\n2. 最大横向误差 (Max Absolute CTE):")
        content.append("   单位: 米 (m)。反映了实验过程中发生的最严重的偏离情况（最差表现）。")
        content.append("\n3. 航向误差均方根 (RMSE of Heading Error):")
        content.append("   单位: 度 (deg)。反映了车头朝向与路径切向的平均偏差，数值越小表示车辆姿态越稳定。")
        content.append("\n4. 控制平滑度 (Control Smoothness / Yaw Rate STD):")
        content.append("   单位: 弧度/秒 (rad/s)。计算角速度的标准差。数值越小，说明方向盘抖动越少，控制越平滑。")
        content.append("\n========== 实验结果数据 ==========")
        content.append("RMSE of CTE       : {:.4f} m".format(metrics["rmse_cte"]))
        content.append("Max Absolute CTE  : {:.4f} m".format(metrics["max_cte"]))
        content.append("RMSE of Heading   : {:.4f} deg".format(metrics["rmse_heading"]))
        content.append("Control Smoothness: {:.4f} rad/s".format(metrics["smoothness"]))
        content.append("==========================================")

        # 4. 写入文件
        try:
            with open(full_path, 'w') as f:
                f.write('\n'.join(content))
            
            rospy.loginfo("评估报告已生成！")
            rospy.loginfo("保存路径: %s", full_path)
            # 在终端也打印一遍结果
            rospy.loginfo("RMSE CTE: %.4f m, Max CTE: %.4f m", metrics["rmse_cte"], metrics["max_cte"])
            
        except IOError as e:
            rospy.logerr("写入文件失败: %s", e)

    def run(self):
        rospy.spin()
        # 节点关闭时 (Ctrl+C) 执行保存
        metrics = self.calculate_final_statistics()
        self.save_metrics_to_txt(metrics)

# ==========================================
# 3. 程序入口
# ==========================================

if __name__ == '__main__':
    try:
        evaluator = PathTrackingEvaluator()
        evaluator.run()
    except rospy.ROSInterruptException:
        pass