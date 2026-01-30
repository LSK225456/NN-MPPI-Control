#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MPPI 测试用模拟环境

功能：
1. 模拟机器人：订阅 /cmd_vel，发布 /odom 和 TF
2. 发布测试路径：/test_path（支持直线、圆形、正弦波、贝塞尔曲线）
3. 发布可视化：机器人 marker、历史轨迹

使用方法：
    终端1: roslaunch mppi test_mppi_demo.launch
    终端2: roslaunch mppi mppi_controller.launch

作者: LSK
"""

import rospy
import numpy as np
import math
import threading
import argparse

from geometry_msgs.msg import Twist, PoseStamped, TransformStamped, Point
from nav_msgs.msg import Odometry, Path
from visualization_msgs.msg import Marker
from tf2_msgs.msg import TFMessage


class SimpleRobotSimulator:
    """简单机器人模拟器（带可视化）"""
    
    def __init__(self, init_pose=(0.0, 0.0, 0.0)):
        # 机器人状态 [x, y, θ, v, w]
        self.state = np.array([
            init_pose[0], init_pose[1], init_pose[2], 0.0, 0.0
        ], dtype=np.float64)
        
        # 一阶滞后模型参数
        self.tau_v = 0.1
        self.tau_w = 0.05
        
        # 控制指令
        self.cmd_v = 0.0
        self.cmd_w = 0.0
        self.cmd_received = False
        
        # 历史轨迹
        self.trajectory = []
        self.max_traj_len = 1000
        
        self.lock = threading.Lock()
        
        # ROS 通信
        self.pub_odom = rospy.Publisher('/odom', Odometry, queue_size=10)
        self.pub_marker = rospy.Publisher('/robot_marker', Marker, queue_size=1)
        self.pub_traj = rospy.Publisher('/robot_trajectory', Path, queue_size=1)
        self.sub_cmd = rospy.Subscriber('/cmd_vel', Twist, self.cb_cmd)
        
        # TF 发布器（不使用 tf 库，直接发布 TFMessage）
        self.pub_tf = rospy.Publisher('/tf', TFMessage, queue_size=10)
        
        # 仿真参数
        self.sim_rate = 50.0
        self.dt = 1.0 / self.sim_rate
        
        rospy.loginfo("=" * 50)
        rospy.loginfo("[Simulator] 机器人模拟器已启动")
        rospy.loginfo("[Simulator] 初始位置: ({:.2f}, {:.2f})".format(init_pose[0], init_pose[1]))
        rospy.loginfo("[Simulator] 等待 /cmd_vel 控制指令...")
        rospy.loginfo("=" * 50)
    
    def cb_cmd(self, msg):
        with self.lock:
            self.cmd_v = msg.linear.x
            self.cmd_w = msg.angular.z
            if not self.cmd_received:
                self.cmd_received = True
                rospy.loginfo("[Simulator] 收到首个控制指令!")
    
    def step(self):
        with self.lock:
            x, y, theta, v, w = self.state
            
            # 一阶滞后
            alpha_v = self.dt / (self.tau_v + self.dt)
            alpha_w = self.dt / (self.tau_w + self.dt)
            v_next = v + alpha_v * (self.cmd_v - v)
            w_next = w + alpha_w * (self.cmd_w - w)
            
            # 运动学
            x_next = x + v_next * math.cos(theta) * self.dt
            y_next = y + v_next * math.sin(theta) * self.dt
            theta_next = theta + w_next * self.dt
            theta_next = math.atan2(math.sin(theta_next), math.cos(theta_next))
            
            self.state = np.array([x_next, y_next, theta_next, v_next, w_next])
            
            # 记录轨迹
            self.trajectory.append([x_next, y_next])
            if len(self.trajectory) > self.max_traj_len:
                self.trajectory.pop(0)
    
    def euler_to_quaternion(self, roll, pitch, yaw):
        """欧拉角转四元数"""
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)
        
        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy
        
        return (qx, qy, qz, qw)
    
    def publish_all(self):
        now = rospy.Time.now()
        
        with self.lock:
            x, y, theta, v, w = self.state
            traj = list(self.trajectory)
        
        # 1. 发布 Odometry
        odom = Odometry()
        odom.header.stamp = now
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_link"
        odom.pose.pose.position.x = x
        odom.pose.pose.position.y = y
        
        qx, qy, qz, qw = self.euler_to_quaternion(0, 0, theta)
        odom.pose.pose.orientation.x = qx
        odom.pose.pose.orientation.y = qy
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw
        
        odom.twist.twist.linear.x = v
        odom.twist.twist.angular.z = w
        self.pub_odom.publish(odom)
        
        # 2. 发布 TF 变换
        tf_msg = TFMessage()
        
        # TF: map -> odom (静态)
        t1 = TransformStamped()
        t1.header.stamp = now
        t1.header.frame_id = "map"
        t1.child_frame_id = "odom"
        t1.transform.translation.x = 0.0
        t1.transform.translation.y = 0.0
        t1.transform.translation.z = 0.0
        t1.transform.rotation.x = 0.0
        t1.transform.rotation.y = 0.0
        t1.transform.rotation.z = 0.0
        t1.transform.rotation.w = 1.0
        
        # TF: odom -> base_link
        t2 = TransformStamped()
        t2.header.stamp = now
        t2.header.frame_id = "odom"
        t2.child_frame_id = "base_link"
        t2.transform.translation.x = x
        t2.transform.translation.y = y
        t2.transform.translation.z = 0.0
        t2.transform.rotation.x = qx
        t2.transform.rotation.y = qy
        t2.transform.rotation.z = qz
        t2.transform.rotation.w = qw
        
        tf_msg.transforms = [t1, t2]
        self.pub_tf.publish(tf_msg)
        
        # 3. 发布机器人 Marker (箭头)
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.header.stamp = now
        marker.ns = "robot"
        marker.id = 0
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = 0.1
        marker.pose.orientation.x = qx
        marker.pose.orientation.y = qy
        marker.pose.orientation.z = qz
        marker.pose.orientation.w = qw
        marker.scale.x = 0.5
        marker.scale.y = 0.15
        marker.scale.z = 0.1
        marker.color.r = 0.2
        marker.color.g = 0.5
        marker.color.b = 1.0
        marker.color.a = 1.0
        self.pub_marker.publish(marker)
        
        # 4. 发布历史轨迹
        if len(traj) > 1:
            path = Path()
            path.header.frame_id = "odom"
            path.header.stamp = now
            for pt in traj:
                pose = PoseStamped()
                pose.header = path.header
                pose.pose.position.x = pt[0]
                pose.pose.position.y = pt[1]
                pose.pose.orientation.w = 1.0
                path.poses.append(pose)
            self.pub_traj.publish(path)
    
    def run(self):
        rate = rospy.Rate(self.sim_rate)
        log_counter = 0
        
        while not rospy.is_shutdown():
            self.step()
            self.publish_all()
            
            # 每秒打印一次状态
            log_counter += 1
            if log_counter % int(self.sim_rate) == 0:
                with self.lock:
                    x, y, theta, v, w = self.state
                rospy.loginfo(
                    "[Simulator] pos=({:.2f}, {:.2f}) | yaw={:.1f}° | v={:.2f} w={:.2f}".format(
                        x, y, math.degrees(theta), v, w
                    )
                )
            
            rate.sleep()


class TestPathPublisher:
    """测试路径发布器"""
    
    def __init__(self, path_type='straight'):
        self.path_type = path_type
        self.pub = rospy.Publisher('/test_path', Path, queue_size=1, latch=True)
        
        rospy.loginfo("[PathPublisher] 准备生成路径，类型: {}".format(path_type))
        
        path = self._generate_path()
        self._publish(path)
        
        rospy.loginfo("[PathPublisher] 发布 {} 路径，{} 个点".format(path_type, len(path)))
    
    def _generate_path(self):
        """生成测试路径"""
        if self.path_type == 'straight':
            x = np.linspace(0, 10, 100)
            y = np.zeros_like(x)
        elif self.path_type == 'circle':
            t = np.linspace(0, 2 * np.pi, 100)
            x = 3.0 * np.cos(t) + 3.0
            y = 3.0 * np.sin(t)
        elif self.path_type == 'sine':
            x = np.linspace(0, 10, 100)
            y = 1.5 * np.sin(0.5 * x)
        elif self.path_type == 'bezier':
            # 贝塞尔曲线路径
            rospy.loginfo("[PathPublisher] 生成贝塞尔曲线路径...")
            return self._generate_bezier_curve()
        else:
            rospy.logwarn("[PathPublisher] 未知路径类型 '{}', 使用直线".format(self.path_type))
            x = np.linspace(0, 10, 100)
            y = np.zeros_like(x)
        
        return np.column_stack((x, y))
    
    def _generate_bezier_curve(self):
        """
        生成贝塞尔曲线路径（参考 mpc_trajectory_generation.py）
        
        Returns:
            np.ndarray: [N, 2] 路径点 (x, y)
        """
        # 从 ROS 参数服务器读取贝塞尔曲线参数
        start_x = rospy.get_param('~bezier_start_x', 0.0)
        start_y = rospy.get_param('~bezier_start_y', 0.0)
        end_x = rospy.get_param('~bezier_end_x', 10.0)
        end_y = rospy.get_param('~bezier_end_y', 5.0)
        
        # 控制点距离系数
        control_dist_ratio = rospy.get_param('~bezier_control_ratio', 0.5)
        
        # 采样步长（越小越密集）
        dt = rospy.get_param('~bezier_sampling_step', 0.01)
        
        rospy.loginfo("[PathPublisher] 贝塞尔参数: start=({:.2f}, {:.2f}), end=({:.2f}, {:.2f}), ratio={:.2f}".format(
            start_x, start_y, end_x, end_y, control_dist_ratio
        ))
        
        # 定义四个控制点 p0, p1, p2, p3
        p0 = Point(x=start_x, y=start_y, z=0)
        p3 = Point(x=end_x, y=end_y, z=0)
        
        # 起点和终点的航向角
        theta0 = 0.0  # 起点朝向 x 正方向
        theta3 = math.atan2(end_y - start_y, end_x - start_x)  # 终点朝向终点方向
        
        # 计算控制点 p1, p2
        dist = math.sqrt((p3.x - p0.x)**2 + (p3.y - p0.y)**2)
        delta = dist * control_dist_ratio
        
        p1 = Point(
            x=p0.x + delta * math.cos(theta0),
            y=p0.y + delta * math.sin(theta0),
            z=0
        )
        p2 = Point(
            x=p3.x - delta * math.cos(theta3),
            y=p3.y - delta * math.sin(theta3),
            z=0
        )
        
        rospy.loginfo("[PathPublisher] 控制点: p0=({:.2f},{:.2f}), p1=({:.2f},{:.2f}), p2=({:.2f},{:.2f}), p3=({:.2f},{:.2f})".format(
            p0.x, p0.y, p1.x, p1.y, p2.x, p2.y, p3.x, p3.y
        ))
        
        # 生成贝塞尔曲线点
        points = []
        for t in np.arange(0, 1.0, dt):
            pt = self._bezier_interpolate(p0, p1, p2, p3, t)
            points.append([pt.x, pt.y])
        
        # 添加终点
        points.append([p3.x, p3.y])
        
        rospy.loginfo("[PathPublisher] 贝塞尔曲线生成完成: 共 {} 个点".format(len(points)))
        
        return np.array(points)
    
    @staticmethod
    def _bezier_interpolate(p0, p1, p2, p3, t):
        """三次贝塞尔曲线插值"""
        u = 1.0 - t
        point = Point()
        point.x = u*u*u*p0.x + 3*u*u*t*p1.x + 3*u*t*t*p2.x + t*t*t*p3.x
        point.y = u*u*u*p0.y + 3*u*u*t*p1.y + 3*u*t*t*p2.y + t*t*t*p3.y
        point.z = 0
        return point
    
    def _publish(self, path):
        """发布路径到 ROS"""
        msg = Path()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "map"
        
        for pt in path:
            pose = PoseStamped()
            pose.header = msg.header
            pose.pose.position.x = pt[0]
            pose.pose.position.y = pt[1]
            pose.pose.orientation.w = 1.0
            msg.poses.append(pose)
        
        self.pub.publish(msg)
        rospy.loginfo("[PathPublisher] 路径已发布到 /test_path")


def main():
    # 解析命令行参数（用于独立运行）
    parser = argparse.ArgumentParser(description='MPPI 测试模拟环境')
    parser.add_argument('--path', '-p', type=str, default=None,
                        choices=['straight', 'circle', 'sine', 'bezier'],
                        help='路径类型')
    parser.add_argument('--x', type=float, default=0.0, help='初始 x')
    parser.add_argument('--y', type=float, default=0.0, help='初始 y')
    parser.add_argument('--yaw', type=float, default=0.0, help='初始 yaw (度)')
    
    args, _ = parser.parse_known_args()
    
    rospy.init_node('mppi_test_simulator', anonymous=False)
    
    # 从 ROS 参数服务器读取路径类型（launch 文件设置）
    # 如果命令行指定了 --path，则优先使用命令行参数
    if args.path is not None:
        path_type = args.path
        rospy.loginfo("[Main] 使用命令行参数路径类型: {}".format(path_type))
    else:
        path_type = rospy.get_param('~path', 'straight')
        rospy.loginfo("[Main] 使用 ROS 参数路径类型: {}".format(path_type))
    
    # 发布路径
    path_pub = TestPathPublisher(path_type)
    
    # 启动模拟器
    init_yaw = math.radians(args.yaw)
    simulator = SimpleRobotSimulator(init_pose=(args.x, args.y, init_yaw))
    simulator.run()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
