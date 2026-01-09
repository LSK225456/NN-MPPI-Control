#!/usr/bin/env python
# -*- coding: UTF-8 -*-


import rospy
import numpy as np

# ROS 消息类型导入
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import PoseStamped, PointStamped, TransformStamped, Pose, Twist, Point

from tf.transformations import quaternion_from_euler, euler_matrix

# 数学库导入
from math import sqrt, pow, cos, sin, atan2, pi

# TF 变换库导入
import tf
import tf.transformations
import tf2_ros
from tf.transformations import quaternion_from_euler
import tf2_geometry_msgs

# --- 全局变量定义 ---
odom_path = Path()      # 记录机器人实际轨迹
desired_path = Path()   # 将在启动时被一次性计算并填充，之后不再改变
robot_odom = Odometry() # 存储最新的里程计信息
odom_count = 0
planning_frame = "map"   # 默认值，会被 launch 参数覆盖
car_frame = "base_link"  # 默认值，会被 launch 参数覆盖

tf_buffer = None      
tf_listener = None    

# 为发布者和订阅者预先声明全局变量，以便回调函数可以访问
desired_path_pub = None
odom_path_pub = None
odom_sub = None


# --- 核心回调函数 ---
def odom_cb(data):
    """
    这个回调函数通过以下步骤，正确记录并发布车辆在 'map' 坐标系下的真实轨迹:
    1. 接收来自里程计的位姿数据 (通常在 'odom' 坐标系下)。
    2. 使用 tf2 将该位姿变换到 'map' 坐标系。
    3. 将变换后的位姿点追加到路径中并发布。
    """
    global odom_path, odom_count, odom_path_pub, tf_buffer
    
    odom_count += 1
    if odom_count % 3 != 0:
        return

    # 1. 创建一个 PoseStamped 对象，它的header包含了原始的坐标系信息 (例如 'odom')
    original_pose = PoseStamped()
    original_pose.header = data.header
    original_pose.pose = data.pose.pose

    try:
        # 2. 使用 tf_buffer.transform() 进行坐标变换
        #    目标坐标系是全局变量 id, 即 "map"
        #    我们给一个短暂的超时时间，以应对轻微的同步问题
        target_pose = tf_buffer.transform(original_pose, planning_frame, rospy.Duration(0.1))

        # 3. 将正确变换后的位姿追加到我们的路径中
        odom_path.header.stamp = rospy.Time.now()
        odom_path.header.frame_id = planning_frame
        odom_path.poses.append(target_pose)
        
        odom_path_pub.publish(odom_path)

    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
        # 如果 tf 变换查询失败 (例如 tf 树尚未连接)，打印警告信息，避免程序崩溃
        rospy.logwarn("坐标变换失败，从 '%s' 到 '%s': %s", data.header.frame_id, planning_frame, e)



def publish_desired_path(event):
    """
    这个函数被定时器周期性调用，只做一件事：发布已经生成好的全局 desired_path
    """
    global desired_path, desired_path_pub
    if desired_path.poses:
        # 每次发布时更新一下时间戳，这是良好实践
        desired_path.header.stamp = rospy.get_rostime()
        desired_path_pub.publish(desired_path)


def get_initial_transform():
    """
    成功时返回base2world矩阵,否则在ROS关闭时返回None。
    此版本使用经典的 tf 库，先等待再查询，启动时更稳定。
    """
    listener = tf.TransformListener()
    # 12.16 修改: 替换硬编码，使用全局变量
    target_frame = planning_frame 
    source_frame = car_frame
    
    rospy.loginfo("Waiting for transform from '%s' to '%s'..." % (target_frame, source_frame))
  
    # 步骤 1: 使用 waitForTransform 明确等待TF树连接成功
    try:
        # 等待最多100秒，直到变换关系可用
        listener.waitForTransform(target_frame, source_frame, rospy.Time(0), rospy.Duration(100.0))
        rospy.loginfo("Transform is available. Getting initial pose.")
    except (tf.Exception) as e:
        rospy.logerr("Could not get transform from '%s' to '%s' after 100s: %s", target_frame, source_frame, e)
        return None

    # 步骤 2: 等待成功后，再次尝试获取变换，这次应该会立即成功
    try:
        (trans, rot) = listener.lookupTransform(target_frame, source_frame, rospy.Time(0))
        
        # 将获取到的 (translation, rotation) 元组直接转换为 4x4 矩阵
        matrix = tf.transformations.quaternion_matrix(rot)
        matrix[:3, 3] = trans
        
        rospy.loginfo("Successfully got the initial transform.")
        return matrix # 获取成功，返回矩阵

    except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
        # 这一步理论上不应该失败，但为了代码健壮性还是加上异常处理
        rospy.logerr("Failed to lookup transform even after waiting: %s", e)
        return None
# ======================= 新增代码 ENDS ============================

def generate_path(base2world):
    """
    它会修改全局变量 desired_path。
    """
    global desired_path # 明确指出要修改全局的 desired_path

    # 12.17 新增：读取角度偏差参数并修正初始位姿矩阵
    # 原理：如果baselink偏左(正)，我们需要将生成路径的基准向右(负)旋转，以对齐物理车头
    heading_offset_deg = rospy.get_param('~heading_offset_deg', 0.0)
    if abs(heading_offset_deg) > 0.001:
        rospy.loginfo("Applying heading offset correction: %.2f degrees", heading_offset_deg)
        offset_rad = heading_offset_deg * (pi / 180.0)
        # 生成绕Z轴旋转的修正矩阵，注意这里取负号进行反向补偿
        correction_matrix = euler_matrix(0, 0, -offset_rad)
        # 将修正矩阵应用到基准变换中 (base2world * correction)
        base2world = np.dot(base2world, correction_matrix)
    # 12.17 修改结束
    
    trajectory_type = rospy.get_param('~trajectory_type', 'line')
    
    if trajectory_type == "line":
        rospy.loginfo("Generating a single straight-line path...")
        line_length = rospy.get_param('~line_length', 10.0) 
        num_points = rospy.get_param('~num_points', 1000)  
        desired_path.header.frame_id = planning_frame
        
        for t in range(num_points):
            pose = PoseStamped()
            x_local = (line_length / (num_points - 1)) * t
            y_local = 0.0
            
            local_pose_matrix = tf.transformations.translation_matrix([x_local, y_local, 0])
            world_pose_matrix = np.dot(base2world, local_pose_matrix)        # 将局部坐标点变换到了 map 全局坐标系下。
            
            trans = tf.transformations.translation_from_matrix(world_pose_matrix)
            qua = tf.transformations.quaternion_from_matrix(world_pose_matrix)

            # todo1021：确保四元数是有效的
            if np.linalg.norm(qua) < 1e-6:  # 如果四元数接近0
                qua = [0.0, 0.0, 0.0, 1.0]  # 使用单位四元数

            pose.header.seq = t
            pose.header.frame_id = planning_frame
            pose.pose.position.x = trans[0]
            pose.pose.position.y = trans[1]
            pose.pose.position.z = trans[2]
            pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z, pose.pose.orientation.w = qua
            desired_path.poses.append(pose)
            
        rospy.loginfo("Path generated with %d poses.", len(desired_path.poses))

    elif trajectory_type == "curve":
        rospy.loginfo("Generating a single Bezier curve path...")
        # 从参数服务器获取x和y的偏移量
        offset_x = rospy.get_param('~goal_offset_x', 10.0)
        offset_y = rospy.get_param('~goal_offset_y', 5.0)
        
        # 1. 计算起点和终点位姿
        start_pose = Pose()
        start_trans = tf.transformations.translation_from_matrix(base2world)
        start_qua = tf.transformations.quaternion_from_matrix(base2world)
        start_pose.position.x, start_pose.position.y, start_pose.position.z = start_trans
        start_pose.orientation.x, start_pose.orientation.y, start_pose.orientation.z, start_pose.orientation.w = start_qua

        end_pose = Pose()
        # 计算终点在世界坐标系下的位置
        local_end_point = np.array([offset_x, offset_y, 0, 1])
        world_end_point = np.dot(base2world, local_end_point)
        end_pose.position.x = world_end_point[0]
        end_pose.position.y = world_end_point[1]
        end_pose.position.z = world_end_point[2]
        # 终点姿态与起点姿态相同
        end_pose.orientation = start_pose.orientation

        # 2. 调用封装好的函数生成路径点列表
        bezier_poses = _generate_bezier_poses(start_pose, end_pose)

        # 3. 将生成的路径点添加到全局路径中
        desired_path.header.frame_id = planning_frame
        desired_path.poses = bezier_poses
        
        rospy.loginfo("Curve path generated with %d poses.", len(desired_path.poses))



def setup_ros_communications():
    """
    封装了所有发布者、订阅者和定时器的设置。
    """
    # 明确指出要初始化全局的发布者和订阅者变量
    global desired_path_pub, odom_path_pub, error_path_pub, odom_sub

    topic_global_path = rospy.get_param('~topic_global_path', '/desired_path')
    topic_odom = rospy.get_param('~topic_odom', '/odom')
    # 设置所有的发布者和订阅者
    desired_path_pub = rospy.Publisher(topic_global_path, Path, queue_size=10)
    odom_path_pub = rospy.Publisher('/recorded_path', Path, queue_size=10)
    error_path_pub = rospy.Publisher('/error_path', Path, queue_size=10)
    odom_sub = rospy.Subscriber(topic_odom, Odometry, odom_cb)
    
    # 使用定时器独立、周期性地发布期望路径
    rospy.Timer(rospy.Duration(1), publish_desired_path)
    # publish_desired_path
    


###  ----------------贝塞尔曲线-----------------  ###
def _bezier_interpolate(p0, p1, p2, p3, t):
    """三次贝塞尔曲线插值函数"""
    u = 1.0 - t
    point = Point()
    point.x = u*u*u*p0.x + 3*u*u*t*p1.x + 3*u*t*t*p2.x + t*t*t*p3.x
    point.y = u*u*u*p0.y + 3*u*u*t*p1.y + 3*u*t*t*p2.y + t*t*t*p3.y
    point.z = 0
    return point

def _compute_max_curvature(p0, p1, p2, p3):
    """计算贝塞尔曲线的最大曲率"""
    max_k = 0.0
    # 在0.1到0.9之间采样，避开端点
    for t in np.arange(0.1, 0.9, 0.1):
        # 使用微小步长计算一阶和二阶导数的近似值
        pt_prev = _bezier_interpolate(p0, p1, p2, p3, t - 0.005)
        pt_curr = _bezier_interpolate(p0, p1, p2, p3, t)
        pt_next = _bezier_interpolate(p0, p1, p2, p3, t + 0.005)
        
        dx1 = pt_curr.x - pt_prev.x
        dy1 = pt_curr.y - pt_prev.y
        dx2 = pt_next.x - pt_curr.x
        dy2 = pt_next.y - pt_curr.y

        # 曲率公式 k = |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)
        # 这里用差分近似导数
        x_prime = (pt_next.x - pt_prev.x) / 0.01
        y_prime = (pt_next.y - pt_prev.y) / 0.01
        x_double_prime = (pt_next.x - 2 * pt_curr.x + pt_prev.x) / (0.005**2)
        y_double_prime = (pt_next.y - 2 * pt_curr.y + pt_prev.y) / (0.005**2)

        numerator = abs(x_prime * y_double_prime - y_prime * x_double_prime)
        denominator = pow(x_prime**2 + y_prime**2, 1.5)

        if denominator < 1e-6:
            continue
        
        k = numerator / denominator
        if k > max_k:
            max_k = k
            
    return max_k

def _generate_bezier_poses(start_pose, end_pose):
    """
    根据给定的起点和终点位姿，生成一段贝塞尔曲线路径点列表
    """
    # 从ROS参数服务器加载贝塞尔曲线参数
    R_ = rospy.get_param('~turning_radius', 1.0)
    dt_ = rospy.get_param('~sampling_step', 0.01)
    max_iter_ = rospy.get_param('~max_adjust_iter', 100)
    factor_ = rospy.get_param('~adjust_factor', 0.9)

    p0 = start_pose.position
    p3 = end_pose.position
    
    # 计算起点和终点的yaw角
    start_q = [start_pose.orientation.x, start_pose.orientation.y, start_pose.orientation.z, start_pose.orientation.w]
    theta0 = tf.transformations.euler_from_quaternion(start_q)[2]

    end_q = [end_pose.orientation.x, end_pose.orientation.y, end_pose.orientation.z, end_pose.orientation.w]
    theta3 = tf.transformations.euler_from_quaternion(end_q)[2]
    
    # 控制点基础偏移量，可根据需要调整
    dist = sqrt(pow(p3.x - p0.x, 2) + pow(p3.y - p0.y, 2))
    delta = dist / 2.0 

    # 生成初始控制点
    p1 = Point(p0.x + delta * cos(theta0), p0.y + delta * sin(theta0), 0)
    p2 = Point(p3.x - delta * cos(theta3), p3.y - delta * sin(theta3), 0)

    # 曲率约束调整
    iter_count = 0
    max_k = _compute_max_curvature(p0, p1, p2, p3)
    while max_k > 1.0/R_ and iter_count < max_iter_:
        delta *= factor_
        p1 = Point(p0.x + delta * cos(theta0), p0.y + delta * sin(theta0), 0)
        p2 = Point(p3.x - delta * cos(theta3), p3.y - delta * sin(theta3), 0)
        max_k = _compute_max_curvature(p0, p1, p2, p3)
        iter_count += 1
    
    if iter_count > 0:
        rospy.loginfo("Bezier curve adjusted %d times to meet curvature constraint.", iter_count)

    # 生成路径点
    poses_list = []
    for t in np.arange(0, 1.0, dt_):
        pose = PoseStamped()
        pose.header.frame_id = planning_frame
        pose.header.stamp = rospy.Time.now()
        pose.pose.position = _bezier_interpolate(p0, p1, p2, p3, t)
        
        # 计算切线方向作为航向
        dx = 3*(1-t)**2 * (p1.x-p0.x) + 6*(1-t)*t * (p2.x-p1.x) + 3*t**2 * (p3.x-p2.x)
        dy = 3*(1-t)**2 * (p1.y-p0.y) + 6*(1-t)*t * (p2.y-p1.y) + 3*t**2 * (p3.y-p2.y)
        yaw = atan2(dy, dx)
        
        q = tf.transformations.quaternion_from_euler(0, 0, yaw)
        pose.pose.orientation.x = q[0]
        pose.pose.orientation.y = q[1]
        pose.pose.orientation.z = q[2]
        pose.pose.orientation.w = q[3]
        
        poses_list.append(pose)
    
    # 强制添加终点以确保精度
    final_pose = PoseStamped()
    final_pose.header.frame_id = planning_frame
    final_pose.header.stamp = rospy.Time.now()
    final_pose.pose = end_pose
    poses_list.append(final_pose)

    return poses_list
###  ----------------贝塞尔曲线-----------------  ###




if __name__ == '__main__':
    rospy.init_node('path_generator_node')
    rospy.loginfo("Path Generator Node Started.")
    
    planning_frame = rospy.get_param('~planning_frame', 'world')
    car_frame = rospy.get_param('~car_frame', 'base_link')
    
    rospy.loginfo("Using Planning Frame: %s, Car Frame: %s", planning_frame, car_frame)

    global tf_buffer, tf_listener
    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)
    
    # 步骤 1: 获取初始变换矩阵
    initial_transform = get_initial_transform()
    
    # 如果获取成功，则继续执行
    if initial_transform is not None:
        # 步骤 2: 使用获取到的变换矩阵，生成一次路径
        generate_path(initial_transform)
        
        # 步骤 3 & 4: 设置所有通信接口和定时器
        setup_ros_communications()
        
        rospy.loginfo("Initialization complete. Spinning and continuously publishing the desired path...")
        rospy.spin()
    else:
        rospy.logerr("Failed to get initial transform. Shutting down.")