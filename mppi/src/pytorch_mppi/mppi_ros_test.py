#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import torch
import numpy as np
# import tf.transformations as tf_trans
import transformations as tf_trans
import math

# ROS 消息类型
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry, Path

from mppi import MPPI
# from mppi_viz import MPPIVisualizer
try:
    from mppi_viz import MPPIVisualizer 
except ImportError:
    from pytorch_mppi.mppi_viz import MPPIVisualizer




class MPPIControllerNode:
    """
    MPPI 控制器 ROS 节点封装
    功能：接收 /odom 定位和全局路径，利用 MPPI 算法输出 /cmd_vel 控制指令
    """
    def __init__(self):
        # 初始化节点
        rospy.init_node('mppi_controller_node', anonymous=True)

        # --- 参数配置 (从 Parameter Server 获取，支持 Launch 文件修改) ---
        
        # 话题名称配置
        self.topic_pose = rospy.get_param("~topic_pose", "/odom")           # 机器人的定位话题
        self.topic_twist = rospy.get_param("~topic_twist", "/wheel_odom")   # 机器人的速度话题
        self.topic_cmd = rospy.get_param("~topic_cmd", "/cmd_vel")          # 输出控制指令话题
        self.topic_path = rospy.get_param("~topic_path", "/global_plan16")      # 全局路径话题

        # 坐标变换参数配置
        self.is_odom_transform = rospy.get_param("~is_odom_transform", False)        # 是否把odom坐标转换
        self.laser2base_dist = rospy.get_param("~laser2base_dist", 0.38)      # 激光雷达相对于底盘的距离 (米)
        self.laser2base_angle = rospy.get_param("~laser2base_angle", 3.4 / 180.0 * math.pi)
        
        # MPPI 算法超参数
        self.horizon = rospy.get_param("~horizon", 30)              # 预测步长 T (例如往后预测30步)
        self.num_samples = rospy.get_param("~num_samples", 500)     # 采样轨迹数量 K
        self.lambda_val = rospy.get_param("~lambda", 0.05)          # 温度系数 (越小越倾向于利用现有最优，越大越随机)
        self.noise_sigma = rospy.get_param("~noise_sigma", [0.5, 0.6]) # 控制噪声标准差 [v_std, w_std]
        self.control_freq = rospy.get_param("~control_freq", 20.0)  # 控制频率 Hz
        
        # 路径跟踪参数
        self.lookahead_dist = rospy.get_param("~lookahead_dist", 1.0) # 前视距离 (米)，在路径上找多远的目标
        self.goal_tolerance = rospy.get_param("~goal_tolerance", 0.2) # 到达终点的判定距离
        self.ref_path_length = rospy.get_param("~ref_path_length", 5.0) # 参考路径长度：米
        self.w_cte = rospy.get_param("~w_cte", 100.0)   # 横向误差权重
        self.w_yaw = rospy.get_param("~w_yaw", 10.0)    # 航向误差权重
        self.w_vel = rospy.get_param("~w_vel", 0.5)     # 参考速度
        
        # 3. 目标状态参数
        # 期望的巡航线速度 (m/s)
        self.target_velocity = rospy.get_param("~target_velocity", 0.5)

        # 代价权重
        self.w_dist = rospy.get_param("~w_dist", 10.0)  # 距离代价权重
        self.w_vel = rospy.get_param("~w_vel", 0.1)     # 速度平滑/激励权重

        # 设备选择 (GPU/CPU)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        rospy.loginfo(f"MPPI 控制器运行设备: {self.device}")

        # --- 机器人状态初始化 ---
        # 状态向量: [x, y, theta]
        self.current_state = np.array([0.0, 0.0, 0.0]) 
        
        # 路径存储
        self.global_path_np = None  # 格式: numpy array [[x, y], [x, y], ...]
        self.local_target = None    # 当前时刻 MPPI 要追踪的临时目标点 [x, y]
        
        # --- MPPI 核心初始化 ---
        # 1. 构建噪声协方差矩阵
        noise_sigma_tensor = torch.tensor(
            [[self.noise_sigma[0], 0.0], [0.0, self.noise_sigma[1]]], 
            device=self.device, dtype=torch.float32
        )
        
        # 2. 设置控制量边界 (v_min, v_max, w_min, w_max)
        u_min = torch.tensor([-0.4, -0.4], device=self.device)  # 线速度，角速度下限
        u_max = torch.tensor([0.5, 0.4], device=self.device)

        # 3. 实例化 MPPI 类
        self.mppi = MPPI(
            dynamics=self.dynamics_model,           # 传入动力学模型函数
            running_cost=self.trajectory_cost_function, # 传入代价函数
            nx=3,                                   # 状态维度: x, y, theta
            noise_sigma=noise_sigma_tensor,         # 噪声矩阵
            num_samples=self.num_samples,           # 采样数
            horizon=self.horizon,                   # 预测步长
            device=self.device,                     # 运算设备
            lambda_=self.lambda_val,                # 温度系数
            u_min=u_min,                            # 控制下限
            u_max=u_max                             # 控制上限
        )

        self.viz = MPPIVisualizer(frame_id="map")

        # --- ROS 通信接口 ---
        self.sub_pose = rospy.Subscriber(self.topic_pose, Odometry, self.cb_pose)
        self.sub_twist = rospy.Subscriber(self.topic_twist, Odometry, self.cb_twist)
        self.sub_path = rospy.Subscriber(self.topic_path, Path, self.cb_path)
        self.pub_cmd = rospy.Publisher(self.topic_cmd, Twist, queue_size=1)

        # 启动控制定时器
        self.timer = rospy.Timer(rospy.Duration(1.0/self.control_freq), self.cb_control_loop)
        
        rospy.loginfo("MPPI 控制节点已启动，等待路径...")

    def dynamics_model(self, state, action):
        """
        动力学预测模型 (PyTorch 实现)
        作用：根据当前状态和控制量，预测下一时刻的状态。
        此处使用简化版差速模型 (Unicycle Model)。
        
        输入参数:
            state (Tensor): [K, 3] -> [x, y, theta] (K为采样数)
            action (Tensor): [K, 2] -> [v, w]
            
        返回参数:
            next_state (Tensor): [K, 3] -> 下一时刻的 [x, y, theta]
        """
        dt = 1.0 / self.control_freq
        
        # 提取状态分量
        x = state[:, 0]
        y = state[:, 1]
        theta = state[:, 2]
        
        # 提取动作分量
        v = action[:, 0]
        w = action[:, 1]
        
        # 运动学方程更新
        # x_{t+1} = x_t + v * cos(theta) * dt
        # y_{t+1} = y_t + v * sin(theta) * dt
        # theta_{t+1} = theta_t + w * dt
        new_x = x + v * torch.cos(theta) * dt
        new_y = y + v * torch.sin(theta) * dt
        new_theta = theta + w * dt

        new_theta = torch.atan2(torch.sin(new_theta), torch.cos(new_theta)) # 将角度限制在 [-pi, pi] 之间
        
        # 组合结果
        next_state = torch.stack([new_x, new_y, new_theta], dim=1)
        return next_state

    # // 11.24 修改：移除倒车惩罚，以 CTE 为绝对核心，允许通过倒车修正轨迹
    def trajectory_cost_function(self, state, action):
        """
        并行轨迹代价函数
        特点：
        1. 无倒车惩罚：MPPI 可自由探索负速度区域。
        2. 强 CTE 约束：只要偏离路径，代价指数级上升。
        """
        # --- 1. 边界条件：异常保护 ---
        if not hasattr(self, 'ref_path_tensor') or self.ref_path_tensor is None:
            return torch.zeros(state.shape[0], device=self.device)
            
        if not torch.is_tensor(state):
            state = torch.tensor(state, dtype=torch.float32, device=self.device)
            
        # --- 2. 核心计算：并行寻找最近点 (CTE) ---
        # state_pos: [K, 1, 2]
        # ref_pos:   [1, N, 2]
        state_pos = state[:, :2].unsqueeze(1)
        ref_pos = self.ref_path_tensor[:, :2].unsqueeze(0)
        
        # 计算所有粒子到参考路径所有点的欧氏距离平方
        diff = state_pos - ref_pos
        dists_sq = torch.sum(diff**2, dim=2) # [K, N]
        
        # 找到每个粒子距离路径最近的距离 (CTE平方) 和对应的点索引
        min_dists_sq, min_indices = torch.min(dists_sq, dim=1)
        
        # --- 3. 核心计算：航向误差 ---
        # 查表获取对应的参考航向
        ref_yaws = self.ref_path_tensor[min_indices, 2]
        
        # 计算偏差并归一化到 [-pi, pi]
        yaw_error = state[:, 2] - ref_yaws
        yaw_error = torch.atan2(torch.sin(yaw_error), torch.cos(yaw_error))
        
        # --- 4. 权重配置 (严格遵循您的优先级) ---
        W_CTE = 100.0       # [最高优先级] 必须死死咬住路径
        W_YAW = 10.0        # [次高优先级] 车头尽量正
        W_VEL = 0.5         # [低优先级] 仅作为驱动力，允许被 CTE 牺牲
        
        # --- 5. 计算各项 Cost ---
        
        # [Cost 1] 横向误差 (CTE)
        cost_cte = self.w_cte * min_dists_sq
        
        # [Cost 2] 航向误差
        cost_yaw = self.w_yaw * (yaw_error**2)
        
        # [Cost 3] 速度驱动
        # 这里的逻辑是：我们给一个期望速度(如 0.5)，
        # 但因为 W_VEL 很小 (0.5) 而 W_CTE 很大 (100.0)，
        # 如果倒车 (v < 0) 能让 min_dists_sq 变小，算法会果断选择倒车，
        # 因为节省的 CTE 代价远大于速度偏差带来的惩罚。
        target_v = 0.5
        cost_vel = self.w_vel * ((action[:, 0] - target_v)**2)
        
        # --- 6. 总代价 ---
        # 不再包含任何 crash/倒车 惩罚项
        total_cost = cost_cte + cost_yaw + cost_vel
        
        return total_cost
        
    

    def cb_pose(self, msg):
        """
        /odom 回调函数
        作用：更新机器人当前的全局位姿 [x, y, theta]
        支持可选的 laser_link 到 base_link 的坐标变换
        """
        try:
            p = msg.pose.pose.position
            o = msg.pose.pose.orientation
            
            # 四元数转欧拉角 (Yaw)
            quaternion = (o.x, o.y, o.z, o.w)
            euler = tf_trans.euler_from_quaternion(quaternion)
            yaw = euler[2]

            x_curr = p.x
            y_curr = p.y

            if self.is_odom_transform:      # 是否执行坐标变换
                yaw += self.laser2base_angle
                x_curr -= self.laser2base_dist * math.cos(yaw)
                y_curr -= self.laser2base_dist * math.sin(yaw)
            
            self.current_state[0] = x_curr
            self.current_state[1] = y_curr
            self.current_state[2] = yaw
            
        except Exception as e:
            rospy.logerr(f"处理 Odom 数据出错: {e}")

    def cb_twist(self, msg):
        """
        /wheel_odom 回调函数
        作用：获取当前实际速度。虽然本简化版 MPPI 暂未使用此数据初始化(主要靠位置推演)，
        但保留接口以便后续扩展动态模型。
        """
        pass

    def cb_path(self, msg):
        """
        /global_16 路径回调函数
        作用：接收并存储全局路径，将其转换为 numpy 数组以便快速计算。
        格式：nav_msgs/Path -> numpy array [[x, y], [x, y], ...]
        """
        if not msg.poses:
            return
        # print("~~~~~~~~~~~~~~~~~~~~~~~~`")
        path_list = []
        for pose_stamped in msg.poses:
            px = pose_stamped.pose.position.x
            py = pose_stamped.pose.position.y
            path_list.append([px, py])
            
        self.global_path_np = np.array(path_list)
        rospy.loginfo(f"收到新路径，包含 {len(path_list)} 个点")

    def update_local_target(self):
        """
        更新局部参考路径 (Local Reference Path)
        作用：从全局路径中截取车辆前方指定长度（米）的路径片段，转换为 Tensor 供 Cost 函数计算。
        """
        if self.global_path_np is None or len(self.global_path_np) == 0:
            self.ref_path_tensor = None
            self.local_target = None
            return

        # 1. 找到离车最近的全局路径点索引
        # self.current_state[:2] -> [x, y]
        diff = self.global_path_np - self.current_state[:2]
        dists_sq = np.sum(diff**2, axis=1)
        min_index = np.argmin(dists_sq)
        
        # 2. 基于物理距离截取路径 (Length-based Slicing)
        # 从最近点开始累加距离，直到达到 self.ref_path_length
        current_dist = 0.0
        end_index = min_index
        
        # 防止索引越界
        max_idx = len(self.global_path_np) - 1
        
        while end_index < max_idx and current_dist < self.ref_path_length:
            p1 = self.global_path_np[end_index]
            p2 = self.global_path_np[end_index + 1]
            dist = np.linalg.norm(p2 - p1)
            current_dist += dist
            end_index += 1
            
        # 确保至少截取两个点以便计算切线，且不超过数组边界
        end_index = min(end_index + 1, len(self.global_path_np))
        if end_index - min_index < 2:
            # 如果路径快走完了，这就可能发生，强制取最后一段
            end_index = len(self.global_path_np)
            min_index = max(0, end_index - 2)

        # 取出这段路径的坐标 [N, 2]
        local_path_segment = self.global_path_np[min_index:end_index]
        
        # 3. 计算这段路径的切线方向 (Yaw)
        # 使用差分计算：yaw[i] = atan2(y[i+1]-y[i], x[i+1]-x[i])
        dx = np.diff(local_path_segment[:, 0])
        dy = np.diff(local_path_segment[:, 1])
        yaws = np.arctan2(dy, dx)
        
        # 补齐最后一个点的航向（沿用倒数第二个点的方向）以保持维度一致
        yaws = np.append(yaws, yaws[-1]) 
        
        # 4. 组合成 Tensor [N, 3] -> [x, y, yaw]
        # device 会自动适配您的 CPU 环境
        ref_data = np.column_stack((local_path_segment, yaws))
        self.ref_path_tensor = torch.tensor(ref_data, dtype=torch.float32, device=self.device)
        
        # 5. 设置用于可视化的局部目标点 (红球)
        # 这里取截取片段的末端，仅仅为了让 Rviz 能够显示一个“目标”
        self.local_target = local_path_segment[-1]



    def cb_control_loop(self, event):
        """
        控制主循环
        作用：
        1. 更新局部目标点
        2. 调用 MPPI 求解最优动作
        3. 发布 Twist 消息
        """
        if self.global_path_np is None:
            # 还没收到路径，发布 0 速度
            # self.pub_cmd.publish(Twist()) 
            return

        # 1. 准备状态 Tensor
        state_tensor = torch.tensor(self.current_state, dtype=torch.float32, device=self.device)
        
        # 2. 更新参考路径
        self.update_local_target()
        
        # 3. 运行 MPPI 规划
        # command() 内部会调用 dynamics_model 进行推演，并用 trajectory_cost_function 打分
        # 返回最优动作 Tensor [v, w]
        action_tensor = self.mppi.command(state_tensor)

        #确认实际的维度
        # if hasattr(self.mppi, 'states') and self.mppi.states is not None:
        #     rospy.logdebug(f"MPPI states shape: {self.mppi.states.shape}")
        #     self.viz.visualize_candidate_trajectories(self.mppi.states, self.mppi.cost_total)

    ### ------------  可视化 ------------  ### 
        # 1. 获取并发布最优预测轨迹 (基于当前最优控制序列推演)
        # 输入: 当前状态; 输出: [1, T, nx] 的状态序列
        optimal_rollout = self.mppi.get_rollouts(state_tensor)
        self.viz.visualize_optimal_trajectory(optimal_rollout)
        
        # 2. 发布采样轨迹簇 (Top-K)
        # self.mppi.states: [K, T, nx] 存储了本次迭代所有采样轨迹的状态
        # self.mppi.cost_total: [K] 存储了对应的代价
        if hasattr(self.mppi, 'states') and self.mppi.states is not None:
            self.viz.visualize_candidate_trajectories(self.mppi.states, self.mppi.cost_total, num_vis=200)
            
        # 3. 发布局部目标点
        self.viz.visualize_local_target(self.local_target)
    ### ------------  可视化 ------------  ### 

        
        # 4. 转换为 ROS 消息并发布
        action_np = action_tensor.cpu().numpy()
        
        cmd = Twist()
        cmd.linear.x = float(action_np[0])
        cmd.angular.z = float(action_np[1])

        self.pub_cmd.publish(cmd)

if __name__ == '__main__':
    try:
        node = MPPIControllerNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass