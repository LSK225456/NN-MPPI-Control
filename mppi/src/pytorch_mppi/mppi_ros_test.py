#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import torch
import numpy as np
import transformations as tf_trans
import math
import time  # 添加 time 模块

# ROS 消息类型
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry, Path

from mppi import MPPI

try:
    from mppi_viz import MPPIVisualizer 
except ImportError:
    from pytorch_mppi.mppi_viz import MPPIVisualizer

# 动力学模型接口
try:
    from dynamics_interface import create_dynamics_model, DynamicsModel
except ImportError:
    from pytorch_mppi.dynamics_interface import create_dynamics_model, DynamicsModel


class MPPIControllerNode:
    """
    MPPI 控制器 ROS 节点封装
    功能：接收 /odom 定位和全局路径，利用 MPPI 算法输出 /cmd_vel 控制指令
    支持：运动学模型 / 神经网络模型 动态切换
    """
    def __init__(self):
        # 初始化节点
        rospy.init_node('mppi_controller_node', anonymous=True)

        # --- 参数配置 (从 Parameter Server 获取 YAML 配置) ---
        
        # 话题名称配置
        self.topic_pose = rospy.get_param("/topics/pose", "/odom_calib")
        self.topic_twist = rospy.get_param("/topics/twist", "/wheel_odom")
        self.topic_cmd = rospy.get_param("/topics/cmd", "/cmd_vel")
        self.topic_path = rospy.get_param("/topics/path", "/global_plan16")

        # 坐标变换参数配置
        self.is_odom_transform = rospy.get_param("/transform/is_odom_transform", False)
        self.laser2base_dist = rospy.get_param("/transform/laser2base_dist", 0.38)
        self.laser2base_angle = rospy.get_param("/transform/laser2base_angle", 0.0593)
        
        # MPPI 算法超参数
        self.horizon = rospy.get_param("/mppi/horizon", 30)
        self.num_samples = rospy.get_param("/mppi/num_samples", 500)
        self.lambda_val = rospy.get_param("/mppi/lambda", 1.0)
        self.noise_sigma = rospy.get_param("/mppi/noise_sigma", [0.5, 0.6])
        self.control_freq = rospy.get_param("/mppi/control_freq", 20.0)
        
        # 路径跟踪参数
        self.lookahead_dist = rospy.get_param("/path_tracking/lookahead_dist", 5.0)
        self.goal_tolerance = rospy.get_param("/path_tracking/goal_tolerance", 0.2)
        self.ref_path_length = rospy.get_param("/path_tracking/ref_path_length", 5.0)
        
        # 代价权重
        self.w_cte = rospy.get_param("/cost_weights/w_cte", 100.0)
        self.w_yaw = rospy.get_param("/cost_weights/w_yaw", 5.0)
        self.w_vel = rospy.get_param("/cost_weights/w_vel", 0.5)
        self.target_velocity = rospy.get_param("/cost_weights/target_velocity", 0.5)
        
        # 控制量限制
        v_min = rospy.get_param("/control_limits/v_min", -0.4)
        v_max = rospy.get_param("/control_limits/v_max", 0.5)
        w_min = rospy.get_param("/control_limits/w_min", -0.4)
        w_max = rospy.get_param("/control_limits/w_max", 0.4)

        # ====== 神经网络动力学模型参数 ======
        self.use_neural_dynamics = rospy.get_param("/dynamics/use_neural", False)
        self.neural_model_path = rospy.get_param("/dynamics/model_path", "")
        self.neural_scaler_x_path = rospy.get_param("/dynamics/scaler_x_path", "")
        self.neural_scaler_y_path = rospy.get_param("/dynamics/scaler_y_path", "")
        
        # ====== PyTorch 后端配置 ======
        self.dynamics_backend = rospy.get_param("/dynamics/backend", "pytorch")
        self.dynamics_device = rospy.get_param("/dynamics/device", "cpu")
        self.pytorch_model_path = rospy.get_param("/dynamics/pytorch_model_path", "")
        
        # ====== 验证神经网络参数 ======
        if self.use_neural_dynamics:
            nn_dt = 0.05
            control_dt = 1.0 / self.control_freq
            if abs(control_dt - nn_dt) > 0.001:
                rospy.logwarn(f"控制频率 dt={control_dt:.4f}s 与神经网络 dt={nn_dt}s 不匹配!")
                rospy.logwarn(f"建议将 control_freq 设置为 {1.0/nn_dt:.1f} Hz")

        # 设备选择 (GPU/CPU)
        self.device = "cuda" if 0 else "cpu"
        rospy.loginfo(f"MPPI 控制器运行设备: {self.device}")

        # --- 机器人状态初始化 ---
        # 状态向量: [x, y, θ, v, w]
        self.current_state = np.zeros(5, dtype=np.float32)
        
        # 上一帧速度（用于计算加速度）
        self.last_velocity = np.zeros(2, dtype=np.float32)  # [v, w]
        
        # 上一帧控制量
        self.last_control = np.zeros(2, dtype=np.float32)  # [cmd_v, cmd_w]
        
        # 路径存储
        self.global_path_np = None
        self.local_target = None
        
        # 初始化标志
        self._initialized = False
        self._pose_received = False
        self._twist_received = False
        
        # ====== 推理性能统计 ======
        self.inference_times = []  # 记录推理耗时
        self.max_history = 100     # 最多保存最近 100 次
        
        # ====== 初始化动力学模型（使用工厂模式） ======
        dynamics_config = {
            'use_neural': self.use_neural_dynamics,
            'backend': self.dynamics_backend,
            'device': self.dynamics_device,
            'model_path': self.neural_model_path,
            'pytorch_model_path': self.pytorch_model_path,
            'scaler_x_path': self.neural_scaler_x_path,
            'scaler_y_path': self.neural_scaler_y_path
        }
        # 传入 horizon 参数以支持批量 Rollout 后端
        self.dynamics_model = create_dynamics_model(
            dynamics_config, 
            self.num_samples,
            horizon=self.horizon
        )
        
        # 创建 MPPI 兼容的动力学包装函数
        dynamics_func = self._create_dynamics_wrapper()

        # --- MPPI 核心初始化 ---
        noise_sigma_tensor = torch.tensor(
            [[self.noise_sigma[0], 0.0], [0.0, self.noise_sigma[1]]], 
            device=self.device, dtype=torch.float32
        )
        
        u_min = torch.tensor([v_min, w_min], device=self.device)
        u_max = torch.tensor([v_max, w_max], device=self.device)

        self.mppi = MPPI(
            dynamics=dynamics_func,
            running_cost=self.trajectory_cost_function,
            nx=5,
            noise_sigma=noise_sigma_tensor,
            num_samples=self.num_samples,
            horizon=self.horizon,
            device=self.device,
            lambda_=self.lambda_val,
            u_min=u_min,
            u_max=u_max
        )

        self.viz = MPPIVisualizer(frame_id="map")

        # --- ROS 通信接口 ---
        self.sub_pose = rospy.Subscriber(self.topic_pose, Odometry, self.cb_pose)
        self.sub_twist = rospy.Subscriber(self.topic_twist, Odometry, self.cb_twist)
        self.sub_path = rospy.Subscriber(self.topic_path, Path, self.cb_path)
        self.pub_cmd = rospy.Publisher(self.topic_cmd, Twist, queue_size=1)

        self.timer = rospy.Timer(rospy.Duration(1.0/self.control_freq), self.cb_control_loop)
        
        rospy.loginfo("MPPI 控制节点已启动，等待路径...")

    def _create_dynamics_wrapper(self):
        """
        创建 MPPI 兼容的动力学函数包装器
        
        Returns:
            dynamics_func: PyTorch Tensor 接口的动力学函数
        """
        dynamics_model = self.dynamics_model
        num_samples = self.num_samples
        
        def dynamics_func(state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
            """
            动力学函数（PyTorch Tensor 接口）
            
            Args:
                state: [K, 5] 或 [M, K, 5]
                action: [K, 2] 或 [M, K, 2]
            
            Returns:
                next_state: 形状与输入 state 相同
            """
            original_shape = action.shape
            
            # 处理可能的额外维度 (M > 1)
            if action.dim() == 3:
                action_2d = action[0]
                state_2d = state[0] if state.dim() == 3 else state
            else:
                action_2d = action
                state_2d = state
            
            # 检查粒子数是否匹配
            K = action_2d.shape[0]
            if K != num_samples:
                # 粒子数不匹配时回退到运动学模型
                return self._kinematic_fallback(state, action)
            
            # 转换为 numpy 并调用动力学模型
            state_np = state_2d.cpu().numpy()
            action_np = action_2d.cpu().numpy()
            next_state_np = dynamics_model.step(state_np, action_np)
            
            # 转回 Tensor
            next_state = torch.tensor(
                next_state_np, 
                dtype=state.dtype, 
                device=state.device
            )
            
            # 恢复原始形状
            if len(original_shape) == 3:
                next_state = next_state.unsqueeze(0)
            
            return next_state
        
        return dynamics_func

    def _kinematic_fallback(self, state, action):
        """运动学模型回退（用于粒子数不匹配的情况）"""
        dt = 1.0 / self.control_freq
        
        x = state[..., 0]
        y = state[..., 1]
        theta = state[..., 2]
        v_curr = state[..., 3]
        w_curr = state[..., 4]
        
        v_cmd = action[..., 0]
        w_cmd = action[..., 1]
        
        tau_v, tau_w = 0.1, 0.05
        alpha_v = dt / (tau_v + dt)
        alpha_w = dt / (tau_w + dt)
        
        v_next = v_curr + alpha_v * (v_cmd - v_curr)
        w_next = w_curr + alpha_w * (w_cmd - w_curr)
        
        new_x = x + v_next * torch.cos(theta) * dt
        new_y = y + v_next * torch.sin(theta) * dt
        new_theta = theta + w_next * dt
        new_theta = torch.atan2(torch.sin(new_theta), torch.cos(new_theta))
        
        return torch.stack([new_x, new_y, new_theta, v_next, w_next], dim=-1)

    def trajectory_cost_function(self, state, action):
        """并行轨迹代价函数"""
        if not hasattr(self, 'ref_path_tensor') or self.ref_path_tensor is None:
            return torch.zeros(state.shape[0], device=self.device)
            
        if not torch.is_tensor(state):
            state = torch.tensor(state, dtype=torch.float32, device=self.device)
            
        state_pos = state[:, :2].unsqueeze(1)
        ref_pos = self.ref_path_tensor[:, :2].unsqueeze(0)
        
        diff = state_pos - ref_pos
        dists_sq = torch.sum(diff**2, dim=2)
        min_dists_sq, min_indices = torch.min(dists_sq, dim=1)
        
        ref_yaws = self.ref_path_tensor[min_indices, 2]
        yaw_error = state[:, 2] - ref_yaws
        yaw_error = torch.atan2(torch.sin(yaw_error), torch.cos(yaw_error))
        
        cost_cte = self.w_cte * min_dists_sq
        cost_yaw = self.w_yaw * (yaw_error**2)
        cost_vel = self.w_vel * ((action[:, 0] - self.target_velocity)**2)
        
        return cost_cte + cost_yaw + cost_vel

    def cb_pose(self, msg):
        """位姿回调"""
        try:
            p = msg.pose.pose.position
            o = msg.pose.pose.orientation
            
            quaternion = (o.x, o.y, o.z, o.w)
            euler = tf_trans.euler_from_quaternion(quaternion)
            yaw = euler[2]

            x_curr, y_curr = p.x, p.y

            if self.is_odom_transform:
                yaw += self.laser2base_angle
                x_curr -= self.laser2base_dist * math.cos(yaw)
                y_curr -= self.laser2base_dist * math.sin(yaw)
            
            self.current_state[0] = x_curr
            self.current_state[1] = y_curr
            self.current_state[2] = yaw
            
            if not self._pose_received:
                self._pose_received = True
                rospy.loginfo("首次收到位姿数据")
            
        except Exception as e:
            rospy.logerr(f"处理 Odom 数据出错: {e}")

    def cb_twist(self, msg):
        """速度回调"""
        v = msg.twist.twist.linear.x
        w = msg.twist.twist.angular.z
        
        self.current_state[3] = float(v)
        self.current_state[4] = float(w)
        
        if not self._twist_received:
            self._twist_received = True
            rospy.loginfo("首次收到速度数据")

    def cb_path(self, msg):
        """路径回调"""
        if not msg.poses:
            return
        path_list = [[ps.pose.position.x, ps.pose.position.y] for ps in msg.poses]
        self.global_path_np = np.array(path_list)
        rospy.loginfo(f"收到新路径，包含 {len(path_list)} 个点")

    def update_local_target(self):
        """更新局部参考路径"""
        if self.global_path_np is None or len(self.global_path_np) == 0:
            self.ref_path_tensor = None
            self.local_target = None
            return

        diff = self.global_path_np - self.current_state[:2]
        dists_sq = np.sum(diff**2, axis=1)
        min_index = np.argmin(dists_sq)
        
        current_dist = 0.0
        end_index = min_index
        max_idx = len(self.global_path_np) - 1
        
        while end_index < max_idx and current_dist < self.ref_path_length:
            p1 = self.global_path_np[end_index]
            p2 = self.global_path_np[end_index + 1]
            current_dist += np.linalg.norm(p2 - p1)
            end_index += 1
            
        end_index = min(end_index + 1, len(self.global_path_np))
        if end_index - min_index < 2:
            end_index = len(self.global_path_np)
            min_index = max(0, end_index - 2)

        local_path_segment = self.global_path_np[min_index:end_index]
        
        dx = np.diff(local_path_segment[:, 0])
        dy = np.diff(local_path_segment[:, 1])
        yaws = np.arctan2(dy, dx)
        yaws = np.append(yaws, yaws[-1])
        
        ref_data = np.column_stack((local_path_segment, yaws))
        self.ref_path_tensor = torch.tensor(ref_data, dtype=torch.float32, device=self.device)
        self.local_target = local_path_segment[-1]

    def cb_control_loop(self, event):
        """控制主循环"""
        if self.global_path_np is None:
            return
        
        if not self._pose_received:
            rospy.logwarn_throttle(5.0, "等待位姿数据...")
            return
            
        if not self._twist_received:
            rospy.logwarn_throttle(5.0, "等待速度数据...")
            return

        state_tensor = torch.tensor(self.current_state, dtype=torch.float32, device=self.device)
        
        self.update_local_target()
        
        # 在 MPPI 优化前重置动力学模型状态
        self.dynamics_model.reset(self.current_state)
        
        # ====== 计时开始 ======
        start_time = time.perf_counter()
        
        # 运行 MPPI 规划
        action_tensor = self.mppi.command(state_tensor)
        
        # ====== 计时结束 ======
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        
        # 记录并打印耗时
        self.inference_times.append(elapsed_ms)
        if len(self.inference_times) > self.max_history:
            self.inference_times.pop(0)
        
        avg_time = np.mean(self.inference_times)
        rospy.loginfo(
            "[MPPI推理] 本次耗时: {:.1f}ms | 平均耗时: {:.1f}ms | "
            "粒子数: {} | horizon: {}".format(
                elapsed_ms, avg_time, self.num_samples, self.horizon
            )
        )

        # 可视化
        optimal_rollout = self.mppi.get_rollouts(state_tensor)
        self.viz.visualize_optimal_trajectory(optimal_rollout)
        
        if hasattr(self.mppi, 'states') and self.mppi.states is not None:
            self.viz.visualize_candidate_trajectories(self.mppi.states, self.mppi.cost_total, num_vis=200)
            
        self.viz.visualize_local_target(self.local_target)
        
        # 发布控制指令
        action_np = action_tensor.cpu().numpy()
        
        cmd = Twist()
        cmd.linear.x = float(action_np[0])
        cmd.angular.z = float(action_np[1])
        self.pub_cmd.publish(cmd)
        
        # 更新真实历史缓冲区
        self._update_neural_history(action_np)
        
        # 保存当前状态用于下次计算
        self.last_control[0] = float(action_np[0])
        self.last_control[1] = float(action_np[1])
        self.last_velocity[0] = self.current_state[3]
        self.last_velocity[1] = self.current_state[4]
    
    def _update_neural_history(self, action_np):
        """
        更新神经网络的真实历史缓冲区
        
        特征顺序: [v, w, dt, cmd_v, cmd_w, a_v, a_w, err_v, err_w, v_x_w]
        """
        dt = 1.0 / self.control_freq
        
        # 当前速度
        v = self.current_state[3]
        w = self.current_state[4]
        
        # 控制量
        cmd_v = float(action_np[0])
        cmd_w = float(action_np[1])
        
        # 计算加速度
        a_v = (v - self.last_velocity[0]) / dt
        a_w = (w - self.last_velocity[1]) / dt
        
        # 限制异常值
        a_v = float(np.clip(a_v, -20.0, 20.0))
        a_w = float(np.clip(a_w, -20.0, 20.0))
        
        # 计算误差
        err_v = cmd_v - v
        err_w = cmd_w - w
        
        # 耦合项
        v_x_w = v * w
        
        # 更新真实历史
        self.dynamics_model.update_real_history(
            float(v), float(w),
            cmd_v, cmd_w,
            a_v, a_w,
            float(err_v), float(err_w),
            float(v_x_w)
        )


if __name__ == '__main__':
    try:
        node = MPPIControllerNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass