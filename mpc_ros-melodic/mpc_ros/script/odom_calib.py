#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rospy
import tf
import math
import tf2_ros
import geometry_msgs.msg
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point, Pose, Quaternion, Twist, Vector3, TransformStamped

# ====================== 可修改参数 ======================
# 输入话题名称
INPUT_TOPIC = "/odom"
# 输出话题名称
OUTPUT_TOPIC = "/odom_calib"
# 校准角度（度）- Base_link相对于物理车头顺时针偏转的角度（正值表示顺时针）
CORRECTION_ANGLE_DEG =  1.8015-1.081 +0.05 # 1.801-6.66+2.577+3.58-3.33  #    1.8015
# 校准后的child_frame_id
OUTPUT_CHILD_FRAME_ID = "base_link_calib"
# TF父坐标系
TF_PARENT_FRAME = "world"
# TF子坐标系
TF_CHILD_FRAME = "base_link_calib"
# 是否发布TF变换
PUBLISH_TF = False   #0104：True
# 是否打印调试信息
DEBUG = False
# ======================================================

class OdometryCalibrator:
    def __init__(self):
        # 将角度转换为弧度（因为安装是顺时针偏，我们需要逆时针校正，所以取负）
        self.correction_angle_rad = -math.radians(CORRECTION_ANGLE_DEG)
        
        # 创建TF广播器
        self.tf_broadcaster = tf.TransformBroadcaster()
        
        # 创建发布器和订阅器
        self.pub = rospy.Publisher(OUTPUT_TOPIC, Odometry, queue_size=10)
        self.sub = rospy.Subscriber(INPUT_TOPIC, Odometry, self.odom_callback)
        
        rospy.loginfo("Odometry校准节点已启动")
        rospy.loginfo("订阅话题: %s", INPUT_TOPIC)
        rospy.loginfo("发布话题: %s", OUTPUT_TOPIC)
        rospy.loginfo("校准角度: %.4f度 (%.6f弧度)", 
                     CORRECTION_ANGLE_DEG, self.correction_angle_rad)
        rospy.loginfo("发布TF变换: %s -> %s", TF_PARENT_FRAME, TF_CHILD_FRAME)
        
    def odom_callback(self, msg):
        """
        处理/odom话题的回调函数
        """
        try:
            # 创建输出消息
            output_msg = Odometry()
            
            # 复制header信息
            output_msg.header = msg.header
            output_msg.header.frame_id = TF_PARENT_FRAME
            output_msg.child_frame_id = TF_CHILD_FRAME
            
            # 复制速度信息（不进行校正）
            output_msg.twist = msg.twist
            
            # 提取原始姿态
            orig_pose = msg.pose.pose
            
            # 如果原始odom的frame_id不是world，需要进行坐标变换
            if msg.header.frame_id != TF_PARENT_FRAME and msg.header.frame_id != "":
                rospy.logwarn("输入odom的frame_id为'%s'，但期望为'%s'。假设它们是同一坐标系。",
                             msg.header.frame_id, TF_PARENT_FRAME)
            
            # 提取原始四元数
            orig_orientation = [
                orig_pose.orientation.x,
                orig_pose.orientation.y,
                orig_pose.orientation.z,
                orig_pose.orientation.w
            ]
            
            # 将四元数转换为欧拉角 (roll, pitch, yaw)
            euler = tf.transformations.euler_from_quaternion(orig_orientation)
            
            if DEBUG:
                rospy.loginfo("原始姿态 - 位置: (%.3f, %.3f, %.3f), 偏航角: %.6f rad (%.6f deg)",
                             orig_pose.position.x, orig_pose.position.y, orig_pose.position.z,
                             euler[2], math.degrees(euler[2]))
            
            # 应用校准：将偏航角加上校准角度
            corrected_yaw = euler[2] + self.correction_angle_rad
            
            # 规范化角度到[-π, π]范围
            corrected_yaw = math.atan2(math.sin(corrected_yaw), math.cos(corrected_yaw))
            
            # 将欧拉角转换回四元数
            corrected_quaternion = tf.transformations.quaternion_from_euler(
                euler[0], euler[1], corrected_yaw
            )
            
            # 更新输出消息的位姿
            output_msg.pose.pose.position = orig_pose.position
            output_msg.pose.pose.orientation.x = corrected_quaternion[0]
            output_msg.pose.pose.orientation.y = corrected_quaternion[1]
            output_msg.pose.pose.orientation.z = corrected_quaternion[2]
            output_msg.pose.pose.orientation.w = corrected_quaternion[3]
            
            # 复制协方差矩阵（假设不需要校正）
            output_msg.pose.covariance = msg.pose.covariance
            output_msg.twist.covariance = msg.twist.covariance
            
            # 发布校准后的odom消息
            self.pub.publish(output_msg)
            
            if DEBUG:
                rospy.loginfo("校准后 - 偏航角: %.6f rad (%.6f deg)",
                             corrected_yaw, math.degrees(corrected_yaw))
                rospy.loginfo("发布校准后的odom数据到话题: %s", OUTPUT_TOPIC)
            
            # 发布TF变换
            if PUBLISH_TF:
                self.publish_tf_transform(
                    output_msg.header.stamp,
                    orig_pose.position,
                    corrected_quaternion
                )
                
        except Exception as e:
            rospy.logerr("处理odom消息时出错: %s", str(e))
    
    def publish_tf_transform(self, stamp, position, orientation):
        """
        发布TF变换
        """
        try:
            self.tf_broadcaster.sendTransform(
                translation=(position.x, position.y, position.z),
                rotation=(orientation[0], orientation[1], orientation[2], orientation[3]),
                time=stamp,
                child=TF_CHILD_FRAME,
                parent=TF_PARENT_FRAME
            )
            
            if DEBUG:
                rospy.loginfo("发布TF变换: %s -> %s", TF_PARENT_FRAME, TF_CHILD_FRAME)
                
        except Exception as e:
            rospy.logerr("发布TF变换时出错: %s", str(e))

def main():
    rospy.init_node('odometry_calibrator', anonymous=True)
    
    try:
        calibrator = OdometryCalibrator()
        rospy.loginfo("Odometry校准节点运行中...")
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("节点被中断")
    except Exception as e:
        rospy.logerr("节点运行出错: %s", str(e))

if __name__ == '__main__':
    main()