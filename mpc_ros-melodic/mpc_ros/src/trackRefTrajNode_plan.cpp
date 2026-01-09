/*
# Copyright 2018 HyphaROS Workshop.
# Developer: HaoChih, LIN (hypha.ros@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
*/

#include <iostream>
#include <map>
#include <math.h>

#include "ros/ros.h"
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Twist.h>
#include <tf/transform_listener.h>
#include <std_msgs/Float32.h>

// #include <tf/transform_datatypes.h>
#include <nav_msgs/Path.h>
#include <nav_msgs/Odometry.h>
//#include <ackermann_msgs/AckermannDriveStamped.h>
#include <visualization_msgs/Marker.h>

#include "trackRefTraj.h"
#include <Eigen/Core>
#include <Eigen/QR>


// inlcude iostream and string libraries
#include <iostream>
#include <fstream>
#include <string>


using namespace std;
using namespace Eigen;


/********************/
/* CLASS DEFINITION */
/********************/
class MPCNode
{
    public:
        MPCNode();
        ~MPCNode();
        int get_thread_numbers();
        
    private:
        ros::NodeHandle _nh;
        ros::Subscriber _sub_odom, _sub_gen_path, _sub_path, _sub_goal, _sub_amcl;
        ros::Subscriber _sub_wheel_odom; // 1104 订阅这个话题的速度信息
        nav_msgs::Odometry _wheel_odom; // 1104 新增轮式里程计数据存储
        ros::Subscriber _sub_odom_path; // 1105 新增：用于历史轨迹记录的odom订阅者
        ros::Publisher _pub_odom_path_history; // 1105 新增：历史轨迹发布者
        nav_msgs::Path _odom_path_history; // 1105 新增：存储历史轨迹
        ros::Publisher _pub_cte_error; //1112 新增：横向误差发布者
        std_msgs::Float32 _cte_error_msg; //1112 新增：横向误差消息     

        double _laser2base; // 1106：laser2base坐标变换，雷达在车体前方X方向的距离

        unsigned int _odom_count;
        ros::Publisher _pub_totalcost, _pub_ctecost, _pub_ethetacost,_pub_odompath, _pub_twist, _pub_ackermann, _pub_mpctraj;
        ros::Timer _timer1;
        tf::TransformListener _tf_listener;

        geometry_msgs::Point _goal_pos;
        nav_msgs::Odometry _odom;
        nav_msgs::Path _odom_path, _mpc_traj; 
	//ackermann_msgs::AckermannDriveStamped _ackermann_msg;
        geometry_msgs::Twist _twist_msg;

        // string _globalPath_topic, _goal_topic;
        // string _map_frame, _odom_frame, _car_frame;

        string _topic_global_path, _topic_goal, _topic_odom; // 新增
        string _planning_frame, _car_frame;                  // 新增: _planning_frame 统一替代 map/odom frame

        MPC _mpc;
        map<string, double> _mpc_params;
        double _mpc_steps, _ref_cte, _ref_etheta, _ref_vel, _w_cte, _w_etheta, _w_vel, 
               _w_angvel, _w_accel, _w_angvel_d, _w_accel_d, _max_angvel, _max_throttle, _bound_value;
        double _heading_offset_deg; // 12.17 新增：存储角度偏差参数

        double _w_cte_int;  //1026
        double _integ_decay, _integ_max, _integ_v_thresh;
        double _prev_cte;   // 用于过零检测，存储上一时刻的cte
        double _min_ref_vel, _v_ref_cte_k, _v_ref_etheta_k;     //新增自适应参考速度参数变量

        //double _Lf; 
        double _dt, _w, _throttle, _speed, _max_speed;
        double _pathLength, _goalRadius, _waypointsDist;
        int _controller_freq, _downSampling, _thread_numbers;
        bool _goal_received, _goal_reached, _path_computed, _pub_twist_flag, _debug_info, _delay_mode;

        double polyeval(Eigen::VectorXd coeffs, double x);
        Eigen::VectorXd polyfit(Eigen::VectorXd xvals, Eigen::VectorXd yvals, int order);

        void odomCB(const nav_msgs::Odometry::ConstPtr& odomMsg);
        void pathCB(const nav_msgs::Path::ConstPtr& pathMsg);
        void desiredPathCB(const nav_msgs::Path::ConstPtr& pathMsg);
        void goalCB(const geometry_msgs::PoseStamped::ConstPtr& goalMsg);
        void amclCB(const geometry_msgs::PoseWithCovarianceStamped::ConstPtr& amclMsg);
        void wheelOdomCB(const nav_msgs::Odometry::ConstPtr& wheelOdomMsg); // 1104
        void controlLoopCB(const ros::TimerEvent&);
        void odomPathCB(const nav_msgs::Odometry::ConstPtr& odomMsg); // 1105 新增：历史轨迹回调函数
        void goalReachedJudge();
        //For making global planner
        nav_msgs::Path _gen_path;
        unsigned int min_idx;
        
        double _mpc_etheta;
        double _mpc_cte;
        fstream file;
        unsigned int idx;
        double _cte_int;  //1026
}; // end of class


MPCNode::MPCNode()
{
    //Private parameters handler
    ros::NodeHandle pn("~");

    //Parameters for control loop
    pn.param("thread_numbers", _thread_numbers, 2); // number of threads for this ROS node
    pn.param("pub_twist_cmd", _pub_twist_flag, true);
    pn.param("debug_info", _debug_info, true);
    pn.param("delay_mode", _delay_mode, true);
    pn.param("max_speed", _max_speed, 0.50); // unit: m/s
    pn.param("waypoints_dist", _waypointsDist, -1.0); // unit: m
    pn.param("path_length", _pathLength, 2.0); // unit: m
    pn.param("goal_radius", _goalRadius, 0.5); // unit: m
    pn.param("controller_freq", _controller_freq, 10);
    //pn.param("vehicle_Lf", _Lf, 0.290); // distance between the front of the vehicle and its center of gravity
    // _dt = double(1.0/_controller_freq); // time step duration dt in s 
    _dt = 0.1;
    
    //Parameter for MPC solver
    pn.param("mpc_steps", _mpc_steps, 20.0);
    pn.param("mpc_ref_cte", _ref_cte, 0.0);
    pn.param("mpc_ref_vel", _ref_vel, 1.0);
    pn.param("mpc_ref_etheta", _ref_etheta, 0.0);
    pn.param("mpc_w_cte", _w_cte, 5000.0);
    pn.param("mpc_w_etheta", _w_etheta, 5000.0);
    pn.param("mpc_w_vel", _w_vel, 1.0);
    pn.param("mpc_w_angvel", _w_angvel, 100.0);
    pn.param("mpc_w_angvel_d", _w_angvel_d, 10.0);
    pn.param("mpc_w_accel", _w_accel, 50.0);
    pn.param("mpc_w_accel_d", _w_accel_d, 10.0);
    pn.param("heading_offset_deg", _heading_offset_deg, 0.0);

    pn.param("mpc_w_cte_int", _w_cte_int, 200.0);   //1026:参数读取
    pn.param("mpc_integ_decay", _integ_decay, 0.95);
    pn.param("mpc_integ_max", _integ_max, 2.0);
    pn.param("mpc_integ_v_thresh", _integ_v_thresh, 0.1);

    pn.param("mpc_min_ref_vel", _min_ref_vel, 0.1);
    pn.param("mpc_v_ref_cte_k", _v_ref_cte_k, 1.5);       // 对应公式中的横向误差系数
    pn.param("mpc_v_ref_etheta_k", _v_ref_etheta_k, 1.0); // 对应公式中的角度误差系数

    pn.param("mpc_max_angvel", _max_angvel, 3.0); // Maximal angvel radian (~30 deg)
    pn.param("mpc_max_throttle", _max_throttle, 1.0); // Maximal throttle accel
    pn.param("mpc_bound_value", _bound_value, 1.0e3); // Bound value for other variables
    pn.param("laser2base", _laser2base, 0.38);           // 1106：laser2base坐标变换

    //Parameter for topics & Frame name
    // pn.param<std::string>("global_path_topic", _globalPath_topic, "/move_base/TrajectoryPlannerROS/global_plan" );
    // pn.param<std::string>("goal_topic", _goal_topic, "/move_base_simple/goal" );
    // pn.param<std::string>("map_frame", _map_frame, "map" ); //*****for mpc, "odom"
    // pn.param<std::string>("odom_frame", _odom_frame, "map");
    // 加载话题和坐标系参数
    pn.param<std::string>("topic_global_path", _topic_global_path, "/desired_path"); 
    pn.param<std::string>("topic_goal", _topic_goal, "/move_base_simple/goal");
    pn.param<std::string>("topic_odom", _topic_odom, "/odom");
    pn.param<std::string>("planning_frame", _planning_frame, "map"); // 统一的规划坐标系
    pn.param<std::string>("car_frame", _car_frame, "base_link" );

    //Display the parameters
    cout << "\n===== Parameters =====" << endl;
    cout << "pub_twist_cmd: "  << _pub_twist_flag << endl;
    cout << "debug_info: "  << _debug_info << endl;
    cout << "delay_mode: "  << _delay_mode << endl;
    //cout << "vehicle_Lf: "  << _Lf << endl;
    cout << "frequency: "   << _dt << endl;
    cout << "mpc_steps: "   << _mpc_steps << endl;
    cout << "mpc_ref_vel: " << _ref_vel << endl;
    cout << "mpc_w_cte: "   << _w_cte << endl;
    cout << "mpc_w_etheta: "  << _w_etheta << endl;
    cout << "mpc_max_angvel: "  << _max_angvel << endl;

    //Publishers and Subscribers
    _sub_odom   = _nh.subscribe(_topic_odom, 1, &MPCNode::odomCB, this);        //todo:之前是/odom
    _sub_wheel_odom = _nh.subscribe("/wheel_odom", 1, &MPCNode::wheelOdomCB, this); // 1104 新增轮式里程计订阅来获取速度
   
    _sub_gen_path   = _nh.subscribe(_topic_global_path, 1, &MPCNode::desiredPathCB, this);//todo16
    // _sub_goal   = _nh.subscribe(_goal_topic, 1, &MPCNode::goalCB, this);
    _sub_amcl   = _nh.subscribe("/amcl_pose", 5, &MPCNode::amclCB, this);
    _sub_odom_path = _nh.subscribe("/odom", 1, &MPCNode::odomPathCB, this); // 1105 新增：历史轨迹记录的odom订阅
    
    _pub_odompath  = _nh.advertise<nav_msgs::Path>("/mpc_reference", 1); // reference path for MPC ///mpc_reference 
    _pub_mpctraj   = _nh.advertise<nav_msgs::Path>("/mpc_trajectory", 1);// MPC trajectory output
    _pub_cte_error = _nh.advertise<std_msgs::Float32>("/cte_error", 1); //1112 新增：初始化横向误差发布者
    //_pub_ackermann = _nh.advertise<ackermann_msgs::AckermannDriveStamped>("/ackermann_cmd", 1);
    if(_pub_twist_flag)
        _pub_twist = _nh.advertise<geometry_msgs::Twist>("/cmd_vel", 1); //for stage (Ackermann msg non-supported)
    
    _pub_totalcost  = _nh.advertise<std_msgs::Float32>("/total_cost", 1); // Global path generated from another source
    _pub_ctecost  = _nh.advertise<std_msgs::Float32>("/cross_track_error", 1); // Global path generated from another source
    _pub_ethetacost  = _nh.advertise<std_msgs::Float32>("/theta_error", 1); // Global path generated from another source
    _pub_odom_path_history = _nh.advertise<nav_msgs::Path>("/recorded_path", 10); // 1105 新增：历史轨迹发布

    //Timer
    _timer1 = _nh.createTimer(ros::Duration((1.0)/_controller_freq), &MPCNode::controlLoopCB, this); // 10Hz //*****mpc

    //Init variables
    _goal_received = false;
    _goal_reached  = false;
    _path_computed = false;
    _throttle = 0.0; 
    _w = 0.0;
    _speed = 0.0;
    _wheel_odom = nav_msgs::Odometry();

    //_ackermann_msg = ackermann_msgs::AckermannDriveStamped();
    _twist_msg = geometry_msgs::Twist();
    _mpc_traj = nav_msgs::Path();

    //Init parameters for MPC object
    _mpc_params["DT"] = _dt;
    //_mpc_params["LF"] = _Lf;
    _mpc_params["STEPS"]    = _mpc_steps;
    _mpc_params["REF_CTE"]  = _ref_cte;
    _mpc_params["REF_ETHETA"] = _ref_etheta;
    _mpc_params["REF_V"]    = _ref_vel;
    _mpc_params["W_CTE"]    = _w_cte;
    _mpc_params["W_EPSI"]   = _w_etheta;
    _mpc_params["W_V"]      = _w_vel;
    _mpc_params["W_ANGVEL"]  = _w_angvel;
    _mpc_params["W_A"]      = _w_accel;
    _mpc_params["W_DANGVEL"] = _w_angvel_d;
    _mpc_params["W_DA"]     = _w_accel_d;
    _mpc_params["W_CTE_INT"] = _w_cte_int;  //1026

    _mpc_params["ANGVEL"]   = _max_angvel;
    _mpc_params["MAXTHR"]   = _max_throttle;
    _mpc_params["BOUND"]    = _bound_value;
    _mpc.LoadParams(_mpc_params);

    _odom_count = 0;
    _odom_path_history = nav_msgs::Path();
    _odom_path_history.header.frame_id = _planning_frame;

    min_idx = 0;
    idx = 0;
    _mpc_etheta = 0;
    _mpc_cte = 0;
    _cte_int = 0.0;  //1026
    _prev_cte = 0.0; 

    cout<< "ros::Time::now().toSec()  "<<ros::Time::now().toNSec() << endl;
}

MPCNode::~MPCNode()
{
    file.close();
    
};


void MPCNode::odomPathCB(const nav_msgs::Odometry::ConstPtr& odomMsg)
{
    _odom_count += 1;
    if (_odom_count % 3 != 0) {
        return;
    }

    try {
        // 使用转换后的 _odom 数据
        geometry_msgs::PoseStamped base_pose;
        base_pose.header = _odom.header;
        base_pose.pose = _odom.pose.pose;

        // 如果需要转换到其他坐标系，可以在这里进行
        // 否则直接使用 base_link 在 map 下的坐标
        _odom_path_history.header.stamp = ros::Time::now();
        _odom_path_history.header.frame_id = _planning_frame; // 使用 map 坐标系
        _odom_path_history.poses.push_back(base_pose);
        
        _pub_odom_path_history.publish(_odom_path_history);

        if(_gen_path.poses.size() > 0) // 确保有全局路径
        {
            double min_cte = std::numeric_limits<double>::max();
            const double px = base_pose.pose.position.x;
            const double py = base_pose.pose.position.y;
            
            // 遍历全局路径点，找到最小距离作为横向误差
            for(int i = 0; i < _gen_path.poses.size(); i++) 
            {
                geometry_msgs::PoseStamped path_pose;
                // 转换到同一坐标系
                _tf_listener.transformPose(_planning_frame, ros::Time(0), 
                                         _gen_path.poses[i], _gen_path.header.frame_id, path_pose);
                
                double dx = path_pose.pose.position.x - px;
                double dy = path_pose.pose.position.y - py;
                double current_dist = sqrt(dx*dx + dy*dy);
                
                if(current_dist < min_cte)
                {
                    min_cte = current_dist;
                }
            }
            
            // 发布横向误差
            _cte_error_msg.data = static_cast<float>(min_cte);
            _pub_cte_error.publish(_cte_error_msg);
        }

    } catch (tf::TransformException &ex) {
        ROS_WARN("Failed to process path history: %s", ex.what());
        return;
    }
}

void MPCNode::wheelOdomCB(const nav_msgs::Odometry::ConstPtr& wheelOdomMsg)
{
    _wheel_odom = *wheelOdomMsg;
}

// Public: return _thread_numbers
int MPCNode::get_thread_numbers()
{
    return _thread_numbers;
}


// Evaluate a polynomial.
double MPCNode::polyeval(Eigen::VectorXd coeffs, double x) 
{
    double result = 0.0;
    for (int i = 0; i < coeffs.size(); i++) 
    {
        result += coeffs[i] * pow(x, i);
    }
    return result;
}


// Fit a polynomial.
// Adapted from
// https://github.com/JuliaMath/Polynomials.jl/blob/master/src/Polynomials.jl#L676-L716
Eigen::VectorXd MPCNode::polyfit(Eigen::VectorXd xvals, Eigen::VectorXd yvals, int order) 
{
    assert(xvals.size() == yvals.size());
    assert(order >= 1 && order <= xvals.size() - 1);
    Eigen::MatrixXd A(xvals.size(), order + 1);

    for (int i = 0; i < xvals.size(); i++)
        A(i, 0) = 1.0;

    for (int j = 0; j < xvals.size(); j++) 
    {
        for (int i = 0; i < order; i++) 
            A(j, i + 1) = A(j, i) * xvals(j);
    }

    auto Q = A.householderQr();
    auto result = Q.solve(yvals);
    return result;
}

void MPCNode::odomCB(const nav_msgs::Odometry::ConstPtr& odomMsg)
{
    /// ----------------------  室内定位下的坐标系补偿/微调 -------------------  ///
    // // 直接应用静态变换计算
    // nav_msgs::Odometry transformed_odom = *odomMsg;
    
    // // 获取当前位姿的方向
    // tf::Pose pose;
    // tf::poseMsgToTF(odomMsg->pose.pose, pose);
    // double yaw = tf::getYaw(pose.getRotation())+0/180*M_PI;       //TODO1112
    
    // // 1106：laser2base坐标变换
    // transformed_odom.pose.pose.position.x -= _laser2base * cos(yaw);
    // transformed_odom.pose.pose.position.y -= _laser2base * sin(yaw);

    // transformed_odom.child_frame_id = _car_frame;
    
    // _odom = transformed_odom;
    /// ----------------------  室内定位下的坐标系补偿/微调 -------------------  ///


    _odom = *odomMsg;

    // /// ----------------------  室外gps定位下的坐标系补偿/微调 -------------------  ///
    // // 12.17 修改：在接收 odom 数据时直接补偿角度偏差
    // // 获取原始的 Yaw 角
    // double raw_yaw = tf::getYaw(odomMsg->pose.pose.orientation);
    
    // // 计算修正后的 Yaw (Raw - Offset)。
    // // 原理：如果 baselink 物理上偏左(正值)，odom读数会偏大，我们需要减去偏差值让它回归0(正前方)
    // double corrected_yaw = raw_yaw - (_heading_offset_deg * M_PI / 180.0);

    // // 将修正后的数据存入 _odom
    // _odom = *odomMsg;
    // // 重写 orientation 四元数
    // _odom.pose.pose.orientation = tf::createQuaternionMsgFromYaw(corrected_yaw);
    //     /// ----------------------  室外gps定位下的坐标系补偿/微调 -------------------  ///

}



// CallBack: Update generated path (conversion to odom frame)
void MPCNode::desiredPathCB(const nav_msgs::Path::ConstPtr& totalPathMsg)
{
    _gen_path = *totalPathMsg;
    
    _goal_received = true;
    _goal_reached = false;
    nav_msgs::Path mpc_path = nav_msgs::Path();   // For generating mpc reference path  
    geometry_msgs::PoseStamped tempPose;
    nav_msgs::Odometry odom = _odom; 

    try
    {
        double total_length = 0.0;
        //find waypoints distance 认为轨迹waypoints均匀
        // 计算相邻轨迹点的距离
        if(_waypointsDist <= 0.0)
        {        
            double gap_x = totalPathMsg->poses[1].pose.position.x - totalPathMsg->poses[0].pose.position.x;
            double gap_y = totalPathMsg->poses[1].pose.position.y - totalPathMsg->poses[0].pose.position.y;
            _waypointsDist = sqrt(gap_x*gap_x + gap_y*gap_y);             
        }                       

        // Find the nearst point for robot position
        double min_val_in_direction = std::numeric_limits<double>::max();
        int min_idx_in_direction = -1; 
        
        // 从里程计获取车辆的线速度和角速度信息
        const double linear_velocity = _wheel_odom.twist.twist.linear.x;
        tf::Pose pose;
        tf::poseMsgToTF(odom.pose.pose, pose);
        const double theta = tf::getYaw(pose.getRotation());
        
        // 定义一个表示车辆行进方向的向量
        double travel_direction_x = cos(theta);
        double travel_direction_y = sin(theta);

        // 判断车辆是在后退还是前进
        // 如果线速度为负（小于一个小的负阈值），则认为车辆在后退
        if (linear_velocity < -0.05) 
        {
            // 后退时，行进方向与车头方向相反
            travel_direction_x = -travel_direction_x;
            travel_direction_y = -travel_direction_y;
        }


        // int min_val = 100; 
        int N = totalPathMsg->poses.size(); // Number of waypoints        
        _goal_pos = totalPathMsg->poses[N-1].pose.position;
 
        const double px = odom.pose.pose.position.x; //pose: odom frame
        const double py = odom.pose.pose.position.y;
        const double ptheta = odom.pose.pose.position.y;
        
        double dx, dy; // difference distance
        double pre_yaw = 0;
        double roll, pitch, yaw = 0;

        // 遍历轨迹点，找到距离最小的轨迹点
        for(int i = 0; i < N; i++) 
        {
            geometry_msgs::PoseStamped pose_in_odom;       // 1015 先把路径转换到odom坐标系下
            _tf_listener.transformPose(_planning_frame, ros::Time(0), 
                                     totalPathMsg->poses[i], totalPathMsg->header.frame_id, pose_in_odom);
            dx = pose_in_odom.pose.position.x - px;
            dy = pose_in_odom.pose.position.y - py;


                
            // 计算“车到路径点”的向量与“车辆行进方向”向量的点积
            double dot_product = dx * travel_direction_x + dy * travel_direction_y;

            // 只有在行进方向前方的点，才参与最小距离的比较
            if (dot_product > 0)
            {
                double current_dist = sqrt(dx*dx + dy*dy);
                // 在这个行驶方向前半部分中，找一个离车最近的点
                if (current_dist < min_val_in_direction)
                {
                    min_val_in_direction = current_dist;
                    min_idx_in_direction = i;
                }
            }

            tf::Quaternion q(
                totalPathMsg->poses[i].pose.orientation.x,
                totalPathMsg->poses[i].pose.orientation.y,
                totalPathMsg->poses[i].pose.orientation.z,
                totalPathMsg->poses[i].pose.orientation.w);
            tf::Matrix3x3 m(q);
            m.getRPY(roll, pitch, yaw);

            if(abs(pre_yaw - yaw) > 5)
            {
                cout << "abs(pre_yaw - yaw)" << abs(pre_yaw - yaw) << endl;
                pre_yaw = yaw;
            }
       
            // if(min_val > sqrt(dx*dx + dy*dy) && abs((int)(i - min_idx)) < 50)
            // {
            //     min_val = sqrt(dx*dx + dy*dy);
            //     min_idx = i;
            // }
        }

        std::cout<< "min_idx" << min_idx << endl;
        
        // 从距离最小的轨迹点开始跟踪，并计算总的total_length，mpc_path中只保留3m内的轨迹点
        for(int i = min_idx_in_direction; i < N-1 ; i++)        // 1015
        {
            if(total_length > _pathLength)
                break;
            // 位姿坐标转换
            // _tf_listener.transformPose(_map_frame, ros::Time(0) , 
            //                                 totalPathMsg->poses[i], _odom_frame, tempPose); 
            _tf_listener.transformPose(_planning_frame, ros::Time(0), 
                         totalPathMsg->poses[i], totalPathMsg->header.frame_id, tempPose);      // 1015
            // tempPose.pose.position.z=0;                    
            mpc_path.poses.push_back(tempPose);

             double gap_x = totalPathMsg->poses[i+1].pose.position.x - totalPathMsg->poses[i].pose.position.x;
            double gap_y = totalPathMsg->poses[i+1].pose.position.y - totalPathMsg->poses[i].pose.position.y;
            _waypointsDist = sqrt(gap_x*gap_x + gap_y*gap_y);                                   
            total_length = total_length + _waypointsDist;           
        }   
        
        if(mpc_path.poses.size() >= _pathLength )
        {
            _odom_path = mpc_path; // Path waypoints in odom frame
            _path_computed = true;
            // publish odom path
            mpc_path.header.frame_id = _planning_frame;     // 1015
            mpc_path.header.stamp = ros::Time::now();
            _pub_odompath.publish(mpc_path);
        }
        else
        {
            cout << "Failed to path generation" << endl;
            _waypointsDist = -1;
        }       
    }
    catch(tf::TransformException &ex)
    {
        ROS_ERROR("%s",ex.what());
        ros::Duration(1.0).sleep();
    }
    
}

// CallBack: Update path waypoints (conversion to odom frame)
void MPCNode::pathCB(const nav_msgs::Path::ConstPtr& pathMsg)
{    
}

// CallBack: Update goal status
void MPCNode::goalCB(const geometry_msgs::PoseStamped::ConstPtr& goalMsg)
{
    _goal_pos = goalMsg->pose.position;
    _goal_received = true;
    _goal_reached = false;
    ROS_INFO("Goal Received :goalCB!");
}


// Callback: Check if the car is inside the goal area or not 
void MPCNode::amclCB(const geometry_msgs::PoseWithCovarianceStamped::ConstPtr& amclMsg)
{
    if(_goal_received)
    {
        double car2goal_x = _goal_pos.x - amclMsg->pose.pose.position.x;
        double car2goal_y = _goal_pos.y - amclMsg->pose.pose.position.y;
        double dist2goal = sqrt(car2goal_x*car2goal_x + car2goal_y*car2goal_y);
        if(dist2goal < _goalRadius)
        {
            _goal_received = false;
            _goal_reached = true;
            _path_computed = false;
            ROS_INFO("Goal Reached !");
        }
    }
}

void MPCNode::goalReachedJudge()
{
    if(_goal_received)
    {
        double car2goal_x = _goal_pos.x - _odom.pose.pose.position.x;
        double car2goal_y = _goal_pos.y - _odom.pose.pose.position.y;
        double dist2goal = sqrt(car2goal_x*car2goal_x + car2goal_y*car2goal_y);

        // cout<< "_goal_pos.x   "<< _goal_pos.x<<endl;
        if(dist2goal < _goalRadius)
        {
            _goal_reached = true;
            _path_computed = false;
            ROS_INFO("Goal Reached !");
        }
    }
}

// Timer: Control Loop (closed loop nonlinear MPC)
void MPCNode::controlLoopCB(const ros::TimerEvent&)
{          
    cout<< "controlLoopCB" << endl;
    if(_goal_received && !_goal_reached && _path_computed ) //received goal & goal not reached    
    {    
         // 当前位姿和期望路径
        nav_msgs::Odometry odom = _odom; 
        nav_msgs::Path odom_path = _odom_path;   

        // Update system states: X=[x, y, theta, v] 根据状态估计结果获取当前的 x y theat ,v
        const double px = odom.pose.pose.position.x; //pose: odom frame
        const double py = odom.pose.pose.position.y;
        tf::Pose pose;
        tf::poseMsgToTF(odom.pose.pose, pose);
        const double theta = tf::getYaw(pose.getRotation());
        double theta_ = atan2(odom_path.poses[1].pose.position.y - odom_path.poses[0].pose.position.y, odom_path.poses[1].pose.position.x - odom_path.poses[0].pose.position.x);
        cout<< "theta" << theta <<endl;
        cout<< "theta_e" << theta_ <<endl;

        // double diff = abs(theta - theta_);
        // cout<< "diff" << diff <<endl;
        // if (diff > M_PI) {  
        //     diff = 2 * M_PI - diff;  
        // }  
        // cout<< "diff_correct" << diff <<endl;
        // if ( diff > M_PI_2)
        // {
        //     cout<< "11" << endl;
        //     _twist_msg.linear.x  = 0; 
        //     _twist_msg.angular.z = 2;
        //     _pub_twist.publish(_twist_msg);
        //     return;
        // }

        double diff = theta_ - theta;  // -2pi ~ 2pi
            // 如果差大于π，则减去2π
        if (diff > M_PI) {
            diff -= 2 * M_PI;
        }
        // 如果差小于-π，则加上2π
        else if (diff < -M_PI) {
            diff += 2 * M_PI;
        }  
        if( abs(diff) > M_PI/3)
        {
            if (diff > 0) 
            {
            cout<< "11" << endl;
            _twist_msg.linear.x  = 0; 
            _twist_msg.angular.z = 0.3;
            _pub_twist.publish(_twist_msg);
            return;
            }
            else
            {
            cout<< "11" << endl;
            _twist_msg.linear.x  = 0; 
            _twist_msg.angular.z = -0.3;
            _pub_twist.publish(_twist_msg);
            return;
            }
            return ;
        }


        const double vx = _wheel_odom.twist.twist.linear.x; // 使用轮式里程计的线速度x
        const double vy = _wheel_odom.twist.twist.linear.y; // 使用轮式里程计的线速度y
        const double v = sqrt(vx*vx+vy*vy); // 计算合速度
        // const double v = odom.twist.twist.linear.x;

        // Update system inputs: U=[w, throttle]  系统输入 方向，油门
        const double w = _w; // steering -> w
        //const double steering = _steering;  // radian
        const double throttle = _throttle; // accel: >0; brake: <0
        const double dt = _dt;  // 0.1
        //const double Lf = _Lf;

        // Waypoints related parameters   mpc参考轨迹（3m内）
        const int N = odom_path.poses.size(); // Number of waypoints
        const double costheta = cos(theta);
        const double sintheta = sin(theta);

        // Convert to the vehicle coordinate system  全局坐标转换到车辆坐标系下  车头为正
        VectorXd x_veh(N); // 数组
        VectorXd y_veh(N);
        for(int i = 0; i < N; i++) 
        {
            const double dx = odom_path.poses[i].pose.position.x - px;
            const double dy = odom_path.poses[i].pose.position.y - py;
            x_veh[i] = dx * costheta + dy * sintheta;
            y_veh[i] = dy * costheta - dx * sintheta;
        }
        
        // Fit waypoints 拟合3阶多项式  依次从低次到高次 y = ax^3 + bx^2 + cx + d coeffs[0]为常数项
        auto coeffs = polyfit(x_veh, y_veh, 5); 
        // 计算给定多项式系数在特定x值处的多项式的值
        const double cte  = polyeval(coeffs, 0.0); // 横向误差 车辆当前位置与拟合路径在y轴方向上的偏差
        const double etheta = atan(coeffs[1]); // 车辆坐标系下拟合曲线的第一个点的切线夹角

        _mpc_cte = cte;
        _mpc_etheta = etheta;

        // _cte_int += cte * dt;   //1026:old: 累积积分误差
        
        ///// ----------------- 积分误差项：抗积分饱和 ----------------- /////
        // 4. 第四级防护：过零清零 (Zero Crossing Reset)
        // 原理：当CTE符号改变时（跨过参考线），之前的积分力已经完成任务，应立即清除以防超调
        if (cte * _prev_cte < 0.0) 
        {
            _cte_int = 0.0; 
        }
        _prev_cte = cte; 
        
        // 1. 速度门限 & 模式门限：只有在行驶(速度>阈值)且有目标且未到达时才积分
        if (abs(v) > _integ_v_thresh && _goal_received && !_goal_reached) 
        {
            // 2. 衰减积分 (Leaky Integrator)：引入衰减系数，让旧误差随时间衰减，解决震荡
            _cte_int = _cte_int * _integ_decay + cte * dt;
        }
        else
        {
            // 停车或无目标时清零，解决静止积分爆炸问题
            _cte_int = 0.0;
        }

        // 3. 硬限幅 (Hard Clamping)：防止数值过大
        if (_cte_int > _integ_max) _cte_int = _integ_max;
        else if (_cte_int < -_integ_max) _cte_int = -_integ_max;
        ///// ----------------- 积分误差项：抗积分饱和 ----------------- /////

        ///// ----------------- 根据曲率和当前误差值自适应计算参考速度 ----------------- /////
        // 目的：误差大或弯道急时主动减速，解决“死不减速”导致的超调
        
        // A. 基于误差减速
        // 公式：v = v_max / (1 + k1*|cte| + k2*|etheta|)
        double error_scaling = 1.0 / (1.0 + _v_ref_cte_k * abs(cte) + _v_ref_etheta_k * abs(etheta));
        double v_ref_error = _ref_vel * error_scaling; 

        // B. 基于曲率减速 
        // 计算当前点的曲率 k = |y''| / (1 + y'^2)^(1.5)
        // 对于多项式 y = c0 + c1*x + c2*x^2... 在车体坐标系原点(x=0):
        double dy = coeffs[1];        // y' = c1
        double ddy = 2 * coeffs[2];   // y'' = 2*c2
        double curvature = abs(ddy) / pow(1 + dy*dy, 1.5);
        
        // 物理公式 v < sqrt(a_lat / k)
        // 这里硬编码侧向加速度限制为 1.5 m/s^2，省略yaml参数以简化调参
        double lat_acc_limit_internal = 1.5; 
        double v_ref_curve = _ref_vel; // 默认不限速
        if (curvature > 0.001) { // 防止除以0
            v_ref_curve = sqrt(lat_acc_limit_internal / curvature);
        }

        // C. 融合与限幅
        // 取两者中的较小值（安全第一），且不低于最小速度，不高于设定最高速度
        double final_ref_vel = std::min(v_ref_error, v_ref_curve);
        final_ref_vel = std::max(final_ref_vel, _min_ref_vel); // 防止停车
        final_ref_vel = std::min(final_ref_vel, _ref_vel);     // 不超速

        cout << "[Adaptive Vel] Final: " << final_ref_vel 
                 << " | Limiter: " << (v_ref_error < v_ref_curve ? "ERROR (误差)" : "CURVATURE (弯道)") 
                 << " | v_err: " << v_ref_error 
                 << " | v_curv: " << v_ref_curve << endl;   // 调试用：看到底是哪项限制了参考速度

        // D. 关键步骤：将动态速度传递给 MPC 求解器
        // 这一步修改了 _mpc_params 中的参数，并重新加载，改变了 Solver 内部的成本函数目标
        _mpc_params["REF_V"] = final_ref_vel;
        _mpc.LoadParams(_mpc_params);

        ///// ----------------- 根据曲率和当前误差值自适应计算参考速度 ----------------- /////


        VectorXd state(7);  //1026:扩展状态向量
        if(_delay_mode)
        {
            // 车辆坐标系下
            // Kinematic model is used to predict vehicle state at the actual moment of control (current time + delay dt) 动力学模型预测
            const double px_act = v * dt;
            const double py_act = 0;
            const double theta_act = w * dt; //(steering) theta_act = v * steering * dt / Lf;
            const double v_act = v + throttle * dt; //v = v + a * dt
            
            const double cte_act = cte + v * sin(etheta) * dt; // 车辆坐标系下的y向误差
            const double etheta_act = etheta - theta_act;  
            const double cte_int_act = _cte_int + cte * dt; //1026:预测积分项在延迟后的状态

            state << px_act, py_act, theta_act, v_act, cte_act, etheta_act, cte_int_act;   // 1026:7自由度状态变量
        }
        else
        {
            state << 0, 0, 0, v, cte, etheta, _cte_int;     //1026
        }
        
        // Solve MPC Problem
        vector<double> mpc_results = _mpc.Solve(state, coeffs);
              
        // MPC result (all described in car frame), output = (acceleration, w)        
        _w = mpc_results[0]; // radian/sec, angular velocity
        _throttle = mpc_results[1]; // acceleration
        _speed = v + _throttle*dt;  // speed
        if (_speed >= _max_speed)
            _speed = _max_speed;
        if(_speed <= 0.0)
            _speed = 0.0;

        if(_debug_info)
        {
            cout << "\n\nDEBUG" << endl;
            cout << "theta: " << theta << endl;
            cout << "V: " << v << endl;
            //cout << "odom_path: \n" << odom_path << endl;
            //cout << "x_points: \n" << x_veh << endl;
            //cout << "y_points: \n" << y_veh << endl;
            cout << "coeffs: \n" << coeffs << endl;
            cout << "_w: \n" << _w << endl;
            cout << "_throttle: \n" << _throttle << endl;
            cout << "_speed: \n" << _speed << endl;
        }

        // Display the MPC predicted trajectory
        _mpc_traj = nav_msgs::Path();
        _mpc_traj.header.frame_id = _car_frame; // points in car coordinate        
        _mpc_traj.header.stamp = ros::Time::now();
        for(int i=0; i<_mpc.mpc_x.size(); i++)
        {
            geometry_msgs::PoseStamped tempPose;
            tempPose.header = _mpc_traj.header;
            tempPose.pose.position.x = _mpc.mpc_x[i];
            tempPose.pose.position.y = _mpc.mpc_y[i];
            tempPose.pose.orientation.w = 1.0;
            _mpc_traj.poses.push_back(tempPose); 
        }     
        // publish the mpc trajectory
        _pub_mpctraj.publish(_mpc_traj);

    }
    else
    {
        _throttle = 0.0;
        _speed = 0.0;
        _w = 0;
        _cte_int = 0.0; //1026:当没有目标或到达目标时，清除积分项

        if(_goal_reached && _goal_received)
            cout << "Goal Reached: control loop !" << endl;
    }


    // publish general cmd_vel 
    if(_pub_twist_flag)
    {
        _twist_msg.linear.x  = _speed; 
        _twist_msg.angular.z = _w;
        _pub_twist.publish(_twist_msg);

        std_msgs::Float32 mpc_total_cost;
        mpc_total_cost.data = static_cast<float>(_mpc._mpc_totalcost);
        _pub_totalcost.publish(mpc_total_cost);

        std_msgs::Float32 mpc_cte_cost;
        mpc_cte_cost.data = static_cast<float>(_mpc._mpc_ctecost);
        _pub_ctecost.publish(mpc_cte_cost);

        std_msgs::Float32 mpc_etheta_cost;
        mpc_etheta_cost.data = static_cast<float>(_mpc._mpc_ethetacost);
        _pub_ethetacost.publish(mpc_etheta_cost);

        //cout << "_mpc_totalcost: "<< _mpc._mpc_totalcost << endl;
        //cout << "_mpc_ctecost: "<< _mpc._mpc_ctecost << endl;
        //cout << "_mpc_ethetacost: "<< _mpc._mpc_ethetacost << endl;
        //cout << "_mpc_velcost: "<< _mpc._mpc_velcost << endl;
        //writefile
        idx++;
        // cout << "idx: "<< idx << endl;

    }
    else
    {
        _twist_msg.linear.x  = 0; 
        _twist_msg.angular.z = 0;
        _pub_twist.publish(_twist_msg);
    }
    
  
    /*
    file.open("/home/geonhee/catkin_ws/src/mpc_ros/write.csv");
    string line;
    while (getline(file, line,'\n')) 
    {
        istringstream templine(line); 
        string data;
        while (getline( templine, data,',')) 
        {
            cout << "data.c_str(): "<< data << endl;
            matrix.push_back(atof(data.c_str()));  
        }
    }
    file.close();*/

}

/*****************/
/* MAIN FUNCTION */
/*****************/
int main(int argc, char **argv)
{
    //Initiate ROS
    ros::init(argc, argv, "MPC_Node");
    MPCNode mpc_node;

    ROS_INFO("Waiting for global path msgs ~");
    ros::AsyncSpinner spinner(mpc_node.get_thread_numbers()); // Use multi threads
    spinner.start();
    ros::waitForShutdown();
    return 0;
}
