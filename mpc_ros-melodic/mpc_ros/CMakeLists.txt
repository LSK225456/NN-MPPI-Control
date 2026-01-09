cmake_minimum_required(VERSION 3.5)
project(mpc_ros)

## Compile as C++11, supported in ROS Kinetic and newer
add_definitions(-std=c++11 -O3)

# options for build configuration
option(BUILD_EXAMPLE "Whether or not building the CppAD & Ipopt example" OFF) 

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  costmap_2d
  dynamic_reconfigure
  geometry_msgs
  nav_msgs
  #ackermann_msgs
  pluginlib
  message_generation
  move_base
  base_local_planner
  roscpp
  rospy
  std_msgs
  tf
  visualization_msgs
)

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories( ${catkin_INCLUDE_DIRS} include )
include_directories(/usr/include)
link_directories(/usr/lib)


# dynamic reconfigure
generate_dynamic_reconfigure_options(
    cfg/MPCPlanner.cfg
)


catkin_package(
   INCLUDE_DIRS include
   LIBRARIES mpc_ros
   CATKIN_DEPENDS costmap_2d dynamic_reconfigure geometry_msgs move_base roscpp rospy std_msgs tf visualization_msgs pluginlib
#  DEPENDS system_lib
)

###########
## Build ##
###########

# General MPC_Node 
ADD_EXECUTABLE( MPC_Node src/MPC.cpp src/MPC_Node.cpp )
TARGET_LINK_LIBRARIES(MPC_Node ipopt ${catkin_LIBRARIES} )

# Total navigation with MPC_Node

# Local planner with MPC_Node for tracking
ADD_EXECUTABLE( tracking_reference_trajectory src/trackRefTraj.cpp src/trackRefTrajNode.cpp )
TARGET_LINK_LIBRARIES(tracking_reference_trajectory ipopt ${catkin_LIBRARIES} )

ADD_EXECUTABLE( tracking_reference_trajectory_plan src/trackRefTraj.cpp src/trackRefTrajNode_plan.cpp )
TARGET_LINK_LIBRARIES(tracking_reference_trajectory_plan ipopt ${catkin_LIBRARIES} )
# Pure Pursuit Node
add_executable(Pure_Pursuit src/Pure_Pursuit.cpp)
target_link_libraries(Pure_Pursuit ${catkin_LIBRARIES})

# Own Global planner for tracking desired trajectory
add_library(global_planner_lib src/global_planner/global_planner.cpp)
TARGET_LINK_LIBRARIES(global_planner_lib ipopt ${catkin_LIBRARIES} )

# MPC Local planner plugin
#############
## Example ##
#############
if(BUILD_EXAMPLE)
    ADD_EXECUTABLE( CppAD_started example/CppAD_started.cpp )
    TARGET_LINK_LIBRARIES(CppAD_started)
    
    ADD_EXECUTABLE( CppAD_Ipopt example/CppAD_Ipopt.cpp )
    TARGET_LINK_LIBRARIES(CppAD_Ipopt ipopt)
endif(BUILD_EXAMPLE)


# =========================================================
# INSTALL RULES (必须添加以下内容，RK3588才能找到文件)
# =========================================================

# 1. 安装编译好的 C++ 可执行文件和库
install(TARGETS 
    MPC_Node 
    tracking_reference_trajectory 
    tracking_reference_trajectory_plan 
    Pure_Pursuit
    global_planner_lib
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# 2. 安装 launch 文件、参数、地图、rviz配置等文件夹
# 注意：务必检查你的文件夹名称是否匹配 (launch, params, maps, rviz, cfg)
install(DIRECTORY launch params maps rviz cfg
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

# 3. 安装头文件 (如果有)
install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)

# 4. 安装 Python 脚本 (使它们可以通过 rosrun/roslaunch 调用)
# 这里列出了你之前文件列表中出现的关键脚本，请根据实际情况增减
catkin_install_python(PROGRAMS
  script/mpc_trajectory_generation.py
  script/dwa_trajectory_generation.py
  script/base_control.py
  script/odom_path_plotter.py
  script/teleop_keyboard.py
  script/car_calib.py
  script/odom_calib.py
  script/evaluate_tracking.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)