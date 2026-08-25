cmake_minimum_required(VERSION 3.8)
project(nova_interfaces)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(action_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/DAGNode.msg"
  "msg/TaskGraph.msg"
  "msg/ToolDescriptor.msg"
  "msg/ExecutorHeartbeat.msg"
  "msg/SkillInfo.msg"
  "msg/SkillCommand.msg"
  "msg/SkillResult.msg"
  "msg/TaskState.msg"
  "srv/ListTools.srv"
  "srv/RunTask.srv"
  "srv/EnvInfo.srv"
  "srv/MapTopics.srv"
  "srv/UnmapTopics.srv"
  "action/MCPExecute.action"
  DEPENDENCIES builtin_interfaces action_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
