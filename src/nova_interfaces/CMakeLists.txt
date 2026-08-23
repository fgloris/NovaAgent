cmake_minimum_required(VERSION 3.8)
project(nova_interfaces)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/SkillInfo.msg"
  "msg/SkillCommand.msg"
  "msg/SkillResult.msg"
  "msg/TaskState.msg"
  "srv/ListSkills.srv"
  "srv/GetSkill.srv"
  "srv/LoadSkills.srv"
  "srv/ValidateSkills.srv"
  "srv/CreateSkill.srv"
  "srv/DeleteSkill.srv"
)

ament_export_dependencies(rosidl_default_runtime)

ament_package()
