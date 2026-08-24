# 一键启动 NovaAgent 演示系统:demo executor + executor_manager + agentos
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    skills_dir = get_package_share_directory("nova_agentos") + "/skills"
    return LaunchDescription(
        [
            Node(
                package="nova_executor_demo",
                executable="nova_executor_demo_node",
                name="nova_executor_demo",
                output="screen",
            ),
            Node(
                package="nova_executor_manager",
                executable="nova_executor_manager_node",
                name="nova_executor_manager",
                output="screen",
            ),
            Node(
                package="nova_agentos",
                executable="nova_agentos_node",
                name="nova_agentos",
                output="screen",
                parameters=[{"skills_dir": skills_dir}],
            ),
        ]
    )
