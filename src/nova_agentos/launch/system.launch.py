# 一键启动 NovaAgent 演示系统:VLA executor + executor_manager + agentos
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    skills_dir = get_package_share_directory("nova_agentos") + "/skills"
    return LaunchDescription(
        [
            Node(
                package="nova_vla_executor",
                executable="nova_vla_executor_node",
                name="nova_vla_executor",
                output="screen",
                parameters=[{"server_url": "http://127.0.0.1:8767"}],
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
