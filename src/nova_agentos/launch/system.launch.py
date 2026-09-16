# 完整 ROS 侧拓扑。RoboCasa sim server 与 Pi server 仍是外部进程。
from pathlib import Path

from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    agentos_share = get_package_share_directory("nova_agentos")
    skills_dir = agentos_share + "/skills"
    agentos_config = Path(agentos_share) / "config" / "agentos.yaml"
    vla_config = Path(get_package_share_directory("nova_vla_executor")) / "config" / "vla.yaml"
    perception_config = (
        Path(get_package_share_directory("nova_perception_executor")) / "config" / "perception.yaml"
    )
    robocasa_config = (
        Path(get_package_share_directory("nova_robocasa_bridge")) / "config" / "bridge.yaml"
    )
    return LaunchDescription(
        [
            Node(
                package="nova_robocasa_bridge",
                executable="robocasa_bridge_node",
                name="robocasa_bridge",
                output="screen",
                parameters=[str(robocasa_config)],
            ),
            Node(
                package="nova_executor_raw",
                executable="nova_executor_raw_node",
                name="nova_executor_raw",
                output="screen",
            ),
            Node(
                package="nova_executor_manager",
                executable="nova_executor_manager_node",
                name="nova_executor_manager",
                output="screen",
            ),
            Node(
                package="nova_perception_executor",
                executable="nova_perception_executor_node",
                name="nova_perception_executor",
                output="screen",
                parameters=[str(perception_config)],
            ),
            Node(
                package="nova_agentos",
                executable="nova_agentos_node",
                name="nova_agentos",
                output="screen",
                parameters=[str(agentos_config), {"skills_dir": skills_dir}],
            ),
        ]
    )
