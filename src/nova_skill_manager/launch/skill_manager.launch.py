from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            Node(
                package="nova_skill_manager",
                executable="skill_manager_node",
                name="nova_skill_manager",
                output="screen",
            )
        ]
    )
