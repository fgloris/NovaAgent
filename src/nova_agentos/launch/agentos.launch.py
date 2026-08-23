from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            Node(
                package="nova_agentos",
                executable="agentos_node",
                name="nova_agentos",
                output="screen",
            )
        ]
    )
