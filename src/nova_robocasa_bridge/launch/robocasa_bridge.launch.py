from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory
from pathlib import Path


def generate_launch_description():
    config = Path(get_package_share_directory("nova_robocasa_bridge")) / "config" / "bridge.yaml"

    return LaunchDescription(
        [
            DeclareLaunchArgument("env_id", default_value="robocasa/PickPlaceCounterToCabinet"),
            DeclareLaunchArgument("server_host", default_value="127.0.0.1"),
            DeclareLaunchArgument("server_port", default_value="8766"),
            Node(
                package="nova_robocasa_bridge",
                executable="robocasa_bridge_node",
                name="robocasa_bridge",
                output="screen",
                parameters=[
                    str(config),
                    {
                        "env_id": LaunchConfiguration("env_id"),
                        "server_host": LaunchConfiguration("server_host"),
                        "server_port": ParameterValue(
                            LaunchConfiguration("server_port"),
                            value_type=int,
                        ),
                    },
                ],
            )
        ]
    )
