from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    config = LaunchConfiguration("config")
    default_config = PathJoinSubstitution([
        FindPackageShare("redrhex_lowlevel_bridge"),
        "config",
        "lowlevel_bridge.yaml",
    ])
    return LaunchDescription([
        DeclareLaunchArgument("config", default_value=default_config),
        Node(
            package="redrhex_lowlevel_bridge",
            executable="lowlevel_bridge_node",
            name="redrhex_lowlevel_bridge",
            output="screen",
            parameters=[config],
        ),
    ])
