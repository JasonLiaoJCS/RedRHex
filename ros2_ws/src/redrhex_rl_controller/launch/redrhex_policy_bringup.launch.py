from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def generate_launch_description():
    config = LaunchConfiguration("config")
    use_fake_sensors = LaunchConfiguration("use_fake_sensors")
    fake_publish_abad_joints = LaunchConfiguration("fake_publish_abad_joints")
    fake_publish_damper_joints = LaunchConfiguration("fake_publish_damper_joints")
    start_bridge = LaunchConfiguration("start_bridge")
    bridge_config = LaunchConfiguration("bridge_config")

    default_config = PathJoinSubstitution([
        FindPackageShare("redrhex_rl_controller"),
        "config",
        "redrhex_policy.yaml",
    ])
    default_bridge_config = PathJoinSubstitution([
        FindPackageShare("redrhex_lowlevel_bridge"),
        "config",
        "lowlevel_bridge.yaml",
    ])

    return LaunchDescription([
        DeclareLaunchArgument("config", default_value=default_config),
        DeclareLaunchArgument("bridge_config", default_value=default_bridge_config),
        DeclareLaunchArgument("use_fake_sensors", default_value="false"),
        DeclareLaunchArgument("fake_publish_abad_joints", default_value="false"),
        DeclareLaunchArgument("fake_publish_damper_joints", default_value="false"),
        DeclareLaunchArgument("start_bridge", default_value="true"),
        Node(
            package="redrhex_rl_controller",
            executable="rl_controller_node",
            name="redrhex_rl_controller",
            output="screen",
            parameters=[config],
        ),
        Node(
            package="redrhex_lowlevel_bridge",
            executable="lowlevel_bridge_node",
            name="redrhex_lowlevel_bridge",
            output="screen",
            parameters=[bridge_config],
            condition=IfCondition(start_bridge),
        ),
        Node(
            package="redrhex_rl_controller",
            executable="fake_sensor_node",
            name="redrhex_fake_sensor_node",
            output="screen",
            parameters=[{
                "publish_abad_joints": fake_publish_abad_joints,
                "publish_damper_joints": fake_publish_damper_joints,
            }],
            condition=IfCondition(use_fake_sensors),
        ),
    ])
