from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def _bool_text(value: str) -> bool:
    return value.strip().lower() in ("1", "true", "yes", "on")


def _maybe_add(params: dict, name: str, value: str, value_type):
    text = value.strip()
    if text == "":
        return
    if value_type is bool:
        params[name] = _bool_text(text)
    elif value_type is float:
        params[name] = float(text)
    else:
        params[name] = text


def _launch_setup(context, *args, **kwargs):
    config = LaunchConfiguration("config").perform(context)
    bridge_config = LaunchConfiguration("bridge_config").perform(context)
    use_fake_sensors = LaunchConfiguration("use_fake_sensors")
    start_bridge = LaunchConfiguration("start_bridge")
    fake_publish_abad_joints = LaunchConfiguration("fake_publish_abad_joints")
    fake_publish_damper_joints = LaunchConfiguration("fake_publish_damper_joints")

    controller_overrides = {}
    _maybe_add(controller_overrides, "policy.onnx_path", LaunchConfiguration("onnx_path").perform(context), str)
    _maybe_add(controller_overrides, "policy.policy_hz", LaunchConfiguration("policy_hz").perform(context), float)
    _maybe_add(controller_overrides, "policy.use_cuda", LaunchConfiguration("use_cuda").perform(context), bool)
    _maybe_add(controller_overrides, "policy.use_tensorrt", LaunchConfiguration("use_tensorrt").perform(context), bool)
    _maybe_add(
        controller_overrides,
        "state_machine.enable_policy_on_start",
        LaunchConfiguration("enable_policy_on_start").perform(context),
        bool,
    )
    _maybe_add(
        controller_overrides,
        "state_machine.enable_motor_output_on_start",
        LaunchConfiguration("enable_motor_output_on_start").perform(context),
        bool,
    )
    _maybe_add(
        controller_overrides,
        "observation.base_lin_vel_source",
        LaunchConfiguration("base_lin_vel_source").perform(context),
        str,
    )
    _maybe_add(
        controller_overrides,
        "observation.abad_feedback_source",
        LaunchConfiguration("abad_feedback_source").perform(context),
        str,
    )
    _maybe_add(
        controller_overrides,
        "safety.require_lowlevel_heartbeat",
        LaunchConfiguration("require_lowlevel_heartbeat").perform(context),
        bool,
    )
    _maybe_add(
        controller_overrides,
        "safety.require_motor_feedback",
        LaunchConfiguration("require_motor_feedback").perform(context),
        bool,
    )

    controller_parameters = [config]
    if controller_overrides:
        controller_parameters.append(controller_overrides)

    bridge_overrides = {}
    _maybe_add(bridge_overrides, "backend", LaunchConfiguration("bridge_backend").perform(context), str)
    _maybe_add(bridge_overrides, "rinbo.allow_enable", LaunchConfiguration("bridge_rinbo_allow_enable").perform(context), bool)
    _maybe_add(bridge_overrides, "rinbo.require_state", LaunchConfiguration("bridge_rinbo_require_state").perform(context), bool)
    _maybe_add(
        bridge_overrides,
        "rinbo.block_if_duplicate_command_publishers",
        LaunchConfiguration("bridge_rinbo_block_if_duplicate_command_publishers").perform(context),
        bool,
    )
    bridge_parameters = [bridge_config]
    if bridge_overrides:
        bridge_parameters.append(bridge_overrides)

    fake_params = {
        "publish_abad_joints": fake_publish_abad_joints,
        "publish_damper_joints": fake_publish_damper_joints,
    }
    _maybe_add(fake_params, "rate_hz", LaunchConfiguration("fake_rate_hz").perform(context), float)
    _maybe_add(fake_params, "cmd_vx", LaunchConfiguration("fake_cmd_vx").perform(context), float)
    _maybe_add(fake_params, "cmd_vy", LaunchConfiguration("fake_cmd_vy").perform(context), float)
    _maybe_add(fake_params, "cmd_wz", LaunchConfiguration("fake_cmd_wz").perform(context), float)

    return [
        Node(
            package="redrhex_rl_controller",
            executable="rl_controller_node",
            name="redrhex_rl_controller",
            output="screen",
            parameters=controller_parameters,
        ),
        Node(
            package="redrhex_lowlevel_bridge",
            executable="lowlevel_bridge_node",
            name="redrhex_lowlevel_bridge",
            output="screen",
            parameters=bridge_parameters,
            condition=IfCondition(start_bridge),
        ),
        Node(
            package="redrhex_rl_controller",
            executable="fake_sensor_node",
            name="redrhex_fake_sensor_node",
            output="screen",
            parameters=[fake_params],
            condition=IfCondition(use_fake_sensors),
        ),
    ]


def generate_launch_description():
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
        DeclareLaunchArgument("fake_rate_hz", default_value=""),
        DeclareLaunchArgument("fake_cmd_vx", default_value=""),
        DeclareLaunchArgument("fake_cmd_vy", default_value=""),
        DeclareLaunchArgument("fake_cmd_wz", default_value=""),
        DeclareLaunchArgument("start_bridge", default_value="true"),
        DeclareLaunchArgument("onnx_path", default_value="", description="Optional override for policy.onnx_path."),
        DeclareLaunchArgument("policy_hz", default_value="", description="Optional override for policy.policy_hz."),
        DeclareLaunchArgument("use_cuda", default_value="", description="Optional bool override for policy.use_cuda."),
        DeclareLaunchArgument("use_tensorrt", default_value="", description="Optional bool override for policy.use_tensorrt."),
        DeclareLaunchArgument("enable_policy_on_start", default_value="", description="Keep false on hardware."),
        DeclareLaunchArgument("enable_motor_output_on_start", default_value="", description="Keep false on hardware."),
        DeclareLaunchArgument("base_lin_vel_source", default_value="", description="zero or odom."),
        DeclareLaunchArgument("abad_feedback_source", default_value="", description="commanded or joint_states."),
        DeclareLaunchArgument("require_lowlevel_heartbeat", default_value="", description="Optional bool safety override."),
        DeclareLaunchArgument("require_motor_feedback", default_value="", description="Optional bool safety override."),
        DeclareLaunchArgument("bridge_backend", default_value="", description="Optional low-level bridge backend override."),
        DeclareLaunchArgument("bridge_rinbo_allow_enable", default_value="", description="Optional bool override for bridge rinbo.allow_enable."),
        DeclareLaunchArgument("bridge_rinbo_require_state", default_value="", description="Optional bool override for bridge rinbo.require_state."),
        DeclareLaunchArgument(
            "bridge_rinbo_block_if_duplicate_command_publishers",
            default_value="",
            description="Optional duplicate /motor/command publisher protection override.",
        ),
        OpaqueFunction(function=_launch_setup),
    ])
