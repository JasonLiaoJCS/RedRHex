from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


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
    elif value_type is int:
        params[name] = int(text)
    else:
        params[name] = text


def _launch_setup(context, *args, **kwargs):
    config = LaunchConfiguration("config").perform(context)
    override_params = {}
    _maybe_add(override_params, "backend", LaunchConfiguration("backend").perform(context), str)
    _maybe_add(override_params, "rinbo.allow_enable", LaunchConfiguration("rinbo_allow_enable").perform(context), bool)
    _maybe_add(override_params, "rinbo.require_state", LaunchConfiguration("rinbo_require_state").perform(context), bool)
    _maybe_add(
        override_params,
        "rinbo.publish_when_disabled",
        LaunchConfiguration("rinbo_publish_when_disabled").perform(context),
        bool,
    )
    _maybe_add(
        override_params,
        "rinbo.block_if_duplicate_command_publishers",
        LaunchConfiguration("rinbo_block_if_duplicate_command_publishers").perform(context),
        bool,
    )
    _maybe_add(
        override_params,
        "rinbo.publish_shutdown_disable",
        LaunchConfiguration("rinbo_publish_shutdown_disable").perform(context),
        bool,
    )
    _maybe_add(override_params, "rinbo.main_pwm_per_rad_s", LaunchConfiguration("rinbo_main_pwm_per_rad_s").perform(context), float)
    _maybe_add(override_params, "rinbo.main_max_pwm", LaunchConfiguration("rinbo_main_max_pwm").perform(context), float)

    parameters = [config]
    if override_params:
        parameters.append(override_params)

    return [
        Node(
            package="redrhex_lowlevel_bridge",
            executable="lowlevel_bridge_node",
            name="redrhex_lowlevel_bridge",
            output="screen",
            parameters=parameters,
        )
    ]


def generate_launch_description():
    default_config = PathJoinSubstitution([
        FindPackageShare("redrhex_lowlevel_bridge"),
        "config",
        "lowlevel_bridge.yaml",
    ])
    return LaunchDescription([
        DeclareLaunchArgument("config", default_value=default_config),
        DeclareLaunchArgument("backend", default_value="", description="Optional override: mock, biorola_ros, rinbo_ros, serial, sbrio_udp."),
        DeclareLaunchArgument("rinbo_allow_enable", default_value="", description="Optional bool override for rinbo.allow_enable."),
        DeclareLaunchArgument("rinbo_require_state", default_value="", description="Optional bool override for rinbo.require_state."),
        DeclareLaunchArgument("rinbo_publish_when_disabled", default_value="", description="Optional bool override for rinbo.publish_when_disabled."),
        DeclareLaunchArgument(
            "rinbo_block_if_duplicate_command_publishers",
            default_value="",
            description="Optional bool override for duplicate /motor/command publisher protection.",
        ),
        DeclareLaunchArgument(
            "rinbo_publish_shutdown_disable",
            default_value="",
            description="Optional bool override for sending disabled packets on shutdown.",
        ),
        DeclareLaunchArgument("rinbo_main_pwm_per_rad_s", default_value="", description="Optional float override for main velocity to PWM conversion."),
        DeclareLaunchArgument("rinbo_main_max_pwm", default_value="", description="Optional float override for max PWM/voltage command."),
        OpaqueFunction(function=_launch_setup),
    ])
