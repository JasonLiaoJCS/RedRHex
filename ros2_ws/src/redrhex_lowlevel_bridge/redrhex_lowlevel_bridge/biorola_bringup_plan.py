"""Generate a copy/paste BioRoLaROS2 + RedRhex bringup checklist."""

from __future__ import annotations

import argparse
import ipaddress
import textwrap


def _validate_ip(value: str, name: str) -> str:
    try:
        ipaddress.ip_address(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{name} must be an IPv4/IPv6 address, got {value!r}") from exc
    return value


def _shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def _strip_template_indent(text: str) -> str:
    lines = text.splitlines()
    return "\n".join(line[8:] if line.startswith("        ") else line for line in lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Print a staged command plan for BioRoLaROS2 sbRIO + RedRhex RL bringup."
    )
    parser.add_argument("--sbrio-ip", required=True, type=lambda x: _validate_ip(x, "sbrio-ip"))
    parser.add_argument("--orin-ip", required=True, type=lambda x: _validate_ip(x, "orin-ip"))
    parser.add_argument("--rinbo-ws", default="~/rinbo_ros_ws")
    parser.add_argument("--redrhex-ws", default="~/RedRhex/RedRhex/ros2_ws")
    parser.add_argument("--onnx-path", default="/home/jetson/redrhex_models/policy.onnx")
    parser.add_argument("--ros-distro", default="humble")
    parser.add_argument("--sbriouser", default="admin")
    parser.add_argument("--include-relay", action="store_true", help="Print the relay power-on command as an active step.")
    parser.add_argument(
        "--enable-hardware-snippets",
        action="store_true",
        help="Also print allow_enable=true and small single-motor test snippets.",
    )
    return parser


def render(args: argparse.Namespace) -> str:
    rinbo_ws = args.rinbo_ws
    redrhex_ws = args.redrhex_ws
    relay_line = (
        "ros2 run redrhex_lowlevel_bridge biorola_power_tool relay --confirm-relay"
        if args.include_relay
        else "# Relay step intentionally commented out until E-stop/current-limit are ready:\n"
        "# ros2 run redrhex_lowlevel_bridge biorola_power_tool relay --confirm-relay"
    )
    hardware_snippets = ""
    if args.enable_hardware_snippets:
        hardware_snippets = f"""

## 8. Hardware snippets after the robot is suspended

Only run this after the robot is lifted, current-limited, and E-stop is in hand.

```bash
source /opt/ros/{args.ros_distro}/setup.bash
source {rinbo_ws}/install/setup.bash
source {redrhex_ws}/install/setup.bash

# Restart the RedRhex bridge with hardware output allowed, but still duplicate-publisher protected.
ros2 launch redrhex_lowlevel_bridge lowlevel_bridge.launch.py \\
  backend:=biorola_ros \\
  rinbo_allow_enable:=true \\
  rinbo_require_state:=true \\
  rinbo_block_if_duplicate_command_publishers:=true

# In another terminal, list policy-order joints.
ros2 run redrhex_rl_controller motor_command_tool list-joints

# Single ABAD dry-run first, then a tiny enabled move.
ros2 run redrhex_rl_controller motor_command_tool single-abad --index 0 --position 0.08 --dry-run
ros2 run redrhex_rl_controller motor_command_tool single-abad --index 0 --position 0.08 --enable --confirm-risk --duration 1.0

# Single main-drive dry-run first, then a tiny enabled velocity command.
ros2 run redrhex_rl_controller motor_command_tool single-main-velocity --index 0 --velocity 0.25 --dry-run
ros2 run redrhex_rl_controller motor_command_tool single-main-velocity --index 0 --velocity 0.25 --enable --confirm-risk --duration 1.0
```
"""

    rendered = textwrap.dedent(
        f"""
        # BioRoLaROS2 + RedRhex RL Bringup Plan

        Generated for:

        ```text
        sbRIO IP : {args.sbrio_ip}
        Orin IP  : {args.orin_ip}
        ROS_DISTRO: {args.ros_distro}
        rinbo_ws : {rinbo_ws}
        redrhex_ws: {redrhex_ws}
        policy.onnx: {args.onnx_path}
        ```

        Keep the robot lifted, current-limited, and E-stop ready before any enabled motor command.

        ## 1. sbRIO terminal

        ```bash
        ssh {args.sbriouser}@{args.sbrio_ip}
        cd ~/rinbo_sbRIO_ws/rinbo_fpga_driver/build
        export CORE_LOCAL_IP={args.sbrio_ip}
        export CORE_MASTER_ADDR={args.sbrio_ip}:50051
        ps -ef | egrep "grpccore|fpga_driver" | grep -v grep
        # If old processes exist, stop duplicates intentionally:
        # pkill -f grpccore
        # pkill -f fpga_driver
        nohup /home/admin/rinbo_sbRIO_ws/install/bin/grpccore >/tmp/grpccore.log 2>&1 &
        nohup /home/admin/rinbo_sbRIO_ws/rinbo_fpga_driver/build/fpga_driver >/tmp/fpga_driver.log 2>&1 &
        ps -ef | egrep "grpccore|fpga_driver" | grep -v grep
        netstat -tn | grep 50051 || echo "NO TCP on 50051"
        ```

        ## 2. Orin terminal A: BioRoLaROS2 bridge

        ```bash
        source /opt/ros/{args.ros_distro}/setup.bash
        source {rinbo_ws}/install/setup.bash
        export CORE_MASTER_ADDR={args.sbrio_ip}:50051
        export CORE_LOCAL_IP={args.orin_ip}
        ros2 run rinbo_ros_bridge rinbo_ros_bridge
        ```

        ## 3. Orin terminal B: RedRhex tools and preflight

        ```bash
        source /opt/ros/{args.ros_distro}/setup.bash
        source {rinbo_ws}/install/setup.bash
        cd {redrhex_ws}
        colcon build --symlink-install --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3
        source install/setup.bash

        ros2 run redrhex_lowlevel_bridge biorola_bringup_check \\
          --master-addr {args.sbrio_ip}:50051 \\
          --local-ip {args.orin_ip} \\
          --message-timeout-s 5.0

        ros2 run redrhex_rl_controller preflight_check \\
          --onnx {_shell_quote(args.onnx_path)} \\
          --config {redrhex_ws}/src/redrhex_rl_controller/config/redrhex_policy.yaml
        ```

        ## 4. Power, still no RL control

        ```bash
        ros2 run redrhex_lowlevel_bridge biorola_power_tool digital
        ros2 run redrhex_lowlevel_bridge biorola_power_tool sensors
        {relay_line}
        ros2 run redrhex_lowlevel_bridge biorola_power_tool status
        ros2 run redrhex_lowlevel_bridge biorola_bringup_check --message-timeout-s 5.0 --require-power-state
        ```

        ## 5. Calibration and standing through existing BioRoLaROS2 FSM

        ```bash
        ros2 run rinbo_fsm rinbo_cali
        ros2 run rinbo_fsm rinbo_standing
        # Stop rinbo_fsm before RedRhex RL tests. Do not leave rinbo_tripod running.
        ros2 topic info /motor/command -v
        ```

        ## 6. RedRhex low-level bridge, preview-only first

        ```bash
        source /opt/ros/{args.ros_distro}/setup.bash
        source {rinbo_ws}/install/setup.bash
        source {redrhex_ws}/install/setup.bash

        ros2 launch redrhex_lowlevel_bridge lowlevel_bridge.launch.py \\
          backend:=biorola_ros \\
          rinbo_allow_enable:=false \\
          rinbo_require_state:=true
        ```

        In a monitor terminal:

        ```bash
        source /opt/ros/{args.ros_distro}/setup.bash
        source {rinbo_ws}/install/setup.bash
        source {redrhex_ws}/install/setup.bash
        ros2 topic echo /redrhex/lowlevel_heartbeat
        ros2 topic echo /joint_states --once
        ros2 topic echo /motor_feedback --once
        ros2 topic echo /redrhex/lowlevel_diagnostics --once
        ```

        ## 7. RL controller dry-run

        ```bash
        source /opt/ros/{args.ros_distro}/setup.bash
        source {rinbo_ws}/install/setup.bash
        source {redrhex_ws}/install/setup.bash

        ros2 launch redrhex_rl_controller redrhex_policy_bringup.launch.py \\
          start_bridge:=false \\
          use_fake_sensors:=false \\
          onnx_path:={_shell_quote(args.onnx_path)} \\
          enable_policy_on_start:=false \\
          enable_motor_output_on_start:=false

        # Enable policy computation only after state reaches POLICY_READY.
        ros2 topic pub --once /redrhex/enable_policy std_msgs/msg/Bool "{{data: true}}"

        # Keep motor output disabled at this stage.
        ros2 topic echo /redrhex/state_machine_state
        ros2 topic echo /redrhex/policy_action_raw --once
        ros2 topic echo /redrhex/motor_commands --once
        ros2 topic echo /redrhex/rinbo_motor_command_preview --once
        ```
        {hardware_snippets}
        ## Stop rule

        Stop immediately if any of these appear:

        ```text
        BioRoLaROS2 hardcoded CORE_IP: ERROR
        /motor/command publishers > 1
        /redrhex/lowlevel_heartbeat=false
        rinbo_actual_publish_state=blocked_no_recent_state
        rinbo_actual_publish_state=blocked_duplicate_publishers
        observation NaN/Inf
        ONNX output NaN/Inf
        roll/pitch over safety limit
        ```

        Software stop helper, in any sourced Orin terminal:

        ```bash
        ros2 run redrhex_rl_controller estop_tool assert
        # Only after the scene is safe:
        ros2 run redrhex_rl_controller estop_tool clear --confirm-clear
        ```
        """
    ).strip()
    return _strip_template_indent(rendered) + "\n"


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    print(render(args))


if __name__ == "__main__":
    main()
