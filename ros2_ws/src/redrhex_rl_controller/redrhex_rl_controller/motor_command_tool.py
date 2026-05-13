"""Manual motor command publisher for staged RedRhex bringup.

This tool deliberately publishes through the same /redrhex/motor_commands
topic as the RL controller, so you can test the low-level bridge before
allowing policy takeover.
"""

from __future__ import annotations

import argparse
import json
import time

import rclpy
from rclpy.node import Node

from redrhex_msgs.msg import RedRhexMotorCommand

from . import redrhex_contract as C


def _base_command(enable: bool, mode: int) -> RedRhexMotorCommand:
    msg = RedRhexMotorCommand()
    msg.header.frame_id = "redrhex_base"
    msg.joint_names = C.MAIN_DRIVE_JOINT_NAMES + C.ABAD_JOINT_NAMES + C.DAMPER_JOINT_NAMES
    msg.target_position_rad = list(C.INIT_MAIN_DRIVE_POS) + list(C.INIT_ABAD_POS) + list(C.INIT_DAMPER_POS)
    msg.target_velocity_rad_s = [0.0] * 18
    msg.kp = [12.0] * 6 + [20.0] * 6 + [50.0] * 6
    msg.kd = [1.0] * 6 + [1.0] * 6 + [2.0] * 6
    msg.effort_limit_nm = [20.0] * 6 + [3.0] * 6 + [10.0] * 6
    msg.enable = bool(enable)
    msg.mode = int(mode)
    return msg


class MotorCommandTool(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("redrhex_motor_command_tool")
        self.args = args
        self.pub = self.create_publisher(RedRhexMotorCommand, "/redrhex/motor_commands", 10)

    def build_command(self) -> RedRhexMotorCommand:
        msg = _base_command(enable=self.args.enable, mode=self.args.mode_id)
        if self.args.mode == "disable":
            msg.enable = False
            msg.kp = [0.0] * 18
            msg.kd = [0.0] * 18
            msg.target_velocity_rad_s = [0.0] * 18
        elif self.args.mode == "init-stand":
            pass
        elif self.args.mode == "single-abad":
            idx = self.args.index
            if idx < 0 or idx >= 6:
                raise ValueError("--index must be 0..5 for single-abad")
            msg.target_position_rad[6 + idx] = float(self.args.position)
            msg.kp[6 + idx] = float(self.args.kp)
            msg.kd[6 + idx] = float(self.args.kd)
            msg.effort_limit_nm[6 + idx] = float(self.args.effort_limit)
        elif self.args.mode == "single-main-velocity":
            idx = self.args.index
            if idx < 0 or idx >= 6:
                raise ValueError("--index must be 0..5 for single-main-velocity")
            msg.kp[idx] = 0.0
            msg.kd[idx] = float(self.args.kd)
            msg.target_velocity_rad_s[idx] = float(self.args.velocity)
            msg.effort_limit_nm[idx] = float(self.args.effort_limit)
        elif self.args.mode == "all-abad":
            pos = float(self.args.position)
            for i in range(6):
                msg.target_position_rad[6 + i] = pos
                msg.kp[6 + i] = float(self.args.kp)
                msg.kd[6 + i] = float(self.args.kd)
                msg.effort_limit_nm[6 + i] = float(self.args.effort_limit)
        elif self.args.mode == "all-main-velocity":
            vel = float(self.args.velocity)
            for i in range(6):
                msg.kp[i] = 0.0
                msg.kd[i] = float(self.args.kd)
                msg.target_velocity_rad_s[i] = vel
                msg.effort_limit_nm[i] = float(self.args.effort_limit)
        else:
            raise ValueError(f"Unsupported mode {self.args.mode}")
        return msg

    @staticmethod
    def summarize_command(msg: RedRhexMotorCommand) -> dict:
        rows = []
        for idx, name in enumerate(msg.joint_names):
            rows.append(
                {
                    "index": idx,
                    "joint": name,
                    "pos_rad": round(float(msg.target_position_rad[idx]), 5),
                    "vel_rad_s": round(float(msg.target_velocity_rad_s[idx]), 5),
                    "kp": round(float(msg.kp[idx]), 5),
                    "kd": round(float(msg.kd[idx]), 5),
                    "effort_nm": round(float(msg.effort_limit_nm[idx]), 5),
                }
            )
        return {
            "enable": bool(msg.enable),
            "mode": int(msg.mode),
            "joint_count": len(msg.joint_names),
            "joints": rows,
        }

    def run(self) -> None:
        msg = self.build_command()
        if self.args.dry_run:
            print(json.dumps(self.summarize_command(msg), indent=2))
            return

        period = 1.0 / max(float(self.args.rate_hz), 1.0)
        end_time = time.monotonic() + max(float(self.args.duration), period)
        self.get_logger().warn(
            f"Publishing manual command mode={self.args.mode} enable={msg.enable} "
            f"duration={self.args.duration:.2f}s. Keep E-stop in hand."
        )
        while rclpy.ok() and time.monotonic() < end_time:
            msg.header.stamp = self.get_clock().now().to_msg()
            self.pub.publish(msg)
            rclpy.spin_once(self, timeout_sec=0.0)
            time.sleep(period)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=[
            "list-joints",
            "disable",
            "init-stand",
            "single-abad",
            "single-main-velocity",
            "all-abad",
            "all-main-velocity",
        ],
    )
    parser.add_argument("--enable", action="store_true", help="Actually enable motor output.")
    parser.add_argument(
        "--confirm-risk",
        action="store_true",
        help="Required together with --enable. Confirms you have E-stop and the robot is safe to move.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the command JSON and do not publish.")
    parser.add_argument("--index", type=int, default=0, help="Leg/joint index 0..5 in policy order.")
    parser.add_argument("--position", type=float, default=0.0, help="Target ABAD position in rad.")
    parser.add_argument("--velocity", type=float, default=0.3, help="Target main-drive velocity in rad/s.")
    parser.add_argument("--kp", type=float, default=8.0)
    parser.add_argument("--kd", type=float, default=0.5)
    parser.add_argument("--effort-limit", type=float, default=2.0)
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--rate-hz", type=float, default=50.0)
    parser.add_argument("--mode-id", type=int, default=2)
    return parser


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.mode == "list-joints":
        for idx, name in enumerate(C.MAIN_DRIVE_JOINT_NAMES + C.ABAD_JOINT_NAMES + C.DAMPER_JOINT_NAMES):
            print(f"{idx:02d}: {name}")
        return
    if args.enable and not args.confirm_risk:
        raise SystemExit("Refusing --enable without --confirm-risk. Keep E-stop in hand and rerun intentionally.")
    if args.enable and args.mode in ("all-main-velocity", "single-main-velocity") and abs(args.velocity) > 1.0:
        raise SystemExit("Refusing velocity > 1.0 rad/s in manual tool. Increase only after editing the code intentionally.")
    if args.enable and args.mode in ("single-abad", "all-abad") and abs(args.position) > 0.25:
        raise SystemExit("Refusing ABAD position > 0.25 rad in manual tool. Increase only after bench validation.")
    if args.duration > 10.0:
        raise SystemExit("Refusing duration > 10 s in manual tool.")

    rclpy.init()
    node = MotorCommandTool(args)
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()
