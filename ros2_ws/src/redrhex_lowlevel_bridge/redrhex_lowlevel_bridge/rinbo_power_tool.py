"""Power command helper for the BioRoLaROS2/RhexROS2 sbRIO stack."""

from __future__ import annotations

import argparse
import json
import time

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

try:
    from rclpy._rclpy_pybind11 import RCLError
except Exception:  # pragma: no cover - depends on rclpy version
    RCLError = RuntimeError


POWER_STATES = {
    "off": (False, False, False),
    "digital": (True, False, False),
    "sensors": (True, True, False),
    "relay": (True, True, True),
}


def _summary(topic: str, digital: bool, signal: bool, power: bool, clean: bool, trigger: bool) -> dict:
    return {
        "topic": topic,
        "digital": bool(digital),
        "signal": bool(signal),
        "power": bool(power),
        "clean": bool(clean),
        "trigger": bool(trigger),
    }


def _power_channels(msg) -> list[dict[str, float]]:
    channels = []
    for idx in range(8):
        voltage = float(getattr(msg, f"v_{idx}", 0.0))
        current = float(getattr(msg, f"i_{idx}", 0.0))
        channels.append({"index": idx, "voltage_v": voltage, "current_a": current})
    return channels


def print_dry_run(args: argparse.Namespace) -> None:
    if args.mode == "status":
        print(json.dumps({"state_topic": args.state_topic, "mode": "status", "dry_run": True}, indent=2))
        return
    modes = ["digital", "sensors"]
    if args.mode == "sequence":
        if args.include_relay:
            modes.append("relay")
    else:
        modes = [args.mode]
    payload = [
        _summary(args.topic, *POWER_STATES[mode], clean=args.clean, trigger=args.trigger)
        for mode in modes
    ]
    print(json.dumps(payload[0] if len(payload) == 1 else payload, indent=2))


class RinboPowerTool(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("redrhex_rinbo_power_tool")
        self.args = args
        try:
            from rinbo_msgs.msg import PowerCmdStamped, PowerStateStamped
        except Exception as exc:  # pragma: no cover - requires external RhexROS2 overlay
            raise RuntimeError(
                "rinbo_msgs is required. Build/source ~/rinbo_ros_ws first: source ~/rinbo_ros_ws/install/setup.bash"
            ) from exc
        self.PowerCmdStamped = PowerCmdStamped
        self.PowerStateStamped = PowerStateStamped
        self.last_power_state = None
        self.pub = self.create_publisher(PowerCmdStamped, args.topic, 10)
        self.create_subscription(PowerStateStamped, args.state_topic, self._on_power_state, 10)
        self.sequence = 0

    def _on_power_state(self, msg) -> None:
        self.last_power_state = msg

    def wait_for_subscriber(self) -> None:
        if self.args.dry_run:
            return
        deadline = time.monotonic() + max(float(self.args.wait_for_subscriber_s), 0.0)
        while rclpy.ok() and time.monotonic() < deadline:
            if self.pub.get_subscription_count() > 0:
                return
            rclpy.spin_once(self, timeout_sec=0.05)
        if self.pub.get_subscription_count() == 0 and not self.args.allow_no_subscriber:
            raise RuntimeError(
                f"No subscriber on {self.args.topic}. Start/source rinbo_ros_bridge before sending power commands."
            )

    def build_msg(self, digital: bool, signal: bool, power: bool):
        self.sequence += 1
        msg = self.PowerCmdStamped()
        msg.header.seq = self.sequence
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "redrhex_power"
        msg.digital = bool(digital)
        msg.signal = bool(signal)
        msg.power = bool(power)
        if hasattr(msg, "clean"):
            msg.clean = bool(self.args.clean)
        if hasattr(msg, "trigger"):
            msg.trigger = bool(self.args.trigger)
        return msg

    def summarize(self, msg) -> dict:
        return _summary(
            self.args.topic,
            bool(msg.digital),
            bool(msg.signal),
            bool(msg.power),
            bool(getattr(msg, "clean", False)),
            bool(getattr(msg, "trigger", False)),
        )

    def summarize_power_state(self, msg) -> dict:
        summary = self.summarize(msg)
        summary["topic"] = self.args.state_topic
        channels = _power_channels(msg)
        summary["channels"] = channels
        summary["max_current_a"] = max((abs(ch["current_a"]) for ch in channels), default=0.0)
        nonzero_voltages = [ch["voltage_v"] for ch in channels if abs(ch["voltage_v"]) > 1.0e-6]
        summary["min_nonzero_voltage_v"] = min(nonzero_voltages) if nonzero_voltages else 0.0
        return summary

    def read_status(self) -> None:
        deadline = time.monotonic() + max(float(self.args.status_timeout_s), 0.0)
        while rclpy.ok() and time.monotonic() < deadline:
            if self.last_power_state is not None:
                print(json.dumps(self.summarize_power_state(self.last_power_state), indent=2))
                return
            rclpy.spin_once(self, timeout_sec=0.05)
        raise RuntimeError(f"No message received on {self.args.state_topic} before timeout.")

    def publish_state(self, digital: bool, signal: bool, power: bool) -> None:
        msg = self.build_msg(digital, signal, power)
        if self.args.dry_run:
            print(json.dumps(self.summarize(msg), indent=2))
            return
        self.wait_for_subscriber()
        for _ in range(max(int(self.args.repeat), 1)):
            msg.header.stamp = self.get_clock().now().to_msg()
            self.pub.publish(msg)
            rclpy.spin_once(self, timeout_sec=0.05)
            time.sleep(max(float(self.args.repeat_delay_s), 0.0))
        self.get_logger().warn(
            f"Published power command digital={msg.digital} signal={msg.signal} power={msg.power}"
        )

    def run(self) -> None:
        if self.args.mode == "status":
            self.read_status()
            return
        if self.args.mode == "sequence":
            sequence = ["digital", "sensors"]
            if self.args.include_relay:
                sequence.append("relay")
            for mode in sequence:
                self.publish_state(*POWER_STATES[mode])
                time.sleep(max(float(self.args.step_delay_s), 0.0))
            return
        self.publish_state(*POWER_STATES[self.args.mode])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Publish BioRoLaROS2/RhexROS2 rinbo_msgs/PowerCmdStamped safely.")
    parser.add_argument("mode", choices=["off", "digital", "sensors", "relay", "sequence", "status"])
    parser.add_argument("--topic", default="/power/command")
    parser.add_argument("--state-topic", default="/power/state")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--repeat-delay-s", type=float, default=0.05)
    parser.add_argument("--step-delay-s", type=float, default=0.5)
    parser.add_argument("--wait-for-subscriber-s", type=float, default=2.0)
    parser.add_argument("--status-timeout-s", type=float, default=3.0)
    parser.add_argument(
        "--allow-no-subscriber",
        action="store_true",
        help="Publish even if /power/command has no visible subscriber. Usually not recommended.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--trigger", action="store_true")
    parser.add_argument(
        "--include-relay",
        action="store_true",
        help="For sequence mode, also publish power=true after digital and signal.",
    )
    parser.add_argument(
        "--confirm-relay",
        action="store_true",
        help="Required for any command that sets power=true.",
    )
    return parser


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    wants_relay = args.mode == "relay" or (args.mode == "sequence" and args.include_relay)
    if wants_relay and not args.confirm_relay and not args.dry_run:
        raise SystemExit("Refusing power=true without --confirm-relay. Keep E-stop ready and rerun intentionally.")
    if args.dry_run:
        print_dry_run(args)
        return

    rclpy.init()
    node = None
    try:
        node = RinboPowerTool(args)
        node.run()
    except (KeyboardInterrupt, ExternalShutdownException, RCLError):
        pass
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
