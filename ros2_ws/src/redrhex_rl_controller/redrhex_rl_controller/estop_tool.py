"""Software E-stop helper for RedRhex ROS2 bringup."""

from __future__ import annotations

import argparse
import json
import time


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Publish RedRhex software E-stop commands.")
    parser.add_argument("mode", choices=["assert", "clear"])
    parser.add_argument("--topic", default="/estop")
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--period-s", type=float, default=0.05)
    parser.add_argument("--wait-for-subscriber-s", type=float, default=2.0)
    parser.add_argument("--allow-no-subscriber", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--confirm-clear",
        action="store_true",
        help="Required for mode=clear. Check physical E-stop, robot support, and scene safety first.",
    )
    return parser


def _print_dry_run(args: argparse.Namespace) -> None:
    print(
        json.dumps(
            {
                "topic": args.topic,
                "estop": bool(args.mode == "assert"),
                "repeat": int(args.repeat),
                "period_s": float(args.period_s),
                "dry_run": True,
            },
            indent=2,
        )
    )


def _run_ros(args: argparse.Namespace) -> None:
    import rclpy
    from rclpy.executors import ExternalShutdownException
    from rclpy.node import Node
    from std_msgs.msg import Bool

    try:
        from rclpy._rclpy_pybind11 import RCLError
    except Exception:  # pragma: no cover - depends on rclpy version
        RCLError = RuntimeError

    class EstopTool(Node):
        def __init__(self, parsed_args: argparse.Namespace) -> None:
            super().__init__("redrhex_estop_tool")
            self.args = parsed_args
            self.pub = self.create_publisher(Bool, parsed_args.topic, 10)

        def wait_for_subscribers(self) -> None:
            deadline = time.monotonic() + max(float(self.args.wait_for_subscriber_s), 0.0)
            while rclpy.ok() and time.monotonic() < deadline:
                if self.pub.get_subscription_count() > 0:
                    return
                rclpy.spin_once(self, timeout_sec=0.05)
            if self.pub.get_subscription_count() == 0 and not self.args.allow_no_subscriber:
                raise RuntimeError(
                    f"No subscriber on {self.args.topic}. Start redrhex_rl_controller first, "
                    "or rerun with --allow-no-subscriber for a graph-level test."
                )

        def publish(self, asserted: bool) -> None:
            self.wait_for_subscribers()
            msg = Bool()
            msg.data = bool(asserted)
            for _ in range(max(1, int(self.args.repeat))):
                self.pub.publish(msg)
                rclpy.spin_once(self, timeout_sec=0.02)
                time.sleep(max(0.0, float(self.args.period_s)))
            state = "ASSERTED" if asserted else "CLEARED"
            self.get_logger().warn(f"Software E-stop {state} on {self.args.topic}")

    rclpy.init()
    node = EstopTool(args)
    try:
        node.publish(args.mode == "assert")
    except (KeyboardInterrupt, ExternalShutdownException, RCLError):
        pass
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.mode == "clear" and not args.confirm_clear and not args.dry_run:
        raise SystemExit("Refusing to clear software E-stop without --confirm-clear.")
    if args.repeat <= 0:
        raise SystemExit("--repeat must be positive.")
    if args.period_s < 0.0:
        raise SystemExit("--period-s must be non-negative.")
    if args.dry_run:
        _print_dry_run(args)
        return
    _run_ros(args)


if __name__ == "__main__":
    main()
