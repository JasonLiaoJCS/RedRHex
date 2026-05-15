"""BioRoLaROS2/RhexROS2 sbRIO bringup checker for RedRhex deployment."""

from __future__ import annotations

import argparse
import os
import re
import socket
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

try:
    from rclpy._rclpy_pybind11 import RCLError
except Exception:  # pragma: no cover - depends on rclpy version
    RCLError = RuntimeError


@dataclass
class CheckResult:
    level: str
    name: str
    detail: str


def _parse_master_addr(value: str | None) -> tuple[str | None, int | None]:
    if not value:
        return None, None
    if ":" not in value:
        return value, 50051
    host, port_text = value.rsplit(":", 1)
    try:
        return host, int(port_text)
    except ValueError:
        return host, None


def _local_ipv4_addrs() -> list[str]:
    addrs: set[str] = set()
    try:
        output = subprocess.check_output(["hostname", "-I"], text=True, timeout=1.0)
        addrs.update(token for token in output.split() if "." in token)
    except Exception:
        pass
    try:
        _hostname, _aliases, host_addrs = socket.gethostbyname_ex(socket.gethostname())
        addrs.update(host_addrs)
    except OSError:
        pass
    addrs.discard("127.0.0.1")
    return sorted(addrs)


class RinboBringupCheck(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("redrhex_rinbo_bringup_check")
        self.args = args
        self.results: list[CheckResult] = []
        self.motor_state_seen = False
        self.power_state_seen = False
        self.last_motor_state = None
        self.last_power_state = None
        try:
            from rinbo_msgs.msg import MotorCmdStamped, MotorStateStamped, PowerCmdStamped, PowerStateStamped
        except Exception as exc:  # pragma: no cover - requires external BioRoLaROS2 overlay
            self.rinbo_import_error = exc
            self.MotorCmdStamped = None
            self.MotorStateStamped = None
            self.PowerCmdStamped = None
            self.PowerStateStamped = None
        else:
            self.rinbo_import_error = None
            self.MotorCmdStamped = MotorCmdStamped
            self.MotorStateStamped = MotorStateStamped
            self.PowerCmdStamped = PowerCmdStamped
            self.PowerStateStamped = PowerStateStamped
            self.create_subscription(MotorStateStamped, self.args.motor_state_topic, self._on_motor_state, 10)
            self.create_subscription(PowerStateStamped, self.args.power_state_topic, self._on_power_state, 10)

    def add(self, level: str, name: str, detail: str) -> None:
        self.results.append(CheckResult(level, name, detail))

    def _on_motor_state(self, msg) -> None:
        self.motor_state_seen = True
        self.last_motor_state = msg

    def _on_power_state(self, msg) -> None:
        self.power_state_seen = True
        self.last_power_state = msg

    def check_env(self) -> None:
        ros_distro = os.environ.get("ROS_DISTRO")
        self.add("OK" if ros_distro else "WARN", "ROS_DISTRO", ros_distro or "not set")

        master = self.args.master_addr or os.environ.get("CORE_MASTER_ADDR")
        local_ip = self.args.local_ip or os.environ.get("CORE_LOCAL_IP")
        core_ip = os.environ.get("CORE_IP")
        host, port = _parse_master_addr(master)
        if host and port:
            self.add("OK", "CORE_MASTER_ADDR", f"{host}:{port}")
        else:
            self.add("ERROR", "CORE_MASTER_ADDR", "not set or invalid; expected <sbRIO_ip>:50051")
        local_addrs = _local_ipv4_addrs()
        if local_ip and (not local_addrs or local_ip in local_addrs):
            detail = local_ip if not local_addrs else f"{local_ip} in host IPs {local_addrs}"
            self.add("OK", "CORE_LOCAL_IP", detail)
        elif local_ip:
            self.add("WARN", "CORE_LOCAL_IP", f"{local_ip} not in host IPs {local_addrs}")
        else:
            self.add("WARN", "CORE_LOCAL_IP", f"not set; host IPs {local_addrs}")
        if core_ip and host and core_ip != host:
            self.add(
                "WARN",
                "CORE_IP",
                f"{core_ip} differs from CORE_MASTER_ADDR host {host}. BioRoLaROS2 bridge source may also hardcode CORE_IP.",
            )
        elif core_ip:
            self.add("OK", "CORE_IP", core_ip)
        else:
            self.add(
                "WARN",
                "CORE_IP",
                "not set in this shell. If BioRoLaROS2 bridge cannot connect, check rinbo_ros_bridge.cpp hardcoded CORE_IP.",
            )

        if host and port:
            try:
                with socket.create_connection((host, port), timeout=self.args.tcp_timeout_s):
                    pass
            except OSError as exc:
                self.add("WARN", "TCP 50051", f"cannot connect to {host}:{port}: {exc}")
            else:
                self.add("OK", "TCP 50051", f"connected to {host}:{port}")

    def check_bridge_source(self) -> None:
        source_arg = str(self.args.bridge_source or "").strip()
        if not source_arg:
            return
        source_path = Path(source_arg).expanduser()
        if not source_path.exists():
            self.add(
                "WARN",
                "BioRoLaROS2 CORE_IP source check",
                f"{source_path} not found; skip hardcoded CORE_IP check",
            )
            return
        try:
            text = source_path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            self.add("WARN", "BioRoLaROS2 CORE_IP source check", f"cannot read {source_path}: {exc}")
            return
        match = re.search(r'setenv\s*\(\s*"CORE_IP"\s*,\s*"([^"]+)"', text)
        master = self.args.master_addr or os.environ.get("CORE_MASTER_ADDR")
        host, _port = _parse_master_addr(master)
        if not match:
            self.add(
                "OK",
                "BioRoLaROS2 CORE_IP source check",
                f"{source_path} has no obvious hardcoded setenv(\"CORE_IP\", ...)",
            )
            return
        hardcoded_ip = match.group(1)
        if not host:
            self.add(
                "WARN",
                "BioRoLaROS2 hardcoded CORE_IP",
                f"{source_path} hardcodes CORE_IP={hardcoded_ip}; set CORE_MASTER_ADDR=<same_ip>:50051 before bringup.",
            )
        elif hardcoded_ip != host:
            self.add(
                "ERROR",
                "BioRoLaROS2 hardcoded CORE_IP",
                (
                    f"{source_path} hardcodes CORE_IP={hardcoded_ip}, but CORE_MASTER_ADDR host is {host}. "
                    "Edit rinbo_ros_bridge.cpp and rebuild BioRoLaROS2 before hardware bringup."
                ),
            )
        else:
            self.add("OK", "BioRoLaROS2 hardcoded CORE_IP", f"{hardcoded_ip} matches CORE_MASTER_ADDR host")

    def check_rinbo_msgs(self) -> None:
        if self.rinbo_import_error is None:
            self.add("OK", "rinbo_msgs", "MotorStateStamped and PowerStateStamped importable")
            self.check_message_contract()
        else:
            self.add("ERROR", "rinbo_msgs", f"not importable: {self.rinbo_import_error}")

    def _missing_fields(self, obj, fields: list[str]) -> list[str]:
        return [field for field in fields if not hasattr(obj, field)]

    def check_message_contract(self) -> None:
        assert self.MotorCmdStamped is not None
        assert self.MotorStateStamped is not None
        assert self.PowerCmdStamped is not None
        assert self.PowerStateStamped is not None

        errors: list[str] = []
        motor_cmd = self.MotorCmdStamped()
        motor_state = self.MotorStateStamped()
        power_cmd = self.PowerCmdStamped()
        power_state = self.PowerStateStamped()

        leg_names = ["l1", "l2", "l3", "r1", "r2", "r3"]
        servo_names = ["sl1", "sl2", "sl3", "sr1", "sr2", "sr3"]
        missing_motor_cmd = self._missing_fields(motor_cmd, ["header", "servo_control_mode"] + leg_names + servo_names)
        if missing_motor_cmd:
            errors.append(f"MotorCmdStamped missing {missing_motor_cmd}")
        missing_motor_state = self._missing_fields(motor_state, ["header", "servo_control_mode"] + leg_names + servo_names)
        if missing_motor_state:
            errors.append(f"MotorStateStamped missing {missing_motor_state}")

        for leg_name in leg_names:
            if hasattr(motor_cmd, leg_name):
                missing = self._missing_fields(
                    getattr(motor_cmd, leg_name),
                    ["enable", "direction", "voltage", "state", "reset_position"],
                )
                if missing:
                    errors.append(f"MotorCmdStamped.{leg_name} missing {missing}")
            if hasattr(motor_state, leg_name):
                missing = self._missing_fields(getattr(motor_state, leg_name), ["position", "tick_count", "hall_effect"])
                if missing:
                    errors.append(f"MotorStateStamped.{leg_name} missing {missing}")

        for servo_name in servo_names:
            if hasattr(motor_cmd, servo_name):
                missing = self._missing_fields(getattr(motor_cmd, servo_name), ["position_encoder"])
                if missing:
                    errors.append(f"MotorCmdStamped.{servo_name} missing {missing}")
            if hasattr(motor_state, servo_name):
                missing = self._missing_fields(getattr(motor_state, servo_name), ["position_encoder"])
                if missing:
                    errors.append(f"MotorStateStamped.{servo_name} missing {missing}")

        missing_power_cmd = self._missing_fields(power_cmd, ["header", "digital", "signal", "power", "clean", "trigger"])
        if missing_power_cmd:
            errors.append(f"PowerCmdStamped missing {missing_power_cmd}")
        power_state_fields = ["header", "digital", "signal", "power", "clean"]
        for idx in range(8):
            power_state_fields.extend([f"v_{idx}", f"i_{idx}"])
        missing_power_state = self._missing_fields(power_state, power_state_fields)
        if missing_power_state:
            errors.append(f"PowerStateStamped missing {missing_power_state}")

        if errors:
            self.add("ERROR", "BioRoLaROS2 message contract", "; ".join(errors))
        else:
            self.add(
                "OK",
                "BioRoLaROS2 message contract",
                "MotorCmd/State, PowerCmd/State fields match expected rinbo_msgs interface",
            )

    def check_topics(self) -> None:
        deadline = time.monotonic() + max(float(self.args.discovery_timeout_s), 0.0)
        topic_map: dict[str, list[str]] = {}
        while rclpy.ok() and time.monotonic() < deadline:
            topic_map = dict(self.get_topic_names_and_types())
            required = [self.args.motor_state_topic, self.args.power_state_topic, self.args.motor_command_topic]
            if all(topic in topic_map for topic in required):
                break
            rclpy.spin_once(self, timeout_sec=0.05)

        for topic in [self.args.motor_command_topic, self.args.motor_state_topic, self.args.power_command_topic, self.args.power_state_topic]:
            if topic in topic_map:
                self.add("OK", f"topic {topic}", ",".join(topic_map[topic]))
            else:
                self.add("WARN", f"topic {topic}", "not discovered yet")

        self.check_graph_endpoints()

    def _endpoint_names(self, infos) -> str:
        names = []
        for info in infos:
            node_name = getattr(info, "node_name", "")
            node_namespace = getattr(info, "node_namespace", "")
            full_name = f"{node_namespace.rstrip('/')}/{node_name}".replace("//", "/")
            names.append(full_name or "<unknown>")
        return ", ".join(names) if names else "none"

    def check_graph_endpoints(self) -> None:
        motor_command_publishers = self.get_publishers_info_by_topic(self.args.motor_command_topic)
        motor_command_publisher_count = len(motor_command_publishers)
        if motor_command_publisher_count <= self.args.max_motor_command_publishers:
            self.add(
                "OK",
                f"publishers {self.args.motor_command_topic}",
                f"{motor_command_publisher_count}: {self._endpoint_names(motor_command_publishers)}",
            )
        else:
            self.add(
                "ERROR",
                f"publishers {self.args.motor_command_topic}",
                (
                    f"{motor_command_publisher_count}: {self._endpoint_names(motor_command_publishers)}. "
                    "Stop rinbo_tripod/rinbo_standing or duplicate RL bridge nodes before continuing."
                ),
            )

        power_command_subscribers = self.get_subscriptions_info_by_topic(self.args.power_command_topic)
        if power_command_subscribers:
            self.add(
                "OK",
                f"subscribers {self.args.power_command_topic}",
                f"{len(power_command_subscribers)}: {self._endpoint_names(power_command_subscribers)}",
            )
        else:
            self.add(
                "WARN",
                f"subscribers {self.args.power_command_topic}",
                "none. Start rinbo_ros_bridge before using biorola_power_tool.",
            )

        motor_state_publishers = self.get_publishers_info_by_topic(self.args.motor_state_topic)
        if motor_state_publishers:
            self.add(
                "OK",
                f"publishers {self.args.motor_state_topic}",
                f"{len(motor_state_publishers)}: {self._endpoint_names(motor_state_publishers)}",
            )
        else:
            self.add("WARN", f"publishers {self.args.motor_state_topic}", "none")

    def wait_for_messages(self) -> None:
        if self.rinbo_import_error is not None:
            return
        deadline = time.monotonic() + max(float(self.args.message_timeout_s), 0.0)
        while rclpy.ok() and time.monotonic() < deadline:
            if self.motor_state_seen and (self.power_state_seen or not self.args.require_power_state):
                break
            rclpy.spin_once(self, timeout_sec=0.05)
        self.add("OK" if self.motor_state_seen else "WARN", self.args.motor_state_topic, "message received" if self.motor_state_seen else "no message before timeout")
        if self.last_motor_state is not None:
            self.add("OK", "motor_state summary", self._summarize_motor_state(self.last_motor_state))
        if self.args.require_power_state:
            self.add("OK" if self.power_state_seen else "WARN", self.args.power_state_topic, "message received" if self.power_state_seen else "no message before timeout")
        if self.last_power_state is not None:
            self.add("OK", "power_state summary", self._summarize_power_state(self.last_power_state))

    def _summarize_power_state(self, msg) -> str:
        channels = []
        for idx in range(8):
            voltage = float(getattr(msg, f"v_{idx}", 0.0))
            current = float(getattr(msg, f"i_{idx}", 0.0))
            if abs(voltage) > 1.0e-6 or abs(current) > 1.0e-6:
                channels.append(f"ch{idx}={voltage:.2f}V/{current:.2f}A")
        rail_summary = "; rails " + ", ".join(channels) if channels else "; rails no nonzero readings"
        return (
            f"digital={bool(getattr(msg, 'digital', False))} "
            f"signal={bool(getattr(msg, 'signal', False))} "
            f"power={bool(getattr(msg, 'power', False))}"
            f"{rail_summary}"
        )

    def _summarize_motor_state(self, msg) -> str:
        positions = []
        for field in ["l1", "l2", "l3", "r1", "r2", "r3"]:
            if hasattr(msg, field):
                leg = getattr(msg, field)
                positions.append(f"{field}={float(getattr(leg, 'position', 0.0)):.1f}")
        servos = []
        for field in ["sl1", "sl2", "sl3", "sr1", "sr2", "sr3"]:
            if hasattr(msg, field):
                servo = getattr(msg, field)
                servos.append(f"{field}={int(getattr(servo, 'position_encoder', 0))}")
        chunks = []
        if positions:
            chunks.append("positions " + ", ".join(positions))
        if servos:
            chunks.append("servos " + ", ".join(servos))
        return "; ".join(chunks) if chunks else "message received"

    def run(self) -> int:
        self.check_env()
        self.check_bridge_source()
        self.check_rinbo_msgs()
        self.check_topics()
        self.wait_for_messages()
        has_error = any(result.level == "ERROR" for result in self.results)
        has_warn = any(result.level == "WARN" for result in self.results)
        for result in self.results:
            print(f"[{result.level:5}] {result.name}: {result.detail}")
        if has_error:
            print("\nFix ERROR items before continuing.")
            return 2
        if has_warn:
            print("\nWARN items may be normal before rinbo_ros_bridge/sbRIO is running, but do not run RL on hardware yet.")
            return 1 if self.args.strict else 0
        print("\nBioRoLaROS2 bridge checks look good.")
        return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check BioRoLaROS2/RhexROS2 sbRIO environment before RedRhex RL bringup.")
    parser.add_argument("--master-addr", default=None, help="Override CORE_MASTER_ADDR, e.g. 192.168.0.100:50051")
    parser.add_argument("--local-ip", default=None, help="Override CORE_LOCAL_IP for display/check only.")
    parser.add_argument("--tcp-timeout-s", type=float, default=1.0)
    parser.add_argument(
        "--bridge-source",
        default="~/rinbo_ros_ws/src/rinbo_ros_bridge/src/rinbo_ros_bridge.cpp",
        help="BioRoLaROS2 rinbo_ros_bridge.cpp path used to detect hardcoded CORE_IP.",
    )
    parser.add_argument("--discovery-timeout-s", type=float, default=2.0)
    parser.add_argument("--message-timeout-s", type=float, default=3.0)
    parser.add_argument("--motor-command-topic", default="/motor/command")
    parser.add_argument("--motor-state-topic", default="/motor/state")
    parser.add_argument("--power-command-topic", default="/power/command")
    parser.add_argument("--power-state-topic", default="/power/state")
    parser.add_argument("--max-motor-command-publishers", type=int, default=1)
    parser.add_argument("--require-power-state", action="store_true")
    parser.add_argument("--strict", action="store_true", help="Return nonzero on WARN as well as ERROR.")
    return parser


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    rclpy.init()
    node = RinboBringupCheck(args)
    try:
        code = node.run()
    except (KeyboardInterrupt, ExternalShutdownException, RCLError):
        code = 130
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    raise SystemExit(code)
