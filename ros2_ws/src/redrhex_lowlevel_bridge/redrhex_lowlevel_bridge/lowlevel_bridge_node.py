"""ROS2 node bridging RedRhexMotorCommand to a replaceable low-level backend."""

from __future__ import annotations

import rclpy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Bool

from redrhex_msgs.msg import RedRhexMotorCommand, RedRhexMotorState

from .mock_bridge import MockLowLevelBridge
from .rinbo_ros_backend import RinboRosBackend
from .serial_bridge import SerialLowLevelBridge
from .sbrio_udp_bridge import SbrioUdpBridge

try:
    from rclpy._rclpy_pybind11 import RCLError
except Exception:  # pragma: no cover - depends on rclpy version
    RCLError = RuntimeError


MAIN_JOINT_NAMES_POLICY_ORDER = [
    "Revolute_15",
    "Revolute_7",
    "Revolute_12",
    "Revolute_18",
    "Revolute_23",
    "Revolute_24",
]


class LowLevelBridgeNode(Node):
    def __init__(self) -> None:
        super().__init__("redrhex_lowlevel_bridge")
        self.declare_parameter("backend", "mock")
        self.declare_parameter("mock.print_every_n", 50)
        self.declare_parameter("serial.port", "/dev/ttyUSB0")
        self.declare_parameter("serial.baudrate", 921600)
        self.declare_parameter("serial.timeout_s", 0.005)
        self.declare_parameter("serial.allow_enable", False)
        self.declare_parameter("sbrio.remote_host", "192.168.0.2")
        self.declare_parameter("sbrio.command_port", 15000)
        self.declare_parameter("sbrio.bind_host", "0.0.0.0")
        self.declare_parameter("sbrio.feedback_port", 15001)
        self.declare_parameter("sbrio.timeout_s", 0.002)
        self.declare_parameter("sbrio.heartbeat_timeout_s", 0.25)
        self.declare_parameter("sbrio.allow_enable", False)
        self.declare_parameter("sbrio.require_feedback", False)
        self.declare_parameter("rinbo.command_topic", "/motor/command")
        self.declare_parameter("rinbo.state_topic", "/motor/state")
        self.declare_parameter("rinbo.joint_state_topic", "/joint_states")
        self.declare_parameter("rinbo.preview_topic", "/redrhex/rinbo_motor_command_preview")
        self.declare_parameter("rinbo.publish_preview", True)
        self.declare_parameter("rinbo.allow_enable", False)
        self.declare_parameter("rinbo.publish_when_disabled", False)
        self.declare_parameter("rinbo.disabled_servo_control_mode", 0)
        self.declare_parameter("rinbo.publish_shutdown_disable", True)
        self.declare_parameter("rinbo.shutdown_disable_repeats", 5)
        self.declare_parameter("rinbo.shutdown_disable_period_s", 0.02)
        self.declare_parameter("rinbo.require_state", True)
        self.declare_parameter("rinbo.block_if_duplicate_command_publishers", True)
        self.declare_parameter("rinbo.state_timeout_s", 0.25)
        self.declare_parameter("rinbo.main_position_counts_per_rev", 54984.83)
        self.declare_parameter("rinbo.main_pwm_per_rad_s", 120.0)
        self.declare_parameter("rinbo.main_max_pwm", 500.0)
        self.declare_parameter("rinbo.main_encoder_zero_counts_rinbo_order", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.declare_parameter("rinbo.main_encoder_sign_rinbo_order", [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0])
        self.declare_parameter("rinbo.main_velocity_sign_policy_order", [1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        self.declare_parameter("rinbo.main_direction_positive_rinbo_order", [True, True, True, False, False, False])
        self.declare_parameter("rinbo.main_velocity_filter_alpha", 0.35)
        self.declare_parameter("rinbo.main_velocity_max_dt_s", 0.20)
        self.declare_parameter("rinbo.main_velocity_clip_rad_s", 80.0)
        self.declare_parameter("rinbo.abad_encoder_zero_rinbo_order", [740, 2565, 3283, 1944, 2071, 989])
        self.declare_parameter("rinbo.abad_encoder_counts_per_rad", 1000.0)
        self.declare_parameter("rinbo.abad_encoder_min", 0)
        self.declare_parameter("rinbo.abad_encoder_max", 65535)
        self.declare_parameter("rinbo.abad_sign_rinbo_order", [1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        self.declare_parameter("rinbo.servo_control_mode", 2)
        self.declare_parameter("feedback_rate_hz", 50.0)

        backend = str(self.get_parameter("backend").value)
        if backend == "mock":
            self.bridge = MockLowLevelBridge(int(self.get_parameter("mock.print_every_n").value))
        elif backend == "serial":
            self.bridge = SerialLowLevelBridge(
                str(self.get_parameter("serial.port").value),
                int(self.get_parameter("serial.baudrate").value),
                float(self.get_parameter("serial.timeout_s").value),
                bool(self.get_parameter("serial.allow_enable").value),
            )
        elif backend == "sbrio_udp":
            self.bridge = SbrioUdpBridge(
                str(self.get_parameter("sbrio.remote_host").value),
                int(self.get_parameter("sbrio.command_port").value),
                str(self.get_parameter("sbrio.bind_host").value),
                int(self.get_parameter("sbrio.feedback_port").value),
                float(self.get_parameter("sbrio.timeout_s").value),
                float(self.get_parameter("sbrio.heartbeat_timeout_s").value),
                bool(self.get_parameter("sbrio.allow_enable").value),
                bool(self.get_parameter("sbrio.require_feedback").value),
            )
        elif backend in ("rinbo_ros", "biorola_ros"):
            self.bridge = RinboRosBackend(
                self,
                str(self.get_parameter("rinbo.command_topic").value),
                str(self.get_parameter("rinbo.state_topic").value),
                str(self.get_parameter("rinbo.joint_state_topic").value),
                str(self.get_parameter("rinbo.preview_topic").value),
                bool(self.get_parameter("rinbo.publish_preview").value),
                bool(self.get_parameter("rinbo.allow_enable").value),
                bool(self.get_parameter("rinbo.publish_when_disabled").value),
                int(self.get_parameter("rinbo.disabled_servo_control_mode").value),
                bool(self.get_parameter("rinbo.publish_shutdown_disable").value),
                int(self.get_parameter("rinbo.shutdown_disable_repeats").value),
                float(self.get_parameter("rinbo.shutdown_disable_period_s").value),
                bool(self.get_parameter("rinbo.require_state").value),
                bool(self.get_parameter("rinbo.block_if_duplicate_command_publishers").value),
                float(self.get_parameter("rinbo.state_timeout_s").value),
                float(self.get_parameter("rinbo.main_position_counts_per_rev").value),
                float(self.get_parameter("rinbo.main_pwm_per_rad_s").value),
                float(self.get_parameter("rinbo.main_max_pwm").value),
                list(self.get_parameter("rinbo.main_encoder_zero_counts_rinbo_order").value),
                list(self.get_parameter("rinbo.main_encoder_sign_rinbo_order").value),
                list(self.get_parameter("rinbo.main_velocity_sign_policy_order").value),
                list(self.get_parameter("rinbo.main_direction_positive_rinbo_order").value),
                float(self.get_parameter("rinbo.main_velocity_filter_alpha").value),
                float(self.get_parameter("rinbo.main_velocity_max_dt_s").value),
                float(self.get_parameter("rinbo.main_velocity_clip_rad_s").value),
                list(self.get_parameter("rinbo.abad_encoder_zero_rinbo_order").value),
                float(self.get_parameter("rinbo.abad_encoder_counts_per_rad").value),
                int(self.get_parameter("rinbo.abad_encoder_min").value),
                int(self.get_parameter("rinbo.abad_encoder_max").value),
                list(self.get_parameter("rinbo.abad_sign_rinbo_order").value),
                int(self.get_parameter("rinbo.servo_control_mode").value),
                MAIN_JOINT_NAMES_POLICY_ORDER,
            )
        else:
            raise ValueError(
                f"Unknown low-level backend '{backend}'. Expected mock, serial, sbrio_udp, rinbo_ros, or biorola_ros."
            )
        self.backend = backend
        self.bridge.connect()

        self.create_subscription(RedRhexMotorCommand, "/redrhex/motor_commands", self._on_motor_command, 10)
        self.feedback_pub = self.create_publisher(RedRhexMotorState, "/motor_feedback", 10)
        self.heartbeat_pub = self.create_publisher(Bool, "/redrhex/lowlevel_heartbeat", 10)
        self.diag_pub = self.create_publisher(DiagnosticArray, "/redrhex/lowlevel_diagnostics", 10)

        rate = float(self.get_parameter("feedback_rate_hz").value)
        self.timer = self.create_timer(1.0 / max(rate, 1.0), self._tick)
        self.get_logger().info(f"Low-level bridge started with backend={backend}")

    def _on_motor_command(self, msg: RedRhexMotorCommand) -> None:
        try:
            self.bridge.send_motor_command(msg)
        except Exception as exc:
            self.get_logger().error(f"Failed to send motor command: {exc}")

    def _tick(self) -> None:
        alive = self.bridge.is_alive()
        hb = Bool()
        hb.data = bool(alive)
        self.heartbeat_pub.publish(hb)

        state = self.bridge.read_motor_state()
        if state is not None:
            state.header.stamp = self.get_clock().now().to_msg()
            state.header.frame_id = "redrhex_base"
            self.feedback_pub.publish(state)

        status = DiagnosticStatus()
        status.name = "redrhex_lowlevel_bridge"
        status.hardware_id = self.backend
        status.level = DiagnosticStatus.OK if alive else DiagnosticStatus.ERROR
        status.message = "alive" if alive else "not alive"
        status.values = [KeyValue(key="backend", value=self.backend)]
        if hasattr(self.bridge, "diagnostic_values"):
            status.values.extend(
                KeyValue(key=key, value=value) for key, value in self.bridge.diagnostic_values().items()
            )
        arr = DiagnosticArray()
        arr.header.stamp = self.get_clock().now().to_msg()
        arr.status = [status]
        self.diag_pub.publish(arr)

    def destroy_node(self) -> bool:
        self.bridge.shutdown()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LowLevelBridgeNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException, RCLError):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
