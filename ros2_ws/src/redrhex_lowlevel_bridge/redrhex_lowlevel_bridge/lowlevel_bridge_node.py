"""ROS2 node bridging RedRhexMotorCommand to a replaceable low-level backend."""

from __future__ import annotations

import rclpy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from rclpy.node import Node
from std_msgs.msg import Bool

from redrhex_msgs.msg import RedRhexMotorCommand, RedRhexMotorState

from .mock_bridge import MockLowLevelBridge
from .serial_bridge import SerialLowLevelBridge


class LowLevelBridgeNode(Node):
    def __init__(self) -> None:
        super().__init__("redrhex_lowlevel_bridge")
        self.declare_parameter("backend", "mock")
        self.declare_parameter("mock.print_every_n", 50)
        self.declare_parameter("serial.port", "/dev/ttyUSB0")
        self.declare_parameter("serial.baudrate", 921600)
        self.declare_parameter("serial.timeout_s", 0.005)
        self.declare_parameter("feedback_rate_hz", 50.0)

        backend = str(self.get_parameter("backend").value)
        if backend == "mock":
            self.bridge = MockLowLevelBridge(int(self.get_parameter("mock.print_every_n").value))
        elif backend == "serial":
            self.bridge = SerialLowLevelBridge(
                str(self.get_parameter("serial.port").value),
                int(self.get_parameter("serial.baudrate").value),
                float(self.get_parameter("serial.timeout_s").value),
            )
        else:
            raise ValueError(f"Unknown low-level backend '{backend}'. Expected mock or serial.")
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
    finally:
        node.destroy_node()
        rclpy.shutdown()
