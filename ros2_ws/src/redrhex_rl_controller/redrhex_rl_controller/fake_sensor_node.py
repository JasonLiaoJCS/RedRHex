"""Fake RedRhex sensors for ROS2 graph and ONNX smoke tests."""

from __future__ import annotations

import math

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import Bool

from . import redrhex_contract as C


class RedRhexFakeSensorNode(Node):
    def __init__(self) -> None:
        super().__init__("redrhex_fake_sensor_node")
        self.declare_parameter("rate_hz", 125.0)
        self.declare_parameter("publish_cmd_vel", True)
        self.declare_parameter("cmd_vx", 0.0)
        self.declare_parameter("cmd_vy", 0.0)
        self.declare_parameter("cmd_wz", 0.0)
        self.rate_hz = float(self.get_parameter("rate_hz").value)
        self.publish_cmd_vel = bool(self.get_parameter("publish_cmd_vel").value)
        self.phase = 0.0

        self.imu_pub = self.create_publisher(Imu, "/imu/data", 10)
        self.joint_pub = self.create_publisher(JointState, "/joint_states", 10)
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.heartbeat_pub = self.create_publisher(Bool, "/redrhex/lowlevel_heartbeat", 10)
        self.timer = self.create_timer(1.0 / self.rate_hz, self._tick)

    def _tick(self) -> None:
        now = self.get_clock().now().to_msg()
        self.phase = (self.phase + 2.0 * math.pi * C.BASE_GAIT_FREQUENCY_HZ / self.rate_hz) % (2.0 * math.pi)

        imu = Imu()
        imu.header.stamp = now
        imu.header.frame_id = "base_link"
        imu.orientation.x = 0.0
        imu.orientation.y = 0.0
        imu.orientation.z = 0.0
        imu.orientation.w = 1.0
        imu.angular_velocity.x = 0.0
        imu.angular_velocity.y = 0.0
        imu.angular_velocity.z = 0.0
        imu.linear_acceleration.z = 9.81
        self.imu_pub.publish(imu)

        js = JointState()
        js.header.stamp = now
        js.name = C.MAIN_DRIVE_JOINT_NAMES + C.ABAD_JOINT_NAMES + C.DAMPER_JOINT_NAMES
        main_pos = [p + 0.05 * math.sin(self.phase) for p in C.INIT_MAIN_DRIVE_POS]
        abad_pos = list(C.INIT_ABAD_POS)
        damper_pos = list(C.INIT_DAMPER_POS)
        js.position = main_pos + abad_pos + damper_pos
        js.velocity = [0.05 * math.cos(self.phase)] * 6 + [0.0] * 12
        self.joint_pub.publish(js)

        if self.publish_cmd_vel:
            cmd = Twist()
            cmd.linear.x = float(self.get_parameter("cmd_vx").value)
            cmd.linear.y = float(self.get_parameter("cmd_vy").value)
            cmd.angular.z = float(self.get_parameter("cmd_wz").value)
            self.cmd_pub.publish(cmd)

        hb = Bool()
        hb.data = True
        self.heartbeat_pub.publish(hb)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RedRhexFakeSensorNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
