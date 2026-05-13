"""ROS2 node wiring RedRhex sensors -> ONNX policy -> motor commands."""

from __future__ import annotations

import traceback

import numpy as np
import rclpy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import BatteryState, Imu, JointState
from std_msgs.msg import Bool, Float32MultiArray, String

from redrhex_msgs.msg import RedRhexMotorCommand, RedRhexMotorState

from . import redrhex_contract as C
from .action_decoder import ActionDecoder, DecodedMotorCommand
from .observation_builder import ObservationBuilder
from .policy_onnx_runner import PolicyONNXRunner
from .safety_filter import SafetyFilter, SafetyState
from .state_machine import RedRhexState, RedRhexStateMachine, StateMachineInputs


def _declare_get(node: Node, name: str, default):
    node.declare_parameter(name, default)
    return node.get_parameter(name).value


class RedRhexRLControllerNode(Node):
    def __init__(self) -> None:
        super().__init__("redrhex_rl_controller")

        self.onnx_path = str(_declare_get(self, "policy.onnx_path", "/home/jetson/redrhex_models/policy.onnx"))
        self.expected_obs_dim = int(_declare_get(self, "policy.expected_obs_dim", C.OBS_DIM_SINGLE))
        self.expected_action_dim = int(_declare_get(self, "policy.expected_action_dim", C.ACTION_DIM))
        self.policy_hz_param = float(_declare_get(self, "policy.policy_hz", 0.0))
        self.policy_hz = self.policy_hz_param if self.policy_hz_param > 0.0 else C.POLICY_HZ
        self.use_cuda = bool(_declare_get(self, "policy.use_cuda", False))
        self.use_tensorrt = bool(_declare_get(self, "policy.use_tensorrt", False))
        self.allow_history_dim = bool(_declare_get(self, "policy.allow_history_dim", True))

        self.enable_policy_param = bool(_declare_get(self, "state_machine.enable_policy_on_start", False))
        self.enable_motor_output_param = bool(_declare_get(self, "state_machine.enable_motor_output_on_start", False))
        self.init_stand_duration_s = float(_declare_get(self, "state_machine.init_stand_duration_s", 2.0))
        self.warmup_duration_s = float(_declare_get(self, "state_machine.warmup_duration_s", 1.0))

        self.sensor_timeout_s = float(_declare_get(self, "safety.sensor_timeout_s", 0.10))
        self.cmd_timeout_s = float(_declare_get(self, "safety.cmd_timeout_s", 0.25))
        self.require_motor_feedback = bool(_declare_get(self, "safety.require_motor_feedback", False))
        self.require_lowlevel_heartbeat = bool(_declare_get(self, "safety.require_lowlevel_heartbeat", False))

        command_limits = {
            "vx_min": float(_declare_get(self, "commands.vx_min", C.COMMAND_LIMITS["vx_min"])),
            "vx_max": float(_declare_get(self, "commands.vx_max", C.COMMAND_LIMITS["vx_max"])),
            "vy_min": float(_declare_get(self, "commands.vy_min", C.COMMAND_LIMITS["vy_min"])),
            "vy_max": float(_declare_get(self, "commands.vy_max", C.COMMAND_LIMITS["vy_max"])),
            "wz_min": float(_declare_get(self, "commands.wz_min", C.COMMAND_LIMITS["wz_min"])),
            "wz_max": float(_declare_get(self, "commands.wz_max", C.COMMAND_LIMITS["wz_max"])),
        }

        builder_cfg = {
            "expected_obs_dim": C.OBS_DIM_SINGLE,
            "policy_input_dim": self.expected_obs_dim,
            "policy_history_length": int(_declare_get(self, "observation.policy_history_length", C.POLICY_HISTORY_LENGTH)),
            "base_lin_vel_source": str(_declare_get(self, "observation.base_lin_vel_source", "zero")),
            "odom_twist_in_body_frame": bool(_declare_get(self, "observation.odom_twist_in_body_frame", True)),
            "command_limits": command_limits,
        }
        decoder_cfg = {
            "action_clip": float(_declare_get(self, "safety.action_clip", 1.0)),
            "main_drive_vel_limit_rad_s": float(_declare_get(self, "safety.main_drive_vel_limit_rad_s", 30.0)),
            "abad_pos_limit": float(_declare_get(self, "safety.abad_pos_limit_rad", C.STAGE_ABAD_POS_LIMIT)),
            "main_drive_slew_rate_rad_s2": float(_declare_get(self, "safety.main_drive_slew_rate_rad_s2", 120.0)),
            "abad_slew_rate_rad_s": float(_declare_get(self, "safety.abad_slew_rate_rad_s", 6.0)),
            "include_damper_command": bool(_declare_get(self, "action.include_damper_command", True)),
            "stand_main_drive_kp": list(_declare_get(self, "action.stand_main_drive_kp", [12.0] * 6)),
            "stand_main_drive_kd": list(_declare_get(self, "action.stand_main_drive_kd", [1.0] * 6)),
            "main_drive_sign": list(_declare_get(self, "action.main_drive_sign", [1.0] * 6)),
            "abad_sign": list(_declare_get(self, "action.abad_sign", [1.0] * 6)),
            "damper_sign": list(_declare_get(self, "action.damper_sign", [1.0] * 6)),
            "main_drive_zero_offset_rad": list(_declare_get(self, "action.main_drive_zero_offset_rad", [0.0] * 6)),
            "abad_zero_offset_rad": list(_declare_get(self, "action.abad_zero_offset_rad", [0.0] * 6)),
            "damper_zero_offset_rad": list(_declare_get(self, "action.damper_zero_offset_rad", [0.0] * 6)),
            "main_drive_kp": list(_declare_get(self, "action.main_drive_kp", [0.0] * 6)),
            "main_drive_kd": list(_declare_get(self, "action.main_drive_kd", [50.0] * 6)),
            "abad_kp": list(_declare_get(self, "action.abad_kp", [40.0] * 6)),
            "abad_kd": list(_declare_get(self, "action.abad_kd", [4.0] * 6)),
        }
        safety_cfg = {
            "sensor_timeout_s": self.sensor_timeout_s,
            "cmd_timeout_s": self.cmd_timeout_s,
            "motor_feedback_timeout_s": float(_declare_get(self, "safety.motor_feedback_timeout_s", 0.25)),
            "heartbeat_timeout_s": float(_declare_get(self, "safety.heartbeat_timeout_s", 0.10)),
            "max_abs_roll_rad": float(_declare_get(self, "safety.max_abs_roll_rad", 0.7)),
            "max_abs_pitch_rad": float(_declare_get(self, "safety.max_abs_pitch_rad", 0.7)),
            "action_clip": float(_declare_get(self, "safety.action_clip", 1.0)),
            "main_drive_vel_limit_rad_s": float(_declare_get(self, "safety.main_drive_vel_limit_rad_s", 30.0)),
            "abad_pos_limit_rad": float(_declare_get(self, "safety.abad_pos_limit_rad", 0.7)),
            "max_motor_temperature_c": float(_declare_get(self, "safety.max_motor_temperature_c", 70.0)),
            "max_motor_current_a": float(_declare_get(self, "safety.max_motor_current_a", 20.0)),
            "max_control_loop_dt_s": float(_declare_get(self, "safety.max_control_loop_dt_s", 0.03)),
            "require_motor_feedback": self.require_motor_feedback,
            "require_lowlevel_heartbeat": self.require_lowlevel_heartbeat,
            "command_limits": command_limits,
        }

        self.observation_builder = ObservationBuilder(builder_cfg)
        self.action_decoder = ActionDecoder(decoder_cfg)
        self.safety_filter = SafetyFilter(safety_cfg)
        self.state_machine = RedRhexStateMachine(
            require_motor_feedback=self.require_motor_feedback,
            require_lowlevel_heartbeat=self.require_lowlevel_heartbeat,
        )

        self.policy_runner: PolicyONNXRunner | None = None
        self.policy_loaded = False
        try:
            self.policy_runner = PolicyONNXRunner(
                self.onnx_path,
                expected_obs_dim=self.expected_obs_dim,
                expected_action_dim=self.expected_action_dim,
                use_cuda=self.use_cuda,
                use_tensorrt=self.use_tensorrt,
                allow_history_dim=self.allow_history_dim,
            )
            if self.policy_runner.obs_dim is not None:
                self.observation_builder.policy_input_dim = self.policy_runner.obs_dim
            self.policy_loaded = True
            self.get_logger().info(f"Loaded ONNX policy: {self.policy_runner.io_info}")
        except Exception as exc:
            self.get_logger().error(f"Failed to load ONNX policy: {exc}")

        self.estop = False
        self.enable_policy = self.enable_policy_param
        self.enable_motor_output = self.enable_motor_output_param
        self.recover_requested = False
        self.last_motor_feedback_time: float | None = None
        self.last_lowlevel_heartbeat_time: float | None = None
        self.motor_temperatures: list[float] = []
        self.motor_currents: list[float] = []
        self.motor_faults: list[bool] = []
        self.battery_state: BatteryState | None = None
        self.last_loop_time: float | None = None
        self.last_diag_values: dict[str, str] = {}
        self.state_enter_time = self._now_s()

        self.create_subscription(Imu, "/imu/data", self._on_imu, 10)
        self.create_subscription(JointState, "/joint_states", self._on_joint_states, 10)
        self.create_subscription(Twist, "/cmd_vel", self._on_cmd_vel, 10)
        self.create_subscription(Bool, "/estop", self._on_estop, 10)
        self.create_subscription(Bool, "/redrhex/enable_policy", self._on_enable_policy, 10)
        self.create_subscription(Bool, "/redrhex/enable_motors", self._on_enable_motors, 10)
        self.create_subscription(Bool, "/redrhex/recover", self._on_recover, 10)
        self.create_subscription(Bool, "/redrhex/lowlevel_heartbeat", self._on_lowlevel_heartbeat, 10)
        self.create_subscription(RedRhexMotorState, "/motor_feedback", self._on_motor_feedback, 10)
        self.create_subscription(BatteryState, "/battery_state", self._on_battery_state, 10)
        self.create_subscription(Odometry, "/odom", self._on_odom, 10)

        self.obs_pub = self.create_publisher(Float32MultiArray, "/redrhex/observation", 10)
        self.raw_action_pub = self.create_publisher(Float32MultiArray, "/redrhex/policy_action_raw", 10)
        self.safe_action_pub = self.create_publisher(Float32MultiArray, "/redrhex/policy_action_safe", 10)
        self.motor_cmd_pub = self.create_publisher(RedRhexMotorCommand, "/redrhex/motor_commands", 10)
        self.state_pub = self.create_publisher(String, "/redrhex/state_machine_state", 10)
        self.diag_pub = self.create_publisher(DiagnosticArray, "/redrhex/diagnostics", 10)

        self.timer = self.create_timer(1.0 / self.policy_hz, self._control_tick)
        self.get_logger().info(f"RedRhex RL controller started at {self.policy_hz:.1f} Hz.")

    def _now_s(self) -> float:
        return self.get_clock().now().nanoseconds * 1.0e-9

    def _on_imu(self, msg: Imu) -> None:
        self.observation_builder.update_imu(msg, self._now_s())

    def _on_joint_states(self, msg: JointState) -> None:
        self.observation_builder.update_joint_state(msg, self._now_s())

    def _on_cmd_vel(self, msg: Twist) -> None:
        self.observation_builder.update_cmd_vel(msg, self._now_s())

    def _on_odom(self, msg: Odometry) -> None:
        self.observation_builder.update_odom(msg, self._now_s())

    def _drop_enable_latches(self, reason: str) -> None:
        if self.enable_policy or self.enable_motor_output:
            self.get_logger().warn(f"Dropping enable latches: {reason}")
        self.enable_policy = False
        self.enable_motor_output = False

    def _policy_enable_allowed(self) -> bool:
        return (not self.estop) and self.state_machine.state in (
            RedRhexState.POLICY_READY,
            RedRhexState.POLICY_RUN,
        )

    def _motor_output_enable_allowed(self) -> bool:
        return (not self.estop) and self.state_machine.state in (
            RedRhexState.INIT_STAND,
            RedRhexState.WARMUP,
            RedRhexState.POLICY_READY,
            RedRhexState.POLICY_RUN,
        )

    def _on_estop(self, msg: Bool) -> None:
        self.estop = bool(msg.data)
        if self.estop:
            self._drop_enable_latches("E-stop asserted")

    def _on_enable_policy(self, msg: Bool) -> None:
        requested = bool(msg.data)
        if not requested:
            self.enable_policy = False
            return
        if not self._policy_enable_allowed():
            self.enable_policy = False
            self.get_logger().warn(
                f"Rejecting policy enable while state={self.state_machine.state.value}, estop={self.estop}"
            )
            return
        self.enable_policy = True

    def _on_enable_motors(self, msg: Bool) -> None:
        requested = bool(msg.data)
        if not requested:
            self.enable_motor_output = False
            return
        if not self._motor_output_enable_allowed():
            self.enable_motor_output = False
            self.get_logger().warn(
                f"Rejecting motor output enable while state={self.state_machine.state.value}, estop={self.estop}"
            )
            return
        self.enable_motor_output = True

    def _on_recover(self, msg: Bool) -> None:
        self.recover_requested = bool(msg.data)

    def _on_lowlevel_heartbeat(self, msg: Bool) -> None:
        if msg.data:
            self.last_lowlevel_heartbeat_time = self._now_s()

    def _on_motor_feedback(self, msg: RedRhexMotorState) -> None:
        self.last_motor_feedback_time = self._now_s()
        self.motor_temperatures = [float(x) for x in msg.temperature_c]
        self.motor_currents = [float(x) for x in msg.current_a]
        self.motor_faults = [bool(x) for x in msg.fault]

    def _on_battery_state(self, msg: BatteryState) -> None:
        self.battery_state = msg

    def _control_tick(self) -> None:
        now_s = self._now_s()
        dt = C.CONTROL_DT if self.last_loop_time is None else max(0.0, now_s - self.last_loop_time)
        self.last_loop_time = now_s

        obs_status = self.observation_builder.status(now_s, self.sensor_timeout_s, self.cmd_timeout_s)
        roll, pitch, _ = self.observation_builder.get_roll_pitch_yaw()
        imu_age = None if self.observation_builder.imu_time is None else now_s - self.observation_builder.imu_time
        joint_age = None if self.observation_builder.joint_time is None else now_s - self.observation_builder.joint_time
        motor_age = None if self.last_motor_feedback_time is None else now_s - self.last_motor_feedback_time
        heartbeat_age = None if self.last_lowlevel_heartbeat_time is None else now_s - self.last_lowlevel_heartbeat_time
        self.last_diag_values = {
            "imu_age_s": self._fmt_optional(imu_age),
            "joint_state_age_s": self._fmt_optional(joint_age),
            "motor_feedback_age_s": self._fmt_optional(motor_age),
            "heartbeat_age_s": self._fmt_optional(heartbeat_age),
            "control_loop_dt_s": f"{dt:.4f}",
            "roll_rad": f"{roll:.4f}",
            "pitch_rad": f"{pitch:.4f}",
            "cmd_vel": ",".join(f"{x:.3f}" for x in self.observation_builder.cmd_vel),
        }

        pre_safety = SafetyState(
            estop=self.estop,
            imu_age_s=imu_age,
            joint_state_age_s=joint_age,
            motor_feedback_age_s=motor_age,
            heartbeat_age_s=heartbeat_age,
            roll_rad=roll,
            pitch_rad=pitch,
            command=self.observation_builder.cmd_vel.copy(),
            motor_temperatures_c=self.motor_temperatures,
            motor_currents_a=self.motor_currents,
            motor_faults=self.motor_faults,
            control_loop_dt_s=dt,
        )
        safety_result = self.safety_filter.check(pre_safety)
        ignore_waiting_timeouts = self.state_machine.state in (RedRhexState.BOOT, RedRhexState.SENSOR_CHECK)
        if not safety_result.ok and not ignore_waiting_timeouts:
            self._drop_enable_latches("; ".join(safety_result.reasons[:3]))
        sm_safety_ok = safety_result.ok or ignore_waiting_timeouts
        if self.estop:
            sm_safety_ok = False

        elapsed_in_state = now_s - self.state_enter_time
        motor_ready = motor_age is not None and motor_age <= self.safety_filter.motor_feedback_timeout_s
        bridge_ready = heartbeat_age is not None and heartbeat_age <= self.safety_filter.heartbeat_timeout_s
        inputs = StateMachineInputs(
            policy_loaded=self.policy_loaded,
            sensors_ready=obs_status.ok,
            motor_feedback_ready=motor_ready,
            lowlevel_alive=bridge_ready,
            estop=self.estop,
            safety_ok=sm_safety_ok,
            fall_detected=abs(roll) > self.safety_filter.max_abs_roll_rad or abs(pitch) > self.safety_filter.max_abs_pitch_rad,
            init_stand_done=elapsed_in_state >= self.init_stand_duration_s,
            warmup_done=elapsed_in_state >= self.warmup_duration_s,
            enable_policy=self.enable_policy,
            recover_requested=self.recover_requested,
            reasons=safety_result.reasons + obs_status.reasons,
        )
        old_state = self.state_machine.state
        state = self.state_machine.update(inputs)
        if state != old_state:
            self.state_enter_time = now_s
            self.get_logger().info(f"State transition: {old_state.value} -> {state.value}: {self.state_machine.last_transition_reason}")
            if state == RedRhexState.INIT_STAND:
                self.observation_builder.reset(gait_phase=0.0)
                self.action_decoder.reset(gait_phase=0.0)
            if state in (RedRhexState.PROTECTIVE_STOP, RedRhexState.FALL_DETECTED, RedRhexState.RECOVER):
                self._drop_enable_latches(self.state_machine.last_transition_reason)

        decoded: DecodedMotorCommand | None = None
        observation_for_pub: np.ndarray | None = None
        raw_action: np.ndarray | None = None
        reasons = list(safety_result.reasons) + list(obs_status.reasons)

        try:
            if state in (RedRhexState.BOOT, RedRhexState.SENSOR_CHECK, RedRhexState.MOTOR_IDLE):
                decoded = self.action_decoder.disabled_command()
            elif state in (RedRhexState.INIT_STAND, RedRhexState.WARMUP, RedRhexState.POLICY_READY):
                decoded = self.action_decoder.init_stand_command(enable=True)
            elif state == RedRhexState.POLICY_RUN:
                if self.policy_runner is None:
                    raise RuntimeError("policy runner is not loaded")
                observation_for_pub = self.observation_builder.build_policy_input(now_s)
                single_obs = observation_for_pub[: C.OBS_DIM_SINGLE]
                raw_action = self.policy_runner.run(observation_for_pub)
                projected_gravity = single_obs[C.OBSERVATION_SLICES["projected_gravity"][0] : C.OBSERVATION_SLICES["projected_gravity"][1]]
                decoded = self.action_decoder.decode(
                    raw_action,
                    self.observation_builder.get_main_drive_positions(),
                    self.observation_builder.get_abad_positions(),
                    self.observation_builder.cmd_vel.copy(),
                    projected_gravity,
                    dt,
                    self.observation_builder.gait_phase,
                )
                post_safety = self.safety_filter.check(pre_safety, single_obs, raw_action, decoded)
                if not post_safety.ok:
                    reasons.extend(post_safety.reasons)
                    self._drop_enable_latches("; ".join(post_safety.reasons[:3]))
                    self.state_machine.transition(RedRhexState.PROTECTIVE_STOP, "; ".join(post_safety.reasons))
                    decoded = self.action_decoder.protective_stop_command(
                        self.observation_builder.get_main_drive_positions(),
                        self.observation_builder.get_abad_positions(),
                    )
                else:
                    self.observation_builder.update_last_actions(raw_action)
                    self.raw_action_pub.publish(Float32MultiArray(data=[float(x) for x in raw_action]))
                    self.safe_action_pub.publish(Float32MultiArray(data=[float(x) for x in decoded.safe_action]))
                    self.obs_pub.publish(Float32MultiArray(data=[float(x) for x in single_obs]))
            else:
                decoded = self.action_decoder.protective_stop_command(
                    self.observation_builder.get_main_drive_positions(),
                    self.observation_builder.get_abad_positions(),
                )
        except Exception as exc:
            reasons.append(str(exc))
            self.get_logger().error(f"Control tick failed: {exc}\n{traceback.format_exc()}")
            self._drop_enable_latches(str(exc))
            self.state_machine.transition(RedRhexState.PROTECTIVE_STOP, str(exc))
            decoded = self.action_decoder.protective_stop_command(
                self.observation_builder.get_main_drive_positions(),
                self.observation_builder.get_abad_positions(),
            )

        self._publish_motor_command(decoded)
        self._publish_state_and_diagnostics(reasons, safety_result.ok and obs_status.ok)
        self.recover_requested = False

    def _publish_motor_command(self, decoded: DecodedMotorCommand) -> None:
        msg = RedRhexMotorCommand()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "redrhex_base"
        msg.joint_names = decoded.joint_names
        msg.target_position_rad = decoded.target_position_rad
        msg.target_velocity_rad_s = decoded.target_velocity_rad_s
        msg.kp = decoded.kp
        msg.kd = decoded.kd
        msg.effort_limit_nm = decoded.effort_limit_nm
        msg.enable = bool(decoded.enable and self.enable_motor_output and not self.estop)
        msg.mode = decoded.mode
        self.motor_cmd_pub.publish(msg)

    @staticmethod
    def _fmt_optional(value: float | None) -> str:
        return "none" if value is None else f"{value:.4f}"

    def _publish_state_and_diagnostics(self, reasons: list[str], ok: bool) -> None:
        state_msg = String()
        state_msg.data = self.state_machine.state.value
        self.state_pub.publish(state_msg)

        status = DiagnosticStatus()
        status.name = "redrhex_rl_controller"
        status.hardware_id = "redrhex"
        status.level = DiagnosticStatus.OK if ok else DiagnosticStatus.WARN
        if self.state_machine.state in (RedRhexState.PROTECTIVE_STOP, RedRhexState.FALL_DETECTED):
            status.level = DiagnosticStatus.ERROR
        status.message = "OK" if not reasons else "; ".join(reasons[:6])
        status.values = [
            KeyValue(key="state", value=self.state_machine.state.value),
            KeyValue(key="last_transition_reason", value=self.state_machine.last_transition_reason),
            KeyValue(key="policy_loaded", value=str(self.policy_loaded)),
            KeyValue(key="policy_enabled", value=str(self.enable_policy)),
            KeyValue(key="motor_output_enabled", value=str(self.enable_motor_output and not self.estop)),
            KeyValue(key="estop", value=str(self.estop)),
            KeyValue(key="onnx_path", value=self.onnx_path),
        ]
        status.values.extend(KeyValue(key=key, value=value) for key, value in self.last_diag_values.items())
        arr = DiagnosticArray()
        arr.header.stamp = self.get_clock().now().to_msg()
        arr.status = [status]
        self.diag_pub.publish(arr)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RedRhexRLControllerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
