"""Adapter backend for JasonLiaoJCS/BioRoLaROS2 rinbo_msgs.

This backend is for the existing BioRoLaROS2/RhexROS2 stack:
  /motor/command  rinbo_msgs/msg/MotorCmdStamped
  /motor/state    rinbo_msgs/msg/MotorStateStamped

It also publishes /joint_states from the six main-drive encoders so the RL
controller can build IsaacLab-compatible observations.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

from sensor_msgs.msg import JointState

from redrhex_msgs.msg import RedRhexMotorState

from .bridge_base import LowLevelBridgeBase


@dataclass(frozen=True)
class RinboLegMapping:
    rinbo_field: str
    policy_index: int
    is_left: bool


class RinboRosBackend(LowLevelBridgeBase):
    """ROS adapter for the existing BioRoLaROS2/RhexROS2 bridge.

    The rinbo bridge expects PWM-like main-drive commands in LegCmd.voltage /
    LegCmd.direction. ABAD commands are sent as ServoCmd.position_encoder.
    """

    POLICY_TO_RINBO_LEGS = [
        RinboLegMapping("r1", 0, False),
        RinboLegMapping("r2", 1, False),
        RinboLegMapping("r3", 2, False),
        RinboLegMapping("l1", 3, True),
        RinboLegMapping("l2", 4, True),
        RinboLegMapping("l3", 5, True),
    ]
    RINBO_LEG_ORDER = ["l1", "l2", "l3", "r1", "r2", "r3"]
    RINBO_SERVO_ORDER = ["sl1", "sl2", "sl3", "sr1", "sr2", "sr3"]
    # Policy ABAD order is RF, RM, RR, LF, LM, LR.
    POLICY_ABAD_INDEX_BY_RINBO_SERVO = [3, 4, 5, 0, 1, 2]

    def __init__(
        self,
        node,
        command_topic: str,
        state_topic: str,
        joint_state_topic: str,
        preview_topic: str,
        publish_preview: bool,
        allow_enable: bool,
        publish_when_disabled: bool,
        disabled_servo_control_mode: int,
        publish_shutdown_disable: bool,
        shutdown_disable_repeats: int,
        shutdown_disable_period_s: float,
        require_state: bool,
        block_if_duplicate_command_publishers: bool,
        state_timeout_s: float,
        main_position_counts_per_rev: float,
        main_pwm_per_rad_s: float,
        main_max_pwm: float,
        main_encoder_zero_counts_rinbo_order: list[float],
        main_encoder_sign_rinbo_order: list[float],
        main_velocity_sign_policy_order: list[float],
        main_direction_positive_rinbo_order: list[bool],
        main_velocity_filter_alpha: float,
        main_velocity_max_dt_s: float,
        main_velocity_clip_rad_s: float,
        abad_encoder_zero_rinbo_order: list[int],
        abad_encoder_counts_per_rad: float,
        abad_encoder_min: int,
        abad_encoder_max: int,
        abad_sign_rinbo_order: list[float],
        servo_control_mode: int,
        main_joint_names_policy_order: list[str],
    ) -> None:
        self.node = node
        self.command_topic = command_topic
        self.state_topic = state_topic
        self.joint_state_topic = joint_state_topic
        self.preview_topic = preview_topic
        self.publish_preview = bool(publish_preview)
        self.allow_enable = bool(allow_enable)
        self.publish_when_disabled = bool(publish_when_disabled)
        self.disabled_servo_control_mode = int(disabled_servo_control_mode)
        self.publish_shutdown_disable = bool(publish_shutdown_disable)
        self.shutdown_disable_repeats = int(shutdown_disable_repeats)
        self.shutdown_disable_period_s = float(shutdown_disable_period_s)
        self.require_state = bool(require_state)
        self.block_if_duplicate_command_publishers = bool(block_if_duplicate_command_publishers)
        self.state_timeout_s = float(state_timeout_s)
        self.main_position_counts_per_rev = float(main_position_counts_per_rev)
        self.main_rad_per_count = 2.0 * math.pi / self.main_position_counts_per_rev
        self.main_pwm_per_rad_s = float(main_pwm_per_rad_s)
        self.main_max_pwm = float(main_max_pwm)
        self.main_encoder_zero_counts_rinbo_order = [float(x) for x in main_encoder_zero_counts_rinbo_order]
        self.main_encoder_sign_rinbo_order = [float(x) for x in main_encoder_sign_rinbo_order]
        self.main_velocity_sign_policy_order = [float(x) for x in main_velocity_sign_policy_order]
        self.main_direction_positive_rinbo_order = [bool(x) for x in main_direction_positive_rinbo_order]
        self.main_velocity_filter_alpha = float(main_velocity_filter_alpha)
        self.main_velocity_max_dt_s = float(main_velocity_max_dt_s)
        self.main_velocity_clip_rad_s = float(main_velocity_clip_rad_s)
        self.abad_encoder_zero_rinbo_order = [int(x) for x in abad_encoder_zero_rinbo_order]
        self.abad_encoder_counts_per_rad = float(abad_encoder_counts_per_rad)
        self.abad_encoder_min = int(abad_encoder_min)
        self.abad_encoder_max = int(abad_encoder_max)
        self.abad_sign_rinbo_order = [float(x) for x in abad_sign_rinbo_order]
        self.servo_control_mode = int(servo_control_mode)
        self.main_joint_names_policy_order = list(main_joint_names_policy_order)

        if len(self.main_encoder_zero_counts_rinbo_order) != 6:
            raise ValueError("main_encoder_zero_counts_rinbo_order must have length 6")
        if len(self.main_encoder_sign_rinbo_order) != 6:
            raise ValueError("main_encoder_sign_rinbo_order must have length 6")
        if len(self.main_velocity_sign_policy_order) != 6:
            raise ValueError("main_velocity_sign_policy_order must have length 6")
        if len(self.main_direction_positive_rinbo_order) != 6:
            raise ValueError("main_direction_positive_rinbo_order must have length 6")
        if len(self.abad_encoder_zero_rinbo_order) != 6:
            raise ValueError("abad_encoder_zero_rinbo_order must have length 6")
        if len(self.abad_sign_rinbo_order) != 6:
            raise ValueError("abad_sign_rinbo_order must have length 6")
        if len(self.main_joint_names_policy_order) != 6:
            raise ValueError("main_joint_names_policy_order must have length 6")
        if self.main_position_counts_per_rev <= 0.0:
            raise ValueError("main_position_counts_per_rev must be positive")
        if self.main_pwm_per_rad_s <= 0.0 or self.main_max_pwm <= 0.0:
            raise ValueError("main PWM conversion parameters must be positive")
        if self.abad_encoder_counts_per_rad <= 0.0:
            raise ValueError("abad_encoder_counts_per_rad must be positive")
        if self.abad_encoder_min >= self.abad_encoder_max:
            raise ValueError("abad_encoder_min must be smaller than abad_encoder_max")
        if not 0.0 <= self.main_velocity_filter_alpha <= 1.0:
            raise ValueError("main_velocity_filter_alpha must be in [0, 1]")
        if self.main_velocity_max_dt_s <= 0.0:
            raise ValueError("main_velocity_max_dt_s must be positive")
        if self.main_velocity_clip_rad_s <= 0.0:
            raise ValueError("main_velocity_clip_rad_s must be positive")
        if self.shutdown_disable_repeats < 0:
            raise ValueError("shutdown_disable_repeats must be non-negative")
        if self.shutdown_disable_period_s < 0.0:
            raise ValueError("shutdown_disable_period_s must be non-negative")

        self.connected = False
        self.sequence = 0
        self.last_state_time: float | None = None
        self.latest_motor_state: RedRhexMotorState | None = None
        self.latest_positions_policy = [0.0] * 6
        self.latest_velocities_policy = [0.0] * 6
        self._prev_positions_policy: list[float] | None = None
        self._prev_state_time: float | None = None
        self.latest_raw_positions_rinbo = [0.0] * 6
        self.latest_servo_positions_rinbo = list(self.abad_encoder_zero_rinbo_order)
        self.last_command_was_enabled = False
        self.last_pwm_rinbo_order = [0.0] * 6
        self.last_abad_encoder_targets_rinbo_order = list(self.abad_encoder_zero_rinbo_order)
        self.last_actual_publish_state = "never"
        self.last_block_reason = ""
        self._last_warned_block_reason = ""

    def connect(self) -> None:
        try:
            from rinbo_msgs.msg import MotorCmdStamped, MotorStateStamped
        except Exception as exc:  # pragma: no cover - requires external BioRoLaROS2 overlay
            raise RuntimeError(
                "rinbo_msgs is required for backend='biorola_ros'/'rinbo_ros'. Build/source BioRoLaROS2 first."
            ) from exc

        self.MotorCmdStamped = MotorCmdStamped
        self.MotorStateStamped = MotorStateStamped
        self.cmd_pub = self.node.create_publisher(MotorCmdStamped, self.command_topic, 10)
        self.preview_pub = self.node.create_publisher(MotorCmdStamped, self.preview_topic, 10)
        self.joint_pub = self.node.create_publisher(JointState, self.joint_state_topic, 10)
        self.state_sub = self.node.create_subscription(MotorStateStamped, self.state_topic, self._on_rinbo_state, 10)
        self.connected = True
        self.node.get_logger().info(
            f"Rinbo ROS backend connected: command={self.command_topic}, state={self.state_topic}"
        )

    def send_motor_command(self, cmd) -> None:
        if not self.connected:
            raise RuntimeError("Rinbo ROS backend is not connected")
        self._validate_command_shape(cmd)
        enabled = bool(cmd.enable)

        preview_msg = self._make_motor_cmd_msg(cmd, enabled=enabled, preview=True)
        if self.publish_preview:
            self.preview_pub.publish(preview_msg)

        if enabled and not self.allow_enable:
            self.last_actual_publish_state = "blocked_allow_enable"
            self.last_block_reason = "rinbo.allow_enable is false"
            self._warn_once(self.last_block_reason)
            self.last_command_was_enabled = False
            return
        if enabled and self.require_state and not self.is_alive():
            self.last_actual_publish_state = "blocked_no_recent_state"
            self.last_block_reason = "no recent /motor/state"
            self._warn_once(self.last_block_reason)
            self.last_command_was_enabled = False
            return
        if enabled and self.block_if_duplicate_command_publishers:
            duplicate_count, endpoint_names = self._command_publisher_count()
            if duplicate_count > 1:
                self.last_actual_publish_state = "blocked_duplicate_publishers"
                self.last_block_reason = f"{duplicate_count} publishers on {self.command_topic}: {endpoint_names}"
                self._warn_once(self.last_block_reason)
                self.last_command_was_enabled = False
                return
        if enabled and self.cmd_pub.get_subscription_count() == 0:
            self.last_actual_publish_state = "blocked_no_command_subscriber"
            self.last_block_reason = f"no subscriber on {self.command_topic}"
            self._warn_once(self.last_block_reason)
            self.last_command_was_enabled = False
            return

        # During dry-run, avoid publishing disabled preview packets because
        # BioRoLaROS2 servo commands have no per-servo enable. If motors were
        # previously enabled, still send one disabled packet to release legs.
        if not enabled and not self.publish_when_disabled and not self.last_command_was_enabled:
            self.last_command_was_enabled = False
            self.last_pwm_rinbo_order = [0.0] * 6
            self.last_actual_publish_state = "preview_only_disabled"
            return

        msg = self._make_motor_cmd_msg(cmd, enabled=enabled, preview=False)
        self.cmd_pub.publish(msg)
        self.last_command_was_enabled = enabled
        self.last_actual_publish_state = "published_enabled" if enabled else "published_disabled"

    def _make_motor_cmd_msg(self, cmd, enabled: bool, preview: bool):
        msg = self.MotorCmdStamped()
        now = self.node.get_clock().now().to_msg()
        if preview:
            seq = self.sequence
        else:
            self.sequence = (self.sequence + 1) & 0xFFFFFFFF
            seq = self.sequence
        msg.header.seq = seq
        msg.header.stamp = now
        msg.header.frame_id = "redrhex_preview" if preview else "redrhex_base"
        msg.servo_control_mode = self.servo_control_mode if enabled else self.disabled_servo_control_mode

        self._disable_all_legs(msg)
        self._set_main_drive_pwm(msg, cmd, enabled)
        if enabled:
            self._set_abad_servo_targets(msg, cmd)
        else:
            self._set_abad_neutral_targets(msg)
        return msg

    def _validate_command_shape(self, cmd) -> None:
        if len(cmd.target_velocity_rad_s) < 6:
            raise ValueError("target_velocity_rad_s must contain at least 6 main-drive values")
        if len(cmd.target_position_rad) < 12:
            raise ValueError("target_position_rad must contain 6 main-drive + 6 ABAD values")

    def read_motor_state(self):
        return self.latest_motor_state

    def is_alive(self) -> bool:
        if not self.connected:
            return False
        if not self.require_state:
            return True
        if self.last_state_time is None:
            return False
        return time.monotonic() - self.last_state_time <= self.state_timeout_s

    def shutdown(self) -> None:
        if self.connected and self.publish_shutdown_disable:
            self._publish_shutdown_disable()
        self.connected = False

    def diagnostic_values(self) -> dict[str, str]:
        state_age = "none" if self.last_state_time is None else f"{time.monotonic() - self.last_state_time:.4f}"
        publisher_count, publisher_names = self._command_publisher_count()
        return {
            "rinbo_command_topic": self.command_topic,
            "rinbo_state_topic": self.state_topic,
            "rinbo_preview_topic": self.preview_topic,
            "rinbo_publish_preview": str(self.publish_preview),
            "rinbo_require_state": str(self.require_state),
            "rinbo_allow_enable": str(self.allow_enable),
            "rinbo_publish_when_disabled": str(self.publish_when_disabled),
            "rinbo_block_if_duplicate_command_publishers": str(self.block_if_duplicate_command_publishers),
            "rinbo_command_subscribers": str(self.cmd_pub.get_subscription_count() if hasattr(self, "cmd_pub") else 0),
            "rinbo_command_publishers": f"{publisher_count}: {publisher_names}",
            "rinbo_main_velocity_filter_alpha": f"{self.main_velocity_filter_alpha:.3f}",
            "rinbo_main_velocity_clip_rad_s": f"{self.main_velocity_clip_rad_s:.3f}",
            "rinbo_last_state_age_s": state_age,
            "rinbo_last_command_enabled": str(self.last_command_was_enabled),
            "rinbo_actual_publish_state": self.last_actual_publish_state,
            "rinbo_last_block_reason": self.last_block_reason,
            "rinbo_last_pwm_l1_l2_l3_r1_r2_r3": ",".join(f"{x:.2f}" for x in self.last_pwm_rinbo_order),
            "rinbo_last_abad_sl1_sl2_sl3_sr1_sr2_sr3": ",".join(str(x) for x in self.last_abad_encoder_targets_rinbo_order),
            "rinbo_servo_state_sl1_sl2_sl3_sr1_sr2_sr3": ",".join(str(x) for x in self.latest_servo_positions_rinbo),
            "rinbo_main_vel_policy_order_rad_s": ",".join(f"{x:.3f}" for x in self.latest_velocities_policy),
        }

    def _warn_once(self, reason: str) -> None:
        if reason != self._last_warned_block_reason:
            self.node.get_logger().warn(f"Blocking enabled BioRoLaROS2 command: {reason}")
            self._last_warned_block_reason = reason

    def _command_publisher_count(self) -> tuple[int, str]:
        try:
            infos = self.node.get_publishers_info_by_topic(self.command_topic)
        except Exception:
            return 0, "unknown"
        names = []
        for info in infos:
            node_name = getattr(info, "node_name", "")
            node_namespace = getattr(info, "node_namespace", "")
            names.append(f"{node_namespace.rstrip('/')}/{node_name}".replace("//", "/") or "<unknown>")
        return len(infos), ",".join(names) if names else "none"

    def _publish_shutdown_disable(self) -> None:
        if not hasattr(self, "cmd_pub") or not hasattr(self, "MotorCmdStamped"):
            return
        msg = self.MotorCmdStamped()
        msg.header.frame_id = "redrhex_shutdown_disable"
        msg.servo_control_mode = self.disabled_servo_control_mode
        self._disable_all_legs(msg)
        self._set_abad_neutral_targets(msg)
        for _ in range(self.shutdown_disable_repeats):
            msg.header.stamp = self.node.get_clock().now().to_msg()
            self.cmd_pub.publish(msg)
            if self.shutdown_disable_period_s > 0.0:
                time.sleep(self.shutdown_disable_period_s)

    def _disable_all_legs(self, msg) -> None:
        for field in self.RINBO_LEG_ORDER:
            leg = getattr(msg, field)
            leg.enable = False
            leg.direction = False
            leg.voltage = 0.0
            leg.state = 0
            leg.reset_position = False

    def _set_main_drive_pwm(self, msg, cmd, enabled: bool) -> None:
        self.last_pwm_rinbo_order = [0.0] * 6
        for mapping in self.POLICY_TO_RINBO_LEGS:
            leg = getattr(msg, mapping.rinbo_field)
            rinbo_idx = self.RINBO_LEG_ORDER.index(mapping.rinbo_field)
            target_velocity = (
                float(cmd.target_velocity_rad_s[mapping.policy_index])
                * self.main_velocity_sign_policy_order[mapping.policy_index]
            )
            pwm = max(-self.main_max_pwm, min(self.main_max_pwm, target_velocity * self.main_pwm_per_rad_s))
            self.last_pwm_rinbo_order[rinbo_idx] = float(pwm)
            if enabled:
                leg.enable = True
                leg.state = 1
                leg.reset_position = False
                direction_positive = self.main_direction_positive_rinbo_order[rinbo_idx]
                leg.direction = direction_positive if pwm >= 0.0 else not direction_positive
                leg.voltage = abs(float(pwm))

    def _set_abad_servo_targets(self, msg, cmd) -> None:
        targets: list[int] = []
        abad_targets_policy = list(cmd.target_position_rad[6:12])
        for servo_idx, field in enumerate(self.RINBO_SERVO_ORDER):
            policy_idx = self.POLICY_ABAD_INDEX_BY_RINBO_SERVO[servo_idx]
            target_rad = float(abad_targets_policy[policy_idx])
            raw = (
                self.abad_encoder_zero_rinbo_order[servo_idx]
                + self.abad_sign_rinbo_order[servo_idx] * target_rad * self.abad_encoder_counts_per_rad
            )
            target = int(max(self.abad_encoder_min, min(self.abad_encoder_max, round(raw))))
            getattr(msg, field).position_encoder = target
            targets.append(target)
        self.last_abad_encoder_targets_rinbo_order = targets

    def _set_abad_neutral_targets(self, msg) -> None:
        targets: list[int] = []
        for servo_idx, field in enumerate(self.RINBO_SERVO_ORDER):
            target = int(max(self.abad_encoder_min, min(self.abad_encoder_max, self.abad_encoder_zero_rinbo_order[servo_idx])))
            getattr(msg, field).position_encoder = target
            targets.append(target)
        self.last_abad_encoder_targets_rinbo_order = targets

    def _on_rinbo_state(self, msg) -> None:
        state_time = time.monotonic()
        self.last_state_time = state_time
        rinbo_positions = [
            float(msg.l1.position),
            float(msg.l2.position),
            float(msg.l3.position),
            float(msg.r1.position),
            float(msg.r2.position),
            float(msg.r3.position),
        ]
        self.latest_raw_positions_rinbo = rinbo_positions
        self.latest_servo_positions_rinbo = [
            int(getattr(msg.sl1, "position_encoder", 0)),
            int(getattr(msg.sl2, "position_encoder", 0)),
            int(getattr(msg.sl3, "position_encoder", 0)),
            int(getattr(msg.sr1, "position_encoder", 0)),
            int(getattr(msg.sr2, "position_encoder", 0)),
            int(getattr(msg.sr3, "position_encoder", 0)),
        ]

        rinbo_rad = [
            (rinbo_positions[i] - self.main_encoder_zero_counts_rinbo_order[i])
            * self.main_encoder_sign_rinbo_order[i]
            * self.main_rad_per_count
            for i in range(6)
        ]
        # Policy order: RF, RM, RR, LF, LM, LR.
        policy_positions = [0.0] * 6
        for mapping in self.POLICY_TO_RINBO_LEGS:
            rinbo_idx = self.RINBO_LEG_ORDER.index(mapping.rinbo_field)
            policy_positions[mapping.policy_index] = rinbo_rad[rinbo_idx]
        policy_velocities = self._estimate_policy_velocities(policy_positions, state_time)
        self.latest_positions_policy = policy_positions
        self.latest_velocities_policy = policy_velocities

        js = JointState()
        js.header.stamp = self.node.get_clock().now().to_msg()
        js.header.frame_id = "redrhex_base"
        js.name = list(self.main_joint_names_policy_order)
        js.position = [float(x) for x in policy_positions]
        js.velocity = [float(x) for x in policy_velocities]
        self.joint_pub.publish(js)

        state = RedRhexMotorState()
        state.header.stamp = js.header.stamp
        state.header.frame_id = "redrhex_base"
        state.joint_names = list(self.main_joint_names_policy_order)
        state.position_rad = [float(x) for x in policy_positions]
        state.velocity_rad_s = [float(x) for x in policy_velocities]
        state.effort_nm = [0.0] * 6
        state.current_a = []
        state.temperature_c = []
        state.fault = [False] * 6
        self.latest_motor_state = state

    def _estimate_policy_velocities(self, policy_positions: list[float], state_time: float) -> list[float]:
        if self._prev_positions_policy is None or self._prev_state_time is None:
            self._prev_positions_policy = list(policy_positions)
            self._prev_state_time = state_time
            return list(self.latest_velocities_policy)

        prev_positions = list(self._prev_positions_policy)
        dt = state_time - self._prev_state_time
        self._prev_positions_policy = list(policy_positions)
        self._prev_state_time = state_time
        if dt <= 1.0e-6 or dt > self.main_velocity_max_dt_s:
            return [0.0] * 6

        raw_vel = [
            max(
                -self.main_velocity_clip_rad_s,
                min(self.main_velocity_clip_rad_s, (float(policy_positions[i]) - float(prev_positions[i])) / dt),
            )
            for i in range(6)
        ]
        alpha = self.main_velocity_filter_alpha
        return [
            alpha * raw_vel[i] + (1.0 - alpha) * self.latest_velocities_policy[i]
            for i in range(6)
        ]
