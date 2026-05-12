"""Safety checks for RedRhex real-robot policy deployment."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import redrhex_contract as C
from .action_decoder import DecodedMotorCommand


@dataclass
class SafetyState:
    estop: bool = False
    imu_age_s: float | None = None
    joint_state_age_s: float | None = None
    motor_feedback_age_s: float | None = None
    heartbeat_age_s: float | None = None
    roll_rad: float = 0.0
    pitch_rad: float = 0.0
    command: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    motor_temperatures_c: list[float] = field(default_factory=list)
    motor_currents_a: list[float] = field(default_factory=list)
    motor_faults: list[bool] = field(default_factory=list)
    control_loop_dt_s: float | None = None


@dataclass
class SafetyResult:
    ok: bool
    reasons: list[str]


class SafetyFilter:
    def __init__(self, config: dict | None = None) -> None:
        self.cfg = config or {}
        self.sensor_timeout_s = float(self.cfg.get("sensor_timeout_s", 0.10))
        self.cmd_timeout_s = float(self.cfg.get("cmd_timeout_s", 0.25))
        self.motor_feedback_timeout_s = float(self.cfg.get("motor_feedback_timeout_s", 0.25))
        self.heartbeat_timeout_s = float(self.cfg.get("heartbeat_timeout_s", 0.10))
        self.max_abs_roll_rad = float(self.cfg.get("max_abs_roll_rad", 0.7))
        self.max_abs_pitch_rad = float(self.cfg.get("max_abs_pitch_rad", 0.7))
        self.action_clip = float(self.cfg.get("action_clip", 1.0))
        self.main_drive_vel_limit_rad_s = float(self.cfg.get("main_drive_vel_limit_rad_s", 30.0))
        self.abad_pos_limit_rad = float(self.cfg.get("abad_pos_limit_rad", 0.7))
        self.max_motor_temperature_c = float(self.cfg.get("max_motor_temperature_c", 70.0))
        self.max_motor_current_a = float(self.cfg.get("max_motor_current_a", 20.0))
        self.max_control_loop_dt_s = float(self.cfg.get("max_control_loop_dt_s", 0.03))
        self.require_motor_feedback = bool(self.cfg.get("require_motor_feedback", False))
        self.require_lowlevel_heartbeat = bool(self.cfg.get("require_lowlevel_heartbeat", False))
        self.command_limits = dict(C.COMMAND_LIMITS)
        self.command_limits.update(self.cfg.get("command_limits", {}))

    def check(
        self,
        state: SafetyState,
        observation: np.ndarray | None = None,
        raw_action: np.ndarray | None = None,
        command: DecodedMotorCommand | None = None,
    ) -> SafetyResult:
        reasons: list[str] = []

        if state.estop:
            reasons.append("E-stop active")
        if state.imu_age_s is None or state.imu_age_s > self.sensor_timeout_s:
            reasons.append("IMU timeout")
        if state.joint_state_age_s is None or state.joint_state_age_s > self.sensor_timeout_s:
            reasons.append("joint_states timeout")
        if self.require_motor_feedback and (
            state.motor_feedback_age_s is None or state.motor_feedback_age_s > self.motor_feedback_timeout_s
        ):
            reasons.append("motor_feedback timeout")
        if self.require_lowlevel_heartbeat and (
            state.heartbeat_age_s is None or state.heartbeat_age_s > self.heartbeat_timeout_s
        ):
            reasons.append("low-level heartbeat timeout")
        if abs(state.roll_rad) > self.max_abs_roll_rad:
            reasons.append(f"roll too large: {state.roll_rad:.3f} rad")
        if abs(state.pitch_rad) > self.max_abs_pitch_rad:
            reasons.append(f"pitch too large: {state.pitch_rad:.3f} rad")
        if state.control_loop_dt_s is not None and state.control_loop_dt_s > self.max_control_loop_dt_s:
            reasons.append(f"control loop deadline miss: {state.control_loop_dt_s:.4f} s")

        cmd = np.asarray(state.command, dtype=np.float64).reshape(3)
        if (
            cmd[0] < self.command_limits["vx_min"]
            or cmd[0] > self.command_limits["vx_max"]
            or cmd[1] < self.command_limits["vy_min"]
            or cmd[1] > self.command_limits["vy_max"]
            or cmd[2] < self.command_limits["wz_min"]
            or cmd[2] > self.command_limits["wz_max"]
        ):
            reasons.append(f"velocity command outside training range: {cmd.tolist()}")

        if observation is not None and not np.isfinite(observation).all():
            reasons.append("observation NaN/Inf")
        if raw_action is not None:
            raw_action = np.asarray(raw_action)
            if not np.isfinite(raw_action).all():
                reasons.append("policy action NaN/Inf")
            if np.max(np.abs(raw_action)) > self.action_clip + 1.0e-4:
                reasons.append("policy action magnitude too large")

        if command is not None:
            main_vel = np.asarray(command.target_main_drive_velocity, dtype=np.float64)
            abad_pos = np.asarray(command.target_abad_position, dtype=np.float64)
            if not np.isfinite(main_vel).all() or not np.isfinite(abad_pos).all():
                reasons.append("decoded command NaN/Inf")
            if np.max(np.abs(main_vel)) > self.main_drive_vel_limit_rad_s + 1.0e-6:
                reasons.append("main drive velocity target exceeds limit")
            if np.max(np.abs(abad_pos)) > self.abad_pos_limit_rad + 1.0e-6:
                reasons.append("ABAD position target exceeds limit")

        if state.motor_temperatures_c and max(state.motor_temperatures_c) > self.max_motor_temperature_c:
            reasons.append("motor temperature too high")
        if state.motor_currents_a and max(abs(x) for x in state.motor_currents_a) > self.max_motor_current_a:
            reasons.append("motor current too high")
        if any(state.motor_faults):
            reasons.append("motor fault flag")

        return SafetyResult(ok=len(reasons) == 0, reasons=reasons)
