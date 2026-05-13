"""Build real-robot observations matching RedRhex IsaacLab policy inputs."""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from . import redrhex_contract as C


@dataclass
class ObservationStatus:
    ok: bool
    reasons: list[str] = field(default_factory=list)


def _stamp_to_float(msg: object | None, fallback: float | None = None) -> float:
    if msg is None:
        return time.monotonic() if fallback is None else fallback
    sec = getattr(getattr(msg, "header", None), "stamp", None)
    if sec is None:
        return time.monotonic() if fallback is None else fallback
    stamp_s = float(getattr(sec, "sec", 0)) + 1.0e-9 * float(getattr(sec, "nanosec", 0))
    return stamp_s if stamp_s > 0.0 else (time.monotonic() if fallback is None else fallback)


def _normalize_quat_xyzw(q: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(q)
    if norm < 1.0e-9:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return q / norm


def _quat_inverse_rotate_xyzw(q_xyzw: np.ndarray, v: np.ndarray) -> np.ndarray:
    x, y, z, w = _normalize_quat_xyzw(q_xyzw)
    # Rotation matrix body->world for ROS xyzw quaternion, then transpose.
    r00 = 1.0 - 2.0 * (y * y + z * z)
    r01 = 2.0 * (x * y - z * w)
    r02 = 2.0 * (x * z + y * w)
    r10 = 2.0 * (x * y + z * w)
    r11 = 1.0 - 2.0 * (x * x + z * z)
    r12 = 2.0 * (y * z - x * w)
    r20 = 2.0 * (x * z - y * w)
    r21 = 2.0 * (y * z + x * w)
    r22 = 1.0 - 2.0 * (x * x + y * y)
    return np.array(
        [
            r00 * v[0] + r10 * v[1] + r20 * v[2],
            r01 * v[0] + r11 * v[1] + r21 * v[2],
            r02 * v[0] + r12 * v[1] + r22 * v[2],
        ],
        dtype=np.float64,
    )


def _quat_to_roll_pitch_yaw(q_xyzw: np.ndarray) -> tuple[float, float, float]:
    x, y, z, w = _normalize_quat_xyzw(q_xyzw)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


class ObservationBuilder:
    """Stateful observation builder.

    Single-frame observation is the 56-D vector from RedrhexEnv._get_observations().
    If the exported ONNX expects 280-D input, build_policy_input() returns
    [current_obs, previous_obs_1, ..., previous_obs_4], matching the RSL-RL
    "policy"+"history" group order used by this repo's newer configs.
    """

    def __init__(self, config: dict | None = None) -> None:
        self.cfg = config or {}
        self.expected_obs_dim = int(self.cfg.get("expected_obs_dim", C.OBS_DIM_SINGLE))
        self.policy_input_dim = int(self.cfg.get("policy_input_dim", self.expected_obs_dim))
        self.history_length = int(self.cfg.get("policy_history_length", C.POLICY_HISTORY_LENGTH))
        self.base_gait_angular_vel = float(self.cfg.get("base_gait_angular_vel", C.BASE_GAIT_ANGULAR_VEL))
        self.abad_pos_scale = float(self.cfg.get("abad_pos_scale", C.ABAD_POS_SCALE))
        self.base_gait_frequency_hz = float(self.cfg.get("base_gait_frequency_hz", C.BASE_GAIT_FREQUENCY_HZ))
        self.base_lin_vel_source = str(self.cfg.get("base_lin_vel_source", "zero"))
        self.odom_twist_in_body_frame = bool(self.cfg.get("odom_twist_in_body_frame", True))
        self.command_limits = dict(C.COMMAND_LIMITS)
        self.command_limits.update(self.cfg.get("command_limits", {}))

        self.main_drive_joint_names = list(self.cfg.get("main_drive_joint_names", C.MAIN_DRIVE_JOINT_NAMES))
        self.abad_joint_names = list(self.cfg.get("abad_joint_names", C.ABAD_JOINT_NAMES))
        self.required_joint_names = self.main_drive_joint_names + self.abad_joint_names
        self._validate_config()

        self.last_actions = np.zeros(C.ACTION_DIM, dtype=np.float32)
        self.gait_phase = 0.0
        self._history: deque[np.ndarray] = deque(maxlen=max(1, self.history_length))
        self._last_build_time: float | None = None

        self.imu_quat_xyzw: np.ndarray | None = None
        self.imu_ang_vel = np.zeros(3, dtype=np.float64)
        self.imu_time: float | None = None
        self.joint_pos: dict[str, float] = {}
        self.joint_vel: dict[str, float] = {}
        self.joint_time: float | None = None
        self.cmd_vel = np.zeros(3, dtype=np.float64)
        self.cmd_time: float | None = None
        self.odom_lin_vel = np.zeros(3, dtype=np.float64)
        self.odom_time: float | None = None

    def _validate_config(self) -> None:
        if self.expected_obs_dim != C.OBS_DIM_SINGLE:
            raise ValueError(f"expected_obs_dim must be {C.OBS_DIM_SINGLE}, got {self.expected_obs_dim}")
        if self.history_length <= 0:
            raise ValueError("policy_history_length must be positive")
        if self.policy_input_dim not in (C.OBS_DIM_SINGLE, C.OBS_DIM_SINGLE * self.history_length):
            raise ValueError(
                f"policy_input_dim must be {C.OBS_DIM_SINGLE} or {C.OBS_DIM_SINGLE * self.history_length}, "
                f"got {self.policy_input_dim}"
            )
        if self.base_lin_vel_source not in ("zero", "odom"):
            raise ValueError("base_lin_vel_source must be 'zero' or 'odom'")

        for name, joint_names in (
            ("main_drive_joint_names", self.main_drive_joint_names),
            ("abad_joint_names", self.abad_joint_names),
        ):
            if len(joint_names) != 6:
                raise ValueError(f"{name} must contain 6 joints, got {len(joint_names)}")
            if len(set(joint_names)) != len(joint_names):
                raise ValueError(f"{name} contains duplicate names: {joint_names}")
        if set(self.main_drive_joint_names).intersection(self.abad_joint_names):
            raise ValueError("main_drive_joint_names and abad_joint_names overlap")

        for key in ("vx", "vy", "wz"):
            lo = self.command_limits[f"{key}_min"]
            hi = self.command_limits[f"{key}_max"]
            if not np.isfinite([lo, hi]).all() or lo > hi:
                raise ValueError(f"invalid command limit for {key}: min={lo}, max={hi}")

    def reset(self, gait_phase: float = 0.0) -> None:
        self.gait_phase = float(gait_phase) % (2.0 * math.pi)
        self.last_actions[:] = 0.0
        self._history.clear()
        self._last_build_time = None

    def update_imu(self, msg: object, now_s: float | None = None) -> None:
        orientation = getattr(msg, "orientation")
        angular_velocity = getattr(msg, "angular_velocity")
        self.imu_quat_xyzw = np.array(
            [orientation.x, orientation.y, orientation.z, orientation.w], dtype=np.float64
        )
        self.imu_ang_vel = np.array([angular_velocity.x, angular_velocity.y, angular_velocity.z], dtype=np.float64)
        self.imu_time = now_s if now_s is not None else _stamp_to_float(msg)

    def update_joint_state(self, msg: object, now_s: float | None = None) -> None:
        names: Iterable[str] = getattr(msg, "name", [])
        positions = list(getattr(msg, "position", []))
        velocities = list(getattr(msg, "velocity", []))
        for idx, name in enumerate(names):
            if idx < len(positions):
                self.joint_pos[str(name)] = float(positions[idx])
            if idx < len(velocities):
                self.joint_vel[str(name)] = float(velocities[idx])
        self.joint_time = now_s if now_s is not None else _stamp_to_float(msg)

    def update_cmd_vel(self, msg: object, now_s: float | None = None) -> None:
        linear = getattr(msg, "linear")
        angular = getattr(msg, "angular")
        cmd = np.array([linear.x, linear.y, angular.z], dtype=np.float64)
        cmd[0] = np.clip(cmd[0], self.command_limits["vx_min"], self.command_limits["vx_max"])
        cmd[1] = np.clip(cmd[1], self.command_limits["vy_min"], self.command_limits["vy_max"])
        cmd[2] = np.clip(cmd[2], self.command_limits["wz_min"], self.command_limits["wz_max"])
        self.cmd_vel = cmd
        self.cmd_time = now_s if now_s is not None else _stamp_to_float(msg)

    def update_odom(self, msg: object, now_s: float | None = None) -> None:
        twist = getattr(getattr(msg, "twist"), "twist")
        lin = np.array([twist.linear.x, twist.linear.y, twist.linear.z], dtype=np.float64)
        if not self.odom_twist_in_body_frame and self.imu_quat_xyzw is not None:
            lin = _quat_inverse_rotate_xyzw(self.imu_quat_xyzw, lin)
        self.odom_lin_vel = lin
        self.odom_time = now_s if now_s is not None else _stamp_to_float(msg)

    def update_last_actions(self, action: np.ndarray) -> None:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape != (C.ACTION_DIM,):
            raise ValueError(f"last action shape must be ({C.ACTION_DIM},), got {action.shape}")
        self.last_actions = action.copy()

    def status(self, now_s: float, sensor_timeout_s: float, cmd_timeout_s: float) -> ObservationStatus:
        reasons: list[str] = []
        if self.imu_time is None:
            reasons.append("waiting for /imu/data")
        elif now_s - self.imu_time > sensor_timeout_s:
            reasons.append("IMU timeout")
        if self.joint_time is None:
            reasons.append("waiting for /joint_states")
        elif now_s - self.joint_time > sensor_timeout_s:
            reasons.append("joint_states timeout")
        missing = [name for name in self.required_joint_names if name not in self.joint_pos]
        if missing:
            reasons.append(f"missing joints: {missing}")
        if self.cmd_time is not None and now_s - self.cmd_time > cmd_timeout_s:
            self.cmd_vel[:] = 0.0
        return ObservationStatus(ok=len(reasons) == 0, reasons=reasons)

    def build_single(self, now_s: float, update_phase: bool = True) -> np.ndarray:
        if self.imu_quat_xyzw is None:
            raise RuntimeError("Cannot build observation before IMU is received.")
        missing = [name for name in self.required_joint_names if name not in self.joint_pos]
        if missing:
            raise RuntimeError(f"Cannot build observation; missing joint_states for {missing}.")

        if update_phase:
            if self._last_build_time is None:
                dt = C.CONTROL_DT
            else:
                dt = max(0.0, min(0.05, now_s - self._last_build_time))
            self.gait_phase = (self.gait_phase + 2.0 * math.pi * self.base_gait_frequency_hz * dt) % (2.0 * math.pi)
            self._last_build_time = now_s

        if self.base_lin_vel_source == "odom" and self.odom_time is not None:
            base_lin_vel = self.odom_lin_vel
        else:
            base_lin_vel = np.zeros(3, dtype=np.float64)

        projected_gravity = _quat_inverse_rotate_xyzw(self.imu_quat_xyzw, np.array([0.0, 0.0, -1.0]))
        main_pos = np.array([self.joint_pos[name] for name in self.main_drive_joint_names], dtype=np.float64)
        main_vel = np.array([self.joint_vel.get(name, 0.0) for name in self.main_drive_joint_names], dtype=np.float64)
        abad_pos = np.array([self.joint_pos[name] for name in self.abad_joint_names], dtype=np.float64)
        abad_vel = np.array([self.joint_vel.get(name, 0.0) for name in self.abad_joint_names], dtype=np.float64)

        obs = np.concatenate(
            [
                base_lin_vel,
                self.imu_ang_vel,
                projected_gravity,
                np.sin(main_pos),
                np.cos(main_pos),
                main_vel / self.base_gait_angular_vel,
                abad_pos / self.abad_pos_scale,
                abad_vel,
                self.cmd_vel,
                np.array([math.sin(self.gait_phase), math.cos(self.gait_phase)], dtype=np.float64),
                self.last_actions.astype(np.float64),
            ]
        ).astype(np.float32)

        if obs.shape != (C.OBS_DIM_SINGLE,):
            raise RuntimeError(f"Observation dim {obs.shape[0]} != {C.OBS_DIM_SINGLE}.")
        if not np.isfinite(obs).all():
            raise RuntimeError("Observation contains NaN or Inf.")
        return obs

    def build_policy_input(self, now_s: float) -> np.ndarray:
        obs = self.build_single(now_s)
        self._history.appendleft(obs.copy())
        if self.policy_input_dim == C.OBS_DIM_SINGLE:
            return obs
        expected_history_dim = C.OBS_DIM_SINGLE * self.history_length
        if self.policy_input_dim != expected_history_dim:
            raise RuntimeError(
                f"Unsupported policy_input_dim={self.policy_input_dim}; expected 56 or {expected_history_dim}."
            )
        frames = list(self._history)
        while len(frames) < self.history_length:
            frames.append(np.zeros(C.OBS_DIM_SINGLE, dtype=np.float32))
        stacked = np.concatenate(frames, axis=0).astype(np.float32)
        if stacked.shape != (expected_history_dim,):
            raise RuntimeError(f"Stacked observation dim {stacked.shape[0]} != {expected_history_dim}.")
        return stacked

    def get_roll_pitch_yaw(self) -> tuple[float, float, float]:
        if self.imu_quat_xyzw is None:
            return 0.0, 0.0, 0.0
        return _quat_to_roll_pitch_yaw(self.imu_quat_xyzw)

    def get_main_drive_positions(self) -> np.ndarray:
        return np.array([self.joint_pos.get(name, 0.0) for name in self.main_drive_joint_names], dtype=np.float64)

    def get_abad_positions(self) -> np.ndarray:
        return np.array([self.joint_pos.get(name, 0.0) for name in self.abad_joint_names], dtype=np.float64)
