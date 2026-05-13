"""Preflight checks before RedRhex real-robot bringup."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from . import redrhex_contract as C
from .action_decoder import ActionDecoder
from .observation_builder import ObservationBuilder
from .policy_onnx_runner import PolicyONNXRunner
from .safety_filter import SafetyFilter


def _append_check(result: dict[str, object], name: str, ok: bool, **fields: object) -> None:
    check = {"name": name, "ok": bool(ok)}
    check.update(fields)
    result["checks"].append(check)


def _load_ros_params(config_path: str | None, result: dict[str, object]) -> dict:
    if not config_path:
        result["warnings"].append("No --config provided; YAML parameter validation skipped.")
        return {}
    path = Path(config_path).expanduser()
    if not path.exists():
        _append_check(result, "config_exists", False, path=str(path))
        return {}
    _append_check(result, "config_exists", True, path=str(path))
    try:
        import yaml
    except Exception as exc:
        result["warnings"].append(f"PyYAML unavailable, YAML validation skipped: {exc}")
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    params = data.get("redrhex_rl_controller", {}).get("ros__parameters", {})
    if not isinstance(params, dict):
        _append_check(result, "config_schema", False, error="missing redrhex_rl_controller.ros__parameters")
        return {}
    _append_check(result, "config_schema", True)
    return params


def _nested(params: dict, *keys: str, default=None):
    cur = params
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _command_limits_from_params(params: dict) -> dict:
    return {
        "vx_min": float(_nested(params, "commands", "vx_min", default=C.COMMAND_LIMITS["vx_min"])),
        "vx_max": float(_nested(params, "commands", "vx_max", default=C.COMMAND_LIMITS["vx_max"])),
        "vy_min": float(_nested(params, "commands", "vy_min", default=C.COMMAND_LIMITS["vy_min"])),
        "vy_max": float(_nested(params, "commands", "vy_max", default=C.COMMAND_LIMITS["vy_max"])),
        "wz_min": float(_nested(params, "commands", "wz_min", default=C.COMMAND_LIMITS["wz_min"])),
        "wz_max": float(_nested(params, "commands", "wz_max", default=C.COMMAND_LIMITS["wz_max"])),
    }


def _validate_deployment_config(params: dict, policy_input_dim: int, result: dict[str, object]) -> None:
    if not params:
        return
    command_limits = _command_limits_from_params(params)
    try:
        ObservationBuilder(
            {
                "expected_obs_dim": C.OBS_DIM_SINGLE,
                "policy_input_dim": int(policy_input_dim),
                "policy_history_length": int(
                    _nested(params, "observation", "policy_history_length", default=C.POLICY_HISTORY_LENGTH)
                ),
                "base_lin_vel_source": str(_nested(params, "observation", "base_lin_vel_source", default="zero")),
                "odom_twist_in_body_frame": bool(
                    _nested(params, "observation", "odom_twist_in_body_frame", default=True)
                ),
                "command_limits": command_limits,
            }
        )
        _append_check(result, "observation_builder_config", True)
    except Exception as exc:
        _append_check(result, "observation_builder_config", False, error=str(exc))

    try:
        ActionDecoder(
            {
                "action_clip": float(_nested(params, "safety", "action_clip", default=1.0)),
                "main_drive_vel_limit_rad_s": float(
                    _nested(params, "safety", "main_drive_vel_limit_rad_s", default=30.0)
                ),
                "abad_pos_limit": float(_nested(params, "safety", "abad_pos_limit_rad", default=C.STAGE_ABAD_POS_LIMIT)),
                "main_drive_slew_rate_rad_s2": float(
                    _nested(params, "safety", "main_drive_slew_rate_rad_s2", default=120.0)
                ),
                "abad_slew_rate_rad_s": float(_nested(params, "safety", "abad_slew_rate_rad_s", default=6.0)),
                "main_drive_sign": list(_nested(params, "action", "main_drive_sign", default=[1.0] * 6)),
                "abad_sign": list(_nested(params, "action", "abad_sign", default=[1.0] * 6)),
                "damper_sign": list(_nested(params, "action", "damper_sign", default=[1.0] * 6)),
                "main_drive_zero_offset_rad": list(
                    _nested(params, "action", "main_drive_zero_offset_rad", default=[0.0] * 6)
                ),
                "abad_zero_offset_rad": list(_nested(params, "action", "abad_zero_offset_rad", default=[0.0] * 6)),
                "damper_zero_offset_rad": list(
                    _nested(params, "action", "damper_zero_offset_rad", default=[0.0] * 6)
                ),
                "main_drive_kp": list(_nested(params, "action", "main_drive_kp", default=[0.0] * 6)),
                "main_drive_kd": list(_nested(params, "action", "main_drive_kd", default=[50.0] * 6)),
                "abad_kp": list(_nested(params, "action", "abad_kp", default=[40.0] * 6)),
                "abad_kd": list(_nested(params, "action", "abad_kd", default=[4.0] * 6)),
                "stand_main_drive_kp": list(
                    _nested(params, "action", "stand_main_drive_kp", default=[12.0] * 6)
                ),
                "stand_main_drive_kd": list(_nested(params, "action", "stand_main_drive_kd", default=[1.0] * 6)),
            }
        )
        _append_check(result, "action_decoder_config", True)
    except Exception as exc:
        _append_check(result, "action_decoder_config", False, error=str(exc))

    try:
        SafetyFilter(
            {
                "sensor_timeout_s": float(_nested(params, "safety", "sensor_timeout_s", default=0.10)),
                "cmd_timeout_s": float(_nested(params, "safety", "cmd_timeout_s", default=0.25)),
                "motor_feedback_timeout_s": float(
                    _nested(params, "safety", "motor_feedback_timeout_s", default=0.25)
                ),
                "heartbeat_timeout_s": float(_nested(params, "safety", "heartbeat_timeout_s", default=0.10)),
                "max_abs_roll_rad": float(_nested(params, "safety", "max_abs_roll_rad", default=0.7)),
                "max_abs_pitch_rad": float(_nested(params, "safety", "max_abs_pitch_rad", default=0.7)),
                "action_clip": float(_nested(params, "safety", "action_clip", default=1.0)),
                "main_drive_vel_limit_rad_s": float(
                    _nested(params, "safety", "main_drive_vel_limit_rad_s", default=30.0)
                ),
                "abad_pos_limit_rad": float(_nested(params, "safety", "abad_pos_limit_rad", default=0.7)),
                "max_motor_temperature_c": float(
                    _nested(params, "safety", "max_motor_temperature_c", default=70.0)
                ),
                "max_motor_current_a": float(_nested(params, "safety", "max_motor_current_a", default=20.0)),
                "max_control_loop_dt_s": float(_nested(params, "safety", "max_control_loop_dt_s", default=0.03)),
                "command_limits": command_limits,
            }
        )
        _append_check(result, "safety_filter_config", True)
    except Exception as exc:
        _append_check(result, "safety_filter_config", False, error=str(exc))

    if bool(_nested(params, "state_machine", "enable_policy_on_start", default=False)):
        result["warnings"].append("enable_policy_on_start is true; keep it false for real-robot bringup.")
    if bool(_nested(params, "state_machine", "enable_motor_output_on_start", default=False)):
        result["warnings"].append("enable_motor_output_on_start is true; keep it false for real-robot bringup.")
    if str(_nested(params, "observation", "base_lin_vel_source", default="zero")) == "zero":
        result["warnings"].append(
            "base_lin_vel_source is zero. This is only safe for bench tests; use odom/leg odometry before serious locomotion."
        )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Check ONNX and RedRhex deployment assumptions.")
    parser.add_argument("--onnx", default=None)
    parser.add_argument("--config", default=None, help="Path to redrhex_policy.yaml for parameter validation.")
    parser.add_argument("--expected-obs-dim", type=int, default=C.OBS_DIM_SINGLE)
    parser.add_argument("--expected-action-dim", type=int, default=C.ACTION_DIM)
    parser.add_argument("--use-cuda", action="store_true")
    parser.add_argument("--use-tensorrt", action="store_true")
    args = parser.parse_args(argv)

    result: dict[str, object] = {
        "python": sys.executable,
        "numpy_version": np.__version__,
        "onnx_path": None,
        "repo_policy_hz": C.POLICY_HZ,
        "repo_control_dt_s": C.CONTROL_DT,
        "single_obs_dim": C.OBS_DIM_SINGLE,
        "history_obs_dim": C.OBS_DIM_SINGLE * C.POLICY_HISTORY_LENGTH,
        "action_dim": C.ACTION_DIM,
        "checks": [],
        "warnings": [],
        "next_steps": [],
    }
    params = _load_ros_params(args.config, result)
    onnx_path = args.onnx or _nested(params, "policy", "onnx_path", default="/home/jetson/redrhex_models/policy.onnx")
    result["onnx_path"] = str(Path(onnx_path).expanduser())

    all_names = C.ALL_CONTROLLED_JOINT_NAMES
    _append_check(
        result,
        "redrhex_contract_joint_names",
        len(C.MAIN_DRIVE_JOINT_NAMES) == 6
        and len(C.ABAD_JOINT_NAMES) == 6
        and len(C.DAMPER_JOINT_NAMES) == 6
        and len(set(all_names)) == len(all_names),
        controlled_joint_count=len(all_names),
    )
    _append_check(
        result,
        "redrhex_contract_dimensions",
        C.OBS_DIM_SINGLE == args.expected_obs_dim and C.ACTION_DIM == args.expected_action_dim,
        obs_dim=C.OBS_DIM_SINGLE,
        action_dim=C.ACTION_DIM,
    )

    path = Path(onnx_path).expanduser()
    if not path.exists():
        _append_check(result, "onnx_exists", False)
        result["next_steps"].append("Copy policy.onnx to /home/jetson/redrhex_models/policy.onnx or pass --onnx.")
        print(json.dumps(result, indent=2))
        return 2
    _append_check(result, "onnx_exists", True, size_bytes=path.stat().st_size)

    try:
        runner = PolicyONNXRunner(
            str(path),
            expected_obs_dim=args.expected_obs_dim,
            expected_action_dim=args.expected_action_dim,
            use_cuda=args.use_cuda,
            use_tensorrt=args.use_tensorrt,
            allow_history_dim=True,
        )
        info = runner.io_info
        result["onnx_io"] = {
            "input_name": info.input_name,
            "input_shape": info.input_shape,
            "output_name": info.output_name,
            "output_shape": info.output_shape,
            "providers": info.providers,
            "obs_dim": info.obs_dim,
            "action_dim": info.action_dim,
        }
        obs_dim = info.obs_dim or args.expected_obs_dim
        action = runner.run(np.zeros(obs_dim, dtype=np.float32))
        _append_check(
            result,
            "zero_observation_inference",
            bool(np.isfinite(action).all() and action.shape == (args.expected_action_dim,)),
            action_min=float(np.min(action)),
            action_max=float(np.max(action)),
        )
        if obs_dim == C.OBS_DIM_SINGLE * C.POLICY_HISTORY_LENGTH:
            result["warnings"].append("ONNX expects 280-D policy+history input; keep policy_history_length=5.")
        elif obs_dim != C.OBS_DIM_SINGLE:
            result["warnings"].append(f"Unexpected obs dim {obs_dim}; verify export and YAML.")
    except Exception as exc:
        _append_check(result, "onnx_load_and_run", False, error=str(exc))
        print(json.dumps(result, indent=2))
        return 3

    _validate_deployment_config(params, obs_dim, result)
    result["next_steps"] = [
        "Launch mock mode with use_fake_sensors:=true.",
        "Do not enable policy on hardware until INIT_STAND, single ABAD, and single main-drive tests pass.",
        "Keep /redrhex/enable_motors false until the robot is suspended, current-limited, and E-stop is ready.",
    ]
    if any(not bool(check.get("ok")) for check in result["checks"]):
        print(json.dumps(result, indent=2))
        return 4
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
