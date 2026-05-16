from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any

from .commands import DEFAULT_TASK
from .rewards import read_reward_scales_from_yaml


_ACTIVE_RUN_STATUSES = {"running", "stopping"}
_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slug(value: str) -> str:
    slug = _SLUG_RE.sub("-", str(value or "").lower().strip()).strip("-")
    return slug or "run"


def _run_label(run: dict[str, Any]) -> str:
    return str(run.get("display_name") or run.get("id") or "run")


def _run_time(run: dict[str, Any]) -> float:
    for key in ("created_at", "updated_at"):
        value = run.get(key)
        if not value:
            continue
        try:
            normalized = str(value).replace("Z", "+00:00")
            return datetime.fromisoformat(normalized).timestamp()
        except ValueError:
            continue
    return 0.0


def _params(run: dict[str, Any]) -> dict[str, Any]:
    params = run.get("params")
    return params if isinstance(params, dict) else {}


def _env_yaml(run: dict[str, Any]) -> Path | None:
    log_dir = run.get("log_dir")
    if not log_dir:
        return None
    path = Path(str(log_dir)) / "params" / "env.yaml"
    return path if path.exists() else None


def _preset_values(presets: list[dict[str, Any]], preset_id: str | None) -> dict[str, Any]:
    if not preset_id:
        return {}
    for preset in presets:
        if str(preset.get("id") or "") == str(preset_id):
            values = preset.get("values")
            return dict(values) if isinstance(values, dict) else {}
    return {}


def _float_values(values: dict[str, Any]) -> dict[str, float]:
    normalized: dict[str, float] = {}
    for key, value in values.items():
        try:
            normalized[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return normalized


def _dict_values(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _first_dict(*values: Any) -> dict[str, Any]:
    for value in values:
        data = _dict_values(value)
        if data:
            return data
    return {}


def reward_values_for_tweak(run: dict[str, Any], reward_presets: list[dict[str, Any]]) -> dict[str, float]:
    """Return editable reward values for a tweak draft, preferring the exact saved run config."""
    env_yaml = _env_yaml(run)
    if env_yaml:
        yaml_values = read_reward_scales_from_yaml(env_yaml)
        if yaml_values:
            return yaml_values

    params = _params(run)
    override_values = _first_dict(params.get("reward_overrides"), run.get("reward_overrides"))
    if override_values:
        return _float_values(override_values)

    preset_id = str(params.get("reward_preset_id") or run.get("reward_preset_id") or "")
    return _float_values(_preset_values(reward_presets, preset_id))


def terrain_values_for_tweak(run: dict[str, Any], terrain_presets: list[dict[str, Any]]) -> dict[str, Any]:
    params = _params(run)
    override_values = _first_dict(params.get("terrain_overrides"), run.get("terrain_overrides"))
    if override_values:
        return override_values
    preset_id = str(params.get("terrain_preset_id") or run.get("terrain_preset_id") or "")
    return _preset_values(terrain_presets, preset_id)


def run_can_seed_tweak(run: dict[str, Any], reward_presets: list[dict[str, Any]]) -> bool:
    if str(run.get("status") or "").lower() in _ACTIVE_RUN_STATUSES:
        return False
    return bool(_params(run) or reward_values_for_tweak(run, reward_presets))


def newest_finished_tweak_run(runs: list[dict[str, Any]], reward_presets: list[dict[str, Any]]) -> dict[str, Any] | None:
    ordered = sorted(runs, key=_run_time, reverse=True)
    for run in ordered:
        if run_can_seed_tweak(run, reward_presets):
            return run
    return None


def build_tweak_payload(
    run: dict[str, Any],
    *,
    reward_presets: list[dict[str, Any]],
    terrain_presets: list[dict[str, Any]],
) -> dict[str, Any]:
    if not run_can_seed_tweak(run, reward_presets):
        raise ValueError("Run does not have usable training or reward data for tweaking")

    params = _params(run)
    source_id = str(run.get("id") or "")
    source_label = _run_label(run)
    reward_values = reward_values_for_tweak(run, reward_presets)
    terrain_preset_id = str(params.get("terrain_preset_id") or run.get("terrain_preset_id") or "baseline")
    terrain_values = terrain_values_for_tweak(run, terrain_presets)
    draft_id = f"tweak-{_slug(source_id or source_label)}"

    training_params = {
        "task": str(params.get("task") or DEFAULT_TASK),
        "num_envs": int(params.get("num_envs") or 4),
        "max_iterations": int(params.get("max_iterations") or 1),
        "device": str(params.get("device") or "cuda:0"),
        "headless": bool(params.get("headless", True)),
        "seed": params.get("seed"),
        "resume": False,
        "checkpoint": "",
        "reward_preset_id": draft_id,
        "reward_overrides": reward_values,
        "terrain_preset_id": terrain_preset_id,
        "terrain_overrides": terrain_values,
        "tweak_source_run_id": source_id,
        "tweak_source_label": source_label,
    }

    if training_params["seed"] in ("", None):
        training_params["seed"] = None

    return {
        "source_run": {
            "id": source_id,
            "display_name": run.get("display_name"),
            "status": run.get("status"),
            "created_at": run.get("created_at"),
            "updated_at": run.get("updated_at"),
            "log_dir": run.get("log_dir"),
        },
        "training_params": training_params,
        "reward_preset": {
            "id": draft_id,
            "name": f"Tweak from {source_label}",
            "description": f"Unsaved reward draft copied from {source_label}.",
            "values": reward_values,
            "built_in": False,
            "draft": True,
            "source_run_id": source_id,
        },
        "terrain_preset_id": terrain_preset_id,
        "terrain_overrides": terrain_values,
        "message": f"Loaded tweak draft from {source_label}.",
    }
