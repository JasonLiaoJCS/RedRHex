"""Tests for reward diff logic — independently runnable, no Isaac Sim required."""
from __future__ import annotations

import pytest

from tools.training_panel.training_panel.rewards import (
    read_reward_scales_from_yaml,
    reward_diff,
)


SAMPLE_YAML = """\
rew_scale_forward_vel: 5.0
rew_scale_vel_tracking: 4.0
rew_scale_alive: 0.15
rew_scale_orientation: -0.3
rew_scale_collision: -1.0
"""

DEFAULTS = {
    "rew_scale_forward_vel": 3.0,
    "rew_scale_vel_tracking": 4.0,
    "rew_scale_alive": 0.15,
    "rew_scale_orientation": -0.3,
    "rew_scale_collision": -1.0,
}


# ---- read_reward_scales_from_yaml ----

def test_parses_positive_and_negative_scales(tmp_path):
    yaml_file = tmp_path / "env.yaml"
    yaml_file.write_text(SAMPLE_YAML)
    scales = read_reward_scales_from_yaml(yaml_file)
    assert scales["rew_scale_forward_vel"] == 5.0
    assert scales["rew_scale_orientation"] == -0.3
    assert scales["rew_scale_collision"] == -1.0


def test_returns_empty_for_missing_file(tmp_path):
    scales = read_reward_scales_from_yaml(tmp_path / "nonexistent.yaml")
    assert scales == {}


def test_ignores_non_reward_scale_lines(tmp_path):
    yaml_file = tmp_path / "env.yaml"
    yaml_file.write_text("num_envs: 64\nrew_scale_alive: 0.5\ntask: foo\n")
    scales = read_reward_scales_from_yaml(yaml_file)
    assert list(scales.keys()) == ["rew_scale_alive"]


def test_handles_zero_values(tmp_path):
    yaml_file = tmp_path / "env.yaml"
    yaml_file.write_text("rew_scale_action_rate: 0.0\n")
    scales = read_reward_scales_from_yaml(yaml_file)
    assert scales["rew_scale_action_rate"] == 0.0


# ---- reward_diff ----

def test_diff_empty_when_all_defaults():
    result = reward_diff(DEFAULTS.copy(), DEFAULTS.copy())
    assert result["changed"] == []
    assert set(result["same"]) == set(DEFAULTS.keys())


def test_diff_detects_overridden_positive_value():
    yaml_scales = {**DEFAULTS, "rew_scale_forward_vel": 5.0}
    result = reward_diff(yaml_scales, DEFAULTS)
    changed_names = [c["name"] for c in result["changed"]]
    assert "rew_scale_forward_vel" in changed_names
    item = next(c for c in result["changed"] if c["name"] == "rew_scale_forward_vel")
    assert item["yaml_value"] == 5.0
    assert item["default_value"] == 3.0
    assert item["delta_pct"] == pytest.approx(66.7, abs=0.2)


def test_diff_detects_overridden_negative_value():
    yaml_scales = {**DEFAULTS, "rew_scale_orientation": -0.6}
    result = reward_diff(yaml_scales, DEFAULTS)
    item = next(c for c in result["changed"] if c["name"] == "rew_scale_orientation")
    assert item["yaml_value"] == -0.6
    assert item["default_value"] == -0.3


def test_diff_unchanged_fields_in_same_list():
    yaml_scales = {**DEFAULTS, "rew_scale_forward_vel": 5.0}
    result = reward_diff(yaml_scales, DEFAULTS)
    assert "rew_scale_vel_tracking" in result["same"]
    assert "rew_scale_forward_vel" not in result["same"]


def test_diff_handles_unknown_yaml_field():
    """A field in YAML not in defaults (new rew_scale_*) appears in changed."""
    yaml_scales = {**DEFAULTS, "rew_scale_new_field": 1.0}
    result = reward_diff(yaml_scales, DEFAULTS)
    item = next((c for c in result["changed"] if c["name"] == "rew_scale_new_field"), None)
    assert item is not None
    assert item["default_value"] is None
    assert item["delta_pct"] is None


def test_diff_skips_fields_missing_from_yaml():
    """If a default field is absent from YAML, it's not in the diff."""
    yaml_scales = {"rew_scale_forward_vel": 3.0}  # only one field
    result = reward_diff(yaml_scales, DEFAULTS)
    names_in_diff = [c["name"] for c in result["changed"]] + result["same"]
    assert "rew_scale_orientation" not in names_in_diff


# ---- HistoryStore.get_reward_config_for_run (integration-ish) ----

def test_get_reward_config_returns_none_for_missing_params_dir(tmp_path):
    from tools.training_panel.training_panel.history import HistoryStore
    from tools.training_panel.training_panel.config import PanelPaths
    from pathlib import Path

    # Create a minimal PanelPaths pointing at tmp_path
    paths = PanelPaths(
        repo_root=tmp_path,
        isaaclab_root=Path("/nonexistent"),
        isaacsim_root=Path("/nonexistent"),
        conda_sh=Path("/nonexistent"),
        conda_env="none",
    )
    store = HistoryStore(paths)
    # Add a run with a log_dir that has no params/env.yaml
    log_dir = tmp_path / "logs" / "rsl_rl" / "redrhex_wheg" / "test-run-01"
    log_dir.mkdir(parents=True)
    store.add_run({
        "id": "test-run-01",
        "source": "rsl_rl",
        "status": "completed",
        "log_dir": str(log_dir),
        "created_at": "2026-05-15T12:00:00",
        "updated_at": "2026-05-15T12:00:00",
    })
    result = store.get_reward_config_for_run("test-run-01")
    assert result is None


def test_get_reward_config_returns_diff_for_existing_yaml(tmp_path):
    from tools.training_panel.training_panel.history import HistoryStore
    from tools.training_panel.training_panel.config import PanelPaths
    from pathlib import Path

    # Create a minimal repo structure with env_cfg.py so reward_defaults can find scales
    cfg_dir = tmp_path / "source" / "RedRhex" / "RedRhex" / "tasks" / "direct" / "redrhex"
    cfg_dir.mkdir(parents=True)
    (cfg_dir / "redrhex_env_cfg.py").write_text(
        "rew_scale_forward_vel = 3.0  # fwd\n"
        "rew_scale_alive = 0.15  # alive\n"
    )

    # Create a run with params/env.yaml that overrides rew_scale_forward_vel
    log_dir = tmp_path / "logs" / "rsl_rl" / "redrhex_wheg" / "test-run-02"
    params_dir = log_dir / "params"
    params_dir.mkdir(parents=True)
    (params_dir / "env.yaml").write_text(
        "rew_scale_forward_vel: 6.0\n"
        "rew_scale_alive: 0.15\n"
    )

    paths = PanelPaths(
        repo_root=tmp_path,
        isaaclab_root=Path("/nonexistent"),
        isaacsim_root=Path("/nonexistent"),
        conda_sh=Path("/nonexistent"),
        conda_env="none",
    )
    store = HistoryStore(paths)
    store.add_run({
        "id": "test-run-02",
        "source": "rsl_rl",
        "status": "completed",
        "log_dir": str(log_dir),
        "created_at": "2026-05-15T12:00:00",
        "updated_at": "2026-05-15T12:00:00",
    })

    result = store.get_reward_config_for_run("test-run-02")
    assert result is not None
    changed_names = [c["name"] for c in result["changed"]]
    assert "rew_scale_forward_vel" in changed_names
    assert "rew_scale_alive" not in changed_names


def test_get_reward_config_can_compare_with_previous_run(tmp_path):
    from tools.training_panel.training_panel.history import HistoryStore
    from tools.training_panel.training_panel.config import PanelPaths
    from pathlib import Path

    cfg_dir = tmp_path / "source" / "RedRhex" / "RedRhex" / "tasks" / "direct" / "redrhex"
    cfg_dir.mkdir(parents=True)
    (cfg_dir / "redrhex_env_cfg.py").write_text(
        "rew_scale_forward_vel = 3.0\n"
        "rew_scale_alive = 0.15\n"
    )

    previous_log = tmp_path / "logs" / "rsl_rl" / "redrhex_wheg" / "previous-run"
    current_log = tmp_path / "logs" / "rsl_rl" / "redrhex_wheg" / "current-run"
    for log_dir in (previous_log, current_log):
        (log_dir / "params").mkdir(parents=True)
    (previous_log / "params" / "env.yaml").write_text(
        "rew_scale_forward_vel: 4.0\n"
        "rew_scale_alive: 0.2\n"
    )
    (current_log / "params" / "env.yaml").write_text(
        "rew_scale_forward_vel: 6.0\n"
        "rew_scale_alive: 0.2\n"
    )

    paths = PanelPaths(
        repo_root=tmp_path,
        isaaclab_root=Path("/nonexistent"),
        isaacsim_root=Path("/nonexistent"),
        conda_sh=Path("/nonexistent"),
        conda_env="none",
    )
    store = HistoryStore(paths)
    store.add_run({
        "id": "previous-run",
        "source": "rsl_rl",
        "status": "completed",
        "log_dir": str(previous_log),
        "created_at": "2026-05-15T12:00:00",
        "updated_at": "2026-05-15T12:00:00",
    })
    store.add_run({
        "id": "current-run",
        "source": "rsl_rl",
        "status": "completed",
        "log_dir": str(current_log),
        "created_at": "2026-05-15T13:00:00",
        "updated_at": "2026-05-15T13:00:00",
    })

    result = store.get_reward_config_for_run("current-run", compare_to="previous")

    assert result["baseline_kind"] == "previous"
    assert result["baseline_run_id"] == "previous-run"
    assert [item["name"] for item in result["changed"]] == ["rew_scale_forward_vel"]
    assert result["changed"][0]["default_value"] == 4.0
    assert result["changed"][0]["delta_pct"] == 50.0


def test_get_reward_config_previous_reports_missing_baseline(tmp_path):
    from tools.training_panel.training_panel.history import HistoryStore
    from tools.training_panel.training_panel.config import PanelPaths
    from pathlib import Path

    cfg_dir = tmp_path / "source" / "RedRhex" / "RedRhex" / "tasks" / "direct" / "redrhex"
    cfg_dir.mkdir(parents=True)
    (cfg_dir / "redrhex_env_cfg.py").write_text("rew_scale_forward_vel = 3.0\n")
    log_dir = tmp_path / "logs" / "rsl_rl" / "redrhex_wheg" / "only-run" / "params"
    log_dir.mkdir(parents=True)
    (log_dir / "env.yaml").write_text("rew_scale_forward_vel: 4.0\n")

    paths = PanelPaths(
        repo_root=tmp_path,
        isaaclab_root=Path("/nonexistent"),
        isaacsim_root=Path("/nonexistent"),
        conda_sh=Path("/nonexistent"),
        conda_env="none",
    )
    store = HistoryStore(paths)
    store.add_run({
        "id": "only-run",
        "source": "rsl_rl",
        "status": "completed",
        "log_dir": str(log_dir.parent),
        "created_at": "2026-05-15T12:00:00",
    })

    result = store.get_reward_config_for_run("only-run", compare_to="previous")

    assert result["baseline_kind"] == "previous"
    assert result["baseline_missing"] is True
    assert result["changed"] == []
