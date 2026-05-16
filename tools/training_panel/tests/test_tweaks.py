from __future__ import annotations

from tools.training_panel.training_panel.tweaks import (
    build_tweak_payload,
    newest_finished_tweak_run,
)


REWARD_PRESETS = [
    {
        "id": "baseline",
        "name": "Baseline",
        "values": {},
    },
    {
        "id": "speed",
        "name": "Speed",
        "values": {"rew_scale_forward_vel": 5.0},
    },
]

TERRAIN_PRESETS = [
    {
        "id": "flat-debug",
        "name": "Flat",
        "values": {"terrain.terrain_type": "plane"},
    }
]


def test_newest_finished_tweak_run_skips_running_runs():
    runs = [
        {
            "id": "running",
            "status": "running",
            "created_at": "2026-05-16T12:00:00",
            "params": {"reward_overrides": {"rew_scale_alive": 0.1}},
        },
        {
            "id": "finished",
            "status": "failed",
            "created_at": "2026-05-16T11:00:00",
            "params": {"reward_overrides": {"rew_scale_alive": 0.2}},
        },
    ]

    selected = newest_finished_tweak_run(runs, REWARD_PRESETS)

    assert selected is not None
    assert selected["id"] == "finished"


def test_tweak_payload_prefers_saved_env_yaml_reward_scales(tmp_path):
    log_dir = tmp_path / "run"
    params_dir = log_dir / "params"
    params_dir.mkdir(parents=True)
    (params_dir / "env.yaml").write_text(
        "rew_scale_forward_vel: 7.0\n"
        "rew_scale_alive: 0.3\n",
        encoding="utf-8",
    )
    run = {
        "id": "run_yaml",
        "display_name": "YAML source",
        "status": "completed",
        "created_at": "2026-05-16T10:00:00",
        "log_dir": str(log_dir),
        "params": {
            "task": "Template-Redrhex-Direct-v0",
            "num_envs": 32,
            "max_iterations": 100,
            "device": "cuda:0",
            "reward_preset_id": "speed",
            "reward_overrides": {"rew_scale_forward_vel": 5.0},
            "terrain_preset_id": "flat-debug",
            "terrain_overrides": {"terrain.terrain_type": "plane"},
        },
    }

    payload = build_tweak_payload(run, reward_presets=REWARD_PRESETS, terrain_presets=TERRAIN_PRESETS)

    assert payload["reward_preset"]["values"]["rew_scale_forward_vel"] == 7.0
    assert payload["reward_preset"]["values"]["rew_scale_alive"] == 0.3
    assert payload["training_params"]["checkpoint"] == ""
    assert payload["training_params"]["resume"] is False
    assert payload["training_params"]["terrain_overrides"] == {"terrain.terrain_type": "plane"}
    assert payload["training_params"]["tweak_source_run_id"] == "run_yaml"


def test_tweak_payload_falls_back_to_recorded_reward_overrides():
    run = {
        "id": "run_overrides",
        "status": "interrupted",
        "created_at": "2026-05-16T09:00:00",
        "params": {
            "reward_preset_id": "speed",
            "reward_overrides": {"rew_scale_alive": 0.25},
            "terrain_preset_id": "flat-debug",
        },
    }

    payload = build_tweak_payload(run, reward_presets=REWARD_PRESETS, terrain_presets=TERRAIN_PRESETS)

    assert payload["reward_preset"]["values"] == {"rew_scale_alive": 0.25}
    assert payload["training_params"]["terrain_overrides"] == {"terrain.terrain_type": "plane"}
