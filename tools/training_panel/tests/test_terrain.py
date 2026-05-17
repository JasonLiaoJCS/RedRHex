from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.training_panel.training_panel.config import PanelPaths
from tools.training_panel.training_panel.history import HistoryStore
from tools.training_panel.training_panel.terrain import (
    TerrainPresetStore,
    apply_terrain_overrides,
    read_terrain_values_from_yaml,
    terrain_defaults,
    terrain_diff,
    terrain_file_index,
)


CFG_SOURCE = """\
REDRHEX_ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(6.0, 6.0),
    border_width=3.0,
    num_rows=6,
    num_cols=12,
    curriculum=True,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 0.15),
    use_cache=False,
    sub_terrains={
        "flat": mesh.MeshPlaneTerrainCfg(proportion=0.20),
        "random_rough": hf.HfRandomUniformTerrainCfg(
            proportion=0.25,
            noise_range=(0.01, 0.06),
            noise_step=0.005,
            border_width=0.25,
        ),
        "wave": hf.HfWaveTerrainCfg(
            proportion=0.15,
            amplitude_range=(0.01, 0.06),
            num_waves=2,
            border_width=0.25,
        ),
        "stairs": hf.HfPyramidStairsTerrainCfg(
            proportion=0.20,
            step_height_range=(0.02, 0.12),
            step_width=0.28,
            platform_width=1.2,
            border_width=0.25,
        ),
        "boxes": mesh.MeshRandomGridTerrainCfg(
            proportion=0.20,
            grid_width=0.45,
            grid_height_range=(0.02, 0.12),
            platform_width=1.5,
        ),
    },
)

class RedrhexEnvCfg:
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=REDRHEX_ROUGH_TERRAINS_CFG,
        collision_group=-1,
        max_init_terrain_level=1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.2,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )
    terrain_curriculum_enable = True
    terrain_curriculum_levels = [0.0, 0.08, 0.20, 0.35, 0.55]
"""


def make_repo(tmp_path: Path) -> Path:
    cfg_dir = tmp_path / "source" / "RedRhex" / "RedRhex" / "tasks" / "direct" / "redrhex"
    cfg_dir.mkdir(parents=True)
    (cfg_dir / "redrhex_env_cfg.py").write_text(CFG_SOURCE, encoding="utf-8")
    (cfg_dir / "redrhex_env.py").write_text("# runtime terrain curriculum\n", encoding="utf-8")
    return tmp_path


def make_paths(root: Path) -> PanelPaths:
    return PanelPaths(
        repo_root=root,
        isaaclab_root=Path("/nonexistent"),
        isaacsim_root=Path("/nonexistent"),
        conda_sh=Path("/nonexistent"),
        conda_env="none",
    )


def test_terrain_defaults_parse_current_cfg_with_ast(tmp_path):
    repo = make_repo(tmp_path)
    defaults = terrain_defaults(repo)
    assert defaults["terrain.terrain_type"] == "generator"
    assert defaults["terrain.terrain_generator.size"] == [6.0, 6.0]
    assert defaults["terrain.terrain_generator.sub_terrains.random_rough.noise_range"] == [0.01, 0.06]
    assert defaults["terrain.physics_material.static_friction"] == 1.2
    assert defaults["terrain_curriculum_levels"] == [0.0, 0.08, 0.20, 0.35, 0.55]
    assert defaults["terrain.terrain_generator.border_height"] == 1.0


def test_terrain_file_index_reports_schema_and_files(tmp_path):
    repo = make_repo(tmp_path)
    index = terrain_file_index(repo)
    keys = {item["key"] for item in index["field_schema"]}
    assert "terrain.terrain_generator.sub_terrains.stairs.step_width" in keys
    assert index["mode"] == "preset-overrides"
    assert index["files"][0]["exists"] is True


def test_stairs_boxes_preset_grid_width_leaves_random_grid_border(tmp_path):
    repo = make_repo(tmp_path)
    preset = TerrainPresetStore(repo / "terrain_presets.json").get_preset("stairs-boxes")
    grid_width = preset["values"]["terrain.terrain_generator.sub_terrains.boxes.grid_width"]
    size = terrain_defaults(repo)["terrain.terrain_generator.size"]

    border_width = float(size[0]) - int(float(size[0]) / grid_width) * grid_width

    assert border_width > 0


def test_reads_terrain_values_from_yaml_and_diffs(tmp_path):
    repo = make_repo(tmp_path)
    env_yaml = tmp_path / "env.yaml"
    env_yaml.write_text(
        """\
terrain:
  terrain_type: plane
  max_init_terrain_level: 0
  terrain_generator:
    difficulty_range:
      - 0.0
      - 0.25
    sub_terrains:
      stairs:
        step_width: 0.32
terrain_curriculum_enable: false
terrain_curriculum_levels: [0.0]
""",
        encoding="utf-8",
    )
    parsed = read_terrain_values_from_yaml(env_yaml)
    assert parsed["terrain.terrain_type"] == "plane"
    assert parsed["terrain.terrain_generator.difficulty_range"] == [0.0, 0.25]
    assert parsed["terrain_curriculum_enable"] is False
    diff = terrain_diff(parsed, terrain_defaults(repo))
    changed_names = {item["name"] for item in diff["changed"]}
    assert "terrain.terrain_type" in changed_names
    assert "terrain.terrain_generator.sub_terrains.stairs.step_width" in changed_names


def test_reads_terrain_yaml_nulls_and_python_tuple_tags(tmp_path):
    env_yaml = tmp_path / "env.yaml"
    env_yaml.write_text(
        """\
terrain:
  terrain_type: plane
  max_init_terrain_level: null
  terrain_generator:
    difficulty_range: !!python/tuple
    - 0.0
    - 0.55
    sub_terrains:
      boxes:
        grid_height_range: !!python/tuple
        - 0.02
        - 0.12
""",
        encoding="utf-8",
    )
    parsed = read_terrain_values_from_yaml(env_yaml)
    assert parsed["terrain.terrain_type"] == "plane"
    assert "terrain.max_init_terrain_level" not in parsed
    assert parsed["terrain.terrain_generator.difficulty_range"] == [0.0, 0.55]
    assert parsed["terrain.terrain_generator.sub_terrains.boxes.grid_height_range"] == [0.02, 0.12]


def test_history_returns_terrain_config_for_run(tmp_path):
    repo = make_repo(tmp_path)
    paths = make_paths(repo)
    store = HistoryStore(paths)
    log_dir = repo / "logs" / "rsl_rl" / "redrhex_wheg" / "terrain-run"
    params_dir = log_dir / "params"
    params_dir.mkdir(parents=True)
    (params_dir / "env.yaml").write_text(
        "terrain:\n  terrain_type: plane\nterrain_curriculum_enable: false\n",
        encoding="utf-8",
    )
    store.add_run(
        {
            "id": "terrain-run",
            "source": "training_panel",
            "status": "completed",
            "log_dir": str(log_dir),
            "terrain_preset_id": "flat-debug",
            "created_at": "2026-05-15T12:00:00",
        }
    )
    config = store.get_terrain_config_for_run("terrain-run")
    assert config is not None
    assert config["preset_id"] == "flat-debug"
    assert any(item["name"] == "terrain.terrain_type" for item in config["changed"])


def test_terrain_preset_store_crud_and_active_persistence(tmp_path):
    preset_file = tmp_path / "terrain_presets.json"
    store = TerrainPresetStore(preset_file)
    assert store.get_preset("baseline")["built_in"] is True
    preset = store.create_preset("My Terrain", "desc", {"terrain.terrain_type": "plane"})
    store.set_active_preset(preset["id"])
    assert store.get_active_preset_id() == preset["id"]
    store2 = TerrainPresetStore(preset_file)
    assert store2.get_active_preset_id() == preset["id"]
    updated = store2.update_preset(
        preset["id"],
        name="Renamed Terrain",
        description="updated",
        values={"terrain.max_init_terrain_level": 0},
    )
    assert updated["name"] == "Renamed Terrain"
    assert updated["description"] == "updated"
    assert updated["values"]["terrain.max_init_terrain_level"] == 0
    with pytest.raises(ValueError):
        store2.delete_preset("baseline")


def test_apply_terrain_overrides_to_fake_env_cfg():
    env_cfg = SimpleNamespace(
        terrain=SimpleNamespace(
            terrain_type="generator",
            max_init_terrain_level=1,
            terrain_generator=SimpleNamespace(
                difficulty_range=(0.0, 0.15),
                sub_terrains={
                    "stairs": SimpleNamespace(step_width=0.28),
                },
            ),
        ),
        terrain_curriculum_enable=True,
    )
    applied = apply_terrain_overrides(
        env_cfg,
        {
            "terrain.terrain_type": "plane",
            "terrain.terrain_generator.difficulty_range": [0.0, 0.3],
            "terrain.terrain_generator.sub_terrains.stairs.step_width": 0.35,
            "terrain_curriculum_enable": False,
        },
    )
    assert "terrain.terrain_type=plane" in applied
    assert env_cfg.terrain.terrain_type == "plane"
    assert env_cfg.terrain.terrain_generator.difficulty_range == (0.0, 0.3)
    assert env_cfg.terrain.terrain_generator.sub_terrains["stairs"].step_width == 0.35
    assert env_cfg.terrain_curriculum_enable is False


def test_apply_terrain_overrides_adjusts_exact_box_grid_divisor():
    env_cfg = SimpleNamespace(
        terrain=SimpleNamespace(
            terrain_type="generator",
            terrain_generator=SimpleNamespace(
                size=(6.0, 6.0),
                sub_terrains={
                    "boxes": SimpleNamespace(grid_width=0.45),
                },
            ),
        ),
    )
    applied = apply_terrain_overrides(
        env_cfg,
        {
            "terrain.terrain_generator.sub_terrains.boxes.grid_width": 0.4,
        },
    )
    grid_width = env_cfg.terrain.terrain_generator.sub_terrains["boxes"].grid_width
    border_width = 6.0 - int(6.0 / grid_width) * grid_width

    assert grid_width == 0.396
    assert border_width > 0
    assert applied == [
        "terrain.terrain_generator.sub_terrains.boxes.grid_width=0.396 (adjusted from 0.4)",
    ]
