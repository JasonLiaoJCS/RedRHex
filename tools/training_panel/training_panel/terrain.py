from __future__ import annotations

import ast
import json
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any


TERRAIN_CFG_RELATIVE_PATH = "source/RedRhex/RedRhex/tasks/direct/redrhex/redrhex_env_cfg.py"
_DIFF_TOLERANCE = 1e-9
_SLUG_RE = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class TerrainField:
    key: str
    label: str
    category: str
    value_type: str
    description: str
    step: float | None = None
    choices: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "category": self.category,
            "type": self.value_type,
            "description": self.description,
            "step": self.step,
            "choices": list(self.choices),
        }


FIELD_SCHEMA: tuple[TerrainField, ...] = (
    TerrainField("terrain.terrain_type", "Terrain Type", "Importer", "choice", "Isaac terrain source.", choices=("generator", "plane", "usd")),
    TerrainField("terrain.prim_path", "Prim Path", "Importer", "string", "USD prim path used for the ground."),
    TerrainField("terrain.collision_group", "Collision Group", "Importer", "int", "Collision group assigned to terrain."),
    TerrainField("terrain.max_init_terrain_level", "Max Init Level", "Importer", "int", "Highest terrain row level available at reset."),
    TerrainField("terrain.debug_vis", "Debug Origins", "Importer", "bool", "Show terrain origin debug visualization."),
    TerrainField("terrain.physics_material.friction_combine_mode", "Friction Combine", "Physics Material", "choice", "How terrain friction combines with robot materials.", choices=("average", "min", "multiply", "max")),
    TerrainField("terrain.physics_material.restitution_combine_mode", "Restitution Combine", "Physics Material", "choice", "How bounce combines with robot materials.", choices=("average", "min", "multiply", "max")),
    TerrainField("terrain.physics_material.static_friction", "Static Friction", "Physics Material", "float", "Static friction coefficient.", 0.01),
    TerrainField("terrain.physics_material.dynamic_friction", "Dynamic Friction", "Physics Material", "float", "Dynamic friction coefficient.", 0.01),
    TerrainField("terrain_curriculum_enable", "Terrain Curriculum", "Curriculum", "bool", "Enable stage-based terrain difficulty updates."),
    TerrainField("terrain_curriculum_levels", "Curriculum Levels", "Curriculum", "list", "Difficulty level per curriculum stage.", 0.01),
    TerrainField("terrain.terrain_generator.size", "Tile Size", "Generator", "range", "Sub-terrain tile width and length in meters.", 0.1),
    TerrainField("terrain.terrain_generator.border_width", "Border Width", "Generator", "float", "Outer terrain border width in meters.", 0.1),
    TerrainField("terrain.terrain_generator.border_height", "Border Height", "Generator", "float", "Outer terrain border height in meters.", 0.1),
    TerrainField("terrain.terrain_generator.num_rows", "Rows", "Generator", "int", "Terrain difficulty rows.", 1),
    TerrainField("terrain.terrain_generator.num_cols", "Columns", "Generator", "int", "Terrain variation columns.", 1),
    TerrainField("terrain.terrain_generator.curriculum", "Generator Curriculum", "Generator", "bool", "Generate terrain rows in curriculum order."),
    TerrainField("terrain.terrain_generator.color_scheme", "Color Scheme", "Generator", "choice", "Terrain visual color scheme.", choices=("height", "random", "none")),
    TerrainField("terrain.terrain_generator.horizontal_scale", "Horizontal Scale", "Generator", "float", "Height-field XY discretization.", 0.001),
    TerrainField("terrain.terrain_generator.vertical_scale", "Vertical Scale", "Generator", "float", "Height-field Z discretization.", 0.001),
    TerrainField("terrain.terrain_generator.slope_threshold", "Slope Threshold", "Generator", "float", "Height-field slope correction threshold.", 0.01),
    TerrainField("terrain.terrain_generator.difficulty_range", "Difficulty Range", "Generator", "range", "Generated terrain difficulty min and max.", 0.01),
    TerrainField("terrain.terrain_generator.use_cache", "Use Cache", "Generator", "bool", "Reuse generated terrain cache when available."),
    TerrainField("terrain.terrain_generator.cache_dir", "Cache Dir", "Generator", "string", "Terrain cache directory."),
    TerrainField("terrain.terrain_generator.sub_terrains.flat.proportion", "Flat Proportion", "Flat", "float", "Sampling weight for flat terrain.", 0.01),
    TerrainField("terrain.terrain_generator.sub_terrains.random_rough.proportion", "Rough Proportion", "Random Rough", "float", "Sampling weight for random rough terrain.", 0.01),
    TerrainField("terrain.terrain_generator.sub_terrains.random_rough.noise_range", "Noise Range", "Random Rough", "range", "Min and max rough terrain height noise in meters.", 0.001),
    TerrainField("terrain.terrain_generator.sub_terrains.random_rough.noise_step", "Noise Step", "Random Rough", "float", "Minimum height change between samples.", 0.001),
    TerrainField("terrain.terrain_generator.sub_terrains.random_rough.border_width", "Border Width", "Random Rough", "float", "Flat border around rough tile.", 0.01),
    TerrainField("terrain.terrain_generator.sub_terrains.wave.proportion", "Wave Proportion", "Wave", "float", "Sampling weight for wave terrain.", 0.01),
    TerrainField("terrain.terrain_generator.sub_terrains.wave.amplitude_range", "Amplitude Range", "Wave", "range", "Min and max wave amplitude in meters.", 0.001),
    TerrainField("terrain.terrain_generator.sub_terrains.wave.num_waves", "Wave Count", "Wave", "int", "Number of waves per terrain tile.", 1),
    TerrainField("terrain.terrain_generator.sub_terrains.wave.border_width", "Border Width", "Wave", "float", "Flat border around wave tile.", 0.01),
    TerrainField("terrain.terrain_generator.sub_terrains.stairs.proportion", "Stairs Proportion", "Stairs", "float", "Sampling weight for pyramid stairs terrain.", 0.01),
    TerrainField("terrain.terrain_generator.sub_terrains.stairs.step_height_range", "Step Height Range", "Stairs", "range", "Min and max stair height in meters.", 0.001),
    TerrainField("terrain.terrain_generator.sub_terrains.stairs.step_width", "Step Width", "Stairs", "float", "Stair tread width in meters.", 0.01),
    TerrainField("terrain.terrain_generator.sub_terrains.stairs.platform_width", "Platform Width", "Stairs", "float", "Central platform width in meters.", 0.1),
    TerrainField("terrain.terrain_generator.sub_terrains.stairs.border_width", "Border Width", "Stairs", "float", "Flat border around stairs tile.", 0.01),
    TerrainField("terrain.terrain_generator.sub_terrains.boxes.proportion", "Boxes Proportion", "Boxes", "float", "Sampling weight for random box grid terrain.", 0.01),
    TerrainField("terrain.terrain_generator.sub_terrains.boxes.grid_width", "Grid Width", "Boxes", "float", "Random grid cell width in meters.", 0.01),
    TerrainField("terrain.terrain_generator.sub_terrains.boxes.grid_height_range", "Grid Height Range", "Boxes", "range", "Min and max random grid height in meters.", 0.001),
    TerrainField("terrain.terrain_generator.sub_terrains.boxes.platform_width", "Platform Width", "Boxes", "float", "Central platform width in meters.", 0.1),
)

FIELD_MAP = {field.key: field for field in FIELD_SCHEMA}
FIELD_ORDER = [field.key for field in FIELD_SCHEMA]
IMPLICIT_DEFAULTS = {
    "terrain.terrain_generator.border_height": 1.0,
    "terrain.terrain_generator.color_scheme": "none",
    "terrain.terrain_generator.cache_dir": "/tmp/isaaclab/terrains",
}
TUPLE_FIELDS = {
    "terrain.terrain_generator.size",
    "terrain.terrain_generator.difficulty_range",
    "terrain.terrain_generator.sub_terrains.random_rough.noise_range",
    "terrain.terrain_generator.sub_terrains.wave.amplitude_range",
    "terrain.terrain_generator.sub_terrains.stairs.step_height_range",
    "terrain.terrain_generator.sub_terrains.boxes.grid_height_range",
}


BUILT_IN_TERRAIN_PRESETS: list[dict[str, Any]] = [
    {
        "id": "baseline",
        "name": "Baseline",
        "description": "The current terrain defaults from redrhex_env_cfg.py.",
        "built_in": True,
        "values": {},
    },
    {
        "id": "flat-debug",
        "name": "Flat Debug",
        "description": "For quick debugging on a plane with terrain curriculum disabled.",
        "built_in": True,
        "values": {
            "terrain.terrain_type": "plane",
            "terrain.max_init_terrain_level": 0,
            "terrain_curriculum_enable": False,
            "terrain_curriculum_levels": [0.0],
        },
    },
    {
        "id": "mild-mixed",
        "name": "Mild Mixed",
        "description": "A gentle rough/wave/stairs/boxes mix for early terrain training.",
        "built_in": True,
        "values": {
            "terrain.terrain_type": "generator",
            "terrain.max_init_terrain_level": 1,
            "terrain.terrain_generator.difficulty_range": [0.0, 0.10],
            "terrain_curriculum_enable": True,
            "terrain_curriculum_levels": [0.0, 0.05, 0.10, 0.16, 0.24],
            "terrain.terrain_generator.sub_terrains.random_rough.noise_range": [0.005, 0.035],
            "terrain.terrain_generator.sub_terrains.wave.amplitude_range": [0.005, 0.035],
            "terrain.terrain_generator.sub_terrains.stairs.step_height_range": [0.01, 0.07],
            "terrain.terrain_generator.sub_terrains.boxes.grid_height_range": [0.01, 0.07],
        },
    },
    {
        "id": "rough-mixed",
        "name": "Rough Mixed",
        "description": "A stronger mixed-terrain profile for robustness work.",
        "built_in": True,
        "values": {
            "terrain.terrain_type": "generator",
            "terrain.max_init_terrain_level": 2,
            "terrain.terrain_generator.difficulty_range": [0.0, 0.30],
            "terrain_curriculum_enable": True,
            "terrain_curriculum_levels": [0.0, 0.12, 0.28, 0.45, 0.70],
            "terrain.terrain_generator.sub_terrains.flat.proportion": 0.10,
            "terrain.terrain_generator.sub_terrains.random_rough.proportion": 0.30,
            "terrain.terrain_generator.sub_terrains.wave.proportion": 0.20,
            "terrain.terrain_generator.sub_terrains.stairs.proportion": 0.20,
            "terrain.terrain_generator.sub_terrains.boxes.proportion": 0.20,
            "terrain.terrain_generator.sub_terrains.random_rough.noise_range": [0.02, 0.08],
            "terrain.terrain_generator.sub_terrains.wave.amplitude_range": [0.02, 0.08],
            "terrain.terrain_generator.sub_terrains.stairs.step_height_range": [0.03, 0.15],
            "terrain.terrain_generator.sub_terrains.boxes.grid_height_range": [0.03, 0.15],
        },
    },
    {
        "id": "stairs-boxes",
        "name": "Stairs + Boxes",
        "description": "Focused obstacle profile for step and box-grid adaptation.",
        "built_in": True,
        "values": {
            "terrain.terrain_type": "generator",
            "terrain.max_init_terrain_level": 2,
            "terrain.terrain_generator.difficulty_range": [0.0, 0.25],
            "terrain_curriculum_enable": True,
            "terrain_curriculum_levels": [0.0, 0.10, 0.22, 0.38, 0.55],
            "terrain.terrain_generator.sub_terrains.flat.proportion": 0.10,
            "terrain.terrain_generator.sub_terrains.random_rough.proportion": 0.10,
            "terrain.terrain_generator.sub_terrains.wave.proportion": 0.05,
            "terrain.terrain_generator.sub_terrains.stairs.proportion": 0.40,
            "terrain.terrain_generator.sub_terrains.boxes.proportion": 0.35,
            "terrain.terrain_generator.sub_terrains.stairs.step_height_range": [0.02, 0.14],
            "terrain.terrain_generator.sub_terrains.stairs.step_width": 0.25,
            "terrain.terrain_generator.sub_terrains.boxes.grid_width": 0.40,
            "terrain.terrain_generator.sub_terrains.boxes.grid_height_range": [0.02, 0.14],
        },
    },
]


def _slugify(name: str) -> str:
    return _SLUG_RE.sub("-", name.lower().strip()).strip("-") or "terrain-preset"


def _node_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _node_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return ""


def _node_value(node: ast.AST, constants: dict[str, Any]) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        value = _node_value(node.operand, constants)
        return -value if isinstance(value, (int, float)) else value
    if isinstance(node, (ast.Tuple, ast.List)):
        return [_node_value(item, constants) for item in node.elts]
    if isinstance(node, ast.Dict):
        return {
            _node_value(key, constants): _node_value(value, constants)
            for key, value in zip(node.keys, node.values)
            if key is not None
        }
    if isinstance(node, ast.Name):
        return constants.get(node.id, node.id)
    if isinstance(node, ast.Call):
        data: dict[str, Any] = {"__type": _node_name(node.func)}
        for keyword in node.keywords:
            if keyword.arg is not None:
                data[keyword.arg] = _node_value(keyword.value, constants)
        return data
    if isinstance(node, ast.Attribute):
        return _node_name(node)
    return None


def _class_assignments(tree: ast.Module, class_name: str, constants: dict[str, Any]) -> dict[str, Any]:
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        values: dict[str, Any] = {}
        for item in node.body:
            if isinstance(item, ast.Assign) and len(item.targets) == 1 and isinstance(item.targets[0], ast.Name):
                values[item.targets[0].id] = _node_value(item.value, constants)
        return values
    return {}


def _flatten(data: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(data, dict):
        flat: dict[str, Any] = {}
        for key, value in data.items():
            if key == "__type":
                continue
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            flat.update(_flatten(value, child_prefix))
        return flat
    return {prefix: data}


def _coerce_field_value(key: str, value: Any) -> Any:
    field = FIELD_MAP.get(key)
    if field is None:
        return _normalize_value(value)
    if value is None:
        return None
    if field.value_type == "bool":
        if isinstance(value, str):
            if value.strip() == "":
                return None
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)
    if field.value_type == "int":
        if isinstance(value, str) and value.strip() == "":
            return None
        return int(value)
    if field.value_type == "float":
        if isinstance(value, str) and value.strip() == "":
            return None
        return float(value)
    if field.value_type in {"range", "list"}:
        if isinstance(value, str):
            if value.strip() == "":
                return []
            parsed = _parse_scalar(value)
            if parsed is None:
                return None
            if isinstance(parsed, list):
                return [float(item) if isinstance(item, (int, float, str)) and str(item).strip() else item for item in parsed]
            return parsed
        if isinstance(value, tuple):
            value = list(value)
        if isinstance(value, list):
            return [_normalize_value(item) for item in value]
    if field.value_type in {"choice", "string"}:
        return str(value)
    return _normalize_value(value)


def _normalize_value(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_normalize_value(item) for item in value]
    if isinstance(value, list):
        return [_normalize_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _normalize_value(item) for key, item in value.items()}
    return value


def _normalize_values(values: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in values.items():
        if key in FIELD_MAP:
            normalized[key] = _coerce_field_value(key, value)
    return normalized


def terrain_defaults(repo_root: Path) -> dict[str, Any]:
    cfg_path = repo_root / TERRAIN_CFG_RELATIVE_PATH
    if not cfg_path.exists():
        return {}
    tree = ast.parse(cfg_path.read_text(encoding="utf-8"), filename=str(cfg_path))
    constants: dict[str, Any] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            constants[node.targets[0].id] = _node_value(node.value, constants)
    assignments = _class_assignments(tree, "RedrhexEnvCfg", constants)
    terrain_values = _flatten(assignments.get("terrain", {}), "terrain")
    terrain_values["terrain_curriculum_enable"] = assignments.get("terrain_curriculum_enable")
    terrain_values["terrain_curriculum_levels"] = assignments.get("terrain_curriculum_levels")
    for key, value in IMPLICIT_DEFAULTS.items():
        terrain_values.setdefault(key, value)
    return {
        key: _coerce_field_value(key, value)
        for key, value in terrain_values.items()
        if key in FIELD_MAP and value is not None
    }


def terrain_file_index(repo_root: Path) -> dict[str, Any]:
    cfg_path = repo_root / TERRAIN_CFG_RELATIVE_PATH
    defaults = terrain_defaults(repo_root)
    return {
        "files": [
            {
                "title": "Terrain defaults and generator presets",
                "path": TERRAIN_CFG_RELATIVE_PATH,
                "absolute_path": str(cfg_path),
                "exists": cfg_path.exists(),
                "why": "Defines RedRHex terrain importer, rough terrain generator, sub-terrain mix, and terrain curriculum defaults.",
            },
            {
                "title": "Terrain curriculum runtime logic",
                "path": "source/RedRhex/RedRhex/tasks/direct/redrhex/redrhex_env.py",
                "absolute_path": str(repo_root / "source/RedRhex/RedRhex/tasks/direct/redrhex/redrhex_env.py"),
                "exists": (repo_root / "source/RedRhex/RedRhex/tasks/direct/redrhex/redrhex_env.py").exists(),
                "why": "Shows how stage-based terrain levels are resolved and applied during training.",
            },
        ],
        "field_schema": [field.to_dict() for field in FIELD_SCHEMA],
        "terrain_defaults": defaults,
        "terrain_values": [
            {
                "key": key,
                "value": defaults[key],
                "relative_path": TERRAIN_CFG_RELATIVE_PATH,
            }
            for key in FIELD_ORDER
            if key in defaults
        ],
        "mode": "preset-overrides",
    }


def _parse_scalar(raw: str) -> Any:
    text = raw.strip()
    if text == "":
        return ""
    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none", "~"}:
        return None
    if (text.startswith("[") and text.endswith("]")) or (text.startswith("(") and text.endswith(")")):
        try:
            parsed = ast.literal_eval(text)
            return list(parsed) if isinstance(parsed, tuple) else parsed
        except (ValueError, SyntaxError):
            return text
    try:
        if any(char in text for char in ".eE"):
            return float(text)
        return int(text)
    except ValueError:
        return text.strip("\"'")


def read_terrain_values_from_yaml(env_yaml_path: Path) -> dict[str, Any]:
    """Parse selected terrain values from params/env.yaml without requiring PyYAML."""
    if not env_yaml_path.exists():
        return {}
    values: dict[str, Any] = {}
    stack: list[tuple[int, str]] = []
    list_key: str | None = None
    for raw_line in env_yaml_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        line = raw_line.strip()
        if line.startswith("- ") and list_key:
            values.setdefault(list_key, []).append(_parse_scalar(line[2:]))
            continue
        while stack and stack[-1][0] >= indent:
            stack.pop()
        list_key = None
        if ":" not in line:
            continue
        key, raw_value = line.split(":", 1)
        key = key.strip()
        path = ".".join([part for _, part in stack] + [key])
        raw_value = raw_value.strip()
        if raw_value == "":
            stack.append((indent, key))
            if path in FIELD_MAP and FIELD_MAP[path].value_type in {"range", "list"}:
                values[path] = []
                list_key = path
            continue
        if (
            path in FIELD_MAP
            and FIELD_MAP[path].value_type in {"range", "list"}
            and raw_value.startswith("!!")
        ):
            values[path] = []
            list_key = path
            continue
        if path in FIELD_MAP:
            parsed_value = _coerce_field_value(path, _parse_scalar(raw_value))
            if parsed_value is not None:
                values[path] = parsed_value
    return values


def _values_equal(left: Any, right: Any) -> bool:
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return abs(float(left) - float(right)) <= _DIFF_TOLERANCE
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(_values_equal(a, b) for a, b in zip(left, right))
    return left == right


def terrain_diff(yaml_values: dict[str, Any], defaults: dict[str, Any]) -> dict[str, list[Any]]:
    changed = []
    same = []
    for key in FIELD_ORDER:
        if key not in defaults or key not in yaml_values:
            continue
        yaml_value = yaml_values[key]
        default_value = defaults[key]
        if _values_equal(yaml_value, default_value):
            same.append(key)
            continue
        delta_pct = None
        if isinstance(yaml_value, (int, float)) and isinstance(default_value, (int, float)):
            delta_pct = round((float(yaml_value) - float(default_value)) / (abs(float(default_value)) + 1e-12) * 100, 1)
        changed.append({"name": key, "yaml_value": yaml_value, "default_value": default_value, "delta_pct": delta_pct})
    for key, yaml_value in yaml_values.items():
        if key not in defaults:
            changed.append({"name": key, "yaml_value": yaml_value, "default_value": None, "delta_pct": None})
    return {"changed": changed, "same": same}


def apply_terrain_overrides(env_cfg: Any, overrides: dict[str, Any]) -> list[str]:
    """Apply flattened terrain override keys to an Isaac env config object."""
    applied: list[str] = []
    for key, raw_value in _normalize_values(overrides).items():
        target = env_cfg
        parts = key.split(".")
        for part in parts[:-1]:
            if isinstance(target, dict):
                target = target.get(part)
            else:
                target = getattr(target, part, None)
            if target is None:
                break
        if target is None:
            continue
        value = raw_value
        if key in TUPLE_FIELDS and isinstance(value, list):
            value = tuple(value)
        if isinstance(target, dict):
            target[parts[-1]] = value
        elif hasattr(target, parts[-1]):
            setattr(target, parts[-1], value)
        else:
            continue
        applied.append(f"{key}={raw_value}")
    return applied


class TerrainPresetStore:
    def __init__(self, preset_file: Path) -> None:
        self._file = preset_file
        self._lock = threading.Lock()
        self._data: dict[str, Any] = self._load()

    def _load(self) -> dict[str, Any]:
        if self._file.exists():
            try:
                data = json.loads(self._file.read_text(encoding="utf-8"))
                existing_ids = {preset["id"] for preset in data.get("presets", [])}
                for built_in in BUILT_IN_TERRAIN_PRESETS:
                    if built_in["id"] not in existing_ids:
                        data.setdefault("presets", []).insert(BUILT_IN_TERRAIN_PRESETS.index(built_in), dict(built_in))
                    else:
                        for index, preset in enumerate(data.get("presets", [])):
                            if preset["id"] == built_in["id"]:
                                data["presets"][index] = dict(built_in)
                data.setdefault("active_preset_id", "baseline")
                return data
            except (json.JSONDecodeError, KeyError, TypeError):
                pass
        return {
            "active_preset_id": "baseline",
            "presets": [dict(preset) for preset in BUILT_IN_TERRAIN_PRESETS],
        }

    def _save(self) -> None:
        self._file.parent.mkdir(parents=True, exist_ok=True)
        self._file.write_text(json.dumps(self._data, indent=2, ensure_ascii=False), encoding="utf-8")

    def list_presets(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(preset) for preset in self._data.get("presets", [])]

    def get_preset(self, preset_id: str) -> dict[str, Any] | None:
        with self._lock:
            for preset in self._data.get("presets", []):
                if preset["id"] == preset_id:
                    return dict(preset)
        return None

    def create_preset(self, name: str, description: str, values: dict[str, Any]) -> dict[str, Any]:
        slug = _slugify(name)
        with self._lock:
            existing_ids = {preset["id"] for preset in self._data.get("presets", [])}
            base, counter = slug, 2
            while slug in existing_ids:
                slug = f"{base}-{counter}"
                counter += 1
            preset = {
                "id": slug,
                "name": name.strip(),
                "description": description.strip(),
                "built_in": False,
                "values": _normalize_values(values),
            }
            self._data.setdefault("presets", []).append(preset)
            self._save()
            return dict(preset)

    def update_preset(self, preset_id: str, **updates: Any) -> dict[str, Any]:
        with self._lock:
            for index, preset in enumerate(self._data.get("presets", [])):
                if preset["id"] != preset_id:
                    continue
                if preset.get("built_in"):
                    raise ValueError(f"Cannot modify built-in terrain preset '{preset_id}'")
                if "name" in updates:
                    preset["name"] = str(updates["name"]).strip()
                if "description" in updates:
                    preset["description"] = str(updates["description"]).strip()
                if "values" in updates:
                    preset["values"] = _normalize_values(dict(updates["values"]))
                self._data["presets"][index] = preset
                self._save()
                return dict(preset)
        raise KeyError(f"Terrain preset '{preset_id}' not found")

    def delete_preset(self, preset_id: str) -> bool:
        with self._lock:
            for preset in self._data.get("presets", []):
                if preset["id"] == preset_id:
                    if preset.get("built_in"):
                        raise ValueError(f"Cannot delete built-in terrain preset '{preset_id}'")
                    self._data["presets"] = [item for item in self._data["presets"] if item["id"] != preset_id]
                    if self._data.get("active_preset_id") == preset_id:
                        self._data["active_preset_id"] = "baseline"
                    self._save()
                    return True
        return False

    def get_active_preset_id(self) -> str:
        with self._lock:
            return self._data.get("active_preset_id") or "baseline"

    def set_active_preset(self, preset_id: str) -> None:
        with self._lock:
            ids = {preset["id"] for preset in self._data.get("presets", [])}
            if preset_id not in ids:
                raise KeyError(f"Terrain preset '{preset_id}' not found")
            self._data["active_preset_id"] = preset_id
            self._save()
