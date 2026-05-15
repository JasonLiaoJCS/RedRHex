from __future__ import annotations

import json
import re
import threading
from pathlib import Path


_SLUG_RE = re.compile(r"[^a-z0-9]+")

BUILT_IN_PRESETS: list[dict] = [
    {
        "id": "baseline",
        "name": "Baseline",
        "description": (
            "The current default reward configuration. "
            "Good starting point — balanced between speed and stability."
        ),
        "built_in": True,
        "values": {},
    },
    {
        "id": "speed-focus",
        "name": "Speed Focus",
        "description": (
            "Emphasises forward velocity and tracking. "
            "Robot learns to move faster but may be less stable. "
            "Try this after a baseline run succeeds."
        ),
        "built_in": True,
        "values": {
            "rew_scale_forward_vel": 5.0,
            "rew_scale_vel_tracking": 6.0,
            "rew_scale_ang_vel_tracking": 3.5,
            "rew_scale_orientation": -0.1,
            "rew_scale_base_height": -0.1,
        },
    },
    {
        "id": "stability-focus",
        "name": "Stability Focus",
        "description": (
            "Strongly penalises tilting and height deviation. "
            "Robot learns to stay upright and balanced first. "
            "Good for early-stage training when the robot keeps falling."
        ),
        "built_in": True,
        "values": {
            "rew_scale_forward_vel": 1.5,
            "rew_scale_vel_tracking": 2.0,
            "rew_scale_orientation": -0.6,
            "rew_scale_base_height": -0.6,
            "rew_scale_lin_vel_z": -0.3,
            "rew_scale_alive": 0.3,
        },
    },
]


def _slugify(name: str) -> str:
    return _SLUG_RE.sub("-", name.lower().strip()).strip("-")


class PresetStore:
    def __init__(self, preset_file: Path) -> None:
        self._file = preset_file
        self._lock = threading.Lock()
        self._data: dict = self._load()

    def _load(self) -> dict:
        if self._file.exists():
            try:
                data = json.loads(self._file.read_text(encoding="utf-8"))
                # Merge built-ins (they may have been updated)
                existing_ids = {p["id"] for p in data.get("presets", [])}
                for bp in BUILT_IN_PRESETS:
                    if bp["id"] not in existing_ids:
                        data.setdefault("presets", []).insert(
                            BUILT_IN_PRESETS.index(bp), dict(bp)
                        )
                    else:
                        for i, p in enumerate(data["presets"]):
                            if p["id"] == bp["id"]:
                                data["presets"][i] = dict(bp)
                return data
            except (json.JSONDecodeError, KeyError):
                pass
        return {
            "active_preset_id": "baseline",
            "presets": [dict(bp) for bp in BUILT_IN_PRESETS],
        }

    def _save(self) -> None:
        self._file.parent.mkdir(parents=True, exist_ok=True)
        self._file.write_text(
            json.dumps(self._data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def list_presets(self) -> list[dict]:
        with self._lock:
            return [dict(p) for p in self._data.get("presets", [])]

    def get_preset(self, preset_id: str) -> dict | None:
        with self._lock:
            for p in self._data.get("presets", []):
                if p["id"] == preset_id:
                    return dict(p)
        return None

    def create_preset(self, name: str, description: str, values: dict[str, float]) -> dict:
        slug = _slugify(name)
        with self._lock:
            existing_ids = {p["id"] for p in self._data.get("presets", [])}
            # Ensure unique ID
            base, counter = slug, 2
            while slug in existing_ids:
                slug = f"{base}-{counter}"
                counter += 1
            preset = {
                "id": slug,
                "name": name.strip(),
                "description": description.strip(),
                "built_in": False,
                "values": {k: float(v) for k, v in values.items()},
            }
            self._data.setdefault("presets", []).append(preset)
            self._save()
            return dict(preset)

    def update_preset(self, preset_id: str, **updates: object) -> dict:
        with self._lock:
            for i, p in enumerate(self._data.get("presets", [])):
                if p["id"] != preset_id:
                    continue
                if p.get("built_in"):
                    raise ValueError(f"Cannot modify built-in preset '{preset_id}'")
                if "name" in updates:
                    p["name"] = str(updates["name"]).strip()
                if "description" in updates:
                    p["description"] = str(updates["description"]).strip()
                if "values" in updates:
                    p["values"] = {k: float(v) for k, v in updates["values"].items()}
                self._data["presets"][i] = p
                self._save()
                return dict(p)
        raise KeyError(f"Preset '{preset_id}' not found")

    def delete_preset(self, preset_id: str) -> bool:
        with self._lock:
            for p in self._data.get("presets", []):
                if p["id"] == preset_id:
                    if p.get("built_in"):
                        raise ValueError(f"Cannot delete built-in preset '{preset_id}'")
                    self._data["presets"] = [
                        x for x in self._data["presets"] if x["id"] != preset_id
                    ]
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
            ids = {p["id"] for p in self._data.get("presets", [])}
            if preset_id not in ids:
                raise KeyError(f"Preset '{preset_id}' not found")
            self._data["active_preset_id"] = preset_id
            self._save()

    def get_active_preset(self) -> dict:
        active_id = self.get_active_preset_id()
        preset = self.get_preset(active_id)
        return preset or self.get_preset("baseline") or BUILT_IN_PRESETS[0]
