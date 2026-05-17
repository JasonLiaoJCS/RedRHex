from __future__ import annotations

import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import PanelPaths
from .rewards import read_reward_scales_from_yaml, reward_defaults, reward_diff
from .terrain import read_terrain_values_from_yaml, terrain_defaults, terrain_diff

MODEL_RE = re.compile(r"model_(\d+)\.pt$")
DISCOVERED_RUN_TIME_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})")
PANEL_RUN_TIME_RE = re.compile(r"^panel_(\d{8})_(\d{6})")
ACTIVE_PANEL_LOG_CLAIM_GRACE_SECONDS = 5
ACTIVE_PANEL_LOG_CLAIM_WINDOW_SECONDS = 10 * 60


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def _safe_note_id(run_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", run_id)


def _is_within(path: Path, root: Path) -> bool:
    resolved = path.resolve()
    resolved_root = root.resolve()
    return resolved == resolved_root or resolved_root in resolved.parents


def _merged_params(*params: Any) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for value in params:
        if isinstance(value, dict):
            merged.update(value)
    return merged


def _has_value(value: Any) -> bool:
    return value not in (None, "", [], {})


def _parse_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.replace(tzinfo=None)


def _datetime_from_discovered_run_id(run_id: str) -> datetime | None:
    match = DISCOVERED_RUN_TIME_RE.match(str(run_id or ""))
    if not match:
        return None
    try:
        return datetime.strptime(f"{match.group(1)}_{match.group(2)}", "%Y-%m-%d_%H-%M-%S")
    except ValueError:
        return None


def _datetime_from_panel_run_id(run_id: str) -> datetime | None:
    match = PANEL_RUN_TIME_RE.match(str(run_id or ""))
    if not match:
        return None
    try:
        return datetime.strptime(f"{match.group(1)}_{match.group(2)}", "%Y%m%d_%H%M%S")
    except ValueError:
        return None


def tail_file(path: Path, max_chars: int = 50000) -> str:
    if not path.exists() or not path.is_file():
        return ""
    with path.open("rb") as file:
        file.seek(0, 2)
        size = file.tell()
        file.seek(max(0, size - max_chars))
        return file.read().decode("utf-8", errors="replace")


def latest_checkpoint(log_dir: Path) -> str | None:
    models = []
    for path in log_dir.glob("model_*.pt"):
        match = MODEL_RE.match(path.name)
        if match:
            models.append((int(match.group(1)), path))
    if not models:
        return None
    return str(max(models, key=lambda item: item[0])[1])


def latest_video(log_dir: Path) -> str | None:
    video_dir = log_dir / "videos" / "play"
    if not video_dir.exists():
        return None
    videos = [path for path in video_dir.glob("*.mp4") if path.is_file()]
    if not videos:
        return None
    return str(max(videos, key=lambda path: path.stat().st_mtime))


def latest_onnx(log_dir: Path) -> str | None:
    onnx = log_dir / "exported" / "policy.onnx"
    return str(onnx) if onnx.is_file() else None


def checkpoint_inventory(log_dir: Path) -> list[tuple[int, Path]]:
    checkpoints = []
    for path in log_dir.glob("model_*.pt"):
        match = MODEL_RE.match(path.name)
        if match and path.is_file():
            checkpoints.append((int(match.group(1)), path))
    return sorted(checkpoints, key=lambda item: item[0])


class HistoryStore:
    def __init__(self, paths: PanelPaths):
        self.paths = paths
        self.paths.ensure_dirs()

    def _load_data(self) -> dict:
        data = _read_json(self.paths.history_file, {"runs": [], "folders": [], "deleted_runs": []})
        if not isinstance(data, dict):
            data = {"runs": [], "folders": [], "deleted_runs": []}
        data.setdefault("runs", [])
        data.setdefault("folders", [])
        data.setdefault("deleted_runs", [])
        return data

    def _save_data(self, data: dict) -> None:
        self.paths.ensure_dirs()
        self.paths.history_file.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _load_records(self) -> list[dict]:
        data = self._load_data()
        return list(data.get("runs", []))

    def _save_records(self, records: list[dict]) -> None:
        data = self._load_data()
        data["runs"] = records
        self._save_data(data)

    def _collapse_duplicate_runs(self, records: list[dict]) -> list[dict]:
        """Collapse panel/discovered records that represent the same RSL-RL log dir."""
        groups: dict[str, list[dict]] = {}
        order: list[str] = []
        for record in records:
            log_dir = str(record.get("log_dir") or "").strip()
            key = f"log:{log_dir}" if log_dir else f"id:{record.get('id')}"
            if key not in groups:
                groups[key] = []
                order.append(key)
            groups[key].append(record)

        collapsed = []
        for key in order:
            group = groups[key]
            if len(group) == 1:
                collapsed.append(group[0])
                continue

            def rank(record: dict) -> tuple[int, str]:
                run_id = str(record.get("id") or "")
                score = 0
                if run_id.startswith("panel_"):
                    score += 4
                if record.get("source") == "training_panel":
                    score += 2
                if record.get("process_log"):
                    score += 1
                return score, str(record.get("created_at") or "")

            primary = max(group, key=rank).copy()
            merged_params = primary.get("params") if isinstance(primary.get("params"), dict) else {}
            for record in group:
                if record is primary:
                    continue
                for field, value in record.items():
                    if field == "params":
                        if isinstance(value, dict):
                            merged_params = _merged_params(value, merged_params)
                        continue
                    if field == "id":
                        continue
                    if not _has_value(primary.get(field)) and _has_value(value):
                        primary[field] = value
            if merged_params:
                primary["params"] = merged_params
            collapsed.append(primary)
        return collapsed

    def _process_log_sample(self, process_log: Any, max_chars: int = 160000) -> str:
        if not process_log:
            return ""
        path = Path(str(process_log))
        if not path.exists() or not path.is_file():
            return ""
        try:
            with path.open("rb") as file:
                head = file.read(max_chars // 2).decode("utf-8", errors="replace")
                file.seek(0, 2)
                size = file.tell()
                file.seek(max(0, size - max_chars // 2))
                tail = file.read().decode("utf-8", errors="replace")
        except OSError:
            return ""
        return f"{head}\n{tail}"

    def _active_panel_discovered_run(self, record: dict, discovered_runs: list[dict]) -> dict | None:
        """Find the discovered RSL-RL folder that an active panel run is already writing."""
        if record.get("source") != "training_panel":
            return None
        if record.get("log_dir"):
            return None
        if str(record.get("status") or "").lower() not in {"running", "stopping"}:
            return None

        candidates = []
        text = self._process_log_sample(record.get("process_log"))
        if text:
            explicit_dirs = set()
            root = re.escape(str(self.paths.rsl_rl_log_root))
            for match in re.findall(rf"({root}/[^\s'\"<>]+)", text):
                path = Path(match)
                while path.parent != self.paths.rsl_rl_log_root and path != self.paths.rsl_rl_log_root:
                    path = path.parent
                if path.parent == self.paths.rsl_rl_log_root:
                    explicit_dirs.add(str(path))

            exact_names = re.findall(r"Exact experiment name requested from command line:\s*(\S+)", text)
            for discovered in discovered_runs:
                log_dir = str(discovered.get("log_dir") or "")
                run_id = str(discovered.get("id") or "")
                if log_dir in explicit_dirs:
                    candidates.append(discovered)
                    continue
                if any(run_id == name or run_id.startswith(f"{name}_") for name in exact_names):
                    candidates.append(discovered)
        if not candidates:
            candidates = self._fresh_discovered_runs_for_active_panel(record, discovered_runs)
        if not candidates:
            return None
        return max(candidates, key=lambda run: str(run.get("created_at") or ""))

    def _fresh_discovered_runs_for_active_panel(self, record: dict, discovered_runs: list[dict]) -> list[dict]:
        panel_start = _parse_datetime(record.get("created_at")) or _datetime_from_panel_run_id(str(record.get("id") or ""))
        if not panel_start:
            return []
        candidates: list[tuple[float, dict]] = []
        for discovered in discovered_runs:
            run_time = _datetime_from_discovered_run_id(str(discovered.get("id") or ""))
            if not run_time:
                run_time = _parse_datetime(discovered.get("created_at"))
            if not run_time:
                continue
            delta = (run_time - panel_start).total_seconds()
            if -ACTIVE_PANEL_LOG_CLAIM_GRACE_SECONDS <= delta <= ACTIVE_PANEL_LOG_CLAIM_WINDOW_SECONDS:
                candidates.append((abs(delta), discovered))
        if not candidates:
            return []
        candidates.sort(key=lambda item: item[0])
        return [candidates[0][1]]

    def _log_dir_conflicts_with_process_log(self, record: dict) -> bool:
        if record.get("source") != "training_panel" or not record.get("log_dir") or not record.get("process_log"):
            return False
        text = self._process_log_sample(record.get("process_log"))
        if not text:
            return False
        log_dir_name = Path(str(record["log_dir"])).name
        exact_names = re.findall(r"Exact experiment name requested from command line:\s*(\S+)", text)
        if not exact_names:
            return False
        return not any(log_dir_name == name or log_dir_name.startswith(f"{name}_") for name in exact_names)

    def _display_record(self, record: dict) -> dict:
        display = record.copy()
        if self._log_dir_conflicts_with_process_log(display):
            display["log_dir"] = None
            display["latest_checkpoint"] = None
            display["latest_video"] = None
            display["onnx_path"] = None
            display["has_video"] = False
            display["has_onnx"] = False
            display["has_tensorboard"] = False
        return display

    @staticmethod
    def _is_panel_record(record: dict) -> bool:
        run_id = str(record.get("id") or "")
        return run_id.startswith("panel_") or bool(record.get("process_log"))

    def _discovered_run_for_id(self, run_id: str) -> dict | None:
        log_dir = self.paths.rsl_rl_log_root / run_id
        if not log_dir.is_dir():
            return None
        return {
            "id": run_id,
            "source": "rsl_rl",
            "status": "completed" if latest_checkpoint(log_dir) else "unknown",
            "created_at": datetime.fromtimestamp(log_dir.stat().st_mtime).isoformat(timespec="seconds"),
            "log_dir": str(log_dir),
        }

    def canonical_run_id(self, run_id: str, updates: dict | None = None) -> str:
        """Resolve discovered RSL-RL folder ids back to the owning panel run when possible."""
        cleaned = str(run_id or "").strip()
        if not cleaned:
            return cleaned

        records = self._load_records()
        for record in records:
            if record.get("id") == cleaned and self._is_panel_record(record):
                return cleaned

        log_dir = str((updates or {}).get("log_dir") or "").strip()
        if not log_dir:
            for record in records:
                if record.get("id") == cleaned and record.get("log_dir"):
                    log_dir = str(record["log_dir"])
                    break
        discovered = self._discovered_run_for_id(cleaned)
        if not log_dir and discovered:
            log_dir = str(discovered["log_dir"])

        if log_dir:
            for record in records:
                if record.get("id") != cleaned and self._is_panel_record(record) and str(record.get("log_dir") or "") == log_dir:
                    return str(record["id"])

        if discovered:
            for record in records:
                if record.get("id") != cleaned and self._is_panel_record(record):
                    if self._active_panel_discovered_run(record, [discovered]):
                        return str(record["id"])

        return cleaned

    def _note_path(self, run_id: str) -> Path:
        return self.paths.notes_dir / f"{_safe_note_id(run_id)}.md"

    def _deleted_run_matches(self, entry: Any, run_id: str, log_dir: str | None = None) -> bool:
        if isinstance(entry, str):
            return entry == run_id
        if not isinstance(entry, dict):
            return False
        if entry.get("id") == run_id:
            return True
        if entry.get("log_dir_name") == run_id:
            return True
        if log_dir and entry.get("log_dir") == log_dir:
            return True
        if log_dir and entry.get("log_dir_name") == Path(log_dir).name:
            return True
        return False

    def _is_deleted_run(self, data: dict, run_id: str, log_dir: str | None = None) -> bool:
        return any(self._deleted_run_matches(entry, run_id, log_dir) for entry in data.get("deleted_runs", []))

    def _forget_deleted_run(self, data: dict, run_id: str, log_dir: str | None = None) -> None:
        data["deleted_runs"] = [
            entry for entry in data.get("deleted_runs", []) if not self._deleted_run_matches(entry, run_id, log_dir)
        ]

    def _prune_deleted_run_records(self, data: dict, run_id: str, log_dir: str | None = None) -> None:
        data["runs"] = [
            record
            for record in data.get("runs", [])
            if not self._deleted_run_matches({"id": record.get("id"), "log_dir": record.get("log_dir")}, run_id, log_dir)
        ]

    def _remember_deleted_run(self, data: dict, run_id: str, run: dict | None = None, log_dir: str | None = None) -> None:
        existing = list(data.get("deleted_runs", []))
        if any(
            (entry == run_id if isinstance(entry, str) else isinstance(entry, dict) and entry.get("id") == run_id)
            for entry in existing
        ):
            data["deleted_runs"] = existing
            return
        entry = {
            "id": run_id,
            "deleted_at": datetime.now().isoformat(timespec="seconds"),
        }
        source = (run or {}).get("source")
        if source:
            entry["source"] = source
        resolved_log_dir = log_dir or (run or {}).get("log_dir")
        if resolved_log_dir:
            entry["log_dir"] = str(resolved_log_dir)
            entry["log_dir_name"] = Path(str(resolved_log_dir)).name
        existing.append(entry)
        data["deleted_runs"] = existing

    def deleted_run_tombstones(self, run_ids: list[str] | None = None) -> list[dict]:
        wanted = {str(run_id) for run_id in (run_ids or []) if str(run_id)}
        tombstones = []
        for entry in self._load_data().get("deleted_runs", []):
            if isinstance(entry, str):
                tombstone = {"id": entry}
            elif isinstance(entry, dict) and entry.get("id"):
                tombstone = dict(entry)
            else:
                continue
            if wanted and tombstone.get("id") not in wanted and tombstone.get("log_dir_name") not in wanted:
                continue
            tombstones.append(tombstone)
        return tombstones

    def add_run(self, record: dict) -> None:
        data = self._load_data()
        self._forget_deleted_run(data, str(record.get("id") or ""), str(record.get("log_dir") or "") or None)
        records = list(data.get("runs", []))
        records.append(record)
        data["runs"] = records
        self._save_data(data)

    def update_run(self, run_id: str, **updates: Any) -> None:
        original_run_id = str(run_id or "").strip()
        run_id = self.canonical_run_id(original_run_id, updates)
        if run_id != original_run_id and updates.get("source") == "rsl_rl":
            updates = dict(updates)
            updates.pop("source", None)
        records = self._load_records()
        for record in records:
            if record.get("id") == run_id:
                record.update(updates)
                record["updated_at"] = datetime.now().isoformat(timespec="seconds")
                break
        self._save_records(records)

    def patch_run_metadata(self, run_id: str, **updates: Any) -> dict:
        original_run_id = str(run_id or "").strip()
        run_id = self.canonical_run_id(original_run_id, updates)
        if run_id != original_run_id and updates.get("source") == "rsl_rl":
            updates = dict(updates)
            updates.pop("source", None)
        data = self._load_data()
        if self._is_deleted_run(data, run_id, str(updates.get("log_dir") or "") or None):
            self._prune_deleted_run_records(data, run_id, str(updates.get("log_dir") or "") or None)
            self._save_data(data)
            return {"id": run_id, "deleted": True, **updates}
        records = self._load_records()
        now = datetime.now().isoformat(timespec="seconds")
        for record in records:
            if record.get("id") == run_id:
                record.update(updates)
                record["updated_at"] = now
                self._save_records(records)
                return record
        discovered = self.get_run(run_id) or {"id": run_id, "source": "training_panel"}
        record = {
            "id": run_id,
            "source": discovered.get("source", "training_panel"),
            "created_at": discovered.get("created_at", now),
            "updated_at": now,
            "log_dir": discovered.get("log_dir"),
            **updates,
        }
        records.append(record)
        self._save_records(records)
        return record

    def assign_runs_to_folder(self, run_ids: list[str], folder: str | None) -> list[dict]:
        normalized_folder = folder.strip() if isinstance(folder, str) else None
        if normalized_folder == "":
            normalized_folder = None
        if normalized_folder:
            self.create_folder(normalized_folder)
        updated = []
        seen: set[str] = set()
        for run_id in run_ids:
            cleaned = str(run_id or "").strip()
            if not cleaned or cleaned in seen:
                continue
            seen.add(cleaned)
            updated.append(self.patch_run_metadata(cleaned, folder=normalized_folder))
        return updated

    def create_folder(self, name: str) -> str:
        folder = name.strip()
        if not folder:
            raise ValueError("folder name is required")
        if len(folder) > 80:
            raise ValueError("folder name must be 80 characters or fewer")
        data = self._load_data()
        folders = {str(existing).strip() for existing in data.get("folders", []) if str(existing).strip()}
        folders.add(folder)
        data["folders"] = sorted(folders, key=str.lower)
        self._save_data(data)
        return folder

    def delete_folder(self, name: str) -> dict:
        folder = name.strip()
        if not folder:
            raise ValueError("folder name is required")
        data = self._load_data()
        folders = {str(existing).strip() for existing in data.get("folders", []) if str(existing).strip()}
        removed = folder in folders
        folders.discard(folder)
        now = datetime.now().isoformat(timespec="seconds")
        moved_count = 0
        records = []
        for record in data.get("runs", []):
            if record.get("folder") == folder:
                record = {**record, "folder": None, "updated_at": now}
                moved_count += 1
            records.append(record)
        data["folders"] = sorted(folders, key=str.lower)
        data["runs"] = records
        self._save_data(data)
        return {"folder": folder, "removed": removed, "moved_count": moved_count}

    def rename_folder(self, old_name: str, new_name: str) -> dict:
        old_folder = old_name.strip()
        new_folder = new_name.strip()
        if not old_folder:
            raise ValueError("folder name is required")
        if not new_folder:
            raise ValueError("new folder name is required")
        if len(new_folder) > 80:
            raise ValueError("folder name must be 80 characters or fewer")
        if old_folder == new_folder:
            return {"old_folder": old_folder, "new_folder": new_folder, "renamed": False, "moved_count": 0}

        existing = self.get_folders()
        if old_folder not in existing:
            raise ValueError("folder not found")
        if any(folder.lower() == new_folder.lower() and folder != old_folder for folder in existing):
            raise ValueError("folder already exists")

        data = self._load_data()
        folders = {str(folder).strip() for folder in data.get("folders", []) if str(folder).strip()}
        folders.discard(old_folder)
        folders.add(new_folder)
        now = datetime.now().isoformat(timespec="seconds")
        moved_count = 0
        records = []
        for record in data.get("runs", []):
            if record.get("folder") == old_folder:
                record = {**record, "folder": new_folder, "updated_at": now}
                moved_count += 1
            records.append(record)
        data["folders"] = sorted(folders, key=str.lower)
        data["runs"] = records
        self._save_data(data)
        return {"old_folder": old_folder, "new_folder": new_folder, "renamed": True, "moved_count": moved_count}

    def link_run_to_log(self, run_id: str, log_dir: str, status: str, returncode: int | None) -> dict:
        records = self._load_records()
        now = datetime.now().isoformat(timespec="seconds")
        log_dir_str = str(Path(log_dir))
        discovered_id = Path(log_dir_str).name
        primary: dict | None = None
        duplicate_metadata: dict[str, Any] = {}
        kept = []
        for record in records:
            is_primary = record.get("id") == run_id
            is_duplicate = record.get("id") == discovered_id or record.get("log_dir") == log_dir_str
            if is_primary:
                primary = record
                continue
            if is_duplicate:
                for key in (
                    "folder",
                    "display_name",
                    "reward_preset_id",
                    "reward_overrides",
                    "terrain_preset_id",
                    "terrain_overrides",
                    "created_by",
                    "requester_label",
                ):
                    if record.get(key) and key not in duplicate_metadata:
                        duplicate_metadata[key] = record[key]
                duplicate_params = record.get("params")
                if isinstance(duplicate_params, dict):
                    duplicate_metadata["params"] = _merged_params(duplicate_metadata.get("params"), duplicate_params)
                continue
            kept.append(record)
        if primary is None:
            primary = {"id": run_id, "source": "training_panel", "created_at": now}
        for key, value in duplicate_metadata.items():
            if key == "params":
                primary[key] = _merged_params(value, primary.get("params"))
                continue
            if not primary.get(key):
                primary[key] = value
        if not primary.get("created_by"):
            params = primary.get("params") if isinstance(primary.get("params"), dict) else {}
            requester_id = params.get("requester_id") or params.get("created_by")
            if requester_id:
                primary["created_by"] = requester_id
        if primary.get("created_by"):
            primary["params"] = _merged_params(primary.get("params"), {"requester_id": primary["created_by"]})
        if primary.get("requester_label"):
            primary["params"] = _merged_params(primary.get("params"), {"requester_label": primary["requester_label"]})
        primary.update(
            {
                "status": status,
                "returncode": returncode,
                "log_dir": log_dir_str,
                "updated_at": now,
            }
        )
        kept.append(primary)
        self._save_records(self._collapse_duplicate_runs(kept))
        return primary

    def rename_run(self, run_id: str, display_name: str) -> dict:
        name = display_name.strip()
        if len(name) > 120:
            raise ValueError("display_name must be 120 characters or fewer")
        self.patch_run_metadata(run_id, display_name=name)
        return self.get_run(run_id) or {"id": run_id, "display_name": name}

    def get_note(self, run_id: str) -> str:
        run_id = self.canonical_run_id(run_id)
        if self._is_deleted_run(self._load_data(), run_id):
            return ""
        path = self._note_path(run_id)
        return path.read_text(encoding="utf-8") if path.exists() else ""

    def set_note(self, run_id: str, text: str) -> None:
        run_id = self.canonical_run_id(run_id)
        data = self._load_data()
        if self._is_deleted_run(data, run_id):
            self._prune_deleted_run_records(data, run_id)
            note_path = self._note_path(run_id)
            if note_path.exists():
                note_path.unlink()
            self._save_data(data)
            return
        self.paths.ensure_dirs()
        path = self._note_path(run_id)
        path.write_text(text, encoding="utf-8")
        self.patch_run_metadata(run_id, notes=text)

    def discover_rsl_runs(self) -> list[dict]:
        if not self.paths.rsl_rl_log_root.exists():
            return []
        runs = []
        for log_dir in sorted(self.paths.rsl_rl_log_root.iterdir(), reverse=True):
            if not log_dir.is_dir():
                continue
            checkpoint = latest_checkpoint(log_dir)
            video = latest_video(log_dir)
            onnx = latest_onnx(log_dir)
            params_dir = log_dir / "params"
            runs.append(
                {
                    "id": log_dir.name,
                    "source": "rsl_rl",
                    "status": "completed" if checkpoint else "unknown",
                    "created_at": datetime.fromtimestamp(log_dir.stat().st_mtime).isoformat(timespec="seconds"),
                    "log_dir": str(log_dir),
                    "latest_checkpoint": checkpoint,
                    "latest_video": video,
                    "has_video": bool(video),
                    "onnx_path": onnx,
                    "has_onnx": bool(onnx),
                    "has_tensorboard": any(log_dir.glob("events.out.tfevents.*")),
                    "has_params": params_dir.exists(),
                }
            )
        return runs

    def list_runs(self) -> list[dict]:
        data = self._load_data()
        records = list(data.get("runs", []))
        discovered_runs = self.discover_rsl_runs()
        discovered_by_id = {run["id"]: run for run in discovered_runs}
        discovered_by_log_dir = {run.get("log_dir"): run for run in discovered_runs if run.get("log_dir")}
        merged = []
        represented_ids = set()
        represented_log_dirs = set()
        for record in records:
            if self._is_deleted_run(data, str(record.get("id") or ""), str(record.get("log_dir") or "") or None):
                continue
            record = self._display_record(record)
            unrepresented_discovered = [
                run
                for run in discovered_runs
                if run["id"] not in represented_ids and run.get("log_dir") not in represented_log_dirs
            ]
            discovered = (
                discovered_by_id.get(record.get("id"))
                or discovered_by_log_dir.get(record.get("log_dir"))
                or self._active_panel_discovered_run(record, unrepresented_discovered)
            )
            merged_record = {**(discovered or {}), **record}
            if discovered and not merged_record.get("log_dir") and discovered.get("log_dir"):
                merged_record["log_dir"] = discovered["log_dir"]
            merged.append(merged_record)
            if discovered:
                represented_ids.add(discovered["id"])
                represented_log_dirs.add(discovered.get("log_dir"))
            represented_ids.add(record.get("id"))
            represented_log_dirs.add(record.get("log_dir"))
        for discovered in discovered_runs:
            if self._is_deleted_run(data, str(discovered.get("id") or ""), str(discovered.get("log_dir") or "") or None):
                continue
            if discovered["id"] in represented_ids or discovered.get("log_dir") in represented_log_dirs:
                continue
            merged.append(discovered)
        merged = self._collapse_duplicate_runs(merged)
        defaults = reward_defaults(self.paths.repo_root)
        terrain_current_defaults = terrain_defaults(self.paths.repo_root)
        for record in merged:
            run_id = record.get("id", "")
            note_text = self.get_note(run_id)
            record["notes"] = note_text
            record["has_notes"] = bool(note_text)
            log_dir = Path(record["log_dir"]) if record.get("log_dir") else None
            if log_dir and log_dir.exists():
                record["latest_checkpoint"] = latest_checkpoint(log_dir)
                record["latest_video"] = latest_video(log_dir)
                record["onnx_path"] = latest_onnx(log_dir)
                record["has_video"] = bool(record["latest_video"])
                record["has_onnx"] = bool(record["onnx_path"])
                record["has_tensorboard"] = any(log_dir.glob("events.out.tfevents.*"))
                env_yaml = log_dir / "params" / "env.yaml"
                if env_yaml.exists() and defaults:
                    yaml_scales = read_reward_scales_from_yaml(env_yaml)
                    diff = reward_diff(yaml_scales, defaults)
                    record["reward_diff_count"] = len(diff["changed"])
                else:
                    record.setdefault("reward_diff_count", 0)
                if env_yaml.exists() and terrain_current_defaults:
                    yaml_terrain = read_terrain_values_from_yaml(env_yaml)
                    terrain_config_diff = terrain_diff(yaml_terrain, terrain_current_defaults)
                    record["terrain_diff_count"] = len(terrain_config_diff["changed"])
                else:
                    record.setdefault("terrain_diff_count", 0)
            else:
                record.setdefault("reward_diff_count", 0)
                record.setdefault("terrain_diff_count", 0)
        return sorted(merged, key=lambda item: item.get("created_at", ""), reverse=True)

    def find_latest_log_after(self, start_time: float) -> str | None:
        candidates = []
        if not self.paths.rsl_rl_log_root.exists():
            return None
        for log_dir in self.paths.rsl_rl_log_root.iterdir():
            if log_dir.is_dir() and log_dir.stat().st_mtime >= start_time:
                candidates.append(log_dir)
        if not candidates:
            return None
        return str(max(candidates, key=lambda path: path.stat().st_mtime))

    def get_run(self, run_id: str) -> dict | None:
        for run in self.list_runs():
            if run.get("id") == run_id:
                return run
        return None

    def get_debug(self, run_id: str) -> dict | None:
        run = self.get_run(run_id)
        if not run:
            return None
        if run.get("onnx_status") in ("exporting", "failed") and run.get("onnx_process_log"):
            onnx_log = Path(run["onnx_process_log"])
            return {
                "id": run_id,
                "display_name": run.get("display_name"),
                "status": f"onnx {run.get('onnx_status')}",
                "pid": run.get("onnx_pid"),
                "returncode": run.get("onnx_returncode"),
                "command": run.get("onnx_command"),
                "log_dir": run.get("log_dir"),
                "process_log": str(onnx_log),
                "process_log_tail": tail_file(onnx_log),
                "attach_command": run.get("onnx_process_attach_command"),
                "tmux_session": run.get("onnx_tmux_session"),
                "debug_hint": "This run is showing the ONNX export process log.",
            }
        if run.get("video_status") in ("recording", "failed") and run.get("video_process_log"):
            video_log = Path(run["video_process_log"])
            return {
                "id": run_id,
                "display_name": run.get("display_name"),
                "status": f"video {run.get('video_status')}",
                "pid": run.get("video_pid"),
                "returncode": run.get("video_returncode"),
                "command": run.get("video_command"),
                "log_dir": run.get("log_dir"),
                "process_log": str(video_log),
                "process_log_tail": tail_file(video_log),
                "attach_command": run.get("video_process_attach_command"),
                "tmux_session": run.get("video_tmux_session"),
                "debug_hint": "This run is showing the post-training video recording process log.",
            }
        process_log = Path(run["process_log"]) if run.get("process_log") else None
        return {
            "id": run_id,
            "display_name": run.get("display_name"),
            "status": run.get("status"),
            "pid": run.get("pid"),
            "returncode": run.get("returncode"),
            "command": run.get("command"),
            "log_dir": run.get("log_dir"),
            "process_log": str(process_log) if process_log else None,
            "process_log_tail": tail_file(process_log) if process_log else "",
            "attach_command": run.get("attach_command"),
            "tmux_session": run.get("tmux_session"),
            "debug_hint": (
                "This run was launched by the panel, so the command and captured terminal output are shown here."
                if process_log
                else "This run was discovered from logs/rsl_rl, so no panel process log is available."
            ),
        }

    def _reward_env_yaml_for_run(self, run: dict) -> Path | None:
        if not run or not run.get("log_dir"):
            return None
        env_yaml = Path(str(run["log_dir"])) / "params" / "env.yaml"
        return env_yaml if env_yaml.exists() else None

    @staticmethod
    def _run_time_key(run: dict) -> float:
        try:
            return datetime.fromisoformat(str(run.get("created_at") or "")).timestamp()
        except ValueError:
            return 0.0

    def _previous_reward_run(self, run_id: str, run: dict) -> dict | None:
        current_time = self._run_time_key(run)
        candidates = []
        for candidate in self.list_runs():
            if candidate.get("id") == run_id:
                continue
            env_yaml = self._reward_env_yaml_for_run(candidate)
            if not env_yaml:
                continue
            candidate_time = self._run_time_key(candidate)
            if current_time and candidate_time >= current_time:
                continue
            candidates.append((candidate_time, candidate))
        if not candidates and current_time:
            for candidate in self.list_runs():
                if candidate.get("id") == run_id:
                    continue
                env_yaml = self._reward_env_yaml_for_run(candidate)
                if env_yaml:
                    candidates.append((self._run_time_key(candidate), candidate))
        if not candidates:
            return None
        return max(candidates, key=lambda item: item[0])[1]

    def get_reward_config_for_run(self, run_id: str, compare_to: str = "default") -> dict | None:
        """Return reward diff for this run against defaults or the previous run with saved rewards."""
        run = self.get_run(run_id)
        env_yaml = self._reward_env_yaml_for_run(run or {})
        if not run or not env_yaml:
            return None
        yaml_scales = read_reward_scales_from_yaml(env_yaml)
        compare_mode = str(compare_to or "default").strip().lower()
        if compare_mode in {"previous", "last", "last-run", "last_run"}:
            previous = self._previous_reward_run(run_id, run)
            if not previous:
                return {
                    "run_id": run_id,
                    "preset_id": run.get("reward_preset_id"),
                    "env_yaml": str(env_yaml),
                    "baseline_kind": "previous",
                    "baseline_missing": True,
                    "baseline_label": "previous run",
                    "changed": [],
                    "same": [],
                }
            baseline_yaml = self._reward_env_yaml_for_run(previous)
            baseline_scales = read_reward_scales_from_yaml(baseline_yaml) if baseline_yaml else {}
            diff = reward_diff(yaml_scales, baseline_scales)
            return {
                "run_id": run_id,
                "preset_id": run.get("reward_preset_id"),
                "env_yaml": str(env_yaml),
                "baseline_kind": "previous",
                "baseline_label": previous.get("display_name") or previous.get("id"),
                "baseline_run_id": previous.get("id"),
                "baseline_created_at": previous.get("created_at"),
                "baseline_env_yaml": str(baseline_yaml) if baseline_yaml else None,
                **diff,
            }

        defaults = reward_defaults(self.paths.repo_root)
        diff = reward_diff(yaml_scales, defaults)
        return {
            "run_id": run_id,
            "preset_id": run.get("reward_preset_id"),
            "env_yaml": str(env_yaml),
            "baseline_kind": "default",
            "baseline_label": "default",
            **diff,
        }

    def get_terrain_config_for_run(self, run_id: str) -> dict | None:
        """Return terrain diff between this run's saved params/env.yaml and current defaults."""
        run = self.get_run(run_id)
        if not run or not run.get("log_dir"):
            return None
        log_dir = Path(run["log_dir"])
        env_yaml = log_dir / "params" / "env.yaml"
        if not env_yaml.exists():
            return None
        defaults = terrain_defaults(self.paths.repo_root)
        yaml_values = read_terrain_values_from_yaml(env_yaml)
        diff = terrain_diff(yaml_values, defaults)
        return {
            "run_id": run_id,
            "preset_id": run.get("terrain_preset_id"),
            "env_yaml": str(env_yaml),
            **diff,
        }

    def get_folders(self) -> list[str]:
        """Return sorted list of explicit and run-assigned folder names."""
        data = self._load_data()
        folders: set[str] = {
            str(folder).strip()
            for folder in data.get("folders", [])
            if str(folder).strip()
        }
        for run in self.list_runs():
            folder = run.get("folder")
            if folder and folder.strip():
                folders.add(folder.strip())
        return sorted(folders, key=str.lower)

    def delete_preview(self, run_id: str) -> dict | None:
        run = self.get_run(run_id)
        if not run:
            return None
        paths = []
        seen_paths: set[str] = set()
        log_dir = Path(run["log_dir"]) if run.get("log_dir") else None
        process_log = Path(run["process_log"]) if run.get("process_log") else None
        note_path = self._note_path(run_id)

        def add_path(kind: str, path: Path | None, is_dir: bool = False) -> None:
            if not path or not path.exists():
                return
            path_text = str(path)
            if path_text in seen_paths:
                return
            seen_paths.add(path_text)
            paths.append({"kind": kind, "path": path_text, "is_dir": is_dir})

        add_path("rsl_rl_log_dir", log_dir, is_dir=True)
        add_path("panel_process_log", process_log)
        add_path("panel_exit_file", Path(run["exit_file"]) if run.get("exit_file") else None)
        add_path("panel_video_process_log", Path(run["video_process_log"]) if run.get("video_process_log") else None)
        add_path("panel_video_exit_file", Path(run["video_exit_file"]) if run.get("video_exit_file") else None)
        add_path("panel_onnx_process_log", Path(run["onnx_process_log"]) if run.get("onnx_process_log") else None)
        add_path("panel_onnx_exit_file", Path(run["onnx_exit_file"]) if run.get("onnx_exit_file") else None)
        add_path("panel_note", note_path)
        return {
            "id": run_id,
            "display_name": run.get("display_name"),
            "status": run.get("status"),
            "source": run.get("source"),
            "log_dir": str(log_dir) if log_dir else None,
            "paths": paths,
            "history_record": True,
            "requires_confirmation": run_id,
            "warning": "This permanently removes the run from the panel and deletes the listed repo-owned log/note files.",
        }

    def bulk_delete_preview(self, run_ids: list[str], delete_logs: bool = True) -> dict:
        cleaned_ids = []
        seen: set[str] = set()
        for raw_id in run_ids:
            run_id = str(raw_id or "").strip()
            if run_id and run_id not in seen:
                cleaned_ids.append(run_id)
                seen.add(run_id)
        if not cleaned_ids:
            raise ValueError("at least one run_id is required")

        previews = []
        missing = []
        total_paths = 0
        for run_id in cleaned_ids:
            preview = self.delete_preview(run_id)
            if not preview:
                missing.append(run_id)
                continue
            preview["delete_logs"] = delete_logs
            previews.append(preview)
            total_paths += len(preview.get("paths") or [])
        return {
            "run_ids": cleaned_ids,
            "runs": previews,
            "missing": missing,
            "run_count": len(previews),
            "path_count": total_paths,
            "delete_logs": delete_logs,
            "warning": "This permanently removes selected runs from the panel and deletes repo-owned log/note files.",
        }

    def delete_run(self, run_id: str, confirmation: str = "", delete_logs: bool = True, confirm: bool = False) -> dict:
        if not confirm and confirmation != run_id:
            raise ValueError("Type the exact run id to confirm deletion")
        run = self.get_run(run_id)
        preview = self.delete_preview(run_id)
        if not preview:
            raise ValueError("Run not found")
        deleted_paths = []
        if delete_logs:
            for item in preview["paths"]:
                path = Path(item["path"])
                if not (
                    _is_within(path, self.paths.rsl_rl_log_root)
                    or _is_within(path, self.paths.panel_log_root)
                ):
                    raise ValueError(f"Refusing to delete path outside repo log roots: {path}")
                if path.is_dir():
                    shutil.rmtree(path)
                elif path.exists():
                    path.unlink()
                deleted_paths.append(str(path))

        data = self._load_data()
        records = list(data.get("runs", []))
        log_dir = next((item["path"] for item in preview["paths"] if item["kind"] == "rsl_rl_log_dir"), None)
        kept = [
            record
            for record in records
            if record.get("id") != run_id and (not log_dir or record.get("log_dir") != log_dir)
        ]
        data["runs"] = kept
        self._remember_deleted_run(data, run_id, run=run, log_dir=log_dir or preview.get("log_dir"))
        self._save_data(data)
        return {"deleted": True, "run_id": run_id, "deleted_paths": deleted_paths}

    def bulk_delete_runs(self, run_ids: list[str], delete_logs: bool = True, confirm: bool = False) -> dict:
        if not confirm:
            raise ValueError("confirm must be true for bulk deletion")
        preview = self.bulk_delete_preview(run_ids, delete_logs=delete_logs)
        affected_run_ids = [str(item["id"]) for item in preview["runs"]]
        deleted = []
        skipped_duplicate_ids = []
        unique_runs = []
        seen_run_keys: set[str] = set()
        for item in preview["runs"]:
            log_dir = str(item.get("log_dir") or "").strip()
            run_key = f"log:{log_dir}" if log_dir else f"id:{item['id']}"
            if run_key in seen_run_keys:
                skipped_duplicate_ids.append(str(item["id"]))
                continue
            seen_run_keys.add(run_key)
            unique_runs.append(item)
            deleted.append(str(item["id"]))

        path_items = []
        seen_paths: set[str] = set()
        if delete_logs:
            for item in unique_runs:
                for path_item in item.get("paths") or []:
                    path_text = str(path_item.get("path") or "")
                    if not path_text or path_text in seen_paths:
                        continue
                    path = Path(path_text)
                    if not (
                        _is_within(path, self.paths.rsl_rl_log_root)
                        or _is_within(path, self.paths.panel_log_root)
                    ):
                        raise ValueError(f"Refusing to delete path outside repo log roots: {path}")
                    seen_paths.add(path_text)
                    path_items.append(path_item)

        deleted_paths = []
        for item in sorted(path_items, key=lambda path_item: len(str(path_item.get("path") or "")), reverse=True):
            path = Path(str(item["path"]))
            if path.is_dir():
                shutil.rmtree(path)
                deleted_paths.append(str(path))
            elif path.exists():
                path.unlink()
                deleted_paths.append(str(path))

        data = self._load_data()
        deleted_log_dirs = {str(item.get("log_dir")) for item in unique_runs if item.get("log_dir")}
        affected_ids = set(affected_run_ids)
        data["runs"] = [
            record
            for record in data.get("runs", [])
            if record.get("id") not in affected_ids
            and (not record.get("log_dir") or str(record.get("log_dir")) not in deleted_log_dirs)
        ]
        for item in preview["runs"]:
            self._remember_deleted_run(data, str(item["id"]), run=item, log_dir=item.get("log_dir"))
        self._save_data(data)
        return {
            "deleted": True,
            "run_ids": affected_run_ids,
            "deleted_run_ids": deleted,
            "missing": preview["missing"],
            "deleted_paths": deleted_paths,
            "deleted_count": len(deleted),
            "skipped_duplicate_ids": skipped_duplicate_ids,
        }

    def compact_preview(self, run_id: str) -> dict:
        run = self.get_run(run_id)
        if not run:
            raise ValueError("Run not found")
        if not run.get("log_dir"):
            raise ValueError("Run has no linked log directory")
        log_dir = Path(run["log_dir"])
        if not log_dir.exists():
            raise ValueError("Run log directory does not exist")
        checkpoints = checkpoint_inventory(log_dir)
        if not checkpoints:
            raise ValueError("No model_*.pt checkpoints found")
        kept_iteration, kept_path = checkpoints[-1]
        delete_items = []
        bytes_to_free = 0
        for iteration, path in checkpoints[:-1]:
            size = path.stat().st_size
            bytes_to_free += size
            delete_items.append(
                {
                    "iteration": iteration,
                    "path": str(path),
                    "bytes": size,
                }
            )
        return {
            "id": run_id,
            "log_dir": str(log_dir),
            "kept_checkpoint": str(kept_path),
            "kept_iteration": kept_iteration,
            "delete_count": len(delete_items),
            "bytes_to_free": bytes_to_free,
            "delete_paths": delete_items,
            "requires_confirmation": run_id,
            "warning": "This permanently deletes old top-level model_*.pt checkpoint files and keeps the newest checkpoint.",
        }

    def compact_run(self, run_id: str, confirmation: str) -> dict:
        if confirmation != run_id:
            raise ValueError("Type the exact run id to confirm compaction")
        preview = self.compact_preview(run_id)
        log_dir = Path(preview["log_dir"])
        if not _is_within(log_dir, self.paths.rsl_rl_log_root):
            raise ValueError(f"Refusing to compact path outside RSL-RL log root: {log_dir}")
        deleted_paths = []
        for item in preview["delete_paths"]:
            path = Path(item["path"])
            if path.parent != log_dir or not MODEL_RE.match(path.name):
                raise ValueError(f"Refusing to delete non-checkpoint path: {path}")
            if not _is_within(path, self.paths.rsl_rl_log_root):
                raise ValueError(f"Refusing to delete path outside RSL-RL log root: {path}")
            if path.exists():
                path.unlink()
                deleted_paths.append(str(path))
        self.patch_run_metadata(
            run_id,
            compacted_at=datetime.now().isoformat(timespec="seconds"),
            compacted_deleted_count=len(deleted_paths),
            compacted_bytes_freed=preview["bytes_to_free"],
        )
        return {
            "compacted": True,
            "run_id": run_id,
            "kept_checkpoint": preview["kept_checkpoint"],
            "deleted_paths": deleted_paths,
            "bytes_freed": preview["bytes_to_free"],
        }
