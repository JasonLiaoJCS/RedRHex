from __future__ import annotations

import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import PanelPaths
from .rewards import read_reward_scales_from_yaml, reward_defaults, reward_diff

MODEL_RE = re.compile(r"model_(\d+)\.pt$")


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


class HistoryStore:
    def __init__(self, paths: PanelPaths):
        self.paths = paths
        self.paths.ensure_dirs()

    def _load_data(self) -> dict:
        data = _read_json(self.paths.history_file, {"runs": [], "folders": []})
        if not isinstance(data, dict):
            data = {"runs": [], "folders": []}
        data.setdefault("runs", [])
        data.setdefault("folders", [])
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

    def _note_path(self, run_id: str) -> Path:
        return self.paths.notes_dir / f"{_safe_note_id(run_id)}.md"

    def add_run(self, record: dict) -> None:
        records = self._load_records()
        records.append(record)
        self._save_records(records)

    def update_run(self, run_id: str, **updates: Any) -> None:
        records = self._load_records()
        for record in records:
            if record.get("id") == run_id:
                record.update(updates)
                record["updated_at"] = datetime.now().isoformat(timespec="seconds")
                break
        self._save_records(records)

    def patch_run_metadata(self, run_id: str, **updates: Any) -> dict:
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
                for key in ("folder", "display_name", "reward_preset_id", "reward_overrides"):
                    if record.get(key) and key not in duplicate_metadata:
                        duplicate_metadata[key] = record[key]
                continue
            kept.append(record)
        if primary is None:
            primary = {"id": run_id, "source": "training_panel", "created_at": now}
        for key, value in duplicate_metadata.items():
            if not primary.get(key):
                primary[key] = value
        primary.update(
            {
                "status": status,
                "returncode": returncode,
                "log_dir": log_dir_str,
                "updated_at": now,
            }
        )
        kept.append(primary)
        self._save_records(kept)
        return primary

    def rename_run(self, run_id: str, display_name: str) -> dict:
        name = display_name.strip()
        if len(name) > 120:
            raise ValueError("display_name must be 120 characters or fewer")
        self.patch_run_metadata(run_id, display_name=name)
        return self.get_run(run_id) or {"id": run_id, "display_name": name}

    def get_note(self, run_id: str) -> str:
        path = self._note_path(run_id)
        return path.read_text(encoding="utf-8") if path.exists() else ""

    def set_note(self, run_id: str, text: str) -> None:
        self.paths.ensure_dirs()
        path = self._note_path(run_id)
        path.write_text(text, encoding="utf-8")

    def discover_rsl_runs(self) -> list[dict]:
        if not self.paths.rsl_rl_log_root.exists():
            return []
        runs = []
        for log_dir in sorted(self.paths.rsl_rl_log_root.iterdir(), reverse=True):
            if not log_dir.is_dir():
                continue
            checkpoint = latest_checkpoint(log_dir)
            video = latest_video(log_dir)
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
                    "has_tensorboard": any(log_dir.glob("events.out.tfevents.*")),
                    "has_params": params_dir.exists(),
                }
            )
        return runs

    def list_runs(self) -> list[dict]:
        records = self._load_records()
        discovered_runs = self.discover_rsl_runs()
        discovered_by_id = {run["id"]: run for run in discovered_runs}
        discovered_by_log_dir = {run.get("log_dir"): run for run in discovered_runs if run.get("log_dir")}
        merged = []
        represented_ids = set()
        represented_log_dirs = set()
        for record in records:
            discovered = discovered_by_id.get(record.get("id")) or discovered_by_log_dir.get(record.get("log_dir"))
            merged_record = {**(discovered or {}), **record}
            merged.append(merged_record)
            if discovered:
                represented_ids.add(discovered["id"])
                represented_log_dirs.add(discovered.get("log_dir"))
            represented_ids.add(record.get("id"))
            represented_log_dirs.add(record.get("log_dir"))
        for discovered in discovered_runs:
            if discovered["id"] in represented_ids or discovered.get("log_dir") in represented_log_dirs:
                continue
            merged.append(discovered)
        defaults = reward_defaults(self.paths.repo_root)
        for record in merged:
            run_id = record.get("id", "")
            record["has_notes"] = bool(self.get_note(run_id))
            log_dir = Path(record["log_dir"]) if record.get("log_dir") else None
            if log_dir and log_dir.exists():
                record["latest_checkpoint"] = latest_checkpoint(log_dir)
                record["latest_video"] = latest_video(log_dir)
                record["has_video"] = bool(record["latest_video"])
                record["has_tensorboard"] = any(log_dir.glob("events.out.tfevents.*"))
                env_yaml = log_dir / "params" / "env.yaml"
                if env_yaml.exists() and defaults:
                    yaml_scales = read_reward_scales_from_yaml(env_yaml)
                    diff = reward_diff(yaml_scales, defaults)
                    record["reward_diff_count"] = len(diff["changed"])
                else:
                    record.setdefault("reward_diff_count", 0)
            else:
                record.setdefault("reward_diff_count", 0)
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

    def get_reward_config_for_run(self, run_id: str) -> dict | None:
        """Return full reward diff between this run's saved params/env.yaml and current defaults."""
        run = self.get_run(run_id)
        if not run or not run.get("log_dir"):
            return None
        log_dir = Path(run["log_dir"])
        env_yaml = log_dir / "params" / "env.yaml"
        if not env_yaml.exists():
            return None
        defaults = reward_defaults(self.paths.repo_root)
        yaml_scales = read_reward_scales_from_yaml(env_yaml)
        diff = reward_diff(yaml_scales, defaults)
        return {
            "run_id": run_id,
            "preset_id": run.get("reward_preset_id"),
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
        log_dir = Path(run["log_dir"]) if run.get("log_dir") else None
        process_log = Path(run["process_log"]) if run.get("process_log") else None
        note_path = self._note_path(run_id)
        if log_dir and log_dir.exists():
            paths.append({"kind": "rsl_rl_log_dir", "path": str(log_dir), "is_dir": True})
        if process_log and process_log.exists():
            paths.append({"kind": "panel_process_log", "path": str(process_log), "is_dir": False})
        if note_path.exists():
            paths.append({"kind": "panel_note", "path": str(note_path), "is_dir": False})
        return {
            "id": run_id,
            "display_name": run.get("display_name"),
            "status": run.get("status"),
            "source": run.get("source"),
            "paths": paths,
            "history_record": True,
            "requires_confirmation": run_id,
            "warning": "This permanently removes the run from the panel and deletes the listed repo-owned log/note files.",
        }

    def delete_run(self, run_id: str, confirmation: str, delete_logs: bool = True) -> dict:
        if confirmation != run_id:
            raise ValueError("Type the exact run id to confirm deletion")
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

        records = self._load_records()
        log_dir = next((item["path"] for item in preview["paths"] if item["kind"] == "rsl_rl_log_dir"), None)
        kept = [
            record
            for record in records
            if record.get("id") != run_id and (not log_dir or record.get("log_dir") != log_dir)
        ]
        self._save_records(kept)
        return {"deleted": True, "run_id": run_id, "deleted_paths": deleted_paths}
