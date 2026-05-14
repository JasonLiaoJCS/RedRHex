from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import PanelPaths

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


class HistoryStore:
    def __init__(self, paths: PanelPaths):
        self.paths = paths
        self.paths.ensure_dirs()

    def _load_records(self) -> list[dict]:
        data = _read_json(self.paths.history_file, {"runs": []})
        return list(data.get("runs", []))

    def _save_records(self, records: list[dict]) -> None:
        self.paths.ensure_dirs()
        self.paths.history_file.write_text(json.dumps({"runs": records}, indent=2), encoding="utf-8")

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

    def rename_run(self, run_id: str, display_name: str) -> dict:
        name = display_name.strip()
        if len(name) > 120:
            raise ValueError("display_name must be 120 characters or fewer")
        return self.patch_run_metadata(run_id, display_name=name)

    def get_note(self, run_id: str) -> str:
        path = self.paths.notes_dir / f"{_safe_note_id(run_id)}.md"
        return path.read_text(encoding="utf-8") if path.exists() else ""

    def set_note(self, run_id: str, text: str) -> None:
        self.paths.ensure_dirs()
        path = self.paths.notes_dir / f"{_safe_note_id(run_id)}.md"
        path.write_text(text, encoding="utf-8")

    def discover_rsl_runs(self) -> list[dict]:
        if not self.paths.rsl_rl_log_root.exists():
            return []
        runs = []
        for log_dir in sorted(self.paths.rsl_rl_log_root.iterdir(), reverse=True):
            if not log_dir.is_dir():
                continue
            checkpoint = latest_checkpoint(log_dir)
            params_dir = log_dir / "params"
            runs.append(
                {
                    "id": log_dir.name,
                    "source": "rsl_rl",
                    "status": "completed" if checkpoint else "unknown",
                    "created_at": datetime.fromtimestamp(log_dir.stat().st_mtime).isoformat(timespec="seconds"),
                    "log_dir": str(log_dir),
                    "latest_checkpoint": checkpoint,
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
        for record in merged:
            run_id = record.get("id", "")
            record["has_notes"] = bool(self.get_note(run_id))
            log_dir = Path(record["log_dir"]) if record.get("log_dir") else None
            if log_dir and log_dir.exists():
                record["latest_checkpoint"] = latest_checkpoint(log_dir)
                record["has_tensorboard"] = any(log_dir.glob("events.out.tfevents.*"))
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
        process_log = Path(run["process_log"]) if run.get("process_log") else None
        return {
            "id": run_id,
            "display_name": run.get("display_name"),
            "status": run.get("status"),
            "command": run.get("command"),
            "log_dir": run.get("log_dir"),
            "process_log": str(process_log) if process_log else None,
            "process_log_tail": tail_file(process_log) if process_log else "",
            "debug_hint": (
                "This run was launched by the panel, so the command and captured terminal output are shown here."
                if process_log
                else "This run was discovered from logs/rsl_rl, so no panel process log is available."
            ),
        }
