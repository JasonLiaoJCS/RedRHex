from __future__ import annotations

import argparse
import json
import mimetypes
import shlex
import shutil
import subprocess
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from tools.training_panel import __version__

from .activity import ActivityStore
from .commands import DEFAULT_TASK, DEFAULT_VIDEO_PRESET, VIDEO_PRESETS, TrainingParams, VideoParams
from .config import PanelPaths
from .history import HistoryStore
from .presets import PresetStore
from .processes import ProcessRegistry, ProcessStartError
from .remote_config import RemoteStateStore
from .remote_manager import RemoteWorkerManager
from .rewards import reward_defaults, reward_file_index
from .terrain import TerrainPresetStore, terrain_defaults, terrain_file_index
from .tweaks import build_tweak_payload, newest_finished_tweak_run


STATIC_DIR = Path(__file__).resolve().parents[1] / "static"
_PRESET_FILE = Path(__file__).resolve().parents[1] / "reward_presets.json"
_TERRAIN_PRESET_FILE = Path(__file__).resolve().parents[1] / "terrain_presets.json"


def route_id(path: str) -> str:
    return unquote(path.split("/")[3])


def route_id2(path: str) -> str:
    """Extract the second path segment ID (index 4), e.g. /api/presets/{id}/delete."""
    return unquote(path.split("/")[4])


def _is_within(path: Path, root: Path) -> bool:
    resolved = path.resolve()
    resolved_root = root.resolve()
    return resolved == resolved_root or resolved_root in resolved.parents


class PanelState:
    def __init__(self, paths: PanelPaths):
        self.paths = paths
        self.history = HistoryStore(paths)
        self.processes = ProcessRegistry(paths, self.history)
        self.presets = PresetStore(_PRESET_FILE)
        self.terrain_presets = TerrainPresetStore(_TERRAIN_PRESET_FILE)
        self.activity = ActivityStore(paths)
        self.remote_state = RemoteStateStore(paths.remote_state_file)
        self.remote_worker = RemoteWorkerManager(paths, self.remote_state)
        self.remote_worker.autostart_if_enabled()


class PanelHandler(BaseHTTPRequestHandler):
    state: PanelState

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            return self._send_static("index.html")
        if parsed.path.startswith("/static/"):
            return self._send_static(parsed.path.removeprefix("/static/"))
        if parsed.path == "/api/system":
            return self._json(
                {
                    "repo_root": str(self.state.paths.repo_root),
                    "rsl_rl_log_root": str(self.state.paths.rsl_rl_log_root),
                    "default_task": DEFAULT_TASK,
                    "version": __version__,
                    "local_url_hint": "http://127.0.0.1:8080",
                    "lan_hint": "Run with --host 0.0.0.0 and open http://<machine-ip>:8080",
                    "ssh_tunnel_hint": "ssh -L 8080:127.0.0.1:8080 user@host",
                }
            )
        if parsed.path == "/api/activity":
            query = parse_qs(parsed.query)
            try:
                limit = int(query.get("limit", ["80"])[0])
            except ValueError:
                limit = 80
            limit = max(1, min(limit, 200))
            include_remote = query.get("remote", ["1"])[0] not in {"0", "false", "False"}
            window = query.get("window", ["7d"])[0]
            if window not in {"today", "7d", "30d"}:
                window = "7d"
            member = query.get("member", [""])[0]
            category = query.get("category", [""])[0]
            if category not in {"", "training", "artifact", "preset", "metadata", "admin", "system"}:
                category = ""
            return self._json(
                self.state.activity.snapshot(
                    limit=limit,
                    include_remote=include_remote,
                    window=window,
                    member=member,
                    category=category,
                )
            )
        if parsed.path == "/api/remote/status":
            processes = self.state.processes.list_processes()
            active_isaac = self.state.processes.running_isaac_processes()
            return self._json(
                {
                    **self.state.remote_worker.status(),
                    "active_process_count": len([p for p in processes if p.get("returncode") is None]),
                    "active_isaac_process_count": len(active_isaac),
                    "active_isaac_processes": active_isaac,
                }
            )
        if parsed.path == "/api/training/defaults":
            return self._json(TrainingParams().to_dict())
        if parsed.path == "/api/video/presets":
            return self._json({"presets": [params.to_dict() for params in VIDEO_PRESETS.values()]})
        if parsed.path == "/api/runs":
            self.state.processes.reconcile_stale_history()
            return self._json({"runs": self.state.history.list_runs()})
        if parsed.path == "/api/tweaks/last-run":
            self.state.processes.reconcile_stale_history()
            reward_presets = self.state.presets.list_presets()
            run = newest_finished_tweak_run(self.state.history.list_runs(), reward_presets)
            if not run:
                return self._json({"error": "No finished run with usable tweak data found"}, status=404)
            try:
                return self._json(
                    build_tweak_payload(
                        run,
                        reward_presets=reward_presets,
                        terrain_presets=self.state.terrain_presets.list_presets(),
                    )
                )
            except ValueError as exc:
                return self._json({"error": str(exc)}, status=400)
        if parsed.path.startswith("/api/runs/") and parsed.path.endswith("/tweak"):
            run_id = route_id(parsed.path)
            run = self.state.history.get_run(run_id)
            if not run:
                return self._json({"error": "Run not found"}, status=404)
            try:
                return self._json(
                    build_tweak_payload(
                        run,
                        reward_presets=self.state.presets.list_presets(),
                        terrain_presets=self.state.terrain_presets.list_presets(),
                    )
                )
            except ValueError as exc:
                return self._json({"error": str(exc)}, status=400)
        if parsed.path.startswith("/api/runs/") and parsed.path.endswith("/notes"):
            run_id = route_id(parsed.path)
            return self._json({"run_id": run_id, "notes": self.state.history.get_note(run_id)})
        if parsed.path.startswith("/api/runs/") and parsed.path.endswith("/debug"):
            run_id = route_id(parsed.path)
            debug = self.state.history.get_debug(run_id)
            if not debug:
                return self._json({"error": "Run not found"}, status=404)
            return self._json(debug)
        if parsed.path.startswith("/api/runs/") and parsed.path.endswith("/reward-config"):
            run_id = route_id(parsed.path)
            query = parse_qs(parsed.query)
            compare_to = query.get("compare", ["default"])[0]
            config = self.state.history.get_reward_config_for_run(run_id, compare_to=compare_to)
            if config is None:
                return self._json({"error": "No reward config found for run"}, status=404)
            return self._json(config)
        if parsed.path.startswith("/api/runs/") and parsed.path.endswith("/terrain-config"):
            run_id = route_id(parsed.path)
            config = self.state.history.get_terrain_config_for_run(run_id)
            if config is None:
                return self._json({"error": "No terrain config found for run"}, status=404)
            return self._json(config)
        if parsed.path == "/api/folders":
            return self._json({"folders": self.state.history.get_folders()})
        if parsed.path.startswith("/api/runs/") and parsed.path.endswith("/video"):
            run_id = route_id(parsed.path)
            return self._send_run_video(run_id)
        if parsed.path.startswith("/api/runs/") and parsed.path.endswith("/delete-preview"):
            run_id = route_id(parsed.path)
            preview = self.state.history.delete_preview(run_id)
            if not preview:
                return self._json({"error": "Run not found"}, status=404)
            return self._json(preview)
        if parsed.path.startswith("/api/runs/") and parsed.path.endswith("/compact-preview"):
            run_id = route_id(parsed.path)
            try:
                return self._json(self.state.history.compact_preview(run_id))
            except ValueError as exc:
                return self._json({"error": str(exc)}, status=404 if str(exc) == "Run not found" else 400)
        if parsed.path == "/api/tweakables":
            return self._json(reward_file_index(self.state.paths.repo_root))
        if parsed.path == "/api/rewards/defaults":
            return self._json(reward_defaults(self.state.paths.repo_root))
        if parsed.path == "/api/terrain":
            return self._json(terrain_file_index(self.state.paths.repo_root))
        if parsed.path == "/api/terrain/defaults":
            return self._json(terrain_defaults(self.state.paths.repo_root))
        if parsed.path == "/api/terrain/presets":
            return self._json({
                "presets": self.state.terrain_presets.list_presets(),
                "active_preset_id": self.state.terrain_presets.get_active_preset_id(),
            })
        if parsed.path.startswith("/api/terrain/presets/") and not parsed.path.endswith("/update") and not parsed.path.endswith("/delete"):
            preset_id = route_id2(parsed.path)
            preset = self.state.terrain_presets.get_preset(preset_id)
            if not preset:
                return self._json({"error": "Terrain preset not found"}, status=404)
            return self._json(preset)
        if parsed.path == "/api/presets":
            return self._json({
                "presets": self.state.presets.list_presets(),
                "active_preset_id": self.state.presets.get_active_preset_id(),
            })
        if parsed.path.startswith("/api/presets/") and not parsed.path.endswith("/update") and not parsed.path.endswith("/delete"):
            preset_id = route_id(parsed.path)
            preset = self.state.presets.get_preset(preset_id)
            if not preset:
                return self._json({"error": "Preset not found"}, status=404)
            return self._json(preset)
        if parsed.path == "/api/processes":
            return self._json({"processes": self.state.processes.list_processes()})
        if parsed.path.startswith("/api/processes/") and parsed.path.endswith("/debug"):
            process_id = route_id(parsed.path)
            debug = self.state.processes.get_process_debug(process_id)
            if not debug:
                return self._json({"error": "Process not found"}, status=404)
            return self._json(debug)
        if parsed.path == "/api/convergence/settings":
            from .convergence import PRESETS, load_convergence_config
            cfg = load_convergence_config(self.state.paths.convergence_config_file)
            from dataclasses import asdict
            return self._json({"config": asdict(cfg), "presets": PRESETS})
        self._not_found()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            payload = self._payload()
            if parsed.path == "/api/training/start":
                params = TrainingParams.from_dict(payload)
                run = self.state.processes.start_training(params)
                self._record_activity(
                    "training_start",
                    summary=f"Started training {run.get('id')}",
                    subject_id=str(run.get("id") or ""),
                    payload={
                        "run_id": run.get("id"),
                        "reward_preset_id": params.reward_preset_id,
                        "terrain_preset_id": params.terrain_preset_id,
                        "params": params.to_dict(),
                    },
                )
                return self._json(run, status=201)
            if parsed.path == "/api/remote/settings":
                state = self.state.remote_worker.save_settings(payload)
                return self._json({"saved": True, "remote_state": state, "status": self.state.remote_worker.status()})
            if parsed.path == "/api/remote/worker/start":
                return self._json(self.state.remote_worker.start(str(payload.get("mode") or "")))
            if parsed.path == "/api/remote/worker/stop":
                return self._json(self.state.remote_worker.stop())
            if parsed.path == "/api/remote/worker/restart":
                return self._json(self.state.remote_worker.restart(str(payload.get("mode") or "")))
            if parsed.path == "/api/presets":
                name = str(payload.get("name") or "").strip()
                description = str(payload.get("description") or "").strip()
                values = {str(k): float(v) for k, v in (payload.get("values") or {}).items()}
                if not name:
                    return self._json({"error": "name is required"}, status=400)
                preset = self.state.presets.create_preset(name, description, values)
                self._record_activity(
                    "reward_preset_create",
                    summary=f"Created reward profile {preset.get('name')}",
                    subject_id=str(preset.get("id") or ""),
                    payload={"reward_preset_id": preset.get("id")},
                )
                return self._json(preset, status=201)
            if parsed.path.startswith("/api/presets/") and parsed.path.endswith("/update"):
                preset_id = route_id(parsed.path)
                updates: dict = {}
                if "name" in payload:
                    updates["name"] = str(payload["name"])
                if "description" in payload:
                    updates["description"] = str(payload["description"])
                if "values" in payload:
                    updates["values"] = {str(k): float(v) for k, v in payload["values"].items()}
                try:
                    preset = self.state.presets.update_preset(preset_id, **updates)
                except (KeyError, ValueError) as exc:
                    return self._json({"error": str(exc)}, status=400)
                self._record_activity(
                    "reward_preset_edit",
                    summary=f"Edited reward profile {preset.get('name')}",
                    subject_id=preset_id,
                    payload={"reward_preset_id": preset_id},
                )
                return self._json(preset)
            if parsed.path.startswith("/api/presets/") and parsed.path.endswith("/delete"):
                preset_id = route_id(parsed.path)
                try:
                    deleted = self.state.presets.delete_preset(preset_id)
                except ValueError as exc:
                    return self._json({"error": str(exc)}, status=400)
                if deleted:
                    self._record_activity(
                        "reward_preset_delete",
                        summary=f"Deleted reward profile {preset_id}",
                        subject_id=preset_id,
                        payload={"reward_preset_id": preset_id},
                    )
                return self._json({"deleted": deleted})
            if parsed.path == "/api/presets/activate":
                preset_id = str(payload.get("preset_id") or "")
                try:
                    self.state.presets.set_active_preset(preset_id)
                except KeyError as exc:
                    return self._json({"error": str(exc)}, status=404)
                self._record_activity(
                    "reward_preset_activate",
                    summary=f"Activated reward profile {preset_id}",
                    subject_id=preset_id,
                    payload={"reward_preset_id": preset_id},
                )
                return self._json({"active_preset_id": preset_id})
            if parsed.path == "/api/terrain/presets":
                name = str(payload.get("name") or "").strip()
                description = str(payload.get("description") or "").strip()
                values = dict(payload.get("values") or {})
                if not name:
                    return self._json({"error": "name is required"}, status=400)
                preset = self.state.terrain_presets.create_preset(name, description, values)
                self._record_activity(
                    "terrain_preset_create",
                    summary=f"Created terrain profile {preset.get('name')}",
                    subject_id=str(preset.get("id") or ""),
                    payload={"terrain_preset_id": preset.get("id")},
                )
                return self._json(preset, status=201)
            if parsed.path.startswith("/api/terrain/presets/") and parsed.path.endswith("/update"):
                preset_id = route_id2(parsed.path)
                updates: dict = {}
                if "name" in payload:
                    updates["name"] = str(payload["name"])
                if "description" in payload:
                    updates["description"] = str(payload["description"])
                if "values" in payload:
                    updates["values"] = dict(payload["values"])
                try:
                    preset = self.state.terrain_presets.update_preset(preset_id, **updates)
                except (KeyError, ValueError) as exc:
                    return self._json({"error": str(exc)}, status=400)
                self._record_activity(
                    "terrain_preset_edit",
                    summary=f"Edited terrain profile {preset.get('name')}",
                    subject_id=preset_id,
                    payload={"terrain_preset_id": preset_id},
                )
                return self._json(preset)
            if parsed.path.startswith("/api/terrain/presets/") and parsed.path.endswith("/delete"):
                preset_id = route_id2(parsed.path)
                try:
                    deleted = self.state.terrain_presets.delete_preset(preset_id)
                except ValueError as exc:
                    return self._json({"error": str(exc)}, status=400)
                if deleted:
                    self._record_activity(
                        "terrain_preset_delete",
                        summary=f"Deleted terrain profile {preset_id}",
                        subject_id=preset_id,
                        payload={"terrain_preset_id": preset_id},
                    )
                return self._json({"deleted": deleted})
            if parsed.path == "/api/terrain/presets/activate":
                preset_id = str(payload.get("preset_id") or "")
                try:
                    self.state.terrain_presets.set_active_preset(preset_id)
                except KeyError as exc:
                    return self._json({"error": str(exc)}, status=404)
                self._record_activity(
                    "terrain_preset_activate",
                    summary=f"Activated terrain profile {preset_id}",
                    subject_id=preset_id,
                    payload={"terrain_preset_id": preset_id},
                )
                return self._json({"active_preset_id": preset_id})
            if parsed.path == "/api/training/stop":
                run_id = str(payload.get("run_id") or "")
                stopped = self.state.processes.stop(run_id)
                if stopped:
                    self._record_activity("process_stop", summary=f"Stopped process {run_id}", subject_id=run_id)
                return self._json({"stopped": stopped})
            if parsed.path == "/api/folders":
                folder = self.state.history.create_folder(str(payload.get("name") or ""))
                self._record_activity("folder_create", summary=f"Created folder {folder}", subject_id=folder)
                return self._json({"folder": folder, "folders": self.state.history.get_folders()}, status=201)
            if parsed.path == "/api/folders/delete":
                result = self.state.history.delete_folder(str(payload.get("folder") or payload.get("name") or ""))
                self._record_activity(
                    "folder_delete",
                    summary=f"Removed folder {result.get('folder')}",
                    subject_id=str(result.get("folder") or ""),
                    payload=result,
                )
                return self._json({**result, "folders": self.state.history.get_folders()})
            if parsed.path == "/api/folders/rename":
                result = self.state.history.rename_folder(
                    str(payload.get("old_name") or payload.get("folder") or ""),
                    str(payload.get("new_name") or payload.get("name") or ""),
                )
                self._record_activity(
                    "folder_rename",
                    summary=f"Renamed folder {result.get('old_folder')} to {result.get('new_folder')}",
                    subject_id=str(result.get("new_folder") or ""),
                    payload=result,
                )
                return self._json({**result, "folders": self.state.history.get_folders()})
            if parsed.path == "/api/folders/assign":
                return self._json(self._assign_folders(payload))
            if parsed.path == "/api/runs/delete-preview":
                run_ids = payload.get("run_ids") or []
                if not isinstance(run_ids, list):
                    return self._json({"error": "run_ids must be a list"}, status=400)
                return self._json(
                    self.state.history.bulk_delete_preview(
                        [str(run_id) for run_id in run_ids],
                        delete_logs=bool(payload.get("delete_logs", True)),
                    )
                )
            if parsed.path == "/api/runs/delete":
                run_ids = payload.get("run_ids") or []
                if not isinstance(run_ids, list):
                    return self._json({"error": "run_ids must be a list"}, status=400)
                running_by_run = self._running_by_run([str(run_id) for run_id in run_ids])
                if running_by_run:
                    return self._json(
                        {"error": "Stop running processes for selected runs before deleting them", "processes": running_by_run},
                        status=409,
                    )
                result = self.state.history.bulk_delete_runs(
                    [str(run_id) for run_id in run_ids],
                    delete_logs=bool(payload.get("delete_logs", True)),
                    confirm=bool(payload.get("confirm")),
                )
                remote_deleted = self._sync_remote_deleted_runs()
                if remote_deleted is not None:
                    result["remote_delete_requests"] = remote_deleted
                self._record_activity(
                    "bulk_run_delete",
                    summary=f"Deleted {result.get('deleted_count', 0)} selected runs",
                    payload=result,
                )
                return self._json(result)
            if parsed.path.startswith("/api/runs/") and parsed.path.endswith("/stop"):
                run_id = route_id(parsed.path)
                stopped = self.state.processes.stop_all_for_run(run_id)
                if stopped:
                    self._record_activity(
                        "run_process_stop",
                        summary=f"Stopped processes for {run_id}",
                        subject_id=run_id,
                        payload={"stopped_ids": stopped},
                    )
                return self._json({"stopped": bool(stopped), "stopped_ids": stopped})
            if parsed.path.startswith("/api/runs/") and parsed.path.endswith("/folder"):
                run_id = route_id(parsed.path)
                data = self._assign_folders({"run_ids": [run_id], "folder": payload.get("folder")})
                return self._json({"folder": data["folder"], "run_id": run_id, "folders": data["folders"]})
            if parsed.path.startswith("/api/runs/") and parsed.path.endswith("/record-video"):
                run_id = route_id(parsed.path)
                run = self.state.history.get_run(run_id)
                if not run or not run.get("latest_checkpoint"):
                    return self._json({"error": "No checkpoint found for run"}, status=404)
                active_media = self.state.processes.running_isaac_processes()
                if active_media:
                    return self._json(
                        {"error": "Stop the active Isaac process before starting another Isaac action.", "processes": active_media},
                        status=409,
                    )
                result = self.state.processes.start_video_recording(
                    run_id=run_id,
                    checkpoint=str(run["latest_checkpoint"]),
                    device=str(payload.get("device") or "cuda:0"),
                    video_params=VideoParams.from_preset(DEFAULT_VIDEO_PRESET),
                )
                self._record_activity("video_record_start", summary=f"Started video recording for {run_id}", subject_id=run_id, payload=result)
                return self._json(result, status=201)
            if parsed.path.startswith("/api/runs/") and parsed.path.endswith("/export-onnx"):
                run_id = route_id(parsed.path)
                run = self.state.history.get_run(run_id)
                if not run or not run.get("latest_checkpoint"):
                    return self._json({"error": "No checkpoint found for run"}, status=404)
                active_media = self.state.processes.running_isaac_processes()
                if active_media:
                    return self._json(
                        {"error": "Stop the active Isaac process before starting another Isaac action.", "processes": active_media},
                        status=409,
                    )
                result = self.state.processes.start_onnx_export(
                    run_id=run_id,
                    checkpoint=str(run["latest_checkpoint"]),
                    device=str(payload.get("device") or "cuda:0"),
                )
                self._record_activity("onnx_export_start", summary=f"Started ONNX export for {run_id}", subject_id=run_id, payload=result)
                return self._json(result, status=201)
            if parsed.path == "/api/open-location":
                return self._json(self._open_location(str(payload.get("path") or "")))
            if parsed.path == "/api/tensorboard/start":
                host = str(payload.get("host") or "127.0.0.1")
                port = int(payload["port"]) if payload.get("port") else None
                return self._json(self.state.processes.start_tensorboard(host=host, port=port))
            if parsed.path.startswith("/api/runs/") and parsed.path.endswith("/rename"):
                run_id = route_id(parsed.path)
                record = self.state.history.rename_run(run_id, str(payload.get("display_name") or ""))
                self._record_activity(
                    "run_rename",
                    summary=f"Renamed run {run_id}",
                    subject_id=run_id,
                    payload={"display_name": record.get("display_name")},
                )
                return self._json({"saved": True, "run": record})
            if parsed.path.startswith("/api/runs/") and parsed.path.endswith("/tensorboard"):
                run_id = route_id(parsed.path)
                run = self.state.history.get_run(run_id)
                if not run or not run.get("log_dir"):
                    return self._json({"error": "No log directory found for run"}, status=404)
                host = str(payload.get("host") or "127.0.0.1")
                port = int(payload["port"]) if payload.get("port") else None
                result = self.state.processes.start_tensorboard(
                    host=host,
                    port=port,
                    logdir=Path(str(run["log_dir"])),
                    source_run_id=run_id,
                )
                self._record_activity("tensorboard_start", summary=f"Started TensorBoard for {run_id}", subject_id=run_id, payload=result)
                return self._json(result)
            if parsed.path.startswith("/api/runs/") and parsed.path.endswith("/delete"):
                run_id = route_id(parsed.path)
                running = self.state.processes.running_for_run(run_id)
                if running:
                    return self._json(
                        {"error": "Stop running processes for this run before deleting it", "processes": running},
                        status=409,
                    )
                result = self.state.history.delete_run(
                    run_id,
                    confirmation=str(payload.get("confirmation") or ""),
                    delete_logs=bool(payload.get("delete_logs", True)),
                    confirm=bool(payload.get("confirm")),
                )
                remote_deleted = self._sync_remote_deleted_runs()
                if remote_deleted is not None:
                    result["remote_delete_requests"] = remote_deleted
                self._record_activity(
                    "run_delete",
                    summary=f"Deleted run {run_id}",
                    subject_id=run_id,
                    payload=result,
                )
                return self._json(result)
            if parsed.path.startswith("/api/runs/") and parsed.path.endswith("/compact"):
                run_id = route_id(parsed.path)
                running = self.state.processes.running_for_run(run_id)
                if running:
                    return self._json(
                        {"error": "Stop running processes for this run before compacting it", "processes": running},
                        status=409,
                    )
                result = self.state.history.compact_run(
                    run_id,
                    confirmation=str(payload.get("confirmation") or ""),
                )
                self._record_activity(
                    "run_compact",
                    summary=f"Compacted run {run_id}",
                    subject_id=run_id,
                    payload=result,
                )
                return self._json(result)
            if parsed.path.startswith("/api/runs/") and parsed.path.endswith("/notes"):
                run_id = route_id(parsed.path)
                self.state.history.set_note(run_id, str(payload.get("notes") or ""))
                return self._json({"saved": True, "run_id": run_id})
            if parsed.path.startswith("/api/runs/") and parsed.path.endswith("/play"):
                run_id = route_id(parsed.path)
                run = self.state.history.get_run(run_id)
                if not run or not run.get("latest_checkpoint"):
                    return self._json({"error": "No checkpoint found for run"}, status=404)
                active_media = self.state.processes.running_isaac_processes()
                if active_media:
                    return self._json(
                        {"error": "Stop the active Isaac process before starting another Isaac action.", "processes": active_media},
                        status=409,
                    )
                result = self.state.processes.start_play(
                    run_id=run_id,
                    checkpoint=str(run["latest_checkpoint"]),
                    device=str(payload.get("device") or "cuda:0"),
                )
                self._record_activity("play_start", summary=f"Started play for {run_id}", subject_id=run_id, payload=result)
                return self._json(result, status=201)
            if parsed.path == "/api/convergence/settings":
                from dataclasses import asdict
                from .convergence import PRESETS, apply_settings
                cfg = apply_settings(payload, self.state.paths.convergence_config_file)
                return self._json({"saved": True, "config": asdict(cfg), "presets": PRESETS})
        except ProcessStartError as exc:
            return self._json(exc.payload, status=500)
        except ValueError as exc:
            return self._json({"error": str(exc)}, status=400)
        self._not_found()

    def log_message(self, fmt: str, *args) -> None:
        print(f"[training-panel] {self.address_string()} - {fmt % args}")

    def _payload(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8"))

    def _json(self, data: dict, status: int = 200) -> None:
        body = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _record_activity(
        self,
        event_type: str,
        *,
        summary: str = "",
        subject_id: str = "",
        payload: dict | None = None,
    ) -> None:
        try:
            self.state.activity.record(
                event_type,
                summary=summary,
                subject_id=subject_id,
                payload=payload or {},
            )
        except Exception as exc:
            print(f"[training-panel] activity log skipped: {exc}")

    def _running_by_run(self, run_ids: list[str]) -> dict[str, list[dict]]:
        running_by_run = {}
        for run_id in run_ids:
            running = self.state.processes.running_for_run(str(run_id))
            if running:
                running_by_run[str(run_id)] = running
        return running_by_run

    def _sync_remote_deleted_runs(self) -> int | None:
        try:
            from .remote_worker import RemoteWorker
            from .supabase_client import SupabaseClient

            config = self.state.remote_worker.config()
            if not config.configured:
                return None
            client = SupabaseClient(config, timeout=5.0)
            worker = RemoteWorker(
                config,
                self.state.paths,
                client,
                state_store=self.state.remote_state,
            )
            return worker.sync_deleted_runs()
        except Exception as exc:
            print(f"[training-panel] remote delete sync skipped: {exc}")
            return None

    def _assign_folders(self, payload: dict) -> dict:
        raw_run_ids = payload.get("run_ids")
        if not isinstance(raw_run_ids, list):
            raise ValueError("run_ids must be a list")
        run_ids = [str(run_id).strip() for run_id in raw_run_ids if str(run_id or "").strip()]
        if not run_ids:
            raise ValueError("at least one run_id is required")
        raw_folder = payload.get("folder")
        folder = str(raw_folder).strip() if raw_folder is not None else None
        if folder == "":
            folder = None
        runs = self.state.history.assign_runs_to_folder(run_ids, folder)
        return {
            "folder": folder,
            "run_ids": [run.get("id") for run in runs],
            "runs": runs,
            "folders": self.state.history.get_folders(),
        }

    def _open_location(self, requested_path: str) -> dict:
        if not requested_path:
            raise ValueError("path is required")
        path = Path(requested_path).expanduser()
        if not path.exists():
            raise ValueError("path does not exist")
        resolved = path.resolve()
        allowed_roots = (self.state.paths.rsl_rl_log_root, self.state.paths.panel_log_root)
        if not any(_is_within(resolved, root) for root in allowed_roots):
            raise ValueError("Refusing to open a path outside repo-owned log roots")
        opener = shutil.which("xdg-open") or shutil.which("gio")
        command = f"xdg-open {shlex.quote(str(resolved))}"
        opened = False
        error = ""
        if opener:
            argv = [opener, str(resolved)] if Path(opener).name == "xdg-open" else [opener, "open", str(resolved)]
            command = " ".join(shlex.quote(part) for part in argv)
            try:
                subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                opened = True
            except OSError as exc:
                error = str(exc)
        return {
            "path": str(resolved),
            "opened": opened,
            "opener": Path(opener).name if opener else None,
            "command": command,
            "error": error,
        }

    def _send_run_video(self, run_id: str) -> None:
        run = self.state.history.get_run(run_id)
        video = Path(str(run.get("latest_video"))) if run and run.get("latest_video") else None
        if not video or not video.exists() or not video.is_file():
            return self._json({"error": "No recorded video found for run"}, status=404)
        resolved_video = video.resolve()
        resolved_root = self.state.paths.rsl_rl_log_root.resolve()
        if resolved_video != resolved_root and resolved_root not in resolved_video.parents:
            return self._json({"error": "Video path is outside the RSL-RL log root"}, status=403)
        self._send_file_response(resolved_video, "video/mp4")

    def _send_file_response(self, path: Path, content_type: str) -> None:
        file_size = path.stat().st_size
        range_header = self.headers.get("Range")
        start = 0
        end = file_size - 1
        status = 200
        if range_header:
            match = range_header.strip().removeprefix("bytes=").split("-", 1)
            try:
                if match[0]:
                    start = int(match[0])
                if len(match) > 1 and match[1]:
                    end = int(match[1])
            except ValueError:
                start = file_size
            end = min(end, file_size - 1)
            if start < 0 or start >= file_size or end < start:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{file_size}")
                self.end_headers()
                return
            status = 206
        length = end - start + 1
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
        self.end_headers()
        with path.open("rb") as file:
            file.seek(start)
            remaining = length
            while remaining > 0:
                chunk = file.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def _send_static(self, relative: str) -> None:
        path = (STATIC_DIR / relative).resolve()
        if not path.is_file() or STATIC_DIR not in path.parents:
            return self._not_found()
        body = path.read_bytes()
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _not_found(self) -> None:
        self._json({"error": "not found"}, status=404)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the RedRHex local training panel.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host. Use 0.0.0.0 for LAN access.")
    parser.add_argument("--port", type=int, default=8080, help="Bind port.")
    args = parser.parse_args()

    paths = PanelPaths.from_env()
    paths.ensure_dirs()
    PanelHandler.state = PanelState(paths)
    server = ThreadingHTTPServer((args.host, args.port), PanelHandler)
    print(f"RedRHex training panel: http://{args.host}:{args.port}")
    print("For SSH tunnel: ssh -L 8080:127.0.0.1:8080 user@host")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping RedRHex training panel.")
    finally:
        server.server_close()
