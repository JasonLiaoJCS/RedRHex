from __future__ import annotations

import argparse
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

from .activity import category_for_job_type, outcome_for_status, score_activity_event
from .commands import DEFAULT_VIDEO_PRESET, TrainingParams, VideoParams
from .config import PanelPaths
from .history import HistoryStore, checkpoint_inventory
from .notifications import notification_events_for_run
from .processes import ProcessRegistry
from .remote_config import (
    RemoteConfig,
    RemoteStateStore,
    heartbeat_payload,
    role_allows,
)
from .supabase_client import SupabaseClient

VIDEO_BUCKET = "redrhex-videos"
GPU_JOB_TYPES = {"start_training", "record_video", "export_onnx"}


def _storage_safe(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "").strip())
    return safe or "run"


def _checkpoint_iteration(path: str) -> int | None:
    match = re.search(r"model_(\d+)\.pt$", str(path or ""))
    return int(match.group(1)) if match else None


def run_artifacts(run: dict) -> list[dict]:
    artifacts = []
    seen: set[tuple[str, str]] = set()

    def add_artifact(kind: str, path: str | None) -> None:
        if not path:
            return
        local_path = str(path)
        key = (kind, local_path)
        if key in seen:
            return
        seen.add(key)
        artifact = {
            "kind": kind,
            "path": local_path,
            "local_path": local_path,
            "run_id": run.get("id"),
        }
        try:
            file_path = Path(local_path)
            if file_path.is_file():
                artifact["bytes"] = file_path.stat().st_size
        except OSError:
            pass
        artifacts.append(artifact)

    mapping = {
        "checkpoint": run.get("latest_checkpoint"),
        "video": run.get("latest_video"),
        "onnx": run.get("onnx_path"),
        "process_log": run.get("process_log"),
    }
    for kind, path in mapping.items():
        add_artifact(kind, path)
    log_dir = Path(str(run.get("log_dir") or ""))
    if log_dir.is_dir():
        for _, checkpoint in checkpoint_inventory(log_dir):
            add_artifact("checkpoint", str(checkpoint))
        video_dir = log_dir / "videos" / "play"
        if video_dir.is_dir():
            for video in sorted(video_dir.glob("*.mp4"), key=lambda item: item.stat().st_mtime, reverse=True):
                if video.is_file():
                    add_artifact("video", str(video))
    return artifacts


def _parse_time(value: str | None) -> float:
    if not value:
        return 0.0
    try:
        normalized = str(value).replace("Z", "+00:00")
        dt = datetime.fromisoformat(normalized)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=datetime.now().astimezone().tzinfo)
        return dt.timestamp()
    except ValueError:
        return 0.0


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _uuid_or_none(value: str | None) -> str | None:
    if not value:
        return None
    try:
        return str(UUID(str(value)))
    except ValueError:
        return None


@dataclass
class RemoteJobResult:
    local_run_id: str | None = None
    process_id: str | None = None
    payload: dict | None = None

    def to_dict(self) -> dict:
        return {
            "local_run_id": self.local_run_id,
            "process_id": self.process_id,
            "payload": self.payload or {},
        }


class RemoteJobExecutor:
    def __init__(self, paths: PanelPaths):
        self.paths = paths
        self.history = HistoryStore(paths)
        self.processes = ProcessRegistry(paths, self.history)

    def gpu_locked(self) -> bool:
        return bool(self.processes.running_isaac_processes())

    def execute(self, job: dict) -> RemoteJobResult:
        job_type = str(job.get("type") or job.get("job_type") or "")
        actor_role = str(job.get("actor_role") or job.get("role") or "viewer")
        payload = job.get("payload") or {}
        if not role_allows(actor_role, job_type):
            raise PermissionError(f"Role '{actor_role}' cannot run remote job '{job_type}'")
        if job_type in GPU_JOB_TYPES and self.gpu_locked():
            raise RuntimeError("Another Isaac/GPU action is already running")

        if job_type == "start_training":
            params = TrainingParams.from_dict(payload)
            actor_id = _uuid_or_none(str(job.get("actor_id") or ""))
            if actor_id:
                params.requester_id = actor_id
                params.requester_label = str(job.get("actor_name") or job.get("actor_role") or "Remote member")
            result = self.processes.start_training(params)
            return RemoteJobResult(local_run_id=result.get("id"), process_id=result.get("id"), payload=result)

        if job_type == "stop_process":
            process_id = str(payload.get("process_id") or payload.get("run_id") or "")
            if not process_id:
                raise ValueError("process_id is required")
            stopped_ids = []
            if self.processes.stop(process_id):
                stopped_ids.append(process_id)
            else:
                stopped_ids = self.processes.stop_all_for_run(process_id)
            return RemoteJobResult(process_id=process_id, payload={"stopped": bool(stopped_ids), "stopped_ids": stopped_ids})

        if job_type in {"record_video", "export_onnx"}:
            run = self._run_with_checkpoint(str(payload.get("run_id") or ""))
            checkpoint = self._checkpoint_for_payload(run, payload)
            if job_type == "record_video":
                result = self.processes.start_video_recording(
                    run_id=str(run["id"]),
                    checkpoint=str(checkpoint),
                    device=str(payload.get("device") or "cuda:0"),
                    video_params=VideoParams.from_preset(DEFAULT_VIDEO_PRESET),
                )
            else:
                result = self.processes.start_onnx_export(
                    run_id=str(run["id"]),
                    checkpoint=str(checkpoint),
                    device=str(payload.get("device") or "cuda:0"),
                )
            result["checkpoint"] = str(checkpoint)
            result["checkpoint_iteration"] = _checkpoint_iteration(str(checkpoint))
            return RemoteJobResult(local_run_id=str(run["id"]), process_id=result.get("id"), payload=result)

        if job_type == "tensorboard":
            run_id = str(payload.get("run_id") or "")
            run = self.history.get_run(run_id) or {}
            log_dir = Path(str(run.get("log_dir") or self.paths.rsl_rl_log_root))
            result = self.processes.start_tensorboard(
                host=str(payload.get("host") or "0.0.0.0"),
                port=int(payload["port"]) if payload.get("port") else None,
                logdir=log_dir,
                source_run_id=run_id or None,
            )
            return RemoteJobResult(local_run_id=run_id or None, process_id=result.get("id"), payload=result)

        if job_type == "compact_run":
            run_id = str(payload.get("run_id") or "")
            if self.processes.running_for_run(run_id):
                raise RuntimeError("Stop running processes for this run before compacting it")
            result = self.history.compact_run(run_id, confirmation=str(payload.get("confirmation") or run_id))
            return RemoteJobResult(local_run_id=run_id, payload=result)

        if job_type == "delete_run":
            run_id = str(payload.get("run_id") or "")
            if self.processes.running_for_run(run_id):
                raise RuntimeError("Stop running processes for this run before deleting it")
            result = self.history.delete_run(
                run_id,
                confirmation=str(payload.get("confirmation") or run_id),
                delete_logs=bool(payload.get("delete_logs", True)),
            )
            return RemoteJobResult(local_run_id=run_id, payload=result)

        raise ValueError(f"Unsupported remote job type: {job_type}")

    def _run_with_checkpoint(self, run_id: str) -> dict:
        run = self.history.get_run(run_id)
        if not run or not run.get("latest_checkpoint"):
            raise ValueError("No checkpoint found for run")
        return run

    def _checkpoint_for_payload(self, run: dict, payload: dict) -> Path:
        requested = str(payload.get("checkpoint") or "").strip()
        log_dir_value = str(run.get("log_dir") or "")
        log_dir = Path(log_dir_value) if log_dir_value else None
        if requested:
            checkpoint = Path(requested).expanduser()
            if not checkpoint.is_absolute() and log_dir:
                checkpoint = log_dir / checkpoint
            if not checkpoint.is_file():
                raise ValueError(f"Checkpoint does not exist: {checkpoint}")
            if log_dir and log_dir.is_dir():
                try:
                    resolved = checkpoint.resolve()
                    resolved_log = log_dir.resolve()
                    if resolved != resolved_log and resolved_log not in resolved.parents:
                        raise ValueError("Checkpoint must be inside the selected run log directory")
                except OSError as exc:
                    raise ValueError(f"Checkpoint could not be resolved: {checkpoint}") from exc
            return checkpoint
        iteration = payload.get("checkpoint_iteration")
        if iteration is not None and log_dir and log_dir.is_dir():
            try:
                target_iteration = int(iteration)
            except (TypeError, ValueError):
                raise ValueError("checkpoint_iteration must be an integer") from None
            for checkpoint_iteration, checkpoint in checkpoint_inventory(log_dir):
                if checkpoint_iteration == target_iteration:
                    return checkpoint
            raise ValueError(f"No checkpoint found for iteration {target_iteration}")
        return Path(str(run["latest_checkpoint"]))

    def sync_runs_payload(self) -> list[dict]:
        runs = self.history.list_runs()
        return [
            {
                "id": run.get("id"),
                "status": run.get("status"),
                "display_name": run.get("display_name"),
                "folder": run.get("folder"),
                "notes": self.history.get_note(str(run.get("id") or "")),
                "created_at": run.get("created_at"),
                "updated_at": run.get("updated_at"),
                "log_dir": run.get("log_dir"),
                "params": run.get("params") or {},
                "latest_checkpoint": run.get("latest_checkpoint"),
                "latest_video": run.get("latest_video"),
                "onnx_path": run.get("onnx_path"),
                "created_by": run.get("created_by") or (run.get("params") or {}).get("requester_id"),
                "artifacts": run_artifacts(run),
            }
            for run in runs
        ]


class RemoteWorker:
    def __init__(
        self,
        config: RemoteConfig,
        paths: PanelPaths,
        client: SupabaseClient,
        executor: RemoteJobExecutor | None = None,
        state_store: RemoteStateStore | None = None,
    ):
        self.config = config
        self.paths = paths
        self.client = client
        self.executor = executor or RemoteJobExecutor(paths)
        self.history = getattr(self.executor, "history", HistoryStore(paths))
        self.state_store = state_store or RemoteStateStore(paths.remote_state_file)
        self.active_job_id: str | None = None
        self.last_sync_at = 0.0
        self.last_sync_error = ""

    def send_heartbeat(self, queue_depth: int = 0, gpu_locked: bool | None = None) -> dict:
        accept_jobs = self.state_store.effective_accept_jobs(self.config)
        payload = heartbeat_payload(
            self.config,
            self.paths,
            active_job_id=self.active_job_id,
            queue_depth=queue_depth,
            gpu_locked=self.executor.gpu_locked() if gpu_locked is None else gpu_locked,
            accept_jobs=accept_jobs,
        )
        self.client.heartbeat(payload)
        return payload

    def pull_remote_run_metadata(self) -> int:
        try:
            rows = self.client.select(
                "runs",
                query={
                    "machine_id": f"eq.{self.config.machine_id}",
                    "select": "id,display_name,folder,notes,updated_at",
                },
            )
        except Exception:
            return 0
        if not isinstance(rows, list):
            return 0
        updated = 0
        for remote in rows:
            run_id = str(remote.get("id") or "")
            if not run_id:
                continue
            local = self.history.get_run(run_id) or {}
            if _parse_time(str(remote.get("updated_at") or "")) <= _parse_time(str(local.get("updated_at") or "")):
                continue
            metadata = {
                key: remote.get(key)
                for key in ("display_name", "folder")
                if key in remote
            }
            if metadata:
                self.history.patch_run_metadata(run_id, **metadata)
            if "notes" in remote and remote.get("notes") is not None:
                self.history.set_note(run_id, str(remote.get("notes") or ""))
            updated += 1
        return updated

    def sync_runs(self) -> list[dict]:
        self.pull_remote_run_metadata()
        self.sync_deleted_runs()
        runs = []
        artifacts = []
        artifact_errors = []
        for run in self.executor.sync_runs_payload():
            run_record = dict(run)
            artifact_records = run_record.pop("artifacts", []) or []
            now = _now_iso()
            run_record["created_at"] = run_record.get("created_at") or now
            run_record["updated_at"] = run_record.get("updated_at") or run_record["created_at"]
            run_record["params"] = run_record.get("params") or {}
            runs.append({**run_record, "machine_id": self.config.machine_id})
            for artifact in artifact_records:
                try:
                    record = self._remote_artifact_record(artifact)
                    if record:
                        artifacts.append(record)
                except Exception as exc:
                    artifact_errors.append(str(exc))
        if runs:
            self.client.upsert("runs", runs)
        if artifacts:
            self.client.upsert("artifacts", artifacts, query={"on_conflict": "run_id,kind,local_path"})
        notification_errors = self._dispatch_notification_events(runs, artifacts)
        if notification_errors:
            artifact_errors.extend(notification_errors)
        if artifact_errors:
            self.last_sync_error = "; ".join(artifact_errors[-3:])
        return runs

    def _dispatch_notification_events(self, runs: list[dict], artifacts: list[dict]) -> list[str]:
        errors: list[str] = []
        artifacts_by_run: dict[str, list[dict]] = {}
        for artifact in artifacts:
            run_id = str(artifact.get("run_id") or "")
            if run_id:
                artifacts_by_run.setdefault(run_id, []).append(artifact)
        for run in runs:
            run_id = str(run.get("id") or "")
            events = notification_events_for_run(
                run,
                machine_id=self.config.machine_id,
                artifacts=artifacts_by_run.get(run_id, []),
                remote_url=self.config.cloudflare_tunnel_host,
                require_video_storage=True,
            )
            for event in events:
                try:
                    self.client.function_request("notify", event)
                except Exception as exc:
                    errors.append(f"notify {event.get('event_type')} {run_id}: {exc}")
        return errors

    def sync_deleted_runs(self) -> int:
        deleted_count = 0
        for tombstone in self.history.deleted_run_tombstones():
            for query in self._deleted_run_queries(tombstone):
                self.client.delete("runs", query=query)
                deleted_count += 1
        return deleted_count

    def _deleted_run_queries(self, tombstone: dict) -> list[dict]:
        queries = []
        seen: set[tuple[tuple[str, str], ...]] = set()

        def add(query: dict) -> None:
            normalized = tuple(sorted((str(key), str(value)) for key, value in query.items()))
            if normalized in seen:
                return
            seen.add(normalized)
            queries.append(query)

        run_id = str(tombstone.get("id") or "").strip()
        log_dir = str(tombstone.get("log_dir") or "").strip()
        log_dir_name = str(tombstone.get("log_dir_name") or "").strip()
        base = {"machine_id": f"eq.{self.config.machine_id}"}
        if run_id:
            add({**base, "id": f"eq.{run_id}"})
        if log_dir_name and log_dir_name != run_id:
            add({**base, "id": f"eq.{log_dir_name}"})
        if log_dir:
            add({**base, "log_dir": f"eq.{log_dir}"})
        return queries

    def sync_if_due(self, force: bool = False) -> str:
        now = time.time()
        if not force and now - self.last_sync_at < self.config.sync_interval_seconds:
            return self.last_sync_error
        self.last_sync_at = now
        try:
            self.sync_runs()
            self.last_sync_error = ""
        except Exception as exc:
            self.last_sync_error = str(exc)
        return self.last_sync_error

    def _remote_artifact_record(self, artifact: dict) -> dict | None:
        run_id = str(artifact.get("run_id") or "")
        local_path = str(artifact.get("local_path") or artifact.get("path") or "")
        kind = str(artifact.get("kind") or "")
        if not run_id or not local_path or not kind:
            return None
        record = {
            "run_id": run_id,
            "machine_id": self.config.machine_id,
            "kind": kind,
            "local_path": local_path,
            "storage_path": None,
            "public_url": None,
            "bytes": None,
        }
        if artifact.get("bytes") is not None:
            record["bytes"] = int(artifact["bytes"])
        elif Path(local_path).is_file():
            record["bytes"] = Path(local_path).stat().st_size
        if kind == "video":
            storage_path = self._ensure_video_storage_path(run_id, local_path)
            if storage_path:
                record["storage_path"] = storage_path
        return record

    def _ensure_video_storage_path(self, run_id: str, local_path: str) -> str | None:
        path = Path(local_path)
        if not path.is_file():
            return None
        existing = self._existing_artifact(run_id, "video", local_path)
        if existing and existing.get("storage_path"):
            return str(existing["storage_path"])
        storage_path = f"runs/{_storage_safe(run_id)}/videos/{_storage_safe(path.name)}"
        self.client.upload_storage_object(VIDEO_BUCKET, storage_path, path, content_type="video/mp4")
        return storage_path

    def _existing_artifact(self, run_id: str, kind: str, local_path: str) -> dict | None:
        try:
            rows = self.client.select(
                "artifacts",
                query={
                    "run_id": f"eq.{run_id}",
                    "kind": f"eq.{kind}",
                    "local_path": f"eq.{local_path}",
                    "select": "run_id,kind,local_path,storage_path",
                    "limit": "1",
                },
            )
        except Exception:
            return None
        if isinstance(rows, list) and rows:
            return rows[0]
        return None

    def _profile_for_job(self, job: dict) -> dict:
        actor_id = _uuid_or_none(str(job.get("actor_id") or ""))
        if not actor_id:
            return {}
        try:
            rows = self.client.select(
                "profiles",
                query={
                    "id": f"eq.{actor_id}",
                    "select": "id,email,display_name,role",
                    "limit": "1",
                },
            )
        except Exception:
            return {}
        if isinstance(rows, list) and rows:
            row = rows[0]
            return row if isinstance(row, dict) else {}
        return {}

    def _record_job_activity(
        self,
        job: dict,
        outcome: str,
        *,
        result: dict | None = None,
        error: str = "",
    ) -> None:
        try:
            job_type = str(job.get("type") or job.get("job_type") or "job")
            payload = job.get("payload") if isinstance(job.get("payload"), dict) else {}
            result = result or {}
            result_payload = result.get("payload") if isinstance(result.get("payload"), dict) else {}
            profile = self._profile_for_job(job)
            actor_id = _uuid_or_none(str(job.get("actor_id") or ""))
            actor_name = str(
                profile.get("display_name")
                or profile.get("email")
                or job.get("actor_name")
                or actor_id
                or job.get("actor_role")
                or "Remote member"
            )
            actor_role = str(profile.get("role") or job.get("actor_role") or "viewer")
            category = category_for_job_type(job_type)
            normalized_outcome = outcome_for_status(outcome)
            run_id = str(
                payload.get("run_id")
                or result.get("local_run_id")
                or result_payload.get("run_id")
                or result_payload.get("source_run_id")
                or ""
            )
            metadata = {
                "job_type": job_type,
                "status": normalized_outcome,
                "run_id": run_id,
                "process_id": result.get("process_id") or result_payload.get("id") or "",
                "reward_preset_id": payload.get("reward_preset_id") or "",
                "terrain_preset_id": payload.get("terrain_preset_id") or "",
                "error": error,
            }
            record = {
                "machine_id": self.config.machine_id,
                "actor_id": actor_id,
                "actor_name": actor_name,
                "actor_role": actor_role if actor_role in {"viewer", "operator", "admin"} else "viewer",
                "event_type": f"remote_job_{normalized_outcome}",
                "category": category,
                "outcome": normalized_outcome,
                "run_id": run_id or None,
                "job_id": _uuid_or_none(str(job.get("id") or "")),
                "points": score_activity_event(job_type, outcome=normalized_outcome, category=category, job_type=job_type),
                "metadata": metadata,
                "created_at": _now_iso(),
            }
            self.client.insert("team_activity_events", record, prefer="return=minimal")
        except Exception:
            # Analytics must never block training, video, export, or stop jobs.
            return

    def poll_once(self) -> dict:
        accept_jobs = self.state_store.effective_accept_jobs(self.config)
        gpu_locked = self.executor.gpu_locked()
        heartbeat = self.send_heartbeat(gpu_locked=gpu_locked)
        self.sync_if_due()
        if not accept_jobs:
            return {"status": "disabled", "heartbeat": heartbeat}
        job = self.client.claim_next_job(self.config.machine_id, gpu_locked=gpu_locked)
        if not job:
            return {"status": "idle", "heartbeat": heartbeat}

        job_id = str(job.get("id"))
        self.active_job_id = job_id
        self._record_job_activity(job, "queued")
        self._record_job_activity(job, "claimed")
        self._record_job_activity(job, "running")
        try:
            result = self.executor.execute(job).to_dict()
            sync_error = self.sync_if_due(force=True)
            if sync_error:
                result.setdefault("payload", {})["sync_error"] = sync_error
            self.client.complete_job(job_id, result)
            self._record_job_activity(job, "completed", result=result)
            return {"status": "completed", "job_id": job_id, "result": result}
        except Exception as exc:
            sync_error = self.sync_if_due(force=True)
            result = {"job": job}
            if sync_error:
                result["sync_error"] = sync_error
            self.client.fail_job(job_id, str(exc), result)
            self._record_job_activity(job, "failed", result=result, error=str(exc))
            return {"status": "failed", "job_id": job_id, "error": str(exc)}
        finally:
            self.active_job_id = None
            self.send_heartbeat()

    def run_forever(self) -> None:
        while True:
            try:
                self.poll_once()
            except Exception as exc:
                print(f"[remote-worker] poll error: {exc}")
            time.sleep(self.config.poll_interval_seconds)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the RedRHex remote training worker.")
    parser.add_argument("--once", action="store_true", help="Poll once and exit.")
    args = parser.parse_args(argv)

    paths = PanelPaths.from_env()
    paths.ensure_dirs()
    config = RemoteConfig.from_env()
    client = SupabaseClient(config)
    worker = RemoteWorker(config, paths, client)
    if args.once:
        print(worker.poll_once())
    else:
        print(f"RedRHex remote worker online for machine {config.machine_id}")
        worker.run_forever()
