from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

from .commands import DEFAULT_VIDEO_PRESET, TrainingParams, VideoParams
from .config import PanelPaths
from .history import HistoryStore
from .processes import ProcessRegistry
from .remote_config import (
    RemoteConfig,
    RemoteStateStore,
    heartbeat_payload,
    role_allows,
)
from .supabase_client import SupabaseClient


def run_artifacts(run: dict) -> list[dict]:
    artifacts = []
    mapping = {
        "checkpoint": run.get("latest_checkpoint"),
        "video": run.get("latest_video"),
        "onnx": run.get("onnx_path"),
        "process_log": run.get("process_log"),
    }
    for kind, path in mapping.items():
        if path:
            artifacts.append({"kind": kind, "path": str(path), "run_id": run.get("id")})
    return artifacts


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

        if job_type == "start_training":
            params = TrainingParams.from_dict(payload)
            result = self.processes.start_training(params)
            return RemoteJobResult(local_run_id=result.get("id"), process_id=result.get("id"), payload=result)

        if job_type == "stop_process":
            process_id = str(payload.get("process_id") or payload.get("run_id") or "")
            if not process_id:
                raise ValueError("process_id is required")
            stopped = self.processes.stop(process_id)
            return RemoteJobResult(process_id=process_id, payload={"stopped": stopped})

        if job_type in {"record_video", "export_onnx"}:
            if self.gpu_locked():
                raise RuntimeError("Another Isaac/GPU action is already running")
            run = self._run_with_checkpoint(str(payload.get("run_id") or ""))
            if job_type == "record_video":
                result = self.processes.start_video_recording(
                    run_id=str(run["id"]),
                    checkpoint=str(run["latest_checkpoint"]),
                    device=str(payload.get("device") or "cuda:0"),
                    video_params=VideoParams.from_preset(DEFAULT_VIDEO_PRESET),
                )
            else:
                result = self.processes.start_onnx_export(
                    run_id=str(run["id"]),
                    checkpoint=str(run["latest_checkpoint"]),
                    device=str(payload.get("device") or "cuda:0"),
                )
            return RemoteJobResult(local_run_id=str(run["id"]), process_id=result.get("id"), payload=result)

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

    def sync_runs_payload(self) -> list[dict]:
        runs = self.history.list_runs()
        return [
            {
                "id": run.get("id"),
                "status": run.get("status"),
                "display_name": run.get("display_name"),
                "created_at": run.get("created_at"),
                "updated_at": run.get("updated_at"),
                "log_dir": run.get("log_dir"),
                "params": run.get("params") or {},
                "latest_checkpoint": run.get("latest_checkpoint"),
                "latest_video": run.get("latest_video"),
                "onnx_path": run.get("onnx_path"),
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
        self.state_store = state_store or RemoteStateStore(paths.remote_state_file)
        self.active_job_id: str | None = None

    def send_heartbeat(self, queue_depth: int = 0) -> dict:
        accept_jobs = self.state_store.effective_accept_jobs(self.config)
        payload = heartbeat_payload(
            self.config,
            self.paths,
            active_job_id=self.active_job_id,
            queue_depth=queue_depth,
            gpu_locked=self.executor.gpu_locked(),
            accept_jobs=accept_jobs,
        )
        self.client.heartbeat(payload)
        return payload

    def sync_runs(self) -> list[dict]:
        runs = []
        artifacts = []
        for run in self.executor.sync_runs_payload():
            artifact_records = run.pop("artifacts", []) or []
            runs.append({**run, "machine_id": self.config.machine_id})
            artifacts.extend({**artifact, "machine_id": self.config.machine_id} for artifact in artifact_records)
        if runs:
            self.client.upsert("runs", runs)
        if artifacts:
            self.client.upsert("artifacts", artifacts, query={"on_conflict": "run_id,kind,local_path"})
        return runs

    def poll_once(self) -> dict:
        accept_jobs = self.state_store.effective_accept_jobs(self.config)
        heartbeat = self.send_heartbeat()
        if not accept_jobs:
            return {"status": "disabled", "heartbeat": heartbeat}
        job = self.client.claim_next_job(self.config.machine_id)
        if not job:
            return {"status": "idle", "heartbeat": heartbeat}

        job_id = str(job.get("id"))
        self.active_job_id = job_id
        try:
            result = self.executor.execute(job).to_dict()
            self.sync_runs()
            self.client.complete_job(job_id, result)
            return {"status": "completed", "job_id": job_id, "result": result}
        except Exception as exc:
            self.sync_runs()
            self.client.fail_job(job_id, str(exc), {"job": job})
            return {"status": "failed", "job_id": job_id, "error": str(exc)}
        finally:
            self.active_job_id = None
            self.send_heartbeat()

    def run_forever(self) -> None:
        while True:
            self.poll_once()
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
