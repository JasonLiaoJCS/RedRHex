from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .commands import (
    TrainingParams,
    display_isaaclab_command,
    play_argv,
    shell_for_command,
    shell_for_isaaclab,
    tensorboard_argv,
    training_argv,
)
from .config import PanelPaths, timestamp_id
from .history import HistoryStore


@dataclass
class ProcessInfo:
    kind: str
    pid: int
    run_id: str
    log_file: str
    started_at: str


class ProcessRegistry:
    def __init__(self, paths: PanelPaths, history: HistoryStore):
        self.paths = paths
        self.history = history
        self._lock = threading.Lock()
        self._processes: dict[str, subprocess.Popen] = {}
        self._infos: dict[str, ProcessInfo] = {}

    def list_processes(self) -> list[dict]:
        with self._lock:
            infos = []
            for run_id, info in self._infos.items():
                proc = self._processes.get(run_id)
                infos.append({**info.__dict__, "returncode": proc.poll() if proc else None})
            return infos

    def start_training(self, params: TrainingParams) -> dict:
        self.paths.ensure_dirs()
        run_id = f"panel_{timestamp_id()}"
        started_at_epoch = time.time()
        started_at = datetime.now().isoformat(timespec="seconds")
        script_argv = training_argv(params)
        shell = shell_for_isaaclab(self.paths, script_argv)
        log_file = self.paths.process_log_dir / f"{run_id}.log"
        record = {
            "id": run_id,
            "source": "training_panel",
            "status": "running",
            "created_at": started_at,
            "updated_at": started_at,
            "params": params.to_dict(),
            "command": display_isaaclab_command(self.paths, script_argv),
            "process_log": str(log_file),
            "log_dir": None,
        }
        self.history.add_run(record)
        proc = self._spawn_shell(shell, log_file)
        self._register(run_id, "training", proc, log_file, started_at)
        thread = threading.Thread(
            target=self._monitor_training,
            args=(run_id, proc, started_at_epoch),
            daemon=True,
        )
        thread.start()
        return {**record, "pid": proc.pid}

    def stop(self, run_id: str) -> bool:
        with self._lock:
            proc = self._processes.get(run_id)
        if not proc or proc.poll() is not None:
            return False
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        self.history.update_run(run_id, status="stopping")
        return True

    def start_tensorboard(self, host: str = "127.0.0.1", port: int = 6006) -> dict:
        run_id = f"tensorboard_{port}"
        with self._lock:
            existing = self._processes.get(run_id)
            if existing and existing.poll() is None:
                return self._tensorboard_response(host, port, existing.pid, already_running=True)
        argv = tensorboard_argv(self.paths.repo_root / "logs" / "rsl_rl", host, port)
        shell = shell_for_command(self.paths, argv)
        log_file = self.paths.process_log_dir / f"{run_id}.log"
        proc = self._spawn_shell(shell, log_file)
        self._register(run_id, "tensorboard", proc, log_file, datetime.now().isoformat(timespec="seconds"))
        return self._tensorboard_response(host, port, proc.pid, already_running=False)

    def start_play(self, run_id: str, checkpoint: str, device: str = "cuda:0") -> dict:
        play_id = f"play_{timestamp_id()}"
        argv = play_argv(checkpoint=checkpoint, device=device)
        shell = shell_for_isaaclab(self.paths, argv)
        log_file = self.paths.process_log_dir / f"{play_id}.log"
        proc = self._spawn_shell(shell, log_file)
        self._register(play_id, "play", proc, log_file, datetime.now().isoformat(timespec="seconds"))
        return {"id": play_id, "source_run_id": run_id, "pid": proc.pid, "process_log": str(log_file)}

    def _spawn_shell(self, shell: str, log_file: Path) -> subprocess.Popen:
        log_handle = log_file.open("w", encoding="utf-8")
        try:
            proc = subprocess.Popen(
                ["bash", "-lc", shell],
                cwd=self.paths.repo_root,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        finally:
            log_handle.close()
        return proc

    def _register(self, run_id: str, kind: str, proc: subprocess.Popen, log_file: Path, started_at: str) -> None:
        info = ProcessInfo(kind=kind, pid=proc.pid, run_id=run_id, log_file=str(log_file), started_at=started_at)
        with self._lock:
            self._processes[run_id] = proc
            self._infos[run_id] = info

    def _monitor_training(self, run_id: str, proc: subprocess.Popen, started_at_epoch: float) -> None:
        returncode = proc.wait()
        status = "completed" if returncode == 0 else "failed"
        log_dir = self.history.find_latest_log_after(started_at_epoch)
        self.history.update_run(run_id, status=status, returncode=returncode, log_dir=log_dir)

    @staticmethod
    def _tensorboard_response(host: str, port: int, pid: int, already_running: bool) -> dict:
        display_host = "127.0.0.1" if host in ("0.0.0.0", "::") else host
        return {
            "pid": pid,
            "already_running": already_running,
            "url": f"http://{display_host}:{port}",
            "host": host,
            "port": port,
        }
