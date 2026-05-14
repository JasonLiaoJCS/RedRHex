from __future__ import annotations

import os
import signal
import socket
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
from .history import HistoryStore, tail_file


@dataclass
class ProcessInfo:
    kind: str
    pid: int
    run_id: str
    log_file: str
    started_at: str
    command: str


class ProcessStartError(RuntimeError):
    def __init__(self, message: str, payload: dict):
        super().__init__(message)
        self.payload = {"error": message, **payload}


class ProcessRegistry:
    def __init__(self, paths: PanelPaths, history: HistoryStore):
        self.paths = paths
        self.history = history
        self._lock = threading.Lock()
        self._processes: dict[str, subprocess.Popen] = {}
        self._infos: dict[str, ProcessInfo] = {}
        self._tensorboards_by_logdir: dict[str, str] = {}

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
        self._register(run_id, "training", proc, log_file, started_at, shell)
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

    def start_tensorboard(self, host: str = "127.0.0.1", port: int | None = None, logdir: Path | None = None) -> dict:
        target_logdir = logdir or (self.paths.repo_root / "logs" / "rsl_rl")
        target_key = str(target_logdir.resolve() if target_logdir.exists() else target_logdir)
        with self._lock:
            existing_id = self._tensorboards_by_logdir.get(target_key)
            existing = self._processes.get(existing_id or "")
            if existing_id and existing and existing.poll() is None:
                existing_info = self._infos[existing_id]
                existing_port = int(existing_info.run_id.rsplit("_", 1)[-1])
                return self._tensorboard_response(
                    host,
                    existing_port,
                    existing.pid,
                    existing_id,
                    existing_info.log_file,
                    existing_info.command,
                    already_running=True,
                )
        selected_port = port or self._find_free_port(6006)
        run_id = f"tensorboard_{selected_port}"
        argv = tensorboard_argv(target_logdir, host, selected_port)
        shell = shell_for_command(self.paths, argv)
        log_file = self.paths.process_log_dir / f"{run_id}.log"
        proc = self._spawn_shell(shell, log_file)
        self._register(run_id, "tensorboard", proc, log_file, datetime.now().isoformat(timespec="seconds"), shell)
        with self._lock:
            self._tensorboards_by_logdir[target_key] = run_id
        self._raise_if_immediate_exit(proc, run_id, "TensorBoard", wait_seconds=2.0)
        return self._tensorboard_response(
            host,
            selected_port,
            proc.pid,
            run_id,
            str(log_file),
            shell,
            already_running=False,
        )

    def start_play(self, run_id: str, checkpoint: str, device: str = "cuda:0") -> dict:
        play_id = f"play_{timestamp_id()}"
        argv = play_argv(checkpoint=checkpoint, device=device)
        shell = shell_for_isaaclab(self.paths, argv)
        log_file = self.paths.process_log_dir / f"{play_id}.log"
        proc = self._spawn_shell(shell, log_file)
        self._register(play_id, "play", proc, log_file, datetime.now().isoformat(timespec="seconds"), shell)
        self._raise_if_immediate_exit(proc, play_id, "Play", wait_seconds=1.0)
        return {"id": play_id, "source_run_id": run_id, "pid": proc.pid, "process_log": str(log_file), "command": shell}

    def get_process_debug(self, process_id: str) -> dict | None:
        with self._lock:
            info = self._infos.get(process_id)
            proc = self._processes.get(process_id)
        if not info:
            return None
        return {
            **info.__dict__,
            "returncode": proc.poll() if proc else None,
            "log_tail": tail_file(Path(info.log_file)),
        }

    def _spawn_shell(self, shell: str, log_file: Path) -> subprocess.Popen:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_handle = log_file.open("w", encoding="utf-8")
        try:
            log_handle.write("$ bash -lc <<'PANEL_COMMAND'\n")
            log_handle.write(shell)
            log_handle.write("\nPANEL_COMMAND\n\n")
            log_handle.flush()
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

    def _register(
        self,
        run_id: str,
        kind: str,
        proc: subprocess.Popen,
        log_file: Path,
        started_at: str,
        command: str,
    ) -> None:
        info = ProcessInfo(
            kind=kind,
            pid=proc.pid,
            run_id=run_id,
            log_file=str(log_file),
            started_at=started_at,
            command=command,
        )
        with self._lock:
            self._processes[run_id] = proc
            self._infos[run_id] = info

    def _raise_if_immediate_exit(
        self,
        proc: subprocess.Popen,
        process_id: str,
        label: str,
        wait_seconds: float,
    ) -> None:
        deadline = time.time() + wait_seconds
        returncode = proc.poll()
        while returncode is None and time.time() < deadline:
            time.sleep(0.1)
            returncode = proc.poll()
        if returncode is None:
            return
        debug = self.get_process_debug(process_id) or {"returncode": returncode}
        raise ProcessStartError(f"{label} exited while starting. Open Debug or check the process log.", debug)

    def _monitor_training(self, run_id: str, proc: subprocess.Popen, started_at_epoch: float) -> None:
        returncode = proc.wait()
        status = "completed" if returncode == 0 else "failed"
        log_dir = self.history.find_latest_log_after(started_at_epoch)
        self.history.update_run(run_id, status=status, returncode=returncode, log_dir=log_dir)

    @staticmethod
    def _find_free_port(start: int) -> int:
        for port in range(start, start + 100):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                try:
                    sock.bind(("127.0.0.1", port))
                except OSError:
                    continue
                return port
        raise RuntimeError("No free port found for TensorBoard")

    @staticmethod
    def _tensorboard_response(
        host: str,
        port: int,
        pid: int,
        process_id: str,
        process_log: str,
        command: str,
        already_running: bool,
    ) -> dict:
        display_host = "127.0.0.1" if host in ("0.0.0.0", "::") else host
        return {
            "pid": pid,
            "id": process_id,
            "already_running": already_running,
            "url": f"http://{display_host}:{port}",
            "host": host,
            "port": port,
            "process_log": process_log,
            "command": command,
        }
