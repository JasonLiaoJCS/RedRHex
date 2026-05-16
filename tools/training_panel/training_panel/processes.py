from __future__ import annotations

import os
import re
import signal
import shlex
import shutil
import socket
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .commands import (
    DEFAULT_VIDEO_PRESET,
    TrainingParams,
    VideoParams,
    display_isaaclab_command,
    export_onnx_argv,
    play_argv,
    shell_for_command,
    shell_for_isaaclab,
    tensorboard_argv,
    training_argv,
)
from .config import PanelPaths, timestamp_id
from .history import HistoryStore, latest_checkpoint, latest_onnx, latest_video, tail_file

EXTERNAL_PLAY_ID_PREFIX = "external_play_"
EXTERNAL_ONNX_ID_PREFIX = "external_onnx_"
EXTERNAL_TRAINING_ID_PREFIX = "external_training_"
EXTERNAL_TENSORBOARD_ID_PREFIX = "external_tensorboard_"
EXTERNAL_ID_PREFIXES = (
    EXTERNAL_PLAY_ID_PREFIX,
    EXTERNAL_ONNX_ID_PREFIX,
    EXTERNAL_TRAINING_ID_PREFIX,
    EXTERNAL_TENSORBOARD_ID_PREFIX,
)

@dataclass
class ProcessInfo:
    kind: str
    pid: int
    run_id: str
    log_file: str
    started_at: str
    command: str
    source_run_id: str | None = None
    tmux_session: str | None = None
    attach_command: str | None = None
    exit_file: str | None = None


@dataclass
class SpawnedProcess:
    proc: subprocess.Popen
    tmux_session: str | None = None
    attach_command: str | None = None
    exit_file: str | None = None


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
                infos.append({**info.__dict__, "returncode": self._returncode(info, proc) if proc else None})
        known_groups = {self._process_group_for_pid(info["pid"]) for info in infos}
        external = [process for process in self._external_processes() if process.get("process_group") not in known_groups]
        return infos + external

    def start_training(self, params: TrainingParams) -> dict:
        self.paths.ensure_dirs()
        run_id = f"panel_{timestamp_id()}"
        started_at_epoch = time.time()
        started_at = datetime.now().isoformat(timespec="seconds")
        script_argv = training_argv(params)
        shell = shell_for_isaaclab(self.paths, script_argv)
        log_file = self.paths.process_log_dir / f"{run_id}.log"
        # Write panel overrides before spawning so train.py reads them on startup.
        self._write_reward_override(params.reward_overrides)
        self._write_terrain_override(params.terrain_overrides)
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
            "reward_preset_id": params.reward_preset_id,
            "reward_overrides": params.reward_overrides,
            "terrain_preset_id": params.terrain_preset_id,
            "terrain_overrides": params.terrain_overrides,
        }
        self.history.add_run(record)
        spawned = self._spawn_shell(run_id, shell, log_file)
        proc = spawned.proc
        self.history.update_run(run_id, pid=proc.pid)
        self.history.update_run(
            run_id,
            tmux_session=spawned.tmux_session,
            attach_command=spawned.attach_command,
            exit_file=spawned.exit_file,
        )
        self._register(run_id, "training", spawned, log_file, started_at, shell)
        thread = threading.Thread(
            target=self._monitor_training,
            args=(run_id, proc, started_at_epoch),
            daemon=True,
        )
        thread.start()
        return {
            **record,
            "pid": proc.pid,
            "tmux_session": spawned.tmux_session,
            "attach_command": spawned.attach_command,
        }

    def _write_reward_override(self, overrides: dict) -> None:
        import json as _json
        override_file = self.paths.reward_override_file
        if overrides:
            override_file.parent.mkdir(parents=True, exist_ok=True)
            override_file.write_text(_json.dumps(overrides, indent=2), encoding="utf-8")
        elif override_file.exists():
            override_file.unlink()

    def _write_terrain_override(self, overrides: dict) -> None:
        import json as _json
        override_file = self.paths.terrain_override_file
        if overrides:
            override_file.parent.mkdir(parents=True, exist_ok=True)
            override_file.write_text(_json.dumps(overrides, indent=2), encoding="utf-8")
        elif override_file.exists():
            override_file.unlink()

    def stop(self, run_id: str) -> bool:
        with self._lock:
            proc = self._processes.get(run_id)
            info = self._infos.get(run_id)
        if not proc and run_id.startswith(EXTERNAL_ID_PREFIXES):
            return self._stop_external_group(run_id)
        if not proc or proc.poll() is not None:
            return False
        process_group = os.getpgid(proc.pid)
        if info and info.tmux_session:
            if not self._send_tmux_interrupt(info.tmux_session):
                os.killpg(process_group, signal.SIGINT)
        else:
            os.killpg(process_group, signal.SIGINT)
        self.history.update_run(run_id, status="stopping")
        threading.Thread(
            target=self._force_stop_after_grace,
            args=(proc, process_group, info.tmux_session if info else None),
            daemon=True,
        ).start()
        return True

    def start_tensorboard(
        self,
        host: str = "127.0.0.1",
        port: int | None = None,
        logdir: Path | None = None,
        source_run_id: str | None = None,
    ) -> dict:
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
                    attach_command=existing_info.attach_command,
                    tmux_session=existing_info.tmux_session,
                )
        selected_port = port or self._find_free_port(6006)
        run_id = f"tensorboard_{selected_port}"
        argv = tensorboard_argv(target_logdir, host, selected_port)
        shell = shell_for_command(self.paths, argv)
        log_file = self.paths.process_log_dir / f"{run_id}.log"
        spawned = self._spawn_shell(run_id, shell, log_file)
        proc = spawned.proc
        self._register(
            run_id,
            "tensorboard",
            spawned,
            log_file,
            datetime.now().isoformat(timespec="seconds"),
            shell,
            source_run_id=source_run_id,
        )
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
            attach_command=spawned.attach_command,
            tmux_session=spawned.tmux_session,
        )

    def start_play(self, run_id: str, checkpoint: str, device: str = "cuda:0") -> dict:
        play_id = f"play_{timestamp_id()}"
        argv = play_argv(checkpoint=checkpoint, device=device)
        shell = shell_for_isaaclab(self.paths, argv)
        log_file = self.paths.process_log_dir / f"{play_id}.log"
        spawned = self._spawn_shell(play_id, shell, log_file)
        proc = spawned.proc
        self._register(
            play_id,
            "play",
            spawned,
            log_file,
            datetime.now().isoformat(timespec="seconds"),
            shell,
            source_run_id=run_id,
        )
        self._raise_if_immediate_exit(proc, play_id, "Play", wait_seconds=1.0)
        return {
            "id": play_id,
            "source_run_id": run_id,
            "pid": proc.pid,
            "process_log": str(log_file),
            "command": shell,
            "tmux_session": spawned.tmux_session,
            "attach_command": spawned.attach_command,
            "exit_file": spawned.exit_file,
        }

    def start_video_recording(
        self,
        run_id: str,
        checkpoint: str,
        device: str = "cuda:0",
        video_params: VideoParams | None = None,
    ) -> dict:
        params = video_params or VideoParams.from_preset(DEFAULT_VIDEO_PRESET)
        params.validate()
        video_id = f"video_{timestamp_id()}"
        argv = play_argv(
            checkpoint=checkpoint,
            device=device,
            num_envs=1,
            headless=True,
            video=True,
            video_length=params.length,
            video_width=params.width,
            video_height=params.height,
            video_fps=params.fps,
            rendering_mode=params.rendering_mode,
        )
        shell = shell_for_isaaclab(self.paths, argv)
        log_file = self.paths.process_log_dir / f"{video_id}.log"
        spawned = self._spawn_shell(video_id, shell, log_file)
        proc = spawned.proc
        self._register(
            video_id,
            "video",
            spawned,
            log_file,
            datetime.now().isoformat(timespec="seconds"),
            shell,
            source_run_id=run_id,
        )
        self.history.update_run(
            run_id,
            video_status="recording",
            video_process_id=video_id,
            video_pid=proc.pid,
            video_process_log=str(log_file),
            video_command=shell,
            video_process_attach_command=spawned.attach_command,
            video_tmux_session=spawned.tmux_session,
            video_exit_file=spawned.exit_file,
            video_preset=params.preset,
            video_params=params.to_dict(),
            video_length=params.length,
            video_checkpoint=str(checkpoint),
            video_checkpoint_iteration=self._checkpoint_iteration(checkpoint),
        )
        thread = threading.Thread(
            target=self._monitor_video,
            args=(run_id, video_id, proc),
            daemon=True,
        )
        thread.start()
        return {
            "id": video_id,
            "source_run_id": run_id,
            "pid": proc.pid,
            "process_log": str(log_file),
            "command": shell,
            "tmux_session": spawned.tmux_session,
            "attach_command": spawned.attach_command,
            "exit_file": spawned.exit_file,
            "video_params": params.to_dict(),
            "checkpoint": str(checkpoint),
            "checkpoint_iteration": self._checkpoint_iteration(checkpoint),
        }

    def start_onnx_export(self, run_id: str, checkpoint: str, device: str = "cuda:0") -> dict:
        onnx_id = f"onnx_{timestamp_id()}"
        argv = export_onnx_argv(checkpoint=checkpoint, device=device)
        shell = shell_for_isaaclab(self.paths, argv)
        log_file = self.paths.process_log_dir / f"{onnx_id}.log"
        spawned = self._spawn_shell(onnx_id, shell, log_file)
        proc = spawned.proc
        self._register(
            onnx_id,
            "onnx",
            spawned,
            log_file,
            datetime.now().isoformat(timespec="seconds"),
            shell,
            source_run_id=run_id,
        )
        self.history.update_run(
            run_id,
            onnx_status="exporting",
            onnx_process_id=onnx_id,
            onnx_pid=proc.pid,
            onnx_process_log=str(log_file),
            onnx_command=shell,
            onnx_process_attach_command=spawned.attach_command,
            onnx_tmux_session=spawned.tmux_session,
            onnx_exit_file=spawned.exit_file,
            onnx_error=None,
        )
        thread = threading.Thread(
            target=self._monitor_onnx,
            args=(run_id, onnx_id, proc),
            daemon=True,
        )
        thread.start()
        return {
            "id": onnx_id,
            "source_run_id": run_id,
            "pid": proc.pid,
            "process_log": str(log_file),
            "command": shell,
            "tmux_session": spawned.tmux_session,
            "attach_command": spawned.attach_command,
            "exit_file": spawned.exit_file,
        }

    def get_process_debug(self, process_id: str) -> dict | None:
        with self._lock:
            info = self._infos.get(process_id)
            proc = self._processes.get(process_id)
        if not info:
            external = next(
                (
                    process
                    for process in self._external_processes()
                    if process.get("run_id") == process_id
                ),
                None,
            )
            if not external:
                return None
            return {
                **external,
                "log_tail": tail_file(Path(external["log_file"])) if external.get("log_file") else "",
            }
        return {
            **info.__dict__,
            "returncode": self._returncode(info, proc) if proc else None,
            "log_tail": tail_file(Path(info.log_file)),
        }

    def running_for_run(self, source_run_id: str, kind: str | None = None) -> list[dict]:
        return [
            process
            for process in self.list_processes()
            if process.get("returncode") is None
            and (process.get("run_id") == source_run_id or process.get("source_run_id") == source_run_id)
            and (kind is None or process.get("kind") == kind)
        ]

    def running_media_processes(self) -> list[dict]:
        return self.running_isaac_processes()

    def running_isaac_processes(self) -> list[dict]:
        return [
            process
            for process in self.list_processes()
            if process.get("returncode") is None and process.get("kind") in ("play", "video", "onnx")
        ]

    def stop_all_for_run(self, source_run_id: str) -> list[str]:
        stopped = []
        for process in self.running_for_run(source_run_id):
            process_id = process.get("run_id", "")
            if process_id and self.stop(process_id):
                stopped.append(process_id)
        return stopped

    def reconcile_stale_history(self) -> None:
        known_process_runs = set()
        for process in self.list_processes():
            if process.get("kind") != "training":
                continue
            if process.get("run_id"):
                known_process_runs.add(process["run_id"])
            if process.get("source_run_id"):
                known_process_runs.add(process["source_run_id"])
        for run in self.history.list_runs():
            if run.get("source") != "training_panel":
                continue
            if run.get("status") not in ("running", "stopping"):
                continue
            exit_code = self._exit_code_from_history(run)
            if exit_code is not None:
                log_dir = run.get("log_dir") or self._completed_log_for_run(run)
                if log_dir:
                    status = "completed" if exit_code == 0 else "failed"
                    self.history.link_run_to_log(run["id"], log_dir, status=status, returncode=exit_code)
                    continue
            if run.get("id") not in known_process_runs:
                self.history.update_run(run["id"], status="interrupted")

    def _exit_code_from_history(self, run: dict) -> int | None:
        exit_file = run.get("exit_file")
        if not exit_file:
            return None
        path = Path(str(exit_file))
        if not path.exists():
            return None
        try:
            return int(path.read_text(encoding="utf-8").strip())
        except ValueError:
            return None

    def _completed_log_for_run(self, run: dict) -> str | None:
        process_log = Path(str(run.get("process_log") or ""))
        if process_log.exists():
            text = self._head_file(process_log, max_chars=120000) + "\n" + tail_file(process_log, max_chars=120000)
            match = re.search(r"Exact experiment name requested from command line:\s*(\S+)", text)
            if match:
                candidates = sorted(self.paths.rsl_rl_log_root.glob(f"{match.group(1)}*"))
                candidates = [path for path in candidates if path.is_dir()]
                if candidates:
                    return str(max(candidates, key=lambda path: path.stat().st_mtime))
        created_at = run.get("created_at")
        if created_at:
            try:
                return self.history.find_latest_log_after(datetime.fromisoformat(str(created_at)).timestamp())
            except ValueError:
                return None
        return None

    @staticmethod
    def _head_file(path: Path, max_chars: int = 120000) -> str:
        if not path.exists() or not path.is_file():
            return ""
        with path.open("rb") as file:
            return file.read(max_chars).decode("utf-8", errors="replace")

    def _external_processes(self) -> list[dict]:
        return [
            *self._external_training_processes(),
            *self._external_onnx_processes(),
            *self._external_play_processes(),
            *self._external_tensorboard_processes(),
        ]

    def _external_training_processes(self) -> list[dict]:
        try:
            output = subprocess.check_output(
                ["ps", "-eo", "pid=,pgid=,stat=,args="],
                text=True,
                errors="replace",
            )
        except (OSError, subprocess.SubprocessError):
            return []
        by_group = {}
        for line in output.splitlines():
            parts = line.strip().split(maxsplit=3)
            if len(parts) < 4:
                continue
            pid_text, pgid_text, stat, command = parts
            if not self._is_repo_training_process(command, pid_text):
                continue
            try:
                pid = int(pid_text)
                process_group = int(pgid_text)
            except ValueError:
                continue
            source_run_id = self._source_run_id_from_training_process(pid, process_group, command)
            if not source_run_id:
                continue
            existing = by_group.get(process_group)
            if existing and existing["pid"] == process_group:
                continue
            log_file = self._matching_training_log(pid, process_group, command)
            by_group[process_group] = {
                "kind": "training",
                "pid": pid,
                "process_group": process_group,
                "run_id": f"{EXTERNAL_TRAINING_ID_PREFIX}{process_group}",
                "source_run_id": source_run_id,
                "log_file": str(log_file) if log_file else "",
                "started_at": "",
                "command": command,
                "returncode": None,
                "external": True,
                "stat": stat,
            }
        return list(by_group.values())

    def _external_play_processes(self) -> list[dict]:
        try:
            output = subprocess.check_output(
                ["ps", "-eo", "pid=,pgid=,stat=,args="],
                text=True,
                errors="replace",
            )
        except (OSError, subprocess.SubprocessError):
            return []
        by_group = {}
        for line in output.splitlines():
            parts = line.strip().split(maxsplit=3)
            if len(parts) < 4:
                continue
            pid_text, pgid_text, stat, command = parts
            if not self._is_repo_play_process(command, pid_text):
                continue
            try:
                pid = int(pid_text)
                process_group = int(pgid_text)
            except ValueError:
                continue
            source_run_id = self._source_run_id_from_command(command)
            if not source_run_id:
                continue
            existing = by_group.get(process_group)
            if existing and existing["pid"] == process_group:
                continue
            log_file = self._matching_process_log(command)
            by_group[process_group] = {
                "kind": "play",
                "pid": pid,
                "process_group": process_group,
                "run_id": f"{EXTERNAL_PLAY_ID_PREFIX}{process_group}",
                "source_run_id": source_run_id,
                "log_file": str(log_file) if log_file else "",
                "started_at": "",
                "command": command,
                "returncode": None,
                "external": True,
                "stat": stat,
            }
        return list(by_group.values())

    def _external_onnx_processes(self) -> list[dict]:
        try:
            output = subprocess.check_output(
                ["ps", "-eo", "pid=,pgid=,stat=,args="],
                text=True,
                errors="replace",
            )
        except (OSError, subprocess.SubprocessError):
            return []
        by_group = {}
        for line in output.splitlines():
            parts = line.strip().split(maxsplit=3)
            if len(parts) < 4:
                continue
            pid_text, pgid_text, stat, command = parts
            if not self._is_repo_onnx_process(command, pid_text):
                continue
            try:
                pid = int(pid_text)
                process_group = int(pgid_text)
            except ValueError:
                continue
            source_run_id = self._source_run_id_from_command(command)
            if not source_run_id:
                continue
            log_file = self._matching_onnx_log(command)
            by_group[process_group] = {
                "kind": "onnx",
                "pid": pid,
                "process_group": process_group,
                "run_id": f"{EXTERNAL_ONNX_ID_PREFIX}{process_group}",
                "source_run_id": source_run_id,
                "log_file": str(log_file) if log_file else "",
                "started_at": "",
                "command": command,
                "returncode": None,
                "external": True,
                "stat": stat,
            }
        return list(by_group.values())

    def _external_tensorboard_processes(self) -> list[dict]:
        try:
            output = subprocess.check_output(
                ["ps", "-eo", "pid=,pgid=,stat=,args="],
                text=True,
                errors="replace",
            )
        except (OSError, subprocess.SubprocessError):
            return []
        by_group = {}
        for line in output.splitlines():
            parts = line.strip().split(maxsplit=3)
            if len(parts) < 4:
                continue
            pid_text, pgid_text, stat, command = parts
            if "tensorboard_data_server" in command:
                continue
            if not self._is_repo_tensorboard_process(command):
                continue
            try:
                pid = int(pid_text)
                process_group = int(pgid_text)
            except ValueError:
                continue
            source_run_id = self._source_run_id_from_tensorboard_command(command)
            if not source_run_id:
                continue
            log_file = self._matching_tensorboard_log(command)
            by_group[process_group] = {
                "kind": "tensorboard",
                "pid": pid,
                "process_group": process_group,
                "run_id": f"{EXTERNAL_TENSORBOARD_ID_PREFIX}{process_group}",
                "source_run_id": source_run_id,
                "log_file": str(log_file) if log_file else "",
                "started_at": "",
                "command": command,
                "returncode": None,
                "external": True,
                "stat": stat,
            }
        return list(by_group.values())

    def _source_run_id_from_command(self, command: str) -> str | None:
        match = re.search(r"logs/rsl_rl/redrhex_wheg/([^/\s]+)/model_\d+\.pt", command)
        return match.group(1) if match else None

    def _source_run_id_from_tensorboard_command(self, command: str) -> str | None:
        match = re.search(r"logs/rsl_rl/redrhex_wheg/([^/\s]+)(?:\s|$)", command)
        return match.group(1) if match else None

    def _source_run_id_from_training_process(self, pid: int, process_group: int, command: str) -> str | None:
        for run in self.history.list_runs():
            if run.get("source") != "training_panel":
                continue
            if run.get("pid") in (pid, process_group):
                return run.get("id")
            recorded_command = run.get("command") or ""
            if recorded_command and self._training_commands_match(recorded_command, command):
                return run.get("id")
        return None

    def _matching_process_log(self, command: str) -> Path | None:
        checkpoint_match = re.search(r"--checkpoint\s+(\S+)", command)
        checkpoint = checkpoint_match.group(1) if checkpoint_match else ""
        candidates = []
        for path in self.paths.process_log_dir.glob("play_*.log"):
            text = tail_file(path, max_chars=20000)
            if checkpoint and checkpoint in text:
                candidates.append(path)
        if not candidates:
            return None
        return max(candidates, key=lambda path: path.stat().st_mtime)

    def _matching_onnx_log(self, command: str) -> Path | None:
        checkpoint_match = re.search(r"--checkpoint\s+(\S+)", command)
        checkpoint = checkpoint_match.group(1) if checkpoint_match else ""
        candidates = []
        for path in self.paths.process_log_dir.glob("onnx_*.log"):
            text = tail_file(path, max_chars=20000)
            if checkpoint and checkpoint in text:
                candidates.append(path)
        if not candidates:
            return None
        return max(candidates, key=lambda path: path.stat().st_mtime)

    def _matching_tensorboard_log(self, command: str) -> Path | None:
        logdir_match = re.search(r"--logdir\s+(\S+)", command)
        logdir = logdir_match.group(1) if logdir_match else ""
        candidates = []
        for path in self.paths.process_log_dir.glob("tensorboard_*.log"):
            text = tail_file(path, max_chars=20000)
            if logdir and logdir in text:
                candidates.append(path)
        if not candidates:
            return None
        return max(candidates, key=lambda path: path.stat().st_mtime)

    def _matching_training_log(self, pid: int, process_group: int, command: str) -> Path | None:
        for run in self.history.list_runs():
            if run.get("source") != "training_panel":
                continue
            if run.get("pid") in (pid, process_group) or self._training_commands_match(run.get("command") or "", command):
                process_log = run.get("process_log")
                if process_log:
                    return Path(process_log)
        return None

    def _is_repo_training_process(self, command: str, pid_text: str) -> bool:
        if self._is_tmux_server_command(command):
            return False
        if "scripts/rsl_rl/train.py" not in command:
            return False
        if "isaaclab.sh -p" not in command and "python" not in command:
            return False
        return self._command_or_cwd_matches_repo(command, pid_text)

    def _is_repo_play_process(self, command: str, pid_text: str) -> bool:
        if self._is_tmux_server_command(command):
            return False
        if "--export_policy_only" in command:
            return False
        if "scripts/rsl_rl/play.py" not in command:
            return False
        if "isaaclab.sh -p" not in command and "python" not in command:
            return False
        return self._command_or_cwd_matches_repo(command, pid_text)

    def _is_repo_onnx_process(self, command: str, pid_text: str) -> bool:
        if self._is_tmux_server_command(command):
            return False
        if "scripts/rsl_rl/play.py" not in command or "--export_policy_only" not in command:
            return False
        if "isaaclab.sh -p" not in command and "python" not in command:
            return False
        return self._command_or_cwd_matches_repo(command, pid_text)

    def _is_repo_tensorboard_process(self, command: str) -> bool:
        return "tensorboard" in command and "--logdir" in command and str(self.paths.repo_root) in command

    @staticmethod
    def _is_tmux_server_command(command: str) -> bool:
        return "tmux new-session" in command or "tmux: server" in command or "/tmux new-session" in command

    def _command_or_cwd_matches_repo(self, command: str, pid_text: str) -> bool:
        if str(self.paths.repo_root) in command:
            return True
        try:
            return Path(f"/proc/{int(pid_text)}/cwd").resolve() == self.paths.repo_root.resolve()
        except (OSError, RuntimeError, ValueError):
            return False

    @staticmethod
    def _training_commands_match(recorded_command: str, process_command: str) -> bool:
        if not recorded_command or "scripts/rsl_rl/train.py" not in process_command:
            return False
        keys = ("--task", "--num_envs", "--max_iterations", "--device", "--seed", "--checkpoint")
        matched = 0
        for key in keys:
            recorded = ProcessRegistry._arg_value(recorded_command, key)
            observed = ProcessRegistry._arg_value(process_command, key)
            if not recorded or not observed:
                continue
            if recorded != observed:
                return False
            if recorded == observed:
                matched += 1
        return matched >= 3

    @staticmethod
    def _arg_value(command: str, key: str) -> str | None:
        match = re.search(rf"{re.escape(key)}(?:=|\s+)([^\s]+)", command)
        return match.group(1) if match else None

    @staticmethod
    def _process_group_for_pid(pid: int) -> int | None:
        try:
            return os.getpgid(pid)
        except ProcessLookupError:
            return None

    def _stop_external_group(self, run_id: str) -> bool:
        process_id = run_id
        for prefix in EXTERNAL_ID_PREFIXES:
            process_id = process_id.removeprefix(prefix)
        try:
            process_group = int(process_id)
        except ValueError:
            return False
        try:
            os.killpg(process_group, signal.SIGINT)
        except ProcessLookupError:
            return False
        threading.Thread(target=self._force_kill_group_after_grace, args=(process_group,), daemon=True).start()
        return True

    def _spawn_shell(self, run_id: str, shell: str, log_file: Path) -> SpawnedProcess:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        tmux = shutil.which("tmux")
        if tmux:
            return self._spawn_tmux(run_id, shell, log_file, tmux)

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
        return SpawnedProcess(proc=proc)

    def _spawn_tmux(self, run_id: str, shell: str, log_file: Path, tmux: str) -> SpawnedProcess:
        session = self._safe_tmux_session(run_id)
        done_signal = f"done_{session}"
        exit_file = self.paths.process_log_dir / f"{run_id}.exit"
        attach_command = f"tmux attach -t {shlex.quote(session)}"
        log_file.write_text("", encoding="utf-8")
        inner = "\n".join(
            [
                "set +e",
                f"exec > >(tee -a {shlex.quote(str(log_file))}) 2>&1",
                "echo \"$ bash -lc <<'PANEL_COMMAND'\"",
                "cat <<'PANEL_COMMAND'",
                shell,
                "PANEL_COMMAND",
                "echo",
                f"bash -lc {shlex.quote(shell)}",
                "status=$?",
                f"printf '%s' \"$status\" > {shlex.quote(str(exit_file))}",
                f"{shlex.quote(tmux)} wait-for -S {shlex.quote(done_signal)}",
                "exit \"$status\"",
            ]
        )
        outer = "\n".join(
            [
                f"{shlex.quote(tmux)} new-session -d -s {shlex.quote(session)} -- bash -lc {shlex.quote(inner)}"
                " || exit $?",
                f"{shlex.quote(tmux)} wait-for {shlex.quote(done_signal)}",
                "status=0",
                f"if [ -f {shlex.quote(str(exit_file))} ]; then status=$(cat {shlex.quote(str(exit_file))}); fi",
                'exit "$status"',
            ]
        )
        proc = subprocess.Popen(
            ["bash", "-lc", outer],
            cwd=self.paths.repo_root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        return SpawnedProcess(
            proc=proc,
            tmux_session=session,
            attach_command=attach_command,
            exit_file=str(exit_file),
        )

    def _register(
        self,
        run_id: str,
        kind: str,
        spawned: SpawnedProcess,
        log_file: Path,
        started_at: str,
        command: str,
        source_run_id: str | None = None,
    ) -> None:
        info = ProcessInfo(
            kind=kind,
            pid=spawned.proc.pid,
            run_id=run_id,
            log_file=str(log_file),
            started_at=started_at,
            command=command,
            source_run_id=source_run_id,
            tmux_session=spawned.tmux_session,
            attach_command=spawned.attach_command,
            exit_file=spawned.exit_file,
        )
        with self._lock:
            self._processes[run_id] = spawned.proc
            self._infos[run_id] = info

    @staticmethod
    def _safe_tmux_session(run_id: str) -> str:
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", run_id).strip("._-")
        return f"redrhex_{safe or timestamp_id()}"[:80]

    @staticmethod
    def _send_tmux_interrupt(session: str) -> bool:
        try:
            result = subprocess.run(
                ["tmux", "send-keys", "-t", session, "C-c"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        except OSError:
            return False
        return result.returncode == 0

    @staticmethod
    def _kill_tmux_session(session: str) -> None:
        try:
            subprocess.run(
                ["tmux", "kill-session", "-t", session],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        except OSError:
            return
        try:
            subprocess.run(
                ["tmux", "wait-for", "-S", f"done_{session}"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        except OSError:
            return

    @staticmethod
    def _returncode(info: ProcessInfo, proc: subprocess.Popen) -> int | None:
        returncode = proc.poll()
        if returncode is None:
            return None
        if info.exit_file:
            exit_path = Path(info.exit_file)
            if exit_path.exists():
                try:
                    return int(exit_path.read_text(encoding="utf-8").strip())
                except ValueError:
                    return returncode
        return returncode

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
        raise ProcessStartError(f"{label} exited while starting. Open Console or check the process log.", debug)

    def _monitor_training(self, run_id: str, proc: subprocess.Popen, started_at_epoch: float) -> None:
        import time
        from .convergence import ConvergenceChecker, load_convergence_config
        from .notifications import send_convergence_notification
        from .remote_config import RemoteConfig

        checker = ConvergenceChecker()
        convergence_detected = False
        convergence_notified_at: float | None = None
        log_dir: str | None = None
        poll_interval = 60  # seconds between TensorBoard reads

        # Poll while training is running, checking for convergence each cycle.
        while proc.poll() is None:
            time.sleep(poll_interval)
            if not log_dir:
                log_dir = self.history.find_latest_log_after(started_at_epoch)
            if log_dir and not convergence_detected:
                try:
                    cfg = load_convergence_config(self.paths.convergence_config_file)
                    if cfg.enabled:
                        cooled = (
                            convergence_notified_at is None
                            or (time.time() - convergence_notified_at) > cfg.cooldown_minutes * 60
                        )
                        if cooled:
                            result = checker.check(Path(log_dir), cfg)
                            if result.detected:
                                convergence_detected = True
                                convergence_notified_at = time.time()
                                self.history.update_run(
                                    run_id,
                                    convergence_detected=True,
                                    convergence_iteration=result.iteration,
                                    convergence_improvement_pct=round(result.improvement_pct, 2),
                                )
                                if cfg.auto_record_video:
                                    self.history.update_run(run_id, queue_video_on_completion=True)
                                remote_cfg = RemoteConfig.from_env()
                                run = self.history.get_run(run_id) or {}
                                send_convergence_notification(
                                    run, result,
                                    discord_webhook=remote_cfg.discord_webhook_url,
                                    resend_key=remote_cfg.resend_api_key,
                                )
                except Exception:
                    pass  # never let convergence logic crash the monitor thread

        returncode = proc.wait()
        status = "completed" if returncode == 0 else "failed"
        if not log_dir:
            log_dir = self.history.find_latest_log_after(started_at_epoch)
        self.history.update_run(run_id, status=status, returncode=returncode, log_dir=log_dir)

        # Record video when: training succeeded normally, OR convergence was detected and
        # auto_record_video was requested (even if training was stopped early).
        run = self.history.get_run(run_id) or {}
        force_video = bool(run.get("queue_video_on_completion")) and convergence_detected
        if (returncode == 0 or force_video) and log_dir:
            checkpoint = latest_checkpoint(Path(log_dir))
            if not checkpoint:
                self.history.update_run(run_id, video_status="missing_checkpoint")
                return
            try:
                params = run.get("params") or {}
                self.start_video_recording(
                    run_id=run_id,
                    checkpoint=checkpoint,
                    device=str(params.get("device") or "cuda:0"),
                    video_params=VideoParams.from_preset(DEFAULT_VIDEO_PRESET),
                )
            except Exception as exc:
                self.history.update_run(run_id, video_status="failed", video_error=str(exc))
        elif not log_dir:
            pass  # no log dir found — nothing to record

    def _monitor_video(self, source_run_id: str, video_id: str, proc: subprocess.Popen) -> None:
        returncode = proc.wait()
        run = self.history.get_run(source_run_id) or {}
        log_dir = Path(run["log_dir"]) if run.get("log_dir") else None
        video = latest_video(log_dir) if log_dir and log_dir.exists() else None
        if video:
            video = self._tag_video_with_checkpoint(Path(video), video_id, run)
        self.history.update_run(
            source_run_id,
            video_status="completed" if returncode == 0 and video else "failed",
            video_returncode=returncode,
            video_process_id=video_id,
            latest_video=video,
            has_video=bool(video),
            video_error=None if returncode == 0 and video else "Video process finished but no MP4 was produced.",
        )

    @staticmethod
    def _checkpoint_iteration(checkpoint: str | Path) -> int | None:
        match = re.search(r"model_(\d+)\.pt$", str(checkpoint or ""))
        return int(match.group(1)) if match else None

    def _tag_video_with_checkpoint(self, video: Path, video_id: str, run: dict) -> str:
        iteration = run.get("video_checkpoint_iteration") or self._checkpoint_iteration(str(run.get("video_checkpoint") or ""))
        if not iteration or not video.is_file():
            return str(video)
        if f"model_{iteration}" in video.name:
            return str(video)
        safe_video_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", video_id).strip("._-") or "video"
        target = video.with_name(f"model_{iteration}_{safe_video_id}{video.suffix}")
        counter = 2
        while target.exists() and target != video:
            target = video.with_name(f"model_{iteration}_{safe_video_id}_{counter}{video.suffix}")
            counter += 1
        try:
            video.rename(target)
            return str(target)
        except OSError:
            return str(video)

    def _monitor_onnx(self, source_run_id: str, onnx_id: str, proc: subprocess.Popen) -> None:
        returncode = proc.wait()
        run = self.history.get_run(source_run_id) or {}
        log_dir = Path(run["log_dir"]) if run.get("log_dir") else None
        onnx_path = latest_onnx(log_dir) if log_dir and log_dir.exists() else None
        self.history.update_run(
            source_run_id,
            onnx_status="completed" if returncode == 0 and onnx_path else "failed",
            onnx_returncode=returncode,
            onnx_process_id=onnx_id,
            onnx_path=onnx_path,
            has_onnx=bool(onnx_path),
            onnx_error=None if returncode == 0 and onnx_path else "ONNX export finished but policy.onnx was not produced.",
        )

    @staticmethod
    def _force_stop_after_grace(proc: subprocess.Popen, process_group: int, tmux_session: str | None = None) -> None:
        try:
            proc.wait(timeout=5)
            return
        except subprocess.TimeoutExpired:
            pass
        if tmux_session:
            ProcessRegistry._kill_tmux_session(tmux_session)
            try:
                proc.wait(timeout=3)
                return
            except subprocess.TimeoutExpired:
                pass
        for sig in (signal.SIGTERM, signal.SIGKILL):
            if proc.poll() is not None:
                return
            try:
                os.killpg(process_group, sig)
            except ProcessLookupError:
                return
            try:
                proc.wait(timeout=3)
                return
            except subprocess.TimeoutExpired:
                continue

    @staticmethod
    def _force_kill_group_after_grace(process_group: int) -> None:
        time.sleep(5)
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(process_group, 0)
            except ProcessLookupError:
                return
            try:
                os.killpg(process_group, sig)
            except ProcessLookupError:
                return
            time.sleep(3)

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
        attach_command: str | None = None,
        tmux_session: str | None = None,
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
            "attach_command": attach_command,
            "tmux_session": tmux_session,
        }
