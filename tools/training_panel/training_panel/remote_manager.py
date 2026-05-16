from __future__ import annotations

import os
import re
import shlex
import shutil
import signal
import subprocess
import time
from pathlib import Path
from typing import Callable, Mapping

from .config import PanelPaths
from .history import tail_file
from .remote_config import RemoteConfig, RemoteStateStore, parse_bool


REMOTE_WORKER_SESSION = "redrhex_remote_worker"
VALID_WORKER_MODES = {"tmux", "child"}
REMOTE_WEB_URL = "https://popcorn-volcano.github.io/redrhex-training-remote/"
DEFAULT_ENV_FILE = Path.home() / ".redrhex_remote.env"


def parse_env_file(path: Path) -> dict[str, str]:
    """Parse simple shell-style KEY=value exports without executing the file."""
    if not path.exists() or not path.is_file():
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            parts = shlex.split(line, comments=True, posix=True)
        except ValueError:
            continue
        if not parts:
            continue
        if parts[0] == "export":
            parts = parts[1:]
        if len(parts) != 1 or "=" not in parts[0]:
            continue
        key, value = parts[0].split("=", 1)
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            values[key] = value
    return values


class RemoteWorkerManager:
    def __init__(
        self,
        paths: PanelPaths,
        state_store: RemoteStateStore,
        *,
        env_file: Path | None = None,
        environment: Mapping[str, str] | None = None,
        run_command: Callable[..., subprocess.CompletedProcess] = subprocess.run,
        popen_factory: Callable[..., subprocess.Popen] = subprocess.Popen,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.paths = paths
        self.state_store = state_store
        self.env_file = env_file or DEFAULT_ENV_FILE
        self.environment = environment if environment is not None else os.environ
        self.run_command = run_command
        self.popen_factory = popen_factory
        self.sleep = sleep

    @property
    def log_file(self) -> Path:
        return self.paths.panel_log_root / "remote_worker.log"

    @property
    def pid_file(self) -> Path:
        return self.paths.panel_log_root / "remote_worker.pid"

    def merged_env(self) -> dict[str, str]:
        values = parse_env_file(self.env_file)
        for key, value in self.environment.items():
            if key.startswith("REDRHEX_"):
                values[key] = value
        return values

    def config(self) -> RemoteConfig:
        return RemoteConfig.from_env(self.merged_env())

    def saved_mode(self) -> str:
        mode = str(self.state_store.load().get("worker_mode") or "tmux")
        return mode if mode in VALID_WORKER_MODES else "tmux"

    def autostart_enabled(self) -> bool:
        return parse_bool(self.state_store.load().get("worker_autostart"), default=False)

    def save_settings(self, updates: dict) -> dict:
        safe_updates: dict = {}
        if "accept_jobs" in updates:
            safe_updates["accept_jobs"] = bool(updates["accept_jobs"])
        if "worker_autostart" in updates:
            safe_updates["worker_autostart"] = bool(updates["worker_autostart"])
        if "worker_mode" in updates:
            safe_updates["worker_mode"] = self._normalize_mode(str(updates["worker_mode"] or ""))
        if not safe_updates:
            raise ValueError("No remote setting was provided")
        return self.state_store.save(safe_updates)

    def status(self) -> dict:
        config = self.config()
        state = self.state_store.load()
        runtime = self.runtime_status()
        log_file = runtime.get("log_file") or str(self.log_file)
        output = tail_file(Path(log_file), max_chars=30000) if log_file else ""
        if runtime.get("mode") == "tmux":
            pane_output = self._capture_tmux_pane()
            if pane_output.strip():
                output = pane_output
        return {
            **config.public_status(self.paths, self.state_store),
            "missing_required_env": config.missing_required_env,
            "env_file_path": str(self.env_file),
            "env_file_exists": self.env_file.exists(),
            "worker_mode": self.saved_mode(),
            "worker_autostart": self.autostart_enabled(),
            "worker_running": runtime["running"],
            "worker_runtime_mode": runtime.get("mode"),
            "worker_pid": runtime.get("pid"),
            "worker_tmux_session": runtime.get("tmux_session"),
            "worker_attach_command": runtime.get("attach_command"),
            "worker_log_file": log_file,
            "worker_output_tail": output,
            "worker_last_error": state.get("worker_last_error", ""),
            "remote_web_url": self.environment.get("REDRHEX_REMOTE_WEB_URL", REMOTE_WEB_URL),
            "setup_checks": self.setup_checks(config),
        }

    def setup_checks(self, config: RemoteConfig | None = None) -> list[dict]:
        config = config or self.config()
        return [
            {"id": "env_file", "label": "Private env file", "ok": self.env_file.exists(), "detail": str(self.env_file)},
            {"id": "supabase_url", "label": "Supabase URL", "ok": bool(config.supabase_url), "detail": config.supabase_url or "missing"},
            {"id": "anon_key", "label": "anon / publishable key", "ok": bool(config.supabase_anon_key), "detail": "configured" if config.supabase_anon_key else "missing"},
            {"id": "machine_token", "label": "machine token", "ok": bool(config.machine_token), "detail": "configured" if config.machine_token else "missing"},
            {"id": "machine_id", "label": "Machine ID", "ok": bool(config.machine_id), "detail": config.machine_id or "missing"},
        ]

    def runtime_status(self) -> dict:
        if self._tmux_running():
            return {
                "running": True,
                "mode": "tmux",
                "pid": None,
                "tmux_session": REMOTE_WORKER_SESSION,
                "attach_command": f"tmux attach -t {REMOTE_WORKER_SESSION}",
                "log_file": str(self.log_file),
            }
        pid = self._child_pid()
        if pid and self._pid_is_remote_worker(pid):
            return {
                "running": True,
                "mode": "child",
                "pid": pid,
                "tmux_session": None,
                "attach_command": None,
                "log_file": str(self.log_file),
            }
        return {
            "running": False,
            "mode": None,
            "pid": None,
            "tmux_session": None,
            "attach_command": None,
            "log_file": str(self.log_file),
        }

    def start(self, mode: str | None = None) -> dict:
        selected_mode = self._normalize_mode(mode or self.saved_mode())
        config = self.config()
        if not config.configured:
            message = "Remote config is incomplete. Missing: " + ", ".join(config.missing_required_env)
            self.state_store.save({"worker_last_error": message, "worker_mode": selected_mode})
            raise ValueError(message)
        if self.runtime_status()["running"]:
            raise ValueError("Remote worker is already running")
        self.paths.ensure_dirs()
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.log_file.touch(exist_ok=True)
        try:
            if selected_mode == "tmux":
                self._start_tmux()
            else:
                self._start_child()
            self.state_store.save(
                {
                    "worker_mode": selected_mode,
                    "worker_last_error": "",
                    "worker_log_file": str(self.log_file),
                }
            )
        except Exception as exc:
            self.state_store.save({"worker_mode": selected_mode, "worker_last_error": str(exc)})
            raise
        return self.status()

    def stop(self) -> dict:
        runtime = self.runtime_status()
        if not runtime["running"]:
            return self.status()
        try:
            if runtime.get("mode") == "tmux":
                self._stop_tmux()
            else:
                self._stop_child(int(runtime["pid"]))
            self.state_store.save({"worker_last_error": ""})
        except Exception as exc:
            self.state_store.save({"worker_last_error": str(exc)})
            raise
        return self.status()

    def restart(self, mode: str | None = None) -> dict:
        selected_mode = self._normalize_mode(mode or self.saved_mode())
        if self.runtime_status()["running"]:
            self.stop()
        return self.start(selected_mode)

    def autostart_if_enabled(self) -> None:
        if not self.autostart_enabled() or self.runtime_status()["running"]:
            return
        if not self.config().configured:
            self.state_store.save({"worker_last_error": "Auto-start skipped: remote config is incomplete"})
            return
        try:
            self.start(self.saved_mode())
        except Exception as exc:
            self.state_store.save({"worker_last_error": f"Auto-start failed: {exc}"})

    def _normalize_mode(self, mode: str) -> str:
        normalized = mode.strip().lower()
        if normalized not in VALID_WORKER_MODES:
            raise ValueError("worker_mode must be 'tmux' or 'child'")
        return normalized

    def _worker_shell(self) -> str:
        env_file = shlex.quote(str(self.env_file))
        repo_root = shlex.quote(str(self.paths.repo_root))
        return " && ".join(
            [
                f"cd {repo_root}",
                f"if [ -f {env_file} ]; then source {env_file}; fi",
                "exec python -u -m tools.training_panel.remote_worker",
            ]
        )

    def _start_tmux(self) -> None:
        tmux = shutil.which("tmux")
        if not tmux:
            raise RuntimeError("tmux is not installed; use child process mode instead")
        log_file = shlex.quote(str(self.log_file))
        shell = self._worker_shell().replace("exec python", "python")
        inner = f"{shell} 2>&1 | tee -a {log_file}"
        result = self.run_command(
            [tmux, "new-session", "-d", "-s", REMOTE_WORKER_SESSION, "--", "bash", "-lc", inner],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError((result.stderr or "tmux failed to start remote worker").strip())

    def _start_child(self) -> None:
        log_handle = self.log_file.open("a", encoding="utf-8")
        try:
            proc = self.popen_factory(
                ["bash", "-lc", self._worker_shell()],
                cwd=self.paths.repo_root,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        finally:
            log_handle.close()
        self.pid_file.write_text(str(proc.pid), encoding="utf-8")
        self.state_store.save({"worker_child_pid": proc.pid})

    def _stop_tmux(self) -> None:
        self.run_command(["tmux", "send-keys", "-t", REMOTE_WORKER_SESSION, "C-c"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        deadline = time.time() + 3.0
        while time.time() < deadline:
            if not self._tmux_running():
                return
            self.sleep(0.2)
        self.run_command(["tmux", "kill-session", "-t", REMOTE_WORKER_SESSION], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)

    def _stop_child(self, pid: int) -> None:
        try:
            process_group = os.getpgid(pid)
            os.killpg(process_group, signal.SIGINT)
        except ProcessLookupError:
            return
        deadline = time.time() + 3.0
        while time.time() < deadline:
            if not self._pid_is_remote_worker(pid):
                return
            self.sleep(0.2)
        try:
            os.killpg(process_group, signal.SIGTERM)
        except ProcessLookupError:
            return

    def _tmux_running(self) -> bool:
        if not shutil.which("tmux"):
            return False
        result = self.run_command(
            ["tmux", "has-session", "-t", REMOTE_WORKER_SESSION],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return result.returncode == 0

    def _capture_tmux_pane(self) -> str:
        if not self._tmux_running():
            return ""
        result = self.run_command(
            ["tmux", "capture-pane", "-pt", REMOTE_WORKER_SESSION, "-S", "-120"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )
        return result.stdout if result.returncode == 0 else ""

    def _child_pid(self) -> int | None:
        state_pid = self.state_store.load().get("worker_child_pid")
        raw_pid = state_pid
        if self.pid_file.exists():
            raw_pid = self.pid_file.read_text(encoding="utf-8").strip()
        try:
            return int(raw_pid)
        except (TypeError, ValueError):
            return None

    def _pid_is_remote_worker(self, pid: int) -> bool:
        proc_cmdline = Path("/proc") / str(pid) / "cmdline"
        if proc_cmdline.exists():
            try:
                command = proc_cmdline.read_text(encoding="utf-8", errors="replace").replace("\x00", " ")
            except OSError:
                return False
            return "tools.training_panel.remote_worker" in command
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        return True
