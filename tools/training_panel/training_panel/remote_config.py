from __future__ import annotations

import json
import os
import socket
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

from tools.training_panel import __version__

from .config import PanelPaths


REMOTE_JOB_TYPES = {
    "start_training",
    "stop_process",
    "record_video",
    "export_onnx",
    "compact_run",
    "delete_run",
}

ROLE_PERMISSIONS = {
    "viewer": set(),
    "operator": {"start_training", "stop_process", "record_video", "export_onnx"},
    "admin": set(REMOTE_JOB_TYPES),
}


def parse_bool(value: str | bool | None, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on", "enabled"}


def role_allows(role: str, job_type: str) -> bool:
    normalized = str(role or "viewer").strip().lower()
    return job_type in ROLE_PERMISSIONS.get(normalized, set())


@dataclass(frozen=True)
class RemoteConfig:
    supabase_url: str = ""
    supabase_anon_key: str = ""
    machine_token: str = ""
    machine_id: str = ""
    accept_jobs: bool = False
    cloudflare_tunnel_host: str = ""
    discord_webhook_url: str = ""
    resend_api_key: str = ""
    poll_interval_seconds: float = 2.0
    sync_interval_seconds: float = 5.0

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "RemoteConfig":
        source = env or os.environ
        machine_id = source.get("REDRHEX_MACHINE_ID") or socket.gethostname()
        return cls(
            supabase_url=source.get("REDRHEX_SUPABASE_URL", "").rstrip("/"),
            supabase_anon_key=source.get("REDRHEX_SUPABASE_ANON_KEY", ""),
            machine_token=source.get("REDRHEX_SUPABASE_MACHINE_TOKEN", ""),
            machine_id=machine_id,
            accept_jobs=parse_bool(source.get("REDRHEX_REMOTE_ACCEPT_JOBS"), default=False),
            cloudflare_tunnel_host=source.get("REDRHEX_CLOUDFLARE_TUNNEL_HOST", "").rstrip("/"),
            discord_webhook_url=source.get("REDRHEX_DISCORD_WEBHOOK_URL", ""),
            resend_api_key=source.get("REDRHEX_RESEND_API_KEY", ""),
            poll_interval_seconds=float(source.get("REDRHEX_REMOTE_POLL_SECONDS", "2")),
            sync_interval_seconds=float(source.get("REDRHEX_REMOTE_SYNC_SECONDS", "5")),
        )

    @property
    def configured(self) -> bool:
        return bool(self.supabase_url and self.supabase_anon_key and self.machine_token and self.machine_id)

    @property
    def missing_required_env(self) -> list[str]:
        missing = []
        if not self.supabase_url:
            missing.append("REDRHEX_SUPABASE_URL")
        if not self.supabase_anon_key:
            missing.append("REDRHEX_SUPABASE_ANON_KEY")
        if not self.machine_token:
            missing.append("REDRHEX_SUPABASE_MACHINE_TOKEN")
        if not self.machine_id:
            missing.append("REDRHEX_MACHINE_ID")
        return missing

    def public_status(self, paths: PanelPaths, state: "RemoteStateStore | None" = None) -> dict:
        accept_jobs = state.effective_accept_jobs(self) if state else self.accept_jobs
        return {
            "version": __version__,
            "machine_id": self.machine_id,
            "configured": self.configured,
            "accept_jobs": accept_jobs,
            "supabase_url": self.supabase_url,
            "supabase_anon_key_configured": bool(self.supabase_anon_key),
            "machine_token_configured": bool(self.machine_token),
            "cloudflare_tunnel_host": self.cloudflare_tunnel_host,
            "discord_configured": bool(self.discord_webhook_url),
            "email_configured": bool(self.resend_api_key),
            "worker_command": "python -m tools.training_panel.remote_worker",
            "cloudflare_tunnel_command": self.tunnel_command(),
            "remote_state_file": str(paths.remote_state_file),
        }

    def tunnel_command(self) -> str:
        if not self.cloudflare_tunnel_host:
            return "cloudflared tunnel --url http://127.0.0.1:8080"
        return f"cloudflared tunnel --url http://127.0.0.1:8080 --hostname {self.cloudflare_tunnel_host}"


class RemoteStateStore:
    def __init__(self, path: Path):
        self.path = path

    def load(self) -> dict:
        if not self.path.exists():
            return {}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
        return data if isinstance(data, dict) else {}

    def save(self, updates: dict) -> dict:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = self.load()
        data.update(updates)
        data["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return data

    def effective_accept_jobs(self, config: RemoteConfig) -> bool:
        data = self.load()
        if "accept_jobs" in data:
            return parse_bool(data["accept_jobs"], default=config.accept_jobs)
        return config.accept_jobs


def heartbeat_payload(
    config: RemoteConfig,
    paths: PanelPaths,
    *,
    active_job_id: str | None = None,
    queue_depth: int = 0,
    gpu_locked: bool = False,
    accept_jobs: bool | None = None,
) -> dict:
    return {
        "machine_id": config.machine_id,
        "online": True,
        "panel_version": __version__,
        "repo_root": str(paths.repo_root),
        "rsl_rl_log_root": str(paths.rsl_rl_log_root),
        "active_job_id": active_job_id,
        "queue_depth": int(queue_depth),
        "gpu_locked": bool(gpu_locked),
        "accept_jobs": config.accept_jobs if accept_jobs is None else bool(accept_jobs),
        "tunnel_host": config.cloudflare_tunnel_host,
        "heartbeat_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
