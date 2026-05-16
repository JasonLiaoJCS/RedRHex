from __future__ import annotations

import json
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import PanelPaths


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


def _actor_label(event: dict[str, Any]) -> str:
    return str(
        event.get("actor_name")
        or event.get("actor_email")
        or event.get("actor_id")
        or event.get("actor_role")
        or event.get("actor")
        or "Local panel"
    )


def _event_sort_key(event: dict[str, Any]) -> str:
    return str(event.get("created_at") or "")


class ActivityStore:
    def __init__(self, paths: PanelPaths):
        self.paths = paths
        self.paths.ensure_dirs()

    def record(
        self,
        event_type: str,
        *,
        summary: str = "",
        subject_id: str = "",
        actor_name: str = "Local panel",
        source: str = "local",
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        event = {
            "id": str(uuid.uuid4()),
            "created_at": _now_iso(),
            "source": source,
            "event_type": str(event_type),
            "summary": summary or str(event_type).replace("_", " ").title(),
            "subject_id": subject_id,
            "actor_name": actor_name,
            "payload": payload or {},
        }
        self.paths.ensure_dirs()
        with self.paths.activity_file.open("a", encoding="utf-8") as file:
            file.write(json.dumps(event, ensure_ascii=False) + "\n")
        return event

    def local_events(self, limit: int = 80) -> list[dict[str, Any]]:
        events = _read_jsonl(self.paths.activity_file)
        return sorted(events, key=_event_sort_key, reverse=True)[:limit]

    def remote_events(self, limit: int = 80) -> list[dict[str, Any]]:
        try:
            from .remote_config import RemoteConfig
            from .supabase_client import SupabaseClient

            config = RemoteConfig.from_env()
            if not config.configured:
                return []
            client = SupabaseClient(config, timeout=5.0)
            jobs = client.select(
                "jobs",
                query={
                    "select": "*",
                    "order": "created_at.desc",
                    "limit": str(limit),
                },
            )
        except Exception:
            return []
        try:
            profiles = client.select("profiles", query={"select": "id,email,display_name,role"})
        except Exception:
            profiles = []

        if not isinstance(jobs, list):
            return []
        profile_rows = profiles if isinstance(profiles, list) else []
        profile_by_id = {
            str(profile.get("id")): profile
            for profile in profile_rows
            if isinstance(profile, dict) and profile.get("id")
        }
        events = []
        for job in jobs:
            if not isinstance(job, dict):
                continue
            actor_id = str(job.get("actor_id") or "")
            profile = profile_by_id.get(actor_id, {})
            actor_name = str(profile.get("display_name") or profile.get("email") or actor_id or job.get("actor_role") or "Remote member")
            job_type = str(job.get("type") or "job")
            run_id = str((job.get("payload") or {}).get("run_id") or (job.get("result") or {}).get("local_run_id") or "")
            events.append(
                {
                    "id": f"remote-job-{job.get('id')}",
                    "created_at": str(job.get("created_at") or job.get("updated_at") or ""),
                    "source": "remote",
                    "event_type": f"remote_{job_type}",
                    "summary": f"{actor_name} requested {job_type.replace('_', ' ')}",
                    "subject_id": run_id or str(job.get("id") or ""),
                    "actor_id": actor_id,
                    "actor_name": actor_name,
                    "actor_role": str(job.get("actor_role") or ""),
                    "status": str(job.get("status") or ""),
                    "payload": {
                        "job_id": job.get("id"),
                        "job_type": job_type,
                        "status": job.get("status"),
                        "run_id": run_id,
                    },
                }
            )
        return events

    def list_events(self, limit: int = 80, include_remote: bool = True) -> list[dict[str, Any]]:
        local = self.local_events(limit=limit)
        remote = self.remote_events(limit=limit) if include_remote else []
        return sorted([*local, *remote], key=_event_sort_key, reverse=True)[:limit]

    def analytics(self, events: list[dict[str, Any]]) -> dict[str, Any]:
        by_actor = Counter(_actor_label(event) for event in events)
        by_type = Counter(str(event.get("event_type") or "unknown") for event in events)
        profile_usage = Counter()
        for event in events:
            payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
            for key in ("reward_preset_id", "terrain_preset_id"):
                value = payload.get(key)
                if value:
                    profile_usage[str(value)] += 1
        return {
            "total_events": len(events),
            "requests_by_member": by_actor.most_common(10),
            "action_counts": by_type.most_common(12),
            "run_starts": sum(1 for event in events if event.get("event_type") in {"training_start", "remote_start_training"}),
            "deletes": sum(1 for event in events if "delete" in str(event.get("event_type") or "")),
            "most_used_profiles": profile_usage.most_common(10),
        }

    def snapshot(self, limit: int = 80, include_remote: bool = True) -> dict[str, Any]:
        events = self.list_events(limit=limit, include_remote=include_remote)
        return {"events": events, "analytics": self.analytics(events)}
