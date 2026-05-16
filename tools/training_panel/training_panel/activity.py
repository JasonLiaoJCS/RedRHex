from __future__ import annotations

import json
import uuid
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .config import PanelPaths


WINDOWS = {
    "today": timedelta(days=1),
    "7d": timedelta(days=7),
    "30d": timedelta(days=30),
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _window_cutoff(window: str) -> datetime:
    if window == "today":
        local_now = datetime.now().astimezone()
        return local_now.replace(hour=0, minute=0, second=0, microsecond=0).astimezone(timezone.utc)
    now = datetime.now(timezone.utc)
    return now - WINDOWS.get(window, WINDOWS["7d"])


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


def category_for_job_type(job_type: str) -> str:
    kind = str(job_type or "").lower()
    if kind == "start_training":
        return "training"
    if kind in {"record_video", "export_onnx"}:
        return "artifact"
    if "reward" in kind or "terrain" in kind or "preset" in kind:
        return "preset"
    if kind in {"rename_run", "update_run", "save_note", "folder_assign"} or any(token in kind for token in ("note", "folder", "rename")):
        return "metadata"
    if kind in {"compact_run", "delete_run"} or any(token in kind for token in ("delete", "compact", "cleanup")):
        return "admin"
    return "system"


def category_for_event_type(event_type: str, metadata: dict[str, Any] | None = None) -> str:
    metadata = metadata or {}
    job_type = str(metadata.get("job_type") or "")
    if job_type:
        return category_for_job_type(job_type)
    kind = str(event_type or "").lower()
    if "training" in kind or "start_training" in kind:
        return "training"
    if "video" in kind or "onnx" in kind:
        return "artifact"
    if "reward" in kind or "terrain" in kind or "preset" in kind:
        return "preset"
    if "note" in kind or "folder" in kind or "rename" in kind or "metadata" in kind:
        return "metadata"
    if "delete" in kind or "compact" in kind or "cleanup" in kind:
        return "admin"
    return "system"


def outcome_for_status(status: str) -> str:
    normalized = str(status or "").lower()
    if normalized in {"completed", "success", "succeeded"}:
        return "completed"
    if normalized in {"failed", "error"}:
        return "failed"
    if normalized in {"cancelled", "canceled", "interrupted", "stopped"}:
        return "interrupted"
    if normalized in {"queued", "claimed", "running"}:
        return normalized
    return normalized or "info"


def score_activity_event(
    event_type: str,
    *,
    outcome: str = "",
    category: str = "",
    job_type: str = "",
) -> int:
    kind = str(job_type or event_type or "").lower()
    category = str(category or category_for_job_type(kind) or "").lower()
    outcome = outcome_for_status(outcome)
    if category == "training":
        if outcome == "completed":
            return 10
        if outcome in {"failed", "interrupted"}:
            return 2
        return 0
    if kind in {"record_video", "export_onnx"} or category == "artifact":
        return 4 if outcome == "completed" and kind in {"record_video", "export_onnx"} else 0
    if category == "preset":
        return 3 if outcome in {"completed", "info", "success"} else 0
    if category == "metadata":
        return 1 if outcome in {"completed", "info", "success"} else 0
    if kind in {"tensorboard", "play", "play_start", "stop_process"} or any(token in kind for token in ("tensorboard", "play", "stop")):
        return 1 if outcome in {"completed", "info", "success"} else 0
    return 0


def _normalize_event(event: dict[str, Any]) -> dict[str, Any]:
    payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
    metadata = event.get("metadata") if isinstance(event.get("metadata"), dict) else payload
    event_type = str(event.get("event_type") or "activity")
    category = str(event.get("category") or category_for_event_type(event_type, metadata))
    outcome = outcome_for_status(str(event.get("outcome") or event.get("status") or metadata.get("status") or "info"))
    job_type = str(metadata.get("job_type") or metadata.get("type") or "")
    points = event.get("points")
    if points is None:
        points = score_activity_event(event_type, outcome=outcome, category=category, job_type=job_type)
    normalized = {
        **event,
        "source": event.get("source") or "local",
        "event_type": event_type,
        "category": category,
        "outcome": outcome,
        "points": int(points or 0),
        "payload": payload or metadata,
        "metadata": metadata,
    }
    if not normalized.get("summary"):
        actor = _actor_label(normalized)
        normalized["summary"] = f"{actor} {event_type.replace('_', ' ')}"
    return normalized


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
        payload = payload or {}
        category = category_for_event_type(event_type, payload)
        event = {
            "id": str(uuid.uuid4()),
            "created_at": _now_iso(),
            "source": source,
            "event_type": str(event_type),
            "category": category,
            "outcome": "info",
            "points": score_activity_event(str(event_type), outcome="info", category=category),
            "summary": summary or str(event_type).replace("_", " ").title(),
            "subject_id": subject_id,
            "actor_name": actor_name,
            "payload": payload,
        }
        self.paths.ensure_dirs()
        with self.paths.activity_file.open("a", encoding="utf-8") as file:
            file.write(json.dumps(event, ensure_ascii=False) + "\n")
        return event

    def local_events(self, limit: int = 80) -> list[dict[str, Any]]:
        events = [_normalize_event(event) for event in _read_jsonl(self.paths.activity_file)]
        return sorted(events, key=_event_sort_key, reverse=True)[:limit]

    def team_activity_events(
        self,
        *,
        limit: int = 200,
        window: str = "7d",
        member: str = "",
        category: str = "",
    ) -> list[dict[str, Any]]:
        try:
            from .remote_config import RemoteConfig
            from .supabase_client import SupabaseClient

            config = RemoteConfig.from_env()
            if not config.configured:
                return []
            client = SupabaseClient(config, timeout=5.0)
            query = {
                "select": "*",
                "order": "created_at.desc",
                "limit": str(limit),
                "created_at": f"gte.{_window_cutoff(window).isoformat()}",
            }
            if member:
                query["actor_id"] = f"eq.{member}"
            if category:
                query["category"] = f"eq.{category}"
            rows = client.select("team_activity_events", query=query)
        except Exception:
            return []
        if not isinstance(rows, list):
            return []
        return [
            _normalize_event(
                {
                    **row,
                    "id": f"team-activity-{row.get('id')}",
                    "source": "remote",
                    "summary": _summary_for_team_event(row),
                    "subject_id": row.get("run_id") or row.get("job_id") or "",
                    "payload": row.get("metadata") if isinstance(row.get("metadata"), dict) else {},
                }
            )
            for row in rows
            if isinstance(row, dict)
        ]

    def remote_job_events(self, limit: int = 80) -> list[dict[str, Any]]:
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
            status = str(job.get("status") or "")
            payload = job.get("payload") if isinstance(job.get("payload"), dict) else {}
            result = job.get("result") if isinstance(job.get("result"), dict) else {}
            run_id = str(payload.get("run_id") or result.get("local_run_id") or "")
            category = category_for_job_type(job_type)
            outcome = outcome_for_status(status)
            points = score_activity_event(job_type, outcome=outcome, category=category, job_type=job_type)
            events.append(
                _normalize_event(
                    {
                        "id": f"remote-job-{job.get('id')}",
                        "created_at": str(job.get("created_at") or job.get("updated_at") or ""),
                        "source": "remote",
                        "event_type": f"remote_{job_type}",
                        "category": category,
                        "outcome": outcome,
                        "points": points,
                        "summary": f"{actor_name} requested {job_type.replace('_', ' ')}",
                        "subject_id": run_id or str(job.get("id") or ""),
                        "actor_id": actor_id,
                        "actor_name": actor_name,
                        "actor_role": str(job.get("actor_role") or ""),
                        "status": status,
                        "payload": {
                            "job_id": job.get("id"),
                            "job_type": job_type,
                            "status": status,
                            "run_id": run_id,
                            **payload,
                        },
                    }
                )
            )
        return events

    def remote_events(
        self,
        limit: int = 80,
        *,
        window: str = "7d",
        member: str = "",
        category: str = "",
    ) -> list[dict[str, Any]]:
        team_events = self.team_activity_events(limit=limit, window=window, member=member, category=category)
        if team_events:
            return team_events
        return self.remote_job_events(limit=limit)

    def list_events(
        self,
        limit: int = 80,
        include_remote: bool = True,
        *,
        window: str = "7d",
        member: str = "",
        category: str = "",
    ) -> list[dict[str, Any]]:
        cutoff = _window_cutoff(window)
        local = self.local_events(limit=limit)
        remote = self.remote_events(limit=limit, window=window, member=member, category=category) if include_remote else []
        events = [event for event in [*local, *remote] if self._event_in_scope(event, cutoff, member, category)]
        return sorted(events, key=_event_sort_key, reverse=True)[:limit]

    def _event_in_scope(self, event: dict[str, Any], cutoff: datetime, member: str, category: str) -> bool:
        created_at = _parse_time(str(event.get("created_at") or ""))
        if created_at and created_at < cutoff:
            return False
        if member and str(event.get("actor_id") or "") != member and _actor_label(event) != member:
            return False
        if category and str(event.get("category") or "") != category:
            return False
        return True

    def analytics(self, events: list[dict[str, Any]]) -> dict[str, Any]:
        by_actor = Counter(_actor_label(event) for event in events)
        by_type = Counter(str(event.get("event_type") or "unknown") for event in events)
        by_category = Counter(str(event.get("category") or "system") for event in events)
        by_outcome = Counter(str(event.get("outcome") or "info") for event in events)
        profile_usage = Counter()
        member_rows: dict[str, dict[str, Any]] = {}
        member_training_keys: dict[str, set[str]] = {}
        training_keys = set()
        completed_training = set()
        failed_training = set()
        artifacts_completed = 0

        for event in events:
            payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
            metadata = event.get("metadata") if isinstance(event.get("metadata"), dict) else payload
            actor_key = str(event.get("actor_id") or _actor_label(event))
            row = member_rows.setdefault(
                actor_key,
                {
                    "actor_id": event.get("actor_id") or "",
                    "name": _actor_label(event),
                    "role": event.get("actor_role") or "",
                    "points": 0,
                    "runs": 0,
                    "completions": 0,
                    "failures": 0,
                    "videos": 0,
                    "preset_edits": 0,
                    "events": 0,
                },
            )
            row["points"] += int(event.get("points") or 0)
            row["events"] += 1
            category = str(event.get("category") or "")
            outcome = str(event.get("outcome") or "")
            run_key = str(event.get("run_id") or event.get("subject_id") or payload.get("run_id") or payload.get("job_id") or event.get("id"))
            event_kind = str(metadata.get("job_type") or event.get("event_type") or "").lower()
            if category == "training":
                training_keys.add(run_key)
                member_training_keys.setdefault(actor_key, set()).add(run_key)
                if outcome == "completed":
                    completed_training.add(run_key)
                    row["completions"] += 1
                if outcome in {"failed", "interrupted"}:
                    failed_training.add(run_key)
                    row["failures"] += 1
            if category == "artifact" and outcome == "completed":
                artifacts_completed += 1
                if "video" in event_kind:
                    row["videos"] += 1
            if category == "preset":
                row["preset_edits"] += 1
            for key in ("reward_preset_id", "terrain_preset_id"):
                value = payload.get(key) or metadata.get(key)
                if value:
                    profile_usage[str(value)] += 1

        completed_count = len(completed_training)
        terminal_training = completed_count + len(failed_training)
        success_rate = round((completed_count / terminal_training) * 100) if terminal_training else 0
        for actor_key, run_ids in member_training_keys.items():
            if actor_key in member_rows:
                member_rows[actor_key]["runs"] = len(run_ids)
        leaderboard = sorted(member_rows.values(), key=lambda item: (-int(item["points"]), str(item["name"])))[:10]
        return {
            "total_events": len(events),
            "requests_by_member": by_actor.most_common(10),
            "action_counts": by_type.most_common(12),
            "run_starts": sum(1 for event in events if event.get("event_type") in {"training_start", "remote_start_training"} or event.get("category") == "training"),
            "deletes": sum(1 for event in events if "delete" in str(event.get("event_type") or "")),
            "most_used_profiles": profile_usage.most_common(10),
            "kpis": {
                "total_points": sum(int(event.get("points") or 0) for event in events),
                "training_runs": len(training_keys),
                "success_rate": success_rate,
                "artifacts_completed": artifacts_completed,
                "active_members": len(member_rows),
            },
            "leaderboard": leaderboard,
            "action_mix": by_category.most_common(),
            "outcome_mix": by_outcome.most_common(),
            "experiment_summary": {
                "most_used_profiles": profile_usage.most_common(10),
                "action_counts": by_type.most_common(12),
                "outcomes": by_outcome.most_common(),
            },
            "recent_failures": [
                event for event in events if str(event.get("outcome") or "") in {"failed", "interrupted"}
            ][:8],
        }

    def snapshot(
        self,
        limit: int = 80,
        include_remote: bool = True,
        *,
        window: str = "7d",
        member: str = "",
        category: str = "",
    ) -> dict[str, Any]:
        events = self.list_events(limit=limit, include_remote=include_remote, window=window, member=member, category=category)
        return {
            "events": events,
            "analytics": self.analytics(events),
            "filters": {"window": window, "member": member, "category": category},
        }


def _summary_for_team_event(row: dict[str, Any]) -> str:
    actor = str(row.get("actor_name") or row.get("actor_role") or "Remote member")
    event_type = str(row.get("event_type") or "activity").replace("_", " ")
    outcome = str(row.get("outcome") or "")
    return f"{actor} {event_type}{f' ({outcome})' if outcome else ''}"
