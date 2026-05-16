from __future__ import annotations

from datetime import datetime, timezone


def completion_event_from_run(run: dict, remote_url: str = "") -> dict | None:
    status = run.get("status")
    if status not in ("completed", "failed", "interrupted"):
        return None
    event_type = "training_completed" if status == "completed" else "training_failed"
    return {
        "event_type": event_type,
        "run_id": run.get("id"),
        "status": status,
        "display_name": run.get("display_name") or run.get("id"),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "payload": {
            "task": (run.get("params") or {}).get("task"),
            "num_envs": (run.get("params") or {}).get("num_envs"),
            "max_iterations": (run.get("params") or {}).get("max_iterations"),
            "returncode": run.get("returncode"),
            "latest_checkpoint": run.get("latest_checkpoint"),
            "latest_video": run.get("latest_video"),
            "onnx_path": run.get("onnx_path"),
            "remote_url": remote_url,
        },
    }


def discord_message(event: dict) -> dict:
    payload = event.get("payload") or {}
    title = "Training completed" if event.get("event_type") == "training_completed" else "Training failed"
    fields = [
        {"name": "Run", "value": str(event.get("display_name") or event.get("run_id") or "-"), "inline": False},
        {"name": "Status", "value": str(event.get("status") or "-"), "inline": True},
        {"name": "Task", "value": str(payload.get("task") or "-"), "inline": True},
        {"name": "Iterations", "value": str(payload.get("max_iterations") or "-"), "inline": True},
    ]
    if payload.get("remote_url"):
        fields.append({"name": "Remote link", "value": str(payload["remote_url"]), "inline": False})
    return {
        "content": f"{title}: {event.get('display_name') or event.get('run_id')}",
        "embeds": [{"title": title, "fields": fields}],
    }


def email_message(event: dict, to_email: str) -> dict:
    payload = event.get("payload") or {}
    subject = f"RedRHex {event.get('status')}: {event.get('display_name') or event.get('run_id')}"
    lines = [
        subject,
        "",
        f"Run: {event.get('run_id')}",
        f"Task: {payload.get('task') or '-'}",
        f"Iterations: {payload.get('max_iterations') or '-'}",
        f"Return code: {payload.get('returncode')}",
        f"Checkpoint: {payload.get('latest_checkpoint') or '-'}",
        f"Video: {payload.get('latest_video') or '-'}",
        f"ONNX: {payload.get('onnx_path') or '-'}",
        f"Remote: {payload.get('remote_url') or '-'}",
    ]
    return {
        "to": to_email,
        "subject": subject,
        "text": "\n".join(lines),
    }
