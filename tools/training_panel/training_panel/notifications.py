from __future__ import annotations

import json
from datetime import datetime, timezone
from urllib.request import Request, urlopen

NOTIFICATION_EVENT_TYPES = {
    "training_converged",
    "training_completed",
    "training_failed",
    "video_ready",
    "test_notification",
}


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


# ---------------------------------------------------------------------------
# Edge function payloads
# ---------------------------------------------------------------------------

def _run_params(run: dict) -> dict:
    params = run.get("params") or {}
    return params if isinstance(params, dict) else {}


def requester_id_for_run(run: dict) -> str:
    params = _run_params(run)
    return str(run.get("created_by") or params.get("requester_id") or params.get("created_by") or "").strip()


def _event_time() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _base_run_payload(run: dict, remote_url: str = "") -> dict:
    params = _run_params(run)
    return {
        "display_name": run.get("display_name") or run.get("id"),
        "status": run.get("status"),
        "task": params.get("task"),
        "num_envs": params.get("num_envs"),
        "max_iterations": params.get("max_iterations"),
        "device": params.get("device"),
        "reward_preset_id": run.get("reward_preset_id") or params.get("reward_preset_id"),
        "terrain_preset_id": run.get("terrain_preset_id") or params.get("terrain_preset_id"),
        "returncode": run.get("returncode"),
        "latest_checkpoint": run.get("latest_checkpoint"),
        "latest_video": run.get("latest_video"),
        "remote_url": remote_url,
    }


def notification_event_key(event_type: str, run: dict, payload: dict | None = None) -> str:
    """Stable idempotency key for worker-to-edge notification dispatch."""
    payload = payload or {}
    run_id = str(run.get("id") or payload.get("run_id") or "")
    if event_type == "training_converged":
        return f"{run_id}:training_converged:{payload.get('iteration') or run.get('convergence_iteration') or 'unknown'}"
    if event_type == "training_completed":
        return f"{run_id}:training_completed:{run.get('returncode', 0)}"
    if event_type == "training_failed":
        return f"{run_id}:training_failed:{run.get('status') or 'failed'}:{run.get('returncode', 'unknown')}"
    if event_type == "video_ready":
        video_id = payload.get("storage_path") or payload.get("latest_video") or run.get("latest_video") or "video"
        return f"{run_id}:video_ready:{video_id}"
    if event_type == "test_notification":
        recipient = payload.get("recipient_id") or requester_id_for_run(run) or "unknown"
        return f"test_notification:{recipient}:{_event_time()}"
    return f"{run_id}:{event_type}"


def edge_event_payload(
    event_type: str,
    run: dict,
    *,
    machine_id: str = "",
    requester_id: str = "",
    payload: dict | None = None,
    event_key: str | None = None,
) -> dict:
    """Build the small event envelope consumed by the Supabase notify function."""
    if event_type not in NOTIFICATION_EVENT_TYPES:
        raise ValueError(f"Unsupported notification event type: {event_type}")
    merged_payload = payload or {}
    return {
        "event_type": event_type,
        "event_key": event_key or notification_event_key(event_type, run, merged_payload),
        "run_id": run.get("id"),
        "machine_id": machine_id or run.get("machine_id") or "",
        "requester_id": requester_id or requester_id_for_run(run),
        "created_at": _event_time(),
        "payload": merged_payload,
    }


def completion_edge_event(run: dict, *, machine_id: str = "", remote_url: str = "") -> dict | None:
    status = str(run.get("status") or "").lower()
    if status not in {"completed", "failed", "interrupted"}:
        return None
    event_type = "training_completed" if status == "completed" else "training_failed"
    payload = _base_run_payload(run, remote_url=remote_url)
    return edge_event_payload(event_type, run, machine_id=machine_id, payload=payload)


def convergence_edge_event(run: dict, *, machine_id: str = "", remote_url: str = "") -> dict | None:
    if not run.get("convergence_detected"):
        return None
    payload = {
        **_base_run_payload(run, remote_url=remote_url),
        "iteration": run.get("convergence_iteration"),
        "improvement_pct": run.get("convergence_improvement_pct"),
    }
    return edge_event_payload("training_converged", run, machine_id=machine_id, payload=payload)


def video_ready_edge_event(
    run: dict,
    artifact: dict | None = None,
    *,
    machine_id: str = "",
    remote_url: str = "",
) -> dict | None:
    artifact = artifact or {}
    storage_path = artifact.get("storage_path")
    local_path = artifact.get("local_path") or artifact.get("path")
    latest_video = storage_path or local_path or run.get("latest_video")
    if not latest_video:
        return None
    payload = {
        **_base_run_payload(run, remote_url=remote_url),
        "latest_video": latest_video,
        "storage_path": storage_path,
        "local_path": local_path,
        "bytes": artifact.get("bytes"),
    }
    return edge_event_payload("video_ready", run, machine_id=machine_id, payload=payload)


def notification_events_for_run(
    run: dict,
    *,
    machine_id: str = "",
    artifacts: list[dict] | None = None,
    remote_url: str = "",
    require_video_storage: bool = False,
) -> list[dict]:
    """Return requester-scoped notification events for a synced run.

    Runs without a requester are local mother runs; those stay in the in-panel
    history/activity surfaces and do not create external notifications.
    """
    requester_id = requester_id_for_run(run)
    if not requester_id:
        return []
    events: list[dict] = []
    for maybe_event in (
        convergence_edge_event(run, machine_id=machine_id, remote_url=remote_url),
        completion_edge_event(run, machine_id=machine_id, remote_url=remote_url),
    ):
        if maybe_event:
            maybe_event["requester_id"] = requester_id
            events.append(maybe_event)
    video_artifacts = [
        artifact for artifact in (artifacts or [])
        if str(artifact.get("kind") or "") == "video"
        and (
            artifact.get("storage_path")
            or (not require_video_storage and (artifact.get("local_path") or artifact.get("path")))
        )
    ]
    if video_artifacts:
        for artifact in video_artifacts:
            event = video_ready_edge_event(run, artifact, machine_id=machine_id, remote_url=remote_url)
            if event:
                event["requester_id"] = requester_id
                events.append(event)
    elif not require_video_storage:
        event = video_ready_edge_event(run, None, machine_id=machine_id, remote_url=remote_url)
        if event:
            event["requester_id"] = requester_id
            events.append(event)
    return events


def build_test_notification_edge_event(*, requester_id: str, machine_id: str, email: str = "") -> dict:
    payload = {"recipient_id": requester_id, "email": email, "display_name": "Notification test"}
    return edge_event_payload(
        "test_notification",
        {"id": "notification-test", "created_by": requester_id, "machine_id": machine_id},
        machine_id=machine_id,
        requester_id=requester_id,
        payload=payload,
    )


# ---------------------------------------------------------------------------
# Convergence notifications
# ---------------------------------------------------------------------------

def convergence_event(run: dict, result: "ConvergenceResult", remote_url: str = "") -> dict:
    """Build a convergence event dict, parallel to completion_event_from_run()."""
    from .convergence import ConvergenceResult  # local import avoids circular dependency
    return {
        "event_type": "training_converged",
        "run_id": run.get("id"),
        "display_name": run.get("display_name") or run.get("id"),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "payload": {
            "task": (run.get("params") or {}).get("task"),
            "iteration": result.iteration,
            "improvement_pct": result.improvement_pct,
            "window_max": result.window_max,
            "window_min": result.window_min,
            "tag": result.tag,
            "reason": result.reason,
            "remote_url": remote_url,
        },
    }


def _convergence_discord_message(event: dict) -> dict:
    payload = event.get("payload") or {}
    label = str(event.get("display_name") or event.get("run_id") or "-")
    fields = [
        {"name": "Run", "value": label, "inline": False},
        {"name": "Converged at iteration", "value": str(payload.get("iteration") or "-"), "inline": True},
        {"name": "Improvement", "value": f"{payload.get('improvement_pct', 0):.1f}%", "inline": True},
        {"name": "Reward range", "value": f"{payload.get('window_min', 0):.3f} – {payload.get('window_max', 0):.3f}", "inline": True},
    ]
    if payload.get("remote_url"):
        fields.append({"name": "Remote link", "value": str(payload["remote_url"]), "inline": False})
    return {
        "content": f"Training converged: {label}",
        "embeds": [{"title": "Training converged", "fields": fields,
                    "footer": {"text": "Training still running — video will record when it ends."}}],
    }


def _convergence_email_message(event: dict, to_email: str) -> dict:
    payload = event.get("payload") or {}
    label = str(event.get("display_name") or event.get("run_id") or "-")
    subject = f"RedRHex converged: {label}"
    lines = [
        subject,
        "",
        f"Run: {event.get('run_id')}",
        f"Task: {payload.get('task') or '-'}",
        f"Converged at iteration: {payload.get('iteration') or '-'}",
        f"Improvement over window: {payload.get('improvement_pct', 0):.1f}%",
        f"Reward range: {payload.get('window_min', 0):.3f} – {payload.get('window_max', 0):.3f}",
        "",
        "Training is still running. A video will be recorded when it ends.",
        f"Remote: {payload.get('remote_url') or '-'}",
    ]
    return {"to": to_email, "subject": subject, "text": "\n".join(lines)}


def send_convergence_notification(
    run: dict,
    result: "ConvergenceResult",
    discord_webhook: str = "",
    resend_key: str = "",
    email_to: str = "",
    email_from: str = "",
    remote_url: str = "",
) -> dict:
    """
    Send Discord and/or email notification for a convergence event.
    Returns {"discord": {ok, status}, "email": {ok, status}} for each channel attempted.
    Network errors are caught — never raises.
    """
    import os
    # Fall back to env vars if callers didn't supply values
    discord_webhook = discord_webhook or os.environ.get("REDRHEX_DISCORD_WEBHOOK_URL", "")
    resend_key = resend_key or os.environ.get("REDRHEX_RESEND_API_KEY", "")
    email_to = email_to or os.environ.get("REDRHEX_NOTIFICATION_EMAIL_TO", "")
    email_from = email_from or os.environ.get("REDRHEX_NOTIFICATION_EMAIL_FROM", "")

    event = convergence_event(run, result, remote_url=remote_url)
    results: dict = {}

    if discord_webhook:
        try:
            body = json.dumps(_convergence_discord_message(event)).encode()
            req = Request(discord_webhook, data=body, method="POST",
                          headers={"Content-Type": "application/json"})
            with urlopen(req, timeout=10) as resp:
                results["discord"] = {"ok": resp.status < 300, "status": resp.status}
        except Exception as exc:
            results["discord"] = {"ok": False, "error": str(exc)}

    if resend_key and email_to and email_from:
        try:
            msg = _convergence_email_message(event, email_to)
            body = json.dumps({
                "from": email_from, "to": [email_to],
                "subject": msg["subject"], "text": msg["text"],
            }).encode()
            req = Request("https://api.resend.com/emails", data=body, method="POST",
                          headers={"Authorization": f"Bearer {resend_key}",
                                   "Content-Type": "application/json"})
            with urlopen(req, timeout=10) as resp:
                results["email"] = {"ok": resp.status < 300, "status": resp.status}
        except Exception as exc:
            results["email"] = {"ok": False, "error": str(exc)}

    return results
