from __future__ import annotations

import json
from datetime import datetime, timezone
from urllib.request import Request, urlopen


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
