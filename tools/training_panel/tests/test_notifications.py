from __future__ import annotations

from pathlib import Path

from tools.training_panel.training_panel.notifications import (
    completion_edge_event,
    convergence_edge_event,
    notification_events_for_run,
    notification_event_key,
    requester_id_for_run,
    build_test_notification_edge_event,
    video_ready_edge_event,
)


REQUESTER_ID = "11111111-1111-4111-8111-111111111111"


def run_record(**updates):
    base = {
        "id": "run_one",
        "machine_id": "lab-pc",
        "created_by": REQUESTER_ID,
        "display_name": "Run One",
        "status": "completed",
        "returncode": 0,
        "params": {
            "task": "Template-Redrhex-Direct-v0",
            "num_envs": 64,
            "max_iterations": 100,
            "device": "cuda:0",
            "reward_preset_id": "speed",
            "terrain_preset_id": "rough",
        },
    }
    base.update(updates)
    return base


def test_requester_id_falls_back_to_params_for_synced_runs():
    run = run_record(created_by=None, params={"requester_id": REQUESTER_ID})
    assert requester_id_for_run(run) == REQUESTER_ID


def test_completion_and_failure_edge_events_include_requester_and_stable_key():
    completed = completion_edge_event(run_record(), machine_id="lab-pc", remote_url="https://child.example.com")
    failed = completion_edge_event(run_record(status="interrupted", returncode=130), machine_id="lab-pc")

    assert completed["event_type"] == "training_completed"
    assert completed["requester_id"] == REQUESTER_ID
    assert completed["event_key"] == "run_one:training_completed:0"
    assert completed["payload"]["remote_url"] == "https://child.example.com"
    assert failed["event_type"] == "training_failed"
    assert failed["event_key"] == "run_one:training_failed:interrupted:130"


def test_convergence_video_and_test_payload_builders():
    converged = convergence_edge_event(
        run_record(convergence_detected=True, convergence_iteration=42, convergence_improvement_pct=1.7),
        machine_id="lab-pc",
    )
    video = video_ready_edge_event(
        run_record(),
        {"kind": "video", "storage_path": "runs/run_one/videos/model_42.mp4", "bytes": 1234},
        machine_id="lab-pc",
    )
    test = build_test_notification_edge_event(requester_id=REQUESTER_ID, machine_id="lab-pc")

    assert converged["event_type"] == "training_converged"
    assert converged["event_key"] == "run_one:training_converged:42"
    assert video["event_type"] == "video_ready"
    assert video["payload"]["storage_path"] == "runs/run_one/videos/model_42.mp4"
    assert test["event_type"] == "test_notification"
    assert test["requester_id"] == REQUESTER_ID
    assert "email" not in test["payload"]


def test_notification_events_skip_local_mother_runs_without_requester():
    assert notification_events_for_run(run_record(created_by=None, params={}), machine_id="lab-pc") == []


def test_notification_event_keys_dedupe_repeated_worker_syncs():
    run = run_record(latest_video="/tmp/video.mp4")
    artifacts = [{"kind": "video", "storage_path": "runs/run_one/videos/video.mp4", "run_id": "run_one"}]

    first = notification_events_for_run(run, machine_id="lab-pc", artifacts=artifacts)
    second = notification_events_for_run(run, machine_id="lab-pc", artifacts=artifacts)

    assert [event["event_key"] for event in first] == [event["event_key"] for event in second]
    assert len({notification_event_key(event["event_type"], run, event["payload"]) for event in first}) == len(first)


def test_edge_function_is_discord_only():
    source = Path("tools/training_panel/supabase/functions/notify/index.ts").read_text(encoding="utf-8")

    assert "api.resend.com" not in source
    assert "REDRHEX_RESEND_API_KEY" not in source
    assert "REDRHEX_NOTIFICATION_EMAIL_FROM" not in source
    assert "REDRHEX_NOTIFICATION_EMAIL_TO" not in source
