from __future__ import annotations

from pathlib import Path

from tools.training_panel.training_panel.activity import ActivityStore, score_activity_event
from tools.training_panel.training_panel.config import PanelPaths


def make_paths(root: Path) -> PanelPaths:
    return PanelPaths(
        repo_root=root,
        isaaclab_root=root / "IsaacLab",
        isaacsim_root=root / "isaacsim",
        conda_sh=root / "conda.sh",
        conda_env="env",
    )


def test_activity_store_records_events_and_analytics(tmp_path):
    store = ActivityStore(make_paths(tmp_path))
    store.record(
        "training_start",
        summary="Started training run_one",
        subject_id="run_one",
        payload={"reward_preset_id": "speed-focus", "terrain_preset_id": "rough"},
    )
    store.record("bulk_run_delete", summary="Deleted runs", payload={"run_ids": ["run_one"]})

    snapshot = store.snapshot(include_remote=False)

    assert len(snapshot["events"]) == 2
    assert snapshot["analytics"]["run_starts"] == 1
    assert snapshot["analytics"]["deletes"] == 1
    assert ["speed-focus", 1] in [list(item) for item in snapshot["analytics"]["most_used_profiles"]]


def test_activity_scoring_rules():
    assert score_activity_event("start_training", outcome="completed", category="training", job_type="start_training") == 10
    assert score_activity_event("start_training", outcome="failed", category="training", job_type="start_training") == 2
    assert score_activity_event("record_video", outcome="completed", category="artifact", job_type="record_video") == 4
    assert score_activity_event("reward_preset_save", outcome="info", category="preset") == 3
    assert score_activity_event("run_metadata_save", outcome="info", category="metadata") == 1
    assert score_activity_event("delete_run", outcome="completed", category="admin", job_type="delete_run") == 0


def test_activity_snapshot_filters_window(tmp_path):
    paths = make_paths(tmp_path)
    paths.ensure_dirs()
    paths.activity_file.write_text(
        "\n".join(
            [
                '{"id":"old","created_at":"2026-01-01T00:00:00+00:00","event_type":"training_start","actor_name":"Old"}',
                '{"id":"new","created_at":"2999-01-01T00:00:00+00:00","event_type":"training_start","actor_name":"New"}',
            ]
        ),
        encoding="utf-8",
    )
    store = ActivityStore(paths)
    snapshot = store.snapshot(include_remote=False, window="7d")
    assert [event["id"] for event in snapshot["events"]] == ["new"]


def test_activity_analytics_counts_unique_training_runs_and_video_artifacts(tmp_path):
    store = ActivityStore(make_paths(tmp_path))
    events = [
        {
            "id": "queued",
            "created_at": "2999-01-01T00:00:00+00:00",
            "event_type": "remote_job_queued",
            "category": "training",
            "outcome": "queued",
            "actor_name": "Jason",
            "subject_id": "run_one",
            "metadata": {"job_type": "start_training"},
        },
        {
            "id": "done",
            "created_at": "2999-01-01T00:01:00+00:00",
            "event_type": "remote_job_completed",
            "category": "training",
            "outcome": "completed",
            "actor_name": "Jason",
            "subject_id": "run_one",
            "metadata": {"job_type": "start_training"},
            "points": 10,
        },
        {
            "id": "video",
            "created_at": "2999-01-01T00:02:00+00:00",
            "event_type": "remote_job_completed",
            "category": "artifact",
            "outcome": "completed",
            "actor_name": "Jason",
            "subject_id": "run_one",
            "metadata": {"job_type": "record_video"},
            "points": 4,
        },
    ]

    analytics = store.analytics(events)

    assert analytics["kpis"]["training_runs"] == 1
    assert analytics["leaderboard"][0]["runs"] == 1
    assert analytics["leaderboard"][0]["videos"] == 1


def test_activity_remote_events_are_enriched_with_profiles(tmp_path, monkeypatch):
    class FakeConfig:
        configured = True

    class FakeRemoteConfig:
        @classmethod
        def from_env(cls):
            return FakeConfig()

    class FakeClient:
        def __init__(self, config, timeout=5.0):
            self.config = config
            self.timeout = timeout

        def select(self, table, query=None):
            if table == "jobs":
                return [
                    {
                        "id": "job_one",
                        "type": "start_training",
                        "status": "queued",
                        "actor_id": "user_one",
                        "actor_role": "operator",
                        "payload": {"run_id": "run_one"},
                        "created_at": "2026-05-16T12:00:00+00:00",
                    }
                ]
            if table == "profiles":
                return [{"id": "user_one", "display_name": "Jacob", "email": "jacob@example.com"}]
            return []

    import tools.training_panel.training_panel.remote_config as remote_config
    import tools.training_panel.training_panel.supabase_client as supabase_client

    monkeypatch.setattr(remote_config, "RemoteConfig", FakeRemoteConfig)
    monkeypatch.setattr(supabase_client, "SupabaseClient", FakeClient)

    store = ActivityStore(make_paths(tmp_path))
    events = store.remote_events()

    assert events[0]["actor_name"] == "Jacob"
    assert events[0]["event_type"] == "remote_start_training"
    assert events[0]["subject_id"] == "run_one"
