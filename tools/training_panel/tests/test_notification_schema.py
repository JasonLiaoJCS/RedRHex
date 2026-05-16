from __future__ import annotations

from pathlib import Path


SCHEMA = Path("tools/training_panel/supabase/schema.sql").read_text(encoding="utf-8")


def test_notification_settings_are_user_scoped_and_webhooks_are_private():
    assert "user_id uuid references auth.users" in SCHEMA
    assert "discord_webhook_url text" in SCHEMA
    assert "users can read own notification settings" in SCHEMA
    assert "using (user_id = auth.uid())" in SCHEMA
    assert "notification_settings readable by authenticated users" not in SCHEMA


def test_notification_settings_support_per_machine_upsert_and_event_toggles():
    assert "idx_notification_settings_user_machine" in SCHEMA
    assert "notify_training_converged boolean" in SCHEMA
    assert "notify_training_completed boolean" in SCHEMA
    assert "notify_training_failed boolean" in SCHEMA
    assert "notify_video_ready boolean" in SCHEMA


def test_run_events_have_idempotency_and_dispatch_status_fields():
    assert "event_key text" in SCHEMA
    assert "idx_run_events_event_key_unique" in SCHEMA
    assert "recipient_id uuid references auth.users" in SCHEMA
    assert "notification_status text" in SCHEMA
    assert "channel_results jsonb" in SCHEMA
    assert "notified_at timestamptz" in SCHEMA


def test_worker_or_service_role_can_record_run_event_status():
    assert "machine can insert run events" in SCHEMA
    assert "machine can update own run events" in SCHEMA
    assert "run_events readable by authenticated users" in SCHEMA
