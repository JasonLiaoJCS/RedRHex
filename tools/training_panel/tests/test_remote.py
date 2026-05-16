import tempfile
import unittest
from pathlib import Path

from tools.training_panel import __version__
from tools.training_panel.training_panel.config import PanelPaths
from tools.training_panel.training_panel.notifications import completion_event_from_run, discord_message
from tools.training_panel.training_panel.remote_config import (
    RemoteConfig,
    RemoteStateStore,
    heartbeat_payload,
    role_allows,
)
from tools.training_panel.training_panel.remote_worker import RemoteJobResult, RemoteWorker, run_artifacts


class FakeClient:
    def __init__(self, job=None):
        self.job = job
        self.heartbeats = []
        self.completed = []
        self.failed = []
        self.claim_calls = 0
        self.upserts = []

    def heartbeat(self, payload):
        self.heartbeats.append(payload)

    def claim_next_job(self, machine_id):
        self.claim_calls += 1
        job, self.job = self.job, None
        return job

    def complete_job(self, job_id, result):
        self.completed.append((job_id, result))

    def fail_job(self, job_id, message, result=None):
        self.failed.append((job_id, message, result))

    def upsert(self, table, payload, **kwargs):
        self.upserts.append((table, payload, kwargs))


class FakeExecutor:
    def __init__(self, result=None, error=None):
        self.result = result or RemoteJobResult(local_run_id="run_one", process_id="proc_one")
        self.error = error

    def gpu_locked(self):
        return False

    def execute(self, job):
        if self.error:
            raise self.error
        return self.result

    def sync_runs_payload(self):
        return [{"id": "run_one", "status": "running", "artifacts": []}]


class RemoteTests(unittest.TestCase):
    def make_paths(self, root: Path) -> PanelPaths:
        return PanelPaths(
            repo_root=root,
            isaaclab_root=root / "IsaacLab",
            isaacsim_root=root / "isaacsim",
            conda_sh=root / "conda.sh",
            conda_env="env",
        )

    def test_remote_config_from_env_and_public_status_hide_secrets(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = self.make_paths(Path(tmp))
            config = RemoteConfig.from_env(
                {
                    "REDRHEX_SUPABASE_URL": "https://example.supabase.co/",
                    "REDRHEX_SUPABASE_ANON_KEY": "anon",
                    "REDRHEX_SUPABASE_MACHINE_TOKEN": "secret",
                    "REDRHEX_MACHINE_ID": "lab-pc",
                    "REDRHEX_REMOTE_ACCEPT_JOBS": "true",
                    "REDRHEX_CLOUDFLARE_TUNNEL_HOST": "https://redrhex.example.com/",
                    "REDRHEX_DISCORD_WEBHOOK_URL": "discord-secret",
                    "REDRHEX_RESEND_API_KEY": "resend-secret",
                }
            )
            status = config.public_status(paths)
            self.assertTrue(config.configured)
            self.assertTrue(status["accept_jobs"])
            self.assertTrue(status["machine_token_configured"])
            self.assertNotIn("secret", str(status))

    def test_heartbeat_payload_reports_version_and_gpu_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = self.make_paths(Path(tmp))
            config = RemoteConfig(machine_id="lab-pc", accept_jobs=True)
            payload = heartbeat_payload(config, paths, active_job_id="job", queue_depth=2, gpu_locked=True)
            self.assertEqual(payload["panel_version"], __version__)
            self.assertEqual(payload["machine_id"], "lab-pc")
            self.assertTrue(payload["gpu_locked"])
            self.assertEqual(payload["queue_depth"], 2)

    def test_remote_roles(self):
        self.assertFalse(role_allows("viewer", "start_training"))
        self.assertTrue(role_allows("operator", "export_onnx"))
        self.assertFalse(role_allows("operator", "delete_run"))
        self.assertTrue(role_allows("admin", "delete_run"))

    def test_worker_refuses_jobs_when_remote_acceptance_is_disabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            config = RemoteConfig(machine_id="lab-pc", accept_jobs=True)
            state = RemoteStateStore(paths.remote_state_file)
            state.save({"accept_jobs": False})
            client = FakeClient(job={"id": "job_one", "type": "start_training", "actor_role": "operator"})
            worker = RemoteWorker(config, paths, client, executor=FakeExecutor(), state_store=state)
            result = worker.poll_once()
            self.assertEqual(result["status"], "disabled")
            self.assertEqual(client.claim_calls, 0)

    def test_worker_claims_and_completes_job(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            config = RemoteConfig(machine_id="lab-pc", accept_jobs=True)
            client = FakeClient(job={"id": "job_one", "type": "export_onnx", "actor_role": "operator", "payload": {}})
            worker = RemoteWorker(config, paths, client, executor=FakeExecutor())
            result = worker.poll_once()
            self.assertEqual(result["status"], "completed")
            self.assertEqual(client.completed[0][0], "job_one")

    def test_artifact_sync_records_paths(self):
        artifacts = run_artifacts(
            {
                "id": "run_one",
                "latest_checkpoint": "/tmp/model_1.pt",
                "latest_video": "/tmp/video.mp4",
                "onnx_path": "/tmp/policy.onnx",
            }
        )
        self.assertEqual({item["kind"] for item in artifacts}, {"checkpoint", "video", "onnx"})

    def test_completion_event_and_discord_payload(self):
        event = completion_event_from_run(
            {
                "id": "run_one",
                "status": "completed",
                "params": {"task": "Template-Redrhex-Direct-v0", "max_iterations": 8},
            },
            remote_url="https://example.com/run_one",
        )
        self.assertIsNotNone(event)
        self.assertEqual(event["event_type"], "training_completed")
        message = discord_message(event)
        self.assertIn("Training completed", message["content"])


if __name__ == "__main__":
    unittest.main()
