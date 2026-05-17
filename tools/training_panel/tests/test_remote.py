import tempfile
import time
import unittest
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

from tools.training_panel import __version__
from tools.training_panel.training_panel.config import PanelPaths
from tools.training_panel.training_panel.notifications import (
    completion_event_from_run,
    discord_message,
)
from tools.training_panel.training_panel.remote_config import (
    RemoteConfig,
    RemoteStateStore,
    heartbeat_payload,
    role_allows,
)
from tools.training_panel.training_panel.remote_manager import (
    REMOTE_WORKER_SESSION,
    RemoteWorkerManager,
    parse_env_file,
)
from tools.training_panel.training_panel.remote_worker import (
    RemoteJobExecutor,
    RemoteJobResult,
    RemoteWorker,
    run_artifacts,
)


class FakeClient:
    def __init__(self, job=None):
        self.job = job
        self.heartbeats = []
        self.completed = []
        self.failed = []
        self.claim_calls = 0
        self.upserts = []
        self.select_rows = []
        self.deletes = []
        self.uploads = []
        self.inserts = []
        self.updates = []
        self.function_calls = []
        self.claim_gpu_locked = []
        self.select_by_table = {}
        self.raise_on_artifacts_upsert = False
        self.raise_on_activity_insert = False

    def heartbeat(self, payload):
        self.heartbeats.append(payload)

    def claim_next_job(self, machine_id, gpu_locked=False):
        self.claim_calls += 1
        self.claim_gpu_locked.append(gpu_locked)
        job, self.job = self.job, None
        return job

    def complete_job(self, job_id, result):
        self.completed.append((job_id, result))

    def fail_job(self, job_id, message, result=None):
        self.failed.append((job_id, message, result))

    def upsert(self, table, payload, **kwargs):
        if table == "artifacts" and self.raise_on_artifacts_upsert:
            raise RuntimeError("artifact sync failed")
        self.upserts.append((table, payload, kwargs))

    def insert(self, table, payload, **kwargs):
        if table == "team_activity_events" and self.raise_on_activity_insert:
            raise RuntimeError("activity schema missing")
        self.inserts.append((table, payload, kwargs))

    def update(self, table, payload, query=None, **kwargs):
        self.updates.append((table, payload, query, kwargs))
        return [payload]

    def mark_job_running(self, job_id):
        self.updates.append(("jobs", {"status": "running"}, {"id": f"eq.{job_id}"}, {}))
        return [{"id": job_id, "status": "running"}]

    def select(self, table, query=None):
        if table in self.select_by_table:
            value = self.select_by_table[table]
            return value(query) if callable(value) else value
        return self.select_rows

    def delete(self, table, query=None):
        self.deletes.append((table, query))

    def upload_storage_object(self, bucket, object_path, file_path, **kwargs):
        self.uploads.append((bucket, object_path, file_path, kwargs))
        return {"Key": object_path}

    def function_request(self, name, payload=None):
        self.function_calls.append((name, payload or {}))
        return {"ok": True, "status": "sent"}


class FakeExecutor:
    def __init__(self, result=None, error=None, gpu_locked=False):
        self.result = result or RemoteJobResult(local_run_id="run_one", process_id="proc_one")
        self.error = error
        self._gpu_locked = gpu_locked

    def gpu_locked(self):
        return self._gpu_locked

    def execute(self, job):
        if self.error:
            raise self.error
        return self.result

    def sync_runs_payload(self):
        return [{"id": "run_one", "status": "running", "artifacts": []}]


class FakePopen:
    def __init__(self, pid=4242):
        self.pid = pid


class FakeTmuxRunner:
    def __init__(self, running=False):
        self.running = running
        self.calls = []

    def __call__(self, args, **kwargs):
        self.calls.append(args)
        command = list(args)
        if command[0].endswith("tmux") and command[1] == "has-session":
            return subprocess.CompletedProcess(args, 0 if self.running else 1, stdout="", stderr="")
        if command[0].endswith("tmux") and command[1] == "new-session":
            self.running = True
            return subprocess.CompletedProcess(args, 0, stdout="", stderr="")
        if command[0].endswith("tmux") and command[1] == "send-keys":
            self.running = False
            return subprocess.CompletedProcess(args, 0, stdout="", stderr="")
        if command[0].endswith("tmux") and command[1] == "capture-pane":
            return subprocess.CompletedProcess(args, 0, stdout="worker output", stderr="")
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")


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
        self.assertTrue(role_allows("operator", "tensorboard"))
        self.assertTrue(role_allows("operator", "compact_run"))
        self.assertTrue(role_allows("operator", "send_missed_notifications"))
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
            self.assertIn(("jobs", {"status": "running"}, {"id": "eq.job_one"}, {}), client.updates)

    def test_worker_claims_job_before_routine_sync(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            config = RemoteConfig(machine_id="lab-pc", accept_jobs=True)
            client = FakeClient(job={"id": "job_one", "type": "export_onnx", "actor_role": "operator", "payload": {}})
            worker = RemoteWorker(config, paths, client, executor=FakeExecutor())
            worker.sync_if_due = MagicMock(return_value="")
            worker.sync_runs = MagicMock(return_value={"runs_changed": 1, "artifacts": 0})

            result = worker.poll_once()

            self.assertEqual(result["status"], "completed")
            worker.sync_if_due.assert_not_called()
            worker.sync_runs.assert_called_once_with(force=True, run_ids={"run_one"}, pull_metadata=False)
            self.assertEqual(client.claim_calls, 1)

    def test_worker_syncs_when_idle_after_claim_attempt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            config = RemoteConfig(machine_id="lab-pc", accept_jobs=True)
            client = FakeClient(job=None)
            worker = RemoteWorker(config, paths, client, executor=FakeExecutor())
            worker.sync_if_due = MagicMock(return_value="")

            result = worker.poll_once()

            self.assertEqual(result["status"], "idle")
            self.assertEqual(client.claim_calls, 1)
            worker.sync_if_due.assert_called_once_with()

    def test_worker_records_team_activity_without_blocking_completion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            config = RemoteConfig(machine_id="lab-pc", accept_jobs=True)
            client = FakeClient(job={"id": "job_one", "type": "start_training", "actor_role": "operator", "payload": {}})
            worker = RemoteWorker(config, paths, client, executor=FakeExecutor())
            result = worker.poll_once()
            self.assertEqual(result["status"], "completed")
            activity_rows = [item for item in client.inserts if item[0] == "team_activity_events"]
            self.assertTrue(activity_rows)
            completed = [item[1] for item in activity_rows if item[1]["outcome"] == "completed"]
            self.assertEqual(completed[0]["points"], 10)

    def test_worker_activity_insert_failure_does_not_block_job(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            config = RemoteConfig(machine_id="lab-pc", accept_jobs=True)
            client = FakeClient(job={"id": "job_one", "type": "export_onnx", "actor_role": "operator", "payload": {}})
            client.raise_on_activity_insert = True
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
        self.assertTrue(all(item.get("local_path") for item in artifacts))

    def test_run_artifacts_include_checkpoint_and_video_inventory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            log_dir = root / "logs" / "rsl_rl" / "redrhex_wheg" / "run_one"
            video_dir = log_dir / "videos" / "play"
            video_dir.mkdir(parents=True)
            for iteration in (10, 20):
                (log_dir / f"model_{iteration}.pt").write_text("checkpoint", encoding="utf-8")
            (video_dir / "model_10_video.mp4").write_bytes(b"mp4")
            artifacts = run_artifacts({"id": "run_one", "log_dir": str(log_dir)})
            paths = {Path(item["local_path"]).name for item in artifacts}
            self.assertIn("model_10.pt", paths)
            self.assertIn("model_20.pt", paths)
            self.assertIn("model_10_video.mp4", paths)

    def test_run_artifacts_include_tensorboard_summary_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            log_dir = root / "logs" / "rsl_rl" / "redrhex_wheg" / "run_one"
            summary = log_dir / "training_panel" / "tensorboard_summary.png"
            summary.parent.mkdir(parents=True)
            summary.write_bytes(b"png")

            artifacts = run_artifacts({"id": "run_one", "log_dir": str(log_dir)})

            summary_artifacts = [item for item in artifacts if item["kind"] == "tensorboard_summary"]
            self.assertEqual(len(summary_artifacts), 1)
            self.assertEqual(summary_artifacts[0]["local_path"], str(summary))

    def test_completion_event_and_discord_payload(self):
        event = completion_event_from_run(
            {
                "id": "run_one",
                "display_name": "Sprint 01",
                "status": "completed",
                "created_at": "2026-05-17T10:00:00+00:00",
                "updated_at": "2026-05-17T10:03:12+00:00",
                "requester_label": "Jason",
                "params": {"task": "Template-Redrhex-Direct-v0", "max_iterations": 8},
            },
            remote_url="https://example.com/run_one",
        )
        self.assertIsNotNone(event)
        self.assertEqual(event["event_type"], "training_completed")
        message = discord_message(event)
        self.assertIn("Training completed", message["content"])
        self.assertNotIn("Sprint 01", message["content"])
        self.assertEqual(message["embeds"][0]["title"], "Sprint 01")
        fields = message["embeds"][0]["fields"]
        self.assertEqual([field["name"] for field in fields], ["Status", "Runtime", "Run by", "Link"])
        self.assertEqual(fields[1]["value"], "3m 12s")
        self.assertEqual(fields[2]["value"], "Jason")
        self.assertEqual(fields[3]["value"], "[Open run](https://example.com/run_one)")
        self.assertNotIn("Task", str(message))
        self.assertEqual(str(message).count("Sprint 01"), 1)

    # ------------------------------------------------------------------
    # RemoteJobExecutor job-type tests
    # ------------------------------------------------------------------

    def _make_executor(self, tmp: Path) -> RemoteJobExecutor:
        paths = self.make_paths(tmp)
        paths.ensure_dirs()
        executor = RemoteJobExecutor(paths)
        return executor

    def test_executor_rejects_unauthorized_role(self):
        with tempfile.TemporaryDirectory() as tmp:
            executor = self._make_executor(Path(tmp))
            with self.assertRaises(PermissionError):
                executor.execute({"type": "delete_run", "actor_role": "operator", "payload": {"run_id": "x"}})

    def test_executor_gpu_lock_blocks_media_job(self):
        with tempfile.TemporaryDirectory() as tmp:
            executor = self._make_executor(Path(tmp))
            executor.processes.running_isaac_processes = MagicMock(return_value=[object()])
            with self.assertRaises(RuntimeError, msg="GPU lock should block record_video"):
                executor.execute({"type": "record_video", "actor_role": "operator", "payload": {"run_id": "x"}})

    def test_executor_gpu_lock_blocks_training_job(self):
        with tempfile.TemporaryDirectory() as tmp:
            executor = self._make_executor(Path(tmp))
            executor.processes.running_isaac_processes = MagicMock(return_value=[object()])
            with self.assertRaises(RuntimeError, msg="GPU lock should block start_training"):
                executor.execute({
                    "type": "start_training",
                    "actor_role": "operator",
                    "payload": {"task": "Template-Redrhex-Direct-v0", "num_envs": 4, "max_iterations": 8},
                })

    def test_executor_stop_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            executor = self._make_executor(Path(tmp))
            executor.processes.stop = MagicMock(return_value=False)
            result = executor.execute({"type": "stop_process", "actor_role": "operator", "payload": {"process_id": "proc_1"}})
            executor.processes.stop.assert_called_once_with("proc_1")
            self.assertEqual(result.process_id, "proc_1")

    def test_executor_start_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            executor = self._make_executor(Path(tmp))
            fake_run = {"id": "run_new", "status": "running"}
            executor.processes.start_training = MagicMock(return_value=fake_run)
            result = executor.execute({
                "type": "start_training",
                "actor_role": "operator",
                "payload": {"task": "Template-Redrhex-Direct-v0", "num_envs": 64, "max_iterations": 100, "device": "cuda:0"},
            })
            self.assertTrue(executor.processes.start_training.called)
            self.assertEqual(result.local_run_id, "run_new")

    def test_executor_preserves_remote_actor_as_training_requester(self):
        with tempfile.TemporaryDirectory() as tmp:
            executor = self._make_executor(Path(tmp))
            executor.processes.start_training = MagicMock(return_value={"id": "run_new", "status": "running"})
            actor_id = "11111111-1111-4111-8111-111111111111"

            executor.execute({
                "type": "start_training",
                "actor_role": "operator",
                "actor_id": actor_id,
                "payload": {"task": "Template-Redrhex-Direct-v0", "num_envs": 64, "max_iterations": 100, "device": "cuda:0"},
            })

            params = executor.processes.start_training.call_args.args[0]
            self.assertEqual(params.requester_id, actor_id)

    def test_executor_uses_payload_requester_when_actor_id_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            executor = self._make_executor(Path(tmp))
            executor.processes.start_training = MagicMock(return_value={"id": "run_new", "status": "running"})
            requester_id = "22222222-2222-4222-8222-222222222222"

            executor.execute({
                "type": "start_training",
                "actor_role": "operator",
                "payload": {
                    "task": "Template-Redrhex-Direct-v0",
                    "num_envs": 64,
                    "max_iterations": 100,
                    "device": "cuda:0",
                    "requester_id": requester_id,
                    "requester_label": "phone user",
                },
            })

            params = executor.processes.start_training.call_args.args[0]
            self.assertEqual(params.requester_id, requester_id)
            self.assertEqual(params.requester_label, "phone user")

    def test_executor_preserves_launch_folder_and_client_request_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            executor = self._make_executor(Path(tmp))
            executor.processes.start_training = MagicMock(return_value={"id": "run_new", "status": "running"})

            executor.execute({
                "type": "start_training",
                "actor_role": "operator",
                "payload": {
                    "task": "Template-Redrhex-Direct-v0",
                    "num_envs": 64,
                    "max_iterations": 100,
                    "device": "cuda:0",
                    "display_name": "Launch A",
                    "folder": "tests",
                    "client_request_id": "child-123",
                },
            })

            params = executor.processes.start_training.call_args.args[0]
            self.assertEqual(params.display_name, "Launch A")
            self.assertEqual(params.folder, "tests")
            self.assertEqual(params.client_request_id, "child-123")

    def test_executor_compact_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            executor = self._make_executor(Path(tmp))
            executor.processes.running_for_run = MagicMock(return_value=[])
            executor.history.compact_run = MagicMock(return_value={"deleted": 0})
            result = executor.execute({"type": "compact_run", "actor_role": "operator", "payload": {"run_id": "run_x"}})
            executor.history.compact_run.assert_called_once()
            self.assertEqual(result.local_run_id, "run_x")

    def test_executor_starts_tensorboard_for_run_log_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            executor = self._make_executor(root)
            log_dir = root / "logs" / "rsl_rl" / "redrhex_wheg" / "run_x"
            log_dir.mkdir(parents=True)
            executor.history.add_run({"id": "run_x", "latest_checkpoint": str(log_dir / "model_1.pt"), "log_dir": str(log_dir)})
            executor.processes.start_tensorboard = MagicMock(return_value={"id": "tensorboard_6006", "url": "http://127.0.0.1:6006"})
            result = executor.execute({"type": "tensorboard", "actor_role": "operator", "payload": {"run_id": "run_x", "host": "0.0.0.0"}})
            executor.processes.start_tensorboard.assert_called_once()
            self.assertEqual(executor.processes.start_tensorboard.call_args.kwargs["logdir"], log_dir)
            self.assertEqual(result.process_id, "tensorboard_6006")

    def test_executor_records_video_for_requested_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            executor = self._make_executor(root)
            log_dir = root / "logs" / "rsl_rl" / "redrhex_wheg" / "run_x"
            log_dir.mkdir(parents=True)
            checkpoint_5 = log_dir / "model_5.pt"
            checkpoint_20 = log_dir / "model_20.pt"
            checkpoint_5.write_text("checkpoint", encoding="utf-8")
            checkpoint_20.write_text("checkpoint", encoding="utf-8")
            executor.history.add_run({"id": "run_x", "latest_checkpoint": str(checkpoint_20), "log_dir": str(log_dir)})
            executor.processes.start_video_recording = MagicMock(return_value={"id": "video_1"})
            result = executor.execute({
                "type": "record_video",
                "actor_role": "operator",
                "payload": {"run_id": "run_x", "checkpoint_iteration": 5},
            })
            executor.processes.start_video_recording.assert_called_once()
            self.assertEqual(executor.processes.start_video_recording.call_args.kwargs["checkpoint"], str(checkpoint_5))
            self.assertEqual(result.payload["checkpoint_iteration"], 5)

    def test_executor_delete_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            executor = self._make_executor(Path(tmp))
            executor.processes.running_for_run = MagicMock(return_value=[])
            executor.history.delete_run = MagicMock(return_value={"deleted": True})
            result = executor.execute({"type": "delete_run", "actor_role": "admin", "payload": {"run_id": "run_x"}})
            executor.history.delete_run.assert_called_once()
            self.assertEqual(result.local_run_id, "run_x")

    # ------------------------------------------------------------------
    # RemoteWorker sync_runs + run_forever
    # ------------------------------------------------------------------

    def test_worker_sync_runs_sets_machine_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            config = RemoteConfig(machine_id="lab-pc", accept_jobs=True)
            client = FakeClient()
            worker = RemoteWorker(config, paths, client, executor=FakeExecutor())
            worker.sync_runs()
            self.assertTrue(len(client.upserts) >= 1)
            runs_upsert = next(u for u in client.upserts if u[0] == "runs")
            for run in runs_upsert[1]:
                self.assertEqual(run["machine_id"], "lab-pc")
                self.assertTrue(run["created_at"])
                self.assertTrue(run["updated_at"])

    def test_worker_immediately_syncs_launched_training_run_with_requester(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            requester_id = "11111111-1111-4111-8111-111111111111"
            config = RemoteConfig(machine_id="lab-pc", accept_jobs=True, sync_interval_seconds=999)
            client = FakeClient(job={
                "id": "job_one",
                "type": "start_training",
                "actor_role": "operator",
                "actor_id": requester_id,
                "payload": {
                    "task": "Template-Redrhex-Direct-v0",
                    "num_envs": 4,
                    "max_iterations": 8,
                    "requester_id": requester_id,
                    "requester_label": "phone user",
                },
            })
            executor = FakeExecutor(result=RemoteJobResult(
                local_run_id="panel_run",
                process_id="panel_run",
                payload={
                    "id": "panel_run",
                    "status": "running",
                    "created_at": "2026-05-17T00:00:00",
                    "params": {"task": "Template-Redrhex-Direct-v0"},
                },
            ))
            executor.sync_runs_payload = MagicMock(return_value=[])
            worker = RemoteWorker(config, paths, client, executor=executor)

            result = worker.poll_once()

            self.assertEqual(result["status"], "completed")
            run_upserts = [item for item in client.upserts if item[0] == "runs"]
            self.assertTrue(run_upserts)
            launched = run_upserts[0][1]
            self.assertEqual(launched["id"], "panel_run")
            self.assertEqual(launched["status"], "running")
            self.assertEqual(launched["created_by"], requester_id)
            self.assertEqual(launched["params"]["requester_id"], requester_id)

    def test_worker_sync_runs_preserves_created_by_and_dispatches_notifications(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "clip.mp4"
            video.write_bytes(b"mp4-data")
            paths = self.make_paths(root)
            config = RemoteConfig(machine_id="lab-pc", accept_jobs=True, cloudflare_tunnel_host="https://child.example.com")
            client = FakeClient()
            actor_id = "11111111-1111-4111-8111-111111111111"
            executor = FakeExecutor()
            executor.sync_runs_payload = MagicMock(return_value=[
                {
                    "id": "run_one",
                    "status": "completed",
                    "created_by": actor_id,
                    "params": {"task": "Template-Redrhex-Direct-v0", "max_iterations": 10, "requester_id": actor_id},
                    "latest_video": str(video),
                    "artifacts": [{"kind": "video", "run_id": "run_one", "local_path": str(video)}],
                }
            ])
            worker = RemoteWorker(config, paths, client, executor=executor)
            worker.sync_runs()

            runs_upsert = next(u for u in client.upserts if u[0] == "runs")
            self.assertEqual(runs_upsert[1][0]["created_by"], actor_id)
            event_types = [call[1]["event_type"] for call in client.function_calls]
            self.assertIn("training_completed", event_types)
            self.assertIn("video_ready", event_types)
            self.assertTrue(all(call[1]["requester_id"] == actor_id for call in client.function_calls))

    def test_worker_missed_notifications_backfills_requester_and_skips_sent_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            requester_id = "11111111-1111-4111-8111-111111111111"
            client = FakeClient(job={
                "id": "job_missed",
                "type": "send_missed_notifications",
                "actor_role": "operator",
                "actor_id": requester_id,
                "payload": {"scope": "latest", "requester_id": requester_id, "requester_label": "phone user"},
            })
            client.select_by_table["jobs"] = [{
                "id": "job_start",
                "type": "start_training",
                "machine_id": "lab-pc",
                "actor_id": requester_id,
                "payload": {"requester_id": requester_id},
                "result": {"local_run_id": "run_one"},
            }]
            client.select_by_table["notification_settings"] = [{
                "user_id": requester_id,
                "machine_id": "lab-pc",
                "discord_enabled": True,
                "notify_training_completed": True,
            }]
            client.select_by_table["run_events"] = [{
                "event_key": "run_one:training_completed:0",
                "notification_status": "sent",
            }]
            executor = FakeExecutor()
            executor.sync_runs_payload = MagicMock(return_value=[
                {
                    "id": "run_one",
                    "status": "completed",
                    "created_at": "2026-05-17T00:00:00",
                    "updated_at": "2026-05-17T00:01:00",
                    "params": {"task": "Template-Redrhex-Direct-v0"},
                    "artifacts": [],
                }
            ])
            worker = RemoteWorker(RemoteConfig(machine_id="lab-pc", accept_jobs=True), paths, client, executor=executor)

            result = worker.poll_once()

            payload = result["result"]["payload"]
            self.assertEqual(payload["skipped_sent"], 1)
            self.assertEqual(payload["sent"], 0)
            self.assertFalse(client.function_calls)
            self.assertEqual(worker.history.get_run("run_one")["created_by"], requester_id)

    def test_worker_missed_notifications_sends_unsent_latest_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            requester_id = "11111111-1111-4111-8111-111111111111"
            client = FakeClient(job={
                "id": "job_missed",
                "type": "send_missed_notifications",
                "actor_role": "operator",
                "actor_id": requester_id,
                "payload": {"scope": "latest", "requester_id": requester_id},
            })
            client.select_by_table["jobs"] = [{
                "id": "job_start",
                "type": "start_training",
                "machine_id": "lab-pc",
                "actor_id": requester_id,
                "result": {"payload": {"id": "run_two"}},
            }]
            client.select_by_table["notification_settings"] = [{
                "user_id": requester_id,
                "machine_id": "lab-pc",
                "discord_enabled": True,
                "notify_training_completed": True,
            }]
            client.select_by_table["run_events"] = []
            executor = FakeExecutor()
            executor.sync_runs_payload = MagicMock(return_value=[
                {
                    "id": "run_two",
                    "status": "completed",
                    "created_by": requester_id,
                    "created_at": "2026-05-17T00:00:00",
                    "updated_at": "2026-05-17T00:01:00",
                    "params": {"requester_id": requester_id},
                    "artifacts": [],
                }
            ])
            worker = RemoteWorker(RemoteConfig(machine_id="lab-pc", accept_jobs=True), paths, client, executor=executor)

            result = worker.poll_once()

            self.assertEqual(result["result"]["payload"]["sent"], 1)
            self.assertEqual(client.function_calls[0][1]["event_type"], "training_completed")
            self.assertEqual(client.function_calls[0][1]["requester_id"], requester_id)

    def test_worker_passes_gpu_lock_to_job_claim(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            config = RemoteConfig(machine_id="lab-pc", accept_jobs=True, sync_interval_seconds=999)
            client = FakeClient()
            worker = RemoteWorker(config, paths, client, executor=FakeExecutor(gpu_locked=True))
            worker.poll_once()
            self.assertEqual(client.claim_gpu_locked, [True])

    def test_worker_artifact_payload_keys_are_uniform(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "clip.mp4"
            video.write_bytes(b"mp4-data")
            paths = self.make_paths(root)
            config = RemoteConfig(machine_id="lab-pc", accept_jobs=True)
            client = FakeClient()
            executor = FakeExecutor()
            executor.sync_runs_payload = MagicMock(return_value=[
                {
                    "id": "run_one",
                    "status": "completed",
                    "artifacts": [
                        {"kind": "video", "run_id": "run_one", "local_path": str(video)},
                        {"kind": "onnx", "run_id": "run_one", "local_path": str(root / "policy.onnx")},
                    ],
                }
            ])
            worker = RemoteWorker(config, paths, client, executor=executor)
            worker.sync_runs()
            artifact_upsert = next(u for u in client.upserts if u[0] == "artifacts")
            key_sets = {tuple(sorted(item.keys())) for item in artifact_upsert[1]}
            self.assertEqual(len(key_sets), 1)

    def test_worker_sync_failure_does_not_block_job_completion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            onnx = root / "policy.onnx"
            onnx.write_text("onnx", encoding="utf-8")
            paths = self.make_paths(root)
            config = RemoteConfig(machine_id="lab-pc", accept_jobs=True)
            client = FakeClient(job={"id": "job_one", "type": "export_onnx", "actor_role": "operator", "payload": {}})
            client.raise_on_artifacts_upsert = True
            executor = FakeExecutor()
            executor.sync_runs_payload = MagicMock(return_value=[
                {
                    "id": "run_one",
                    "status": "completed",
                    "artifacts": [{"kind": "onnx", "run_id": "run_one", "local_path": str(onnx)}],
                }
            ])
            worker = RemoteWorker(config, paths, client, executor=executor)
            result = worker.poll_once()
            self.assertEqual(result["status"], "completed")
            self.assertEqual(client.completed[0][0], "job_one")
            self.assertIn("sync_error", client.completed[0][1]["payload"])

    def test_worker_pulls_newer_remote_run_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            paths.ensure_dirs()
            executor = FakeExecutor()
            executor.sync_runs_payload = MagicMock(return_value=[
                {"id": "run_one", "status": "completed", "updated_at": "2026-05-16T00:00:00+00:00", "artifacts": []}
            ])
            history = executor.history = RemoteJobExecutor(paths).history
            history.add_run({
                "id": "run_one",
                "display_name": "old",
                "folder": None,
                "notes": "old note",
                "updated_at": "2026-05-16T00:00:00+00:00",
            })
            config = RemoteConfig(machine_id="lab-pc", accept_jobs=True)
            client = FakeClient()
            client.select_rows = [{
                "id": "run_one",
                "display_name": "new",
                "folder": "team",
                "notes": "new note",
                "updated_at": "2026-05-16T12:00:00+00:00",
            }]
            worker = RemoteWorker(config, paths, client, executor=executor)
            worker.sync_runs()
            run = history.get_run("run_one")
            self.assertEqual(run["display_name"], "new")
            self.assertEqual(run["folder"], "team")
            self.assertEqual(history.get_note("run_one"), "new note")

    def test_worker_does_not_pull_metadata_from_remote_log_dir_alias(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            paths.ensure_dirs()
            log_dir = paths.rsl_rl_log_root / "2026_alias_log"
            log_dir.mkdir(parents=True)
            executor = FakeExecutor()
            history = executor.history = RemoteJobExecutor(paths).history
            history.patch_run_metadata("panel_canonical", log_dir=str(log_dir), source="training_panel", folder="tests")
            config = RemoteConfig(machine_id="lab-pc", accept_jobs=True)
            client = FakeClient()
            client.select_rows = [{
                "id": "2026_alias_log",
                "display_name": "stale alias",
                "folder": "weird_USD",
                "notes": "stale note",
                "log_dir": str(log_dir),
                "updated_at": "2026-05-17T12:00:00+00:00",
            }]
            worker = RemoteWorker(config, paths, client, executor=executor)

            updated = worker.pull_remote_run_metadata()

            self.assertEqual(updated, 0)
            run = history.get_run("panel_canonical")
            self.assertEqual(run["folder"], "tests")
            self.assertEqual(history.get_note("panel_canonical"), "")

    def test_worker_sync_deletes_tombstoned_remote_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            paths.ensure_dirs()
            executor = FakeExecutor()
            executor.sync_runs_payload = MagicMock(return_value=[])
            history = executor.history = RemoteJobExecutor(paths).history
            log_dir = paths.rsl_rl_log_root / "2026_deleted_run"
            log_dir.mkdir(parents=True)
            (log_dir / "events.out.tfevents.test").write_text("x", encoding="utf-8")
            history.patch_run_metadata("panel_deleted", log_dir=str(log_dir), source="training_panel")
            history.delete_run("panel_deleted", confirm=True, delete_logs=False)
            config = RemoteConfig(machine_id="lab-pc", accept_jobs=True)
            client = FakeClient()
            worker = RemoteWorker(config, paths, client, executor=executor)

            worker.sync_runs()

            self.assertIn(("runs", {"machine_id": "eq.lab-pc", "id": "eq.panel_deleted"}), client.deletes)
            self.assertIn(("runs", {"machine_id": "eq.lab-pc", "id": "eq.2026_deleted_run"}), client.deletes)
            self.assertIn(("runs", {"machine_id": "eq.lab-pc", "log_dir": f"eq.{log_dir}"}), client.deletes)
            deletion_upsert = next(item for item in client.upserts if item[0] == "run_deletions")
            self.assertEqual(deletion_upsert[1][0]["id"], "panel_deleted")
            self.assertEqual(deletion_upsert[1][0]["log_dir_name"], "2026_deleted_run")

    def test_worker_sync_removes_remote_alias_for_canonical_log_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            log_dir = paths.rsl_rl_log_root / "2026_alias_log"
            config = RemoteConfig(machine_id="lab-pc", accept_jobs=True)
            client = FakeClient()

            def runs_select(query=None):
                if query and query.get("select") == "id,display_name,folder,notes,log_dir,updated_at":
                    return []
                return [
                    {"id": "2026_alias_log", "log_dir": str(log_dir), "folder": "weird_USD"},
                    {"id": "panel_canonical", "log_dir": str(log_dir), "folder": "tests"},
                ]

            client.select_by_table["runs"] = runs_select
            executor = FakeExecutor()
            executor.sync_runs_payload = MagicMock(return_value=[
                {
                    "id": "panel_canonical",
                    "status": "completed",
                    "folder": "tests",
                    "log_dir": str(log_dir),
                    "artifacts": [],
                }
            ])
            worker = RemoteWorker(config, paths, client, executor=executor)

            summary = worker.sync_runs()

            self.assertEqual(summary["deleted_remote_alias_rows"], 1)
            self.assertIn(("runs", {"machine_id": "eq.lab-pc", "id": "eq.2026_alias_log"}), client.deletes)
            self.assertIn(("artifacts", {"machine_id": "eq.lab-pc", "run_id": "eq.2026_alias_log"}), client.deletes)
            self.assertNotIn(("runs", {"machine_id": "eq.lab-pc", "id": "eq.panel_canonical"}), client.deletes)

    def test_worker_sync_deleted_runs_can_limit_to_current_tombstone(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            paths.ensure_dirs()
            history = RemoteJobExecutor(paths).history
            old_log = paths.rsl_rl_log_root / "old_log"
            new_log = paths.rsl_rl_log_root / "new_log"
            old_log.mkdir(parents=True)
            new_log.mkdir(parents=True)
            history.patch_run_metadata("old_panel", log_dir=str(old_log), source="training_panel")
            history.patch_run_metadata("new_panel", log_dir=str(new_log), source="training_panel")
            history.delete_run("old_panel", confirm=True, delete_logs=False)
            history.delete_run("new_panel", confirm=True, delete_logs=False)
            config = RemoteConfig(machine_id="lab-pc", accept_jobs=True)
            client = FakeClient()
            worker = RemoteWorker(config, paths, client, executor=FakeExecutor())

            summary = worker.sync_deleted_runs(tombstones=history.deleted_run_tombstones(run_ids=["new_panel"]))

            self.assertEqual(summary["deleted_remote_rows"], 3)
            self.assertEqual(summary["tombstones"], 1)
            self.assertIn(("runs", {"machine_id": "eq.lab-pc", "id": "eq.new_panel"}), client.deletes)
            self.assertIn(("runs", {"machine_id": "eq.lab-pc", "id": "eq.new_log"}), client.deletes)
            self.assertNotIn(("runs", {"machine_id": "eq.lab-pc", "id": "eq.old_panel"}), client.deletes)

    def test_worker_sync_runs_skips_unchanged_runs_until_forced(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            config = RemoteConfig(machine_id="lab-pc", accept_jobs=True)
            client = FakeClient()
            executor = FakeExecutor()
            executor.sync_runs_payload = MagicMock(return_value=[
                {
                    "id": "run_one",
                    "status": "completed",
                    "updated_at": "2026-05-17T00:00:00+00:00",
                    "params": {"task": "Template-Redrhex-Direct-v0"},
                    "artifacts": [],
                }
            ])
            worker = RemoteWorker(config, paths, client, executor=executor)

            first = worker.sync_runs()
            second = worker.sync_runs()
            forced = worker.sync_runs(force=True)

            run_upserts = [item for item in client.upserts if item[0] == "runs"]
            self.assertEqual(len(run_upserts), 2)
            self.assertEqual(first["runs_changed"], 1)
            self.assertEqual(second["runs_unchanged"], 1)
            self.assertEqual(forced["runs_changed"], 1)

    def test_worker_groups_run_upserts_by_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            config = RemoteConfig(machine_id="lab-pc", accept_jobs=True)
            client = FakeClient()
            executor = FakeExecutor()
            executor.sync_runs_payload = MagicMock(return_value=[
                {
                    "id": "run_with_folder",
                    "status": "completed",
                    "folder": "tests",
                    "updated_at": "2026-05-17T00:00:00+00:00",
                    "artifacts": [],
                },
                {
                    "id": "run_without_folder",
                    "status": "completed",
                    "updated_at": "2026-05-17T00:01:00+00:00",
                    "artifacts": [],
                },
            ])
            worker = RemoteWorker(config, paths, client, executor=executor)

            worker.sync_runs()

            run_upserts = [item for item in client.upserts if item[0] == "runs"]
            self.assertEqual(len(run_upserts), 2)
            self.assertTrue(all(isinstance(item[1], list) for item in run_upserts))
            self.assertNotEqual(set(run_upserts[0][1][0].keys()), set(run_upserts[1][1][0].keys()))

    def test_worker_uploads_latest_video_and_records_storage_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "videos" / "play" / "clip.mp4"
            video.parent.mkdir(parents=True)
            video.write_bytes(b"mp4-data")
            paths = self.make_paths(root)
            config = RemoteConfig(machine_id="lab-pc", accept_jobs=True)
            client = FakeClient()
            executor = FakeExecutor()
            executor.sync_runs_payload = MagicMock(return_value=[
                {
                    "id": "run one",
                    "status": "completed",
                    "artifacts": [{"kind": "video", "run_id": "run one", "local_path": str(video)}],
                }
            ])
            worker = RemoteWorker(config, paths, client, executor=executor)
            worker.sync_runs()
            self.assertEqual(len(client.uploads), 1)
            self.assertEqual(client.uploads[0][0], "redrhex-videos")
            self.assertEqual(client.uploads[0][1], "runs/run_one/videos/clip.mp4")
            artifact_upsert = next(u for u in client.upserts if u[0] == "artifacts")
            self.assertEqual(artifact_upsert[1][0]["storage_path"], "runs/run_one/videos/clip.mp4")
            self.assertEqual(artifact_upsert[1][0]["bytes"], len(b"mp4-data"))

    def test_worker_uploads_tensorboard_summary_and_records_storage_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = root / "logs" / "rsl_rl" / "redrhex_wheg" / "run one" / "training_panel" / "tensorboard_summary.png"
            summary.parent.mkdir(parents=True)
            summary.write_bytes(b"png-data")
            paths = self.make_paths(root)
            config = RemoteConfig(
                machine_id="lab-pc",
                accept_jobs=True,
                cloudflare_tunnel_host="https://mother.example.com",
            )
            client = FakeClient()
            executor = FakeExecutor()
            executor.sync_runs_payload = MagicMock(return_value=[
                {
                    "id": "run one",
                    "status": "completed",
                    "artifacts": [{"kind": "tensorboard_summary", "run_id": "run one", "local_path": str(summary)}],
                }
            ])
            worker = RemoteWorker(config, paths, client, executor=executor)

            worker.sync_runs()

            self.assertEqual(len(client.uploads), 1)
            self.assertEqual(client.uploads[0][0], "redrhex-videos")
            self.assertEqual(client.uploads[0][1], "runs/run_one/tensorboard/tensorboard_summary.png")
            self.assertEqual(client.uploads[0][3]["content_type"], "image/png")
            artifact_upsert = next(u for u in client.upserts if u[0] == "artifacts")
            self.assertEqual(artifact_upsert[1][0]["storage_path"], "runs/run_one/tensorboard/tensorboard_summary.png")
            self.assertEqual(
                artifact_upsert[1][0]["public_url"],
                "https://mother.example.com/api/runs/run%20one/tensorboard-summary.png",
            )
            self.assertEqual(artifact_upsert[1][0]["bytes"], len(b"png-data"))

    def test_worker_idle_media_pulse_uploads_recent_video_without_full_sync(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "clip.mp4"
            video.write_bytes(b"mp4-data")
            paths = self.make_paths(root)
            paths.ensure_dirs()
            config = RemoteConfig(machine_id="lab-pc", accept_jobs=True, sync_interval_seconds=9999)
            client = FakeClient()
            executor = RemoteJobExecutor(paths)
            executor.processes.running_isaac_processes = MagicMock(return_value=[])
            executor.history.add_run({
                "id": "run_video",
                "status": "completed",
                "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "latest_video": str(video),
                "video_status": "completed",
            })
            worker = RemoteWorker(config, paths, client, executor=executor)
            worker.last_sync_at = time.time()

            result = worker.poll_once()

            self.assertEqual(result["status"], "idle")
            self.assertEqual(len(client.uploads), 1)
            self.assertEqual(worker.last_sync_summary["targeted_run_ids"], ["run_video"])
            artifact_upsert = next(u for u in client.upserts if u[0] == "artifacts")
            self.assertEqual(artifact_upsert[1][0]["storage_path"], "runs/run_video/videos/clip.mp4")

    def test_worker_skips_video_upload_when_artifact_has_storage_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "clip.mp4"
            video.write_bytes(b"mp4-data")
            paths = self.make_paths(root)
            config = RemoteConfig(machine_id="lab-pc", accept_jobs=True)
            client = FakeClient()
            client.select_rows = [{"storage_path": "runs/run_one/videos/clip.mp4"}]
            executor = FakeExecutor()
            executor.sync_runs_payload = MagicMock(return_value=[
                {
                    "id": "run_one",
                    "status": "completed",
                    "artifacts": [{"kind": "video", "run_id": "run_one", "local_path": str(video)}],
                }
            ])
            worker = RemoteWorker(config, paths, client, executor=executor)
            worker.sync_runs()
            self.assertEqual(client.uploads, [])

    def test_run_forever_continues_after_poll_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            config = RemoteConfig(machine_id="lab-pc", accept_jobs=False, poll_interval_seconds=0)
            client = FakeClient()
            worker = RemoteWorker(config, paths, client, executor=FakeExecutor())
            call_count = 0

            def failing_poll():
                nonlocal call_count
                call_count += 1
                if call_count < 3:
                    raise RuntimeError("transient error")
                raise KeyboardInterrupt  # BaseException — not caught by except Exception

            worker.poll_once = failing_poll
            with self.assertRaises(KeyboardInterrupt):
                worker.run_forever()
            self.assertEqual(call_count, 3)

    # ------------------------------------------------------------------
    # RemoteStateStore persistence
    # ------------------------------------------------------------------

    def test_state_store_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            state_file = Path(tmp) / "remote_state.json"
            store = RemoteStateStore(state_file)
            store.save({"accept_jobs": True})
            store2 = RemoteStateStore(state_file)
            state = store2.load()
            self.assertTrue(state["accept_jobs"])
            self.assertIn("updated_at", state)

    def test_state_store_corrupt_file_returns_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            state_file = Path(tmp) / "remote_state.json"
            state_file.write_text("not valid json {{{{")
            store = RemoteStateStore(state_file)
            state = store.load()
            self.assertIsInstance(state, dict)

    # ------------------------------------------------------------------
    # RemoteWorkerManager
    # ------------------------------------------------------------------

    def _write_env_file(self, tmp: Path) -> Path:
        env_file = tmp / ".redrhex_remote.env"
        env_file.write_text(
            "\n".join(
                [
                    'export REDRHEX_SUPABASE_URL="https://example.supabase.co"',
                    'export REDRHEX_SUPABASE_ANON_KEY="anon-key"',
                    'export REDRHEX_SUPABASE_MACHINE_TOKEN="secret-token"',
                    'export REDRHEX_MACHINE_ID="lab-pc"',
                ]
            ),
            encoding="utf-8",
        )
        return env_file

    def test_parse_env_file_handles_exported_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_file = self._write_env_file(Path(tmp))
            parsed = parse_env_file(env_file)
            self.assertEqual(parsed["REDRHEX_MACHINE_ID"], "lab-pc")
            self.assertEqual(parsed["REDRHEX_SUPABASE_URL"], "https://example.supabase.co")

    def test_manager_status_hides_secrets_and_reports_setup(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            paths.ensure_dirs()
            env_file = self._write_env_file(root)
            manager = RemoteWorkerManager(paths, RemoteStateStore(paths.remote_state_file), env_file=env_file)
            status = manager.status()
            self.assertTrue(status["configured"])
            self.assertTrue(status["machine_token_configured"])
            self.assertNotIn("secret-token", str(status))
            self.assertTrue(any(check["id"] == "env_file" and check["ok"] for check in status["setup_checks"]))

    def test_manager_detects_existing_tmux_worker(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            paths.ensure_dirs()
            env_file = self._write_env_file(root)
            runner = FakeTmuxRunner(running=True)
            with patch("tools.training_panel.training_panel.remote_manager.shutil.which", return_value="/usr/bin/tmux"):
                manager = RemoteWorkerManager(paths, RemoteStateStore(paths.remote_state_file), env_file=env_file, run_command=runner)
                status = manager.status()
            self.assertTrue(status["worker_running"])
            self.assertEqual(status["worker_runtime_mode"], "tmux")
            self.assertEqual(status["worker_tmux_session"], REMOTE_WORKER_SESSION)

    def test_manager_start_tmux_builds_session_and_refuses_duplicate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            paths.ensure_dirs()
            env_file = self._write_env_file(root)
            runner = FakeTmuxRunner(running=False)
            with patch("tools.training_panel.training_panel.remote_manager.shutil.which", return_value="/usr/bin/tmux"):
                manager = RemoteWorkerManager(paths, RemoteStateStore(paths.remote_state_file), env_file=env_file, run_command=runner)
                manager.start("tmux")
                with self.assertRaises(ValueError):
                    manager.start("tmux")
            self.assertTrue(any(call[1] == "new-session" for call in runner.calls))

    def test_manager_save_settings_persists_worker_controls(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            paths.ensure_dirs()
            state = RemoteStateStore(paths.remote_state_file)
            manager = RemoteWorkerManager(paths, state, env_file=self._write_env_file(root))
            manager.save_settings({"worker_mode": "child", "worker_autostart": True, "accept_jobs": True})
            saved = state.load()
            self.assertEqual(saved["worker_mode"], "child")
            self.assertTrue(saved["worker_autostart"])
            self.assertTrue(saved["accept_jobs"])

    def test_manager_start_child_writes_pid_and_reports_running(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            paths.ensure_dirs()
            env_file = self._write_env_file(root)
            manager = RemoteWorkerManager(
                paths,
                RemoteStateStore(paths.remote_state_file),
                env_file=env_file,
                popen_factory=MagicMock(return_value=FakePopen(pid=777)),
            )
            manager._pid_is_remote_worker = MagicMock(return_value=True)
            with patch("tools.training_panel.training_panel.remote_manager.shutil.which", return_value=None):
                status = manager.start("child")
            self.assertEqual(paths.panel_log_root.joinpath("remote_worker.pid").read_text(encoding="utf-8"), "777")
            self.assertEqual(status["worker_runtime_mode"], "child")

    def test_manager_autostart_respects_saved_setting(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            paths.ensure_dirs()
            env_file = self._write_env_file(root)
            state = RemoteStateStore(paths.remote_state_file)
            state.save({"worker_autostart": True, "worker_mode": "child"})
            manager = RemoteWorkerManager(
                paths,
                state,
                env_file=env_file,
                popen_factory=MagicMock(return_value=FakePopen(pid=888)),
            )
            manager._pid_is_remote_worker = MagicMock(return_value=True)
            with patch("tools.training_panel.training_panel.remote_manager.shutil.which", return_value=None):
                manager.autostart_if_enabled()
                status = manager.status()
            self.assertTrue(status["worker_running"])
            self.assertEqual(status["worker_runtime_mode"], "child")

if __name__ == "__main__":
    unittest.main()
