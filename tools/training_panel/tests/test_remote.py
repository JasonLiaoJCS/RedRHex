import tempfile
import unittest
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

from tools.training_panel import __version__
from tools.training_panel.training_panel.config import PanelPaths
from tools.training_panel.training_panel.notifications import (
    completion_event_from_run,
    discord_message,
    email_message,
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
        self.claim_gpu_locked = []
        self.raise_on_artifacts_upsert = False

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

    def select(self, table, query=None):
        return self.select_rows

    def delete(self, table, query=None):
        self.deletes.append((table, query))

    def upload_storage_object(self, bucket, object_path, file_path, **kwargs):
        self.uploads.append((bucket, object_path, file_path, kwargs))
        return {"Key": object_path}


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
        self.assertTrue(role_allows("operator", "tensorboard"))
        self.assertTrue(role_allows("operator", "compact_run"))
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
            history.patch_run_metadata("run_one", display_name="old", folder=None, notes="old note")
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

    # ------------------------------------------------------------------
    # email_message
    # ------------------------------------------------------------------

    def test_email_message_content(self):
        event = completion_event_from_run(
            {"id": "run_two", "status": "failed", "params": {"task": "T", "max_iterations": 5}},
        )
        msg = email_message(event, to_email="user@example.com")
        self.assertEqual(msg["to"], "user@example.com")
        self.assertIn("failed", msg["subject"])
        self.assertIn("run_two", msg["text"])
        self.assertIn("Return code", msg["text"])


if __name__ == "__main__":
    unittest.main()
