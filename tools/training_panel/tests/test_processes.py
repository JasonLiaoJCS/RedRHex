import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from tools.training_panel.training_panel.commands import TrainingParams, VideoParams
from tools.training_panel.training_panel.config import PanelPaths
from tools.training_panel.training_panel.history import HistoryStore
from tools.training_panel.training_panel.processes import EXTERNAL_TRAINING_ID_PREFIX, ProcessInfo, ProcessRegistry, SpawnedProcess


class ProcessRegistryTests(unittest.TestCase):
    def make_paths(self, root: Path) -> PanelPaths:
        conda_sh = root / "conda.sh"
        conda_sh.write_text("conda() { :; }\n", encoding="utf-8")
        isaaclab_root = root / "IsaacLab"
        isaaclab_root.mkdir()
        launcher = isaaclab_root / "isaaclab.sh"
        launcher.write_text(
            "#!/usr/bin/env bash\n"
            "echo fake isaaclab \"$@\"\n"
            "sleep 3\n",
            encoding="utf-8",
        )
        os.chmod(launcher, 0o755)
        return PanelPaths(
            repo_root=root,
            isaaclab_root=isaaclab_root,
            isaacsim_root=root / "isaacsim",
            conda_sh=conda_sh,
            conda_env="env",
        )

    def test_training_record_preserves_tweak_metadata(self):
        class FakeProcess:
            pid = 12345

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            history = HistoryStore(paths)
            registry = ProcessRegistry(paths, history)
            params = TrainingParams.from_dict(
                {
                    "task": "Template-Redrhex-Direct-v0",
                    "num_envs": 4,
                    "max_iterations": 8,
                    "device": "cpu",
                    "reward_preset_id": "tweak-run-one",
                    "reward_overrides": {"rew_scale_alive": 0.2},
                    "tweak_source_run_id": "run_one",
                    "tweak_source_label": "Run One",
                    "requester_id": "11111111-1111-4111-8111-111111111111",
                    "requester_label": "Jason",
                    "display_name": "stair warmup",
                }
            )
            with patch.object(registry, "_spawn_shell", return_value=SpawnedProcess(proc=FakeProcess())), patch("threading.Thread") as thread_cls:
                thread_cls.return_value.start = Mock()
                run = registry.start_training(params)

            record = history.get_run(run["id"])
            self.assertEqual(record["params"]["tweak_source_run_id"], "run_one")
            self.assertEqual(record["params"]["tweak_source_label"], "Run One")
            self.assertEqual(record["created_by"], "11111111-1111-4111-8111-111111111111")
            self.assertEqual(record["requester_label"], "Jason")
            self.assertEqual(record["reward_preset_id"], "tweak-run-one")
            self.assertEqual(record["display_name"], "stair warmup")
            self.assertEqual(record["params"]["display_name"], "stair warmup")

    def test_queue_training_starts_immediately_when_gpu_is_free(self):
        class FakeProcess:
            pid = 12345

            def poll(self):
                return None

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            history = HistoryStore(paths)
            registry = ProcessRegistry(paths, history)
            params = TrainingParams.from_dict({"task": "Template-Redrhex-Direct-v0", "num_envs": 4, "max_iterations": 8})
            with patch.object(registry, "_spawn_shell", return_value=SpawnedProcess(proc=FakeProcess())), patch("threading.Thread") as thread_cls:
                thread_cls.return_value.start = Mock()
                run = registry.queue_training(params)

            record = history.get_run(run["id"])
            self.assertEqual(record["status"], "running")
            self.assertEqual([process["kind"] for process in registry.running_isaac_processes()], ["training"])

    def test_queue_training_waits_behind_active_gpu_process(self):
        class FakeProcess:
            pid = 12345

            def poll(self):
                return None

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            history = HistoryStore(paths)
            registry = ProcessRegistry(paths, history)
            params = TrainingParams.from_dict({"task": "Template-Redrhex-Direct-v0", "num_envs": 4, "max_iterations": 8})
            with patch.object(registry, "_spawn_shell", return_value=SpawnedProcess(proc=FakeProcess())), patch("threading.Thread") as thread_cls:
                thread_cls.return_value.start = Mock()
                active = registry.start_training(params)
                queued = registry.queue_training(params)

            self.assertEqual(history.get_run(active["id"])["status"], "running")
            self.assertEqual(history.get_run(queued["id"])["status"], "queued")
            self.assertIsNone(history.get_run(queued["id"]).get("pid"))

    def test_cancel_queued_training(self):
        class FakeProcess:
            pid = 12345

            def poll(self):
                return None

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            history = HistoryStore(paths)
            registry = ProcessRegistry(paths, history)
            params = TrainingParams.from_dict({"task": "Template-Redrhex-Direct-v0", "num_envs": 4, "max_iterations": 8})
            with patch.object(registry, "_spawn_shell", return_value=SpawnedProcess(proc=FakeProcess())), patch("threading.Thread") as thread_cls:
                thread_cls.return_value.start = Mock()
                registry.start_training(params)
                queued = registry.queue_training(params)

            self.assertTrue(registry.cancel_queued_training(queued["id"]))
            self.assertEqual(history.get_run(queued["id"])["status"], "cancelled")

    def test_start_next_queued_training_when_gpu_becomes_free(self):
        class MutableProcess:
            pid = 12345
            returncode = None

            def poll(self):
                return self.returncode

        class RunningProcess:
            pid = 12346

            def poll(self):
                return None

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            history = HistoryStore(paths)
            registry = ProcessRegistry(paths, history)
            params = TrainingParams.from_dict({"task": "Template-Redrhex-Direct-v0", "num_envs": 4, "max_iterations": 8})
            active_proc = MutableProcess()
            with patch.object(registry, "_spawn_shell", return_value=SpawnedProcess(proc=active_proc)), patch("threading.Thread") as thread_cls:
                thread_cls.return_value.start = Mock()
                registry.start_training(params)
                queued = registry.queue_training(params)
            active_proc.returncode = 0
            with patch.object(registry, "_spawn_shell", return_value=SpawnedProcess(proc=RunningProcess())), patch("threading.Thread") as thread_cls:
                thread_cls.return_value.start = Mock()
                started = registry.start_next_queued_training()

            self.assertEqual(started["id"], queued["id"])
            self.assertEqual(history.get_run(queued["id"])["status"], "running")
            self.assertEqual(history.get_run(queued["id"])["pid"], 12346)

    def test_play_process_debug_streams_log_tail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            history = HistoryStore(paths)
            log_dir = paths.rsl_rl_log_root / "run_one"
            params_dir = log_dir / "params"
            params_dir.mkdir(parents=True)
            checkpoint = log_dir / "model_10.pt"
            checkpoint.write_text("x", encoding="utf-8")
            (params_dir / "env.yaml").write_text(
                "terrain:\n  terrain_type: plane\nterrain_curriculum_enable: false\n",
                encoding="utf-8",
            )
            history.add_run({"id": "run_one", "source": "training_panel", "status": "completed", "log_dir": str(log_dir)})
            registry = ProcessRegistry(paths, history)
            result = registry.start_play("run_one", str(checkpoint), device="cpu")
            try:
                debug = registry.get_process_debug(result["id"])
                self.assertIsNotNone(debug)
                self.assertEqual(debug["kind"], "play")
                self.assertIsNone(debug["returncode"])
                self.assertEqual(debug["source_run_id"], "run_one")
                self.assertIn("scripts/rsl_rl/play.py", debug["command"])
                self.assertIn("--terrain_override_file", debug["command"])
                self.assertIn("--camera_follow_robot", debug["command"])
                self.assertIn("--camera_eye -3.0 -2.4 1.6", debug["command"])
                override_files = list(paths.process_override_dir.glob("*_terrain.json"))
                self.assertEqual(len(override_files), 1)
                self.assertIn("terrain.terrain_type", override_files[0].read_text(encoding="utf-8"))
                self.assertIn("fake isaaclab", debug["log_tail"])
            finally:
                proc = registry._processes.get(result["id"])
                registry.stop(result["id"])
                if proc:
                    proc.wait(timeout=8)
                time.sleep(0.1)

    def test_terrain_override_file_is_written_and_cleared(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            registry = ProcessRegistry(paths, HistoryStore(paths))
            registry._write_terrain_override({"terrain.terrain_type": "plane", "terrain_curriculum_enable": False})
            self.assertTrue(paths.terrain_override_file.exists())
            text = paths.terrain_override_file.read_text(encoding="utf-8")
            self.assertIn("terrain.terrain_type", text)
            registry._write_terrain_override({})
            self.assertFalse(paths.terrain_override_file.exists())

    def test_video_recording_process_uses_headless_video_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            history = HistoryStore(paths)
            log_dir = paths.rsl_rl_log_root / "run_one"
            params_dir = log_dir / "params"
            params_dir.mkdir(parents=True)
            checkpoint = log_dir / "model_10.pt"
            checkpoint.write_text("x", encoding="utf-8")
            (params_dir / "env.yaml").write_text(
                "terrain:\n  terrain_type: plane\nterrain_curriculum_enable: false\n",
                encoding="utf-8",
            )
            history.add_run(
                {
                    "id": "run_one",
                    "source": "training_panel",
                    "status": "completed",
                    "created_at": "2026-05-15T11:00:00",
                    "log_dir": str(log_dir),
                }
            )
            registry = ProcessRegistry(paths, history)
            result = registry.start_video_recording(
                "run_one",
                str(checkpoint),
                device="cpu",
                video_params=VideoParams.from_preset("high"),
            )
            try:
                debug = registry.get_process_debug(result["id"])
                self.assertIsNotNone(debug)
                self.assertEqual(debug["kind"], "video")
                self.assertEqual(debug["source_run_id"], "run_one")
                self.assertIn("--headless", debug["command"])
                self.assertIn("--video", debug["command"])
                self.assertIn("--video_length 1200", debug["command"])
                self.assertIn("--video_width 1920", debug["command"])
                self.assertIn("--video_height 1080", debug["command"])
                self.assertIn("--video_fps 30", debug["command"])
                self.assertIn("--rendering_mode quality", debug["command"])
                self.assertIn("--terrain_override_file", debug["command"])
                self.assertIn("--camera_follow_robot", debug["command"])
                self.assertIn("--camera_lookat 0.45 0.0 0.35", debug["command"])
                override_files = list(paths.process_override_dir.glob("*_terrain.json"))
                self.assertEqual(len(override_files), 1)
                self.assertIn("terrain.terrain_type", override_files[0].read_text(encoding="utf-8"))
                self.assertIn("attach_command", debug)
                run = history.get_run("run_one")
                self.assertEqual(run["video_status"], "recording")
                self.assertEqual(run["video_process_id"], result["id"])
                self.assertEqual(run["video_preset"], "high")
            finally:
                proc = registry._processes.get(result["id"])
                registry.stop(result["id"])
                if proc:
                    proc.wait(timeout=8)
                time.sleep(0.1)

    def test_process_terrain_override_falls_back_to_run_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            history = HistoryStore(paths)
            history.add_run(
                {
                    "id": "run_one",
                    "source": "training_panel",
                    "status": "completed",
                    "created_at": "2026-05-15T11:00:00",
                    "terrain_overrides": {"terrain.terrain_type": "plane"},
                }
            )
            registry = ProcessRegistry(paths, history)
            path = registry._write_process_terrain_override("play_test", "run_one")
            self.assertIsNotNone(path)
            self.assertIn("run metadata", Path(path).read_text(encoding="utf-8"))
            self.assertIn("terrain.terrain_type", Path(path).read_text(encoding="utf-8"))

    def test_process_terrain_override_prefers_panel_metadata_over_env_yaml(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            history = HistoryStore(paths)
            log_dir = paths.rsl_rl_log_root / "run_one"
            params_dir = log_dir / "params"
            params_dir.mkdir(parents=True)
            (params_dir / "env.yaml").write_text(
                "terrain:\n  terrain_type: generator\n  max_init_terrain_level: 3\n",
                encoding="utf-8",
            )
            history.add_run(
                {
                    "id": "run_one",
                    "source": "training_panel",
                    "status": "completed",
                    "created_at": "2026-05-15T11:00:00",
                    "log_dir": str(log_dir),
                    "terrain_overrides": {
                        "terrain.terrain_type": "plane",
                        "terrain.max_init_terrain_level": 0,
                    },
                }
            )
            registry = ProcessRegistry(paths, history)

            path = registry._write_process_terrain_override("play_test", "run_one")
            payload = json.loads(Path(path).read_text(encoding="utf-8"))

            self.assertEqual(payload["source"], "run metadata")
            self.assertEqual(payload["overrides"]["terrain.terrain_type"], "plane")
            self.assertEqual(payload["overrides"]["terrain.max_init_terrain_level"], 0)

    def test_process_terrain_override_absent_for_old_runs_without_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            history = HistoryStore(paths)
            history.add_run({"id": "old_run", "source": "rsl_rl", "status": "completed"})
            registry = ProcessRegistry(paths, history)
            self.assertIsNone(registry._write_process_terrain_override("play_test", "old_run"))

    def test_onnx_export_process_uses_export_only_flags_and_updates_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            history = HistoryStore(paths)
            log_dir = paths.rsl_rl_log_root / "run_one"
            log_dir.mkdir(parents=True)
            checkpoint = log_dir / "model_10.pt"
            checkpoint.write_text("x", encoding="utf-8")
            history.add_run(
                {
                    "id": "run_one",
                    "source": "training_panel",
                    "status": "completed",
                    "created_at": "2026-05-15T11:00:00",
                    "log_dir": str(log_dir),
                }
            )
            registry = ProcessRegistry(paths, history)
            result = registry.start_onnx_export("run_one", str(checkpoint), device="cpu")
            try:
                debug = registry.get_process_debug(result["id"])
                self.assertIsNotNone(debug)
                self.assertEqual(debug["kind"], "onnx")
                self.assertEqual(debug["source_run_id"], "run_one")
                self.assertIn("--headless", debug["command"])
                self.assertIn("--export_policy_only", debug["command"])
                self.assertIn("attach_command", debug)
                run = history.get_run("run_one")
                self.assertEqual(run["onnx_status"], "exporting")
                self.assertEqual(run["onnx_process_id"], result["id"])
                self.assertEqual(run["onnx_pid"], result["pid"])
            finally:
                proc = registry._processes.get(result["id"])
                registry.stop(result["id"])
                if proc:
                    proc.wait(timeout=8)
                time.sleep(0.1)

    def test_successful_training_monitor_starts_video_recording(self):
        class CompletedProcess:
            def poll(self):
                return 0  # non-None → while loop body skipped immediately

            def wait(self):
                return 0

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            history = HistoryStore(paths)
            log_dir = paths.rsl_rl_log_root / "run_one"
            log_dir.mkdir(parents=True)
            checkpoint = log_dir / "model_7.pt"
            checkpoint.write_text("x", encoding="utf-8")
            history.add_run(
                {
                    "id": "panel_train",
                    "source": "training_panel",
                    "status": "running",
                    "created_at": "2026-05-15T11:00:00",
                    "params": {"device": "cpu"},
                }
            )
            registry = ProcessRegistry(paths, history)
            registry.start_video_recording = Mock()
            registry._monitor_training("panel_train", CompletedProcess(), 0)

            registry.start_video_recording.assert_called_once_with(
                run_id="panel_train",
                checkpoint=str(checkpoint),
                device="cpu",
                video_params=VideoParams.from_preset("high"),
            )
            run = history.get_run("panel_train")
            self.assertEqual(run["status"], "completed")
            self.assertEqual(run["returncode"], 0)
            self.assertEqual(run["log_dir"], str(log_dir))

    def test_stop_all_for_run_stops_linked_play_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            registry = ProcessRegistry(paths, HistoryStore(paths))
            result = registry.start_play("run_one", "/tmp/checkpoint.pt", device="cpu")
            proc = registry._processes.get(result["id"])
            try:
                stopped = registry.stop_all_for_run("run_one")
                self.assertEqual(stopped, [result["id"]])
                if proc:
                    proc.wait(timeout=8)
            finally:
                if proc and proc.poll() is None:
                    registry.stop(result["id"])
                    proc.wait(timeout=8)

    def test_running_for_run_filters_by_process_kind(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            registry = ProcessRegistry(paths, HistoryStore(paths))
            result = registry.start_play("run_one", "/tmp/checkpoint.pt", device="cpu")
            proc = registry._processes.get(result["id"])
            try:
                play_processes = registry.running_for_run("run_one", kind="play")
                video_processes = registry.running_for_run("run_one", kind="video")
                self.assertEqual([process["run_id"] for process in play_processes], [result["id"]])
                self.assertEqual(video_processes, [])
                self.assertEqual([process["run_id"] for process in registry.running_media_processes()], [result["id"]])
            finally:
                registry.stop(result["id"])
                if proc:
                    proc.wait(timeout=8)

    def test_running_for_log_dir_blocks_unlinked_active_training_log(self):
        class FakeProcess:
            def poll(self):
                return None

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            paths.ensure_dirs()
            history = HistoryStore(paths)
            registry = ProcessRegistry(paths, history)
            history.add_run({
                "id": "panel_run",
                "source": "training_panel",
                "status": "running",
                "created_at": "2026-05-17T01:35:27",
                "process_log": str(paths.process_log_dir / "panel_run.log"),
                "log_dir": None,
            })
            log_dir = paths.rsl_rl_log_root / "2026-05-17_01-35-34_wheg_locomotion_reform_v1"
            log_dir.mkdir(parents=True)
            os.utime(log_dir, (1778952937, 1778952937))
            process_log = paths.process_log_dir / "panel_run.log"
            process_log.write_text(f"Writing events to {log_dir}\n", encoding="utf-8")
            registry._processes["panel_run"] = FakeProcess()
            registry._infos["panel_run"] = ProcessInfo(
                kind="training",
                pid=12345,
                run_id="panel_run",
                log_file=str(process_log),
                started_at="2026-05-17T01:35:27",
                command="train.py --task Template-Redrhex-Direct-v0",
            )

            running = registry.running_for_log_dir(log_dir)

            self.assertEqual([process["run_id"] for process in running], ["panel_run"])

    def test_running_for_log_dir_uses_saved_start_time_for_external_training(self):
        class FakeProcess:
            def poll(self):
                return None

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            paths.ensure_dirs()
            history = HistoryStore(paths)
            registry = ProcessRegistry(paths, history)
            history.add_run({
                "id": "panel_run",
                "source": "training_panel",
                "status": "running",
                "created_at": "2026-05-17T01:35:27",
                "process_log": str(paths.process_log_dir / "panel_run.log"),
                "log_dir": None,
            })
            log_dir = paths.rsl_rl_log_root / "2026-05-17_01-35-34_wheg_locomotion_reform_v1"
            log_dir.mkdir(parents=True)
            os.utime(log_dir, (1778952937, 1778952937))
            process_log = paths.process_log_dir / "panel_run.log"
            process_log.write_text(
                "Exact experiment name requested from command line: 2026-05-17_01-35-34\n",
                encoding="utf-8",
            )
            registry._processes["external_training_123"] = FakeProcess()
            registry._infos["external_training_123"] = ProcessInfo(
                kind="training",
                pid=12345,
                run_id="external_training_123",
                source_run_id="panel_run",
                log_file=str(process_log),
                started_at="",
                command="train.py --task Template-Redrhex-Direct-v0",
            )

            running = registry.running_for_log_dir(log_dir)

            self.assertEqual([process["run_id"] for process in running], ["external_training_123"])

    def test_log_dir_from_process_log_recovers_deleted_event_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            paths.ensure_dirs()
            process_log = paths.process_log_dir / "panel_run.log"
            deleted_log_dir = paths.rsl_rl_log_root / "2026-05-17_01-35-34_wheg_locomotion_reform_v1"
            process_log.write_text(
                "FileNotFoundError: [Errno 2] No such file or directory: "
                f"b'{deleted_log_dir}/events.out.tfevents.1778952937.host.2617302.0'\n",
                encoding="utf-8",
            )
            history = HistoryStore(paths)
            history.add_run({
                "id": "panel_run",
                "source": "training_panel",
                "status": "running",
                "created_at": "2026-05-17T01:35:27",
                "process_log": str(process_log),
                "log_dir": None,
            })
            registry = ProcessRegistry(paths, history)

            self.assertEqual(registry._log_dir_from_process_log("panel_run"), str(deleted_log_dir))

    def test_training_command_match_rejects_conflicting_args(self):
        recorded = (
            "/IsaacLab/isaaclab.sh -p scripts/rsl_rl/train.py "
            "--task Template-Redrhex-Direct-v0 --num_envs 4 --max_iterations 10 --device cuda:0 --headless"
        )
        observed = (
            "bash /IsaacLab/isaaclab.sh -p scripts/rsl_rl/train.py "
            "--task Template-Redrhex-Direct-v0 --num_envs 4 --max_iterations 1 --device cuda:0 --headless"
        )
        self.assertFalse(ProcessRegistry._training_commands_match(recorded, observed))

    def test_training_command_match_accepts_same_core_args(self):
        recorded = (
            "/IsaacLab/isaaclab.sh -p scripts/rsl_rl/train.py "
            "--task Template-Redrhex-Direct-v0 --num_envs 4 --max_iterations 10 --device cuda:0 --headless"
        )
        observed = (
            "python scripts/rsl_rl/train.py "
            "--device cuda:0 --max_iterations 10 --num_envs 4 --task Template-Redrhex-Direct-v0"
        )
        self.assertTrue(ProcessRegistry._training_commands_match(recorded, observed))

    def test_running_isaac_processes_includes_onnx_exports(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            registry = ProcessRegistry(paths, HistoryStore(paths))
            result = registry.start_onnx_export("run_one", "/tmp/checkpoint.pt", device="cpu")
            proc = registry._processes.get(result["id"])
            try:
                self.assertEqual([process["run_id"] for process in registry.running_isaac_processes()], [result["id"]])
            finally:
                registry.stop(result["id"])
                if proc:
                    proc.wait(timeout=8)

    def test_stop_all_for_run_returns_empty_for_nonexistent_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            registry = ProcessRegistry(paths, HistoryStore(paths))
            self.assertEqual(registry.stop_all_for_run("missing_run"), [])

    def test_stop_all_for_run_ignores_already_exited_processes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            registry = ProcessRegistry(paths, HistoryStore(paths))
            result = registry.start_play("run_one", "/tmp/checkpoint.pt", device="cpu")
            proc = registry._processes.get(result["id"])
            self.assertIsNotNone(proc)
            registry.stop(result["id"])
            proc.wait(timeout=8)
            self.assertEqual(registry.stop_all_for_run("run_one"), [])

    def test_source_run_id_from_play_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            registry = ProcessRegistry(paths, HistoryStore(paths))
            command = (
                "python scripts/rsl_rl/play.py --checkpoint "
                "/home/lab_user1/Py/RedRHex/logs/rsl_rl/redrhex_wheg/2026_run/model_12.pt"
            )
            self.assertEqual(registry._source_run_id_from_command(command), "2026_run")

    def test_source_run_id_from_tensorboard_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            registry = ProcessRegistry(paths, HistoryStore(paths))
            command = "tensorboard --logdir /repo/logs/rsl_rl/redrhex_wheg/2026_run --port 6008"
            self.assertEqual(registry._source_run_id_from_tensorboard_command(command), "2026_run")

    def test_source_run_id_from_training_process_uses_panel_record_pid(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            history = HistoryStore(paths)
            history.add_run(
                {
                    "id": "panel_train",
                    "source": "training_panel",
                    "pid": 123,
                    "created_at": "2026-05-15T11:00:00",
                    "command": f"{paths.isaaclab_launcher} -p scripts/rsl_rl/train.py --task Template-Redrhex-Direct-v0",
                }
            )
            registry = ProcessRegistry(paths, history)
            command = "python scripts/rsl_rl/train.py --task Template-Redrhex-Direct-v0"
            self.assertEqual(registry._source_run_id_from_training_process(123, 123, command), "panel_train")

    def test_external_training_process_maps_to_history_and_debug_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            history = HistoryStore(paths)
            process_log = paths.process_log_dir / "panel_train.log"
            process_log.write_text("training is still running\n", encoding="utf-8")
            history.add_run(
                {
                    "id": "panel_train",
                    "source": "training_panel",
                    "pid": 222,
                    "created_at": "2026-05-15T11:00:00",
                    "process_log": str(process_log),
                    "command": (
                        f"{paths.isaaclab_launcher} -p scripts/rsl_rl/train.py --task Template-Redrhex-Direct-v0 "
                        "--num_envs 4 --max_iterations 8 --device cuda:0"
                    ),
                }
            )
            output = (
                f"222 222 Ss bash {paths.isaaclab_launcher} -p scripts/rsl_rl/train.py "
                "--task Template-Redrhex-Direct-v0 --num_envs 4 --max_iterations 8 --device cuda:0\n"
                "223 222 Rl python scripts/rsl_rl/train.py --task Template-Redrhex-Direct-v0 "
                "--num_envs 4 --max_iterations 8 --device cuda:0\n"
            )
            registry = ProcessRegistry(paths, history)
            with patch(
                "tools.training_panel.training_panel.processes.subprocess.check_output",
                return_value=output,
            ):
                processes = registry.list_processes()
                training = next(process for process in processes if process["kind"] == "training")
                self.assertEqual(training["run_id"], f"{EXTERNAL_TRAINING_ID_PREFIX}222")
                self.assertEqual(training["source_run_id"], "panel_train")
                debug = registry.get_process_debug(training["run_id"])
            self.assertIsNotNone(debug)
            self.assertIn("training is still running", debug["log_tail"])

    def test_tmux_server_title_does_not_count_as_training_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            registry = ProcessRegistry(paths, HistoryStore(paths))
            command = (
                f"/usr/bin/tmux new-session -d -s redrhex_panel_fake -- bash -lc "
                f"{paths.isaaclab_launcher} -p scripts/rsl_rl/train.py --task Template-Redrhex-Direct-v0"
            )
            self.assertFalse(registry._is_repo_training_process(command, "123"))

    def test_reconcile_links_completed_panel_run_to_discovered_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            history = HistoryStore(paths)
            process_log = paths.process_log_dir / "panel_done.log"
            process_log.write_text(
                "Exact experiment name requested from command line: 2026-05-15_13-58-07\n" + ("training...\n" * 20000),
                encoding="utf-8",
            )
            exit_file = paths.process_log_dir / "panel_done.exit"
            exit_file.write_text("0", encoding="utf-8")
            log_dir = paths.rsl_rl_log_root / "2026-05-15_13-58-07_wheg_locomotion"
            log_dir.mkdir(parents=True)
            (log_dir / "model_9999.pt").write_text("x", encoding="utf-8")
            newer_log_dir = paths.rsl_rl_log_root / "2026-05-15_15-18-22_wheg_locomotion"
            newer_log_dir.mkdir(parents=True)
            (newer_log_dir / "model_99.pt").write_text("x", encoding="utf-8")
            history.add_run(
                {
                    "id": "panel_done",
                    "source": "training_panel",
                    "status": "running",
                    "created_at": "2026-05-15T13:58:01",
                    "process_log": str(process_log),
                    "exit_file": str(exit_file),
                    "folder": "experiments",
                }
            )
            history.patch_run_metadata(log_dir.name, source="rsl_rl", log_dir=str(log_dir), folder="experiments")
            registry = ProcessRegistry(paths, history)
            with patch("tools.training_panel.training_panel.processes.subprocess.check_output", return_value=""):
                registry.reconcile_stale_history()

            runs = history.list_runs()
            panel = next(run for run in runs if run["id"] == "panel_done")
            self.assertEqual(panel["status"], "completed")
            self.assertEqual(panel["returncode"], 0)
            self.assertEqual(panel["log_dir"], str(log_dir))
            self.assertEqual(panel["latest_checkpoint"], str(log_dir / "model_9999.pt"))
            self.assertFalse(any(run["id"] == log_dir.name for run in runs))

    def test_reconcile_persists_running_panel_log_dir_before_exit(self):
        class FakeProcess:
            pid = 12345

            def poll(self):
                return None

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            history = HistoryStore(paths)
            process_log = paths.process_log_dir / "panel_running.log"
            process_log.write_text(
                "Exact experiment name requested from command line: 2026-05-17_01-35-34\n",
                encoding="utf-8",
            )
            log_dir = paths.rsl_rl_log_root / "2026-05-17_01-35-34_wheg_locomotion_reform_v1"
            log_dir.mkdir(parents=True)
            (log_dir / "model_1.pt").write_text("x", encoding="utf-8")
            history.add_run(
                {
                    "id": "panel_running",
                    "source": "training_panel",
                    "status": "running",
                    "created_at": "2026-05-17T01:35:27",
                    "process_log": str(process_log),
                    "log_dir": None,
                }
            )
            registry = ProcessRegistry(paths, history)
            registry._processes["panel_running"] = FakeProcess()
            registry._infos["panel_running"] = ProcessInfo(
                kind="training",
                pid=12345,
                run_id="panel_running",
                log_file=str(process_log),
                started_at="2026-05-17T01:35:27",
                command="train.py --task Template-Redrhex-Direct-v0",
            )

            with patch("tools.training_panel.training_panel.processes.subprocess.check_output", return_value=""):
                registry.reconcile_stale_history()

            raw = next(record for record in history._load_data()["runs"] if record["id"] == "panel_running")
            panel = history.get_run("panel_running")
            self.assertEqual(raw["status"], "running")
            self.assertEqual(raw["log_dir"], str(log_dir))
            self.assertEqual(panel["latest_checkpoint"], str(log_dir / "model_1.pt"))

    def test_reconcile_persists_fresh_discovered_log_before_exact_name(self):
        class FakeProcess:
            pid = 12346

            def poll(self):
                return None

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            history = HistoryStore(paths)
            process_log = paths.process_log_dir / "panel_running.log"
            process_log.write_text("Isaac startup has not printed the experiment name yet.\n", encoding="utf-8")
            log_dir = paths.rsl_rl_log_root / "2026-05-17_10-21-24_wheg_locomotion_reform_v1"
            log_dir.mkdir(parents=True)
            (log_dir / "model_0.pt").write_text("x", encoding="utf-8")
            history.add_run(
                {
                    "id": "panel_20260517_102117_740732",
                    "source": "training_panel",
                    "status": "running",
                    "created_at": "2026-05-17T10:21:17",
                    "process_log": str(process_log),
                    "log_dir": None,
                }
            )
            registry = ProcessRegistry(paths, history)
            registry._processes["panel_20260517_102117_740732"] = FakeProcess()
            registry._infos["panel_20260517_102117_740732"] = ProcessInfo(
                kind="training",
                pid=12346,
                run_id="panel_20260517_102117_740732",
                log_file=str(process_log),
                started_at="2026-05-17T10:21:17",
                command="train.py --task Template-Redrhex-Direct-v0",
            )

            with patch("tools.training_panel.training_panel.processes.subprocess.check_output", return_value=""):
                registry.reconcile_stale_history()

            raw = next(record for record in history._load_data()["runs"] if record["id"] == "panel_20260517_102117_740732")
            runs = history.list_runs()
            self.assertEqual([run["id"] for run in runs], ["panel_20260517_102117_740732"])
            self.assertEqual(raw["status"], "running")
            self.assertEqual(raw["log_dir"], str(log_dir))

    def test_reconcile_failed_run_does_not_time_fallback_to_neighbor_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            history = HistoryStore(paths)
            process_log = paths.process_log_dir / "panel_failed.log"
            process_log.write_text(
                "Exact experiment name requested from command line: 2026-05-17_10-31-01\n"
                "CUDA error: out of memory\n",
                encoding="utf-8",
            )
            exit_file = paths.process_log_dir / "panel_failed.exit"
            exit_file.write_text("1", encoding="utf-8")
            neighbor_log = paths.rsl_rl_log_root / "2026-05-17_10-30-57_wheg_locomotion_reform_v1"
            neighbor_log.mkdir(parents=True)
            (neighbor_log / "model_19.pt").write_text("x", encoding="utf-8")
            history.add_run(
                {
                    "id": "panel_failed",
                    "source": "training_panel",
                    "status": "running",
                    "created_at": "2026-05-17T10:30:55",
                    "process_log": str(process_log),
                    "exit_file": str(exit_file),
                    "log_dir": None,
                }
            )
            registry = ProcessRegistry(paths, history)

            with patch("tools.training_panel.training_panel.processes.subprocess.check_output", return_value=""):
                registry.reconcile_stale_history()

            run = history.get_run("panel_failed")
            self.assertEqual(run["status"], "failed")
            self.assertEqual(run["returncode"], 1)
            self.assertIsNone(run["log_dir"])
            self.assertIsNone(run["latest_checkpoint"])

    def test_reconcile_stale_history_marks_missing_panel_process_interrupted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            history = HistoryStore(paths)
            history.add_run(
                {
                    "id": "panel_stale",
                    "source": "training_panel",
                    "status": "running",
                    "created_at": "2026-05-15T11:00:00",
                    "pid": 456,
                }
            )
            registry = ProcessRegistry(paths, history)
            with patch("tools.training_panel.training_panel.processes.subprocess.check_output", return_value=""):
                registry.reconcile_stale_history()
            self.assertEqual(history.get_run("panel_stale")["status"], "interrupted")


if __name__ == "__main__":
    unittest.main()
