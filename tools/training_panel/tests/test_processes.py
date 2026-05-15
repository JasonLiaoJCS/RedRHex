import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from tools.training_panel.training_panel.commands import VideoParams
from tools.training_panel.training_panel.config import PanelPaths
from tools.training_panel.training_panel.history import HistoryStore
from tools.training_panel.training_panel.processes import EXTERNAL_TRAINING_ID_PREFIX, ProcessRegistry


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

    def test_play_process_debug_streams_log_tail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            registry = ProcessRegistry(paths, HistoryStore(paths))
            result = registry.start_play("run_one", "/tmp/checkpoint.pt", device="cpu")
            try:
                debug = registry.get_process_debug(result["id"])
                self.assertIsNotNone(debug)
                self.assertEqual(debug["kind"], "play")
                self.assertIsNone(debug["returncode"])
                self.assertEqual(debug["source_run_id"], "run_one")
                self.assertIn("scripts/rsl_rl/play.py", debug["command"])
                self.assertIn("fake isaaclab", debug["log_tail"])
            finally:
                proc = registry._processes.get(result["id"])
                registry.stop(result["id"])
                if proc:
                    proc.wait(timeout=8)
                time.sleep(0.1)

    def test_video_recording_process_uses_headless_video_flags(self):
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
