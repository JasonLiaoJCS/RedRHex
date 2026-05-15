import os
import tempfile
import unittest
from pathlib import Path

from tools.training_panel.training_panel.config import PanelPaths
from tools.training_panel.training_panel.history import HistoryStore, latest_checkpoint, latest_video, tail_file
from tools.training_panel.training_panel.server import PanelHandler, PanelState, route_id


class HistoryTests(unittest.TestCase):
    def make_paths(self, root: Path) -> PanelPaths:
        return PanelPaths(
            repo_root=root,
            isaaclab_root=root / "IsaacLab",
            isaacsim_root=root / "isaacsim",
            conda_sh=root / "conda.sh",
            conda_env="env",
        )

    def test_notes_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = HistoryStore(self.make_paths(Path(tmp)))
            store.set_note("run one", "observed stable gait")
            self.assertEqual(store.get_note("run one"), "observed stable gait")

    def test_latest_checkpoint_uses_highest_iteration(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            (run / "model_0.pt").write_text("x", encoding="utf-8")
            (run / "model_99.pt").write_text("x", encoding="utf-8")
            self.assertTrue(latest_checkpoint(run).endswith("model_99.pt"))

    def test_latest_video_uses_newest_play_mp4(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            video_dir = run / "videos" / "play"
            video_dir.mkdir(parents=True)
            old = video_dir / "rl-video-step-0.mp4"
            new = video_dir / "rl-video-step-600.mp4"
            old.write_text("old", encoding="utf-8")
            new.write_text("new", encoding="utf-8")
            os.utime(old, (100, 100))
            os.utime(new, (200, 200))
            self.assertTrue(latest_video(run).endswith("rl-video-step-600.mp4"))

    def test_rename_discovered_run_preserves_log_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "logs" / "rsl_rl" / "redrhex_wheg" / "2026_run"
            run.mkdir(parents=True)
            (run / "model_0.pt").write_text("x", encoding="utf-8")
            video_dir = run / "videos" / "play"
            video_dir.mkdir(parents=True)
            (video_dir / "rl-video-step-0.mp4").write_text("video", encoding="utf-8")
            store = HistoryStore(self.make_paths(root))
            returned = store.rename_run("2026_run", "first useful gait")
            renamed = store.get_run("2026_run")
            self.assertEqual(returned["display_name"], "first useful gait")
            self.assertTrue(returned["latest_checkpoint"].endswith("model_0.pt"))
            self.assertTrue(returned["latest_video"].endswith("rl-video-step-0.mp4"))
            self.assertEqual(renamed["display_name"], "first useful gait")
            self.assertTrue(renamed["latest_checkpoint"].endswith("model_0.pt"))
            self.assertTrue(renamed["has_video"])

    def test_create_empty_folder_persists_without_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = HistoryStore(self.make_paths(Path(tmp)))
            created = store.create_folder("Ideas")
            self.assertEqual(created, "Ideas")
            self.assertIn("Ideas", store.get_folders())
            self.assertEqual(store.list_runs(), [])

    def test_delete_folder_moves_runs_to_uncategorized(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "logs" / "rsl_rl" / "redrhex_wheg" / "2026_run"
            run.mkdir(parents=True)
            (run / "model_0.pt").write_text("x", encoding="utf-8")
            store = HistoryStore(self.make_paths(root))
            store.create_folder("Archive")
            store.assign_runs_to_folder(["2026_run"], "Archive")

            result = store.delete_folder("Archive")
            listed = {item["id"]: item for item in store.list_runs()}

            self.assertTrue(result["removed"])
            self.assertEqual(result["moved_count"], 1)
            self.assertNotIn("Archive", store.get_folders())
            self.assertIsNone(listed["2026_run"].get("folder"))

    def test_assign_discovered_run_to_folder_persists_after_list_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "logs" / "rsl_rl" / "redrhex_wheg" / "2026_run"
            run.mkdir(parents=True)
            (run / "model_0.pt").write_text("x", encoding="utf-8")
            store = HistoryStore(self.make_paths(root))

            updated = store.assign_runs_to_folder(["2026_run"], "Good Runs")
            listed = {item["id"]: item for item in store.list_runs()}

            self.assertEqual(updated[0]["folder"], "Good Runs")
            self.assertEqual(listed["2026_run"]["folder"], "Good Runs")
            self.assertTrue(listed["2026_run"]["latest_checkpoint"].endswith("model_0.pt"))

    def test_bulk_assign_and_clear_folders_through_handler(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for run_id in ("run_a", "run_b"):
                run = root / "logs" / "rsl_rl" / "redrhex_wheg" / run_id
                run.mkdir(parents=True)
                (run / "model_0.pt").write_text("x", encoding="utf-8")
            handler = object.__new__(PanelHandler)
            handler.state = PanelState(self.make_paths(root))

            assigned = handler._assign_folders({"run_ids": ["run_a", "run_b"], "folder": "Batch"})
            self.assertEqual(assigned["folder"], "Batch")
            self.assertEqual(set(assigned["run_ids"]), {"run_a", "run_b"})
            self.assertIn("Batch", assigned["folders"])
            listed = {item["id"]: item for item in handler.state.history.list_runs()}
            self.assertEqual(listed["run_a"]["folder"], "Batch")
            self.assertEqual(listed["run_b"]["folder"], "Batch")

            cleared = handler._assign_folders({"run_ids": ["run_a", "run_b"], "folder": None})
            self.assertIsNone(cleared["folder"])
            listed = {item["id"]: item for item in handler.state.history.list_runs()}
            self.assertIsNone(listed["run_a"].get("folder"))
            self.assertIsNone(listed["run_b"].get("folder"))
            self.assertIn("Batch", handler.state.history.get_folders())

    def test_tail_file_limits_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "process.log"
            path.write_text("abcdef", encoding="utf-8")
            self.assertEqual(tail_file(path, max_chars=3), "def")

    def test_route_id_decodes_encoded_ids(self):
        self.assertEqual(route_id("/api/runs/run%20one/notes"), "run one")

    def test_open_location_rejects_paths_outside_log_roots(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            outside = root / "outside.txt"
            outside.write_text("x", encoding="utf-8")
            handler = object.__new__(PanelHandler)
            handler.state = PanelState(self.make_paths(root))
            with self.assertRaises(ValueError):
                handler._open_location(str(outside))

    def test_delete_run_requires_confirmation_and_removes_repo_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "logs" / "rsl_rl" / "redrhex_wheg" / "failed_run"
            run.mkdir(parents=True)
            (run / "events.out.tfevents.test").write_text("x", encoding="utf-8")
            store = HistoryStore(self.make_paths(root))
            store.set_note("failed_run", "bad start")

            with self.assertRaises(ValueError):
                store.delete_run("failed_run", confirmation="wrong")

            preview = store.delete_preview("failed_run")
            self.assertEqual(preview["requires_confirmation"], "failed_run")
            self.assertTrue(any(item["kind"] == "rsl_rl_log_dir" for item in preview["paths"]))

            result = store.delete_run("failed_run", confirmation="failed_run")
            self.assertTrue(result["deleted"])
            self.assertFalse(run.exists())
            self.assertEqual(store.get_run("failed_run"), None)
            self.assertEqual(store.get_note("failed_run"), "")


if __name__ == "__main__":
    unittest.main()
