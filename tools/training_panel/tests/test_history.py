import os
import tempfile
import unittest
from pathlib import Path

from tools.training_panel.training_panel.config import PanelPaths
from tools.training_panel.training_panel.history import HistoryStore, latest_checkpoint, latest_onnx, latest_video, tail_file
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

    def test_latest_onnx_discovers_exported_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            exported = run / "exported"
            exported.mkdir()
            (exported / "policy.onnx").write_text("onnx", encoding="utf-8")
            self.assertTrue(latest_onnx(run).endswith("exported/policy.onnx"))

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

    def test_rename_folder_updates_assigned_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "logs" / "rsl_rl" / "redrhex_wheg" / "2026_run"
            run.mkdir(parents=True)
            (run / "model_0.pt").write_text("x", encoding="utf-8")
            store = HistoryStore(self.make_paths(root))
            store.assign_runs_to_folder(["2026_run"], "Drafts")

            result = store.rename_folder("Drafts", "Reviewed")
            listed = {item["id"]: item for item in store.list_runs()}

            self.assertTrue(result["renamed"])
            self.assertEqual(result["moved_count"], 1)
            self.assertIn("Reviewed", store.get_folders())
            self.assertNotIn("Drafts", store.get_folders())
            self.assertEqual(listed["2026_run"]["folder"], "Reviewed")

    def test_rename_folder_rejects_duplicate_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = HistoryStore(self.make_paths(Path(tmp)))
            store.create_folder("A")
            store.create_folder("B")
            with self.assertRaises(ValueError):
                store.rename_folder("A", "B")

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

    def test_delete_run_can_use_boolean_confirmation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "logs" / "rsl_rl" / "redrhex_wheg" / "failed_run"
            run.mkdir(parents=True)
            (run / "events.out.tfevents.test").write_text("x", encoding="utf-8")
            store = HistoryStore(self.make_paths(root))

            result = store.delete_run("failed_run", confirm=True)

            self.assertTrue(result["deleted"])
            self.assertFalse(run.exists())

    def test_delete_run_tombstone_prevents_metadata_recreation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = HistoryStore(self.make_paths(root))
            store.add_run({"id": "failed_panel_run", "source": "training_panel", "created_at": "2026-05-16T12:00:00"})

            result = store.delete_run("failed_panel_run", confirm=True)
            store.patch_run_metadata("failed_panel_run", display_name="remote rename")
            store.set_note("failed_panel_run", "remote note")

            self.assertTrue(result["deleted"])
            self.assertIsNone(store.get_run("failed_panel_run"))
            self.assertEqual(store.get_note("failed_panel_run"), "")
            self.assertFalse(any(record.get("id") == "failed_panel_run" for record in store._load_data()["runs"]))

    def test_delete_run_removes_panel_sidecar_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = HistoryStore(self.make_paths(root))
            process_log = root / "logs" / "training_panel" / "process_logs" / "panel_run.log"
            exit_file = root / "logs" / "training_panel" / "process_logs" / "panel_run.exit"
            video_log = root / "logs" / "training_panel" / "process_logs" / "video_run.log"
            for path in (process_log, exit_file, video_log):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("x", encoding="utf-8")
            store.add_run(
                {
                    "id": "failed_panel_run",
                    "source": "training_panel",
                    "created_at": "2026-05-16T12:00:00",
                    "process_log": str(process_log),
                    "exit_file": str(exit_file),
                    "video_process_log": str(video_log),
                }
            )

            preview = store.delete_preview("failed_panel_run")
            kinds = {item["kind"] for item in preview["paths"]}
            result = store.delete_run("failed_panel_run", confirm=True)

            self.assertIn("panel_process_log", kinds)
            self.assertIn("panel_exit_file", kinds)
            self.assertIn("panel_video_process_log", kinds)
            self.assertEqual(len(result["deleted_paths"]), 3)
            self.assertFalse(process_log.exists())
            self.assertFalse(exit_file.exists())
            self.assertFalse(video_log.exists())

    def test_bulk_delete_preview_and_confirm_delete_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for run_id in ("run_a", "run_b"):
                run = root / "logs" / "rsl_rl" / "redrhex_wheg" / run_id
                run.mkdir(parents=True)
                (run / "events.out.tfevents.test").write_text("x", encoding="utf-8")
            store = HistoryStore(self.make_paths(root))

            preview = store.bulk_delete_preview(["run_a", "run_b"], delete_logs=True)
            self.assertEqual(preview["run_count"], 2)
            self.assertEqual(preview["path_count"], 2)

            with self.assertRaises(ValueError):
                store.bulk_delete_runs(["run_a", "run_b"], confirm=False)

            result = store.bulk_delete_runs(["run_a", "run_b"], confirm=True)
            self.assertEqual(result["deleted_count"], 2)
            self.assertEqual(store.get_run("run_a"), None)
            self.assertEqual(store.get_run("run_b"), None)

    def test_bulk_delete_running_guard_groups_active_processes(self):
        class FakeProcesses:
            def running_for_run(self, run_id):
                if run_id == "run_b":
                    return [{"id": "proc_b", "source_run_id": "run_b", "kind": "training"}]
                return []

        handler = object.__new__(PanelHandler)
        handler.state = type("FakeState", (), {"processes": FakeProcesses()})()

        self.assertEqual(
            handler._running_by_run(["run_a", "run_b"]),
            {"run_b": [{"id": "proc_b", "source_run_id": "run_b", "kind": "training"}]},
        )

    def test_compact_preview_keeps_highest_iteration_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "logs" / "rsl_rl" / "redrhex_wheg" / "compact_run"
            run.mkdir(parents=True)
            (run / "model_0.pt").write_text("old", encoding="utf-8")
            (run / "model_10.pt").write_text("new", encoding="utf-8")
            (run / "model_2.pt").write_text("mid", encoding="utf-8")
            store = HistoryStore(self.make_paths(root))

            preview = store.compact_preview("compact_run")

            self.assertTrue(preview["kept_checkpoint"].endswith("model_10.pt"))
            self.assertEqual(preview["delete_count"], 2)
            self.assertEqual([item["iteration"] for item in preview["delete_paths"]], [0, 2])

    def test_compact_run_deletes_only_old_top_level_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "logs" / "rsl_rl" / "redrhex_wheg" / "compact_run"
            run.mkdir(parents=True)
            (run / "model_0.pt").write_text("old", encoding="utf-8")
            (run / "model_10.pt").write_text("new", encoding="utf-8")
            (run / "events.out.tfevents.test").write_text("event", encoding="utf-8")
            params = run / "params"
            params.mkdir()
            (params / "env.yaml").write_text("params", encoding="utf-8")
            video_dir = run / "videos" / "play"
            video_dir.mkdir(parents=True)
            (video_dir / "rl-video-step-0.mp4").write_text("video", encoding="utf-8")
            exported = run / "exported"
            exported.mkdir()
            (exported / "policy.pt").write_text("jit", encoding="utf-8")
            (exported / "policy.onnx").write_text("onnx", encoding="utf-8")
            store = HistoryStore(self.make_paths(root))
            store.set_note("compact_run", "keep this")

            with self.assertRaises(ValueError):
                store.compact_run("compact_run", confirmation="wrong")

            result = store.compact_run("compact_run", confirmation="compact_run")

            self.assertTrue(result["compacted"])
            self.assertFalse((run / "model_0.pt").exists())
            self.assertTrue((run / "model_10.pt").exists())
            self.assertTrue((run / "events.out.tfevents.test").exists())
            self.assertTrue((params / "env.yaml").exists())
            self.assertTrue((video_dir / "rl-video-step-0.mp4").exists())
            self.assertTrue((exported / "policy.pt").exists())
            self.assertTrue((exported / "policy.onnx").exists())
            self.assertEqual(store.get_note("compact_run"), "keep this")


if __name__ == "__main__":
    unittest.main()
