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

    def test_link_run_to_log_preserves_requester_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            requester_id = "11111111-1111-4111-8111-111111111111"
            log_dir = root / "logs" / "rsl_rl" / "redrhex_wheg" / "real_log"
            log_dir.mkdir(parents=True)
            store = HistoryStore(self.make_paths(root))
            store.add_run({
                "id": "panel_run",
                "source": "training_panel",
                "created_by": requester_id,
                "requester_label": "phone user",
                "params": {"requester_id": requester_id, "task": "Template-Redrhex-Direct-v0"},
            })

            linked = store.link_run_to_log("panel_run", str(log_dir), status="completed", returncode=0)
            listed = {item["id"]: item for item in store.list_runs()}

            self.assertEqual(linked["created_by"], requester_id)
            self.assertEqual(linked["params"]["requester_id"], requester_id)
            self.assertEqual(linked["requester_label"], "phone user")
            self.assertEqual(listed["panel_run"]["created_by"], requester_id)

    def test_list_runs_collapses_panel_and_log_name_duplicate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            log_dir = root / "logs" / "rsl_rl" / "redrhex_wheg" / "2026-05-17_01-18-16_wheg_locomotion_reform_v1"
            log_dir.mkdir(parents=True)
            (log_dir / "model_13.pt").write_text("x", encoding="utf-8")
            store = HistoryStore(self.make_paths(root))
            store.add_run(
                {
                    "id": "panel_20260517_011805",
                    "source": "training_panel",
                    "status": "completed",
                    "created_at": "2026-05-17T01:18:05",
                    "log_dir": str(log_dir),
                    "terrain_preset_id": "flat-debug",
                }
            )
            store.add_run(
                {
                    "id": log_dir.name,
                    "source": "training_panel",
                    "created_at": "2026-05-17T01:18:46",
                    "log_dir": None,
                    "display_name": "duplicate shell",
                }
            )

            runs = store.list_runs()
            ids = [run["id"] for run in runs]
            panel = next(run for run in runs if run["id"] == "panel_20260517_011805")

            self.assertEqual(ids.count("panel_20260517_011805"), 1)
            self.assertNotIn(log_dir.name, ids)
            self.assertEqual(panel["log_dir"], str(log_dir))
            self.assertEqual(panel["terrain_preset_id"], "flat-debug")
            self.assertTrue(panel["latest_checkpoint"].endswith("model_13.pt"))

    def test_running_panel_run_claims_discovered_log_by_exact_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = HistoryStore(self.make_paths(root))
            process_log = store.paths.process_log_dir / "panel_run.log"
            process_log.write_text(
                "Exact experiment name requested from command line: 2026-05-17_01-35-34\n",
                encoding="utf-8",
            )
            log_dir = root / "logs" / "rsl_rl" / "redrhex_wheg" / "2026-05-17_01-35-34_wheg_locomotion_reform_v1"
            log_dir.mkdir(parents=True)
            (log_dir / "model_4.pt").write_text("x", encoding="utf-8")
            store.add_run(
                {
                    "id": "panel_run",
                    "source": "training_panel",
                    "status": "running",
                    "created_at": "2026-05-17T01:35:27",
                    "process_log": str(process_log),
                    "log_dir": None,
                }
            )

            runs = store.list_runs()
            ids = [run["id"] for run in runs]
            panel = next(run for run in runs if run["id"] == "panel_run")

            self.assertEqual(ids, ["panel_run"])
            self.assertEqual(panel["status"], "running")
            self.assertEqual(panel["log_dir"], str(log_dir))
            self.assertTrue(panel["latest_checkpoint"].endswith("model_4.pt"))

    def test_running_panel_run_claims_discovered_log_by_explicit_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = HistoryStore(self.make_paths(root))
            log_dir = root / "logs" / "rsl_rl" / "redrhex_wheg" / "2026-05-17_01-35-34_wheg_locomotion_reform_v1"
            log_dir.mkdir(parents=True)
            process_log = store.paths.process_log_dir / "panel_run.log"
            process_log.write_text(f"Writing TensorBoard data under {log_dir}/events.out.tfevents.test\n", encoding="utf-8")
            store.add_run(
                {
                    "id": "panel_run",
                    "source": "training_panel",
                    "status": "running",
                    "created_at": "2026-05-17T01:35:27",
                    "process_log": str(process_log),
                    "log_dir": None,
                }
            )

            runs = store.list_runs()

            self.assertEqual([run["id"] for run in runs], ["panel_run"])
            self.assertEqual(runs[0]["log_dir"], str(log_dir))

    def test_running_panel_run_claims_fresh_discovered_log_before_log_mentions_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = HistoryStore(self.make_paths(root))
            process_log = store.paths.process_log_dir / "panel_run.log"
            process_log.write_text("Isaac startup is still loading...\n", encoding="utf-8")
            log_dir = (
                root
                / "logs"
                / "rsl_rl"
                / "redrhex_wheg"
                / "2026-05-17_10-21-24_wheg_locomotion_reform_v1"
            )
            log_dir.mkdir(parents=True)
            (log_dir / "model_0.pt").write_text("x", encoding="utf-8")
            store.add_run(
                {
                    "id": "panel_20260517_102117_740732",
                    "source": "training_panel",
                    "status": "running",
                    "created_at": "2026-05-17T10:21:17",
                    "process_log": str(process_log),
                    "log_dir": None,
                }
            )

            runs = store.list_runs()
            ids = [run["id"] for run in runs]
            panel = runs[0]

            self.assertEqual(ids, ["panel_20260517_102117_740732"])
            self.assertEqual(panel["status"], "running")
            self.assertEqual(panel["log_dir"], str(log_dir))
            self.assertTrue(panel["latest_checkpoint"].endswith("model_0.pt"))

    def test_failed_panel_run_does_not_display_mismatched_completed_log_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = HistoryStore(self.make_paths(root))
            log_dir = root / "logs" / "rsl_rl" / "redrhex_wheg" / "2026-05-17_10-30-57_wheg_locomotion_reform_v1"
            log_dir.mkdir(parents=True)
            (log_dir / "model_19.pt").write_text("x", encoding="utf-8")
            completed_log = store.paths.process_log_dir / "panel_completed.log"
            completed_log.write_text(
                "Exact experiment name requested from command line: 2026-05-17_10-30-57\n",
                encoding="utf-8",
            )
            failed_log = store.paths.process_log_dir / "panel_failed.log"
            failed_log.write_text(
                "Exact experiment name requested from command line: 2026-05-17_10-31-01\n"
                "CUDA error: out of memory\n",
                encoding="utf-8",
            )
            store.add_run(
                {
                    "id": "panel_completed",
                    "source": "training_panel",
                    "status": "completed",
                    "returncode": 0,
                    "created_at": "2026-05-17T10:30:50",
                    "process_log": str(completed_log),
                    "log_dir": str(log_dir),
                }
            )
            store.add_run(
                {
                    "id": "panel_failed",
                    "source": "training_panel",
                    "status": "failed",
                    "returncode": 1,
                    "created_at": "2026-05-17T10:30:55",
                    "process_log": str(failed_log),
                    "log_dir": str(log_dir),
                }
            )

            runs = {run["id"]: run for run in store.list_runs()}

            self.assertEqual(runs["panel_completed"]["log_dir"], str(log_dir))
            self.assertTrue(runs["panel_completed"]["latest_checkpoint"].endswith("model_19.pt"))
            self.assertIsNone(runs["panel_failed"]["log_dir"])
            self.assertIsNone(runs["panel_failed"]["latest_checkpoint"])

    def test_patch_discovered_id_updates_canonical_panel_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = HistoryStore(self.make_paths(root))
            log_dir = root / "logs" / "rsl_rl" / "redrhex_wheg" / "2026-05-17_01-35-34_wheg_locomotion_reform_v1"
            log_dir.mkdir(parents=True)
            process_log = store.paths.process_log_dir / "panel_run.log"
            process_log.write_text(
                "Exact experiment name requested from command line: 2026-05-17_01-35-34\n",
                encoding="utf-8",
            )
            store.add_run(
                {
                    "id": "panel_run",
                    "source": "training_panel",
                    "status": "running",
                    "created_at": "2026-05-17T01:35:27",
                    "process_log": str(process_log),
                    "log_dir": None,
                }
            )

            updated = store.patch_run_metadata(log_dir.name, folder="Good Runs", display_name="canonical name")
            records = store._load_data()["runs"]
            listed = {run["id"]: run for run in store.list_runs()}

            self.assertEqual(updated["id"], "panel_run")
            self.assertEqual(listed["panel_run"]["folder"], "Good Runs")
            self.assertEqual(listed["panel_run"]["display_name"], "canonical name")
            self.assertFalse(any(record.get("id") == log_dir.name for record in records))

    def test_set_note_discovered_id_updates_canonical_panel_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = HistoryStore(self.make_paths(root))
            log_dir = root / "logs" / "rsl_rl" / "redrhex_wheg" / "2026-05-17_01-35-34_wheg_locomotion_reform_v1"
            log_dir.mkdir(parents=True)
            store.add_run(
                {
                    "id": "panel_run",
                    "source": "training_panel",
                    "status": "running",
                    "created_at": "2026-05-17T01:35:27",
                    "log_dir": str(log_dir),
                }
            )

            store.set_note(log_dir.name, "team note")

            self.assertEqual(store.get_note("panel_run"), "team note")
            self.assertEqual(store.get_note(log_dir.name), "team note")
            self.assertFalse((store.paths.notes_dir / f"{log_dir.name}.md").exists())

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

    def test_deleted_run_tombstones_can_filter_to_current_delete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = HistoryStore(self.make_paths(root))
            old_log = root / "logs" / "rsl_rl" / "redrhex_wheg" / "old_run"
            new_log = root / "logs" / "rsl_rl" / "redrhex_wheg" / "new_run"
            old_log.mkdir(parents=True)
            new_log.mkdir(parents=True)
            store.patch_run_metadata("old_panel", log_dir=str(old_log), source="training_panel")
            store.patch_run_metadata("new_panel", log_dir=str(new_log), source="training_panel")
            store.delete_run("old_panel", confirm=True, delete_logs=False)
            store.delete_run("new_panel", confirm=True, delete_logs=False)

            current = store.deleted_run_tombstones(run_ids=["new_panel"])

            self.assertEqual([item["id"] for item in current], ["new_panel"])
            self.assertEqual(current[0]["log_dir_name"], "new_run")
            self.assertEqual(len(store.deleted_run_tombstones()), 2)

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

    def test_bulk_delete_tolerates_stale_discovered_alias_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = HistoryStore(self.make_paths(root))
            log_dir = root / "logs" / "rsl_rl" / "redrhex_wheg" / "shared_log"
            log_dir.mkdir(parents=True)
            (log_dir / "events.out.tfevents.test").write_text("x", encoding="utf-8")
            store.add_run({
                "id": "panel_run",
                "source": "training_panel",
                "created_at": "2026-05-17T10:00:00",
                "log_dir": str(log_dir),
            })

            result = store.bulk_delete_runs(["panel_run", "shared_log"], confirm=True)

            self.assertEqual(result["deleted_count"], 1)
            self.assertEqual(result["missing"], ["shared_log"])
            self.assertEqual(result["run_ids"], ["panel_run"])
            self.assertFalse(log_dir.exists())
            self.assertIsNone(store.get_run("panel_run"))
            tombstones = store.deleted_run_tombstones(run_ids=["shared_log"])
            self.assertTrue(any(item.get("log_dir_name") == "shared_log" for item in tombstones))

    def test_bulk_delete_deletes_shared_paths_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = HistoryStore(self.make_paths(root))
            process_log = root / "logs" / "training_panel" / "process_logs" / "shared.log"
            process_log.parent.mkdir(parents=True, exist_ok=True)
            process_log.write_text("x", encoding="utf-8")
            for run_id in ("run_a", "run_b"):
                store.add_run({
                    "id": run_id,
                    "source": "training_panel",
                    "created_at": "2026-05-17T10:00:00",
                    "process_log": str(process_log),
                })

            result = store.bulk_delete_runs(["run_a", "run_b"], confirm=True)

            self.assertEqual(result["deleted_count"], 2)
            self.assertEqual(result["deleted_paths"], [str(process_log)])
            self.assertFalse(process_log.exists())
            self.assertIsNone(store.get_run("run_a"))
            self.assertIsNone(store.get_run("run_b"))

    def test_bulk_delete_tombstones_each_requested_id_for_same_log_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = HistoryStore(self.make_paths(root))
            data = {"runs": [], "folders": [], "deleted_runs": []}
            log_dir = root / "logs" / "rsl_rl" / "redrhex_wheg" / "shared_log"

            store._remember_deleted_run(data, "panel_a", run={"source": "training_panel"}, log_dir=str(log_dir))
            store._remember_deleted_run(data, "panel_b", run={"source": "training_panel"}, log_dir=str(log_dir))

            self.assertEqual([item["id"] for item in data["deleted_runs"]], ["panel_a", "panel_b"])
            self.assertTrue(all(item["log_dir_name"] == "shared_log" for item in data["deleted_runs"]))

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

    def test_delete_running_guard_checks_log_dir_matches(self):
        class FakeProcesses:
            def running_for_run(self, run_id):
                return []

            def running_for_log_dir(self, log_dir):
                if str(log_dir).endswith("active_log"):
                    return [{"run_id": "panel_run", "kind": "training"}]
                return []

        class FakeHistory:
            def delete_preview(self, run_id):
                return {
                    "paths": [
                        {"kind": "rsl_rl_log_dir", "path": "/tmp/active_log", "is_dir": True},
                    ]
                }

        handler = object.__new__(PanelHandler)
        handler.state = type("FakeState", (), {"processes": FakeProcesses(), "history": FakeHistory()})()

        self.assertEqual(
            handler._running_for_run_or_log_dir("active_log"),
            [{"run_id": "panel_run", "kind": "training"}],
        )

    def test_bulk_delete_running_guard_checks_log_dir_matches(self):
        class FakeProcesses:
            def running_for_run(self, run_id):
                return []

            def running_for_log_dir(self, log_dir):
                if Path(str(log_dir)).name == "active_log_dir":
                    return [{"run_id": "panel_run", "kind": "training"}]
                return []

        class FakeHistory:
            def delete_preview(self, run_id):
                return {
                    "paths": [
                        {"kind": "rsl_rl_log_dir", "path": f"/tmp/{run_id}_dir", "is_dir": True},
                    ]
                }

        handler = object.__new__(PanelHandler)
        handler.state = type("FakeState", (), {"processes": FakeProcesses(), "history": FakeHistory()})()

        self.assertEqual(
            handler._running_by_run_or_log_dir(["inactive_log", "active_log"]),
            {"active_log": [{"run_id": "panel_run", "kind": "training"}]},
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
