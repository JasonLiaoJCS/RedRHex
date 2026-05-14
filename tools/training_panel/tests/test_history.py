import tempfile
import unittest
from pathlib import Path

from tools.training_panel.training_panel.config import PanelPaths
from tools.training_panel.training_panel.history import HistoryStore, latest_checkpoint, tail_file


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

    def test_rename_discovered_run_preserves_log_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "logs" / "rsl_rl" / "redrhex_wheg" / "2026_run"
            run.mkdir(parents=True)
            (run / "model_0.pt").write_text("x", encoding="utf-8")
            store = HistoryStore(self.make_paths(root))
            store.rename_run("2026_run", "first useful gait")
            renamed = store.get_run("2026_run")
            self.assertEqual(renamed["display_name"], "first useful gait")
            self.assertTrue(renamed["latest_checkpoint"].endswith("model_0.pt"))

    def test_tail_file_limits_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "process.log"
            path.write_text("abcdef", encoding="utf-8")
            self.assertEqual(tail_file(path, max_chars=3), "def")


if __name__ == "__main__":
    unittest.main()
