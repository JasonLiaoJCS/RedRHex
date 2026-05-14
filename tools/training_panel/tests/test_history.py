import tempfile
import unittest
from pathlib import Path

from tools.training_panel.training_panel.config import PanelPaths
from tools.training_panel.training_panel.history import HistoryStore, latest_checkpoint


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


if __name__ == "__main__":
    unittest.main()

