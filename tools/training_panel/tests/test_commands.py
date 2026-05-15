import tempfile
import unittest
from pathlib import Path

from tools.training_panel.training_panel.commands import VideoParams, play_argv, shell_for_command
from tools.training_panel.training_panel.config import PanelPaths


class CommandTests(unittest.TestCase):
    def make_paths(self, root: Path) -> PanelPaths:
        return PanelPaths(
            repo_root=root,
            isaaclab_root=root / "IsaacLab",
            isaacsim_root=root / "isaacsim",
            conda_sh=root / "miniconda3" / "etc" / "profile.d" / "conda.sh",
            conda_env="env",
        )

    def test_shell_activates_conda_env_and_keeps_conda_lib_first(self):
        with tempfile.TemporaryDirectory() as tmp:
            shell = shell_for_command(self.make_paths(Path(tmp)), ["tensorboard", "--version"])
            self.assertIn("source ", shell)
            self.assertIn("conda activate env", shell)
            self.assertIn('export PATH="$CONDA_PREFIX/bin:$PATH"', shell)
            self.assertIn('export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"', shell)

    def test_play_argv_supports_headless_video_recording(self):
        params = VideoParams.from_preset("high")
        argv = play_argv(
            "/tmp/model_10.pt",
            device="cuda:0",
            headless=True,
            video=True,
            video_length=params.length,
            video_width=params.width,
            video_height=params.height,
            video_fps=params.fps,
            rendering_mode=params.rendering_mode,
        )
        self.assertIn("scripts/rsl_rl/play.py", argv)
        self.assertIn("--headless", argv)
        self.assertIn("--video", argv)
        self.assertIn("--video_length", argv)
        self.assertEqual(argv[argv.index("--video_length") + 1], "1200")
        self.assertEqual(argv[argv.index("--video_width") + 1], "1920")
        self.assertEqual(argv[argv.index("--video_height") + 1], "1080")
        self.assertEqual(argv[argv.index("--video_fps") + 1], "30")
        self.assertEqual(argv[argv.index("--rendering_mode") + 1], "quality")
        self.assertEqual(argv[argv.index("--checkpoint") + 1], "/tmp/model_10.pt")

    def test_video_default_is_high_quality(self):
        high = VideoParams.from_preset(None)
        self.assertEqual(high.preset, "high")
        self.assertEqual(high.width, 1920)
        self.assertEqual(high.height, 1080)
        self.assertEqual(high.length, 1200)
        self.assertEqual(high.rendering_mode, "quality")


if __name__ == "__main__":
    unittest.main()
