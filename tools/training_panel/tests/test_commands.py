import tempfile
import unittest
from pathlib import Path

from tools.training_panel.training_panel.commands import TrainingParams, VideoParams, export_onnx_argv, play_argv, shell_for_command, training_argv
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

    def test_play_argv_supports_terrain_replay_and_follow_camera(self):
        argv = play_argv(
            "/tmp/model_10.pt",
            terrain_override_file="/tmp/terrain.json",
            camera_follow_robot=True,
            camera_eye=(-3.0, -2.4, 1.6),
            camera_lookat=(0.45, 0.0, 0.35),
        )
        self.assertEqual(argv[argv.index("--terrain_override_file") + 1], "/tmp/terrain.json")
        self.assertIn("--camera_follow_robot", argv)
        self.assertEqual(argv[argv.index("--camera_eye") + 1 : argv.index("--camera_eye") + 4], ["-3.0", "-2.4", "1.6"])
        self.assertEqual(argv[argv.index("--camera_lookat") + 1 : argv.index("--camera_lookat") + 4], ["0.45", "0.0", "0.35"])

    def test_video_default_is_high_quality(self):
        high = VideoParams.from_preset(None)
        self.assertEqual(high.preset, "high")
        self.assertEqual(high.width, 1920)
        self.assertEqual(high.height, 1080)
        self.assertEqual(high.length, 1200)
        self.assertEqual(high.rendering_mode, "quality")

    def test_export_onnx_argv_uses_headless_export_only_play(self):
        argv = export_onnx_argv("/tmp/model_10.pt", device="cuda:0")
        self.assertIn("scripts/rsl_rl/play.py", argv)
        self.assertIn("--headless", argv)
        self.assertIn("--export_policy_only", argv)
        self.assertEqual(argv[argv.index("--device") + 1], "cuda:0")
        self.assertEqual(argv[argv.index("--checkpoint") + 1], "/tmp/model_10.pt")

    def test_training_params_accept_terrain_overrides(self):
        params = TrainingParams.from_dict(
            {
                "task": "Template-Redrhex-Direct-v0",
                "num_envs": 4,
                "max_iterations": 1,
                "device": "cuda:0",
                "terrain_preset_id": "flat-debug",
                "terrain_overrides": {
                    "terrain.terrain_type": "plane",
                    "terrain_curriculum_enable": False,
                    "terrain_curriculum_levels": [0.0],
                },
            }
        )
        self.assertEqual(params.terrain_preset_id, "flat-debug")
        self.assertEqual(params.terrain_overrides["terrain.terrain_type"], "plane")
        self.assertEqual(params.terrain_overrides["terrain_curriculum_enable"], False)
        self.assertEqual(params.terrain_overrides["terrain_curriculum_levels"], [0.0])

    def test_training_params_preserve_tweak_metadata_without_changing_argv(self):
        params = TrainingParams.from_dict(
            {
                "task": "Template-Redrhex-Direct-v0",
                "num_envs": 4,
                "max_iterations": 8,
                "device": "cuda:0",
                "tweak_source_run_id": "panel_123",
                "tweak_source_label": "Baseline trial",
            }
        )
        self.assertEqual(params.tweak_source_run_id, "panel_123")
        self.assertEqual(params.tweak_source_label, "Baseline trial")
        self.assertNotIn("tweak_source_run_id", " ".join(training_argv(params)))

    def test_training_params_preserve_requester_without_changing_argv(self):
        params = TrainingParams.from_dict(
            {
                "task": "Template-Redrhex-Direct-v0",
                "num_envs": 4,
                "max_iterations": 8,
                "device": "cuda:0",
                "requester_id": "11111111-1111-4111-8111-111111111111",
                "requester_label": "Jason",
            }
        )
        self.assertEqual(params.requester_id, "11111111-1111-4111-8111-111111111111")
        self.assertEqual(params.requester_label, "Jason")
        self.assertNotIn("requester_id", " ".join(training_argv(params)))

    def test_training_params_accept_display_name_without_changing_argv(self):
        params = TrainingParams.from_dict(
            {
                "task": "Template-Redrhex-Direct-v0",
                "num_envs": 4,
                "max_iterations": 8,
                "device": "cuda:0",
                "display_name": "  stair warmup  ",
            }
        )
        self.assertEqual(params.display_name, "stair warmup")
        self.assertEqual(params.to_dict()["display_name"], "stair warmup")
        self.assertNotIn("stair warmup", " ".join(training_argv(params)))

    def test_training_params_preserve_folder_and_client_request_id_without_changing_argv(self):
        params = TrainingParams.from_dict(
            {
                "task": "Template-Redrhex-Direct-v0",
                "num_envs": 4,
                "max_iterations": 8,
                "device": "cuda:0",
                "folder": "  tests  ",
                "client_request_id": "child-123",
            }
        )
        self.assertEqual(params.folder, "tests")
        self.assertEqual(params.client_request_id, "child-123")
        argv = " ".join(training_argv(params))
        self.assertNotIn("tests", argv)
        self.assertNotIn("child-123", argv)

    def test_training_params_reject_display_name_over_limit(self):
        with self.assertRaises(ValueError):
            TrainingParams.from_dict(
                {
                    "task": "Template-Redrhex-Direct-v0",
                    "num_envs": 4,
                    "max_iterations": 8,
                    "device": "cuda:0",
                    "display_name": "x" * 121,
                }
            )


if __name__ == "__main__":
    unittest.main()
