import unittest

from tools.training_panel.training_panel.commands import TrainingParams, training_argv


class TrainingCommandTests(unittest.TestCase):
    def test_training_argv_smoke(self):
        params = TrainingParams(num_envs=4, max_iterations=1, headless=True)
        argv = training_argv(params)
        self.assertIn("--headless", argv)
        self.assertEqual(argv[argv.index("--num_envs") + 1], "4")
        self.assertEqual(argv[argv.index("--max_iterations") + 1], "1")

    def test_resume_requires_checkpoint(self):
        with self.assertRaises(ValueError):
            TrainingParams.from_dict({"resume": True})


if __name__ == "__main__":
    unittest.main()

