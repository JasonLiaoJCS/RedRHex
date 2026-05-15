import unittest
from pathlib import Path

from tools.training_panel.training_panel.rewards import reward_file_index


class RewardIndexTests(unittest.TestCase):
    def test_finds_reward_scales_in_repo(self):
        repo = Path(__file__).resolve().parents[3]
        index = reward_file_index(repo)
        names = {item["name"] for item in index["reward_scales"]}
        self.assertIn("rew_scale_forward_vel", names)
        self.assertEqual(index["mode"], "read-only")


if __name__ == "__main__":
    unittest.main()

