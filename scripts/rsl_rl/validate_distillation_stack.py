# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Standalone smoke validation for RedRhex distillation."""

import argparse
import json
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPO_ROOT / "source" / "RedRhex"
for path in (str(REPO_ROOT), str(PACKAGE_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Validate RedRhex teacher-student distillation stack.")
parser.add_argument("--task", type=str, default="Template-Redrhex-Direct-v0", help="Gym task name.")
parser.add_argument("--num_envs", type=int, default=32, help="Number of environments for validation.")
parser.add_argument("--steps", type=int, default=8, help="Number of rollout steps for distillation smoke test.")
parser.add_argument("--seed", type=int, default=42, help="Validation seed.")
parser.add_argument("--teacher_ckpt", type=str, required=True, help="Path to privileged teacher checkpoint.")
parser.add_argument("--log_dir", type=str, default="/tmp/redrhex_reform_distill", help="Log directory.")
parser.add_argument(
    "--json_out",
    type=str,
    default="/tmp/redrhex_validate_distill_only_stats.json",
    help="Path to dump validation stats.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym

from rsl_rl.runners import DistillationRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
import RedRhex.tasks  # noqa: F401

from RedRhex.tasks.direct.redrhex.agents.rsl_rl_distillation_cfg import (
    RedrhexDistillationRunnerCfg,
    RedrhexForwardFastDistillationRunnerCfg,
)
from RedRhex.tasks.direct.redrhex.redrhex_env_cfg import RedrhexEnvCfg, RedrhexForwardFastEnvCfg


def _build_cfgs(task_name: str):
    if "ForwardFast" in task_name:
        return RedrhexForwardFastEnvCfg(), RedrhexForwardFastDistillationRunnerCfg()
    return RedrhexEnvCfg(), RedrhexDistillationRunnerCfg()


def main():
    env_cfg, distill_cfg = _build_cfgs(args_cli.task)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    distill_cfg.device = args_cli.device if args_cli.device is not None else distill_cfg.device
    distill_cfg.max_iterations = 1
    distill_cfg.num_steps_per_env = min(distill_cfg.num_steps_per_env, args_cli.steps)

    os.makedirs(args_cli.log_dir, exist_ok=True)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    wrapped_env = RslRlVecEnvWrapper(env, clip_actions=distill_cfg.clip_actions)

    runner = DistillationRunner(wrapped_env, distill_cfg.to_dict(), log_dir=args_cli.log_dir, device=distill_cfg.device)
    runner.load(args_cli.teacher_ckpt, load_optimizer=False)
    runner.learn(num_learning_iterations=1, init_at_random_ep_len=False)

    student_ckpt = os.path.join(args_cli.log_dir, "distill_smoke.pt")
    runner.save(student_ckpt)

    stats = {
        "distill_smoke": 1.0,
        "teacher_ckpt": args_cli.teacher_ckpt,
        "student_ckpt": student_ckpt,
    }
    with open(args_cli.json_out, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, sort_keys=True)
    print("[validate_distillation_stack] Distillation smoke test completed.")
    print(f"[validate_distillation_stack] Wrote stats to {args_cli.json_out}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
