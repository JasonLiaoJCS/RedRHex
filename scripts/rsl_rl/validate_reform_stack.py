# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Smoke validation for the RedRhex reform training stack."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPO_ROOT / "source" / "RedRhex"
for path in (str(REPO_ROOT), str(PACKAGE_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Validate RedRhex asymmetric critic/history/symmetry stack.")
parser.add_argument("--task", type=str, default="Template-Redrhex-Direct-v0", help="Gym task name.")
parser.add_argument("--num_envs", type=int, default=32, help="Number of environments for validation.")
parser.add_argument("--steps", type=int, default=32, help="Random-action rollout steps.")
parser.add_argument("--seed", type=int, default=42, help="Validation seed.")
parser.add_argument("--runner_smoke", action="store_true", default=False, help="Run 1 PPO iteration smoke test.")
parser.add_argument("--runner_steps", type=int, default=8, help="Number of rollout steps for PPO smoke test.")
parser.add_argument(
    "--distill_smoke",
    action="store_true",
    default=False,
    help="Run privileged-teacher PPO + distillation smoke test.",
)
parser.add_argument("--distill_steps", type=int, default=8, help="Number of rollout steps for distillation smoke test.")
parser.add_argument("--log_dir", type=str, default="/tmp/redrhex_reform_smoke", help="Runner smoke log directory.")
parser.add_argument("--json_out", type=str, default="/tmp/redrhex_validate_stats.json", help="Path to dump validation stats.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym
import torch

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
import RedRhex.tasks  # noqa: F401

from RedRhex.tasks.direct.redrhex.agents.rsl_rl_distillation_cfg import (
    RedrhexDistillationRunnerCfg,
    RedrhexForwardFastDistillationRunnerCfg,
)
from RedRhex.tasks.direct.redrhex.agents.rsl_rl_ppo_cfg import (
    PPORunnerCfg,
    PPORunnerForwardFastCfg,
    PPORunnerForwardFastPrivilegedTeacherCfg,
    PPORunnerPrivilegedTeacherCfg,
)
from RedRhex.tasks.direct.redrhex.redrhex_env_cfg import RedrhexEnvCfg, RedrhexForwardFastEnvCfg


def _build_cfgs(task_name: str):
    if "ForwardFast" in task_name:
        return (
            RedrhexForwardFastEnvCfg(),
            PPORunnerForwardFastCfg(),
            PPORunnerForwardFastPrivilegedTeacherCfg(),
            RedrhexForwardFastDistillationRunnerCfg(),
        )
    return RedrhexEnvCfg(), PPORunnerCfg(), PPORunnerPrivilegedTeacherCfg(), RedrhexDistillationRunnerCfg()


def _configure_cfgs(task_name: str):
    env_cfg, agent_cfg, teacher_cfg, distill_cfg = _build_cfgs(task_name)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    agent_cfg.device = args_cli.device if args_cli.device is not None else agent_cfg.device
    teacher_cfg.device = args_cli.device if args_cli.device is not None else teacher_cfg.device
    distill_cfg.device = args_cli.device if args_cli.device is not None else distill_cfg.device
    return env_cfg, agent_cfg, teacher_cfg, distill_cfg


def _assert_obs_shapes(obs_dict: dict[str, torch.Tensor], env_cfg) -> None:
    expected_policy = int(env_cfg.observation_space)
    expected_history = int(env_cfg.history_observation_space)
    expected_critic = int(env_cfg.critic_privileged_observation_space)
    expected_teacher = int(env_cfg.teacher_observation_space)

    assert obs_dict["policy"].shape[1] == expected_policy, obs_dict["policy"].shape
    assert obs_dict["history"].shape[1] == expected_history, obs_dict["history"].shape
    assert obs_dict["critic"].shape[1] == expected_critic, obs_dict["critic"].shape
    assert obs_dict["teacher"].shape[1] == expected_teacher, obs_dict["teacher"].shape


def _random_rollout(env, env_cfg, steps: int) -> dict[str, float]:
    obs_dict, _ = env.reset()
    _assert_obs_shapes(obs_dict, env_cfg)

    unwrapped = env.unwrapped
    max_abs_reward = 0.0
    for _ in range(steps):
        actions = 2.0 * torch.rand((unwrapped.num_envs, env_cfg.action_space), device=unwrapped.device) - 1.0
        obs_dict, reward, terminated, truncated, _ = env.step(actions)
        _assert_obs_shapes(obs_dict, env_cfg)
        assert torch.isfinite(obs_dict["policy"]).all()
        assert torch.isfinite(obs_dict["history"]).all()
        assert torch.isfinite(obs_dict["critic"]).all()
        assert torch.isfinite(obs_dict["teacher"]).all()
        assert torch.isfinite(reward).all()
        max_abs_reward = max(max_abs_reward, float(torch.abs(reward).max().item()))
        if torch.any(terminated | truncated):
            # Isaac Lab auto-resets terminated sub-envs; we only care that the rollout continues cleanly.
            pass

    return {
        "terrain_type": str(unwrapped.cfg.terrain.terrain_type),
        "fault_env_ratio": float(unwrapped._fault_mask.any(dim=1).float().mean().item()),
        "mean_fault_leg_count": float(unwrapped._fault_mask.float().sum(dim=1).mean().item()),
        "mean_main_strength": float(unwrapped._main_strength_scale.mean().item()),
        "mean_abad_strength": float(unwrapped._abad_strength_scale.mean().item()),
        "max_abs_reward": max_abs_reward,
        "history_dim": float(obs_dict["history"].shape[1]),
        "critic_dim": float(obs_dict["critic"].shape[1]),
        "teacher_dim": float(obs_dict["teacher"].shape[1]),
    }


def _runner_smoke(env, agent_cfg, log_dir: str) -> None:
    os.makedirs(log_dir, exist_ok=True)
    wrapped_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    agent_cfg.max_iterations = 1
    agent_cfg.num_steps_per_env = min(agent_cfg.num_steps_per_env, args_cli.runner_steps)
    runner = OnPolicyRunner(wrapped_env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.learn(num_learning_iterations=1, init_at_random_ep_len=False)


def _teacher_runner_smoke(env, teacher_cfg, log_dir: str) -> str:
    os.makedirs(log_dir, exist_ok=True)
    wrapped_env = RslRlVecEnvWrapper(env, clip_actions=teacher_cfg.clip_actions)

    teacher_cfg.max_iterations = 1
    teacher_cfg.num_steps_per_env = min(teacher_cfg.num_steps_per_env, args_cli.runner_steps)
    runner = OnPolicyRunner(wrapped_env, teacher_cfg.to_dict(), log_dir=log_dir, device=teacher_cfg.device)
    runner.learn(num_learning_iterations=1, init_at_random_ep_len=False)

    teacher_ckpt = os.path.join(log_dir, "teacher_smoke.pt")
    runner.save(teacher_ckpt)
    return teacher_ckpt


def _distillation_smoke(env, distill_cfg, teacher_checkpoint: str, log_dir: str) -> str:
    del env, distill_cfg
    os.makedirs(log_dir, exist_ok=True)

    helper_script = REPO_ROOT / "scripts" / "rsl_rl" / "validate_distillation_stack.py"
    json_out = os.path.join(log_dir, "distill_stats.json")
    cmd = [
        sys.executable,
        str(helper_script),
        "--task",
        args_cli.task,
        "--num_envs",
        str(args_cli.num_envs),
        "--steps",
        str(args_cli.distill_steps),
        "--seed",
        str(args_cli.seed),
        "--teacher_ckpt",
        teacher_checkpoint,
        "--log_dir",
        log_dir,
        "--json_out",
        json_out,
    ]
    if args_cli.headless:
        cmd.append("--headless")
    if args_cli.device is not None:
        cmd.extend(["--device", str(args_cli.device)])

    env_vars = os.environ.copy()
    env_vars.setdefault("TERM", "xterm")
    env_vars.setdefault("PYTHONUNBUFFERED", "1")
    subprocess.run(cmd, check=True, env=env_vars)

    with open(json_out, "r", encoding="utf-8") as f:
        distill_stats = json.load(f)
    return str(distill_stats["student_ckpt"])


def main():
    env_cfg, agent_cfg, teacher_cfg, distill_cfg = _configure_cfgs(args_cli.task)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    stats = _random_rollout(env, env_cfg, args_cli.steps)

    print("[validate_reform_stack] Random rollout smoke stats:")
    for key, value in stats.items():
        print(f"  - {key}: {value}")

    if args_cli.runner_smoke:
        _runner_smoke(env, agent_cfg, args_cli.log_dir)
        stats["runner_smoke"] = 1.0
        print("[validate_reform_stack] PPO runner smoke test completed.")
    else:
        stats["runner_smoke"] = 0.0

    if args_cli.distill_smoke:
        teacher_log_dir = os.path.join(args_cli.log_dir, "teacher")
        distill_log_dir = os.path.join(args_cli.log_dir, "distill")
        teacher_ckpt = _teacher_runner_smoke(env, teacher_cfg, teacher_log_dir)
        student_ckpt = _distillation_smoke(env, distill_cfg, teacher_ckpt, distill_log_dir)

        stats["distill_smoke"] = 1.0
        print(f"[validate_reform_stack] Distillation smoke test completed.")
        print(f"[validate_reform_stack] Teacher checkpoint: {teacher_ckpt}")
        print(f"[validate_reform_stack] Student checkpoint: {student_ckpt}")
    else:
        stats["distill_smoke"] = 0.0

    with open(args_cli.json_out, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, sort_keys=True)
    print(f"[validate_reform_stack] Wrote stats to {args_cli.json_out}")
    if env is not None:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
