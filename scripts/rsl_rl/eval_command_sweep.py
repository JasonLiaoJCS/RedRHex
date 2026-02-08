# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate a trained RSL-RL policy using locomotion acceptance metrics."""

import argparse
import csv
import math
import os
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip


parser = argparse.ArgumentParser(description="Evaluate RSL-RL policy with command sweep.")
parser.add_argument("--num_envs", type=int, default=256, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--sweep_steps", type=int, default=600, help="Evaluation steps per command.")
parser.add_argument("--warmup_steps", type=int, default=120, help="Warm-up steps per command.")
parser.add_argument("--csv", type=str, default=None, help="Optional command-table CSV path.")
parser.add_argument("--accept_duration_s", type=float, default=2.0, help="Minimum success duration for lateral/yaw tests.")
parser.add_argument("--accept_vy_abs", type=float, default=0.15, help="Lateral speed threshold for acceptance.")
parser.add_argument("--accept_wz_abs", type=float, default=0.40, help="Yaw rate threshold for acceptance.")
parser.add_argument("--accept_yaw_tilt_bound", type=float, default=0.60, help="Max |roll|/|pitch| bound during yaw acceptance.")
parser.add_argument("--accept_diag_sign_ratio", type=float, default=0.70, help="Required sign-match ratio for diagonal commands.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym
import torch

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import RedRhex.tasks  # noqa: F401


def circular_distance(a: torch.Tensor, b: float | torch.Tensor) -> torch.Tensor:
    return torch.abs(torch.atan2(torch.sin(a - b), torch.cos(a - b)))


def in_stance_phase(unwrapped_env, phase: torch.Tensor) -> torch.Tensor:
    if hasattr(unwrapped_env, "_in_stance_phase"):
        return unwrapped_env._in_stance_phase(phase)
    stance_start = float(unwrapped_env.stance_phase_start)
    stance_end = float(unwrapped_env.stance_phase_end)
    if stance_start < 0:
        stance_start += 2.0 * math.pi
        return (phase >= stance_start) | (phase < stance_end)
    return (phase >= stance_start) & (phase < stance_end)


def summarize_contact_hist(contact_hist: torch.Tensor) -> str:
    total = int(contact_hist.sum().item())
    if total == 0:
        return "n/a"
    parts = []
    for k in range(contact_hist.shape[0]):
        pct = 100.0 * float(contact_hist[k].item()) / float(total)
        parts.append(f"{k}:{pct:.1f}%")
    return " ".join(parts)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    env_cfg.log_dir = os.path.dirname(resume_path)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    unwrapped_env = env.unwrapped
    if hasattr(unwrapped_env, "external_control"):
        unwrapped_env.external_control = True

    command_set = [
        ("forward", (0.40, 0.00, 0.00)),
        ("left", (0.00, 0.30, 0.00)),
        ("right", (0.00, -0.30, 0.00)),
        ("diag_left", (0.30, 0.20, 0.00)),
        ("diag_right", (0.30, -0.20, 0.00)),
        ("yaw_ccw", (0.00, 0.00, 0.80)),
        ("yaw_cw", (0.00, 0.00, -0.80)),
    ]

    results = []
    num_envs = unwrapped_env.num_envs
    device = unwrapped_env.device
    total_steps = args_cli.warmup_steps + args_cli.sweep_steps
    step_dt = float(getattr(unwrapped_env, "step_dt", unwrapped_env.cfg.sim.dt * unwrapped_env.cfg.decimation))

    # Global acceptance accumulators
    sample_count = 0
    err_vx_sum = 0.0
    err_vy_sum = 0.0
    err_wz_sum = 0.0

    base_h_sum = 0.0
    base_h_sq_sum = 0.0
    pitch_sq_sum = 0.0
    roll_sq_sum = 0.0
    stability_count = 0

    fall_events = 0
    episode_ends = 0

    contact_hist = torch.zeros(7, dtype=torch.long)
    transition_samples = 0
    transition_contact_ge4 = 0

    action_rate_sum = 0.0
    effort_proxy_sum = 0.0
    energy_count = 0
    effort_proxy_name = "mean(|target_vel * omega|)"

    forward_phase_diff_sum = 0.0
    forward_phase_err_to_pi_sum = 0.0
    forward_stance_frac_sum = 0.0
    forward_stance_frac_err_sum = 0.0
    forward_stance_speed_sum = 0.0
    forward_swing_speed_sum = 0.0
    forward_count = 0

    for name, cmd in command_set:
        env.reset()
        obs = env.get_observations()

        cmd_tensor = torch.tensor(cmd, device=device, dtype=torch.float32).unsqueeze(0).repeat(num_envs, 1)
        cmd_err_vx = 0.0
        cmd_err_vy = 0.0
        cmd_err_wz = 0.0
        cmd_samples = 0
        cmd_success_vy_steps = 0
        cmd_success_wz_steps = 0
        cmd_diag_sign_match = 0
        cmd_diag_sign_total = 0
        cmd_fall_events = 0
        cmd_episode_ends = 0

        last_actions = None

        for step in range(total_steps):
            if hasattr(unwrapped_env, "commands"):
                unwrapped_env.commands[:] = cmd_tensor

            # Use no_grad instead of inference_mode: inference tensors can break subsequent env.reset() writes.
            with torch.no_grad():
                actions = policy(obs)
                obs, _, dones, _ = env.step(actions)
                policy_nn.reset(dones)

            if step < args_cli.warmup_steps:
                last_actions = actions.clone()
                continue

            actual_vx = unwrapped_env.base_lin_vel[:, 0]
            actual_vy = unwrapped_env.base_lin_vel[:, 1]
            actual_wz = unwrapped_env.base_ang_vel[:, 2]

            dvx = torch.abs(actual_vx - cmd[0])
            dvy = torch.abs(actual_vy - cmd[1])
            dwz = torch.abs(actual_wz - cmd[2])

            cmd_err_vx += dvx.sum().item()
            cmd_err_vy += dvy.sum().item()
            cmd_err_wz += dwz.sum().item()
            cmd_samples += num_envs

            err_vx_sum += dvx.sum().item()
            err_vy_sum += dvy.sum().item()
            err_wz_sum += dwz.sum().item()
            sample_count += num_envs

            # Stability
            base_h = unwrapped_env.robot.data.root_pos_w[:, 2]
            gravity_body = unwrapped_env.projected_gravity
            roll = torch.atan2(gravity_body[:, 1], -gravity_body[:, 2])
            pitch = torch.atan2(-gravity_body[:, 0], torch.sqrt(gravity_body[:, 1] ** 2 + gravity_body[:, 2] ** 2))

            base_h_sum += base_h.sum().item()
            base_h_sq_sum += (base_h**2).sum().item()
            pitch_sq_sum += (pitch**2).sum().item()
            roll_sq_sum += (roll**2).sum().item()
            stability_count += num_envs

            terminated = torch.zeros(num_envs, dtype=torch.bool, device=device)
            time_outs = torch.zeros(num_envs, dtype=torch.bool, device=device)
            if hasattr(unwrapped_env, "reset_terminated") and hasattr(unwrapped_env, "reset_time_outs"):
                terminated = unwrapped_env.reset_terminated
                time_outs = unwrapped_env.reset_time_outs
                fall_events += int(torch.count_nonzero(terminated).item())
                episode_ends += int(torch.count_nonzero(terminated | time_outs).item())
                cmd_fall_events += int(torch.count_nonzero(terminated).item())
                cmd_episode_ends += int(torch.count_nonzero(terminated | time_outs).item())

            # Contact statistics
            if hasattr(unwrapped_env, "_contact_count"):
                contact_count = torch.clamp(torch.round(unwrapped_env._contact_count), min=0, max=6).to(torch.long)
            elif hasattr(unwrapped_env, "_current_leg_in_stance"):
                contact_count = unwrapped_env._current_leg_in_stance.float().sum(dim=1).to(torch.long)
            else:
                contact_count = torch.zeros(num_envs, dtype=torch.long, device=device)
            contact_hist += torch.bincount(contact_count.cpu(), minlength=7)

            # Forward gait metrics
            if name == "forward" and hasattr(unwrapped_env, "_main_drive_indices"):
                main_pos = unwrapped_env.joint_pos[:, unwrapped_env._main_drive_indices]
                main_vel = unwrapped_env.joint_vel[:, unwrapped_env._main_drive_indices]
                direction_multiplier = unwrapped_env._direction_multiplier
                leg_phase = torch.remainder(main_pos * direction_multiplier, 2.0 * math.pi)
                leg_in_stance = in_stance_phase(unwrapped_env, leg_phase)

                phase_a = leg_phase[:, unwrapped_env._tripod_a_indices]
                phase_b = leg_phase[:, unwrapped_env._tripod_b_indices]
                mean_phase_a = torch.atan2(torch.sin(phase_a).mean(dim=1), torch.cos(phase_a).mean(dim=1))
                mean_phase_b = torch.atan2(torch.sin(phase_b).mean(dim=1), torch.cos(phase_b).mean(dim=1))

                phase_diff = circular_distance(mean_phase_a, mean_phase_b)
                forward_phase_diff_sum += phase_diff.mean().item()
                forward_phase_err_to_pi_sum += torch.abs(phase_diff - math.pi).mean().item()

                stance_fraction = leg_in_stance.float().mean(dim=1)
                forward_stance_frac_sum += stance_fraction.mean().item()
                forward_stance_frac_err_sum += torch.abs(stance_fraction - 0.65).mean().item()

                signed_speed = torch.abs(main_vel * direction_multiplier)
                stance_mask = leg_in_stance.float()
                swing_mask = (~leg_in_stance).float()
                stance_speed = (signed_speed * stance_mask).sum(dim=1) / stance_mask.sum(dim=1).clamp(min=1.0)
                swing_speed = (signed_speed * swing_mask).sum(dim=1) / swing_mask.sum(dim=1).clamp(min=1.0)

                forward_stance_speed_sum += stance_speed.mean().item()
                forward_swing_speed_sum += swing_speed.mean().item()
                forward_count += 1

                start_phase = unwrapped_env.stance_phase_start
                if start_phase < 0:
                    start_phase += 2.0 * math.pi
                end_phase = unwrapped_env.stance_phase_end
                transition_window = float(getattr(unwrapped_env.cfg, "forward_transition_window", 0.35))

                dist_a = torch.minimum(circular_distance(mean_phase_a, start_phase), circular_distance(mean_phase_a, end_phase))
                dist_b = torch.minimum(circular_distance(mean_phase_b, start_phase), circular_distance(mean_phase_b, end_phase))
                transition_mask = torch.minimum(dist_a, dist_b) < transition_window

                transition_samples += int(torch.count_nonzero(transition_mask).item())
                transition_contact_ge4 += int(torch.count_nonzero((contact_count >= 4) & transition_mask).item())

            # Energy / smoothness
            if last_actions is not None:
                action_rate = torch.linalg.vector_norm(actions - last_actions, dim=1)
                action_rate_sum += action_rate.mean().item()
            last_actions = actions.clone()

            # Per-command acceptance counters
            not_fallen = ~terminated
            if abs(cmd[1]) > 1e-5 and abs(cmd[0]) < 1e-5 and abs(cmd[2]) < 1e-5:
                cmd_success_vy_steps += int(torch.count_nonzero((torch.abs(actual_vy) > args_cli.accept_vy_abs) & not_fallen).item())

            if abs(cmd[2]) > 1e-5 and abs(cmd[0]) < 1e-5 and abs(cmd[1]) < 1e-5:
                yaw_ok = (
                    (torch.abs(actual_wz) > args_cli.accept_wz_abs)
                    & (torch.abs(roll) < args_cli.accept_yaw_tilt_bound)
                    & (torch.abs(pitch) < args_cli.accept_yaw_tilt_bound)
                    & not_fallen
                )
                cmd_success_wz_steps += int(torch.count_nonzero(yaw_ok).item())

            if abs(cmd[0]) > 1e-5 and abs(cmd[1]) > 1e-5:
                diag_sign_ok = (
                    (torch.sign(actual_vx) == torch.sign(torch.tensor(cmd[0], device=device)))
                    & (torch.sign(actual_vy) == torch.sign(torch.tensor(cmd[1], device=device)))
                )
                diag_mag_ok = (torch.abs(actual_vx) > 0.05) & (torch.abs(actual_vy) > 0.05)
                cmd_diag_sign_match += int(torch.count_nonzero(diag_sign_ok & diag_mag_ok).item())
                cmd_diag_sign_total += int(torch.count_nonzero(diag_mag_ok).item())

            if hasattr(unwrapped_env.robot.data, "applied_torque"):
                effort_proxy_name = "mean(|tau * omega|)"
                torques = unwrapped_env.robot.data.applied_torque[:, unwrapped_env._main_drive_indices]
                omegas = unwrapped_env.joint_vel[:, unwrapped_env._main_drive_indices]
                effort_proxy = torch.mean(torch.abs(torques * omegas), dim=1)
                effort_proxy_sum += effort_proxy.mean().item()
            elif hasattr(unwrapped_env, "_target_drive_vel"):
                effort_proxy_name = "mean(|target_vel * omega|)"
                omegas = unwrapped_env.joint_vel[:, unwrapped_env._main_drive_indices]
                effort_proxy = torch.mean(torch.abs(unwrapped_env._target_drive_vel * omegas), dim=1)
                effort_proxy_sum += effort_proxy.mean().item()
            else:
                effort_proxy_sum += 0.0

            energy_count += 1

        denom = float(max(1, cmd_samples))
        result = {
            "command": name,
            "cmd_vx": cmd[0],
            "cmd_vy": cmd[1],
            "cmd_wz": cmd[2],
            "mae_vx": cmd_err_vx / denom,
            "mae_vy": cmd_err_vy / denom,
            "mae_wz": cmd_err_wz / denom,
        }
        result["success_vy_duration_s"] = float(cmd_success_vy_steps) * step_dt / float(max(1, num_envs))
        result["success_wz_duration_s"] = float(cmd_success_wz_steps) * step_dt / float(max(1, num_envs))
        result["diag_sign_match_ratio"] = float(cmd_diag_sign_match) / float(max(1, cmd_diag_sign_total))
        result["fall_rate"] = float(cmd_fall_events) / float(max(1, cmd_episode_ends))

        if abs(cmd[1]) > 1e-5 and abs(cmd[0]) < 1e-5 and abs(cmd[2]) < 1e-5:
            result["accept_pass"] = (
                (result["success_vy_duration_s"] >= args_cli.accept_duration_s)
                and (result["fall_rate"] < 0.2)
            )
        elif abs(cmd[2]) > 1e-5 and abs(cmd[0]) < 1e-5 and abs(cmd[1]) < 1e-5:
            result["accept_pass"] = (
                (result["success_wz_duration_s"] >= args_cli.accept_duration_s)
                and (result["fall_rate"] < 0.2)
            )
        elif abs(cmd[0]) > 1e-5 and abs(cmd[1]) > 1e-5:
            result["accept_pass"] = result["diag_sign_match_ratio"] >= args_cli.accept_diag_sign_ratio
        else:
            result["accept_pass"] = True
        results.append(result)

    safe_samples = max(1, sample_count)
    mean_abs_vx = err_vx_sum / safe_samples
    mean_abs_vy = err_vy_sum / safe_samples
    mean_abs_wz = err_wz_sum / safe_samples

    safe_stability = max(1, stability_count)
    base_h_mean = base_h_sum / safe_stability
    base_h_var = max(0.0, base_h_sq_sum / safe_stability - base_h_mean * base_h_mean)
    base_h_std = math.sqrt(base_h_var)
    pitch_rms = math.sqrt(pitch_sq_sum / safe_stability)
    roll_rms = math.sqrt(roll_sq_sum / safe_stability)

    fall_rate = float(fall_events) / float(max(1, episode_ends))

    safe_forward = max(1, forward_count)
    phase_diff_mean = forward_phase_diff_sum / safe_forward
    phase_err_to_pi = forward_phase_err_to_pi_sum / safe_forward
    stance_fraction_mean = forward_stance_frac_sum / safe_forward
    stance_fraction_err = forward_stance_frac_err_sum / safe_forward
    stance_speed_mean = forward_stance_speed_sum / safe_forward
    swing_speed_mean = forward_swing_speed_sum / safe_forward
    swing_to_stance_ratio = swing_speed_mean / max(stance_speed_mean, 1e-6)

    transition_ratio_ge4 = float(transition_contact_ge4) / float(max(1, transition_samples))

    action_rate_mean = action_rate_sum / float(max(1, energy_count))
    effort_proxy_mean = effort_proxy_sum / float(max(1, energy_count))

    print("\n=== Command Tracking (MAE) ===")
    print(f"{'command':<12} {'cmd(vx,vy,wz)':<24} {'|vx-vx*|':>10} {'|vy-vy*|':>10} {'|wz-wz*|':>10}")
    for row in results:
        cmd_str = f"({row['cmd_vx']:.2f},{row['cmd_vy']:.2f},{row['cmd_wz']:.2f})"
        print(
            f"{row['command']:<12} {cmd_str:<24} "
            f"{row['mae_vx']:>10.4f} {row['mae_vy']:>10.4f} {row['mae_wz']:>10.4f}"
        )

    print("\n=== Skill Acceptance (Command-level) ===")
    for row in results:
        extra = ""
        if abs(row["cmd_vy"]) > 1e-5 and abs(row["cmd_vx"]) < 1e-5 and abs(row["cmd_wz"]) < 1e-5:
            extra = f"vy_success_s={row['success_vy_duration_s']:.2f}, fall_rate={row['fall_rate']:.3f}"
        elif abs(row["cmd_wz"]) > 1e-5 and abs(row["cmd_vx"]) < 1e-5 and abs(row["cmd_vy"]) < 1e-5:
            extra = f"wz_success_s={row['success_wz_duration_s']:.2f}, fall_rate={row['fall_rate']:.3f}"
        elif abs(row["cmd_vx"]) > 1e-5 and abs(row["cmd_vy"]) > 1e-5:
            extra = f"diag_sign_match={row['diag_sign_match_ratio']:.3f}"
        status = "PASS" if row["accept_pass"] else "FAIL"
        print(f"{row['command']:<12} {status:<4} {extra}")

    print("\n=== Acceptance Metrics Summary ===")
    print(f"tracking.mean|vx-vx_cmd|: {mean_abs_vx:.6f}")
    print(f"tracking.mean|vy-vy_cmd|: {mean_abs_vy:.6f}")
    print(f"tracking.mean|wz-wz_cmd|: {mean_abs_wz:.6f}")
    print(f"forward.phase_diff_mean(rad): {phase_diff_mean:.6f}")
    print(f"forward.phase_diff_abs_to_pi(rad): {phase_err_to_pi:.6f}")
    print(f"forward.stance_fraction_mean: {stance_fraction_mean:.6f}")
    print(f"forward.stance_fraction_abs_err_to_0.65: {stance_fraction_err:.6f}")
    print(f"forward.stance_speed_mean(rad/s): {stance_speed_mean:.6f}")
    print(f"forward.swing_speed_mean(rad/s): {swing_speed_mean:.6f}")
    print(f"forward.swing_to_stance_speed_ratio: {swing_to_stance_ratio:.6f}")
    print(f"stability.fall_rate: {fall_rate:.6f}")
    print(f"stability.base_height_mean(m): {base_h_mean:.6f}")
    print(f"stability.base_height_std(m): {base_h_std:.6f}")
    print(f"stability.base_height_var(m^2): {base_h_var:.6f}")
    print(f"stability.pitch_rms(rad): {pitch_rms:.6f}")
    print(f"stability.roll_rms(rad): {roll_rms:.6f}")
    print(f"contact.histogram: {summarize_contact_hist(contact_hist)}")
    print(f"contact.transition_ratio_ge4: {transition_ratio_ge4:.6f}")
    print(f"energy.action_rate_mean: {action_rate_mean:.6f}")
    print(f"energy.effort_proxy_mean [{effort_proxy_name}]: {effort_proxy_mean:.6f}")

    if args_cli.csv:
        csv_path = os.path.abspath(args_cli.csv)
        csv_dir = os.path.dirname(csv_path)
        if csv_dir:
            os.makedirs(csv_dir, exist_ok=True)

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "command",
                    "cmd_vx",
                    "cmd_vy",
                    "cmd_wz",
                    "mae_vx",
                    "mae_vy",
                    "mae_wz",
                    "success_vy_duration_s",
                    "success_wz_duration_s",
                    "diag_sign_match_ratio",
                    "fall_rate",
                    "accept_pass",
                ],
            )
            writer.writeheader()
            writer.writerows(results)

        summary_path = os.path.splitext(csv_path)[0] + "_summary.csv"
        summary_rows = [
            {"metric": "tracking.mean_abs_vx", "value": mean_abs_vx},
            {"metric": "tracking.mean_abs_vy", "value": mean_abs_vy},
            {"metric": "tracking.mean_abs_wz", "value": mean_abs_wz},
            {"metric": "forward.phase_diff_mean", "value": phase_diff_mean},
            {"metric": "forward.phase_diff_abs_to_pi", "value": phase_err_to_pi},
            {"metric": "forward.stance_fraction_mean", "value": stance_fraction_mean},
            {"metric": "forward.stance_fraction_abs_err_to_0.65", "value": stance_fraction_err},
            {"metric": "forward.stance_speed_mean", "value": stance_speed_mean},
            {"metric": "forward.swing_speed_mean", "value": swing_speed_mean},
            {"metric": "forward.swing_to_stance_speed_ratio", "value": swing_to_stance_ratio},
            {"metric": "stability.fall_rate", "value": fall_rate},
            {"metric": "stability.base_height_mean", "value": base_h_mean},
            {"metric": "stability.base_height_std", "value": base_h_std},
            {"metric": "stability.base_height_var", "value": base_h_var},
            {"metric": "stability.pitch_rms", "value": pitch_rms},
            {"metric": "stability.roll_rms", "value": roll_rms},
            {"metric": "contact.transition_ratio_ge4", "value": transition_ratio_ge4},
            {"metric": "energy.action_rate_mean", "value": action_rate_mean},
            {"metric": "energy.effort_proxy_mean", "value": effort_proxy_mean},
            {"metric": "energy.effort_proxy_name", "value": effort_proxy_name},
            {"metric": "contact.histogram", "value": summarize_contact_hist(contact_hist)},
        ]

        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["metric", "value"])
            writer.writeheader()
            writer.writerows(summary_rows)

        print(f"[INFO] Wrote command table: {csv_path}")
        print(f"[INFO] Wrote summary table: {summary_path}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
