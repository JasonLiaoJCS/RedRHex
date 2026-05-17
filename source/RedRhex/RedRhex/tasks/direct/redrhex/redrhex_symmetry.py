# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from tensordict import TensorDict

__all__ = ["compute_symmetric_states"]


def _left_right_leg_indices(device: torch.device) -> torch.Tensor:
    return torch.tensor([3, 4, 5, 0, 1, 2], device=device, dtype=torch.long)


def _swap_left_right(values: torch.Tensor) -> torch.Tensor:
    return values.index_select(dim=-1, index=_left_right_leg_indices(values.device))


def _transform_policy_obs_left_right(obs: torch.Tensor) -> torch.Tensor:
    if obs.shape[-1] != 56:
        return obs.clone()
    obs = obs.clone()
    device = obs.device

    obs[:, 0:3] = obs[:, 0:3] * torch.tensor([1.0, -1.0, 1.0], device=device)
    obs[:, 3:6] = obs[:, 3:6] * torch.tensor([-1.0, 1.0, -1.0], device=device)
    obs[:, 6:9] = obs[:, 6:9] * torch.tensor([1.0, -1.0, 1.0], device=device)
    obs[:, 9:15] = -_swap_left_right(obs[:, 9:15])
    obs[:, 15:21] = _swap_left_right(obs[:, 15:21])
    obs[:, 21:27] = -_swap_left_right(obs[:, 21:27])
    obs[:, 27:33] = -_swap_left_right(obs[:, 27:33])
    obs[:, 33:39] = -_swap_left_right(obs[:, 33:39])
    obs[:, 39:42] = obs[:, 39:42] * torch.tensor([1.0, -1.0, -1.0], device=device)
    obs[:, 44:50] = -_swap_left_right(obs[:, 44:50])
    obs[:, 50:56] = -_swap_left_right(obs[:, 50:56])
    return obs


def _transform_history_obs_left_right(history_obs: torch.Tensor, policy_obs_dim: int) -> torch.Tensor:
    history_obs = history_obs.clone()
    batch_size = history_obs.shape[0]
    history_obs = history_obs.view(batch_size, -1, policy_obs_dim)
    history_obs = _transform_policy_obs_left_right(history_obs.reshape(-1, policy_obs_dim))
    return history_obs.view(batch_size, -1)


def _transform_critic_obs_left_right(critic_obs: torch.Tensor) -> torch.Tensor:
    if critic_obs.shape[-1] != 47:
        return critic_obs.clone()
    critic_obs = critic_obs.clone()
    critic_obs[:, 0:6] = -_swap_left_right(critic_obs[:, 0:6])
    critic_obs[:, 6:12] = -_swap_left_right(critic_obs[:, 6:12])
    critic_obs[:, 12:18] = _swap_left_right(critic_obs[:, 12:18])
    critic_obs[:, 18:24] = _swap_left_right(critic_obs[:, 18:24])
    critic_obs[:, 24:30] = _swap_left_right(critic_obs[:, 24:30])
    critic_obs[:, 30:36] = _swap_left_right(critic_obs[:, 30:36])
    return critic_obs


def _transform_teacher_obs_left_right(
    teacher_obs: torch.Tensor,
    policy_obs_dim: int,
    history_obs_dim: int,
) -> torch.Tensor:
    teacher_obs = teacher_obs.clone()
    policy_obs = teacher_obs[:, :policy_obs_dim]
    history_obs = teacher_obs[:, policy_obs_dim : policy_obs_dim + history_obs_dim]
    critic_obs = teacher_obs[:, policy_obs_dim + history_obs_dim :]
    policy_obs = _transform_policy_obs_left_right(policy_obs)
    history_obs = _transform_history_obs_left_right(history_obs, policy_obs_dim)
    critic_obs = _transform_critic_obs_left_right(critic_obs)
    return torch.cat([policy_obs, history_obs, critic_obs], dim=-1)


def _transform_actions_left_right(actions: torch.Tensor) -> torch.Tensor:
    if actions.shape[-1] != 12:
        return actions.clone()
    actions = actions.clone()
    actions[:, 0:6] = -_swap_left_right(actions[:, 0:6])
    actions[:, 6:12] = -_swap_left_right(actions[:, 6:12])
    return actions


@torch.no_grad()
def compute_symmetric_states(env, obs: TensorDict | None = None, actions: torch.Tensor | None = None):
    """Create left-right mirrored observations/actions for RedRhex."""

    policy_obs_dim = int(env.unwrapped.cfg.observation_space)
    history_obs_dim = int(getattr(env.unwrapped.cfg, "history_observation_space", 0))

    if obs is not None:
        batch_size = obs.batch_size[0]
        obs_aug = obs.repeat(2)
        for key in obs.keys():
            obs_aug[key][:batch_size] = obs[key]

        if "policy" in obs.keys():
            obs_aug["policy"][batch_size:] = _transform_policy_obs_left_right(obs["policy"])
        if "history" in obs.keys():
            obs_aug["history"][batch_size:] = _transform_history_obs_left_right(obs["history"], policy_obs_dim)
        if "critic" in obs.keys():
            obs_aug["critic"][batch_size:] = _transform_critic_obs_left_right(obs["critic"])
        if "teacher" in obs.keys():
            obs_aug["teacher"][batch_size:] = _transform_teacher_obs_left_right(
                obs["teacher"], policy_obs_dim, history_obs_dim
            )
    else:
        obs_aug = None

    if actions is not None:
        batch_size = actions.shape[0]
        actions_aug = torch.zeros(batch_size * 2, actions.shape[1], device=actions.device, dtype=actions.dtype)
        actions_aug[:batch_size] = actions
        actions_aug[batch_size:] = _transform_actions_left_right(actions)
    else:
        actions_aug = None

    return obs_aug, actions_aug
