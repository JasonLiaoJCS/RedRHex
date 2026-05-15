# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RedRhex RHex-style Wheg Locomotion PPO 訓練配置."""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlSymmetryCfg,
)

from .. import redrhex_symmetry


@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """RedRhex reform PPO config with history actor and asymmetric critic."""

    num_steps_per_env = 24
    max_iterations = 2500
    save_interval = 100
    experiment_name = "redrhex_wheg"
    run_name = "wheg_locomotion_reform_v1"
    logger = "tensorboard"
    clip_actions = 1.0
    obs_groups = {
        "policy": ["policy", "history"],
        "critic": ["policy", "history", "critic"],
    }

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.6,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.003,
        num_learning_epochs=6,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=True,
            data_augmentation_func=redrhex_symmetry.compute_symmetric_states,
        ),
    )


@configclass
class PPORunnerPrivilegedTeacherCfg(RslRlOnPolicyRunnerCfg):
    """Privileged teacher PPO config for later student distillation."""

    num_steps_per_env = 24
    max_iterations = 2500
    save_interval = 100
    experiment_name = "redrhex_wheg_teacher"
    run_name = "wheg_privileged_teacher_v1"
    logger = "tensorboard"
    clip_actions = 1.0
    obs_groups = {
        "policy": ["teacher"],
        "critic": ["teacher"],
    }

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.6,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.003,
        num_learning_epochs=6,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=True,
            data_augmentation_func=redrhex_symmetry.compute_symmetric_states,
        ),
    )


@configclass
class PPORunnerForwardFastCfg(RslRlOnPolicyRunnerCfg):
    """Forward-only fast convergence PPO config."""

    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 50
    experiment_name = "redrhex_forward_fast"
    run_name = "forward_fast_reform_v1"
    logger = "tensorboard"
    clip_actions = 1.0
    obs_groups = {
        "policy": ["policy", "history"],
        "critic": ["policy", "history", "critic"],
    }

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.55,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.003,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=True,
            data_augmentation_func=redrhex_symmetry.compute_symmetric_states,
        ),
    )


@configclass
class PPORunnerForwardFastPrivilegedTeacherCfg(RslRlOnPolicyRunnerCfg):
    """Forward-only privileged teacher PPO config for student distillation."""

    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 50
    experiment_name = "redrhex_forward_fast_teacher"
    run_name = "forward_fast_privileged_teacher_v1"
    logger = "tensorboard"
    clip_actions = 1.0
    obs_groups = {
        "policy": ["teacher"],
        "critic": ["teacher"],
    }

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.55,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.003,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=True,
            data_augmentation_func=redrhex_symmetry.compute_symmetric_states,
        ),
    )
