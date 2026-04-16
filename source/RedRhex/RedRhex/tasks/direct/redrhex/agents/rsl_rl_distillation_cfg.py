# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RedRhex teacher-student distillation configs."""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlDistillationAlgorithmCfg,
    RslRlDistillationRunnerCfg,
    RslRlDistillationStudentTeacherCfg,
)


@configclass
class RedrhexDistillationRunnerCfg(RslRlDistillationRunnerCfg):
    """Two-stage distillation config from privileged teacher to deployable student."""

    num_steps_per_env = 24
    max_iterations = 800
    save_interval = 50
    experiment_name = "redrhex_wheg_distill"
    run_name = "wheg_student_distill_v1"
    logger = "tensorboard"
    clip_actions = 1.0
    obs_groups = {
        "policy": ["policy", "history"],
        "teacher": ["teacher"],
    }
    policy = RslRlDistillationStudentTeacherCfg(
        init_noise_std=0.6,
        noise_std_type="scalar",
        student_obs_normalization=True,
        teacher_obs_normalization=True,
        student_hidden_dims=[512, 256, 128],
        teacher_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=2,
        learning_rate=1.0e-3,
        gradient_length=12,
        max_grad_norm=1.0,
        loss_type="huber",
    )


@configclass
class RedrhexForwardFastDistillationRunnerCfg(RslRlDistillationRunnerCfg):
    """Forward-only distillation config for fast student refinement."""

    num_steps_per_env = 24
    max_iterations = 400
    save_interval = 50
    experiment_name = "redrhex_forward_fast_distill"
    run_name = "forward_fast_student_distill_v1"
    logger = "tensorboard"
    clip_actions = 1.0
    obs_groups = {
        "policy": ["policy", "history"],
        "teacher": ["teacher"],
    }
    policy = RslRlDistillationStudentTeacherCfg(
        init_noise_std=0.55,
        noise_std_type="scalar",
        student_obs_normalization=True,
        teacher_obs_normalization=True,
        student_hidden_dims=[256, 128, 128],
        teacher_hidden_dims=[256, 128, 128],
        activation="elu",
    )
    algorithm = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=2,
        learning_rate=1.0e-3,
        gradient_length=12,
        max_grad_norm=1.0,
        loss_type="huber",
    )
