# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
RedRhex RHex-style Wheg Locomotion PPO 訓練配置

動作空間 (12):
- [0:6] 主驅動速度調整
- [6:12] ABAD 位置

觀測空間 (56):
- base_lin_vel (3)
- base_ang_vel (3)
- projected_gravity (3)
- main_drive_pos_sin (6)
- main_drive_pos_cos (6)
- main_drive_vel (6)
- abad_pos (6)
- abad_vel (6)
- velocity_command (3)
- gait_phase (2)
- last_actions (12)
"""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """RedRhex RHex-style Wheg Locomotion PPO 訓練配置"""

    num_steps_per_env = 24
    max_iterations = 2000
    save_interval = 100
    experiment_name = "redrhex_wheg"
    run_name = "wheg_locomotion"
    logger = "tensorboard"

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.8,  # 稍低的初始噪聲，因為動作範圍較小
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        # 網路大小適合 12 DOF 動作和 56 維觀測
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,  # 較低的熵係數，因為動作空間較小
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,  # 稍低的學習率以獲得更穩定的訓練
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )