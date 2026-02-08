# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
RedRhex RHex-style Wheg Locomotion PPO 訓練配置 v3.0

★★★ 大刀闘斧改革版本 ★★★
- 增加 entropy_coef 促進探索
- 增加 learning_rate 加速學習
- 簡化網路架構避免過擬合

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
    """RedRhex RHex-style Wheg Locomotion PPO 訓練配置 v3.1"""

    num_steps_per_env = 24
    max_iterations = 2000
    save_interval = 100
    experiment_name = "redrhex_wheg"
    run_name = "wheg_locomotion_v3"
    logger = "tensorboard"

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.8,  # 降低初始噪音，避免後段 curriculum 發散
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        # ★ 簡化網路：從 [256,128,64] → [128,64]（避免過擬合）
        actor_hidden_dims=[128, 64],
        critic_hidden_dims=[128, 64],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,  # 避免策略標準差爆增
        num_learning_epochs=6,
        num_mini_batches=4,
        learning_rate=5.0e-4,  # 降低高階段整合時的不穩定
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
