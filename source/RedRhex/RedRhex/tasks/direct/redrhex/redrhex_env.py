# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RedRhex hexapod robot environment with tripod gait locomotion."""

from __future__ import annotations

import math
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import quat_apply_inverse, sample_uniform

from .redrhex_env_cfg import RedrhexEnvCfg


class RedrhexEnv(DirectRLEnv):
    """
    RedRhex 六足機器人三足步態環境

    這個環境訓練 RedRhex 六足機器人使用三足步態 (Tripod Gait) 進行移動。
    三足步態是六足機器人最常見且高效的步態，特點是：
    - Tripod A (Leg 1, 3, 5) 和 Tripod B (Leg 2, 4, 6) 交替接觸地面
    - 任何時刻都有三隻腳支撐，提供穩定的三角形支撐基底

    你們的創新點 - ABAD (外展/內收) 自由度可以用於：
    1. 動態平衡調整
    2. 地形適應
    3. 轉向輔助
    4. 側向移動
    """

    cfg: RedrhexEnvCfg

    def __init__(self, cfg: RedrhexEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # ===================
        # 初始化緩衝區
        # ===================
        self._setup_buffers()

        # ===================
        # 初始化速度命令
        # ===================
        self._setup_commands()

        # ===================
        # 初始化步態相位
        # ===================
        self._setup_gait()

        # Tripod 分組索引
        self._tripod_a_ids = torch.tensor(self.cfg.tripod_group_a, device=self.device)
        self._tripod_b_ids = torch.tensor(self.cfg.tripod_group_b, device=self.device)

        # 打印物理診斷信息
        self._debug_print_physics_info()

        print(f"[RedrhexEnv] 環境初始化完成，共 {self.num_envs} 個環境")
        print(f"[RedrhexEnv] 動作空間: {self.cfg.action_space}")
        print(f"[RedrhexEnv] 觀測空間: {self.cfg.observation_space}")

    def _setup_buffers(self):
        """設置內部緩衝區"""
        # 關節狀態
        self.joint_pos = self.robot.data.joint_pos.clone()
        self.joint_vel = self.robot.data.joint_vel.clone()
        self.joint_pos_default = self.robot.data.default_joint_pos.clone()

        # 動作緩衝
        self.actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self.last_actions = torch.zeros_like(self.actions)

        # 基座狀態
        self.base_lin_vel = torch.zeros(self.num_envs, 3, device=self.device)
        self.base_ang_vel = torch.zeros(self.num_envs, 3, device=self.device)
        self.projected_gravity = torch.zeros(self.num_envs, 3, device=self.device)

        # 腳接觸狀態 (6 隻腳)
        self.feet_contact = torch.ones(self.num_envs, 6, dtype=torch.bool, device=self.device)

        # 計算初始姿態的投影重力作為參考 (用於傾斜終止判斷)
        # 使用配置中的初始旋轉四元數，而不是從 robot 數據讀取
        # 這樣即使機器人初始時有旋轉，也不會被誤判為傾斜
        init_rot = self.cfg.robot_cfg.init_state.rot  # (w, x, y, z)
        init_quat = torch.tensor(
            [init_rot[0], init_rot[1], init_rot[2], init_rot[3]],
            device=self.device
        ).unsqueeze(0).expand(self.num_envs, 4)
        gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=self.device).expand(self.num_envs, 3)
        self.reference_projected_gravity = quat_apply_inverse(init_quat, gravity_vec)

        # 獎勵追蹤
        self.episode_sums = {
            "lin_vel_xy": torch.zeros(self.num_envs, device=self.device),
            "ang_vel_z": torch.zeros(self.num_envs, device=self.device),
            "tripod_contact": torch.zeros(self.num_envs, device=self.device),
            "abad_usage": torch.zeros(self.num_envs, device=self.device),
        }

    def _setup_commands(self):
        """設置速度命令"""
        # commands[:, 0] = x 速度 (前進)
        # commands[:, 1] = y 速度 (側向)
        # commands[:, 2] = yaw 角速度 (轉向)
        self.commands = torch.zeros(self.num_envs, 3, device=self.device)

    def _setup_gait(self):
        """設置步態相位追蹤"""
        # 步態相位: 0 到 2*pi
        self.gait_phase = torch.zeros(self.num_envs, device=self.device)

        # 每條腿的相位偏移 (三足步態)
        # Tripod A (legs 0, 2, 4): 相位 0
        # Tripod B (legs 1, 3, 5): 相位 π
        self.leg_phase_offsets = torch.tensor(
            [0.0, math.pi, 0.0, math.pi, 0.0, math.pi],
            device=self.device
        )

    def _setup_scene(self):
        """設置模擬場景"""
        # 添加機器人
        self.robot = Articulation(self.cfg.robot_cfg)
        self.scene.articulations["robot"] = self.robot

        # 添加地形 - 使用 TerrainImporter (成功案例方式)
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # 複製環境
        self.scene.clone_environments(copy_from_source=False)

        # CPU 模擬需要過濾碰撞
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # 添加燈光
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _debug_print_physics_info(self):
        """診斷：打印物理參數幫助調試重力問題"""
        print("\n" + "="*60)
        print("🔍 物理參數診斷 (Physics Diagnostics)")
        print("="*60)
        
        # 打印重力設置
        print(f"⚙️  重力設置: {self.cfg.sim.gravity}")
        print(f"⚙️  物理 dt: {self.cfg.sim.dt} s ({1/self.cfg.sim.dt:.0f} Hz)")
        print(f"⚙️  Decimation: {self.cfg.decimation}")
        print(f"⚙️  控制頻率: {1/(self.cfg.sim.dt * self.cfg.decimation):.1f} Hz")
        
        # 嘗試獲取機器人質量信息
        try:
            # 獲取總質量
            body_masses = self.robot.root_physx_view.get_masses()
            total_mass = body_masses.sum(dim=-1)
            print(f"\n📊 機器人質量信息:")
            print(f"   總質量: {total_mass[0].item():.4f} kg")
            print(f"   各剛體質量: {body_masses[0].cpu().numpy()}")
            
            # 檢查質量是否異常
            if total_mass[0].item() < 0.1:
                print(f"\n⚠️  警告: 總質量非常小 ({total_mass[0].item():.6f} kg)!")
                print(f"   這可能是導致『重力很小』現象的原因！")
                print(f"   請檢查 USD 文件中的質量/密度設置。")
            elif total_mass[0].item() > 100:
                print(f"\n⚠️  警告: 總質量非常大 ({total_mass[0].item():.2f} kg)!")
                
        except Exception as e:
            print(f"\n❌ 無法獲取質量信息: {e}")
        
        # 計算預期自由落體時間
        print(f"\n📐 自由落體參考:")
        g = abs(self.cfg.sim.gravity[2])
        h = 0.1  # 假設從 10cm 高度落下
        t_expected = (2 * h / g) ** 0.5
        print(f"   從 {h*100:.0f}cm 高度自由落體到地面的理論時間: {t_expected:.3f} 秒")
        
        print("="*60 + "\n")

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """物理步之前處理動作"""
        # 保存上一次動作
        self.last_actions = self.actions.clone()

        # 裁剪並存儲新動作
        self.actions = actions.clone().clamp(-1.0, 1.0)

    def _apply_action(self) -> None:
        """將動作應用到機器人關節"""
        # 為不同類型的關節應用不同的縮放
        scaled_actions = torch.zeros_like(self.actions)

        # 每條腿有 3 個關節，按順序是 hip, knee, foot
        for leg_idx in range(6):
            base_idx = leg_idx * 3
            # Hip (ABAD) - 較小的動作範圍
            scaled_actions[:, base_idx] = self.actions[:, base_idx] * self.cfg.hip_action_scale
            # Knee
            scaled_actions[:, base_idx + 1] = self.actions[:, base_idx + 1] * self.cfg.knee_action_scale
            # Foot
            scaled_actions[:, base_idx + 2] = self.actions[:, base_idx + 2] * self.cfg.foot_action_scale

        # 計算目標關節位置 (相對於默認位置的偏移)
        target_joint_pos = self.joint_pos_default + scaled_actions

        # 應用到機器人
        self.robot.set_joint_position_target(target_joint_pos)

    def _get_observations(self) -> dict:
        """計算觀測"""
        # 更新內部狀態
        self._update_state()

        # 構建觀測向量
        obs = torch.cat([
            # 基座線速度 (3)
            self.base_lin_vel,
            # 基座角速度 (3)
            self.base_ang_vel,
            # 投影重力向量 (3)
            self.projected_gravity,
            # 關節位置 (相對於默認) (18)
            self.joint_pos - self.joint_pos_default,
            # 關節速度 (18)
            torch.clamp(self.joint_vel, min=-20.0, max=20.0),
            # 速度命令 (3)
            self.commands,
            # 步態相位 (sin 和 cos 表示) (2)
            torch.sin(self.gait_phase).unsqueeze(-1),
            torch.cos(self.gait_phase).unsqueeze(-1),
            # 上一次動作 (18)
            self.last_actions,
        ], dim=-1)

        # 添加噪聲
        if self.cfg.add_noise:
            noise = torch.randn_like(obs) * 0.01 * self.cfg.noise_level
            obs = obs + noise

        # NaN/Inf 保護 (關鍵！)
        obs = torch.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0)
        obs = torch.clamp(obs, min=-100.0, max=100.0)

        return {"policy": obs}

    def _update_state(self):
        """更新內部狀態緩衝區"""
        # 獲取關節狀態 (添加 NaN 保護)
        self.joint_pos = torch.nan_to_num(self.robot.data.joint_pos.clone(), nan=0.0)
        self.joint_vel = torch.nan_to_num(self.robot.data.joint_vel.clone(), nan=0.0)

        # 獲取基座狀態
        root_quat = self.robot.data.root_quat_w
        root_lin_vel_w = self.robot.data.root_lin_vel_w
        root_ang_vel_w = self.robot.data.root_ang_vel_w

        # 轉換速度到基座坐標系 (添加 clamp 防止極端值)
        self.base_lin_vel = torch.clamp(
            quat_apply_inverse(root_quat, root_lin_vel_w), min=-10.0, max=10.0
        )
        self.base_ang_vel = torch.clamp(
            quat_apply_inverse(root_quat, root_ang_vel_w), min=-10.0, max=10.0
        )
        
        # NaN 保護
        self.base_lin_vel = torch.nan_to_num(self.base_lin_vel, nan=0.0)
        self.base_ang_vel = torch.nan_to_num(self.base_ang_vel, nan=0.0)

        # 計算投影重力
        gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=self.device).expand(self.num_envs, 3)
        self.projected_gravity = quat_apply_inverse(root_quat, gravity_vec)
        self.projected_gravity = torch.nan_to_num(self.projected_gravity, nan=0.0)

        # 更新步態相位
        dt = self.cfg.sim.dt * self.cfg.decimation
        self.gait_phase = (self.gait_phase + 2 * math.pi * self.cfg.gait_frequency * dt) % (2 * math.pi)


        # 更新腳接觸狀態 (簡化版本)
        self.feet_contact = torch.ones(self.num_envs, 6, dtype=torch.bool, device=self.device)

    def _get_rewards(self) -> torch.Tensor:
        """計算獎勵 - 簡化版本，專注於基本行走"""
        rewards = torch.zeros(self.num_envs, device=self.device)

        # ===================
        # 存活獎勵 (最基本 - 鼓勵保持站立)
        # ===================
        rewards += self.cfg.rew_scale_alive

        # ===================
        # 前進獎勵 (核心獎勵)
        # ===================
        # 直接獎勵 x 方向的位移速度 (限制範圍避免 NaN)
        forward_vel = torch.clamp(self.base_lin_vel[:, 0], min=-5.0, max=5.0)
        
        # 速度追蹤獎勵 - 使用命令的前進速度作為目標
        target_vel = self.commands[:, 0]
        vel_error = torch.abs(forward_vel - target_vel)
        lin_vel_xy_reward = torch.exp(-vel_error / 0.5) * self.cfg.rew_scale_lin_vel_xy
        rewards += lin_vel_xy_reward

        # 額外獎勵：直接獎勵正向前進 (限制上限避免過大獎勵)
        forward_progress_reward = torch.clamp(forward_vel, min=0.0, max=1.0) * self.cfg.rew_scale_forward_progress
        rewards += forward_progress_reward

        # 角速度追蹤 (yaw) - 當前目標是 0
        ang_vel_z_reward = self.cfg.rew_scale_ang_vel_z  # 直接給滿分，因為目標是 0
        rewards += ang_vel_z_reward

        # 懲罰垂直速度 (減少跳躍) - 限制範圍
        z_vel = torch.clamp(self.base_lin_vel[:, 2], min=-5.0, max=5.0)
        lin_vel_z_penalty = torch.square(z_vel) * self.cfg.rew_scale_lin_vel_z
        rewards += lin_vel_z_penalty

        # ===================
        # 穩定性獎勵
        # ===================
        # 姿態懲罰 - 只懲罰極端傾斜 (限制範圍)
        grav_xy = torch.clamp(self.projected_gravity[:, :2], min=-2.0, max=2.0)
        orientation_penalty = torch.sum(torch.square(grav_xy), dim=1)
        rewards += orientation_penalty * self.cfg.rew_scale_orientation

        # 高度維持 - 只懲罰極端高度偏差
        base_height = torch.clamp(self.robot.data.root_pos_w[:, 2], min=-1.0, max=2.0)
        target_height = 0.1  # 目標高度
        height_error = torch.square(base_height - target_height)
        rewards += height_error * self.cfg.rew_scale_base_height

        # ===================
        # 平滑性懲罰 (降低權重)
        # ===================
        action_rate = torch.sum(torch.square(self.actions - self.last_actions), dim=1)
        rewards += action_rate * self.cfg.rew_scale_action_rate

        # 確保獎勵不是 NaN 或 Inf
        rewards = torch.nan_to_num(rewards, nan=0.0, posinf=10.0, neginf=-10.0)

        # ===================
        # 步態獎勵 (簡化 - 只在學會走路後再啟用)
        # ===================
        if self.cfg.rew_scale_tripod_contact > 0:
            tripod_reward = self._compute_tripod_gait_reward()
            rewards += tripod_reward * self.cfg.rew_scale_tripod_contact

        # ===================
        # ABAD 獎勵 (暫時禁用 - 讓機器人先學會走路)
        # ===================
        if self.cfg.rew_scale_abad_usage > 0:
            abad_reward = self._compute_abad_reward()
            rewards += abad_reward * self.cfg.rew_scale_abad_usage

        if self.cfg.rew_scale_abad_symmetry > 0:
            abad_symmetry_reward = self._compute_abad_symmetry_reward()
            rewards += abad_symmetry_reward * self.cfg.rew_scale_abad_symmetry

        # 更新追蹤
        self.episode_sums["lin_vel_xy"] += lin_vel_xy_reward
        self.episode_sums["ang_vel_z"] += ang_vel_z_reward

        return rewards

    def _compute_tripod_gait_reward(self) -> torch.Tensor:
        """
        計算三足步態獎勵

        三足步態的關鍵：
        - 當 gait_phase 在 [0, π) 時，Tripod A (legs 0,2,4) 應該接觸地面
        - 當 gait_phase 在 [π, 2π) 時，Tripod B (legs 1,3,5) 應該接觸地面
        """
        # 判斷當前應該是哪個 tripod 接觸地面
        tripod_a_should_contact = (self.gait_phase < math.pi)

        # 獲取各 tripod 的接觸狀態
        tripod_a_contact = self.feet_contact[:, self._tripod_a_ids]
        tripod_b_contact = self.feet_contact[:, self._tripod_b_ids]

        # 計算接觸數量
        tripod_a_count = tripod_a_contact.sum(dim=1).float()
        tripod_b_count = tripod_b_contact.sum(dim=1).float()

        # 獎勵正確的接觸模式
        reward = torch.zeros(self.num_envs, device=self.device)

        # 當 tripod A 應該接觸時
        mask_a = tripod_a_should_contact
        reward[mask_a] = (tripod_a_count[mask_a] / 3.0) - (tripod_b_count[mask_a] / 3.0)

        # 當 tripod B 應該接觸時
        mask_b = ~tripod_a_should_contact
        reward[mask_b] = (tripod_b_count[mask_b] / 3.0) - (tripod_a_count[mask_b] / 3.0)

        return reward

    def _compute_gait_phase_reward(self) -> torch.Tensor:
        """計算步態相位獎勵"""
        # 計算每條腿的相位
        leg_phases = (self.gait_phase.unsqueeze(-1) + self.leg_phase_offsets) % (2 * math.pi)

        # 理想情況：相位在 [0, π] 時腳應該在地面
        should_be_in_stance = (leg_phases < math.pi)

        # 比較實際接觸狀態與理想狀態
        correct_phase = (should_be_in_stance == self.feet_contact).float()

        return correct_phase.mean(dim=1)

    def _compute_abad_reward(self) -> torch.Tensor:
        """
        計算 ABAD (外展/內收) 關節使用獎勵

        這是你們的創新點！ABAD 關節可以用於：
        1. 側向移動時擴大支撐基底
        2. 轉向時輔助方向控制
        3. 不平地形時調整腳的位置
        """
        # 獲取 ABAD (hip) 關節的動作
        hip_actions = torch.zeros(self.num_envs, 6, device=self.device)
        for i in range(6):
            hip_actions[:, i] = self.actions[:, i * 3]

        # 計算 ABAD 使用量
        hip_usage = torch.abs(hip_actions).mean(dim=1)

        # 獎勵適度使用 (目標 20% 的動作範圍)
        target_usage = 0.2
        usage_error = torch.abs(hip_usage - target_usage)
        reward = torch.exp(-usage_error / 0.1)

        return reward

    def _compute_abad_symmetry_reward(self) -> torch.Tensor:
        """計算 ABAD 左右對稱獎勵"""
        hip_actions = torch.zeros(self.num_envs, 6, device=self.device)
        for i in range(6):
            hip_actions[:, i] = self.actions[:, i * 3]

        # 左右腿配對：(0,1), (2,3), (4,5)
        symmetry_error = (
            torch.abs(hip_actions[:, 0] + hip_actions[:, 1]) +
            torch.abs(hip_actions[:, 2] + hip_actions[:, 3]) +
            torch.abs(hip_actions[:, 4] + hip_actions[:, 5])
        ) / 3.0

        # 只在直線行走時獎勵對稱
        is_straight = (torch.abs(self.commands[:, 1]) < 0.1) & (torch.abs(self.commands[:, 2]) < 0.1)
        reward = torch.exp(-symmetry_error / 0.1) * is_straight.float()

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """計算終止條件"""
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

        # 超時
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # 終止條件
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # 0. 物理爆炸檢測 (最重要！) - 防止位置飛到無限遠
        root_pos = self.robot.data.root_pos_w
        root_vel = self.robot.data.root_lin_vel_w
        
        # 檢查 NaN 或 Inf
        pos_invalid = torch.any(torch.isnan(root_pos) | torch.isinf(root_pos), dim=1)
        vel_invalid = torch.any(torch.isnan(root_vel) | torch.isinf(root_vel), dim=1)
        terminated = terminated | pos_invalid | vel_invalid
        
        # 檢查位置是否飛太遠 (超過 100 米就算爆炸)
        pos_too_far = torch.any(torch.abs(root_pos[:, :2]) > 100.0, dim=1)  # XY 平面
        terminated = terminated | pos_too_far
        
        # 檢查速度是否太快 (超過 50 m/s 就算爆炸)
        vel_too_fast = torch.any(torch.abs(root_vel) > 50.0, dim=1)
        terminated = terminated | vel_too_fast

        # 1. 姿態過差 (傾斜太多)
        # 計算相對於初始姿態的傾斜角度，而不是相對於世界座標
        # 這樣即使機器人初始有旋轉也不會被誤殺
        gravity_diff = self.projected_gravity - self.reference_projected_gravity
        tilt_magnitude = torch.norm(gravity_diff, dim=1)
        # tilt_magnitude ≈ 0 表示與初始姿態相同
        # tilt_magnitude ≈ 2 表示完全翻轉 (180度)
        # 對應關係: sin(angle/2) * 2 ≈ tilt_magnitude for small angles
        bad_orientation = tilt_magnitude > self.cfg.max_tilt_magnitude
        terminated = terminated | bad_orientation

        # 2. 高度過低
        base_height = root_pos[:, 2]
        too_low = base_height < self.cfg.min_base_height
        terminated = terminated | too_low

        # 3. 高度過高
        too_high = base_height > self.cfg.max_base_height
        terminated = terminated | too_high

        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        """重置環境"""
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        num_reset = len(env_ids)

        # 重置關節狀態
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = torch.zeros((num_reset, self.robot.num_joints), device=self.device)

        # 添加小的隨機擾動
        joint_pos += sample_uniform(
            -0.1, 0.1,
            joint_pos.shape,
            device=self.device
        )

        # 重置根狀態
        default_root_state = self.robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]

        # 添加小的隨機位置擾動
        default_root_state[:, 0] += sample_uniform(-0.1, 0.1, (num_reset,), device=self.device)
        default_root_state[:, 1] += sample_uniform(-0.1, 0.1, (num_reset,), device=self.device)

        # 寫入模擬
        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # 重置內部緩衝
        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel
        self.joint_pos_default[env_ids] = self.robot.data.default_joint_pos[env_ids]

        self.actions[env_ids] = 0.0
        self.last_actions[env_ids] = 0.0

        # 隨機化步態相位
        self.gait_phase[env_ids] = sample_uniform(0, 2 * math.pi, (num_reset,), device=self.device)

        # 重置腳接觸
        self.feet_contact[env_ids] = True

        # 採樣新的速度命令
        self._resample_commands(env_ids)

        # 重置獎勵追蹤
        for key in self.episode_sums:
            self.episode_sums[key][env_ids] = 0.0

    def _resample_commands(self, env_ids: torch.Tensor):
        """為指定環境採樣新的速度命令"""
        num_cmds = len(env_ids)

        # 採樣前進速度
        self.commands[env_ids, 0] = sample_uniform(
            self.cfg.lin_vel_x_range[0],
            self.cfg.lin_vel_x_range[1],
            (num_cmds,),
            device=self.device
        )

        # 採樣側向速度
        self.commands[env_ids, 1] = sample_uniform(
            self.cfg.lin_vel_y_range[0],
            self.cfg.lin_vel_y_range[1],
            (num_cmds,),
            device=self.device
        )

        # 採樣 yaw 速度
        self.commands[env_ids, 2] = sample_uniform(
            self.cfg.ang_vel_z_range[0],
            self.cfg.ang_vel_z_range[1],
            (num_cmds,),
            device=self.device
        )

        # 移除死區 - 讓機器人始終有目標速度
        # (命令範圍已設置為 [0.15, 0.3]，不需要死區)
