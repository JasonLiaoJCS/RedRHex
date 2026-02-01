# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""
RedRhex hexapod robot environment with RHex-style wheg locomotion.

RHex 機器人的核心運動原理：
1. 主驅動關節持續旋轉（類似輪子），不是傳統的步行
2. 使用交替三足步態（alternating tripod gait）
3. 半圓形 C 型腿在旋轉時產生前進位移

控制架構：
- 主驅動關節 (15, 7, 12, 18, 23, 24): 速度控制，持續旋轉
- ABAD 關節 (14, 6, 11, 17, 22, 21): 位置控制，RL 探索最佳使用方式
- 避震關節 (5, 8, 13, 25, 26, 27): 被動高阻尼，吸收衝擊
"""

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
    RedRhex 六足機器人 RHex 風格運動環境
    
    這個環境訓練機器人使用「旋轉步態」前進：
    - 主驅動關節像輪子一樣連續旋轉
    - Tripod A 和 Tripod B 以 180° 相位差交替
    - ABAD 關節用於穩定性和轉向（由 RL 探索）
    """

    cfg: RedrhexEnvCfg

    def __init__(self, cfg: RedrhexEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # 獲取關節索引
        self._setup_joint_indices()
        
        # 初始化緩衝區
        self._setup_buffers()

        # 初始化速度命令
        self._setup_commands()

        # 初始化步態相位
        self._setup_gait()

        # 打印診斷信息
        self._debug_print_info()

        print(f"[RedrhexEnv] 環境初始化完成")
        print(f"[RedrhexEnv] 動作空間: {self.cfg.action_space} (6 main_drive + 6 ABAD)")
        print(f"[RedrhexEnv] 觀測空間: {self.cfg.observation_space}")

    def _setup_joint_indices(self):
        """設置關節索引映射"""
        # 獲取所有關節名稱
        joint_names = self.robot.data.joint_names
        
        # 主驅動關節索引
        self._main_drive_indices = []
        for name in self.cfg.main_drive_joint_names:
            if name in joint_names:
                self._main_drive_indices.append(joint_names.index(name))
            else:
                print(f"⚠️ 警告: 找不到主驅動關節 {name}")
        self._main_drive_indices = torch.tensor(
            self._main_drive_indices, device=self.device, dtype=torch.long
        )
        
        # ABAD 關節索引
        self._abad_indices = []
        for name in self.cfg.abad_joint_names:
            if name in joint_names:
                self._abad_indices.append(joint_names.index(name))
            else:
                print(f"⚠️ 警告: 找不到 ABAD 關節 {name}")
        self._abad_indices = torch.tensor(
            self._abad_indices, device=self.device, dtype=torch.long
        )
        
        # 避震關節索引
        self._damper_indices = []
        for name in self.cfg.damper_joint_names:
            if name in joint_names:
                self._damper_indices.append(joint_names.index(name))
            else:
                print(f"⚠️ 警告: 找不到避震關節 {name}")
        self._damper_indices = torch.tensor(
            self._damper_indices, device=self.device, dtype=torch.long
        )
        
        # Tripod 分組
        self._tripod_a_indices = torch.tensor(
            self.cfg.tripod_a_leg_indices, device=self.device, dtype=torch.long
        )
        self._tripod_b_indices = torch.tensor(
            self.cfg.tripod_b_leg_indices, device=self.device, dtype=torch.long
        )
        
        # 方向乘數 - 從配置讀取
        # 右側腿 (idx 0,1,2) → -1, 左側腿 (idx 3,4,5) → +1
        self._direction_multiplier = torch.tensor(
            self.cfg.leg_direction_multiplier, device=self.device
        ).unsqueeze(0)  # Shape: [1, 6]
        
        print(f"[關節索引] 主驅動: {self._main_drive_indices.tolist()}")
        print(f"[關節索引] ABAD: {self._abad_indices.tolist()}")
        print(f"[關節索引] 避震: {self._damper_indices.tolist()}")
        print(f"[方向乘數] {self.cfg.leg_direction_multiplier}")
        print(f"[Tripod A] indices: {self._tripod_a_indices.tolist()} (joints 15, 18, 24)")
        print(f"[Tripod B] indices: {self._tripod_b_indices.tolist()} (joints 7, 12, 23)")

    def _setup_buffers(self):
        """設置內部緩衝區"""
        # 關節狀態
        self.joint_pos = self.robot.data.joint_pos.clone()
        self.joint_vel = self.robot.data.joint_vel.clone()
        
        # 動作緩衝 (12 維: 6 main_drive + 6 ABAD)
        self.actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self.last_actions = torch.zeros_like(self.actions)
        
        # 主驅動上一次速度 (用於計算加速度)
        self.last_main_drive_vel = torch.zeros(self.num_envs, 6, device=self.device)

        # 避震關節的初始位置（從 config 中讀取）
        # 這些關節需要保持在初始角度，不能被拉直
        # 順序要匹配 damper_joint_names: ["Revolute_5", "Revolute_13", "Revolute_25", "Revolute_26", "Revolute_27", "Revolute_8"]
        damper_init_angles = []
        for joint_name in self.cfg.damper_joint_names:
            angle = self.cfg.robot_cfg.init_state.joint_pos.get(joint_name, 0.0)
            damper_init_angles.append(angle)
        self._damper_initial_pos = torch.tensor(damper_init_angles, device=self.device).unsqueeze(0)
        print(f"[避震關節初始角度] {[f'{a*180/3.14159:.1f}°' for a in damper_init_angles]}")

        # 基座狀態
        self.base_lin_vel = torch.zeros(self.num_envs, 3, device=self.device)
        self.base_ang_vel = torch.zeros(self.num_envs, 3, device=self.device)
        self.projected_gravity = torch.zeros(self.num_envs, 3, device=self.device)

        # 計算初始參考重力方向
        init_rot = self.cfg.robot_cfg.init_state.rot
        init_quat = torch.tensor(
            [init_rot[0], init_rot[1], init_rot[2], init_rot[3]],
            device=self.device
        ).unsqueeze(0).expand(self.num_envs, 4)
        gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=self.device).expand(self.num_envs, 3)
        self.reference_projected_gravity = quat_apply_inverse(init_quat, gravity_vec)

        # 獎勵追蹤 - 追蹤所有獎勵分量以便在 TensorBoard 中查看
        self.episode_sums = {
            # 核心獎勵
            "rew_alive": torch.zeros(self.num_envs, device=self.device),
            "rew_forward_vel": torch.zeros(self.num_envs, device=self.device),
            "rew_vel_tracking": torch.zeros(self.num_envs, device=self.device),
            # 步態獎勵
            "rew_gait_sync": torch.zeros(self.num_envs, device=self.device),
            "rew_rotation_dir": torch.zeros(self.num_envs, device=self.device),
            "rew_correct_dir": torch.zeros(self.num_envs, device=self.device),  # 新增
            "rew_all_legs": torch.zeros(self.num_envs, device=self.device),
            "rew_tripod_sync": torch.zeros(self.num_envs, device=self.device),
            "rew_mean_vel": torch.zeros(self.num_envs, device=self.device),
            "rew_min_vel": torch.zeros(self.num_envs, device=self.device),
            "rew_continuous_support": torch.zeros(self.num_envs, device=self.device),
            "rew_smooth_rotation": torch.zeros(self.num_envs, device=self.device),
            # 穩定性懲罰
            "rew_orientation": torch.zeros(self.num_envs, device=self.device),
            "rew_base_height": torch.zeros(self.num_envs, device=self.device),
            "rew_lin_vel_z": torch.zeros(self.num_envs, device=self.device),
            "rew_ang_vel_xy": torch.zeros(self.num_envs, device=self.device),
            # ABAD 獎勵
            "rew_abad_action": torch.zeros(self.num_envs, device=self.device),
            "rew_abad_stability": torch.zeros(self.num_envs, device=self.device),
            # 平滑性
            "rew_action_rate": torch.zeros(self.num_envs, device=self.device),
            # 診斷指標 (非獎勵)
            "diag_forward_vel": torch.zeros(self.num_envs, device=self.device),
            "diag_base_height": torch.zeros(self.num_envs, device=self.device),
            "diag_tilt": torch.zeros(self.num_envs, device=self.device),
            "diag_drive_vel_mean": torch.zeros(self.num_envs, device=self.device),
            "diag_rotating_legs": torch.zeros(self.num_envs, device=self.device),
            "diag_min_leg_vel": torch.zeros(self.num_envs, device=self.device),
        }

    def _setup_commands(self):
        """設置速度命令"""
        self.commands = torch.zeros(self.num_envs, 3, device=self.device)

    def _setup_gait(self):
        """設置步態相位"""
        # 全局步態相位計數器
        self.gait_phase = torch.zeros(self.num_envs, device=self.device)
        
        # 每條腿的目標相位偏移
        # Tripod A (legs 0, 3, 5): 相位 0
        # Tripod B (legs 1, 2, 4): 相位 π
        self.leg_phase_offsets = torch.zeros(6, device=self.device)
        self.leg_phase_offsets[self._tripod_a_indices] = 0.0
        self.leg_phase_offsets[self._tripod_b_indices] = math.pi

    def _setup_scene(self):
        """設置模擬場景"""
        self.robot = Articulation(self.cfg.robot_cfg)
        self.scene.articulations["robot"] = self.robot

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        self.scene.clone_environments(copy_from_source=False)

        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _debug_print_info(self):
        """打印診斷信息"""
        print("\n" + "=" * 70)
        print("🤖 RedRhex RHex-style Wheg Locomotion Environment")
        print("=" * 70)
        print(f"⚙️  控制頻率: {1 / (self.cfg.sim.dt * self.cfg.decimation):.1f} Hz")
        print(f"⚙️  基礎步態頻率: {self.cfg.base_gait_frequency} Hz")
        print(f"⚙️  基礎角速度: {self.cfg.base_gait_angular_vel:.2f} rad/s")
        
        print(f"\n📐 腿部配置:")
        print(f"   主驅動關節順序: {self.cfg.main_drive_joint_names}")
        print(f"   方向乘數: {self.cfg.leg_direction_multiplier}")
        print(f"   (右側腿 idx 0,1,2 = -1, 左側腿 idx 3,4,5 = +1)")
        
        print(f"\n🦿 Tripod 分組:")
        print(f"   Tripod A (idx {self._tripod_a_indices.tolist()}): 關節 15, 18, 24")
        print(f"   Tripod B (idx {self._tripod_b_indices.tolist()}): 關節 7, 12, 23")
        
        print(f"\n🎮 動作空間 ({self.cfg.action_space}):")
        print(f"   [0:6] 主驅動速度 (scale: ±{self.cfg.main_drive_vel_scale} rad/s)")
        print(f"   [6:12] ABAD 位置 (scale: ±{self.cfg.abad_pos_scale} rad)")
        
        print(f"\n💡 RHex 步態原理:")
        print(f"   - C型腿持續旋轉（非擺動），像輪子一樣推進")
        print(f"   - Stance phase (0~π): 腿接觸地面，穩定推進")
        print(f"   - Swing phase (π~2π): 腿離地，快速轉到落地位置")
        print(f"   - 兩組 Tripod 交替支撐，確保持續接地")
        print("=" * 70 + "\n")

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """物理步之前處理動作"""
        self.last_actions = self.actions.clone()
        self.actions = actions.clone().clamp(-1.0, 1.0)

    def _apply_action(self) -> None:
        """
        將動作應用到機器人關節
        
        動作格式 (12 維):
        - [0:6]: 主驅動目標角速度 (相對於基礎速度的調整)
        - [6:12]: ABAD 目標位置
        
        注意：左右側腿需要相反的旋轉方向才能前進！
        - 右側 (Legs 1,2,3): 負向旋轉
        - 左側 (Legs 4,5,6): 正向旋轉
        """
        # ===== 主驅動關節：速度控制 =====
        # 動作 [-1, 1] 映射到速度調整
        drive_actions = self.actions[:, :6]
        
        # 基礎速度
        base_vel = self.cfg.base_gait_angular_vel
        
        # 使用配置中的方向乘數（已在 _setup_joint_indices 中初始化）
        # 右側 (idx 0,1,2) → -1, 左側 (idx 3,4,5) → +1
        
        # 計算目標速度：基礎速度 * 方向 + 動作調整 * 方向
        target_drive_vel = (base_vel + drive_actions * self.cfg.main_drive_vel_scale) * self._direction_multiplier
        
        # 限制速度範圍以防止物理爆炸
        target_drive_vel = torch.clamp(target_drive_vel, min=-10.0, max=10.0)
        
        # 應用速度目標到主驅動關節
        # 注意：當指定 joint_ids 時，target 的形狀應該是 [num_envs, len(joint_ids)]
        self.robot.set_joint_velocity_target(target_drive_vel, joint_ids=self._main_drive_indices)
        
        # ===== ABAD 關節：位置控制 =====
        abad_actions = self.actions[:, 6:12]
        target_abad_pos = abad_actions * self.cfg.abad_pos_scale
        
        # 限制位置範圍
        target_abad_pos = torch.clamp(target_abad_pos, min=-0.5, max=0.5)
        
        # 應用位置目標到 ABAD 關節
        self.robot.set_joint_position_target(target_abad_pos, joint_ids=self._abad_indices)
        
        # ===== 避震關節：保持在初始角度 =====
        # 重要：ImplicitActuator 的 stiffness 會把關節拉向位置目標
        # 如果不設置目標，默認是 0（拉直），這是錯誤的！
        # 必須設置位置目標為初始角度，讓關節保持形狀
        self.robot.set_joint_position_target(
            self._damper_initial_pos.expand(self.num_envs, -1), 
            joint_ids=self._damper_indices
        )

    def _get_observations(self) -> dict:
        """計算觀測"""
        self._update_state()

        # 主驅動關節狀態
        main_drive_pos = self.joint_pos[:, self._main_drive_indices]
        main_drive_vel = self.joint_vel[:, self._main_drive_indices]
        
        # 用 sin/cos 表示主驅動位置（因為是循環的）
        main_drive_pos_sin = torch.sin(main_drive_pos)
        main_drive_pos_cos = torch.cos(main_drive_pos)
        
        # ABAD 關節狀態
        abad_pos = self.joint_pos[:, self._abad_indices]
        abad_vel = self.joint_vel[:, self._abad_indices]

        # 構建觀測向量
        obs = torch.cat([
            self.base_lin_vel,                              # (3)
            self.base_ang_vel,                              # (3)
            self.projected_gravity,                         # (3)
            main_drive_pos_sin,                             # (6)
            main_drive_pos_cos,                             # (6)
            main_drive_vel / self.cfg.base_gait_angular_vel,  # (6) 正規化
            abad_pos / self.cfg.abad_pos_scale,             # (6) 正規化
            abad_vel,                                       # (6)
            self.commands,                                  # (3)
            torch.sin(self.gait_phase).unsqueeze(-1),       # (1)
            torch.cos(self.gait_phase).unsqueeze(-1),       # (1)
            self.last_actions,                              # (12)
        ], dim=-1)

        # 噪聲
        if self.cfg.add_noise:
            noise = torch.randn_like(obs) * 0.01 * self.cfg.noise_level
            obs = obs + noise

        # NaN/Inf 保護
        obs = torch.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0)
        obs = torch.clamp(obs, min=-100.0, max=100.0)

        return {"policy": obs}

    def _update_state(self):
        """更新內部狀態"""
        # 關節狀態
        self.joint_pos = torch.nan_to_num(self.robot.data.joint_pos.clone(), nan=0.0)
        self.joint_vel = torch.nan_to_num(self.robot.data.joint_vel.clone(), nan=0.0)

        # 基座狀態
        root_quat = self.robot.data.root_quat_w
        root_lin_vel_w = self.robot.data.root_lin_vel_w
        root_ang_vel_w = self.robot.data.root_ang_vel_w

        self.base_lin_vel = torch.clamp(
            quat_apply_inverse(root_quat, root_lin_vel_w), min=-10.0, max=10.0
        )
        self.base_ang_vel = torch.clamp(
            quat_apply_inverse(root_quat, root_ang_vel_w), min=-10.0, max=10.0
        )
        
        self.base_lin_vel = torch.nan_to_num(self.base_lin_vel, nan=0.0)
        self.base_ang_vel = torch.nan_to_num(self.base_ang_vel, nan=0.0)

        # 投影重力
        gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=self.device).expand(self.num_envs, 3)
        self.projected_gravity = quat_apply_inverse(root_quat, gravity_vec)
        self.projected_gravity = torch.nan_to_num(self.projected_gravity, nan=0.0)

        # 更新步態相位
        dt = self.cfg.sim.dt * self.cfg.decimation
        self.gait_phase = (self.gait_phase + 2 * math.pi * self.cfg.base_gait_frequency * dt) % (2 * math.pi)

    def _get_rewards(self) -> torch.Tensor:
        """
        ===== RHex 機器人運動原理（極簡版）=====
        
        【機構】
        RHex 是六足機器人，每隻腳是半圓形的 C 型腿。
        
        腿的驅動方式：
        - 主驅動關節（持續 360° 旋轉）：15, 12, 7（右側）; 18, 23, 24（左側）
        - 腿通過連續旋轉向前移動（像輪子，不是走路）
        - 旋轉方向：右腿負向，左腿正向 → 都是往後踩地推動機器人前進
        
        Tripod 分組（交替三足步態）：
        - Tripod A：腿 0, 3, 5（關節 15, 18, 24）
        - Tripod B：腿 1, 2, 4（關節 7, 12, 23）
        
        【動態步態核心】
        不是簡單的 180° 相位差！而是速度調節：
        
        1. 當腿在地面（Stance）：較慢、穩定的速度旋轉
           → 提供推進力，避免打滑
        
        2. 當腿離地（Swing）：快速旋轉
           → 迅速轉到即將落地位置，準備接力
        
        這樣確保永遠有腿在支撐，不會有滯空期。
        
        【獎勵設計原則】
        極度簡化！只獎勵：
        1. 前進（最重要）
        2. 腿在旋轉
        3. 不翻車
        """
        rewards = torch.zeros(self.num_envs, device=self.device)

        # ===== 獲取狀態 =====
        main_drive_vel = self.joint_vel[:, self._main_drive_indices]  # [N, 6]
        main_drive_pos = self.joint_pos[:, self._main_drive_indices]  # [N, 6]
        
        # 有效速度（考慮旋轉方向）
        # 正值 = 往前進方向旋轉
        effective_vel = main_drive_vel * self._direction_multiplier  # [N, 6]
        vel_magnitude = torch.abs(effective_vel)  # [N, 6]
        mean_vel = vel_magnitude.mean(dim=1)
        min_vel = vel_magnitude.min(dim=1).values
        num_active_legs = (vel_magnitude > 0.3).float().sum(dim=1)
        
        # ===== 1. 前進速度（最重要！）=====
        forward_vel = self.base_lin_vel[:, 0]
        
        # 簡單直接：前進 = 獎勵，後退 = 懲罰
        rew_forward_vel = forward_vel * 10.0  # 大權重
        rewards += rew_forward_vel
        
        # 達到目標速度的獎勵
        target_vel = self.commands[:, 0]
        vel_error = torch.abs(forward_vel - target_vel)
        rew_vel_tracking = torch.exp(-vel_error * 2.0) * 2.0
        rewards += rew_vel_tracking

        # ===== 2. 腿旋轉獎勵（簡化）=====
        
        # 2.1 正確方向旋轉
        correct_direction = effective_vel > 0.5  # 往前進方向轉
        rew_rotation_dir = correct_direction.float().sum(dim=1) * 0.5  # 每條腿 0.5
        rewards += rew_rotation_dir
        
        # 2.2 所有腿都要動
        rew_all_legs = num_active_legs * 0.3  # 每條活動的腿 0.3
        rewards += rew_all_legs
        
        # 2.3 最慢的腿也要動（防止罷工）
        rew_min_vel = torch.clamp(min_vel, max=3.0) * 0.5
        rewards += rew_min_vel
        
        # 2.4 平均旋轉速度
        rew_mean_vel = torch.clamp(mean_vel, max=5.0) * 0.3
        rewards += rew_mean_vel
        
        # 為了 TensorBoard 相容性
        rew_correct_dir = rew_mean_vel  # 合併

        # ===== 3. 簡單的穩定性（輕微懲罰）=====
        
        # 3.1 不要翻車（傾斜懲罰）
        grav_xy = self.projected_gravity[:, :2]
        tilt = torch.norm(grav_xy, dim=1)
        rew_orientation = -tilt * 0.5  # 輕微懲罰
        rewards += rew_orientation
        
        # 3.2 保持高度
        base_height = self.robot.data.root_pos_w[:, 2]
        target_height = 0.12
        height_error = torch.abs(base_height - target_height)
        rew_base_height = -height_error * 0.5
        rewards += rew_base_height
        
        # 3.3 不要亂跳（垂直速度懲罰）
        z_vel = self.base_lin_vel[:, 2]
        rew_lin_vel_z = -torch.abs(z_vel) * 0.2
        rewards += rew_lin_vel_z
        
        # 3.4 不要亂轉（角速度懲罰）
        ang_vel_xy = self.base_ang_vel[:, :2]
        rew_ang_vel_xy = -torch.norm(ang_vel_xy, dim=1) * 0.1
        rewards += rew_ang_vel_xy

        # ===== 4. 存活獎勵（小）=====
        rew_alive = torch.ones(self.num_envs, device=self.device) * 0.2
        rewards += rew_alive

        # ===== 5. 步態協調（可選，權重很低）=====
        # Tripod 相位
        effective_pos = main_drive_pos * self._direction_multiplier
        leg_phase = torch.remainder(effective_pos, 2 * math.pi)
        
        phase_a = leg_phase[:, self._tripod_a_indices]  # [N, 3]
        phase_b = leg_phase[:, self._tripod_b_indices]  # [N, 3]
        
        # 同組腿相位一致性
        def phase_coherence(phases):
            sin_mean = torch.sin(phases).mean(dim=1)
            cos_mean = torch.cos(phases).mean(dim=1)
            return torch.sqrt(sin_mean**2 + cos_mean**2)
        
        coherence_a = phase_coherence(phase_a)
        coherence_b = phase_coherence(phase_b)
        rew_tripod_sync = (coherence_a + coherence_b) * 0.2  # 低權重
        rewards += rew_tripod_sync
        
        # 兩組相位差
        mean_phase_a = torch.atan2(torch.sin(phase_a).mean(dim=1), torch.cos(phase_a).mean(dim=1))
        mean_phase_b = torch.atan2(torch.sin(phase_b).mean(dim=1), torch.cos(phase_b).mean(dim=1))
        phase_diff = torch.abs(mean_phase_a - mean_phase_b)
        phase_diff = torch.min(phase_diff, 2 * math.pi - phase_diff)
        phase_diff_error = torch.abs(phase_diff - math.pi)
        rew_gait_sync = torch.exp(-phase_diff_error) * 0.1  # 很低權重
        rewards += rew_gait_sync
        
        # 持續支撐（有腿在地面）
        in_stance = leg_phase < math.pi
        stance_a = in_stance[:, self._tripod_a_indices].float().sum(dim=1)
        stance_b = in_stance[:, self._tripod_b_indices].float().sum(dim=1)
        has_support = ((stance_a >= 1) | (stance_b >= 1)).float()
        rew_continuous_support = has_support * 0.2
        rewards += rew_continuous_support

        # 佔位符（為了 TensorBoard 相容）
        rew_abad_action = torch.zeros(self.num_envs, device=self.device)
        rew_abad_stability = torch.zeros(self.num_envs, device=self.device)
        rew_action_rate = torch.zeros(self.num_envs, device=self.device)
        rew_smooth_rotation = torch.zeros(self.num_envs, device=self.device)

        # NaN 保護
        rewards = torch.nan_to_num(rewards, nan=0.0, posinf=10.0, neginf=-10.0)

        # ===== 更新 TensorBoard =====
        self.episode_sums["rew_alive"] += rew_alive
        self.episode_sums["rew_forward_vel"] += rew_forward_vel
        self.episode_sums["rew_vel_tracking"] += rew_vel_tracking
        self.episode_sums["rew_gait_sync"] += rew_gait_sync
        self.episode_sums["rew_rotation_dir"] += rew_rotation_dir
        self.episode_sums["rew_all_legs"] += rew_all_legs
        self.episode_sums["rew_correct_dir"] += rew_correct_dir
        self.episode_sums["rew_tripod_sync"] += rew_tripod_sync
        self.episode_sums["rew_mean_vel"] += rew_mean_vel
        self.episode_sums["rew_min_vel"] += rew_min_vel
        self.episode_sums["rew_continuous_support"] += rew_continuous_support
        self.episode_sums["rew_smooth_rotation"] += rew_smooth_rotation
        self.episode_sums["rew_orientation"] += rew_orientation
        self.episode_sums["rew_base_height"] += rew_base_height
        self.episode_sums["rew_lin_vel_z"] += rew_lin_vel_z
        self.episode_sums["rew_ang_vel_xy"] += rew_ang_vel_xy
        self.episode_sums["rew_abad_action"] += rew_abad_action
        self.episode_sums["rew_abad_stability"] += rew_abad_stability
        self.episode_sums["rew_action_rate"] += rew_action_rate
        
        # 診斷
        self.episode_sums["diag_forward_vel"] += forward_vel
        self.episode_sums["diag_base_height"] += base_height
        self.episode_sums["diag_tilt"] += tilt
        self.episode_sums["diag_drive_vel_mean"] += mean_vel
        self.episode_sums["diag_rotating_legs"] += num_active_legs
        self.episode_sums["diag_min_leg_vel"] += min_vel
        
        self.last_main_drive_vel = main_drive_vel.clone()

        return rewards

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """計算終止條件 - 大幅放寬以允許探索"""
        # 超時
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # 終止條件 - 只在真正壞掉時終止
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        root_pos = self.robot.data.root_pos_w
        root_vel = self.robot.data.root_lin_vel_w
        
        # 1. 物理爆炸檢測（NaN/Inf）
        pos_invalid = torch.any(torch.isnan(root_pos) | torch.isinf(root_pos), dim=1)
        vel_invalid = torch.any(torch.isnan(root_vel) | torch.isinf(root_vel), dim=1)
        terminated = terminated | pos_invalid | vel_invalid
        
        # 2. 位置過遠（跑到仿真邊界外）
        pos_too_far = torch.any(torch.abs(root_pos[:, :2]) > 50.0, dim=1)
        terminated = terminated | pos_too_far
        
        # 3. 速度過快（物理失控）- 放寬閾值
        vel_too_fast = torch.any(torch.abs(root_vel) > 30.0, dim=1)
        terminated = terminated | vel_too_fast

        # 4. 翻車檢測 - 只在完全翻過來時終止
        # projected_gravity 的 z 分量：正立時約 -1，完全翻轉時約 +1
        # 當 z > 0.5 表示翻過來超過 60°
        flipped_over = self.projected_gravity[:, 2] > 0.5
        terminated = terminated | flipped_over

        # 5. 高度終止 - 放寬範圍
        base_height = root_pos[:, 2]
        too_low = base_height < 0.01  # 只有地面以下才終止
        too_high = base_height > 1.0   # 只有飛太高才終止
        terminated = terminated | too_low | too_high

        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        """重置環境"""
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        num_reset = len(env_ids)

        # 重置關節狀態 - 使用配置文件中定義的默認位置
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = torch.zeros((num_reset, self.robot.num_joints), device=self.device)
        
        # Debug: 打印第一次重置時的初始關節位置
        if not hasattr(self, '_printed_init_pos'):
            self._printed_init_pos = True
            print("\n[DEBUG] Initial joint positions from config:")
            joint_names = self.robot.data.joint_names
            for i, name in enumerate(joint_names):
                pos_deg = joint_pos[0, i].item() * 180 / math.pi
                print(f"  {name}: {joint_pos[0, i].item():.3f} rad ({pos_deg:.1f}°)")
            print("")

        # 減少隨機擾動
        joint_pos += sample_uniform(-0.02, 0.02, joint_pos.shape, device=self.device)

        # 重置根狀態
        default_root_state = self.robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]

        default_root_state[:, 0] += sample_uniform(-0.1, 0.1, (num_reset,), device=self.device)
        default_root_state[:, 1] += sample_uniform(-0.1, 0.1, (num_reset,), device=self.device)

        # 寫入模擬
        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # 重置內部緩衝
        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel

        self.actions[env_ids] = 0.0
        self.last_actions[env_ids] = 0.0
        self.last_main_drive_vel[env_ids] = 0.0  # 從零開始

        # 隨機化步態相位
        self.gait_phase[env_ids] = sample_uniform(0, 2 * math.pi, (num_reset,), device=self.device)

        # 採樣新的速度命令
        self._resample_commands(env_ids)

        # ===== TensorBoard Logging =====
        # 計算並記錄 episode 獎勵總和到 extras["log"]
        # RSL-RL 的 Logger 會自動從 extras["log"] 讀取並寫入 TensorBoard
        extras = dict()
        for key in self.episode_sums.keys():
            # 計算被重置環境的平均 episode 獎勵
            episodic_sum_avg = torch.mean(self.episode_sums[key][env_ids])
            # 使用 "/" 前綴讓 RSL-RL 直接記錄到 TensorBoard
            # 格式: "Episode_Reward/rew_forward_vel" -> TensorBoard 會顯示在 Episode_Reward 分類下
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
        
        # 初始化 extras["log"] 並更新
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        
        # 記錄終止原因統計
        termination_extras = dict()
        termination_extras["Episode_Termination/terminated"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        termination_extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        self.extras["log"].update(termination_extras)
        
        # 重置獎勵追蹤 (在記錄後重置)
        for key in self.episode_sums:
            self.episode_sums[key][env_ids] = 0.0

    def _resample_commands(self, env_ids: torch.Tensor):
        """為指定環境採樣新的速度命令"""
        num_cmds = len(env_ids)

        self.commands[env_ids, 0] = sample_uniform(
            self.cfg.lin_vel_x_range[0],
            self.cfg.lin_vel_x_range[1],
            (num_cmds,),
            device=self.device
        )

        self.commands[env_ids, 1] = sample_uniform(
            self.cfg.lin_vel_y_range[0],
            self.cfg.lin_vel_y_range[1],
            (num_cmds,),
            device=self.device
        )

        self.commands[env_ids, 2] = sample_uniform(
            self.cfg.ang_vel_z_range[0],
            self.cfg.ang_vel_z_range[1],
            (num_cmds,),
            device=self.device
        )
