# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""
=============================================================================
RedRhex 六足機器人環境 - 使用 RHex 風格的「旋轉步態」運動方式
=============================================================================

【給初學者的說明】
這個檔案定義了機器人如何在虛擬環境中「學習走路」。
想像你在教一隻機器狗學走路：
- 你給它一個「訓練場」（這個環境）
- 告訴它「往前走」的命令（速度指令）
- 當它做對了就給「獎勵」，做錯就「扣分」（獎勵函數）
- 機器人透過不斷嘗試，學會如何走得又快又穩

【RHex 機器人的特殊之處】
一般機器人的腿是「擺動」的（像人走路），但 RHex 的腿是「旋轉」的！

想像這樣：
┌────────────────────────────────────────────────────────────────┐
│ 普通走路機器人：腿前後擺動 ← → ← →                            │
│ RHex 機器人：腿像輪子一樣旋轉 ↻ ↻ ↻                          │
│                                                                │
│ RHex 的 C 型腿（半圓形）旋轉時：                              │
│   1. 腿的底部接觸地面 → 把身體往前推                          │
│   2. 腿的頂部離開地面 → 快速轉到下一個位置                    │
│   3. 重複這個過程 → 機器人就前進了！                          │
└────────────────────────────────────────────────────────────────┘

【三種關節的功能】
┌─────────────────────────────────────────────────────────────────┐
│ 1. 主驅動關節 (Main Drive)                                     │
│    - 編號: 15, 7, 12, 18, 23, 24                               │
│    - 功能: 讓腿持續旋轉，像馬達帶動輪子                        │
│    - 控制方式: 速度控制（告訴它轉多快）                        │
│                                                                 │
│ 2. ABAD 關節 (外展/內收)                                        │
│    - 編號: 14, 6, 11, 17, 22, 21                               │
│    - 功能: 讓腿往外或往內擺，用於轉彎和保持平衡                │
│    - 控制方式: 位置控制（告訴它擺到什麼角度）                  │
│                                                                 │
│ 3. 避震關節 (Damper)                                           │
│    - 編號: 5, 8, 13, 25, 26, 27                                │
│    - 功能: 吸收衝擊，保護機身（像汽車的避震器）                │
│    - 控制方式: 被動式（不用控制，自動吸震）                    │
└─────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import math
import torch
from collections.abc import Sequence

# =============================================================================
# 【匯入模組說明】
# =============================================================================
# Isaac Lab 是 NVIDIA 開發的機器人模擬平台，以下是各模組的功能：

import isaaclab.sim as sim_utils                    # 模擬工具（設定物理環境）
from isaaclab.assets import Articulation            # 關節式機器人（有關節可動的機器人）
from isaaclab.envs import DirectRLEnv               # 強化學習環境基礎類別
# ContactSensor 暫時禁用，等待 USD 檔案添加 contact reporter API
# from isaaclab.sensors import ContactSensor        # 接觸感測器（偵測碰撞）
from isaaclab.utils.math import quat_apply_inverse, quat_apply, sample_uniform  # 數學工具
import isaaclab.utils.math as math_utils           # 更多數學工具（四元數、旋轉等）

# 可視化工具：用來在畫面上畫箭頭，顯示速度方向
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import GREEN_ARROW_X_MARKER_CFG, RED_ARROW_X_MARKER_CFG

# 從同目錄匯入配置檔案（定義了機器人的各種參數）
from .redrhex_env_cfg import RedrhexEnvCfg


class RedrhexEnv(DirectRLEnv):
    """
    ==========================================================================
    RedRhex 六足機器人強化學習環境
    ==========================================================================
    
    【這個類別是什麼？】
    這是機器人的「訓練場」！它定義了：
    1. 機器人能「看到」什麼（觀測空間）
    2. 機器人能「做」什麼（動作空間）
    3. 什麼是「好」的行為（獎勵函數）
    4. 什麼時候「遊戲結束」（終止條件）
    
    【運動方式說明 - RHex 非對稱 Duty Cycle 步態】
    ┌─────────────────────────────────────────────────────────────────┐
    │ RHex 旋轉步態的工作原理：                                       │
    │                                                                 │
    │ 六隻腳分成兩組，交替運動：                                      │
    │ • Tripod A（三角支撐組 A）：Leg 0, 3, 5 一起動                  │
    │ • Tripod B（三角支撐組 B）：Leg 1, 2, 4 一起動                  │
    │                                                                 │
    │ ★ 核心機制（非對稱 Duty Cycle）：                               │
    │ • 著地相位：腿在小角度範圍內（如 -30°~+30°）緩慢轉動           │
    │            佔時間的 65%，但只轉過約 60° 的角度                  │
    │ • 擺動相位：腿快速轉過剩餘的 300°，準備下一次著地              │
    │            佔時間的 35%，速度是著地的 ~10 倍                    │
    │                                                                 │
    │ 由於 duty_cycle > 50%，兩組著地時間有 30% 重疊                  │
    │ → 任何時刻都至少有一組著地，永不騰空！                          │
    │                                                                 │
    │ ABAD 關節的作用：                                               │
    │ • 直走時：不需要動                                              │
    │ • 轉彎時：調整腿的角度來改變方向                                │
    │ • 側移時：讓腿往外擺，產生側向推力                              │
    └─────────────────────────────────────────────────────────────────┘
    """

    cfg: RedrhexEnvCfg

    def __init__(self, cfg: RedrhexEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # ContactSensor 暫時禁用，改用高度/姿態檢測
        # self._base_id, _ = self._contact_sensor.find_bodies("base_link")
        # print(f"[Contact Sensor] Base body ID: {self._base_id}")
        print("[INFO] ContactSensor disabled - using height/orientation for body contact detection")

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
        
        # 自動啟用 debug visualization（如果配置啟用且有 GUI）
        if hasattr(self.cfg, 'draw_debug_vis') and self.cfg.draw_debug_vis:
            if self.sim.has_gui():
                self.set_debug_vis(True)
                print("[RedrhexEnv] Debug visualization 已啟用")
            else:
                print("[RedrhexEnv] 無 GUI 模式，跳過 debug visualization")
        
        # ★★★ 簡化獎勵模式初始化 ★★★
        self._use_simplified_rewards = getattr(self.cfg, 'use_simplified_rewards', False)
        self._ablation_flags = getattr(self.cfg, 'ablation_flags', {})
        
        if self._use_simplified_rewards:
            print("=" * 70)
            print("[RedrhexEnv] ★★★ 簡化獎勵模式已啟用 ★★★")
            print("[RedrhexEnv] 只使用 8 項核心獎勵，移除 20+ 項冗餘/衝突獎勵")
            active_rewards = [k for k, v in self._ablation_flags.items() if v]
            print(f"[RedrhexEnv] 啟用的獎勵: {active_rewards}")
            print("=" * 70)
        else:
            print("[RedrhexEnv] 使用完整獎勵模式（向後相容）")

    def _setup_joint_indices(self):
        """
        【設置關節索引映射】
        
        這個函數的目的：找出每個關節在「關節列表」中的位置（索引）
        
        為什麼需要索引？
        想像機器人有 18 個關節，程式需要知道「主驅動關節」是第幾個，
        才能正確地讀取它的狀態或發送控制命令。
        
        比喻：就像點名簿，我們需要知道「小明」是第幾號，才能找到他的資料。
        """
        # 獲取所有關節名稱（像是拿到一份點名簿）
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
        """
        【設置內部緩衝區】
        
        緩衝區（Buffer）= 用來暫時存放資料的「記憶空間」
        
        為什麼需要緩衝區？
        1. 儲存機器人的當前狀態（位置、速度等）
        2. 記住上一次的動作（讓動作更平滑）
        3. 追蹤各種獎勵的累積值（用於訓練分析）
        
        這就像是機器人的「短期記憶」，讓它知道自己現在是什麼狀態。
        """
        # 關節位置和速度（機器人現在各個關節的狀態）
        self.joint_pos = self.robot.data.joint_pos.clone()  # clone() = 複製一份
        self.joint_vel = self.robot.data.joint_vel.clone()
        
        # =================================================================
        # 動作緩衝區（儲存 AI 輸出的控制指令）
        # =================================================================
        # actions 是 AI 神經網路輸出的動作，總共 12 個數值：
        # - 前 6 個: 控制 6 個主驅動關節的旋轉速度
        # - 後 6 個: 控制 6 個 ABAD 關節的位置
        self.actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        # last_actions = 上一次的動作（用來計算動作變化率，讓動作更平滑）
        self.last_actions = torch.zeros_like(self.actions)
        
        # 主驅動關節上一次的速度（用來計算加速度，避免動作太劇烈）
        self.last_main_drive_vel = torch.zeros(self.num_envs, 6, device=self.device)

        # =================================================================
        # 避震關節的初始位置
        # =================================================================
        # 避震關節不被 AI 控制，需要保持在初始角度
        # 這就像汽車的避震器，你不需要操控它，它自己會吸收震動
        damper_init_angles = []
        for joint_name in self.cfg.damper_joint_names:
            angle = self.cfg.robot_cfg.init_state.joint_pos.get(joint_name, 0.0)
            damper_init_angles.append(angle)
        self._damper_initial_pos = torch.tensor(damper_init_angles, device=self.device).unsqueeze(0)
        print(f"[避震關節初始角度] {[f'{a*180/3.14159:.1f}°' for a in damper_init_angles]}")

        # =================================================================
        # 機身狀態緩衝區
        # =================================================================
        # base_lin_vel = 機身的線速度（移動速度），3 維向量 [vx, vy, vz]
        #   vx = 前後速度，vy = 左右速度，vz = 上下速度
        self.base_lin_vel = torch.zeros(self.num_envs, 3, device=self.device)
        
        # base_ang_vel = 機身的角速度（旋轉速度），3 維向量 [wx, wy, wz]
        #   wx = 繞 X 軸旋轉（側滾），wy = 繞 Y 軸旋轉（俯仰），wz = 繞 Z 軸旋轉（偏航）
        self.base_ang_vel = torch.zeros(self.num_envs, 3, device=self.device)
        
        # projected_gravity = 投影重力方向
        # 用來判斷機器人有沒有傾斜（如果傾斜，重力方向就不是正下方）
        self.projected_gravity = torch.zeros(self.num_envs, 3, device=self.device)

        # 計算初始狀態下的參考重力方向（用來比較現在傾斜了多少）
        init_rot = self.cfg.robot_cfg.init_state.rot
        init_quat = torch.tensor(
            [init_rot[0], init_rot[1], init_rot[2], init_rot[3]],
            device=self.device
        ).unsqueeze(0).expand(self.num_envs, 4)
        gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=self.device).expand(self.num_envs, 3)
        self.reference_projected_gravity = quat_apply_inverse(init_quat, gravity_vec)

        # =================================================================
        # 獎勵追蹤緩衝區（用於訓練監控和分析）
        # =================================================================
        # TensorBoard 是一個視覺化工具，可以畫出訓練過程中各種數值的變化曲線
        # 這裡追蹤各種獎勵和診斷數據，方便我們了解機器人學得怎麼樣
        # 
        # 命名規則：
        # - rew_xxx = 獎勵項目（正值是獎勵，負值是懲罰）
        # - diag_xxx = 診斷數據（不是獎勵，只是用來觀察的）
        self.episode_sums = {
            # 核心獎勵
            "rew_alive": torch.zeros(self.num_envs, device=self.device),
            "rew_forward_vel": torch.zeros(self.num_envs, device=self.device),
            "rew_vel_tracking": torch.zeros(self.num_envs, device=self.device),
            # 步態獎勵 - RHex 非對稱 Duty Cycle
            "rew_gait_sync": torch.zeros(self.num_envs, device=self.device),
            "rew_tripod_sync": torch.zeros(self.num_envs, device=self.device),
            "rew_tripod_support": torch.zeros(self.num_envs, device=self.device),      # 連續支撐
            "rew_airborne_penalty": torch.zeros(self.num_envs, device=self.device),    # 騰空懲罰
            "rew_double_support": torch.zeros(self.num_envs, device=self.device),      # 雙支撐獎勵
            "rew_velocity_match": torch.zeros(self.num_envs, device=self.device),      # 速度匹配
            "rew_alternation": torch.zeros(self.num_envs, device=self.device),         # 交替步態
            "rew_frequency": torch.zeros(self.num_envs, device=self.device),           # 頻率一致
            # 舊版步態獎勵（保留向後相容）
            "rew_rotation_dir": torch.zeros(self.num_envs, device=self.device),
            "rew_correct_dir": torch.zeros(self.num_envs, device=self.device),
            "rew_all_legs": torch.zeros(self.num_envs, device=self.device),
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
            # ★★★ 新增：側移專用獎勵 ★★★
            "rew_abad_alternation": torch.zeros(self.num_envs, device=self.device),    # ABAD 交替
            "rew_abad_amplitude": torch.zeros(self.num_envs, device=self.device),      # ABAD 幅度
            "rew_abad_jitter": torch.zeros(self.num_envs, device=self.device),         # ABAD 抖動懲罰
            "rew_sync_jitter": torch.zeros(self.num_envs, device=self.device),         # 全身抖動懲罰
            "rew_abad_action_rate": torch.zeros(self.num_envs, device=self.device),    # ABAD 變化率懲罰
            "rew_lateral_alternating_stance": torch.zeros(self.num_envs, device=self.device),  # 側移交替著地
            # ★★★ 新增：時間基準步態獎勵 ★★★
            "rew_gait_phase": torch.zeros(self.num_envs, device=self.device),          # 步態相位正確性
            "rew_abad_smooth": torch.zeros(self.num_envs, device=self.device),         # ABAD 平滑獎勵
            "rew_both_stance_penalty": torch.zeros(self.num_envs, device=self.device), # 雙側著地懲罰
            "rew_lateral_direction": torch.zeros(self.num_envs, device=self.device),   # 側移方向一致性
            # ★★★ 新增：直走專用獎勵 ★★★
            "rew_high_stance": torch.zeros(self.num_envs, device=self.device),         # 高站姿獎勵
            "rew_abad_zero_forward": torch.zeros(self.num_envs, device=self.device),   # 直走 ABAD 歸零
            "rew_abad_diagonal": torch.zeros(self.num_envs, device=self.device),       # 斜向 ABAD 使用
            # ★★★ 新增：側移專用獎勵 ★★★
            "rew_lateral_drive_lock": torch.zeros(self.num_envs, device=self.device),  # 側移主驅動鎖定
            "rew_lateral_low_freq": torch.zeros(self.num_envs, device=self.device),    # 側移低頻獎勵
            "rew_lateral_correct_dir": torch.zeros(self.num_envs, device=self.device), # 側移正確方向
            # ★★★ 新增：旋轉專用獎勵 ★★★
            "rew_rotation_slow_penalty": torch.zeros(self.num_envs, device=self.device),  # 旋轉太慢懲罰
            "rew_rotation_abad_assist": torch.zeros(self.num_envs, device=self.device),   # 旋轉 ABAD 輔助
            "rew_rotation_correct": torch.zeros(self.num_envs, device=self.device),       # 旋轉方向正確
            # ★★★ 新增：移動獎勵（防消極）★★★
            "rew_leg_moving": torch.zeros(self.num_envs, device=self.device),            # 腿轉動獎勵
            "rew_direction_bonus": torch.zeros(self.num_envs, device=self.device),       # 方向正確額外獎勵
            # 平滑性
            "rew_action_rate": torch.zeros(self.num_envs, device=self.device),
            # 診斷指標 (非獎勵)
            "diag_forward_vel": torch.zeros(self.num_envs, device=self.device),
            "diag_lateral_vel": torch.zeros(self.num_envs, device=self.device),
            "diag_cmd_vx": torch.zeros(self.num_envs, device=self.device),
            "diag_cmd_vy": torch.zeros(self.num_envs, device=self.device),
            "diag_vel_error": torch.zeros(self.num_envs, device=self.device),
            "diag_base_height": torch.zeros(self.num_envs, device=self.device),
            "diag_tilt": torch.zeros(self.num_envs, device=self.device),
            "diag_drive_vel_mean": torch.zeros(self.num_envs, device=self.device),
            "diag_rotating_legs": torch.zeros(self.num_envs, device=self.device),
            "diag_min_leg_vel": torch.zeros(self.num_envs, device=self.device),
            "diag_abad_magnitude": torch.zeros(self.num_envs, device=self.device),
            # 旋轉追蹤診斷
            "diag_cmd_wz": torch.zeros(self.num_envs, device=self.device),
            "diag_actual_wz": torch.zeros(self.num_envs, device=self.device),
            "diag_wz_error": torch.zeros(self.num_envs, device=self.device),
            # 腿速度診斷
            "diag_target_leg_vel": torch.zeros(self.num_envs, device=self.device),
            "diag_leg_vel_error": torch.zeros(self.num_envs, device=self.device),
            # ★★★ 新增：RHex 步態診斷 ★★★
            "diag_stance_count_a": torch.zeros(self.num_envs, device=self.device),    # A組著地腿數
            "diag_stance_count_b": torch.zeros(self.num_envs, device=self.device),    # B組著地腿數
            "diag_phase_diff": torch.zeros(self.num_envs, device=self.device),        # 相位差
            "diag_mean_velocity": torch.zeros(self.num_envs, device=self.device),     # 平均腿速
            "diag_stance_velocity": torch.zeros(self.num_envs, device=self.device),   # 著地組速度
            "diag_swing_velocity": torch.zeros(self.num_envs, device=self.device),    # 擺動組速度
            "diag_airborne_count": torch.zeros(self.num_envs, device=self.device),    # 騰空次數
            "diag_lateral_lift_count": torch.zeros(self.num_envs, device=self.device),  # 側移抬腿數
        }

        # 初始化目標速度緩衝
        self._target_drive_vel = torch.zeros(self.num_envs, 6, device=self.device)
        self._base_velocity = torch.zeros(self.num_envs, 6, device=self.device)  # 基礎速度（未經AI調節）

    def _setup_commands(self):
        """
        【設置速度命令系統】
        
        這個系統負責「告訴機器人要往哪裡走」。
        
        訓練過程中，系統會隨機給機器人不同的移動命令：
        - 「往前走」
        - 「往左走」
        - 「原地轉圈」
        - 等等...
        
        機器人必須學會「聽命令」，這樣訓練完成後，
        我們才能用命令控制機器人去任何地方！
        """
        # 速度命令向量 [vx, vy, wz]：
        # - vx = 前後速度（正值向前，負值向後）
        # - vy = 左右速度（正值向左，負值向右）
        # - wz = 旋轉速度（正值逆時針，負值順時針）
        self.commands = torch.zeros(self.num_envs, 3, device=self.device)
        
        # 命令切換計時器
        self.command_time_left = torch.zeros(self.num_envs, device=self.device)
        
        # 離散方向（10個方向：8個移動方向 + 2個原地旋轉）
        if hasattr(self.cfg, 'discrete_directions') and self.cfg.use_discrete_directions:
            self.discrete_directions = torch.tensor(
                self.cfg.discrete_directions, device=self.device, dtype=torch.float32
            )
            self.num_directions = self.discrete_directions.shape[0]
            
            # 檢查方向格式（是否包含 wz）
            if self.discrete_directions.shape[1] == 2:
                # 舊格式 [vx, vy]，添加 wz=0
                zeros = torch.zeros(self.num_directions, 1, device=self.device)
                self.discrete_directions = torch.cat([self.discrete_directions, zeros], dim=1)
            
            print(f"[命令系統] 使用離散方向模式，共 {self.num_directions} 個方向")
            if hasattr(self.cfg, 'direction_names'):
                print(f"   方向: {', '.join(self.cfg.direction_names)}")
        else:
            self.discrete_directions = None
            self.num_directions = 0
            print(f"[命令系統] 使用連續速度範圍")
        
        # 當前方向索引（用於追蹤）
        self.current_direction_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        
        # ★★★ 外部控制標誌 ★★★
        # 當設為 True 時，禁用自動命令重採樣，讓外部（如鍵盤）控制
        self.external_control = False
        
        # 初始化命令
        self._resample_commands(torch.arange(self.num_envs, device=self.device))

    def _resample_commands(self, env_ids: torch.Tensor):
        """
        【重新採樣速度命令】
        
        功能：隨機給指定的環境一個新的移動命令
        
        參數：
            env_ids: 需要更換命令的環境編號列表
            （訓練時會同時跑很多個環境，每個環境有自己的編號）
        
        這就像教練隨機喊口令：「往前跑！」「往左移！」「原地轉！」
        機器人必須學會正確執行每個口令。
        """
        if len(env_ids) == 0:
            return
            
        # 重置計時器
        self.command_time_left[env_ids] = self.cfg.command_resample_time
        
        if self.discrete_directions is not None and self.cfg.use_discrete_directions:
            # 離散方向模式：隨機選擇一個方向
            dir_indices = torch.randint(0, self.num_directions, (len(env_ids),), device=self.device)
            self.current_direction_idx[env_ids] = dir_indices
            
            # 設置 vx, vy, wz（直接從 discrete_directions 獲取全部三個值）
            self.commands[env_ids, 0] = self.discrete_directions[dir_indices, 0]
            self.commands[env_ids, 1] = self.discrete_directions[dir_indices, 1]
            self.commands[env_ids, 2] = self.discrete_directions[dir_indices, 2]
            
            # 打印方向切換信息（只打印前幾個環境，避免刷屏）
            if len(env_ids) > 0 and env_ids[0] == 0 and hasattr(self.cfg, 'direction_names'):
                idx = dir_indices[0].item()
                name = self.cfg.direction_names[idx] if idx < len(self.cfg.direction_names) else f"Dir{idx}"
                print(f"[命令切換] env0 → {name} (vx={self.commands[0,0]:.2f}, vy={self.commands[0,1]:.2f}, wz={self.commands[0,2]:.2f})")
        else:
            # 連續範圍模式
            self.commands[env_ids, 0] = sample_uniform(
                self.cfg.lin_vel_x_range[0],
                self.cfg.lin_vel_x_range[1],
                (len(env_ids),),
                self.device
            )
            self.commands[env_ids, 1] = sample_uniform(
                self.cfg.lin_vel_y_range[0],
                self.cfg.lin_vel_y_range[1],
                (len(env_ids),),
                self.device
            )
            self.commands[env_ids, 2] = sample_uniform(
                self.cfg.ang_vel_z_range[0],
                self.cfg.ang_vel_z_range[1],
                (len(env_ids),),
                self.device
            )

    def _update_commands(self):
        """更新命令
        
        ★★★ 重大改變：不再定期切換方向！★★★
        每個環境專注訓練同一個方向，直到被重置
        """
        # 外部控制時跳過
        if self.external_control:
            return
        
        # ★★★ 新模式：不再定期切換，只在重置時切換 ★★★
        # 這樣每個環境可以專注學習當前方向，直到失敗或超時
        # 命令重採樣只在 _reset_idx 中進行
        pass

    def _setup_gait(self):
        """
        【設置步態相位系統】 - RHex 非對稱 Duty Cycle 步態
        
        ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        ★ RHex 步態的核心概念                                      ★
        ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        
        【傳統錯誤理解】
        很多人以為 RHex 的兩組腿（Tripod A 和 B）是簡單的 180° 反相，
        或者以為著地佔 65% 時間就轉過 65% 的角度。
        這些都是錯的！
        
        【正確的 RHex 步態】★★★ 角度 vs 時間 的區別 ★★★
        
        RHex 使用「非對稱 duty cycle」步態，關鍵是：
        著地時間長，但轉過的角度少！
        
        1. 著地相位（Stance Phase）
           - 時間佔比：~65%（時間長）
           - 角度範圍：~60°（如 -30° ~ +30°）（角度小！）
           - 腿底部在地面上，只能在小範圍內緩慢轉動
           - 速度約為基礎速度的 15%（非常慢！）
        
        2. 擺動相位（Swing Phase）
           - 時間佔比：~35%（時間短）
           - 角度範圍：~300°（角度大！）
           - 腿要快速轉過大部分角度，回到著地位置
           - 速度約為基礎速度的 150%（10 倍於著地速度！）
        
        【關鍵：為什麼不會騰空？】
        因為 duty_cycle > 0.5，兩組的著地時間有「重疊」！
        
        重疊時間 = (2 × 0.65 - 1) × T = 0.30 × T
        
        這表示在每個週期中，有 30% 的時間是「兩組都著地」的超穩定狀態。
        
        ┌───────────────────────────────────────────────────────────────┐
        │ 時間軸：                                                      │
        │                                                               │
        │ A組: ████████████████████████████████░░░░░░░░░░░░░░           │
        │      ←──── 著地 (65%) ──────────────→←─ 擺動 ─→              │
        │                                                               │
        │ B組: ░░░░░░░░░░████████████████████████████████████░░░░       │
        │      ←擺動→←────────── 著地 (65%) ─────────────→             │
        │                                                               │
        │ 支撐:████████████████████████████████████████████████████     │
        │      ←─ A ─→←─ 重疊 ─→←─── B ───→←─ 重疊 ─→←─ A ─→          │
        │              ↑                    ↑                          │
        │         兩組都著地            兩組都著地                      │
        │         (超級穩定!)           (超級穩定!)                     │
        └───────────────────────────────────────────────────────────────┘
        """
        # 全局步態相位計數器（主時鐘）
        # 這是一個從 0 到 2π 循環的計數器，代表整個步態週期的進度
        self.gait_phase = torch.zeros(self.num_envs, device=self.device)
        
        # 每條腿的相位偏移量
        # Tripod A (腿 0, 3, 5): 偏移 0（跟著主時鐘）
        # Tripod B (腿 1, 2, 4): 偏移值由 tripod_phase_offset 決定
        self.leg_phase_offsets = torch.zeros(6, device=self.device)
        self.leg_phase_offsets[self._tripod_a_indices] = 0.0
        self.leg_phase_offsets[self._tripod_b_indices] = self.cfg.tripod_phase_offset
        
        # =====================================================================
        # 【預計算步態參數】
        # =====================================================================

        # 著地相位邊界（弧度）
        # 注意：stance_phase_start 可能是負數（如 -π/6）
        # 需要正規化到 [0, 2π] 範圍進行比較
        self.stance_phase_start = self.cfg.stance_phase_start  # 如 -π/6 (-30°)
        self.stance_phase_end = self.cfg.stance_phase_end      # 如 +π/6 (+30°)

        # 著地角度區間大小（弧度）
        stance_angle_range = self.stance_phase_end - self.stance_phase_start  # 如 π/3 (60°)
        swing_angle_range = 2 * math.pi - stance_angle_range  # 如 5π/3 (300°)

        # 著地和擺動的目標速度（弧度/秒）
        base_vel = self.cfg.base_gait_angular_vel  # 6.28 rad/s
        self.stance_velocity = base_vel * self.cfg.stance_velocity_ratio  # ~0.94 rad/s (很慢)
        self.swing_velocity = base_vel * self.cfg.swing_velocity_ratio    # ~9.42 rad/s (快)

        # 速度比值（用於獎勵計算）
        self.velocity_ratio = self.swing_velocity / self.stance_velocity  # ~10x

        # 記錄上一步的相位狀態（用於檢測相位轉換）
        self.last_leg_in_stance = torch.ones(self.num_envs, 6, dtype=torch.bool, device=self.device)

        print(f"\n[步態系統初始化] ★ 著地角度小、時間長；擺動角度大、時間短 ★")
        print(f"  著地相位角度範圍: {math.degrees(self.stance_phase_start):.1f}° ~ {math.degrees(self.stance_phase_end):.1f}° (共 {math.degrees(stance_angle_range):.1f}°)")
        print(f"  擺動相位角度範圍: {math.degrees(self.stance_phase_end):.1f}° ~ {math.degrees(self.stance_phase_start + 2*math.pi):.1f}° (共 {math.degrees(swing_angle_range):.1f}°)")
        print(f"  著地時間佔比: {self.cfg.stance_duty_cycle * 100:.1f}%")
        print(f"  著地速度: {self.stance_velocity:.2f} rad/s ({math.degrees(self.stance_velocity):.1f}°/s)")
        print(f"  擺動速度: {self.swing_velocity:.2f} rad/s ({math.degrees(self.swing_velocity):.1f}°/s)")
        print(f"  速度比值 (swing/stance): {self.velocity_ratio:.1f}x")

    def _setup_scene(self):
        """
        【設置模擬場景】
        
        這個函數創建虛擬世界中的所有東西：
        1. 機器人本身
        2. 地面
        3. 燈光
        4. 感測器（目前禁用）
        
        就像在遊戲裡「生成」角色和地圖一樣！
        """
        # 創建機器人並加入場景
        self.robot = Articulation(self.cfg.robot_cfg)
        self.scene.articulations["robot"] = self.robot
        
        # 注意：ContactSensor（接觸感測器）暫時禁用
        # 原本用來偵測「機器人有沒有碰到東西」
        # 但 USD 模型檔案還沒設定好，所以改用高度和姿態來判斷
        # self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        # self.scene.sensors["contact_sensor"] = self._contact_sensor

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        self.scene.clone_environments(copy_from_source=False)

        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        
    def _post_physics_step(self):
        """物理步之後更新狀態"""
        # 必須調用父類的 post physics step
        pass  # DirectRLEnv 會自動處理

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
        print(f"   [0:6] 主驅動速度調節因子 (±50%)")
        print(f"   [6:12] ABAD 位置 (scale: ±{self.cfg.abad_pos_scale} rad)")
        
        print(f"\n💡 RHex 非對稱 Duty Cycle 步態:")
        print(f"   ┌────────────────────────────────────────────────────────┐")
        print(f"   │ 著地相位 (Stance): 佔 {self.cfg.stance_duty_cycle*100:.0f}% 週期             │")
        print(f"   │   - 速度: {self.stance_velocity:.2f} rad/s (慢轉)                  │")
        print(f"   │   - 功能: 提供穩定支撐和推進力                       │")
        print(f"   │                                                        │")
        print(f"   │ 擺動相位 (Swing): 佔 {(1-self.cfg.stance_duty_cycle)*100:.0f}% 週期              │")
        print(f"   │   - 速度: {self.swing_velocity:.2f} rad/s (快轉！)                │")
        print(f"   │   - 功能: 快速回到準備著地的位置                     │")
        print(f"   │                                                        │")
        print(f"   │ 速度比: {self.velocity_ratio:.1f}x (擺動是著地的 {self.velocity_ratio:.1f} 倍速)       │")
        print(f"   │ 重疊期: {(2*self.cfg.stance_duty_cycle-1)*100:.0f}% (兩組同時著地的超穩定期)      │")
        print(f"   └────────────────────────────────────────────────────────┘")
        
        print(f"\n📊 步態時序圖:")
        print(f"   時間 →")
        print(f"   A組: ████████████████░░░░░░░░  (著地65% + 擺動35%)")
        print(f"   B組: ░░░░░░░░████████████████  (先擺動 + 後著地)")
        print(f"   支撐: ████████████████████████  (始終有支撐!)")
        print("=" * 70 + "\n")

    def _is_reward_enabled(self, reward_name: str) -> bool:
        """
        檢查獎勵是否啟用（用於 ablation 測試）
        
        Args:
            reward_name: 獎勵名稱（對應 ablation_flags 中的 key）
            
        Returns:
            bool: True 表示啟用，False 表示禁用
        """
        # 如果使用簡化模式，檢查 ablation_flags
        if self._use_simplified_rewards:
            return self._ablation_flags.get(reward_name, False)
        # 完整模式下所有獎勵都啟用
        return True

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """
        【物理模擬前的準備工作】
        
        這個函數在每次物理計算之前被呼叫。
        
        做兩件事：
        1. 記住上一次的動作（之後用來計算「動作變化率」）
        2. 接收新的動作，並確保數值在合理範圍內 [-1, 1]
        
        為什麼要 clamp（限制範圍）？
        神經網路有時候會輸出很大或很小的數值，
        限制在 [-1, 1] 可以防止失控。
        """
        self.last_actions = self.actions.clone()           # 記住舊動作
        self.actions = actions.clone().clamp(-1.0, 1.0)    # 接收並限制新動作

    def _apply_action(self) -> None:
        """
        【將 AI 的動作指令轉換成實際的關節控制】
        
        ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        ★ RHex 非對稱 Duty Cycle 步態控制邏輯                               ★
        ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        
        【控制架構】
        ┌─────────────────────────────────────────────────────────────────┐
        │ AI 動作 [0:6]  → 速度調節因子（微調基礎速度 ±50%）             │
        │ AI 動作 [6:12] → ABAD 關節目標位置                              │
        └─────────────────────────────────────────────────────────────────┘
        
        【核心邏輯：根據相位決定基礎速度】
        1. 計算每隻腿當前的相位角度
        2. 判斷是在「著地相位」還是「擺動相位」
        3. 著地相位 → 使用慢速 (stance_velocity)
        4. 擺動相位 → 使用快速 (swing_velocity)
        5. AI 只能微調（±50%），不能完全停止
        
        【為什麼這樣設計？】
        - 確保步態結構正確（著地慢、擺動快）
        - 給 AI 調整空間（適應不同地形和速度需求）
        - 防止 AI 學會「停下來偷懶」的策略
        """
        # =====================================================================
        # 步驟 1：計算每隻腿的當前相位
        # =====================================================================
        # 從關節角度計算相位（考慮方向乘數）
        main_drive_pos = self.joint_pos[:, self._main_drive_indices]  # [N, 6]
        effective_pos = main_drive_pos * self._direction_multiplier   # 修正左右方向
        
        # 將角度正規化到 [0, 2π] 範圍
        leg_phase = torch.remainder(effective_pos, 2 * math.pi)  # [N, 6]
        
        # =====================================================================
        # 步驟 2：判斷每隻腿是否在著地相位
        # =====================================================================
        # 著地相位範圍可能跨越 0°/360° 邊界
        # 例如：stance_phase_start = -30° (-π/6), stance_phase_end = +30° (π/6)
        # 需要特殊處理！

        if self.stance_phase_start < 0:
            # 著地區間跨越 0° 邊界：例如 330° ~ 30° (即 -30° ~ +30°)
            # 正規化 start 到 [0, 2π]
            normalized_start = self.stance_phase_start + 2 * math.pi  # 如 330° (11π/6)

            # 腿相位在著地區間內的條件：
            # phase >= normalized_start (如 >= 330°) 或 phase < stance_phase_end (如 < 30°)
            in_stance_phase = (leg_phase >= normalized_start) | (leg_phase < self.stance_phase_end)
        else:
            # 著地區間不跨越邊界：正常比較
            in_stance_phase = (leg_phase >= self.stance_phase_start) & (leg_phase < self.stance_phase_end)

        # in_stance_phase: [N, 6] 布林張量

        # 記錄相位狀態（用於獎勵計算）
        self._current_leg_in_stance = in_stance_phase
        
        # =====================================================================
        # 步驟 3：根據相位選擇基礎速度
        # =====================================================================
        # 著地相位 → 很慢 (stance_velocity ≈ 0.94 rad/s ≈ 54°/s)
        #           只轉一小段角度 (60°)，但花 65% 的時間
        # 擺動相位 → 很快 (swing_velocity ≈ 9.42 rad/s ≈ 540°/s)
        #           要轉大段角度 (300°)，只花 35% 的時間
        base_velocity = torch.where(
            in_stance_phase,
            torch.full_like(leg_phase, self.stance_velocity),   # 著地：很慢
            torch.full_like(leg_phase, self.swing_velocity)     # 擺動：很快
        )  # [N, 6]
        
        # =====================================================================
        # 步驟 4：應用 AI 的速度調節因子
        # =====================================================================
        # AI 動作 [0:6] 範圍 [-1, 1]
        # 轉換成速度乘數 [0.5, 1.5]，讓 AI 可以微調但不能停下來
        drive_actions = self.actions[:, :6]
        speed_scale = 1.0 + drive_actions * 0.5  # 範圍 [0.5, 1.5]
        
        # 計算目標速度
        target_speed = base_velocity * speed_scale  # [N, 6]
        
        # =====================================================================
        # 步驟 5：應用方向乘數（左右腿轉向相反）
        # =====================================================================
        # 右側腿（索引 0,1,2）乘以 -1 → 逆時針轉
        # 左側腿（索引 3,4,5）乘以 +1 → 順時針轉
        target_drive_vel = target_speed * self._direction_multiplier
        
        # 安全限制：防止速度過快
        max_vel = self.swing_velocity * 1.5  # 允許最大 1.5 倍擺動速度
        target_drive_vel = torch.clamp(target_drive_vel, min=-max_vel, max=max_vel)
        
        # =====================================================================
        # ★★★ 正左/正右側移時鎖住主驅動關節 ★★★
        # =====================================================================
        # 當命令是純側移（X 速度 ≈ 0，Y 速度 ≠ 0，旋轉 ≈ 0）時，
        # 主驅動關節轉動會產生前進速度，這與目標衝突！
        # 純側移應該只靠 ABAD 關節（腳往外擺）來推動。
        #
        # 重要：側移前要先把腳調回預設位置，確保六隻腳都接觸地面！
        # 右側腿：45°，左側腿：-45°
        #
        # 判斷條件：
        # - |cmd_x| < 0.05 (前後速度接近零)
        # - |cmd_y| > 0.1  (有明顯的側移需求)
        # - |cmd_yaw| < 0.1 (不旋轉)
        
        cmd_x = self.commands[:, 0]    # 前後速度指令
        cmd_y = self.commands[:, 1]    # 左右速度指令
        cmd_yaw = self.commands[:, 2]  # 旋轉速度指令
        
        # 判斷是否為純側移模式
        is_pure_lateral = (
            (torch.abs(cmd_x) < 0.05) &      # 前後速度接近零
            (torch.abs(cmd_y) > 0.1) &       # 有側移需求
            (torch.abs(cmd_yaw) < 0.1)       # 不旋轉
        )  # [N]
        
        # =====================================================================
        # ★★★ 側移步態控制邏輯 - 完全重寫版 ★★★
        # =====================================================================
        # 
        # 核心設計理念（根據用戶反饋）：
        # 
        # 側移 ≠ 腿快速轉動 + 摩擦滑動（這是 reward exploit！）
        # 側移 = 類似「側併步」的步態
        # 
        # 正確的側移動作：
        # 1. 先讓主驅動回到初始位置（六腳著地）
        # 2. 確認到位後，鎖住主驅動（腿保持在著地位置不轉）
        # 3. 左右交替進行：
        #    - 一側的腿踩穩地面，ABAD 內收（向身體內側推）
        #    - 另一側的腿稍微抬起（可選），ABAD 外展（準備跨出）
        # 4. 頻率要慢（約 0.25 Hz = 4秒一週期），一步一步來
        # 
        # AI 的控制權：
        # - 主驅動：側移時 AI 完全沒有控制權（硬編碼鎖住速度=0）
        # - ABAD：AI 有部分控制權，但受到相位引導
        # =====================================================================
        
        lateral_mask = is_pure_lateral.unsqueeze(1).expand(-1, 6)  # [N, 6]
        
        # 保存側移狀態（用於診斷）
        self._is_pure_lateral = is_pure_lateral
        
        # 診斷計數器
        if not hasattr(self, '_lateral_debug_counter'):
            self._lateral_debug_counter = 0
        self._lateral_debug_counter += 1
        
        # 保存目標速度（用於獎勵計算和診斷）
        self._target_drive_vel = target_drive_vel.clone()
        self._base_velocity = base_velocity.clone()
        
        # =====================================================================
        # 初始化側移相關狀態
        # =====================================================================
        if not hasattr(self, '_lateral_gait_phase'):
            self._lateral_gait_phase = torch.zeros(self.num_envs, device=self.device)
            self._lateral_gait_freq = 0.25  # 0.25 Hz = 4 秒一個完整週期
        
        # 主驅動初始位置（側移時腿要回到的目標位置）
        # 順序：[右前, 右中, 右後, 左前, 左中, 左後]
        # 右側 45°，左側 -45°，確保六腳都向下著地
        if not hasattr(self, '_main_drive_initial_pos'):
            self._main_drive_initial_pos = torch.tensor(
                [45.0, 45.0, 45.0, -45.0, -45.0, -45.0],
                device=self.device
            ) * math.pi / 180.0  # 轉換為弧度
            self._main_drive_initial_pos = self._main_drive_initial_pos.unsqueeze(0)  # [1, 6]
        
        # 側移準備狀態追蹤（是否已到達初始位置）
        if not hasattr(self, '_lateral_ready'):
            self._lateral_ready = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        # 構建最終的速度目標
        final_drive_vel = target_drive_vel.clone()
        
        if is_pure_lateral.any():
            # ====== 讀取當前主驅動位置 ======
            current_main_drive_pos = self.joint_pos[:, self._main_drive_indices]  # [N, 6]
            
            # ====== 計算與初始位置的誤差 ======
            # 注意：角度是循環的，需要處理 wrap-around
            pos_error = current_main_drive_pos - self._main_drive_initial_pos  # [N, 6]
            # 將誤差限制在 [-π, π] 範圍內
            pos_error = torch.atan2(torch.sin(pos_error), torch.cos(pos_error))
            pos_error_abs = torch.abs(pos_error)  # [N, 6]
            
            # ====== 判斷是否已到達初始位置 ======
            # 所有腿的誤差都小於閾值才算準備好
            pos_threshold = 0.15  # 約 8.6 度的容許誤差
            all_legs_ready = (pos_error_abs.max(dim=1).values < pos_threshold)  # [N]
            
            # 更新準備狀態（一旦準備好就保持，直到離開側移模式）
            self._lateral_ready = torch.where(
                is_pure_lateral,
                self._lateral_ready | all_legs_ready,  # 保持或更新
                torch.zeros_like(self._lateral_ready)  # 離開側移模式時重置
            )
            
            # ====== 準備階段：驅動腿回到初始位置 ======
            # 對於還沒準備好的環境，使用位置控制讓腿回到初始位置
            preparing_mask = is_pure_lateral & (~self._lateral_ready)  # [N]
            
            if preparing_mask.any():
                # 計算回到初始位置所需的速度
                # 使用 P 控制器：速度 ∝ 位置誤差
                p_gain = 3.0  # 比例增益
                return_vel = -pos_error * p_gain  # [N, 6]
                # 限制最大速度，避免過快
                return_vel = torch.clamp(return_vel, min=-2.0, max=2.0)
                
                # 準備中的環境使用回歸速度
                preparing_mask_expanded = preparing_mask.unsqueeze(1).expand(-1, 6)
                final_drive_vel = torch.where(
                    preparing_mask_expanded,
                    return_vel,
                    final_drive_vel
                )
            
            # ====== 側移執行階段：鎖住主驅動 ======
            # 對於已準備好的環境，主驅動完全停止
            ready_mask = is_pure_lateral & self._lateral_ready  # [N]
            
            if ready_mask.any():
                # 更新側移步態相位
                dt_sim = self.cfg.sim.dt * self.cfg.decimation
                phase_increment = 2 * math.pi * self._lateral_gait_freq * dt_sim
                
                self._lateral_gait_phase = torch.where(
                    ready_mask,
                    (self._lateral_gait_phase + phase_increment) % (2 * math.pi),
                    self._lateral_gait_phase
                )
                
                # ★★★ 核心：側移時主驅動完全停止（速度=0）！★★★
                ready_mask_expanded = ready_mask.unsqueeze(1).expand(-1, 6)
                final_drive_vel = torch.where(
                    ready_mask_expanded,
                    torch.zeros(self.num_envs, 6, device=self.device),
                    final_drive_vel
                )
            
            # 記錄側移狀態
            self._is_lateral_mode = ready_mask.clone()  # 只有準備好的才算真正側移模式
            self._lateral_phase_for_abad = self._lateral_gait_phase.clone()
            
            # 診斷
            if self._lateral_debug_counter % 500 == 1 and is_pure_lateral[0]:
                max_err = pos_error_abs[0].max().item() * 180 / math.pi
                ready_status = "✓ 已就位" if self._lateral_ready[0] else f"準備中 (誤差: {max_err:.1f}°)"
                print(f"[側移模式] {ready_status}, 相位: {self._lateral_gait_phase[0]:.2f}")
        else:
            self._is_lateral_mode = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            # 離開側移模式時重置準備狀態
            self._lateral_ready = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        # 發送速度指令
        self.robot.set_joint_velocity_target(final_drive_vel, joint_ids=self._main_drive_indices)
        
        # =====================================================================
        # ABAD 關節控制 - 根據模式調整
        # =====================================================================
        abad_actions = self.actions[:, 6:12]
        base_abad_pos = abad_actions * self.cfg.abad_pos_scale
        
        # ★★★ 側移時的 ABAD 特殊處理 ★★★
        # 只有在準備好後才執行側移 ABAD 動作
        if hasattr(self, '_is_lateral_mode') and self._is_lateral_mode.any():
            ready_mask = self._is_lateral_mode  # 只處理已準備好的環境
            lateral_mask_for_abad = ready_mask.unsqueeze(1).expand(-1, 6)
            
            lateral_dir = torch.sign(cmd_y)  # +1 = 向左，-1 = 向右
            phase_sin = torch.sin(self._lateral_gait_phase)
            abad_amplitude = 0.3  # 約 17 度
            
            # 計算側移的理想 ABAD 位置
            right_abad_target = -lateral_dir * phase_sin * abad_amplitude
            left_abad_target = lateral_dir * phase_sin * abad_amplitude
            
            lateral_abad_pos = torch.stack([
                right_abad_target, right_abad_target, right_abad_target,
                left_abad_target, left_abad_target, left_abad_target
            ], dim=1)
            
            # 混合：60% 硬編碼 + 40% AI
            blended_abad = 0.6 * lateral_abad_pos + 0.4 * base_abad_pos
            
            target_abad_pos = torch.where(lateral_mask_for_abad, blended_abad, base_abad_pos)
        else:
            target_abad_pos = base_abad_pos
        
        target_abad_pos = torch.clamp(target_abad_pos, min=-0.5, max=0.5)
        self.robot.set_joint_position_target(target_abad_pos, joint_ids=self._abad_indices)
        
        # =====================================================================
        # 避震關節控制：保持固定位置
        # =====================================================================
        self.robot.set_joint_position_target(
            self._damper_initial_pos.expand(self.num_envs, -1), 
            joint_ids=self._damper_indices
        )

    def _get_observations(self) -> dict:
        """
        【計算觀測值】
        
        觀測值 = 機器人能「感知」到的所有資訊
        
        這就像機器人的「眼睛」和「感覺」：
        - 它知道自己在動還是靜止（速度）
        - 它知道自己有沒有傾斜（重力方向）
        - 它知道腿現在轉到哪裡（關節位置）
        - 它知道上一次做了什麼動作（用於預測）
        
        AI 神經網路會根據這些觀測值，決定下一步要怎麼做。
        
        返回：
            dict: 包含 "policy" 鍵的字典，值是觀測向量
        """
        # 先更新內部狀態（從模擬器讀取最新數據）
        self._update_state()

        # 讀取主驅動關節的位置和速度
        main_drive_pos = self.joint_pos[:, self._main_drive_indices]
        main_drive_vel = self.joint_vel[:, self._main_drive_indices]
        
        # 【為什麼用 sin/cos 表示旋轉位置？】
        # 
        # 問題：旋轉角度是「循環」的（0° = 360°）
        # 如果直接用角度值，神經網路會以為 0° 和 359° 差很遠，
        # 但其實它們只差 1°！
        # 
        # 解決方案：用 sin 和 cos 來表示
        # • sin(0°) = 0, cos(0°) = 1
        # • sin(359°) ≈ 0, cos(359°) ≈ 1  ← 很接近！
        # 
        # 這樣神經網路就能理解角度的「循環」性質了。
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

        # 【添加觀測噪音】
        # 為什麼要加噪音？模擬真實世界感測器的誤差！
        # 這樣訓練出來的 AI 更能適應真實環境的不完美感測。
        if self.cfg.add_noise:
            noise = torch.randn_like(obs) * 0.01 * self.cfg.noise_level
            obs = obs + noise

        # 【數值保護：防止異常值】
        # nan = 「不是數字」（計算錯誤時會出現）
        # inf = 「無限大」（除以零等情況會出現）
        # 這些異常值會讓神經網路爆炸，所以要處理掉
        obs = torch.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0)
        obs = torch.clamp(obs, min=-100.0, max=100.0)  # 限制在合理範圍

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
        
        # 更新速度命令（定期切換方向）
        self._update_commands()

    def _compute_simplified_rewards(self) -> torch.Tensor:
        """
        =================================================================
        【獎勵系統 v3.0】★★★ 大刀闘斧改革版 ★★★
        =================================================================
        
        v3.0 改變：
        1. 移除 × dt：讓獎勵更顯著（原本 × 0.004 太小）
        2. 增加獎勵權重：讓 gradient 更明確
        3. 簡化結構：只有 6 項獎勵
        
        獎勵結構：
        1. forward_progress  - 往命令方向移動就給獎勵（核心！）
        2. velocity_tracking - 速度越準獎勵越多
        3. height_maintain   - 站穩給獎勵
        4. leg_moving        - 腿在轉就給獎勵（防消極）
        5. action_smooth     - 動作平滑懲罰
        6. fall              - 摔倒懲罰
        """
        total_reward = torch.zeros(self.num_envs, device=self.device)
        
        # 獲取獎勵權重（v3.0 - 更大的權重！）
        scales = getattr(self.cfg, 'v2_reward_scales', {
            "forward_progress": 5.0,
            "velocity_tracking": 3.0,
            "height_maintain": 1.0,
            "leg_moving": 1.0,
            "action_smooth": -0.01,
            "fall": -10.0,
        })
        
        # =================================================================
        # 獲取狀態
        # =================================================================
        main_drive_vel = self.joint_vel[:, self._main_drive_indices]
        
        cmd_vx = self.commands[:, 0]
        cmd_vy = self.commands[:, 1]  
        cmd_wz = self.commands[:, 2]
        
        actual_vx = self.base_lin_vel[:, 0]
        actual_vy = self.base_lin_vel[:, 1]
        actual_wz = self.base_ang_vel[:, 2]
        
        base_height = self.robot.data.root_pos_w[:, 2]
        
        # =================================================================
        # R1: 前進獎勵 ★★★ 最重要！★★★
        # =================================================================
        # 往命令方向移動就給獎勵，反向就扣分
        
        forward_sign = torch.sign(cmd_vx + 1e-8)
        forward_progress = actual_vx * forward_sign
        
        lateral_sign = torch.sign(cmd_vy + 1e-8)
        lateral_progress = actual_vy * lateral_sign
        
        rotation_sign = torch.sign(cmd_wz + 1e-8)
        rotation_progress = actual_wz * rotation_sign
        
        # ★★★ 移除 dt，直接給獎勵 ★★★
        rew_forward = (
            forward_progress * 5.0 +
            lateral_progress * 3.0 +
            rotation_progress * 2.0
        ) * scales.get("forward_progress", 5.0) / 5.0
        
        total_reward += rew_forward
        
        # =================================================================
        # R2: 速度追蹤獎勵（線性）
        # =================================================================
        vel_error_x = torch.abs(cmd_vx - actual_vx)
        vel_error_y = torch.abs(cmd_vy - actual_vy)
        wz_error = torch.abs(cmd_wz - actual_wz)
        
        tolerance = 0.5
        
        tracking_x = torch.clamp(1.0 - vel_error_x / tolerance, min=0.0)
        tracking_y = torch.clamp(1.0 - vel_error_y / tolerance, min=0.0)
        tracking_wz = torch.clamp(1.0 - wz_error / 1.0, min=0.0)
        
        # ★★★ 移除 dt ★★★
        rew_tracking = (tracking_x + tracking_y * 0.5 + tracking_wz * 0.3) * scales.get("velocity_tracking", 3.0)
        total_reward += rew_tracking
        
        # =================================================================
        # R3: 站立高度獎勵
        # =================================================================
        min_height = 0.05
        target_height = 0.15
        
        height_ratio = torch.clamp(
            (base_height - min_height) / (target_height - min_height),
            min=0.0, max=1.0
        )
        # ★★★ 移除 dt ★★★
        rew_height = height_ratio * scales.get("height_maintain", 1.0)
        total_reward += rew_height
        
        # =================================================================
        # R4: 腿轉動獎勵（防消極）
        # =================================================================
        leg_speed = torch.abs(main_drive_vel * self._direction_multiplier).mean(dim=1)
        # ★★★ 移除 dt，增加權重 ★★★
        rew_leg_moving = torch.clamp(leg_speed * scales.get("leg_moving", 1.0), max=5.0)
        total_reward += rew_leg_moving
        
        # =================================================================
        # R5: 動作平滑懲罰
        # =================================================================
        action_rate = torch.sum(torch.square(self.actions - self.last_actions), dim=1)
        # ★★★ 移除 dt ★★★
        rew_smooth = action_rate * scales.get("action_smooth", -0.01)
        total_reward += rew_smooth
        
        # =================================================================
        # R6: 摔倒懲罰
        # =================================================================
        gravity_alignment = torch.sum(
            self.projected_gravity * self.reference_projected_gravity, dim=1
        )
        body_tilt = 1.0 - gravity_alignment
        
        is_fallen = (base_height < 0.03) | (body_tilt > 1.5)
        # ★★★ 移除 dt，增加懲罰 ★★★
        rew_fall = is_fallen.float() * scales.get("fall", -10.0)
        total_reward += rew_fall
        
        # 保存用於終止條件
        self._body_contact = is_fallen
        self._body_tilt = body_tilt
        
        # NaN 保護
        total_reward = torch.nan_to_num(total_reward, nan=0.0, posinf=20.0, neginf=-20.0)
        
        # TensorBoard 記錄
        self.episode_sums["rew_forward"] = self.episode_sums.get("rew_forward", torch.zeros_like(total_reward)) + rew_forward
        self.episode_sums["rew_tracking"] = self.episode_sums.get("rew_tracking", torch.zeros_like(total_reward)) + rew_tracking
        self.episode_sums["rew_height"] = self.episode_sums.get("rew_height", torch.zeros_like(total_reward)) + rew_height
        self.episode_sums["rew_leg_moving"] = self.episode_sums.get("rew_leg_moving", torch.zeros_like(total_reward)) + rew_leg_moving
        self.episode_sums["rew_smooth"] = self.episode_sums.get("rew_smooth", torch.zeros_like(total_reward)) + rew_smooth
        self.episode_sums["rew_fall"] = self.episode_sums.get("rew_fall", torch.zeros_like(total_reward)) + rew_fall
        
        # 診斷
        self.episode_sums["diag_forward_vel"] = self.episode_sums.get("diag_forward_vel", torch.zeros_like(total_reward)) + actual_vx
        self.episode_sums["diag_lateral_vel"] = self.episode_sums.get("diag_lateral_vel", torch.zeros_like(total_reward)) + actual_vy
        self.episode_sums["diag_cmd_vx"] = self.episode_sums.get("diag_cmd_vx", torch.zeros_like(total_reward)) + cmd_vx
        self.episode_sums["diag_cmd_vy"] = self.episode_sums.get("diag_cmd_vy", torch.zeros_like(total_reward)) + cmd_vy
        self.episode_sums["diag_vel_error"] = self.episode_sums.get("diag_vel_error", torch.zeros_like(total_reward)) + vel_error_x
        self.episode_sums["diag_base_height"] = self.episode_sums.get("diag_base_height", torch.zeros_like(total_reward)) + base_height
        self.episode_sums["diag_tilt"] = self.episode_sums.get("diag_tilt", torch.zeros_like(total_reward)) + body_tilt
        self.episode_sums["diag_leg_speed"] = self.episode_sums.get("diag_leg_speed", torch.zeros_like(total_reward)) + leg_speed
        
        self.last_main_drive_vel = main_drive_vel.clone()
        
        return total_reward


    def _get_rewards(self) -> torch.Tensor:
        """
        =================================================================
        【獎勵函數】強化學習的核心！！
        =================================================================
        
        根據 use_simplified_rewards 配置選擇：
        - True: 使用簡化的 8 項核心獎勵
        - False: 使用完整的 50+ 項獎勵（向後相容）
        """
        # ★★★ 簡化模式 ★★★
        if self._use_simplified_rewards:
            return self._compute_simplified_rewards()
        
        # ★★★ 完整模式（向後相容）★★★
        # 初始化總獎勵（所有環境都從 0 開始累加）
        total_reward = torch.zeros(self.num_envs, device=self.device)
        dt = self.step_dt  # 時間步長（用於把獎勵縮放到正確的量級）


        # =================================================================
        # 【獲取當前狀態】
        # =================================================================
        # 讀取各種關節和機身的狀態，用於計算獎勵
        
        # 主驅動關節的速度和位置
        main_drive_vel = self.joint_vel[:, self._main_drive_indices]  # 形狀 [環境數, 6]
        main_drive_pos = self.joint_pos[:, self._main_drive_indices]  # 形狀 [環境數, 6]
        
        # ABAD 關節的位置和速度
        abad_pos = self.joint_pos[:, self._abad_indices]  # 形狀 [環境數, 6]
        abad_vel = self.joint_vel[:, self._abad_indices]  # 形狀 [環境數, 6]
        
        # 獲取左右兩側的 ABAD 位置（用於旋轉和側移獎勵）
        # 右側：索引 0, 1, 2；左側：索引 3, 4, 5
        abad_right = abad_pos[:, :3]  # 右側 ABAD
        abad_left = abad_pos[:, 3:]   # 左側 ABAD
        abad_right_mean = abad_right.mean(dim=1)
        abad_left_mean = abad_left.mean(dim=1)
        
        # 【目標速度命令】（這是 AI 要追蹤的目標）
        cmd_vx = self.commands[:, 0]  # 目標前進速度（正 = 前，負 = 後）
        cmd_vy = self.commands[:, 1]  # 目標側向速度（正 = 左，負 = 右）
        cmd_wz = self.commands[:, 2]  # 目標旋轉速度（正 = 逆時針，負 = 順時針）
        
        # 【實際速度】（機器人現在的速度）
        # 注意：這是「本體座標系」，意思是從機器人自己的角度看
        actual_vx = self.base_lin_vel[:, 0]  # 實際前後速度
        actual_vy = self.base_lin_vel[:, 1]  # 實際左右速度
        actual_vz = self.base_lin_vel[:, 2]  # 實際上下速度（理想情況應該 ≈ 0）
        actual_wz = self.base_ang_vel[:, 2]  # 實際旋轉速度
        
        # 【任務需求強度 S】
        # 這個數值表示「當前命令有多複雜」
        # • 純直走：S ≈ 0（不需要側移或旋轉）
        # • 側移 + 旋轉：S 很大（需要 ABAD 關節幫忙）
        S = torch.abs(cmd_vy) + 0.5 * torch.abs(cmd_wz)
        S0 = 0.3  # 歸一化閾值（超過這個值就算「複雜任務」）

        # =================================================================
        # G1: 速度追蹤獎勵（核心獎勵！）
        # =================================================================
        # 目標：讓機器人學會「聽命令」
        # 方法：命令速度和實際速度越接近，獎勵越高
        # 
        # 使用 exp（指數函數）映射的好處：
        # • 誤差 = 0 時，獎勵 = 1（完美！）
        # • 誤差越大，獎勵快速下降趨近 0
        # 這樣可以讓 AI 很清楚地知道「越準確越好」

        # 獲取 tracking_sigma 參數（預設 0.25，來自 legged_gym）
        tracking_sigma = getattr(self.cfg, 'tracking_sigma', 0.25)

        # G1.1 線速度追蹤（前後 + 左右）
        # 計算 XY 方向的速度誤差平方和
        lin_vel_error = torch.sum(
            torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1
        )
        # 公式：獎勵 = exp(-誤差² / sigma)
        # 當誤差 = 0 時，獎勵 = 1
        # sigma 控制衰減速度
        lin_vel_error_mapped = torch.exp(-lin_vel_error / tracking_sigma)
        rew_track_lin_vel = lin_vel_error_mapped * self.cfg.rew_scale_track_lin_vel * dt
        if self._is_reward_enabled("track_lin_vel"):
            total_reward += rew_track_lin_vel

        # G1.2 角速度追蹤（旋轉）
        # 計算旋轉速度的誤差
        yaw_rate_error = torch.square(cmd_wz - actual_wz)
        yaw_rate_error_mapped = torch.exp(-yaw_rate_error / tracking_sigma)
        rew_track_ang_vel = yaw_rate_error_mapped * self.cfg.rew_scale_track_ang_vel * dt
        if self._is_reward_enabled("track_ang_vel"):
            total_reward += rew_track_ang_vel
        
        # ★★★ G1.3 新增：旋轉專用獎勵（用戶說旋轉太慢）★★★
        # 判斷是否為純旋轉模式：只有旋轉命令，沒有線速度
        is_pure_rotation = (
            (torch.abs(cmd_vx) < 0.05) &
            (torch.abs(cmd_vy) < 0.05) &
            (torch.abs(cmd_wz) > 0.2)
        )
        
        # G1.3.1 旋轉速度不足懲罰
        # 如果命令要求旋轉但實際旋轉太慢，大懲罰
        rotation_speed_ratio = torch.abs(actual_wz) / (torch.abs(cmd_wz) + 0.01)
        rotation_too_slow = (rotation_speed_ratio < 0.5) & is_pure_rotation
        rew_rotation_slow_penalty = rotation_too_slow.float() * getattr(self.cfg, 'rew_scale_rotation_slow_penalty', -2.0) * dt
        if self._is_reward_enabled("rotation_slow_penalty"):  # 簡化模式下禁用
            total_reward += rew_rotation_slow_penalty
        
        # G1.3.2 旋轉時 ABAD 輔助獎勵
        # 原地旋轉時，用 ABAD 可以幫助產生旋轉力矩
        # 旋轉方向與 ABAD 配置應該一致
        # cmd_wz > 0 (逆時針)：右側 ABAD 外展，左側 ABAD 內收
        # cmd_wz < 0 (順時針)：右側 ABAD 內收，左側 ABAD 外展
        abad_for_rotation = (abad_right_mean - abad_left_mean) * torch.sign(cmd_wz)
        rew_rotation_abad_assist = torch.where(
            is_pure_rotation & (abad_for_rotation > 0.05),
            abad_for_rotation * getattr(self.cfg, 'rew_scale_rotation_abad_assist', 2.0),
            torch.zeros_like(abad_for_rotation)
        ) * dt
        if self._is_reward_enabled("rotation_abad_assist"):  # 簡化模式下禁用
            total_reward += rew_rotation_abad_assist
        
        # G1.3.3 旋轉方向正確大獎勵
        # 確保旋轉方向與命令一致
        rotation_direction_correct = torch.sign(cmd_wz) * actual_wz  # 方向正確時為正
        rew_rotation_correct = torch.where(
            torch.abs(cmd_wz) > 0.1,
            torch.clamp(rotation_direction_correct * 3.0, min=-2.0, max=4.0),
            torch.zeros_like(rotation_direction_correct)
        ) * dt
        total_reward += rew_rotation_correct

        # =================================================================
        # G2: 姿態與穩定性懲罰
        # =================================================================
        # 目標：讓機器人保持平穩，不要亂翻、亂跳、亂晃
        # 
        # 這些都是「懲罰」（負值），所以越少越好！
        
        # G2.1 直立性懲罰（不要傾斜）
        # 原理：如果機器人完全直立，重力方向 = [0, 0, -1]
        #       projected_gravity 的 xy 分量 = [0, 0]（都是 0）
        #       如果傾斜了，xy 分量就會變大
        # 所以：xy 分量越大 = 傾斜越多 = 懲罰越重
        flat_orientation = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
        rew_upright = flat_orientation * self.cfg.rew_scale_upright * dt
        if self._is_reward_enabled("upright"):  # 簡化模式下禁用
            total_reward += rew_upright
        
        # G2.2 垂直彈跳懲罰（不要亂跳）
        # 機器人應該平穩移動，上下速度（vz）應該接近 0
        z_vel_error = torch.square(actual_vz)
        rew_z_vel = z_vel_error * self.cfg.rew_scale_z_vel * dt
        if self._is_reward_enabled("z_vel"):  # 簡化模式下禁用
            total_reward += rew_z_vel
        
        # G2.3 XY 軸角速度懲罰（不要翻滾）
        # 機器人不應該繞 X 軸或 Y 軸旋轉（那是翻滾），只允許繞 Z 軸旋轉（正常轉彎）
        ang_vel_xy_error = torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
        rew_ang_vel_xy = ang_vel_xy_error * self.cfg.rew_scale_ang_vel_xy * dt
        if self._is_reward_enabled("ang_vel_xy"):  # 簡化模式下禁用
            total_reward += rew_ang_vel_xy
        
        # G2.4 高度維持懲罰（保持正常站立高度）
        # 正常站立高度約 0.12 公尺，偏離太多就扣分
        base_height = self.robot.data.root_pos_w[:, 2]  # 機身離地面的高度
        target_height = getattr(self.cfg, 'target_base_height', 0.12)  # 目標高度
        height_error = torch.square(base_height - target_height)
        rew_base_height = height_error * self.cfg.rew_scale_base_height * dt
        if self._is_reward_enabled("base_height"):  # 簡化模式下禁用
            total_reward += rew_base_height
        
        # ★★★ G2.5 新增：直走時高站姿獎勵 ★★★
        # 當命令是直走（|vx| 大，|vy| 和 |wz| 小）時，獎勵身體抬高
        # 這是用戶特別要求的：「直走時保持身體高度盡量抬高」
        is_forward_walk = (
            (torch.abs(cmd_vx) > 0.1) &      # 有前進命令
            (torch.abs(cmd_vy) < 0.1) &       # 側移很小
            (torch.abs(cmd_wz) < 0.2)         # 旋轉很小
        )
        
        # 直走時的目標高度更高
        target_height_forward = getattr(self.cfg, 'target_base_height_forward', 0.13)
        
        # 計算高站姿獎勵（越高越好，但有上限）
        height_above_target = torch.clamp(base_height - target_height, min=0.0, max=0.05)
        rew_high_stance = torch.where(
            is_forward_walk,
            height_above_target * getattr(self.cfg, 'rew_scale_high_stance', 2.0) * 20.0,  # 放大係數
            torch.zeros_like(base_height)
        ) * dt
        total_reward += rew_high_stance
        
        # G2.5 偏航角速度過大懲罰（當不需要旋轉時）
        # 若 |wz*| 很小，則懲罰 |wz| 過大
        wz_tol = 0.3
        unwanted_yaw = torch.where(
            torch.abs(cmd_wz) < wz_tol,
            torch.square(torch.clamp(torch.abs(actual_wz) - wz_tol, min=0.0)),
            torch.zeros_like(actual_wz)
        )
        rew_unwanted_yaw = -unwanted_yaw * 2.0 * dt
        total_reward += rew_unwanted_yaw

        # =================================================================
        # G3: 身體觸地懲罰（超級重要！！）
        # =================================================================
        # 目標：防止機器人「摔倒」或「翻車」
        # 
        # 什麼是「身體觸地」？
        # • 機身（不是腿）碰到地面 = 摔倒了！
        # • 這是非常糟糕的情況，要大力懲罰
        # 
        # 檢測方法（因為沒有接觸感測器，用間接方式判斷）：
        # 1. 高度太低 → 可能趴在地上
        # 2. 傾斜太大 → 可能翻倒了
        body_height = base_height
        
        # 【計算傾斜程度】
        # 方法：比較「現在的重力方向」和「正常站立時的重力方向」
        # 使用「點積」（Dot Product）來計算相似度：
        # • 點積 = 1  → 完全對齊（0° 傾斜，完美！）
        # • 點積 = 0  → 垂直（90° 傾斜，快翻了！）
        # • 點積 = -1 → 完全相反（180° 傾斜，完全翻過來了！）
        gravity_alignment = torch.sum(
            self.projected_gravity * self.reference_projected_gravity, dim=1
        )  # 範圍 [-1, 1]
        
        # 轉換成「傾斜程度」：
        # • body_tilt = 0 → 完全對齊（沒傾斜）
        # • body_tilt = 1 → 傾斜 90 度
        # • body_tilt = 2 → 翻轉 180 度
        body_tilt = 1.0 - gravity_alignment  # 範圍 [0, 2]
        
        # 【判斷「身體觸地」的條件】
        # 滿足任一條件就視為摔倒：
        # 
        # 條件 1：高度太低（< 0.01 公尺 = 1 公分）
        #         正常站立高度約 12 公分，低於 1 公分肯定是趴著了
        height_threshold = getattr(self.cfg, 'body_contact_height_threshold', 0.01)
        height_contact = body_height < height_threshold
        
        # 條件 2：傾斜太大
        # ★★★ 放寬到 1.2（約 100 度），讓機器人有更多學習機會 ★★★
        #         body_tilt > 1.2 對應傾斜超過約 100 度（幾乎翻轉）
        severe_tilt = body_tilt > 1.2
        
        # 任一條件成立 = 身體觸地！
        body_contact = height_contact | severe_tilt
        
        # 【身體觸地懲罰】摔倒是嚴重的錯誤，但不終止讓 AI 學習恢復
        rew_body_contact = body_contact.float() * self.cfg.rew_scale_body_contact * dt
        if self._is_reward_enabled("body_contact"):  # 簡化模式下保留
            total_reward += rew_body_contact
        
        # 【連續傾斜懲罰】傾斜越多扣分越多（鼓勵保持平衡）
        # ★ 降低懲罰強度，讓機器人更敢動
        # 傾斜小於 30 度：沒事
        # 傾斜超過 30 度：開始扣分
        tilt_penalty = torch.clamp(body_tilt - 0.3, min=0.0) * 2.0  # ★ 從 5.0 降到 2.0，門檻從 0.2 放寬到 0.3
        total_reward -= tilt_penalty * dt
        
        # 記錄用於終止條件
        self._body_contact = body_contact
        self._body_tilt = body_tilt  # 保存用於 _get_dones

        # =================================================================
        # G4: 能耗與動作平滑懲罰
        # =================================================================
        # 目標：讓機器人的動作更省力、更平順
        # 
        # 為什麼這很重要？
        # 1. 省電：真實機器人電池有限，不能浪費
        # 2. 保護硬體：劇烈動作會損壞馬達和關節
        # 3. 看起來更自然：平滑的動作比抖動好看
        
        # G4.1 力矩懲罰（不要用太大力）
        # 馬達出力越大，耗電越多，所以要懲罰大力矩
        if hasattr(self.robot.data, 'applied_torque'):
            joint_torques = torch.sum(torch.square(self.robot.data.applied_torque), dim=1)
            rew_torque = joint_torques * self.cfg.rew_scale_torque * dt
            if self._is_reward_enabled("torque"):  # 簡化模式下保留
                total_reward += rew_torque
        
        # G4.2 動作變化率懲罰（不要抖動）
        # 比較這次動作和上次動作，變化越大懲罰越重
        # 這樣可以讓動作更平滑，不會忽大忽小
        action_rate = torch.sum(torch.square(self.actions - self.last_actions), dim=1)
        rew_action_rate = action_rate * self.cfg.rew_scale_action_rate * dt
        if self._is_reward_enabled("action_rate"):  # 簡化模式下保留
            total_reward += rew_action_rate
        
        # G4.3 關節加速度懲罰（不要急加速）
        # 加速度太大 = 動作太劇烈，對機械結構不好
        if hasattr(self.robot.data, 'joint_acc'):
            joint_accel = torch.sum(torch.square(self.robot.data.joint_acc), dim=1)
            rew_joint_acc = joint_accel * self.cfg.rew_scale_joint_acc * dt
            total_reward += rew_joint_acc
        
        # ★★★ G4.4 高頻關節速度懲罰 ★★★
        # ★ 大幅降低！這個懲罰會讓機器人不敢動
        main_drive_speed = torch.abs(main_drive_vel).mean(dim=1)
        abad_speed = torch.abs(abad_vel).mean(dim=1)
        actual_move_speed = torch.sqrt(actual_vx**2 + actual_vy**2)
        
        # 效率指標：實際移動速度 / 關節速度
        joint_total_speed = main_drive_speed + abad_speed * 2.0
        efficiency = actual_move_speed / (joint_total_speed + 0.1)
        
        # 只懲罰極端低效率的情況（閾值更嚴格）
        cmd_has_velocity = (torch.abs(cmd_vx) > 0.05) | (torch.abs(cmd_vy) > 0.05)
        inefficient_motion = cmd_has_velocity & (joint_total_speed > 3.0) & (efficiency < 0.03)  # ★ 更嚴格的條件
        rew_sliding_penalty = -inefficient_motion.float() * 1.0 * dt  # ★ 從 -5.0 降到 -1.0
        total_reward += rew_sliding_penalty
        
        # ★★★ G4.5 高頻動作懲罰 - 完全移除！★★★
        # 這個懲罰是造成消極的主要原因之一
        # if hasattr(self, 'last_actions'):
        #     action_change = self.actions - self.last_actions
        #     action_change_magnitude = torch.sum(torch.square(action_change), dim=1)
        #     action_magnitude = torch.sum(torch.square(self.actions), dim=1)
        #     is_high_freq_jitter = (action_change_magnitude > 0.1) & (action_magnitude < 0.5)
        #     rew_high_freq_penalty = -is_high_freq_jitter.float() * 3.0 * dt
        #     total_reward += rew_high_freq_penalty

        # =================================================================
        # G5: 步態結構獎勵 ★★★ RHex 非對稱 Duty Cycle 核心獎勵 ★★★
        # =================================================================
        # 
        # 【目標】確保 RHex 風格的「著地慢轉、擺動快轉」步態正確執行
        # 
        # ┌───────────────────────────────────────────────────────────────┐
        # │ G5.1 組內同步：同組三腳相位一致                              │
        # │ G5.2 連續支撐：任何時刻至少一組著地 ★最重要★               │
        # │ G5.3 正確速度：著地慢轉、擺動快轉                            │
        # │ G5.4 交替步態：兩組交替著地                                  │
        # │ G5.5 頻率一致：整體步態頻率正確                              │
        # └───────────────────────────────────────────────────────────────┘
        
        # ---------------------------------------------------------------------
        # 計算每隻腿的「相位」（考慮方向乘數）
        # ---------------------------------------------------------------------
        effective_pos = main_drive_pos * self._direction_multiplier
        leg_phase = torch.remainder(effective_pos, 2 * math.pi)  # [N, 6]
        
        # 分開兩組的相位
        phase_a = leg_phase[:, self._tripod_a_indices]  # Tripod A: 腿 0, 3, 5
        phase_b = leg_phase[:, self._tripod_b_indices]  # Tripod B: 腿 1, 2, 4
        
        # ---------------------------------------------------------------------
        # G5.1 組內同步獎勵（同組的三隻腳應該相位一致）
        # ---------------------------------------------------------------------
        # 使用「相位一致性」(Phase Coherence) 來衡量同步程度
        # coherence = |mean(e^(i*phase))| 
        # = 1 時表示所有相位完全相同，= 0 表示完全分散
        
        def phase_coherence(phases):
            """計算相位一致性（0~1）"""
            sin_mean = torch.sin(phases).mean(dim=1)
            cos_mean = torch.cos(phases).mean(dim=1)
            return torch.sqrt(sin_mean**2 + cos_mean**2)
        
        coherence_a = phase_coherence(phase_a)  # A 組同步程度
        coherence_b = phase_coherence(phase_b)  # B 組同步程度
        
        # 獎勵：兩組都同步 → 給獎勵
        rew_tripod_sync = (coherence_a + coherence_b) * self.cfg.rew_scale_tripod_sync * dt
        total_reward += rew_tripod_sync
        
        # ---------------------------------------------------------------------
        # G5.2 連續支撐獎勵 ★★★ 最重要的步態獎勵！★★★
        # ---------------------------------------------------------------------
        # 確保任何時刻都有至少一組在「著地相位」
        # 
        # 判斷每隻腿是否在著地相位
        if hasattr(self, '_current_leg_in_stance'):
            leg_in_stance = self._current_leg_in_stance  # [N, 6]
        else:
            # 回退方案
            leg_in_stance = (leg_phase >= self.stance_phase_start) & (leg_phase < self.stance_phase_end)
        
        # 計算每組有幾隻腳在著地
        stance_count_a = leg_in_stance[:, self._tripod_a_indices].float().sum(dim=1)  # [N]
        stance_count_b = leg_in_stance[:, self._tripod_b_indices].float().sum(dim=1)  # [N]
        
        # 判斷每組是否「有效著地」（至少 2 隻腳在著地相位）
        a_in_stance = stance_count_a >= 2
        b_in_stance = stance_count_b >= 2
        
        # ★ 連續支撐獎勵：至少一組有效著地
        at_least_one_stance = (a_in_stance | b_in_stance).float()
        rew_tripod_support = at_least_one_stance * self.cfg.rew_scale_tripod_support * dt
        if self._is_reward_enabled("tripod_support"):  # 簡化模式下保留
            total_reward += rew_tripod_support
        
        # ★★ 騰空懲罰：如果兩組都不在著地相位 → 大懲罰！
        both_airborne = (~a_in_stance & ~b_in_stance).float()
        rew_airborne_penalty = both_airborne * getattr(self.cfg, 'rew_scale_airborne_penalty', -10.0) * dt
        if self._is_reward_enabled("airborne_penalty"):  # 簡化模式下禁用
            total_reward += rew_airborne_penalty
        
        # ★★★ 雙支撐獎勵：兩組都著地時是超級穩定狀態
        both_in_stance = (a_in_stance & b_in_stance).float()
        rew_double_support = both_in_stance * 1.0 * dt  # 額外獎勵重疊期
        total_reward += rew_double_support
        
        # ---------------------------------------------------------------------
        # G5.3 正確速度比例獎勵（著地慢轉、擺動快轉）
        # ---------------------------------------------------------------------
        # 獎勵腿在正確的相位使用正確的速度
        # 
        # 期望：
        # - 著地相位的腿：速度 ≈ stance_velocity
        # - 擺動相位的腿：速度 ≈ swing_velocity
        
        if hasattr(self, '_base_velocity'):
            # 使用 _apply_action 中計算的基礎速度
            expected_velocity = self._base_velocity  # [N, 6]
        else:
            # 回退方案
            expected_velocity = torch.where(
                leg_in_stance,
                torch.full_like(leg_phase, self.stance_velocity),
                torch.full_like(leg_phase, self.swing_velocity)
            )
        
        # 計算實際速度與期望速度的誤差（考慮方向）
        actual_signed_vel = main_drive_vel * self._direction_multiplier  # 修正方向
        velocity_error = torch.abs(torch.abs(actual_signed_vel) - expected_velocity)
        
        # 正規化誤差並計算獎勵
        normalized_vel_error = velocity_error / self.swing_velocity  # 正規化
        velocity_match = torch.exp(-2.0 * normalized_vel_error.mean(dim=1))  # 指數映射
        rew_velocity = velocity_match * self.cfg.rew_scale_duty_cycle_velocity * dt
        total_reward += rew_velocity
        
        # ---------------------------------------------------------------------
        # G5.4 交替步態獎勵（兩組應該交替著地，不是同時）
        # ---------------------------------------------------------------------
        # 理想情況：
        # - 一組在著地相位末端（即將進入擺動）
        # - 另一組在著地相位中段（準備接手支撐）
        # 
        # 這不是強制 180° 反相，而是獎勵「平滑交接」
        
        # 計算兩組的平均相位
        mean_phase_a = torch.atan2(torch.sin(phase_a).mean(dim=1), torch.cos(phase_a).mean(dim=1))
        mean_phase_b = torch.atan2(torch.sin(phase_b).mean(dim=1), torch.cos(phase_b).mean(dim=1))
        
        # 計算相位差（應該接近某個值，但不強制是 π）
        phase_diff = torch.abs(mean_phase_a - mean_phase_b)
        phase_diff = torch.min(phase_diff, 2 * math.pi - phase_diff)  # 處理循環
        
        # 獎勵相位差在合理範圍內（π ± 0.5）
        # 這比強制 180° 更寬鬆，允許步態有一定的靈活性
        target_phase_diff = math.pi
        phase_diff_tolerance = 0.8  # 允許 ±0.8 弧度（約 ±46°）的誤差
        phase_diff_error = torch.abs(phase_diff - target_phase_diff)
        phase_diff_in_range = (phase_diff_error < phase_diff_tolerance).float()
        rew_alternation = phase_diff_in_range * getattr(self.cfg, 'rew_scale_tripod_alternation', 1.5) * dt
        total_reward += rew_alternation
        
        # 舊版 antiphase 獎勵（權重通常為 0，保留向後相容）
        if self.cfg.rew_scale_tripod_antiphase != 0:
            phase_diff_error_old = torch.abs(phase_diff - math.pi)
            rew_antiphase = torch.exp(-phase_diff_error_old) * self.cfg.rew_scale_tripod_antiphase * dt
            total_reward += rew_antiphase
        
        # ---------------------------------------------------------------------
        # G5.5 步態頻率一致性獎勵
        # ---------------------------------------------------------------------
        # 整體的「平均轉速」應該接近目標頻率
        mean_abs_vel = torch.abs(main_drive_vel * self._direction_multiplier).mean(dim=1)
        
        # 目標平均速度（考慮 duty cycle 的加權平均）
        target_mean_vel = (self.cfg.stance_duty_cycle * self.stance_velocity + 
                          (1 - self.cfg.stance_duty_cycle) * self.swing_velocity)
        
        freq_error = torch.abs(mean_abs_vel - target_mean_vel) / target_mean_vel
        freq_match = torch.exp(-2.0 * freq_error)
        rew_frequency = freq_match * getattr(self.cfg, 'rew_scale_gait_frequency', 1.0) * dt
        total_reward += rew_frequency
        
        # ---------------------------------------------------------------------
        # 診斷：記錄步態狀態
        # ---------------------------------------------------------------------
        if not hasattr(self, '_gait_debug_counter'):
            self._gait_debug_counter = 0
        self._gait_debug_counter += 1
        
        # 每 500 步打印一次步態診斷
        if self._gait_debug_counter % 500 == 1:
            print(f"[步態診斷] A組著地: {stance_count_a[0]:.0f}/3, B組著地: {stance_count_b[0]:.0f}/3, "
                  f"相位差: {phase_diff[0]:.2f} rad ({phase_diff[0]*180/math.pi:.1f}°), "
                  f"平均速度: {mean_abs_vel[0]:.2f} rad/s")

        # =================================================================
        # G6: ABAD 使用策略獎勵
        # =================================================================
        # ABAD 關節的作用：幫助機器人側移和轉彎
        # 
        # 【設計原則 - 2025年大改版】
        # ★★★ 核心變化 ★★★
        # 1. 直走時：ABAD 應該保持在零度附近（用戶明確要求）
        # 2. 斜向移動時：用 ABAD 產生側向分量
        # 3. 純側移時：用 ABAD 配合側併步推動
        # 4. 旋轉時：用 ABAD 輔助旋轉
        
        # 計算 ABAD 的「使用量」（關節動得多不多）
        U_abad = torch.sum(torch.square(abad_vel), dim=1)  # 用速度平方和表示
        abad_magnitude = torch.abs(abad_pos).mean(dim=1)   # 用位置絕對值表示
        
        # ★★★ G6.0 直走時 ABAD 歸零獎勵（最重要的新獎勵！）★★★
        # 用戶特別強調：「直走時 ABAD 維持在一個正常零度」
        # 
        # 判斷直走條件：
        # - |vx| > 0.1（有前進命令）
        # - |vy| < 0.1（側移很小）
        # - |wz| < 0.2（旋轉很小）
        is_forward_walk_abad = (
            (torch.abs(cmd_vx) > 0.1) &
            (torch.abs(cmd_vy) < 0.1) &
            (torch.abs(cmd_wz) < 0.2)
        )
        
        # 計算 ABAD 偏離零度的程度
        abad_deviation = torch.sum(torch.square(abad_pos), dim=1)  # 所有 ABAD 位置的平方和
        
        # 直走時獎勵 ABAD 接近零度
        # 使用 exp(-deviation) 讓越接近零獎勵越高
        abad_zero_reward = torch.exp(-abad_deviation * 10.0)  # deviation=0 時獎勵=1
        rew_abad_zero_forward = torch.where(
            is_forward_walk_abad,
            abad_zero_reward * getattr(self.cfg, 'rew_scale_abad_zero_forward', 3.0),
            torch.zeros_like(abad_zero_reward)
        ) * dt
        total_reward += rew_abad_zero_forward
        
        # ★★★ G6.0.5 斜向移動時獎勵使用 ABAD ★★★
        # 用戶說：「左前右前就是要在直走的步態下,多加使用ABAD自由度來達成這個目標」
        # 判斷斜向移動：同時有前進和側移命令
        is_diagonal = (
            (torch.abs(cmd_vx) > 0.1) &   # 有前進
            (torch.abs(cmd_vy) > 0.1) &   # 有側移
            (torch.abs(cmd_wz) < 0.2)     # 不旋轉
        )
        
        # 斜向時獎勵 ABAD 有適當幅度
        # ABAD 幅度應該與側移命令成正比
        expected_abad_for_diagonal = torch.abs(cmd_vy) * 0.3  # 預期的 ABAD 幅度
        abad_diagonal_reward = torch.where(
            abad_magnitude > expected_abad_for_diagonal * 0.5,  # 至少要有一半
            torch.clamp(abad_magnitude / (expected_abad_for_diagonal + 0.01), max=2.0),
            torch.zeros_like(abad_magnitude)
        )
        rew_abad_diagonal = torch.where(
            is_diagonal,
            abad_diagonal_reward * getattr(self.cfg, 'rew_scale_abad_diagonal_use', 2.0),
            torch.zeros_like(abad_diagonal_reward)
        ) * dt
        total_reward += rew_abad_diagonal
        
        # G6.1 聰明使用獎勵（需要側移/轉彎時用 ABAD）
        # S 代表「任務複雜度」，S 越大 = 越需要 ABAD
        # 當 S 大且 ABAD 有在用 → 給獎勵
        rew_abad_smart = S * torch.tanh(0.5 * U_abad) * self.cfg.rew_scale_abad_smart_use * dt
        total_reward += rew_abad_smart
        
        # G6.2 浪費懲罰（不需要時亂用 ABAD）
        # 當 S 小（直走）但 ABAD 亂動 → 給懲罰
        # 加強：直走時（is_forward_walk_abad）額外懲罰
        waste_factor = 1.0 - torch.clamp(S / S0, max=1.0)
        rew_abad_waste = waste_factor * U_abad * self.cfg.rew_scale_abad_waste * dt
        total_reward += rew_abad_waste
        
        # G6.3 側向速度追蹤獎勵（ABAD 產生側向速度）
        vy_sign_match = (cmd_vy * actual_vy) > 0
        lateral_tracking = torch.where(
            torch.abs(cmd_vy) > 0.05,
            vy_sign_match.float() * torch.abs(actual_vy) * 2.0,
            torch.zeros_like(actual_vy)
        ) * dt
        total_reward += lateral_tracking

        # =================================================================
        # G6.5: 側移專用獎勵 ★★★ 時間基準步態版本 ★★★
        # =================================================================
        # 核心改變：側移現在使用時間基準的交替步態（0.3 Hz 週期）
        # 
        # 新策略優勢：
        # 1. 使用全局步態相位控制左右交替，不依賴 ABAD 角度
        # 2. 使用 sin 曲線平滑插值，避免硬切換導致的抖動
        # 3. ABAD 只負責提供側向推力，不控制步態切換
        # 
        # 獎勵目標：
        # 1. 獎勵 ABAD 產生足夠的側向推力
        # 2. 獎勵跟隨時間基準步態的正確相位
        # 3. 懲罰高頻動作變化（但不要太嚴格以免凍住）
        
        # 檢測是否處於純側移模式
        is_lateral_mode = getattr(self, '_is_lateral_mode', torch.zeros(self.num_envs, dtype=torch.bool, device=self.device))
        
        # 計算 ABAD 幅度（對所有模式）
        abad_amplitude = torch.abs(abad_pos).mean(dim=1)
        
        # abad_right, abad_left, abad_right_mean, abad_left_mean 已在函數開頭定義
        
        if is_lateral_mode.any():
            # ==============================================================
            # G6.5.1 ABAD 交替獎勵（外八/內八）
            # ==============================================================
            # 獎勵左右 ABAD 反向動作，產生側向推力
            # 不再是控制步態的核心，只是輔助推力
            abad_product = abad_right_mean * abad_left_mean
            
            # 乘積為負數 = 反向 = 好 → 給獎勵
            abad_alternation_reward = torch.where(
                abad_product < -0.01,  # 降低閾值，更容易達成
                torch.sqrt(-abad_product + 0.01) * 2.0,
                torch.zeros_like(abad_product)
            )
            rew_abad_alternation = abad_alternation_reward * getattr(self.cfg, 'rew_scale_abad_alternation', 2.0) * dt
            rew_abad_alternation = torch.where(is_lateral_mode, rew_abad_alternation, torch.zeros_like(rew_abad_alternation))
            total_reward += rew_abad_alternation
            
            # ==============================================================
            # G6.5.2 ABAD 幅度獎勵
            # ==============================================================
            min_abad_amplitude = getattr(self.cfg, 'lateral_min_abad_amplitude', 0.15)
            
            # 獎勵公式：幅度超過閾值後給獎勵
            amplitude_bonus = torch.clamp(abad_amplitude - min_abad_amplitude, min=0.0)
            rew_abad_amplitude = amplitude_bonus * getattr(self.cfg, 'rew_scale_abad_amplitude', 2.0) * dt
            rew_abad_amplitude = torch.where(is_lateral_mode, rew_abad_amplitude, torch.zeros_like(rew_abad_amplitude))
            total_reward += rew_abad_amplitude
            
            # ==============================================================
            # G6.5.3 時間基準步態相位獎勵 ★★★ 新增 ★★★
            # ==============================================================
            # 獎勵機器人跟隨時間基準步態的正確相位
            # 使用 _lateral_lift_mask 來判斷腿是否在正確的抬起/著地狀態
            
            if hasattr(self, '_lateral_lift_mask') and hasattr(self, '_current_leg_in_stance'):
                expected_lift = self._lateral_lift_mask  # [N, 6] 根據時間相位，哪些腿應該抬起
                actual_stance = self._current_leg_in_stance  # [N, 6] 實際哪些腿著地
                
                # 正確的狀態：應該抬起的腿不著地，應該著地的腿著地
                # expected_lift=True → 應該抬起 → actual_stance=False 是正確的
                # expected_lift=False → 應該著地 → actual_stance=True 是正確的
                correct_phase = (expected_lift != actual_stance).float()  # [N, 6]
                phase_correctness = correct_phase.mean(dim=1)  # [N]
                
                rew_gait_phase = phase_correctness * getattr(self.cfg, 'rew_scale_lateral_gait_phase', 1.5) * dt
                rew_gait_phase = torch.where(is_lateral_mode, rew_gait_phase, torch.zeros_like(rew_gait_phase))
                total_reward += rew_gait_phase
            else:
                rew_gait_phase = torch.zeros(self.num_envs, device=self.device)
            
            # ==============================================================
            # G6.5.4 左右交替著地獎勵
            # ==============================================================
            # 現在由時間基準步態保證，這裡只是確認
            if hasattr(self, '_current_leg_in_stance'):
                right_legs_stance = self._current_leg_in_stance[:, :3].float().mean(dim=1)
                left_legs_stance = self._current_leg_in_stance[:, 3:].float().mean(dim=1)
            else:
                right_legs_stance = leg_in_stance[:, :3].float().mean(dim=1)
                left_legs_stance = leg_in_stance[:, 3:].float().mean(dim=1)
            
            # 獎勵一側著地另一側懸空
            one_side_stance = (
                (right_legs_stance > 0.5) & (left_legs_stance < 0.5) |
                (left_legs_stance > 0.5) & (right_legs_stance < 0.5)
            ).float()
            rew_lateral_alternating_stance = one_side_stance * getattr(self.cfg, 'rew_scale_lateral_alternating_stance', 3.0) * dt
            rew_lateral_alternating_stance = torch.where(is_lateral_mode, rew_lateral_alternating_stance, torch.zeros_like(rew_lateral_alternating_stance))
            total_reward += rew_lateral_alternating_stance
            
            # 懲罰兩側都著地的狀態（抖動特徵）
            both_stance = (right_legs_stance > 0.7) & (left_legs_stance > 0.7)
            rew_both_stance_penalty = -both_stance.float() * 3.0 * dt
            rew_both_stance_penalty = torch.where(is_lateral_mode, rew_both_stance_penalty, torch.zeros_like(rew_both_stance_penalty))
            total_reward += rew_both_stance_penalty
            
            # ==============================================================
            # G6.5.4.5 ★★★ 新增：側移時主驅動低速獎勵 ★★★
            # ==============================================================
            # 用戶說：「純側移應該是腿不轉，只有 ABAD 在推」
            # 獎勵主驅動保持低速（接近零）
            main_drive_speed = torch.abs(main_drive_vel).mean(dim=1)
            
            # 主驅動速度越小獎勵越高
            drive_low_speed_reward = torch.exp(-main_drive_speed * 2.0)  # speed=0 時獎勵=1
            rew_lateral_drive_lock = drive_low_speed_reward * getattr(self.cfg, 'rew_scale_lateral_drive_lock', 3.0) * dt
            rew_lateral_drive_lock = torch.where(is_lateral_mode, rew_lateral_drive_lock, torch.zeros_like(rew_lateral_drive_lock))
            total_reward += rew_lateral_drive_lock
            
            # ==============================================================
            # G6.5.4.6 ★★★ 新增：側移低頻獎勵 ★★★
            # ==============================================================
            # 用戶說：「頻率要慢，一步一步來」
            # 使用側移步態相位的變化率來判斷頻率
            # 側移相位增量越接近目標（低頻）獎勵越高
            rew_lateral_low_freq = torch.zeros(self.num_envs, device=self.device)  # 預先初始化
            if hasattr(self, '_lateral_gait_phase'):
                # 目標側移頻率是 0.3 Hz（在 config 中定義）
                target_lateral_freq = getattr(self.cfg, 'lateral_gait_frequency', 0.3)
                
                # 獎勵較慢的動作變化（ABAD 變化率低）
                if hasattr(self, 'last_actions'):
                    abad_rate = torch.sqrt(torch.sum(torch.square(self.actions[:, 6:12] - self.last_actions[:, 6:12]), dim=1))
                    # 變化率低獎勵高
                    low_freq_reward = torch.exp(-abad_rate * 3.0)
                    rew_lateral_low_freq = low_freq_reward * getattr(self.cfg, 'rew_scale_lateral_low_freq', 2.0) * dt
                    rew_lateral_low_freq = torch.where(is_lateral_mode, rew_lateral_low_freq, torch.zeros_like(rew_lateral_low_freq))
                    total_reward += rew_lateral_low_freq
            
            # ==============================================================
            # G6.5.4.7 ★★★ 新增：側移正確方向大獎勵 ★★★
            # ==============================================================
            # 用戶說側移效果很差，大幅獎勵實際產生正確的側移速度
            lateral_speed_correct = torch.sign(cmd_vy) * actual_vy  # 方向正確時為正
            rew_lateral_correct_dir = torch.where(
                torch.abs(cmd_vy) > 0.05,  # 有側移命令時
                torch.clamp(lateral_speed_correct * getattr(self.cfg, 'rew_scale_lateral_correct_dir', 5.0), min=-3.0, max=5.0),
                torch.zeros_like(lateral_speed_correct)
            ) * dt
            rew_lateral_correct_dir = torch.where(is_lateral_mode, rew_lateral_correct_dir, torch.zeros_like(rew_lateral_correct_dir))
            total_reward += rew_lateral_correct_dir
            
            # ==============================================================
            # G6.5.5 動作平滑獎勵（代替嚴格的抖動懲罰）
            # ==============================================================
            # 使用更溫和的方式：獎勵平滑動作而非嚴厲懲罰抖動
            if hasattr(self, 'last_actions'):
                abad_action_current = self.actions[:, 6:12]
                abad_action_last = self.last_actions[:, 6:12]
                abad_action_rate = torch.sum(torch.square(abad_action_current - abad_action_last), dim=1)
                
                # 動作越平滑獎勵越高
                smoothness = torch.exp(-abad_action_rate * 5.0)  # [0, 1]
                rew_abad_smooth = smoothness * 0.5 * dt
                rew_abad_smooth = torch.where(is_lateral_mode, rew_abad_smooth, torch.zeros_like(rew_abad_smooth))
                total_reward += rew_abad_smooth
                
                # 只對極端抖動給予懲罰
                extreme_jitter = (abad_action_rate > 0.2) & (abad_amplitude < 0.1)
                rew_abad_jitter = -extreme_jitter.float() * getattr(self.cfg, 'rew_scale_abad_jitter', -5.0) * dt
                rew_abad_jitter = torch.where(is_lateral_mode, rew_abad_jitter, torch.zeros_like(rew_abad_jitter))
                total_reward += rew_abad_jitter
            else:
                rew_abad_jitter = torch.zeros(self.num_envs, device=self.device)
            
            # ==============================================================
            # G6.5.6 側移方向一致性獎勵
            # ==============================================================
            lateral_direction = torch.sign(cmd_vy)
            abad_direction_correct = (lateral_direction * (abad_right_mean - abad_left_mean)) > 0.05
            rew_lateral_direction = abad_direction_correct.float() * 1.0 * dt
            rew_lateral_direction = torch.where(is_lateral_mode, rew_lateral_direction, torch.zeros_like(rew_lateral_direction))
            total_reward += rew_lateral_direction
            
            # 全身同步抖動懲罰（只對極端情況）
            all_action_rate = torch.sum(torch.square(self.actions - self.last_actions), dim=1)
            main_drive_amplitude = torch.abs(main_drive_vel).mean(dim=1)
            
            # 如果所有動作變化都大且沒有有效移動 → 全身抖動
            is_sync_jitter = (all_action_rate > 0.3) & (main_drive_amplitude < 0.5) & (abad_amplitude < 0.15)
            rew_sync_jitter = is_sync_jitter.float() * getattr(self.cfg, 'rew_scale_sync_jitter', -20.0) * dt
            rew_sync_jitter = torch.where(is_lateral_mode, rew_sync_jitter, torch.zeros_like(rew_sync_jitter))
            total_reward += rew_sync_jitter
            
            # ==============================================================
            # G6.5.6 側移方向一致性獎勵
            # ==============================================================
            # 確保 ABAD 的外展方向與側移方向一致
            # 向左側移（cmd_vy > 0）：左側應該內收（推動身體向左）
            # 向右側移（cmd_vy < 0）：右側應該內收（推動身體向右）
            lateral_direction = torch.sign(cmd_vy)
            
            # 正確的 ABAD 配置：
            # cmd_vy > 0（向左）→ 左側內收（abad_left < 0），右側外展（abad_right > 0）→ 推身體向左
            # cmd_vy < 0（向右）→ 左側外展（abad_left > 0），右側內收（abad_right < 0）→ 推身體向右
            abad_direction_correct = (lateral_direction * (abad_right_mean - abad_left_mean)) > 0.1
            rew_lateral_direction = abad_direction_correct.float() * 1.5 * dt
            rew_lateral_direction = torch.where(is_lateral_mode, rew_lateral_direction, torch.zeros_like(rew_lateral_direction))
            total_reward += rew_lateral_direction
        else:
            # 初始化這些獎勵為零
            rew_abad_alternation = torch.zeros(self.num_envs, device=self.device)
            rew_abad_amplitude = torch.zeros(self.num_envs, device=self.device)
            rew_abad_jitter = torch.zeros(self.num_envs, device=self.device)
            rew_sync_jitter = torch.zeros(self.num_envs, device=self.device)
            rew_lateral_alternating_stance = torch.zeros(self.num_envs, device=self.device)
            # ★★★ 新增：時間基準步態獎勵初始化 ★★★
            rew_gait_phase = torch.zeros(self.num_envs, device=self.device)
            rew_abad_smooth = torch.zeros(self.num_envs, device=self.device)
            rew_both_stance_penalty = torch.zeros(self.num_envs, device=self.device)
            rew_lateral_direction = torch.zeros(self.num_envs, device=self.device)
            # ★★★ 新增：側移專用獎勵初始化 ★★★
            rew_lateral_drive_lock = torch.zeros(self.num_envs, device=self.device)
            rew_lateral_low_freq = torch.zeros(self.num_envs, device=self.device)
            rew_lateral_correct_dir = torch.zeros(self.num_envs, device=self.device)
        
        # G6.6 ABAD 動作變化率額外懲罰（對所有模式生效）
        if hasattr(self, 'last_actions'):
            abad_action_rate_all = torch.sum(torch.square(self.actions[:, 6:12] - self.last_actions[:, 6:12]), dim=1)
            rew_abad_action_rate = abad_action_rate_all * getattr(self.cfg, 'rew_scale_abad_action_rate', -0.1) * dt
            total_reward += rew_abad_action_rate
        else:
            rew_abad_action_rate = torch.zeros(self.num_envs, device=self.device)

        # ========================================================
        # 存活獎勵
        # ========================================================
        rew_alive = torch.ones(self.num_envs, device=self.device) * self.cfg.rew_scale_alive * dt
        if self._is_reward_enabled("alive"):  # 簡化模式下保留
            total_reward += rew_alive

        # =================================================================
        # ★★★ G9: 移動獎勵（防止消極的關鍵！）★★★
        # =================================================================
        # 參考 ANYmal/Cassie：用「移動獎勵」代替「靜止懲罰」
        # 核心原則：獎勵好行為 > 懲罰壞行為
        
        # 計算命令要求的速度
        cmd_speed = torch.sqrt(cmd_vx**2 + cmd_vy**2 + 0.1 * cmd_wz**2)
        # 計算機器人實際移動速度
        actual_speed = torch.sqrt(actual_vx**2 + actual_vy**2)
        
        # G9.1 靜止懲罰（大幅降低！）
        # 只懲罰極端不動的情況
        not_moving = (cmd_speed > 0.15) & (actual_speed < 0.02)
        rew_stationary_penalty = not_moving.float() * (-1.0) * dt  # ★ 從 -3.0 降低到 -1.0
        total_reward += rew_stationary_penalty
        
        # G9.2 ★★★ 腿轉動獎勵（核心防消極獎勵！）★★★
        # 只要腿在轉就給獎勵，不管方向對不對
        # 這會讓機器人願意嘗試移動
        actual_leg_vel = torch.abs(main_drive_vel * self._direction_multiplier).mean(dim=1)
        
        # 線性獎勵：腳轉越快獎勵越高（有上限）
        rew_leg_moving = torch.clamp(actual_leg_vel * getattr(self.cfg, 'rew_scale_leg_moving', 2.0), max=3.0) * dt
        if self._is_reward_enabled("leg_moving"):  # 簡化模式下保留
            total_reward += rew_leg_moving
        
        # G9.3 方向正確的移動額外獎勵
        # 如果實際速度方向與命令一致，給更大獎勵
        velocity_direction_correct = (
            (cmd_vx * actual_vx > 0) |  # 前後方向正確
            (cmd_vy * actual_vy > 0)    # 左右方向正確
        ).float()
        rew_direction_bonus = velocity_direction_correct * actual_speed * 3.0 * dt
        total_reward += rew_direction_bonus
        
        # G9.4 旋轉時的移動獎勵
        # 旋轉時只要在轉就給獎勵
        rotation_moving_reward = torch.where(
            torch.abs(cmd_wz) > 0.1,
            torch.clamp(torch.abs(actual_wz) * 2.0, max=2.0),
            torch.zeros_like(actual_wz)
        ) * dt
        total_reward += rotation_moving_reward

        # ========================================================
        # NaN 保護
        # ========================================================
        total_reward = torch.nan_to_num(total_reward, nan=0.0, posinf=10.0, neginf=-10.0)

        # ========================================================
        # 更新 TensorBoard 記錄（兼容舊格式）
        # ========================================================
        # 計算兼容舊版的變量
        vel_error_2d = torch.sqrt(lin_vel_error)
        mean_vel = torch.abs(main_drive_vel * self._direction_multiplier).mean(dim=1)
        num_active_legs = (torch.abs(main_drive_vel) > 0.3).float().sum(dim=1)
        min_vel = torch.abs(main_drive_vel).min(dim=1).values
        tilt = body_tilt  # 使用新的傾斜計算
        
        # 兼容舊版獎勵名稱
        rew_forward_vel = actual_vx * torch.sign(cmd_vx) * 3.0 * dt
        rew_vel_tracking = lin_vel_error_mapped * 2.0 * dt
        rew_gait_sync = rew_alternation  # 使用新的交替步態獎勵
        rew_rotation_dir = rew_track_ang_vel
        rew_all_legs = num_active_legs * 0.2 * dt
        rew_correct_dir = lateral_tracking
        rew_mean_vel = mean_vel * 0.2 * dt
        rew_min_vel = min_vel * 0.3 * dt
        rew_continuous_support = (coherence_a + coherence_b > 1.0).float() * 0.15 * dt
        rew_smooth_rotation = torch.zeros(self.num_envs, device=self.device)
        rew_orientation = rew_upright
        rew_lin_vel_z = rew_z_vel
        rew_abad_action = rew_abad_smart
        rew_abad_stability = rew_abad_waste
        
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
        
        # ★★★ 新增：RHex 步態獎勵記錄 ★★★
        self.episode_sums["rew_tripod_support"] += rew_tripod_support
        self.episode_sums["rew_airborne_penalty"] += rew_airborne_penalty
        self.episode_sums["rew_double_support"] += rew_double_support
        self.episode_sums["rew_velocity_match"] += rew_velocity
        self.episode_sums["rew_alternation"] += rew_alternation
        self.episode_sums["rew_frequency"] += rew_frequency
        
        # ★★★ 新增：側移專用獎勵記錄 ★★★
        self.episode_sums["rew_abad_alternation"] += rew_abad_alternation
        self.episode_sums["rew_abad_amplitude"] += rew_abad_amplitude
        self.episode_sums["rew_abad_jitter"] += rew_abad_jitter
        self.episode_sums["rew_sync_jitter"] += rew_sync_jitter
        self.episode_sums["rew_abad_action_rate"] += rew_abad_action_rate
        self.episode_sums["rew_lateral_alternating_stance"] += rew_lateral_alternating_stance
        
        # ★★★ 新增：時間基準步態獎勵記錄 ★★★
        self.episode_sums["rew_gait_phase"] += rew_gait_phase
        self.episode_sums["rew_abad_smooth"] += rew_abad_smooth
        self.episode_sums["rew_both_stance_penalty"] += rew_both_stance_penalty
        self.episode_sums["rew_lateral_direction"] += rew_lateral_direction
        
        # ★★★ 新增：直走專用獎勵記錄 ★★★
        self.episode_sums["rew_high_stance"] += rew_high_stance
        self.episode_sums["rew_abad_zero_forward"] += rew_abad_zero_forward
        self.episode_sums["rew_abad_diagonal"] += rew_abad_diagonal
        
        # ★★★ 新增：側移專用獎勵記錄 ★★★
        # 使用 locals() 檢查變數是否存在
        if 'rew_lateral_drive_lock' not in dir():
            rew_lateral_drive_lock = torch.zeros(self.num_envs, device=self.device)
        if 'rew_lateral_low_freq' not in dir():
            rew_lateral_low_freq = torch.zeros(self.num_envs, device=self.device)
        if 'rew_lateral_correct_dir' not in dir():
            rew_lateral_correct_dir = torch.zeros(self.num_envs, device=self.device)
        self.episode_sums["rew_lateral_drive_lock"] += rew_lateral_drive_lock
        self.episode_sums["rew_lateral_low_freq"] += rew_lateral_low_freq
        self.episode_sums["rew_lateral_correct_dir"] += rew_lateral_correct_dir
        
        # ★★★ 新增：旋轉專用獎勵記錄 ★★★
        self.episode_sums["rew_rotation_slow_penalty"] += rew_rotation_slow_penalty
        self.episode_sums["rew_rotation_abad_assist"] += rew_rotation_abad_assist
        self.episode_sums["rew_rotation_correct"] += rew_rotation_correct
        
        # ★★★ 新增：移動獎勵記錄（防消極）★★★
        self.episode_sums["rew_leg_moving"] += rew_leg_moving
        self.episode_sums["rew_direction_bonus"] += rew_direction_bonus
        
        # 診斷
        self.episode_sums["diag_forward_vel"] += actual_vx
        self.episode_sums["diag_lateral_vel"] += actual_vy
        self.episode_sums["diag_cmd_vx"] += cmd_vx
        self.episode_sums["diag_cmd_vy"] += cmd_vy
        self.episode_sums["diag_vel_error"] += vel_error_2d
        self.episode_sums["diag_base_height"] += base_height
        self.episode_sums["diag_tilt"] += tilt
        self.episode_sums["diag_drive_vel_mean"] += mean_vel
        self.episode_sums["diag_rotating_legs"] += num_active_legs
        self.episode_sums["diag_min_leg_vel"] += min_vel
        self.episode_sums["diag_abad_magnitude"] += abad_magnitude
        self.episode_sums["diag_cmd_wz"] += cmd_wz
        self.episode_sums["diag_actual_wz"] += actual_wz
        self.episode_sums["diag_wz_error"] += torch.abs(actual_wz - cmd_wz)
        
        # 腿速度診斷
        target_leg_vel_abs = torch.abs(self._target_drive_vel).mean(dim=1)
        leg_vel_error = torch.abs(torch.abs(main_drive_vel) - torch.abs(self._target_drive_vel)).mean(dim=1)
        
        self.episode_sums["diag_target_leg_vel"] += target_leg_vel_abs
        self.episode_sums["diag_leg_vel_error"] += leg_vel_error
        
        # ★★★ 新增：RHex 步態診斷 ★★★
        self.episode_sums["diag_stance_count_a"] += stance_count_a
        self.episode_sums["diag_stance_count_b"] += stance_count_b
        self.episode_sums["diag_phase_diff"] += phase_diff
        self.episode_sums["diag_mean_velocity"] += mean_abs_vel
        self.episode_sums["diag_airborne_count"] += both_airborne
        
        # 計算著地/擺動組的平均速度
        stance_mask = leg_in_stance.float()
        swing_mask = (~leg_in_stance).float()
        actual_abs_vel = torch.abs(main_drive_vel * self._direction_multiplier)
        
        stance_vel_sum = (actual_abs_vel * stance_mask).sum(dim=1)
        stance_count = stance_mask.sum(dim=1).clamp(min=1)  # 避免除以0
        swing_vel_sum = (actual_abs_vel * swing_mask).sum(dim=1)
        swing_count = swing_mask.sum(dim=1).clamp(min=1)
        
        self.episode_sums["diag_stance_velocity"] += stance_vel_sum / stance_count
        self.episode_sums["diag_swing_velocity"] += swing_vel_sum / swing_count
        
        self.last_main_drive_vel = main_drive_vel.clone()

        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        =================================================================
        【判斷是否結束】
        =================================================================
        
        強化學習中的「Episode」（回合）什麼時候結束？
        
        1. 超時：玩太久了（例如 30 秒），強制結束
        2. 終止：發生嚴重問題（摔倒、飛走、物理爆炸），提前結束
        
        【為什麼要區分這兩種？】
        • 超時 = 正常結束，不代表失敗
        • 終止 = 失敗！AI 要學會避免這種情況
        
        返回：
            terminated: 哪些環境因為「失敗」而結束
            time_out: 哪些環境因為「超時」而結束
        """
        # 【超時檢查】已經玩了最大時間步數了嗎？
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # 初始化終止標記（全部設為 False，等等再根據條件設為 True）
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # 讀取機器人當前狀態
        root_pos = self.robot.data.root_pos_w    # 機身位置
        root_vel = self.robot.data.root_lin_vel_w # 機身速度
        
        # =================================================================
        # 終止條件 1：物理模擬爆炸（出現 NaN 或 Inf）
        # =================================================================
        # NaN = Not a Number（計算錯誤的結果）
        # Inf = Infinity（無限大）
        # 這些都是模擬器出問題的徵兆，必須立即終止
        pos_invalid = torch.any(torch.isnan(root_pos) | torch.isinf(root_pos), dim=1)
        vel_invalid = torch.any(torch.isnan(root_vel) | torch.isinf(root_vel), dim=1)
        
        # =================================================================
        # 終止條件 2：速度過快（物理失控）
        # =================================================================
        # 如果速度超過 30 m/s，肯定是出問題了（正常機器人不可能這麼快）
        vel_too_fast = torch.any(torch.abs(root_vel) > 30.0, dim=1)

        # =================================================================
        # 終止條件 3：翻車（傾斜太大）
        # =================================================================
        # 使用之前計算的 body_tilt 來判斷
        # ★★★ 放寬到 1.5（約 115 度），配合 config 中的 max_tilt_magnitude ★★★
        if hasattr(self, '_body_tilt'):
            max_tilt = getattr(self.cfg, 'max_tilt_magnitude', 2.0)
            flipped_over = self._body_tilt > max_tilt
        else:
            # 回退方案：第一次調用時可能還沒有 _body_tilt
            gravity_alignment = torch.sum(
                self.projected_gravity * self.reference_projected_gravity, dim=1
            )
            flipped_over = gravity_alignment < -0.5  # 放寬

        # =================================================================
        # 終止條件 4：高度異常
        # =================================================================
        base_height = root_pos[:, 2]   # 機身高度（Z 座標）
        min_height = getattr(self.cfg, 'min_base_height', -0.05)
        too_low = base_height < min_height  # 使用 config 中的值
        too_high = base_height > 2.0   # 高於 2 公尺（飛上天了？）
        
        # =================================================================
        # 終止條件 5：身體觸地（摔倒）
        # =================================================================
        # 這是最重要的終止條件！
        # 機器人摔倒了就應該結束，不然它會學會「躺著不動」的偷懶策略
        body_contact_terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if self.cfg.terminate_on_body_contact and hasattr(self, '_body_contact'):
            body_contact_terminated = self._body_contact
        
        # 每隔一段時間打印一次終止原因統計
        if not hasattr(self, '_term_debug_counter'):
            self._term_debug_counter = 0
        self._term_debug_counter += 1
        if self._term_debug_counter % 1000 == 1:
            print(f"[Term Debug] pos_invalid: {pos_invalid.sum().item()}, "
                  f"vel_invalid: {vel_invalid.sum().item()}, "
                  f"vel_too_fast: {vel_too_fast.sum().item()}, "
                  f"flipped: {flipped_over.sum().item()}, "
                  f"too_low: {too_low.sum().item()}, "
                  f"too_high: {too_high.sum().item()}, "
                  f"body_contact: {body_contact_terminated.sum().item()}, "
                  f"base_h_mean: {base_height.mean().item():.3f}")
        
        terminated = pos_invalid | vel_invalid | vel_too_fast | flipped_over | too_low | too_high | body_contact_terminated

        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        """
        【重置環境】
        
        當一個環境結束（超時或終止）後，需要「重置」它：
        • 把機器人放回起點
        • 清除所有狀態
        • 給一個新的速度命令
        
        這樣這個環境就可以開始新的一輪訓練了！
        
        參數：
            env_ids: 需要重置的環境編號列表
        """
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES  # 如果沒指定，就重置全部
        super()._reset_idx(env_ids)  # 呼叫父類別的重置方法

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

        # ★★★ 關鍵修正：隨機化機器人的初始朝向（yaw）★★★
        # 這樣訓練時機器人會面向各種方向，迫使它學習「本體座標系」的追蹤
        # 而不是「世界座標系」的追蹤
        random_yaw = sample_uniform(-math.pi, math.pi, (num_reset,), device=self.device)
        
        # 創建 yaw 旋轉的四元數 [w, x, y, z] = [cos(yaw/2), 0, 0, sin(yaw/2)]
        half_yaw = random_yaw * 0.5
        yaw_w = torch.cos(half_yaw)
        yaw_z = torch.sin(half_yaw)
        
        # ★★★ 重要：將 yaw 旋轉疊加到原本的初始旋轉上 ★★★
        # 原本的初始旋轉是繞 X 軸 90 度：(w0, x0, y0, z0) = (0.7071068, 0.7071068, 0, 0)
        # yaw 旋轉四元數：(yaw_w, 0, 0, yaw_z)
        # 
        # 四元數乘法：q_yaw * q_init
        # w = yaw_w*w0 - yaw_z*0 = yaw_w * w0
        # x = yaw_w*x0 + yaw_z*0 = yaw_w * x0
        # y = yaw_z*x0           (注意：正號！)
        # z = yaw_z*w0
        w0, x0 = 0.7071068, 0.7071068  # 原本的 X 軸 90 度旋轉
        default_root_state[:, 3] = w0 * yaw_w              # w
        default_root_state[:, 4] = x0 * yaw_w              # x
        default_root_state[:, 5] = x0 * yaw_z              # y (修正：正號)
        default_root_state[:, 6] = w0 * yaw_z              # z

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
        # ★★★ 外部控制時不重新採樣命令，保持用戶設置的命令 ★★★
        if not self.external_control:
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

    # ===================================================================
    # 【調試可視化功能】
    # ===================================================================
    # 這個功能會在模擬畫面上畫出箭頭，幫助我們「看見」機器人的狀態：
    # • 綠色箭頭 = 目標速度（你要它往哪走）
    # • 紅色箭頭 = 實際速度（它實際往哪走）
    # 
    # 如果兩個箭頭方向一致 = 追蹤得很好！
    # 如果兩個箭頭方向不同 = 還在學習中...
    
    def _set_debug_vis_impl(self, debug_vis: bool):
        """
        【開啟或關閉調試可視化】
        
        這是 Isaac Lab 的官方介面，用來控制畫面上的調試標記。
        
        參數：
            debug_vis: True = 顯示箭頭，False = 隱藏箭頭
        """
        if debug_vis:
            # 第一次創建 markers
            if not hasattr(self, "goal_vel_visualizer"):
                # 目標速度箭頭（綠色）- 細長箭頭
                goal_marker_cfg = GREEN_ARROW_X_MARKER_CFG.copy()
                goal_marker_cfg.prim_path = "/Visuals/Command/velocity_goal"
                goal_marker_cfg.markers["arrow"].scale = (0.8, 0.25, 0.25)  # 長=0.8, 寬高=0.25
                self.goal_vel_visualizer = VisualizationMarkers(goal_marker_cfg)
                
                # 實際速度箭頭（紅色）- 細長箭頭
                current_marker_cfg = RED_ARROW_X_MARKER_CFG.copy()
                current_marker_cfg.prim_path = "/Visuals/Command/velocity_current"
                current_marker_cfg.markers["arrow"].scale = (0.8, 0.2, 0.2)  # 稍小以區分
                self.current_vel_visualizer = VisualizationMarkers(current_marker_cfg)
                
                print("[可視化] Debug visualization markers 創建成功")
                print("   綠色箭頭 = 目標速度方向")
                print("   紅色箭頭 = 實際速度方向")
            
            # 設置可見
            self.goal_vel_visualizer.set_visibility(True)
            self.current_vel_visualizer.set_visibility(True)
        else:
            # 隱藏 markers
            if hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer.set_visibility(False)
                self.current_vel_visualizer.set_visibility(False)
    
    def _debug_vis_callback(self, event):
        """
        【更新調試箭頭】每一幀都會被呼叫
        
        這個函數會：
        1. 讀取機器人當前位置
        2. 計算目標速度和實際速度的方向
        3. 更新箭頭的位置和旋轉
        
        這樣箭頭就會跟著機器人移動，並且指向正確的方向！
        
        ★★★ 原地旋轉 vs 側移 的視覺區分 ★★★
        
        【側移命令】(vx≈0, vy≠0, wz=0)
        - 箭頭「固定」指向左或右
        - 靜止不動的箭頭 = 線性移動
        
        【旋轉命令】(vx=0, vy=0, wz≠0)  
        - 箭頭會「持續繞圈旋轉」！
        - 逆時針命令 (wz>0)：箭頭逆時針轉
        - 順時針命令 (wz<0)：箭頭順時針轉
        - 旋轉的箭頭 = 旋轉命令！
        """
        # 檢查機器人是否已初始化
        if not self.robot.is_initialized:
            return
        
        # 獲取機器人位置（箭頭起點在機器人上方）
        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5  # 箭頭高度
        
        # 計算目標速度箭頭的縮放和旋轉
        # 傳入完整命令 (vx, vy, wz) 以便處理旋轉視覺化
        goal_arrow_scale, goal_arrow_quat = self._resolve_xy_velocity_to_arrow(
            self.commands[:, :2], is_goal=True, ang_vel=self.commands[:, 2]
        )
        
        # 計算實際速度箭頭的縮放和旋轉
        current_arrow_scale, current_arrow_quat = self._resolve_xy_velocity_to_arrow(
            self.base_lin_vel[:, :2], is_goal=False, ang_vel=self.base_ang_vel[:, 2]
        )
        
        # 更新可視化 markers
        self.goal_vel_visualizer.visualize(base_pos_w, goal_arrow_quat, goal_arrow_scale)
        
        # 實際速度箭頭稍微高一點，避免重疊
        base_pos_w_current = base_pos_w.clone()
        base_pos_w_current[:, 2] += 0.1
        self.current_vel_visualizer.visualize(base_pos_w_current, current_arrow_quat, current_arrow_scale)
    
    def _resolve_xy_velocity_to_arrow(self, xy_velocity: torch.Tensor, is_goal: bool = True, ang_vel: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        【把速度向量轉換成箭頭的外觀】
        
        這個函數計算箭頭應該「多長」和「指向哪裡」。
        
        參數：
            xy_velocity: XY 方向的速度向量 [環境數, 2]
            is_goal: 是不是目標速度的箭頭？
                    （True = 綠色目標箭頭，False = 紅色實際箭頭）
            ang_vel: 角速度 wz [環境數]（可選，用於旋轉可視化）
        
        返回：
            arrow_scale: 箭頭的大小 [環境數, 3]（長、寬、高）
            arrow_quat: 箭頭的旋轉（四元數格式）[環境數, 4]
        
        ★★★ 視覺區分：側移 vs 旋轉 ★★★
        
        【側移】箭頭固定指向移動方向
        ┌─────────────────────────────┐
        │    ← ← ← 🤖                 │  向左側移：箭頭靜止指左
        │            → → →           │  向右側移：箭頭靜止指右
        └─────────────────────────────┘
        
        【旋轉】箭頭持續繞圈轉動！
        ┌─────────────────────────────┐
        │      ↖ ↑ ↗                  │  
        │    ←  🤖  →   逆時針：      │  箭頭逆時針繞圈
        │      ↙ ↓ ↘                  │  
        └─────────────────────────────┘
        
        這樣一眼就能看出：
        - 箭頭不動 = 線性移動命令
        - 箭頭繞圈 = 旋轉命令！
        """
        # 基礎縮放：只改變長度，寬高固定
        if is_goal:
            base_length = 0.8   # 綠色目標箭頭基礎長度
            width_height = 0.25  # 固定寬高
        else:
            base_length = 0.8   # 紅色實際箭頭基礎長度
            width_height = 0.2  # 固定寬高（稍小）
        
        # 計算 XY 速度大小
        speed = torch.linalg.norm(xy_velocity, dim=1)
        
        # 判斷是否為「純旋轉」命令（XY 速度很小，但有角速度）
        is_pure_rotation = (speed < 0.05)  # XY 速度閾值
        
        # 處理角速度可視化
        if ang_vel is not None:
            # 對於純旋轉命令，用 |wz| 來決定箭頭長度
            rotation_speed = torch.abs(ang_vel)
            # 純旋轉時使用角速度決定長度，否則使用線速度
            effective_speed = torch.where(is_pure_rotation, rotation_speed * 0.5, speed)
        else:
            effective_speed = speed
        
        # 箭頭長度根據速度調整：最小 0.3 倍，速度加成 2.0x
        length_scale = base_length * (0.3 + effective_speed * 2.0)
        
        # 創建 scale tensor: [length, width, height]
        arrow_scale = torch.zeros(xy_velocity.shape[0], 3, device=self.device)
        arrow_scale[:, 0] = length_scale  # 長度隨速度變化
        arrow_scale[:, 1] = width_height  # 寬度固定
        arrow_scale[:, 2] = width_height  # 高度固定
        
        # =====================================================================
        # 箭頭方向計算 - 只使用 YAW 角度，忽略 pitch/roll
        # =====================================================================
        # 
        # ★★★ 重要修正 ★★★
        # 之前的方法用 quat_apply 會把 pitch/roll 也應用上去，
        # 導致當機器人晃動時，箭頭方向會瘋狂抖動甚至反向！
        # 
        # 正確做法：只提取 yaw 角度，用純 yaw 旋轉來轉換速度向量
        # 這樣箭頭才會穩定地指向機器人的本體坐標系方向
        #
        
        # 獲取機器人的姿態四元數
        base_quat_w = self.robot.data.root_quat_w
        
        # ★★★ 只提取 YAW 角度（忽略 pitch 和 roll）★★★
        # Isaac Lab 四元數格式: [w, x, y, z]
        w = base_quat_w[:, 0]
        x = base_quat_w[:, 1]
        y = base_quat_w[:, 2]
        z = base_quat_w[:, 3]
        base_yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        
        # 計算本體座標系速度的方向角度（相對於機器人前方）
        # atan2(vy, vx) 給出本體座標系中的速度方向
        body_heading = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        
        # 世界座標系的方向 = 機器人 yaw + 本體速度方向
        linear_world_angle = base_yaw + body_heading
        
        # 處理純旋轉的特殊情況
        if ang_vel is not None:
            # 使用模擬時間讓箭頭持續旋轉
            sim_time = self.episode_length_buf.float() * self.cfg.sim.dt * self.cfg.decimation
            
            # 純旋轉時的角度：箭頭以 wz 速度持續旋轉（相對於機器人朝向）
            rotation_angle = ang_vel * sim_time * 2.0  # 乘以 2 讓旋轉更明顯
            
            # 純旋轉時的世界座標系角度（繞圈動畫）
            rotation_world_angle = base_yaw + rotation_angle
            
            # 根據是否純旋轉選擇角度
            world_heading = torch.where(is_pure_rotation, rotation_world_angle, linear_world_angle)
        else:
            world_heading = linear_world_angle
        
        # 創建只有 yaw 旋轉的四元數（箭頭永遠水平）
        zeros = torch.zeros_like(world_heading)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, world_heading)
        
        return arrow_scale, arrow_quat
