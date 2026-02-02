# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""
Configuration for RedRhex hexapod robot environment with RHex-style wheg locomotion.

RHex 機器人核心概念：
- 每隻腳的主驅動關節持續旋轉（像輪子），而非傳統步行
- 使用交替三足步態（alternating tripod gait）
- 半圓形腿與地面的接觸產生前進位移

關節分組：
- 主驅動關節（連續旋轉）: 15, 12, 18, 23, 24, 7
  - Tripod A: 15, 18, 24
  - Tripod B: 12, 23, 7
- ABAD 關節（外展/內收，RL 探索）: 14, 11, 17, 22, 21, 6
- 避震關節（被動/高剛性）: 5, 13, 25, 26, 27, 8
"""

from __future__ import annotations

import math
import os
import isaaclab.sim as sim_utils

# Import terrain configuration
from .terrain_cfg import TerrainCfg, TerrainType

# Get the USD path - config file is at source/RedRhex/RedRhex/tasks/direct/redrhex/
# USD file is at project root: /home/jasonliao/RedRhex/RedRhex/RedRhex.usd
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REDRHEX_USD_PATH = os.path.normpath(os.path.join(
    _THIS_DIR, "..", "..", "..", "..", "..", "..", "RedRhex.usd"
))
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from isaaclab.sensors import ContactSensorCfg

##
# RedRhex Robot Configuration
##

REDRHEX_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=_REDRHEX_USD_PATH,
        activate_contact_sensors=True,
        mass_props=sim_utils.MassPropertiesCfg(
            # 整機目標質量約 13.5-15 kg：機身 12 kg + 6 腿各 ~0.35 kg
            # 設定較高密度以增加法向力，抑制無效彈跳
            density=2500.0,  # kg/m³ - 提高密度使機身約 12 kg
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.2,       # 降低阻尼讓機器人可以移動
            angular_damping=0.3,      # 適中的角阻尼
            max_linear_velocity=10.0,  # 提高最大線速度
            max_angular_velocity=20.0,  # 提高最大角速度以允許腿旋轉
            max_depenetration_velocity=1.0,  # 適度的穿透恢復速度
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,  # 暫時關閉自碰撞，減少不穩定
            solver_position_iteration_count=16,  # 增加迭代次數
            solver_velocity_iteration_count=8,
            fix_root_link=False,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),  # 提高初始高度以避免穿透地面
        # 四元數 (w, x, y, z) rotate 90 deg around x axis
        rot=(0.7071068, 0.7071068, 0.0, 0.0),
        # 明確設置所有關節位置，匹配 USD 文件中的默認值
        # USD 顯示的是度數，這裡轉換為弧度
        joint_pos={
            # ===== 主驅動關節 (Main Drive) =====
            # 右側: 45°, 左側: -45°
            "Revolute_15": 45.0 * math.pi / 180,   # 右前 - 45°
            "Revolute_12": 45.0 * math.pi / 180,   # 右後 - 45°
            "Revolute_7": 45.0 * math.pi / 180,    # 右中 - 45°
            "Revolute_18": -45.0 * math.pi / 180,  # 左前 - -45°
            "Revolute_23": -45.0 * math.pi / 180,  # 左中 - -45°
            "Revolute_24": -45.0 * math.pi / 180,  # 左後 - -45°
            # ===== ABAD 關節 - 全部 0° =====
            "Revolute_14": 0.0,
            "Revolute_6": 0.0,
            "Revolute_11": 0.0,
            "Revolute_17": 0.0,
            "Revolute_22": 0.0,
            "Revolute_21": 0.0,
            # ===== 避震關節 (Damper) =====
            "Revolute_5": 45.0 * math.pi / 180,    # 45°
            "Revolute_13": -45.0 * math.pi / 180,  # -45°
            "Revolute_8": 45.0 * math.pi / 180,    # 45°
            "Revolute_25": 45.0 * math.pi / 180,   # 45°
            "Revolute_26": 45.0 * math.pi / 180,   # 45°
            "Revolute_27": 45.0 * math.pi / 180,   # 45°
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        # 主驅動關節 - 速度控制，允許連續旋轉
        # RHex 腿需要足夠扭矩來驅動 ~12kg 機身
        # ★★★ 重要：damping 決定速度控制的驅動力！★★★
        # 力矩 = damping * (target_vel - current_vel)
        # damping 太低會導致腿轉不動
        "main_drive": ImplicitActuatorCfg(
            joint_names_expr=[
                "Revolute_15", "Revolute_12", "Revolute_18",
                "Revolute_23", "Revolute_24", "Revolute_7"
            ],
            effort_limit=100.0,      # 再次提高力矩限制！(was 50.0)
            velocity_limit=30.0,     # 提高速度限制
            stiffness=0.0,           # 純速度控制，無位置剛性
            damping=50.0,            # ★★★ 大幅提高阻尼！(was 10.0) ★★★
        ),
        # ABAD 關節 - 位置控制，小範圍調節
        "abad": ImplicitActuatorCfg(
            joint_names_expr=[
                "Revolute_14", "Revolute_11", "Revolute_17",
                "Revolute_22", "Revolute_21", "Revolute_6"
            ],
            effort_limit=8.0,        # 提高力矩以有效調節姿態
            velocity_limit=5.0,
            stiffness=40.0,          # 較高剛性維持位置
            damping=4.0,
        ),
        # 避震關節 - 固定形狀，不在 action space 中
        # 這些關節不能被驅動，使用高剛性+高阻尼讓它們保持初始角度
        # 目標：盡可能保持形狀不變
        "damper": ImplicitActuatorCfg(
            joint_names_expr=[
                "Revolute_5", "Revolute_13", "Revolute_25",
                "Revolute_26", "Revolute_27", "Revolute_8"
            ],
            effort_limit=50.0,       # 高力矩限制以維持位置
            velocity_limit=1.0,      # 低速度限制防止快速移動
            stiffness=200.0,         # 很高剛性 - 保持初始角度
            damping=20.0,            # 高阻尼 - 抑制任何振動
        ),
    },
)


@configclass
class RedrhexEnvCfg(DirectRLEnvCfg):
    """
    Configuration for the RedRhex hexapod RHex-style locomotion environment.
    
    控制架構：
    - 主驅動關節：速度控制（RL 輸出目標角速度）
    - ABAD 關節：位置控制（RL 輸出目標位置偏移）
    - 避震關節：被動（不在 action space 中）
    
    質量估計（UPE 材質 ~940 kg/m³）：
    - 機身 (base_link): ~12 kg
    - 每隻腿 (C-leg): ~0.35 kg
    - 整機總質量: ~14 kg
    """

    # ===================
    # Environment Settings
    # ===================
    # 120 Hz 模擬，decimation=2 → 60 Hz 控制頻率
    decimation = 2
    episode_length_s = 30.0  # 增加 episode 長度讓機器人有更多學習時間

    # ===================
    # Action Space: 6 main drive velocities + 6 ABAD positions = 12
    # ===================
    action_space = 12
    
    # ===================
    # Observation Space:
    # - base_lin_vel (3)
    # - base_ang_vel (3) 
    # - projected_gravity (3)
    # - main_drive_pos_sin (6) - 用 sin 表示循環相位
    # - main_drive_pos_cos (6) - 用 cos 表示循環相位
    # - main_drive_vel (6)
    # - abad_pos (6)
    # - abad_vel (6)
    # - velocity_command (3)
    # - gait_phase (2) - sin/cos
    # - last_actions (12)
    # Total: 3+3+3+6+6+6+6+6+3+2+12 = 56
    # ===================
    observation_space = 56
    state_space = 0

    # ===================
    # Simulation Settings
    # ===================
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,  # 120 Hz 模擬頻率（較低以獲得穩定性）
        render_interval=2,
        gravity=(0.0, 0.0, -9.81),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.2,   # 提高摩擦力幫助 C-leg 抓地
            dynamic_friction=1.0,  # 提高動摩擦
            restitution=0.0,       # 無彈跳
        ),
    )

    # ===================
    # Robot Configuration
    # ===================
    robot_cfg: ArticulationCfg = REDRHEX_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # ===================
    # Contact Sensor Configuration (關鍵！用於檢測身體觸地)
    # ===================
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=3,
        update_period=0.0,  # 每個仿真步都更新
        track_air_time=True,
    )
    
    # 身體部位名稱（用於觸地檢測）- 這些部位不應觸地
    body_names = ["base_link"]  # 主機身
    # 腿部觸地是正常的，不需要懲罰
    leg_names = [".*leg.*", ".*Leg.*"]  # 腿部

    # ===================
    # Scene Configuration
    # ===================
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096,
        env_spacing=2.5,
        replicate_physics=True,
    )

    # ===================
    # Terrain Configuration
    # ===================
    # Default flat ground (Isaac Lab built-in)
    # This is used when procedural_terrain.terrain_type == FLAT
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )
    
    # ===================
    # Procedural Terrain Configuration (for curriculum learning)
    # ===================
    # IMPORTANT: Default is FLAT for backward compatibility!
    # To enable procedural terrain, set terrain_type to ROUGH, STAIRS, OBSTACLES, or MIXED
    procedural_terrain: TerrainCfg = TerrainCfg(
        terrain_type=TerrainType.OBSTACLES,  # 🎮 Try: ROUGH, STAIRS, OBSTACLES, MIXED
        difficulty_scale=0.3,           # 0.0=easy, 1.0=hard (start moderate)
        horizontal_scale=0.5,           # Grid resolution in meters
        vertical_scale=0.15,            # Max height variation at difficulty=1.0
        friction=0.8,                   # Surface friction coefficient
        grid_size=(15.0, 15.0),         # Terrain size in meters
        spawn_height_offset=0.15,       # Extra spawn height above terrain
        obstacle_density=0.3,           # Obstacle density (0.0-1.0)
        min_obstacle_size=0.1,          # Min obstacle dimension
        max_obstacle_size=0.4,          # Max obstacle dimension
        debug_visualize=True,           # 👁️ Visualize terrain bounds
    )

    # ===================
    # Joint Names Mapping (按功能分組)
    # ===================
    # 
    # 機器人腿的物理佈局:
    #   前方 (Forward +X)
    #        ^
    #        |
    #   [Leg1] [Leg4]    (前排)
    #   [Leg2] [Leg5]    (中排)
    #   [Leg3] [Leg6]    (後排)
    #   右側    左側
    #
    # 關節對應:
    # - 主驅動 (360° 旋轉): 15, 12, 7 (右側); 18, 23, 24 (左側)
    # - ABAD (外展內收): 14, 11, 6 (右側); 17, 22, 21 (左側)
    # - 避震 (被動): 5, 13, 8 (右側); 25, 26, 27 (左側)
    #
    # Tripod 分組 (交替支撐):
    # - Tripod A: Leg1(15), Leg4(18), Leg6(24) → 前右 + 前左 + 後左
    # - Tripod B: Leg2(7), Leg3(12), Leg5(23) → 中右 + 後右 + 中左
    
    # 主驅動關節 (連續旋轉) - 順序: [右前, 右中, 右後, 左前, 左中, 左後]
    # 索引:                        [  0,    1,    2,    3,    4,    5  ]
    main_drive_joint_names = [
        "Revolute_15",  # idx 0 - Leg 1 (右前) - Tripod A
        "Revolute_7",   # idx 1 - Leg 2 (右中) - Tripod B
        "Revolute_12",  # idx 2 - Leg 3 (右後) - Tripod B
        "Revolute_18",  # idx 3 - Leg 4 (左前) - Tripod A
        "Revolute_23",  # idx 4 - Leg 5 (左中) - Tripod B
        "Revolute_24",  # idx 5 - Leg 6 (左後) - Tripod A
    ]
    
    # 方向乘數 (前進時的旋轉方向)
    # 右側腿 (idx 0,1,2): 負向旋轉 → -1
    # 左側腿 (idx 3,4,5): 正向旋轉 → +1
    leg_direction_multiplier = [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0]
    
    # ABAD 關節 (位置控制) - 順序對應主驅動
    abad_joint_names = [
        "Revolute_14",  # Leg 1 (右前)
        "Revolute_6",   # Leg 2 (右中)
        "Revolute_11",  # Leg 3 (右後)
        "Revolute_17",  # Leg 4 (左前)
        "Revolute_22",  # Leg 5 (左中)
        "Revolute_21",  # Leg 6 (左後)
    ]
    
    # 避震關節 (被動) - 順序對應主驅動
    damper_joint_names = [
        "Revolute_5",   # Leg 1 (右前)
        "Revolute_8",   # Leg 2 (右中)
        "Revolute_13",  # Leg 3 (右後)
        "Revolute_25",  # Leg 4 (左前)
        "Revolute_26",  # Leg 5 (左中)
        "Revolute_27",  # Leg 6 (左後)
    ]

    # ===================
    # Tripod Groups (索引到 main_drive_joint_names)
    # ===================
    # RHex 交替三足步態：
    # - Tripod A: 15(idx0), 18(idx3), 24(idx5) = 右前 + 左前 + 左後
    # - Tripod B: 7(idx1), 12(idx2), 23(idx4) = 右中 + 右後 + 左中
    # 這樣的分組確保任何時刻都有對角線支撐
    tripod_a_leg_indices = [0, 3, 5]  # joints 15, 18, 24
    tripod_b_leg_indices = [1, 2, 4]  # joints 7, 12, 23
    
    # ===================
    # Leg Side Groups (for motor direction)
    # ===================
    # 基於 main_drive_joint_names 的索引
    # Right side: idx 0, 1, 2 → joints 15, 7, 12
    # Left side: idx 3, 4, 5 → joints 18, 23, 24
    right_leg_indices = [0, 1, 2]  # 右側腿
    left_leg_indices = [3, 4, 5]   # 左側腿

    # ===================
    # Action Scaling
    # ===================
    # 主驅動：目標角速度 (rad/s)
    # 網路輸出 [-1, 1]，乘以 scale 得到目標速度
    # RHex 腿需要持續旋轉，scale 要足夠大
    main_drive_vel_scale = 8.0      # ±8 rad/s - 允許快速旋轉 (~1.3 轉/秒)
    
    # ABAD：目標位置偏移 (rad)
    abad_pos_scale = 0.3            # ±0.3 rad ≈ ±17° - 增加範圍

    # ===================
    # Velocity Command Ranges (多方向追蹤)
    # ===================
    # 9 個移動方向：前、後、左、右、右前、右後、左前、左後、原地旋轉
    # 命令會定期隨機切換方向
    
    # 線速度範圍 (m/s)
    lin_vel_x_range = [-0.3, 0.5]   # 前後速度：正值前進，負值後退
    lin_vel_y_range = [-0.3, 0.3]   # 側向速度：正值向左，負值向右
    ang_vel_z_range = [-1.0, 1.0]   # 旋轉速度 (rad/s)：正值逆時針，負值順時針
    
    # 命令切換設置
    command_resample_time = 3.0     # 每 3 秒隨機切換一次目標方向（縮短以便觀察）
    
    # 離散方向模式（如果啟用，則使用固定方向）
    use_discrete_directions = True
    # 9 個方向的速度設定 [vx, vy, wz]
    # 包含原地旋轉（vx=0, vy=0, wz≠0）
    discrete_directions = [
        [0.4, 0.0, 0.0],     # 0: 前
        [-0.3, 0.0, 0.0],    # 1: 後
        [0.0, 0.3, 0.0],     # 2: 左
        [0.0, -0.3, 0.0],    # 3: 右
        [0.3, 0.2, 0.0],     # 4: 左前
        [0.3, -0.2, 0.0],    # 5: 右前
        [-0.2, 0.2, 0.0],    # 6: 左後
        [-0.2, -0.2, 0.0],   # 7: 右後
        [0.0, 0.0, 0.8],     # 8: 原地逆時針旋轉
        [0.0, 0.0, -0.8],    # 9: 原地順時針旋轉
    ]
    
    # 方向名稱（用於調試輸出）
    direction_names = ["前", "後", "左", "右", "左前", "右前", "左後", "右後", "逆時針旋轉", "順時針旋轉"]
    
    # 可視化設置
    draw_debug_vis = True           # 是否繪製調試箭頭
    debug_vis_interval = 5          # 每隔幾步更新一次可視化（減少性能開銷）

    # ===================
    # Gait Parameters
    # ===================
    # 基礎步態頻率 - 決定腿的旋轉速度
    # RHex 腿需要足夠快的旋轉才能產生穩定推進
    base_gait_frequency = 1.0       # Hz - 每秒 1 圈
    # 對應的角速度 (更新為匹配新頻率)
    base_gait_angular_vel = 2 * math.pi * 1.0  # ≈ 6.28 rad/s
    
    # Tripod 相位差 (180°)
    tripod_phase_offset = math.pi

    # ===================
    # Reward Scales (按照用戶需求重新設計)
    # ===================
    
    # --- G1: 追蹤項（核心）---
    # r_vel = exp(-k_v * e_v^2), r_yaw = exp(-k_w * e_w^2)
    # ★★★ 進一步提高追蹤獎勵，讓機器人積極移動！★★★
    rew_scale_track_lin_vel = 8.0       # 再提高！線速度追蹤
    rew_scale_track_ang_vel = 4.0       # 再提高！角速度追蹤
    
    # --- G2: 姿態與穩定性（避免偏航亂翻、避免彈跳）---
    # ★★★ 降低姿態懲罰，避免機器人因為怕移動而靜止 ★★★
    rew_scale_upright = -2.0            # 降低！俯仰/側滾穩定
    rew_scale_z_vel = -1.0              # 降低！垂直彈跳抑制
    rew_scale_ang_vel_xy = -0.02        # 降低！xy 角速度懲罰
    rew_scale_base_height = -0.5        # 降低！高度偏離懲罰
    
    # --- G3: 身體觸地（必須強烈！）---
    rew_scale_body_contact = -50.0      # 身體觸地大負值！！
    # ★★★ 啟用身體觸地終止！★★★
    # 正常站立高度約 0.12m，當高度低於 0.01m 或傾斜超過 55° 視為觸地
    terminate_on_body_contact = True    # 身體觸地時終止 episode
    body_contact_height_threshold = 0.01  # 高度低於此值視為身體觸地 (m)
    
    # --- G4: 能耗與動作平滑 ---
    rew_scale_torque = -2.5e-5          # 力矩懲罰（anymal_c 是 -2.5e-5）
    rew_scale_action_rate = -0.01       # 動作變化率懲罰
    rew_scale_joint_acc = -2.5e-7       # 關節加速度懲罰
    
    # --- G5: 步態相位結構（打擊六腿同相）---
    rew_scale_tripod_sync = 1.0         # 組內一致獎勵
    rew_scale_tripod_antiphase = 1.0    # 組間反相獎勵
    
    # --- G6: ABAD 使用策略 ---
    # 當 |vy*| 或 |wz*| 大時，鼓勵 ABAD 使用
    # 當 vy*=0 且 wz*=0 時，抑制 ABAD 亂動
    # ★★★ 降低 ABAD 浪費懲罰，讓機器人更自由探索 ★★★
    rew_scale_abad_smart_use = 1.5      # 智能使用 ABAD 獎勵
    rew_scale_abad_waste = -0.2         # 降低！ABAD 浪費懲罰
    
    # --- 存活獎勵 ---
    # ★★★ 完全禁用存活獎勵！靠速度追蹤和運動獲得獎勵 ★★★
    rew_scale_alive = 0.0               # 禁用！不給躺平機會
    
    # --- 兼容舊變量（會被新系統覆蓋）---
    rew_scale_forward_vel = 8.0
    rew_scale_vel_tracking = 3.0
    rew_scale_gait_sync = 0.2
    rew_scale_smooth_rotation = 0.0
    rew_scale_rotation_direction = 3.0
    rew_scale_orientation = -0.05
    rew_scale_lin_vel_z = -0.02
    rew_scale_abad_action = 0.0
    rew_scale_abad_stability = 0.1
    rew_scale_drive_acc = 0.0
    rew_scale_collision = -1.0

    # ===================
    # Termination Conditions
    # ===================
    max_tilt_magnitude = 1.5            # 放寬最大傾斜量（允許更多探索）
    min_base_height = 0.01              # 降低最低高度 (m)
    max_base_height = 0.8               # 提高最高高度 (m)

    # ===================
    # Domain Randomization
    # ===================
    randomize_mass = True
    mass_range = [0.9, 1.1]

    randomize_friction = True
    friction_range = [0.5, 1.25]

    randomize_joint_friction = True
    joint_friction_range = [0.0, 0.05]

    push_robots = True
    push_interval_s = 15.0
    max_push_vel_xy = 0.5

    # ===================
    # Noise Settings
    # ===================
    add_noise = True
    noise_level = 1.0

    noise_lin_vel = 0.1
    noise_ang_vel = 0.2
    noise_gravity = 0.05
    noise_joint_pos = 0.01
    noise_joint_vel = 1.5
