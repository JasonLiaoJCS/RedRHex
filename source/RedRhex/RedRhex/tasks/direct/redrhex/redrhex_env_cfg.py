# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for RedRhex hexapod robot environment with tripod gait."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

##
# RedRhex Robot Configuration
##

REDRHEX_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/jasonliao/RedRhex/RedRhex.usd",#dd
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,      # 恢復為 0 - 讓重力正常作用！
            angular_damping=0.05,    # 輕微角阻尼防止旋轉過快
            max_linear_velocity=10.0,  # 適度限制最大速度
            max_angular_velocity=10.0, # 適度限制最大角速度
            max_depenetration_velocity=1.0,  # 正常穿透恢復速度
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,   # 保持迭代次數
            solver_velocity_iteration_count=4,
            fix_root_link=False,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.08),  # 稍微提高初始高度
        # 四元數 (w, x, y, z) rotate 90 deg around x axis
        rot=( 0.7071068, 0.7071068, 0.0, 0.0),
        joint_pos={
            # Leg 1 (Rear Right) - Tripod Group A
            "Revolute_14": 0.0,   # hip (ABAD)
            "Revolute_15": 0.0,   # knee
            "Revolute_5": 0.0,    # foot
            # Leg 2 (Rear Left) - Tripod Group B
            "Revolute_6": 0.0,
            "Revolute_7": 0.0,
            "Revolute_8": 0.0,
            # Leg 3 (Mid Right) - Tripod Group A
            "Revolute_11": 0.0,
            "Revolute_12": 0.0,
            "Revolute_13": 0.0,
            # Leg 4 (Mid Left) - Tripod Group B
            "Revolute_17": 0.0,
            "Revolute_18": 0.0,
            "Revolute_25": 0.0,
            # Leg 5 (Front Right) - Tripod Group A
            "Revolute_22": 0.0,
            "Revolute_23": 0.0,
            "Revolute_26": 0.0,
            # Leg 6 (Front Left) - Tripod Group B
            "Revolute_21": 0.0,
            "Revolute_24": 0.0,
            "Revolute_27": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=["Revolute.*"],
            effort_limit=10.0,       # 降低力矩限制防止爆炸
            velocity_limit=5.0,      # 嚴格限制速度
            stiffness=30.0,          # 適度剛度
            damping=3.0,             # 增加阻尼減少震盪
        ),
    },
)


@configclass
class RedrhexEnvCfg(DirectRLEnvCfg):
    """Configuration for the RedRhex hexapod tripod gait environment."""

    # ===================
    # Environment Settings
    # ===================
    decimation = 4  # 控制頻率 = 物理頻率 / decimation
    episode_length_s = 20.0  # 每個 episode 持續 20 秒

    # Action/Observation space
    # 18 joints (6 legs × 3 joints)
    action_space = 18
    # Observations: base state + joint state + velocity commands + phase
    # base_lin_vel(3) + base_ang_vel(3) + projected_gravity(3) +
    # joint_pos(18) + joint_vel(18) + velocity_command(3) + gait_phase(2) + last_action(18)
    observation_space = 68
    state_space = 0

    # ===================
    # Simulation Settings
    # ===================
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 250,  # 250 Hz 物理模擬 (提高穩定性)
        render_interval=5,
        gravity=(0.0, 0.0, -9.81),  # 標準重力 (不要乘以任何係數！)
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.5,     # 增加摩擦力
            dynamic_friction=1.2,    # 增加動態摩擦
            restitution=0.0,
        ),
    )

    # ===================
    # Robot Configuration
    # ===================
    robot_cfg: ArticulationCfg = REDRHEX_CFG.replace(prim_path="/World/envs/env_.*/Robot")

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
    # Joint Names Mapping
    # ===================
    # Tripod Group A: Leg 1, 3, 5 (右後、右中、右前)
    # Tripod Group B: Leg 2, 4, 6 (左後、左中、左前)

    # Hip joints (ABAD - 你們的創新點!)
    hip_joint_names = [
        "Revolute_14",  # Leg 1 hip
        "Revolute_6",   # Leg 2 hip
        "Revolute_11",  # Leg 3 hip
        "Revolute_17",  # Leg 4 hip
        "Revolute_22",  # Leg 5 hip
        "Revolute_21",  # Leg 6 hip
    ]

    # Knee joints
    knee_joint_names = [
        "Revolute_15",  # Leg 1 knee
        "Revolute_7",   # Leg 2 knee
        "Revolute_12",  # Leg 3 knee
        "Revolute_18",  # Leg 4 knee
        "Revolute_23",  # Leg 5 knee
        "Revolute_24",  # Leg 6 knee
    ]

    # Foot joints
    foot_joint_names = [
        "Revolute_5",   # Leg 1 foot
        "Revolute_8",   # Leg 2 foot
        "Revolute_13",  # Leg 3 foot
        "Revolute_25",  # Leg 4 foot
        "Revolute_26",  # Leg 5 foot
        "Revolute_27",  # Leg 6 foot
    ]

    # All joint names in order (for action mapping)
    all_joint_names = [
        # Leg 1
        "Revolute_14", "Revolute_15", "Revolute_5",
        # Leg 2
        "Revolute_6", "Revolute_7", "Revolute_8",
        # Leg 3
        "Revolute_11", "Revolute_12", "Revolute_13",
        # Leg 4
        "Revolute_17", "Revolute_18", "Revolute_25",
        # Leg 5
        "Revolute_22", "Revolute_23", "Revolute_26",
        # Leg 6
        "Revolute_21", "Revolute_24", "Revolute_27",
    ]

    # Tripod groups (indices into the 6 legs)
    tripod_group_a = [0, 2, 4]  # Leg 1, 3, 5 (右後、右中、右前)
    tripod_group_b = [1, 3, 5]  # Leg 2, 4, 6 (左後、左中、左前)

    # ===================
    # Action Scaling (降低動作幅度防止爆炸)
    # ===================
    action_scale = 0.25  # 大幅降低動作縮放因子
    hip_action_scale = 0.15  # ABAD 關節更小的動作範圍
    knee_action_scale = 0.25
    foot_action_scale = 0.2

    # ===================
    # Velocity Command Ranges (降低目標難度)
    # ===================
    # 速度命令範圍 [min, max] - 降低目標讓機器人更容易達成
    lin_vel_x_range = [0.15, 0.3]   # 前進速度 - 降低目標 (m/s)
    lin_vel_y_range = [0.0, 0.0]    # 側向速度 - 先學直走 (m/s)
    ang_vel_z_range = [0.0, 0.0]    # 旋轉速度 - 先學直走 (rad/s)

    # ===================
    # Gait Parameters
    # ===================
    gait_frequency = 2.0  # 步態頻率 (Hz) - 三足步態的切換頻率
    stance_duration_ratio = 0.5  # 站立相位占比

    # ===================
    # Reward Scales (極度簡化 - 只專注前進和存活)
    # ===================
    # 核心獎勵 - 只獎勵正確行為，減少懲罰
    rew_scale_lin_vel_xy = 3.0      # 追蹤線速度 - 主要獎勵
    rew_scale_ang_vel_z = 0.3       # 追蹤角速度
    rew_scale_lin_vel_z = -0.2      # 輕微懲罰垂直振動

    # 存活獎勵 (關鍵!)
    rew_scale_alive = 1.0           # 每步存活獎勵 - 提高
    
    # 前進獎勵 (最重要!)
    rew_scale_forward_progress = 2.0  # 大幅獎勵向前移動
    
    # 步態獎勵 - 全部禁用，讓機器人自己探索
    rew_scale_tripod_contact = 0.0  
    rew_scale_gait_phase = 0.0      
    rew_scale_feet_air_time = 0.0   

    # ABAD 獎勵 - 全部禁用
    rew_scale_abad_usage = 0.0      
    rew_scale_abad_symmetry = 0.0   

    # 穩定性 - 極輕懲罰
    rew_scale_orientation = -0.05   # 幾乎不懲罰傾斜
    rew_scale_base_height = -0.05   # 幾乎不懲罰高度

    # 平滑性 - 極輕懲罰
    rew_scale_action_rate = -0.001  
    rew_scale_joint_acc = -1e-8     
    rew_scale_joint_torque = -1e-6

    # 碰撞懲罰
    rew_scale_collision = -1.0      # 懲罰身體碰撞
    rew_scale_stumble = -0.5        # 懲罰絆倒

    # ===================
    # Termination Conditions
    # ===================
    # 使用相對於初始姿態的傾斜量來判斷終止
    # max_tilt_magnitude 是投影重力向量與初始投影重力向量的差距
    # 0 = 與初始姿態相同, 1.0 ≈ 45度傾斜, 1.414 ≈ 90度傾斜, 2.0 = 完全翻轉
    max_tilt_magnitude = 1.2  # 允許約 50-60 度的傾斜
    min_base_height = 0.02  # 最低高度 (m) - 放寬限制
    max_base_height = 0.5   # 最高高度 (m)

    # ===================
    # Domain Randomization
    # ===================
    # 質量隨機化
    randomize_mass = True
    mass_range = [0.9, 1.1]  # 質量變化範圍 (比例)

    # 摩擦力隨機化
    randomize_friction = True
    friction_range = [0.5, 1.25]

    # 關節屬性隨機化
    randomize_joint_friction = True
    joint_friction_range = [0.0, 0.05]

    # 推力擾動
    push_robots = True
    push_interval_s = 15.0  # 每 15 秒推一次
    max_push_vel_xy = 0.5   # 最大推力速度 (m/s)

    # ===================
    # Noise Settings
    # ===================
    add_noise = True
    noise_level = 1.0

    # 觀測噪聲
    noise_lin_vel = 0.1
    noise_ang_vel = 0.2
    noise_gravity = 0.05
    noise_joint_pos = 0.01
    noise_joint_vel = 1.5