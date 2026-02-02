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
from typing import Optional

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
# ContactSensor 暫時禁用，等待 USD 檔案添加 contact reporter API
# from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse, quat_apply, sample_uniform
import isaaclab.utils.math as math_utils

# Visualization Markers for debug arrows
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import GREEN_ARROW_X_MARKER_CFG, RED_ARROW_X_MARKER_CFG

# Terrain configuration (doesn't require Isaac Lab runtime)
from .terrain_cfg import TerrainCfg, TerrainType

# Note: TerrainGenerator is imported lazily in _setup_procedural_terrain()
# because it requires omni.usd which is only available after simulation starts

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
    
    # Terrain generator instance (only created if procedural terrain is enabled)
    _terrain_generator: Optional[TerrainGenerator] = None

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
        
        # 打印地形配置信息
        self._print_terrain_info()
        
        # 自動啟用 debug visualization（如果配置啟用且有 GUI）
        if hasattr(self.cfg, 'draw_debug_vis') and self.cfg.draw_debug_vis:
            if self.sim.has_gui():
                self.set_debug_vis(True)
                print("[RedrhexEnv] Debug visualization 已啟用")
            else:
                print("[RedrhexEnv] 無 GUI 模式，跳過 debug visualization")
        
        # Initialize terrain debug visualization if enabled
        if hasattr(self.cfg, 'procedural_terrain') and self.cfg.procedural_terrain.debug_visualize:
            if self.sim.has_gui():
                self._setup_terrain_debug_vis()
                print("[RedrhexEnv] Terrain debug visualization 已啟用")

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
            "rew_correct_dir": torch.zeros(self.num_envs, device=self.device),
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
            # ★★★ 新增：腿速度診斷 ★★★
            "diag_target_leg_vel": torch.zeros(self.num_envs, device=self.device),
            "diag_leg_vel_error": torch.zeros(self.num_envs, device=self.device),
        }

        # 初始化目標速度緩衝
        self._target_drive_vel = torch.zeros(self.num_envs, 6, device=self.device)

    def _setup_commands(self):
        """設置多方向速度命令系統"""
        # 速度命令 [vx, vy, wz]
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
        
        # 初始化命令
        self._resample_commands(torch.arange(self.num_envs, device=self.device))

    def _resample_commands(self, env_ids: torch.Tensor):
        """重新採樣速度命令"""
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
        """更新命令（定期切換方向）"""
        dt = self.cfg.sim.dt * self.cfg.decimation
        self.command_time_left -= dt
        
        # 找出需要重新採樣的環境
        resample_ids = (self.command_time_left <= 0).nonzero(as_tuple=False).flatten()
        if len(resample_ids) > 0:
            self._resample_commands(resample_ids)

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
        """
        設置模擬場景
        
        This method handles:
        1. Robot articulation setup
        2. Terrain setup (flat or procedural based on config)
        3. Environment cloning
        4. Lighting
        
        The terrain type is determined by cfg.procedural_terrain.terrain_type:
        - FLAT: Uses standard Isaac Lab plane ground (backward compatible)
        - ROUGH/STAIRS/OBSTACLES/MIXED: Uses procedural TerrainGenerator
        """
        self.robot = Articulation(self.cfg.robot_cfg)
        self.scene.articulations["robot"] = self.robot
        
        # 注意：ContactSensor 暫時禁用，因為 USD 檔案缺少 contact reporter API
        # 用高度和姿態檢測來代替
        # self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        # self.scene.sensors["contact_sensor"] = self._contact_sensor

        # =====================================================================
        # TERRAIN SETUP: Flat vs Procedural
        # =====================================================================
        # Check if procedural terrain is enabled
        use_procedural_terrain = (
            hasattr(self.cfg, 'procedural_terrain')
            and self.cfg.procedural_terrain.is_procedural()
        )
        
        if use_procedural_terrain:
            # Initialize and generate procedural terrain
            self._setup_procedural_terrain()
        else:
            # Use default flat terrain (backward compatible)
            self.cfg.terrain.num_envs = self.scene.cfg.num_envs
            self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
            self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
            self._terrain_generator = None  # No procedural terrain

        self.scene.clone_environments(copy_from_source=False)

        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
    
    def _setup_procedural_terrain(self) -> None:
        """
        Initialize and generate procedural terrain based on configuration.
        
        This method:
        1. Creates TerrainGenerator with parameters from cfg.procedural_terrain
        2. Generates terrain with current difficulty level
        3. Adjusts robot spawn height based on terrain height at origin
        """
        # Lazy import - TerrainGenerator requires omni.usd which is only
        # available after Isaac Lab simulation is initialized
        from .terrain_manager import TerrainGenerator, TerrainConfig
        
        terrain_cfg = self.cfg.procedural_terrain
        
        # Convert TerrainCfg to TerrainConfig for TerrainGenerator
        generator_config = TerrainConfig(
            grid_size=terrain_cfg.grid_size,
            cell_size=terrain_cfg.horizontal_scale,
            base_friction=terrain_cfg.friction,
            max_height_variance=terrain_cfg.vertical_scale,
            max_stair_height=terrain_cfg.max_stair_height,
            max_stair_depth=terrain_cfg.max_stair_depth,
            max_obstacle_density=terrain_cfg.obstacle_density,
            min_obstacle_size=terrain_cfg.min_obstacle_size,
            max_obstacle_size=terrain_cfg.max_obstacle_size,
            terrain_prim_path=terrain_cfg.terrain_prim_path,
            random_seed=terrain_cfg.random_seed,
        )
        
        # Create terrain generator
        self._terrain_generator = TerrainGenerator(generator_config)
        
        # Generate terrain with current difficulty
        terrain_type_str = terrain_cfg.get_terrain_type_string()
        self._terrain_generator.generate(
            difficulty=terrain_cfg.difficulty_scale,
            terrain_type=terrain_type_str,
        )
        
        print(f"[Terrain] Generated {terrain_type_str} terrain with difficulty {terrain_cfg.difficulty_scale:.2f}")
        
        # Also create the default flat terrain (needed for collision filtering)
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
    
    def regenerate_terrain(self, difficulty: float, terrain_type: Optional[str] = None) -> None:
        """
        Regenerate procedural terrain with new difficulty level.
        
        Use this method during curriculum learning to increase terrain difficulty
        as the agent improves.
        
        Args:
            difficulty: New difficulty level from 0.0 to 1.0
            terrain_type: Optional new terrain type. If None, uses current type.
        
        Raises:
            RuntimeError: If procedural terrain is not enabled
        """
        if self._terrain_generator is None:
            raise RuntimeError(
                "Cannot regenerate terrain: procedural terrain is not enabled. "
                "Set cfg.procedural_terrain.terrain_type to ROUGH, STAIRS, OBSTACLES, or MIXED."
            )
        
        # Update config
        self.cfg.procedural_terrain.difficulty_scale = difficulty
        
        # Get terrain type
        if terrain_type is None:
            terrain_type = self.cfg.procedural_terrain.get_terrain_type_string()
        
        # Regenerate
        self._terrain_generator.generate(
            difficulty=difficulty,
            terrain_type=terrain_type,
        )
        
        print(f"[Terrain] Regenerated {terrain_type} terrain with difficulty {difficulty:.2f}")
    
    def get_terrain_height_at_position(self, x: float, y: float) -> float:
        """
        Get the terrain height at a specific (x, y) position.
        
        Useful for adjusting spawn positions or checking terrain elevation.
        
        Args:
            x: X coordinate in world frame
            y: Y coordinate in world frame
            
        Returns:
            Terrain height at (x, y), or 0.0 if flat terrain
        """
        if self._terrain_generator is None:
            return 0.0  # Flat terrain
        
        # For now, return a conservative estimate
        # TODO: Implement actual height sampling from terrain mesh
        return self.cfg.procedural_terrain.vertical_scale * self.cfg.procedural_terrain.difficulty_scale
    
    def _print_terrain_info(self) -> None:
        """Print terrain configuration information."""
        if not hasattr(self.cfg, 'procedural_terrain'):
            print(f"[Terrain] Using default flat terrain")
            return
            
        terrain_cfg = self.cfg.procedural_terrain
        print(f"\n🏔️  Terrain Configuration:")
        print(f"   Type: {terrain_cfg.terrain_type.name}")
        print(f"   Procedural: {terrain_cfg.is_procedural()}")
        
        if terrain_cfg.is_procedural():
            print(f"   Difficulty: {terrain_cfg.difficulty_scale:.2f}")
            print(f"   Grid Size: {terrain_cfg.grid_size[0]:.1f}m x {terrain_cfg.grid_size[1]:.1f}m")
            print(f"   Cell Size: {terrain_cfg.horizontal_scale:.2f}m")
            print(f"   Max Height: {terrain_cfg.vertical_scale:.3f}m")
            print(f"   Friction: {terrain_cfg.friction:.2f}")
            print(f"   Debug Vis: {terrain_cfg.debug_visualize}")
        else:
            print(f"   Using Isaac Lab default plane ground")
    
    def _setup_terrain_debug_vis(self) -> None:
        """
        Setup debug visualization for terrain bounds.
        
        Draws red lines around the active terrain area when
        cfg.procedural_terrain.debug_visualize is True.
        """
        # This will be called after terrain is generated
        # to draw boundary markers
        if not hasattr(self, '_terrain_debug_markers'):
            self._terrain_debug_markers = None
        
        # Visualization will be updated in _draw_terrain_bounds()
        print("[Terrain] Debug visualization markers initialized")
    
    def visualize_terrain_bounds(self) -> None:
        """
        Draw red lines around the active terrain area.
        
        This method uses Isaac Lab's debug drawing tools to visualize
        the terrain boundaries. Useful for verifying the robot is
        seeing the correct map.
        
        Only works when:
        1. Simulation has GUI enabled
        2. cfg.procedural_terrain.debug_visualize is True
        """
        if not self.sim.has_gui():
            return
            
        if not hasattr(self.cfg, 'procedural_terrain'):
            return
            
        terrain_cfg = self.cfg.procedural_terrain
        if not terrain_cfg.debug_visualize:
            return
        
        try:
            from omni.isaac.debug_draw import _debug_draw
            draw = _debug_draw.acquire_debug_draw_interface()
            
            # Get terrain bounds
            half_x = terrain_cfg.grid_size[0] / 2
            half_y = terrain_cfg.grid_size[1] / 2
            z = 0.1  # Draw slightly above ground
            
            # Define corner points
            corners = [
                (-half_x, -half_y, z),
                (half_x, -half_y, z),
                (half_x, half_y, z),
                (-half_x, half_y, z),
            ]
            
            # Draw boundary lines (red)
            color = terrain_cfg.debug_vis_color + (1.0,)  # Add alpha
            line_width = 3.0
            
            for i in range(4):
                start = corners[i]
                end = corners[(i + 1) % 4]
                draw.draw_line(start, color, end, color)
            
            # Draw spawn area marker (green)
            spawn_size = terrain_cfg.spawn_area_size / 2
            spawn_color = (0.0, 1.0, 0.0, 1.0)
            spawn_corners = [
                (-spawn_size, -spawn_size, z + 0.01),
                (spawn_size, -spawn_size, z + 0.01),
                (spawn_size, spawn_size, z + 0.01),
                (-spawn_size, spawn_size, z + 0.01),
            ]
            for i in range(4):
                start = spawn_corners[i]
                end = spawn_corners[(i + 1) % 4]
                draw.draw_line(start, spawn_color, end, spawn_color)
                
        except ImportError:
            print("[Terrain] Debug draw interface not available")
        
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
        base_vel = self.cfg.base_gait_angular_vel  # 6.28 rad/s
        
        # ★★★ 重新設計動作解釋 ★★★
        # 動作應該只調整速度大小，不能完全停止腿！
        # 動作 -1 → 最低速度 (base_vel * 0.5)
        # 動作  0 → 基礎速度 (base_vel)
        # 動作 +1 → 最高速度 (base_vel * 1.5)
        # 這樣腿永遠在轉，RL 只能調整快慢
        speed_scale = 1.0 + drive_actions * 0.5  # [0.5, 1.5]
        target_speed = base_vel * speed_scale  # [3.14, 9.42] rad/s
        
        # 應用方向乘數
        # 右側 (idx 0,1,2) → -1, 左側 (idx 3,4,5) → +1
        target_drive_vel = target_speed * self._direction_multiplier
        
        # 限制速度範圍
        target_drive_vel = torch.clamp(target_drive_vel, min=-15.0, max=15.0)
        
        # 保存目標速度用於診斷
        self._target_drive_vel = target_drive_vel.clone()
        
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
        
        # 更新速度命令（定期切換方向）
        self._update_commands()

    def _get_rewards(self) -> torch.Tensor:
        """
        ===== RHex 機器人多方向速度追蹤 Reward（按照用戶需求重新設計）=====
        
        【設計原則】
        G1: 追蹤項（核心）- 線速度 + 角速度
        G2: 姿態與穩定性 - 避免偏航亂翻、避免彈跳
        G3: 身體觸地 - 必須強烈懲罰！！
        G4: 能耗與動作平滑
        G5: 步態相位結構 - 打擊六腿同相
        G6: ABAD 使用策略 - 有 lateral/yaw 分量就鼓勵 ABAD
        
        【關節定義】
        - 主驅動 (15,12,18,23,24,7): 360° 連續旋轉推進
        - 避震 (5,13,25,26,27,8): 只吸震，不推進
        - ABAD (14,11,17,22,21,6): 側向與穩定性輔助
        
        【Tripod 分組】
        - Tripod A (同相): 15, 18, 24
        - Tripod B (同相): 12, 23, 7
        - A 與 B 相差 180°
        """
        # 初始化總獎勵
        total_reward = torch.zeros(self.num_envs, device=self.device)
        dt = self.step_dt  # 時間步長

        # ===== 獲取狀態 =====
        main_drive_vel = self.joint_vel[:, self._main_drive_indices]  # [N, 6]
        main_drive_pos = self.joint_pos[:, self._main_drive_indices]  # [N, 6]
        abad_pos = self.joint_pos[:, self._abad_indices]  # [N, 6]
        abad_vel = self.joint_vel[:, self._abad_indices]  # [N, 6]
        
        # 目標速度命令
        cmd_vx = self.commands[:, 0]  # 目標前進速度
        cmd_vy = self.commands[:, 1]  # 目標側向速度
        cmd_wz = self.commands[:, 2]  # 目標旋轉速度
        
        # 實際速度（本體座標系）
        actual_vx = self.base_lin_vel[:, 0]
        actual_vy = self.base_lin_vel[:, 1]
        actual_vz = self.base_lin_vel[:, 2]
        actual_wz = self.base_ang_vel[:, 2]
        
        # 計算任務需求強度 S = α*|vy*| + β*|wz*|
        S = torch.abs(cmd_vy) + 0.5 * torch.abs(cmd_wz)
        S0 = 0.3  # 歸一化閾值

        # ========================================================
        # G1: 追蹤項（核心）- 參考 anymal_c 的 exp 映射
        # ========================================================
        
        # G1.1 線速度 XY 追蹤: r_vel = exp(-|e_v|^2 / 0.25)
        lin_vel_error = torch.sum(
            torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1
        )
        lin_vel_error_mapped = torch.exp(-lin_vel_error / 0.25)
        rew_track_lin_vel = lin_vel_error_mapped * self.cfg.rew_scale_track_lin_vel * dt
        total_reward += rew_track_lin_vel
        
        # G1.2 角速度 Z 追蹤: r_yaw = exp(-|e_w|^2 / 0.25)
        yaw_rate_error = torch.square(cmd_wz - actual_wz)
        yaw_rate_error_mapped = torch.exp(-yaw_rate_error / 0.25)
        rew_track_ang_vel = yaw_rate_error_mapped * self.cfg.rew_scale_track_ang_vel * dt
        total_reward += rew_track_ang_vel

        # ========================================================
        # G2: 姿態與穩定性（避免偏航亂翻、避免彈跳）
        # ========================================================
        
        # G2.1 俯仰/側滾穩定: r_upright = -(p^2 + r^2)
        # projected_gravity 的 xy 分量反映 roll/pitch
        flat_orientation = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
        rew_upright = flat_orientation * self.cfg.rew_scale_upright * dt
        total_reward += rew_upright
        
        # G2.2 垂直彈跳抑制: r_smooth = -vz^2
        z_vel_error = torch.square(actual_vz)
        rew_z_vel = z_vel_error * self.cfg.rew_scale_z_vel * dt
        total_reward += rew_z_vel
        
        # G2.3 xy 角速度懲罰
        ang_vel_xy_error = torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
        rew_ang_vel_xy = ang_vel_xy_error * self.cfg.rew_scale_ang_vel_xy * dt
        total_reward += rew_ang_vel_xy
        
        # G2.4 高度保持
        base_height = self.robot.data.root_pos_w[:, 2]
        target_height = 0.12
        height_error = torch.square(base_height - target_height)
        rew_base_height = height_error * self.cfg.rew_scale_base_height * dt
        total_reward += rew_base_height
        
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

        # ========================================================
        # G3: 身體觸地（必須強烈！！）- 用高度/姿態檢測代替 ContactSensor
        # ========================================================
        
        # ★★★ 重要：RHex 正常站立高度只有約 1.6cm！★★★
        # 基於高度的"觸地"檢測對 RHex 不適用
        # ★★★ 身體觸地檢測（高度 + 傾斜）★★★
        # 注意：機器人初始姿態是繞 X 軸旋轉 90°，所以不能假設 projected_gravity = [0,0,-1]
        # 必須使用 reference_projected_gravity 做相對比較
        body_height = base_height
        
        # 傾斜程度 = 1 - 當前重力與參考重力的點積
        # 點積 = 1 表示完全對齊（0° 傾斜）
        # 點積 = 0 表示垂直（90° 傾斜）
        # 點積 = -1 表示完全翻轉（180° 傾斜）
        gravity_alignment = torch.sum(
            self.projected_gravity * self.reference_projected_gravity, dim=1
        )  # [-1, 1]
        
        # 傾斜程度: 0 = 完全對齊, 1 = 90°, 2 = 180°
        body_tilt = 1.0 - gravity_alignment  # [0, 2]
        
        # 身體觸地條件：
        # 1. 高度過低（< 0.01m，正常站立約 0.12m）- 身體趴地
        # 2. 傾斜過大（> 0.5，表示傾斜超過約 60°）- 側翻或前後翻
        height_threshold = getattr(self.cfg, 'body_contact_height_threshold', 0.01)
        height_contact = body_height < height_threshold
        severe_tilt = body_tilt > 0.5  # cos(60°) = 0.5, 所以 1 - 0.5 = 0.5
        body_contact = height_contact | severe_tilt  # 任一條件都算觸地
        
        # 身體觸地懲罰（即使不終止也要懲罰）
        rew_body_contact = body_contact.float() * self.cfg.rew_scale_body_contact * dt
        total_reward += rew_body_contact
        
        # 連續傾斜懲罰：傾斜越大懲罰越大（鼓勵保持平衡）
        # body_tilt 範圍 [0, 2]，0.2 約等於 25° 傾斜
        tilt_penalty = torch.clamp(body_tilt - 0.2, min=0.0) * 5.0  # 傾斜超過 25° 開始懲罰
        total_reward -= tilt_penalty * dt
        
        # 記錄用於終止條件
        self._body_contact = body_contact
        self._body_tilt = body_tilt  # 保存用於 _get_dones

        # ========================================================
        # G4: 能耗與動作平滑
        # ========================================================
        
        # G4.1 力矩懲罰（如果有）
        if hasattr(self.robot.data, 'applied_torque'):
            joint_torques = torch.sum(torch.square(self.robot.data.applied_torque), dim=1)
            rew_torque = joint_torques * self.cfg.rew_scale_torque * dt
            total_reward += rew_torque
        
        # G4.2 動作變化率懲罰
        action_rate = torch.sum(torch.square(self.actions - self.last_actions), dim=1)
        rew_action_rate = action_rate * self.cfg.rew_scale_action_rate * dt
        total_reward += rew_action_rate
        
        # G4.3 關節加速度懲罰
        if hasattr(self.robot.data, 'joint_acc'):
            joint_accel = torch.sum(torch.square(self.robot.data.joint_acc), dim=1)
            rew_joint_acc = joint_accel * self.cfg.rew_scale_joint_acc * dt
            total_reward += rew_joint_acc

        # ========================================================
        # G5: 步態相位結構（打擊六腿同相！）
        # ========================================================
        
        # 計算主髖 joint 的相位
        effective_pos = main_drive_pos * self._direction_multiplier
        leg_phase = torch.remainder(effective_pos, 2 * math.pi)
        
        phase_a = leg_phase[:, self._tripod_a_indices]  # Tripod A: idx 0,3,5
        phase_b = leg_phase[:, self._tripod_b_indices]  # Tripod B: idx 1,2,4
        
        # G5.1 組內一致性（鼓勵 A 組內三腿同相、B 組內三腿同相）
        # 使用相位向量的長度來衡量一致性（越接近 1 越一致）
        def phase_coherence(phases):
            sin_mean = torch.sin(phases).mean(dim=1)
            cos_mean = torch.cos(phases).mean(dim=1)
            return torch.sqrt(sin_mean**2 + cos_mean**2)
        
        coherence_a = phase_coherence(phase_a)
        coherence_b = phase_coherence(phase_b)
        rew_tripod_sync = (coherence_a + coherence_b) * self.cfg.rew_scale_tripod_sync * dt
        total_reward += rew_tripod_sync
        
        # G5.2 組間反相（鼓勵 A 與 B 相差 π）
        mean_phase_a = torch.atan2(torch.sin(phase_a).mean(dim=1), torch.cos(phase_a).mean(dim=1))
        mean_phase_b = torch.atan2(torch.sin(phase_b).mean(dim=1), torch.cos(phase_b).mean(dim=1))
        phase_diff = torch.abs(mean_phase_a - mean_phase_b)
        phase_diff = torch.min(phase_diff, 2 * math.pi - phase_diff)  # wrap to [0, π]
        phase_diff_error = torch.abs(phase_diff - math.pi)  # 與 π 的差距
        rew_antiphase = torch.exp(-phase_diff_error) * self.cfg.rew_scale_tripod_antiphase * dt
        total_reward += rew_antiphase

        # ========================================================
        # G6: ABAD 使用策略（重點：有 lateral/yaw 分量就鼓勵）
        # ========================================================
        
        # ABAD 使用量
        U_abad = torch.sum(torch.square(abad_vel), dim=1)  # 使用角速度作為使用量
        abad_magnitude = torch.abs(abad_pos).mean(dim=1)
        
        # G6.1 需要時鼓勵使用 ABAD
        # r_abad_use = +w * S * tanh(c * U_abad)
        rew_abad_smart = S * torch.tanh(0.5 * U_abad) * self.cfg.rew_scale_abad_smart_use * dt
        total_reward += rew_abad_smart
        
        # G6.2 不需要時抑制 ABAD 亂動
        # p_abad_waste = -w * (1 - clamp(S/S0)) * U_abad
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

        # ========================================================
        # 存活獎勵
        # ========================================================
        rew_alive = torch.ones(self.num_envs, device=self.device) * self.cfg.rew_scale_alive * dt
        total_reward += rew_alive

        # ========================================================
        # ★★★ 新增：靜止懲罰（打擊躺平策略！）★★★
        # ========================================================
        # 當命令要求移動但機器人幾乎不動時，給予懲罰
        cmd_speed = torch.sqrt(cmd_vx**2 + cmd_vy**2 + 0.1 * cmd_wz**2)  # 命令速度
        actual_speed = torch.sqrt(actual_vx**2 + actual_vy**2)  # 實際速度
        
        # 如果命令速度 > 0.1 但實際速度 < 0.05，懲罰
        not_moving = (cmd_speed > 0.1) & (actual_speed < 0.05)
        rew_stationary_penalty = not_moving.float() * (-3.0) * dt
        total_reward += rew_stationary_penalty
        
        # 額外：鼓勵腿積極轉動（當有命令時）
        # 腿旋轉速度越接近目標越好
        target_leg_vel = 6.28 * torch.clamp(cmd_speed / 0.4, max=1.0)  # 最高 6.28 rad/s
        actual_leg_vel = torch.abs(main_drive_vel * self._direction_multiplier).mean(dim=1)
        leg_vel_reward = torch.where(
            cmd_speed > 0.05,
            torch.clamp(actual_leg_vel / (target_leg_vel + 0.1), max=1.5) * 1.5,
            torch.zeros_like(actual_leg_vel)
        ) * dt
        total_reward += leg_vel_reward

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
        rew_gait_sync = rew_antiphase
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
        
        # ★★★ 新增：腿速度診斷 ★★★
        target_leg_vel_abs = torch.abs(self._target_drive_vel).mean(dim=1)
        leg_vel_error = torch.abs(torch.abs(main_drive_vel) - torch.abs(self._target_drive_vel)).mean(dim=1)
        
        self.episode_sums["diag_target_leg_vel"] += target_leg_vel_abs
        self.episode_sums["diag_leg_vel_error"] += leg_vel_error
        
        self.last_main_drive_vel = main_drive_vel.clone()

        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        ===== 終止條件（重新設計：加入身體觸地終止！）=====
        
        【關鍵改動】
        - 身體觸地 (body contact) 必須終止！這是防止「翻車取巧」的核心
        - 翻轉超過閾值終止
        - 物理失控終止
        """
        # 超時
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # 終止條件
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        root_pos = self.robot.data.root_pos_w
        root_vel = self.robot.data.root_lin_vel_w
        
        # 1. 物理爆炸檢測（NaN/Inf）
        pos_invalid = torch.any(torch.isnan(root_pos) | torch.isinf(root_pos), dim=1)
        vel_invalid = torch.any(torch.isnan(root_vel) | torch.isinf(root_vel), dim=1)
        
        # 2. 速度過快（物理失控）
        vel_too_fast = torch.any(torch.abs(root_vel) > 30.0, dim=1)

        # 3. 翻車檢測 - 使用參考重力方向
        # 注意：機器人初始姿態是繞 X 軸旋轉 90°，所以不能假設 projected_gravity[:, 2] 的值
        # 使用重力對齊度：點積 < -0.2 表示翻轉超過約 100°
        if hasattr(self, '_body_tilt'):
            flipped_over = self._body_tilt > 1.2  # 傾斜 > 1.2 表示翻轉超過約 100°
        else:
            # 回退方案：第一次調用時可能還沒有 _body_tilt
            gravity_alignment = torch.sum(
                self.projected_gravity * self.reference_projected_gravity, dim=1
            )
            flipped_over = gravity_alignment < -0.2

        # 4. 高度終止
        base_height = root_pos[:, 2]
        too_low = base_height < -0.1  # 地面以下 10cm
        too_high = base_height > 2.0   # 飛太高
        
        # 5. ★★★ 身體觸地終止（關鍵！） ★★★
        # 如果啟用 terminate_on_body_contact，則身體觸地就終止
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

    # ===================================================================
    # Debug Visualization (Official Isaac Lab Method)
    # ===================================================================
    
    def _set_debug_vis_impl(self, debug_vis: bool):
        """創建或設置 debug visualization markers 的可見性
        
        這是 Isaac Lab DirectRLEnv 的官方接口。當 debug_vis=True 時，
        創建 VisualizationMarkers 並設置可見。當 debug_vis=False 時隱藏。
        
        綠色箭頭 = 目標速度方向
        紅色箭頭 = 實際速度方向
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
        """每個渲染幀更新 debug visualization markers
        
        這個回調函數由 Isaac Lab 自動訂閱到 post_update_event。
        在每次渲染後調用，用於更新箭頭的位置和方向。
        """
        # 檢查機器人是否已初始化
        if not self.robot.is_initialized:
            return
        
        # 獲取機器人位置（箭頭起點在機器人上方）
        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5  # 箭頭高度
        
        # 計算目標速度箭頭的縮放和旋轉
        goal_arrow_scale, goal_arrow_quat = self._resolve_xy_velocity_to_arrow(
            self.commands[:, :2], is_goal=True  # [vx, vy]
        )
        
        # 計算實際速度箭頭的縮放和旋轉
        current_arrow_scale, current_arrow_quat = self._resolve_xy_velocity_to_arrow(
            self.base_lin_vel[:, :2], is_goal=False  # 本體坐標系下的 [vx, vy]
        )
        
        # 更新可視化 markers
        self.goal_vel_visualizer.visualize(base_pos_w, goal_arrow_quat, goal_arrow_scale)
        
        # 實際速度箭頭稍微高一點，避免重疊
        base_pos_w_current = base_pos_w.clone()
        base_pos_w_current[:, 2] += 0.1
        self.current_vel_visualizer.visualize(base_pos_w_current, current_arrow_quat, current_arrow_scale)
    
    def _resolve_xy_velocity_to_arrow(self, xy_velocity: torch.Tensor, is_goal: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        """將 XY 速度向量轉換為箭頭的縮放和旋轉
        
        Args:
            xy_velocity: 本體坐標系下的 XY 速度 [N, 2]
            is_goal: 是否為目標速度箭頭（影響基礎縮放）
        
        Returns:
            arrow_scale: 箭頭縮放 [N, 3]
            arrow_quat: 箭頭旋轉四元數（世界坐標系）[N, 4]
        """
        # 基礎縮放：只改變長度，寬高固定
        if is_goal:
            base_length = 0.8   # 綠色目標箭頭基礎長度
            width_height = 0.25  # 固定寬高
        else:
            base_length = 0.8   # 紅色實際箭頭基礎長度
            width_height = 0.2  # 固定寬高（稍小）
        
        # 計算速度大小
        speed = torch.linalg.norm(xy_velocity, dim=1)
        
        # 箭頭長度根據速度調整：最小 0.3 倍，速度加成 2.0x
        length_scale = base_length * (0.3 + speed * 2.0)
        
        # 創建 scale tensor: [length, width, height]
        arrow_scale = torch.zeros(xy_velocity.shape[0], 3, device=self.device)
        arrow_scale[:, 0] = length_scale  # 長度隨速度變化
        arrow_scale[:, 1] = width_height  # 寬度固定
        arrow_scale[:, 2] = width_height  # 高度固定
        
        # 箭頭方向：根據速度方向計算偏航角（只在 XY 平面上）
        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        
        # 獲取機器人的偏航角（只取 yaw，忽略 roll/pitch）
        # 這樣箭頭永遠在水平面上
        base_quat_w = self.robot.data.root_quat_w
        # 從四元數提取 yaw 角度
        # quat = [w, x, y, z]
        # yaw = atan2(2*(w*z + x*y), 1 - 2*(y^2 + z^2))
        w = base_quat_w[:, 0]
        x = base_quat_w[:, 1]
        y = base_quat_w[:, 2]
        z = base_quat_w[:, 3]
        base_yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        
        # 組合箭頭方向（本體坐標系）和機器人 yaw（世界坐標系）
        world_heading = base_yaw + heading_angle
        
        # 創建只有 yaw 旋轉的四元數（箭頭永遠水平）
        zeros = torch.zeros_like(world_heading)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, world_heading)
        
        return arrow_scale, arrow_quat
