# Copyright (c) 2024, RedRhex Project
# SPDX-License-Identifier: BSD-3-Clause

"""
Terrain Manager Module for RHex Hexapod Robot Training

This module provides procedural terrain generation for Reinforcement Learning
training with Curriculum Learning support. Terrain difficulty scales dynamically
based on the agent's learning progress.

Framework: NVIDIA Isaac Lab (Isaac Sim)
Robot: RHex Hexapod

Key Features:
- Procedural terrain generation with difficulty scaling (0.0 to 1.0)
- Multiple terrain types: rough ground, stairs, discrete obstacles
- Proper physics materials and collision APIs
- Memory-safe cleanup mechanism for long training sessions
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import random

# Isaac Lab imports for simulation and USD manipulation
import isaaclab.sim as sim_utils

# USD/Pixar imports for direct scene manipulation
from pxr import UsdPhysics, UsdGeom, Gf, Sdf, UsdShade, PhysxSchema, Usd

# Omniverse stage utilities (available after Isaac Lab is loaded)
import omni.usd


def _get_current_stage():
    """Get the current USD stage using Omniverse API."""
    return omni.usd.get_context().get_stage()


def _create_xform_prim(stage, prim_path: str):
    """Create an Xform prim at the given path."""
    return UsdGeom.Xform.Define(stage, prim_path)


def _create_cube_prim(
    stage,
    prim_path: str,
    position: Tuple[float, float, float],
    scale: Tuple[float, float, float],
):
    """
    Create a cube prim with collision enabled.
    
    Args:
        stage: USD stage
        prim_path: Path for the cube
        position: (x, y, z) position in world space
        scale: (sx, sy, sz) half-extents (cube default size is 2x2x2, so scale=1 gives size 2)
    """
    cube = UsdGeom.Cube.Define(stage, prim_path)
    
    # Set position using translation
    xformable = UsdGeom.Xformable(cube.GetPrim())
    xformable.ClearXformOpOrder()
    
    # Add translate operation
    translate_op = xformable.AddTranslateOp()
    translate_op.Set(Gf.Vec3d(position[0], position[1], position[2]))
    
    # Add scale operation (cube has default size 2, so we scale by half-extents)
    scale_op = xformable.AddScaleOp()
    scale_op.Set(Gf.Vec3d(scale[0], scale[1], scale[2]))
    
    return cube.GetPrim()


def _create_cylinder_prim(
    stage,
    prim_path: str,
    position: Tuple[float, float, float],
    radius: float,
    height: float,
):
    """Create a cylinder prim with collision enabled."""
    cylinder = UsdGeom.Cylinder.Define(stage, prim_path)
    
    # Set size attributes
    cylinder.GetRadiusAttr().Set(radius)
    cylinder.GetHeightAttr().Set(height)
    
    # Set position
    xformable = UsdGeom.Xformable(cylinder.GetPrim())
    xformable.ClearXformOpOrder()
    translate_op = xformable.AddTranslateOp()
    translate_op.Set(Gf.Vec3d(position[0], position[1], position[2]))
    
    return cylinder.GetPrim()


# =============================================================================
# CONFIGURATION DATACLASS
# =============================================================================

class TerrainConfig:
    """
    Configuration container for terrain generation parameters.
    
    This class holds all the tunable parameters that control terrain generation.
    Using a config class makes it easy to:
    1. Save/load configurations for reproducibility
    2. Modify parameters without changing code
    3. Pass settings cleanly between functions
    
    Attributes:
        grid_size: The total size of the terrain grid in meters (width x length)
        cell_size: Size of each terrain cell/tile in meters
        base_friction: Default friction coefficient for terrain surfaces
        max_height_variance: Maximum height variation for rough terrain at difficulty=1.0
        max_stair_height: Maximum stair step height at difficulty=1.0
        max_obstacle_density: Maximum obstacle density (0-1) at difficulty=1.0
        terrain_prim_path: USD path where terrain prims will be created
    """
    
    def __init__(
        self,
        grid_size: Tuple[float, float] = (10.0, 10.0),
        cell_size: float = 0.5,
        base_friction: float = 0.8,
        max_height_variance: float = 0.15,
        max_stair_height: float = 0.12,
        max_stair_depth: float = 0.3,
        max_obstacle_density: float = 0.3,
        min_obstacle_size: float = 0.05,
        max_obstacle_size: float = 0.2,
        terrain_prim_path: str = "/World/ProceduralTerrain",
        random_seed: Optional[int] = None,
    ):
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.base_friction = base_friction
        self.max_height_variance = max_height_variance
        self.max_stair_height = max_stair_height
        self.max_stair_depth = max_stair_depth
        self.max_obstacle_density = max_obstacle_density
        self.min_obstacle_size = min_obstacle_size
        self.max_obstacle_size = max_obstacle_size
        self.terrain_prim_path = terrain_prim_path
        self.random_seed = random_seed
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "TerrainConfig":
        """Create a TerrainConfig from a dictionary."""
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization."""
        return {
            "grid_size": self.grid_size,
            "cell_size": self.cell_size,
            "base_friction": self.base_friction,
            "max_height_variance": self.max_height_variance,
            "max_stair_height": self.max_stair_height,
            "max_stair_depth": self.max_stair_depth,
            "max_obstacle_density": self.max_obstacle_density,
            "min_obstacle_size": self.min_obstacle_size,
            "max_obstacle_size": self.max_obstacle_size,
            "terrain_prim_path": self.terrain_prim_path,
            "random_seed": self.random_seed,
        }


# =============================================================================
# MAIN TERRAIN GENERATOR CLASS
# =============================================================================

class TerrainGenerator:
    """
    Procedural Terrain Generator for RHex Hexapod Robot Training.
    
    This class handles all terrain generation for RL training with curriculum
    learning support. The difficulty parameter (0.0 to 1.0) controls how
    challenging the generated terrain will be.
    
    Difficulty Scaling Philosophy:
    - 0.0 = Flat, easy terrain (agent learns basic locomotion)
    - 0.5 = Moderate terrain (agent learns adaptation)
    - 1.0 = Maximum difficulty (agent masters complex locomotion)
    
    Usage Example:
        >>> config = TerrainConfig(grid_size=(8.0, 8.0), base_friction=0.7)
        >>> generator = TerrainGenerator(config)
        >>> generator.generate(difficulty=0.3)  # Generate easy-moderate terrain
        >>> # ... training loop ...
        >>> generator.generate(difficulty=0.8)  # Increase difficulty as agent improves
    
    Attributes:
        config: TerrainConfig instance with generation parameters
        _generated_prims: List of created prim paths for cleanup
    """
    
    def __init__(self, config: Optional[TerrainConfig] = None):
        """
        Initialize the TerrainGenerator.
        
        Args:
            config: TerrainConfig instance. If None, uses default configuration.
        """
        # Use provided config or create default
        self.config = config if config is not None else TerrainConfig()
        
        # Initialize random number generator for reproducibility
        if self.config.random_seed is not None:
            np.random.seed(self.config.random_seed)
            random.seed(self.config.random_seed)
        
        # Track all generated prims for cleanup
        self._generated_prims: List[str] = []
        
        # Physics material path
        self._physics_material_path = f"{self.config.terrain_prim_path}/PhysicsMaterial"
        
        # Flag to track initialization
        self._initialized = False
    
    # =========================================================================
    # PUBLIC INTERFACE
    # =========================================================================
    
    def generate(self, difficulty: float, terrain_type: str = "mixed") -> None:
        """
        Generate terrain with specified difficulty level.
        
        Args:
            difficulty: Float from 0.0 (easy) to 1.0 (hard)
            terrain_type: "rough", "stairs", "obstacles", or "mixed"
        """
        if not 0.0 <= difficulty <= 1.0:
            raise ValueError(f"Difficulty must be between 0.0 and 1.0, got {difficulty}")
        
        # Clear existing terrain
        self.clear_terrain()
        
        # Setup base structure
        self._setup_base_structure()
        
        # Generate terrain based on type
        if terrain_type == "rough":
            self._generate_rough_terrain(difficulty)
        elif terrain_type == "stairs":
            self._generate_stairs(difficulty)
        elif terrain_type == "obstacles":
            self._generate_discrete_obstacles(difficulty)
        elif terrain_type == "mixed":
            self._generate_mixed_terrain(difficulty)
        else:
            raise ValueError(f"Unknown terrain type: {terrain_type}")
        
        self._initialized = True
    
    def reset(self, difficulty: float, terrain_type: str = "mixed") -> None:
        """Alias for generate() for RL environment compatibility."""
        self.generate(difficulty, terrain_type)
    
    def clear_terrain(self) -> None:
        """Remove all previously generated terrain prims."""
        stage = _get_current_stage()
        if stage is None:
            return
        
        # Remove the entire terrain parent prim
        terrain_prim = stage.GetPrimAtPath(self.config.terrain_prim_path)
        if terrain_prim.IsValid():
            stage.RemovePrim(self.config.terrain_prim_path)
        
        self._generated_prims.clear()
        self._initialized = False
    
    def get_terrain_bounds(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Get the XY bounds of the generated terrain."""
        half_x = self.config.grid_size[0] / 2
        half_y = self.config.grid_size[1] / 2
        return ((-half_x, half_x), (-half_y, half_y))
    
    # =========================================================================
    # PRIVATE: BASE STRUCTURE SETUP
    # =========================================================================
    
    def _setup_base_structure(self) -> None:
        """Set up the foundational terrain structure."""
        stage = _get_current_stage()
        
        # Create parent Xform
        _create_xform_prim(stage, self.config.terrain_prim_path)
        self._generated_prims.append(self.config.terrain_prim_path)
        
        # Create physics material
        self._create_physics_material(stage)
        
        # Create ground plane
        self._create_ground_plane(stage)
    
    def _create_physics_material(self, stage) -> None:
        """Create a shared physics material for terrain surfaces."""
        material_prim = UsdShade.Material.Define(stage, self._physics_material_path)
        
        # Apply physics material API
        physics_material = UsdPhysics.MaterialAPI.Apply(material_prim.GetPrim())
        physics_material.CreateStaticFrictionAttr().Set(self.config.base_friction)
        physics_material.CreateDynamicFrictionAttr().Set(self.config.base_friction * 0.8)
        physics_material.CreateRestitutionAttr().Set(0.1)
        
        # Apply PhysX material properties
        physx_material = PhysxSchema.PhysxMaterialAPI.Apply(material_prim.GetPrim())
        physx_material.CreateFrictionCombineModeAttr().Set("average")
        
        self._generated_prims.append(self._physics_material_path)
    
    def _create_ground_plane(self, stage) -> None:
        """Create the base ground plane."""
        ground_path = f"{self.config.terrain_prim_path}/GroundPlane"
        
        ground_prim = _create_cube_prim(
            stage,
            ground_path,
            position=(0.0, 0.0, -0.025),
            scale=(
                self.config.grid_size[0] / 2,
                self.config.grid_size[1] / 2,
                0.025,
            ),
        )
        
        # Apply collision
        UsdPhysics.CollisionAPI.Apply(ground_prim)
        
        # Bind physics material
        self._bind_physics_material(stage, ground_path)
        
        self._generated_prims.append(ground_path)
    
    def _bind_physics_material(self, stage, prim_path: str) -> None:
        """Bind the shared physics material to a prim using collision API."""
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            return
        
        # Use PhysxSchema to bind physics material directly
        material_prim = stage.GetPrimAtPath(self._physics_material_path)
        if material_prim.IsValid():
            # Create collision API if needed and set material
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI.Apply(prim)
            
            # Bind material using relationship
            material_binding = UsdShade.MaterialBindingAPI.Apply(prim)
            material = UsdShade.Material(material_prim)
            # Use default binding (physics materials work through collision API)
            material_binding.Bind(material)
    
    # =========================================================================
    # TERRAIN GENERATION FUNCTIONS
    # =========================================================================
    
    def _generate_rough_terrain(self, difficulty: float) -> None:
        """Generate uneven terrain using height variations."""
        stage = _get_current_stage()
        
        num_cells_x = int(self.config.grid_size[0] / self.config.cell_size)
        num_cells_y = int(self.config.grid_size[1] / self.config.cell_size)
        
        height_variance = self.config.max_height_variance * difficulty
        height_map = self._generate_noise_heightmap(num_cells_x, num_cells_y, difficulty)
        
        rough_terrain_path = f"{self.config.terrain_prim_path}/RoughTerrain"
        _create_xform_prim(stage, rough_terrain_path)
        self._generated_prims.append(rough_terrain_path)
        
        start_x = -self.config.grid_size[0] / 2 + self.config.cell_size / 2
        start_y = -self.config.grid_size[1] / 2 + self.config.cell_size / 2
        
        for i in range(num_cells_x):
            for j in range(num_cells_y):
                x = start_x + i * self.config.cell_size
                y = start_y + j * self.config.cell_size
                height = height_map[i, j] * height_variance
                
                cell_path = f"{rough_terrain_path}/Cell_{i}_{j}"
                cell_height = 0.1 + abs(height)
                z_position = height / 2
                
                cell_prim = _create_cube_prim(
                    stage,
                    cell_path,
                    position=(x, y, z_position),
                    scale=(
                        self.config.cell_size / 2,
                        self.config.cell_size / 2,
                        cell_height / 2,
                    ),
                )
                
                UsdPhysics.CollisionAPI.Apply(cell_prim)
                self._bind_physics_material(stage, cell_path)
                self._generated_prims.append(cell_path)
    
    def _generate_noise_heightmap(
        self, 
        width: int, 
        height: int, 
        difficulty: float
    ) -> np.ndarray:
        """Generate a noise-based heightmap."""
        noise = np.random.randn(width, height)
        
        # Apply smoothing
        kernel_size = max(1, int(3 - 2 * difficulty))
        if kernel_size > 1:
            try:
                from scipy.ndimage import uniform_filter
                noise = uniform_filter(noise, size=kernel_size)
            except ImportError:
                pass  # Use unfiltered noise if scipy not available
        
        # Normalize
        noise = noise / (np.max(np.abs(noise)) + 1e-6)
        return noise
    
    def _generate_stairs(self, difficulty: float) -> None:
        """Generate stair-like terrain."""
        stage = _get_current_stage()
        
        step_height = self.config.max_stair_height * (0.2 + 0.8 * difficulty)
        step_depth = self.config.max_stair_depth * (1.5 - 0.5 * difficulty)
        
        num_steps = int(self.config.grid_size[0] / step_depth)
        
        stairs_path = f"{self.config.terrain_prim_path}/Stairs"
        _create_xform_prim(stage, stairs_path)
        self._generated_prims.append(stairs_path)
        
        start_x = -self.config.grid_size[0] / 2
        
        for i in range(num_steps):
            step_path = f"{stairs_path}/Step_{i}"
            
            if i < num_steps // 2:
                current_height = step_height * (i + 1)
            else:
                current_height = step_height * (num_steps - i)
            
            x = start_x + step_depth / 2 + i * step_depth
            z = current_height / 2
            
            step_prim = _create_cube_prim(
                stage,
                step_path,
                position=(x, 0.0, z),
                scale=(
                    step_depth / 2,
                    self.config.grid_size[1] / 2,
                    current_height / 2,
                ),
            )
            
            UsdPhysics.CollisionAPI.Apply(step_prim)
            self._bind_physics_material(stage, step_path)
            self._generated_prims.append(step_path)
    
    def _generate_discrete_obstacles(self, difficulty: float) -> None:
        """Generate discrete obstacles scattered on terrain."""
        stage = _get_current_stage()
        
        obstacles_path = f"{self.config.terrain_prim_path}/Obstacles"
        _create_xform_prim(stage, obstacles_path)
        self._generated_prims.append(obstacles_path)
        
        base_count = 5
        max_additional = 30
        num_obstacles = int(base_count + max_additional * difficulty)
        
        min_size = self.config.min_obstacle_size
        max_size = self.config.min_obstacle_size + (
            self.config.max_obstacle_size - self.config.min_obstacle_size
        ) * difficulty
        
        half_x = self.config.grid_size[0] / 2 * 0.8
        half_y = self.config.grid_size[1] / 2 * 0.8
        
        for i in range(num_obstacles):
            x = np.random.uniform(-half_x, half_x)
            y = np.random.uniform(-half_y, half_y)
            
            size = np.random.uniform(min_size, max_size)
            height = np.random.uniform(size * 0.5, size * 2.0)
            
            obstacle_type = random.choice(["Cube", "Cylinder"])
            obstacle_path = f"{obstacles_path}/Obstacle_{i}"
            
            if obstacle_type == "Cube":
                obstacle_prim = _create_cube_prim(
                    stage,
                    obstacle_path,
                    position=(x, y, height / 2),
                    scale=(size / 2, size / 2, height / 2),
                )
            else:
                obstacle_prim = _create_cylinder_prim(
                    stage,
                    obstacle_path,
                    position=(x, y, height / 2),
                    radius=size / 2,
                    height=height,
                )
            
            UsdPhysics.CollisionAPI.Apply(obstacle_prim)
            self._bind_physics_material(stage, obstacle_path)
            self._generated_prims.append(obstacle_path)
    
    def _generate_mixed_terrain(self, difficulty: float) -> None:
        """Generate mixed terrain combining multiple types."""
        stage = _get_current_stage()
        
        # Create spawn area (flat)
        self._create_spawn_area(stage)
        
        # Generate rough terrain in back-left region
        self._generate_rough_terrain_region(
            stage, 
            difficulty,
            x_range=(-self.config.grid_size[0]/2, -self.config.grid_size[0]/8),
            y_range=(-self.config.grid_size[1]/2, 0),
            region_name="RoughRegion"
        )
        
        # Generate stairs in back-right region
        self._generate_stairs_region(
            stage,
            difficulty,
            x_range=(self.config.grid_size[0]/8, self.config.grid_size[0]/2),
            y_range=(-self.config.grid_size[1]/2, 0),
            region_name="StairsRegion"
        )
        
        # Scatter obstacles in front region
        self._generate_obstacles_region(
            stage,
            difficulty,
            x_range=(-self.config.grid_size[0]/2 * 0.8, self.config.grid_size[0]/2 * 0.8),
            y_range=(self.config.grid_size[1]/8, self.config.grid_size[1]/2 * 0.8),
            region_name="ObstacleRegion"
        )
    
    def _create_spawn_area(self, stage) -> None:
        """Create flat spawn area at center."""
        spawn_path = f"{self.config.terrain_prim_path}/SpawnArea"
        spawn_size = 1.5
        
        spawn_prim = _create_cube_prim(
            stage,
            spawn_path,
            position=(0.0, 0.0, 0.01),
            scale=(spawn_size / 2, spawn_size / 2, 0.01),
        )
        
        UsdPhysics.CollisionAPI.Apply(spawn_prim)
        self._bind_physics_material(stage, spawn_path)
        self._generated_prims.append(spawn_path)
    
    def _generate_rough_terrain_region(
        self,
        stage,
        difficulty: float,
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        region_name: str,
    ) -> None:
        """Generate rough terrain within a region."""
        region_path = f"{self.config.terrain_prim_path}/{region_name}"
        _create_xform_prim(stage, region_path)
        self._generated_prims.append(region_path)
        
        region_width = x_range[1] - x_range[0]
        region_height = y_range[1] - y_range[0]
        
        num_cells_x = int(region_width / self.config.cell_size)
        num_cells_y = int(region_height / self.config.cell_size)
        
        height_variance = self.config.max_height_variance * difficulty
        height_map = self._generate_noise_heightmap(num_cells_x, num_cells_y, difficulty)
        
        start_x = x_range[0] + self.config.cell_size / 2
        start_y = y_range[0] + self.config.cell_size / 2
        
        for i in range(num_cells_x):
            for j in range(num_cells_y):
                x = start_x + i * self.config.cell_size
                y = start_y + j * self.config.cell_size
                height = height_map[i, j] * height_variance
                
                cell_path = f"{region_path}/Cell_{i}_{j}"
                cell_height = 0.1 + abs(height)
                
                cell_prim = _create_cube_prim(
                    stage,
                    cell_path,
                    position=(x, y, height / 2),
                    scale=(
                        self.config.cell_size / 2,
                        self.config.cell_size / 2,
                        cell_height / 2,
                    ),
                )
                
                UsdPhysics.CollisionAPI.Apply(cell_prim)
                self._bind_physics_material(stage, cell_path)
                self._generated_prims.append(cell_path)
    
    def _generate_stairs_region(
        self,
        stage,
        difficulty: float,
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        region_name: str,
    ) -> None:
        """Generate stairs within a region."""
        region_path = f"{self.config.terrain_prim_path}/{region_name}"
        _create_xform_prim(stage, region_path)
        self._generated_prims.append(region_path)
        
        step_height = self.config.max_stair_height * (0.2 + 0.8 * difficulty)
        step_depth = self.config.max_stair_depth * (1.5 - 0.5 * difficulty)
        
        region_length = x_range[1] - x_range[0]
        region_width = y_range[1] - y_range[0]
        num_steps = int(region_length / step_depth)
        
        start_x = x_range[0]
        
        for i in range(num_steps):
            step_path = f"{region_path}/Step_{i}"
            
            if i < num_steps // 2:
                current_height = step_height * (i + 1)
            else:
                current_height = step_height * (num_steps - i)
            
            x = start_x + step_depth / 2 + i * step_depth
            y_center = (y_range[0] + y_range[1]) / 2
            z = current_height / 2
            
            step_prim = _create_cube_prim(
                stage,
                step_path,
                position=(x, y_center, z),
                scale=(step_depth / 2, region_width / 2, current_height / 2),
            )
            
            UsdPhysics.CollisionAPI.Apply(step_prim)
            self._bind_physics_material(stage, step_path)
            self._generated_prims.append(step_path)
    
    def _generate_obstacles_region(
        self,
        stage,
        difficulty: float,
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        region_name: str,
    ) -> None:
        """Generate obstacles within a region."""
        region_path = f"{self.config.terrain_prim_path}/{region_name}"
        _create_xform_prim(stage, region_path)
        self._generated_prims.append(region_path)
        
        base_count = 3
        max_additional = 15
        num_obstacles = int(base_count + max_additional * difficulty)
        
        min_size = self.config.min_obstacle_size
        max_size = self.config.min_obstacle_size + (
            self.config.max_obstacle_size - self.config.min_obstacle_size
        ) * difficulty
        
        for i in range(num_obstacles):
            x = np.random.uniform(x_range[0], x_range[1])
            y = np.random.uniform(y_range[0], y_range[1])
            
            size = np.random.uniform(min_size, max_size)
            height = np.random.uniform(size * 0.5, size * 2.0)
            
            obstacle_type = random.choice(["Cube", "Cylinder"])
            obstacle_path = f"{region_path}/Obstacle_{i}"
            
            if obstacle_type == "Cube":
                obstacle_prim = _create_cube_prim(
                    stage,
                    obstacle_path,
                    position=(x, y, height / 2),
                    scale=(size / 2, size / 2, height / 2),
                )
            else:
                obstacle_prim = _create_cylinder_prim(
                    stage,
                    obstacle_path,
                    position=(x, y, height / 2),
                    radius=size / 2,
                    height=height,
                )
            
            UsdPhysics.CollisionAPI.Apply(obstacle_prim)
            self._bind_physics_material(stage, obstacle_path)
            self._generated_prims.append(obstacle_path)
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def get_random_spawn_position(self, margin: float = 0.5) -> Tuple[float, float, float]:
        """Get a random valid spawn position."""
        half_x = self.config.grid_size[0] / 2 - margin
        half_y = self.config.grid_size[1] / 2 - margin
        
        x = np.random.uniform(-half_x, half_x)
        y = np.random.uniform(-half_y, half_y)
        z = 0.2
        
        return (x, y, z)
    
    def set_random_seed(self, seed: int) -> None:
        """Set random seed for reproducibility."""
        self.config.random_seed = seed
        np.random.seed(seed)
        random.seed(seed)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_terrain_from_config(config_dict: Dict[str, Any]) -> TerrainGenerator:
    """Factory function to create TerrainGenerator from config dict."""
    config = TerrainConfig.from_dict(config_dict)
    return TerrainGenerator(config)


def quick_terrain(
    difficulty: float = 0.5,
    terrain_type: str = "mixed",
    grid_size: Tuple[float, float] = (10.0, 10.0),
) -> TerrainGenerator:
    """Quick utility to generate terrain with minimal setup."""
    config = TerrainConfig(grid_size=grid_size)
    generator = TerrainGenerator(config)
    generator.generate(difficulty, terrain_type)
    return generator
