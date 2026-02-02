# Copyright (c) 2024, RedRhex Project
# SPDX-License-Identifier: BSD-3-Clause

"""
Terrain Configuration for RHex Hexapod Robot Training

This module provides a dedicated configuration dataclass for procedural terrain
generation with curriculum learning support. The configuration controls terrain
type, difficulty, and physical properties while maintaining backward compatibility
with flat ground setups.

Usage:
    from terrain_cfg import TerrainCfg, TerrainType
    
    # Default flat terrain (backward compatible)
    terrain_cfg = TerrainCfg()
    
    # Enable procedural terrain with difficulty scaling
    terrain_cfg = TerrainCfg(
        terrain_type=TerrainType.ROUGH,
        difficulty_scale=0.5,
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Tuple, Optional


class TerrainType(Enum):
    """
    Enumeration of available terrain types.
    
    Each terrain type provides different challenges for the robot:
    
    FLAT: 
        Simple plane ground. Default for backward compatibility.
        Use for initial training or debugging.
        
    ROUGH:
        Uneven terrain with height variations based on noise.
        Trains the robot to handle small bumps and irregular surfaces.
        
    STAIRS:
        Step-like terrain for climbing and descending.
        Trains leg coordination and force control.
        
    OBSTACLES:
        Discrete obstacles (boxes, cylinders) scattered on the ground.
        Trains collision avoidance and navigation.
        
    MIXED:
        Combination of rough, stairs, and obstacles.
        Provides diverse training scenarios in a single environment.
    """
    FLAT = auto()       # Simple plane (default, backward compatible)
    ROUGH = auto()      # Noise-based height variations
    STAIRS = auto()     # Step-like terrain
    OBSTACLES = auto()  # Discrete obstacles (boxes, cylinders)
    MIXED = auto()      # Combination of all terrain types


@dataclass
class TerrainCfg:
    """
    Configuration dataclass for procedural terrain generation.
    
    This class encapsulates all parameters needed to generate training terrains
    for the RHex hexapod robot. It follows the Separation of Concerns principle:
    TerrainCfg holds configuration data, TerrainGenerator processes it.
    
    Key Design Decisions:
    1. Default terrain_type is FLAT for backward compatibility
    2. All parameters are typed for IDE support and validation
    3. Docstrings explain what each parameter controls
    
    Example Usage:
        >>> # Basic usage with defaults (flat terrain)
        >>> cfg = TerrainCfg()
        
        >>> # Configure rough terrain for curriculum learning
        >>> cfg = TerrainCfg(
        ...     terrain_type=TerrainType.ROUGH,
        ...     difficulty_scale=0.5,
        ...     horizontal_scale=0.4,
        ... )
        
        >>> # Access properties
        >>> print(f"Using {cfg.terrain_type.name} terrain at difficulty {cfg.difficulty_scale}")
    
    Attributes:
        terrain_type: Type of terrain to generate (FLAT, ROUGH, STAIRS, etc.)
        difficulty_scale: Curriculum difficulty from 0.0 (easy) to 1.0 (hard)
        horizontal_scale: Distance in meters between discrete grid points
        vertical_scale: Maximum height variation in meters at difficulty=1.0
        friction: Surface friction coefficient for robot-terrain interaction
        grid_size: Total terrain size in meters (width, length)
        spawn_height_offset: Additional height offset for robot spawn position
        debug_visualize: Toggle visualization markers for debugging
        random_seed: Seed for reproducible terrain generation
    """
    
    # ==========================================================================
    # Core Terrain Settings
    # ==========================================================================
    
    terrain_type: TerrainType = TerrainType.FLAT
    """
    Type of terrain to generate.
    
    IMPORTANT: Default is FLAT to maintain backward compatibility.
    When FLAT is selected, no procedural terrain is generated and the
    standard Isaac Lab plane ground is used instead.
    
    Options:
        - FLAT: Simple plane ground (default)
        - ROUGH: Noise-based height variations
        - STAIRS: Step-like terrain
        - OBSTACLES: Scattered boxes and cylinders
        - MIXED: Combination of all types
    """
    
    difficulty_scale: float = 0.0
    """
    Curriculum difficulty level from 0.0 (easy) to 1.0 (hard).
    
    This parameter scales all terrain features:
    - 0.0: Minimal variations (nearly flat)
    - 0.5: Moderate challenge (typical training)
    - 1.0: Maximum difficulty (expert-level)
    
    During curriculum learning, gradually increase this value
    as the agent's performance improves.
    """
    
    # ==========================================================================
    # Geometric Parameters
    # ==========================================================================
    
    horizontal_scale: float = 0.5
    """
    Distance in meters between discrete grid points.
    
    Controls the resolution of the terrain mesh:
    - Smaller values: Finer detail, more compute
    - Larger values: Coarser terrain, faster generation
    
    Recommended range: 0.1 to 1.0 meters
    For RHex robot (~40cm leg span), 0.3-0.5m is appropriate.
    """
    
    vertical_scale: float = 0.15
    """
    Maximum height variation in meters at difficulty_scale=1.0.
    
    Actual height variation = vertical_scale * difficulty_scale
    
    For rough terrain:
        - Sets the maximum bump/dip height
    For stairs:
        - Sets the maximum step height
    For obstacles:
        - Influences obstacle height range
        
    Recommended range: 0.05 to 0.3 meters for RHex.
    """
    
    grid_size: Tuple[float, float] = (10.0, 10.0)
    """
    Total terrain size in meters (width, length).
    
    This defines the playable area where terrain features are generated.
    Should be large enough for the robot to explore but not so large
    that it causes memory issues.
    
    Recommended: 8-15 meters for single-robot training.
    For multi-env training, each environment gets its own terrain.
    """
    
    # ==========================================================================
    # Physics Parameters
    # ==========================================================================
    
    friction: float = 0.8
    """
    Surface friction coefficient for terrain surfaces.
    
    Controls how much grip the robot has on the terrain:
    - Low (0.3-0.5): Slippery surfaces (ice-like)
    - Medium (0.6-0.8): Normal surfaces (default)
    - High (0.9-1.2): High-grip surfaces (rubber)
    
    This value is applied to both static and dynamic friction,
    with dynamic friction being 80% of this value.
    """
    
    restitution: float = 0.1
    """
    Surface restitution (bounciness) coefficient.
    
    Controls how much energy is preserved in collisions:
    - 0.0: No bounce (absorbs all energy)
    - 0.5: Medium bounce
    - 1.0: Perfect elastic bounce
    
    For terrain, low values (0.0-0.2) are recommended to
    prevent unrealistic bouncing.
    """
    
    # ==========================================================================
    # Spawn Configuration
    # ==========================================================================
    
    spawn_height_offset: float = 0.1
    """
    Additional height offset for robot spawn position in meters.
    
    The robot's spawn height is automatically calculated as:
        spawn_z = terrain_height_at_origin + spawn_height_offset
    
    This ensures the robot doesn't spawn inside the ground.
    Increase if the robot starts partially buried.
    """
    
    spawn_area_size: float = 1.5
    """
    Size of the flat spawn area in meters.
    
    A flat area is always created at the origin to ensure
    stable robot initialization regardless of terrain type.
    """
    
    # ==========================================================================
    # Stairs-Specific Parameters
    # ==========================================================================
    
    max_stair_height: float = 0.12
    """
    Maximum stair step height in meters at difficulty=1.0.
    
    Actual step height = max_stair_height * (0.2 + 0.8 * difficulty)
    This ensures even at difficulty=0, stairs have minimal height.
    """
    
    max_stair_depth: float = 0.3
    """
    Maximum stair step depth (tread length) in meters.
    
    Step depth decreases with difficulty:
        depth = max_stair_depth * (1.5 - 0.5 * difficulty)
    
    Deeper steps are easier to climb.
    """
    
    # ==========================================================================
    # Obstacles-Specific Parameters
    # ==========================================================================
    
    obstacle_density: float = 0.3
    """
    Obstacle density coefficient at difficulty=1.0.
    
    Controls how many obstacles are placed per unit area.
    Actual count scales with difficulty.
    """
    
    min_obstacle_size: float = 0.05
    """
    Minimum obstacle dimension in meters.
    
    Obstacles are generated with random sizes between
    min_obstacle_size and max_obstacle_size (scaled by difficulty).
    """
    
    max_obstacle_size: float = 0.2
    """
    Maximum obstacle dimension in meters at difficulty=1.0.
    
    Actual max size = min_obstacle_size + (max_obstacle_size - min_obstacle_size) * difficulty
    """
    
    # ==========================================================================
    # Debug and Visualization
    # ==========================================================================
    
    debug_visualize: bool = False
    """
    Toggle visualization markers for debugging.
    
    When enabled, draws:
    - Red lines around active terrain boundaries
    - Spawn point markers
    - Terrain height map visualization (if available)
    
    Note: Only works when simulation has GUI enabled.
    Disable in headless training for performance.
    """
    
    debug_vis_color: Tuple[float, float, float] = (1.0, 0.0, 0.0)
    """
    Color for debug visualization lines (RGB, 0.0-1.0).
    Default is red (1.0, 0.0, 0.0).
    """
    
    # ==========================================================================
    # Reproducibility
    # ==========================================================================
    
    random_seed: Optional[int] = None
    """
    Random seed for reproducible terrain generation.
    
    When set, the same seed will generate identical terrain
    for the same difficulty level. Useful for:
    - Debugging specific terrain configurations
    - Reproducible experiments
    - Comparing different policies on identical terrain
    
    Set to None for random terrain each time.
    """
    
    # ==========================================================================
    # USD Paths (Internal)
    # ==========================================================================
    
    terrain_prim_path: str = "/World/ProceduralTerrain"
    """
    USD path where procedural terrain prims are created.
    
    This path is used internally by TerrainGenerator.
    Should not conflict with other scene elements.
    """
    
    # ==========================================================================
    # Methods
    # ==========================================================================
    
    def is_procedural(self) -> bool:
        """
        Check if procedural terrain generation is needed.
        
        Returns:
            True if terrain_type is not FLAT, False otherwise.
        """
        return self.terrain_type != TerrainType.FLAT
    
    def get_terrain_type_string(self) -> str:
        """
        Get the terrain type as a lowercase string.
        
        Used for compatibility with TerrainGenerator's generate() method.
        
        Returns:
            Terrain type string: "rough", "stairs", "obstacles", "mixed", or "flat"
        """
        return self.terrain_type.name.lower()
    
    def get_spawn_height(self, terrain_height_at_origin: float = 0.0) -> float:
        """
        Calculate the appropriate spawn height for the robot.
        
        Args:
            terrain_height_at_origin: Height of terrain at (0, 0) position
            
        Returns:
            Recommended spawn height in meters
        """
        return terrain_height_at_origin + self.spawn_height_offset
    
    def validate(self) -> bool:
        """
        Validate configuration parameters.
        
        Returns:
            True if all parameters are valid
            
        Raises:
            ValueError: If any parameter is out of valid range
        """
        if not 0.0 <= self.difficulty_scale <= 1.0:
            raise ValueError(f"difficulty_scale must be in [0.0, 1.0], got {self.difficulty_scale}")
        
        if self.horizontal_scale <= 0:
            raise ValueError(f"horizontal_scale must be positive, got {self.horizontal_scale}")
        
        if self.vertical_scale < 0:
            raise ValueError(f"vertical_scale must be non-negative, got {self.vertical_scale}")
        
        if not 0.0 <= self.friction <= 2.0:
            raise ValueError(f"friction should be in [0.0, 2.0], got {self.friction}")
        
        if self.grid_size[0] <= 0 or self.grid_size[1] <= 0:
            raise ValueError(f"grid_size must be positive, got {self.grid_size}")
        
        return True
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self.validate()


# =============================================================================
# Convenience Factory Functions
# =============================================================================

def create_flat_terrain_cfg() -> TerrainCfg:
    """
    Create a flat terrain configuration (default, backward compatible).
    
    Returns:
        TerrainCfg with FLAT terrain type
    """
    return TerrainCfg(terrain_type=TerrainType.FLAT)


def create_rough_terrain_cfg(difficulty: float = 0.5) -> TerrainCfg:
    """
    Create a rough terrain configuration.
    
    Args:
        difficulty: Difficulty level from 0.0 to 1.0
        
    Returns:
        TerrainCfg configured for rough terrain
    """
    return TerrainCfg(
        terrain_type=TerrainType.ROUGH,
        difficulty_scale=difficulty,
    )


def create_stairs_terrain_cfg(difficulty: float = 0.5) -> TerrainCfg:
    """
    Create a stairs terrain configuration.
    
    Args:
        difficulty: Difficulty level from 0.0 to 1.0
        
    Returns:
        TerrainCfg configured for stairs terrain
    """
    return TerrainCfg(
        terrain_type=TerrainType.STAIRS,
        difficulty_scale=difficulty,
    )


def create_mixed_terrain_cfg(difficulty: float = 0.5) -> TerrainCfg:
    """
    Create a mixed terrain configuration.
    
    Args:
        difficulty: Difficulty level from 0.0 to 1.0
        
    Returns:
        TerrainCfg configured for mixed terrain
    """
    return TerrainCfg(
        terrain_type=TerrainType.MIXED,
        difficulty_scale=difficulty,
    )


def create_curriculum_terrain_cfg(
    terrain_type: TerrainType = TerrainType.ROUGH,
    initial_difficulty: float = 0.0,
) -> TerrainCfg:
    """
    Create a terrain configuration for curriculum learning.
    
    Starts with low difficulty that can be gradually increased
    during training.
    
    Args:
        terrain_type: Type of terrain to use
        initial_difficulty: Starting difficulty level
        
    Returns:
        TerrainCfg configured for curriculum learning
    """
    return TerrainCfg(
        terrain_type=terrain_type,
        difficulty_scale=initial_difficulty,
        debug_visualize=True,  # Enable for initial debugging
    )
