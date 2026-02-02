# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""
RedRhex Hexapod Robot Environment with RHex-style Wheg Locomotion.

=============================================================================
ARCHITECTURAL OVERVIEW
=============================================================================

This environment implements a reinforcement learning training setup for the 
RedRhex robot, a hexapod that uses "wheg" (wheel-leg hybrid) locomotion.

KEY CONCEPTS:
-------------
1. **Wheg Locomotion**: Unlike traditional walking robots, RHex uses C-shaped 
   legs that rotate continuously (like wheels) rather than stepping discretely.

2. **Alternating Tripod Gait**: The six legs are divided into two groups that 
   alternate support, ensuring the robot always has stable ground contact:
   - Tripod A: Legs 0, 3, 5 (right-front, left-front, left-rear)
   - Tripod B: Legs 1, 2, 4 (right-mid, right-rear, left-mid)

3. **Joint Types**:
   - Main Drive Joints (6): Continuous rotation for propulsion
   - ABAD Joints (6): Position-controlled for stability and steering
   - Damper Joints (6): Passive shock absorption

CONTROL ARCHITECTURE:
--------------------
- Main drives: Velocity control with base angular velocity ± RL adjustments
- ABAD joints: Position control within limited range (RL explores optimal usage)
- Dampers: Maintain initial position with high damping (passive)

COORDINATE CONVENTION:
---------------------
- X-axis: Forward direction
- Y-axis: Lateral (left positive)
- Z-axis: Vertical (up positive)
- Right legs rotate negatively, left legs rotate positively for forward motion

=============================================================================
"""

from __future__ import annotations

import math
import torch
from typing import Dict, List, Optional, Tuple
from collections.abc import Sequence
from dataclasses import dataclass

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import quat_apply_inverse, sample_uniform

from .redrhex_env_cfg import RedrhexEnvCfg


# =============================================================================
# CONSTANTS
# =============================================================================

# Numerical safety bounds to prevent physics explosion
_MAX_VELOCITY_CLAMP = 10.0  # rad/s for joint velocities
_MAX_OBSERVATION_CLAMP = 100.0  # Observation space bounds
_MAX_POSITION_BOUND = 50.0  # meters from origin before termination
_MAX_ROOT_VELOCITY = 30.0  # m/s before considering physics unstable


# =============================================================================
# REWARD CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class RewardWeights:
    """
    Centralized reward weight configuration.
    
    WHY: Separating weights from logic makes tuning easier and prevents 
    magic numbers scattered throughout the reward function. The frozen=True
    ensures these weights are immutable during training.
    
    DESIGN PHILOSOPHY:
    - Forward velocity has the highest weight (primary objective)
    - Stability penalties are mild (allow exploration)
    - Gait synchronization has low weight (let RL discover patterns)
    """
    # Primary locomotion rewards
    forward_velocity: float = 10.0
    velocity_tracking: float = 2.0
    
    # Leg rotation rewards
    correct_rotation_direction: float = 0.5  # per leg
    active_legs: float = 0.3  # per active leg
    minimum_velocity: float = 0.5
    mean_velocity: float = 0.3
    
    # Stability penalties (negative applied internally)
    orientation_penalty: float = 0.5
    height_penalty: float = 0.5
    vertical_velocity_penalty: float = 0.2
    angular_velocity_penalty: float = 0.1
    
    # Gait coordination (low weight - let RL explore)
    tripod_synchronization: float = 0.2
    gait_phase_sync: float = 0.1
    continuous_support: float = 0.2
    
    # Base survival reward
    alive_bonus: float = 0.2


# Global reward weights instance
REWARD_WEIGHTS = RewardWeights()


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def compute_phase_coherence(phases: torch.Tensor) -> torch.Tensor:
    """
    Compute the phase coherence (synchronization) of a group of oscillators.
    
    This measures how well-synchronized the phases are using the order parameter
    from Kuramoto oscillator theory. A value of 1.0 means perfect synchronization,
    0.0 means completely desynchronized.
    
    WHY THIS APPROACH:
    - Using sin/cos averaging handles the circular nature of phases correctly
    - Directly measuring phase differences would fail at the 0/2π boundary
    - This is the standard method in oscillator synchronization research
    
    Args:
        phases: Tensor of shape [N, num_oscillators] containing phase angles in radians
        
    Returns:
        Tensor of shape [N] with coherence values in range [0, 1]
    """
    sin_mean = torch.sin(phases).mean(dim=1)
    cos_mean = torch.cos(phases).mean(dim=1)
    return torch.sqrt(sin_mean ** 2 + cos_mean ** 2)


def compute_circular_mean(phases: torch.Tensor) -> torch.Tensor:
    """
    Compute the circular (angular) mean of phase angles.
    
    WHY: Regular arithmetic mean fails for circular quantities 
    (e.g., mean of 350° and 10° should be 0°, not 180°).
    
    Args:
        phases: Tensor of shape [N, num_phases] containing angles in radians
        
    Returns:
        Tensor of shape [N] with mean angles in radians
    """
    return torch.atan2(
        torch.sin(phases).mean(dim=1),
        torch.cos(phases).mean(dim=1)
    )


def safe_normalize(tensor: torch.Tensor, 
                   nan_value: float = 0.0,
                   clamp_min: float = -10.0,
                   clamp_max: float = 10.0) -> torch.Tensor:
    """
    Apply NaN protection and clamping to a tensor.
    
    WHY: Physics simulation can produce NaN/Inf values during unstable states.
    Propagating these through the network causes training to fail catastrophically.
    """
    tensor = torch.nan_to_num(tensor, nan=nan_value, posinf=clamp_max, neginf=clamp_min)
    return torch.clamp(tensor, min=clamp_min, max=clamp_max)


# =============================================================================
# MAIN ENVIRONMENT CLASS
# =============================================================================

class RedrhexEnv(DirectRLEnv):
    """
    RedRhex Hexapod Robot Environment for RHex-style Wheg Locomotion Training.
    
    This environment trains the robot to move forward using continuous leg rotation
    with an alternating tripod gait pattern. The RL agent controls:
    - Main drive velocities (how fast each leg rotates)
    - ABAD joint positions (lateral leg positioning)
    
    The reward structure prioritizes forward motion while encouraging stable,
    coordinated leg movements.
    """

    cfg: RedrhexEnvCfg

    # =========================================================================
    # INITIALIZATION
    # =========================================================================

    def __init__(self, cfg: RedrhexEnvCfg, render_mode: Optional[str] = None, **kwargs):
        """
        Initialize the RedRhex environment.
        
        The initialization order is critical:
        1. Parent class init (creates simulation infrastructure)
        2. Joint indices (maps joint names to tensor indices)
        3. Buffers (pre-allocates memory for efficiency)
        4. Commands (velocity targets for the robot)
        5. Gait (phase tracking for tripod coordination)
        """
        super().__init__(cfg, render_mode, **kwargs)

        # Setup in dependency order
        self._setup_joint_indices()
        self._setup_state_buffers()
        self._setup_reward_tracking()
        self._setup_velocity_commands()
        self._setup_gait_phase_tracking()

        # Print diagnostic information for debugging
        self._log_initialization_info()

    def _setup_joint_indices(self) -> None:
        """
        Map joint names to tensor indices for efficient access.
        
        WHY SEPARATE METHOD: Joint index lookup is done once at init rather than
        every step for performance. Storing as tensors enables vectorized operations.
        
        ROBUSTNESS: We validate that all expected joints exist and warn if not,
        rather than silently failing or crashing later.
        """
        joint_names = self.robot.data.joint_names
        
        # Helper function to find indices with validation
        def find_joint_indices(names: List[str], joint_type: str) -> torch.Tensor:
            indices = []
            for name in names:
                if name in joint_names:
                    indices.append(joint_names.index(name))
                else:
                    print(f"⚠️ WARNING: {joint_type} joint '{name}' not found in robot model")
            return torch.tensor(indices, device=self.device, dtype=torch.long)
        
        # Main drive joints (continuous rotation for propulsion)
        self._main_drive_indices = find_joint_indices(
            self.cfg.main_drive_joint_names, "Main Drive"
        )
        
        # ABAD joints (abduction/adduction for stability)
        self._abad_indices = find_joint_indices(
            self.cfg.abad_joint_names, "ABAD"
        )
        
        # Damper joints (passive shock absorption)
        self._damper_indices = find_joint_indices(
            self.cfg.damper_joint_names, "Damper"
        )
        
        # Tripod groupings (indices into the 6 main drive joints, not global joint indices)
        # WHY: Tripod groups reference the LOCAL ordering [0-5] of main drives
        self._tripod_a_indices = torch.tensor(
            self.cfg.tripod_a_leg_indices, device=self.device, dtype=torch.long
        )
        self._tripod_b_indices = torch.tensor(
            self.cfg.tripod_b_leg_indices, device=self.device, dtype=torch.long
        )
        
        # Direction multiplier for left/right leg rotation
        # WHY: Left and right legs must rotate in opposite directions for forward motion
        # Right legs (idx 0,1,2) → -1, Left legs (idx 3,4,5) → +1
        self._direction_multiplier = torch.tensor(
            self.cfg.leg_direction_multiplier, device=self.device
        ).unsqueeze(0)  # Shape: [1, 6] for broadcasting
        
        # Log configuration for debugging
        print(f"[Joint Indices] Main Drive: {self._main_drive_indices.tolist()}")
        print(f"[Joint Indices] ABAD: {self._abad_indices.tolist()}")
        print(f"[Joint Indices] Damper: {self._damper_indices.tolist()}")
        print(f"[Direction Multiplier] {self.cfg.leg_direction_multiplier}")
        print(f"[Tripod A] indices: {self._tripod_a_indices.tolist()}")
        print(f"[Tripod B] indices: {self._tripod_b_indices.tolist()}")

    def _setup_state_buffers(self) -> None:
        """
        Pre-allocate tensors for state tracking.
        
        WHY PRE-ALLOCATE: Creating tensors during the step loop causes memory
        allocation overhead and fragmentation. Pre-allocating once is faster.
        """
        # Joint state buffers (cloned to avoid aliasing issues)
        self.joint_pos = self.robot.data.joint_pos.clone()
        self.joint_vel = self.robot.data.joint_vel.clone()
        
        # Action buffers for tracking action history
        self.actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self.last_actions = torch.zeros_like(self.actions)
        self.last_main_drive_vel = torch.zeros(self.num_envs, 6, device=self.device)

        # Damper joint target positions (maintain initial configuration)
        # WHY: Dampers should stay at their initial angles to preserve leg geometry
        damper_init_angles = [
            self.cfg.robot_cfg.init_state.joint_pos.get(joint_name, 0.0)
            for joint_name in self.cfg.damper_joint_names
        ]
        self._damper_target_positions = torch.tensor(
            damper_init_angles, device=self.device
        ).unsqueeze(0)
        print(f"[Damper Initial Angles] {[f'{a*180/math.pi:.1f}°' for a in damper_init_angles]}")

        # Base state buffers
        self.base_lin_vel = torch.zeros(self.num_envs, 3, device=self.device)
        self.base_ang_vel = torch.zeros(self.num_envs, 3, device=self.device)
        self.projected_gravity = torch.zeros(self.num_envs, 3, device=self.device)

        # Reference gravity for orientation comparison
        self._compute_reference_gravity()

    def _compute_reference_gravity(self) -> None:
        """
        Compute the expected gravity projection when robot is properly oriented.
        
        WHY: By comparing current gravity projection to reference, we can detect
        tilting and flipping without complex angle calculations.
        """
        init_rot = self.cfg.robot_cfg.init_state.rot
        init_quat = torch.tensor(
            [init_rot[0], init_rot[1], init_rot[2], init_rot[3]],
            device=self.device
        ).unsqueeze(0).expand(self.num_envs, 4)
        
        gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=self.device).expand(self.num_envs, 3)
        self.reference_projected_gravity = quat_apply_inverse(init_quat, gravity_vec)

    def _setup_reward_tracking(self) -> None:
        """
        Initialize reward component tracking for TensorBoard visualization.
        
        WHY TRACK COMPONENTS: Monitoring individual reward components helps
        diagnose training issues (e.g., if forward_vel is high but orientation
        penalty is killing total reward, we know the robot is moving but unstable).
        """
        reward_keys = [
            # Core locomotion rewards
            "rew_alive", "rew_forward_vel", "rew_vel_tracking",
            # Gait rewards
            "rew_gait_sync", "rew_rotation_dir", "rew_correct_dir",
            "rew_all_legs", "rew_tripod_sync", "rew_mean_vel",
            "rew_min_vel", "rew_continuous_support", "rew_smooth_rotation",
            # Stability penalties
            "rew_orientation", "rew_base_height", "rew_lin_vel_z", "rew_ang_vel_xy",
            # ABAD control
            "rew_abad_action", "rew_abad_stability", "rew_action_rate",
            # Diagnostic metrics (not rewards, but useful to track)
            "diag_forward_vel", "diag_base_height", "diag_tilt",
            "diag_drive_vel_mean", "diag_rotating_legs", "diag_min_leg_vel",
        ]
        
        self.episode_sums = {
            key: torch.zeros(self.num_envs, device=self.device) 
            for key in reward_keys
        }

    def _setup_velocity_commands(self) -> None:
        """
        Initialize velocity command buffer.
        
        Commands are [vx, vy, omega_z] representing target velocities:
        - vx: Forward/backward velocity
        - vy: Lateral velocity (currently limited range)
        - omega_z: Yaw rotation rate (currently limited range)
        """
        self.commands = torch.zeros(self.num_envs, 3, device=self.device)

    def _setup_gait_phase_tracking(self) -> None:
        """
        Initialize gait phase tracking for tripod coordination.
        
        The gait phase is a global clock that increments continuously.
        Each tripod group has a phase offset (0 for A, π for B) to create
        the alternating pattern.
        
        WHY TRACK PHASE: The reward function uses phase to encourage proper
        tripod synchronization and detect if legs are properly alternating.
        """
        self.gait_phase = torch.zeros(self.num_envs, device=self.device)
        
        # Phase offsets for each leg (indexed by main drive order)
        self.leg_phase_offsets = torch.zeros(6, device=self.device)
        self.leg_phase_offsets[self._tripod_a_indices] = 0.0
        self.leg_phase_offsets[self._tripod_b_indices] = math.pi

    def _setup_scene(self) -> None:
        """
        Configure the simulation scene with robot, terrain, and lighting.
        
        This is called by the parent class during initialization.
        """
        # Create robot articulation
        self.robot = Articulation(self.cfg.robot_cfg)
        self.scene.articulations["robot"] = self.robot

        # Setup terrain
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # Clone environments for parallel simulation
        self.scene.clone_environments(copy_from_source=False)

        # CPU-specific collision filtering
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # Add scene lighting
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _log_initialization_info(self) -> None:
        """
        Print diagnostic information about the environment configuration.
        
        WHY: This helps verify that the environment is configured correctly
        and makes debugging easier when things go wrong.
        """
        print("\n" + "=" * 70)
        print("🤖 RedRhex RHex-style Wheg Locomotion Environment")
        print("=" * 70)
        
        # Timing configuration
        control_freq = 1 / (self.cfg.sim.dt * self.cfg.decimation)
        print(f"⚙️  Control frequency: {control_freq:.1f} Hz")
        print(f"⚙️  Base gait frequency: {self.cfg.base_gait_frequency} Hz")
        print(f"⚙️  Base angular velocity: {self.cfg.base_gait_angular_vel:.2f} rad/s")
        
        # Joint configuration
        print(f"\n📐 Joint Configuration:")
        print(f"   Main drives: {self._main_drive_indices.tolist()}")
        print(f"   ABAD joints: {self._abad_indices.tolist()}")
        print(f"   Damper joints: {self._damper_indices.tolist()}")
        print(f"   Direction multiplier: {self.cfg.leg_direction_multiplier}")
        
        # Tripod groups
        print(f"\n🦿 Tripod Groups (alternating support):")
        print(f"   Tripod A (phase 0): indices {self._tripod_a_indices.tolist()}")
        print(f"   Tripod B (phase π): indices {self._tripod_b_indices.tolist()}")
        
        # Action space
        print(f"\n🎮 Action Space ({self.cfg.action_space} dimensions):")
        print(f"   [0:6]  Main drive velocity adjustment (±{self.cfg.main_drive_vel_scale} rad/s)")
        print(f"   [6:12] ABAD position targets (±{self.cfg.abad_pos_scale} rad)")
        
        print("=" * 70 + "\n")

    # =========================================================================
    # SIMULATION STEP METHODS
    # =========================================================================

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """
        Process actions before physics simulation step.
        
        WHY CLAMP: RL algorithms can output extreme values during exploration.
        Clamping prevents physically impossible commands from destabilizing simulation.
        """
        self.last_actions = self.actions.clone()
        self.actions = actions.clone().clamp(-1.0, 1.0)

    def _apply_action(self) -> None:
        """
        Convert RL actions to joint commands and apply to robot.
        
        Action mapping:
        - actions[0:6]: Main drive velocity adjustments (added to base velocity)
        - actions[6:12]: ABAD joint position targets
        
        CRITICAL: Left and right legs must rotate in opposite directions!
        The direction_multiplier handles this automatically.
        """
        # === Main Drive Joints: Velocity Control ===
        drive_actions = self.actions[:, :6]
        
        # Compute target velocity: (base + adjustment) * direction
        # The direction_multiplier ensures left/right legs rotate correctly
        target_drive_vel = (
            self.cfg.base_gait_angular_vel + 
            drive_actions * self.cfg.main_drive_vel_scale
        ) * self._direction_multiplier
        
        # Safety clamp to prevent physics explosion
        target_drive_vel = torch.clamp(target_drive_vel, -_MAX_VELOCITY_CLAMP, _MAX_VELOCITY_CLAMP)
        
        self.robot.set_joint_velocity_target(
            target_drive_vel, joint_ids=self._main_drive_indices
        )
        
        # === ABAD Joints: Position Control ===
        abad_actions = self.actions[:, 6:12]
        target_abad_pos = abad_actions * self.cfg.abad_pos_scale
        target_abad_pos = torch.clamp(target_abad_pos, -0.5, 0.5)
        
        self.robot.set_joint_position_target(
            target_abad_pos, joint_ids=self._abad_indices
        )
        
        # === Damper Joints: Maintain Initial Position ===
        # WHY: Without explicit targets, the actuator would try to straighten
        # the joints, destroying the leg geometry. We hold them at init pose.
        self.robot.set_joint_position_target(
            self._damper_target_positions.expand(self.num_envs, -1),
            joint_ids=self._damper_indices
        )

    # =========================================================================
    # OBSERVATION COMPUTATION
    # =========================================================================

    def _get_observations(self) -> Dict[str, torch.Tensor]:
        """
        Compute observation vector for the policy network.
        
        Observation components (56 total):
        - Base linear velocity (3): Movement speed in body frame
        - Base angular velocity (3): Rotation rates in body frame  
        - Projected gravity (3): Orientation indicator
        - Main drive position sin/cos (12): Circular encoding of leg angles
        - Main drive velocity (6): Normalized leg rotation speeds
        - ABAD position (6): Normalized lateral leg positions
        - ABAD velocity (6): Lateral leg movement rates
        - Commands (3): Target velocities
        - Gait phase sin/cos (2): Global timing signal
        - Last actions (12): Action history for smoothness
        
        WHY SIN/COS ENCODING: Leg positions are circular (wrap around at 2π).
        Direct angle values would have a discontinuity; sin/cos is continuous.
        """
        self._update_state()

        # Get joint states for controlled joints
        main_drive_pos = self.joint_pos[:, self._main_drive_indices]
        main_drive_vel = self.joint_vel[:, self._main_drive_indices]
        abad_pos = self.joint_pos[:, self._abad_indices]
        abad_vel = self.joint_vel[:, self._abad_indices]

        # Build observation vector
        obs = torch.cat([
            # Base state (9)
            self.base_lin_vel,
            self.base_ang_vel,
            self.projected_gravity,
            # Main drive state with circular encoding (18)
            torch.sin(main_drive_pos),
            torch.cos(main_drive_pos),
            main_drive_vel / self.cfg.base_gait_angular_vel,  # Normalize
            # ABAD state (12)
            abad_pos / self.cfg.abad_pos_scale,  # Normalize
            abad_vel,
            # Commands (3)
            self.commands,
            # Gait timing (2)
            torch.sin(self.gait_phase).unsqueeze(-1),
            torch.cos(self.gait_phase).unsqueeze(-1),
            # Action history (12)
            self.last_actions,
        ], dim=-1)

        # Add observation noise for robustness
        if self.cfg.add_noise:
            obs = obs + torch.randn_like(obs) * 0.01 * self.cfg.noise_level

        # Safety: prevent NaN/Inf from propagating to network
        obs = safe_normalize(obs, clamp_max=_MAX_OBSERVATION_CLAMP, clamp_min=-_MAX_OBSERVATION_CLAMP)

        return {"policy": obs}

    def _update_state(self) -> None:
        """
        Update internal state buffers from simulation data.
        
        This method synchronizes our cached state with the physics simulation
        and advances the gait phase clock.
        """
        # Joint states with NaN protection
        self.joint_pos = torch.nan_to_num(self.robot.data.joint_pos.clone(), nan=0.0)
        self.joint_vel = torch.nan_to_num(self.robot.data.joint_vel.clone(), nan=0.0)

        # Transform base velocities to body frame
        root_quat = self.robot.data.root_quat_w
        
        self.base_lin_vel = safe_normalize(
            quat_apply_inverse(root_quat, self.robot.data.root_lin_vel_w)
        )
        self.base_ang_vel = safe_normalize(
            quat_apply_inverse(root_quat, self.robot.data.root_ang_vel_w)
        )

        # Compute projected gravity (orientation indicator)
        gravity_world = torch.tensor([0.0, 0.0, -1.0], device=self.device).expand(self.num_envs, 3)
        self.projected_gravity = torch.nan_to_num(
            quat_apply_inverse(root_quat, gravity_world), nan=0.0
        )

        # Advance gait phase clock
        dt = self.cfg.sim.dt * self.cfg.decimation
        phase_increment = 2 * math.pi * self.cfg.base_gait_frequency * dt
        self.gait_phase = (self.gait_phase + phase_increment) % (2 * math.pi)

    # =========================================================================
    # REWARD COMPUTATION
    # =========================================================================

    def _get_rewards(self) -> torch.Tensor:
        """
        Compute the reward signal for the RL agent.
        
        REWARD DESIGN PHILOSOPHY:
        -------------------------
        The reward structure follows a hierarchy of importance:
        
        1. PRIMARY: Forward velocity (highest weight)
           - This is what we ultimately want the robot to do
           
        2. SECONDARY: Leg activation and rotation
           - Necessary mechanism for achieving forward motion
           
        3. TERTIARY: Stability (mild penalties)
           - Don't punish too hard, let the robot explore
           
        4. AUXILIARY: Gait coordination (low weight)
           - Encourage tripod pattern but don't force it
           
        WHY THIS STRUCTURE:
        - Heavy forward reward creates strong gradient toward goal
        - Mild stability penalties prevent only catastrophic failures
        - Low gait weights let RL discover its own coordination strategy
        """
        total_reward = torch.zeros(self.num_envs, device=self.device)

        # Compute state features used by multiple reward components
        leg_state = self._compute_leg_state_features()
        
        # === Primary Rewards: Locomotion ===
        total_reward += self._compute_forward_velocity_reward()
        total_reward += self._compute_velocity_tracking_reward()
        
        # === Secondary Rewards: Leg Rotation ===
        rotation_rewards = self._compute_leg_rotation_rewards(leg_state)
        for reward in rotation_rewards.values():
            total_reward += reward
        
        # === Tertiary: Stability Penalties ===
        stability_penalties = self._compute_stability_penalties()
        for penalty in stability_penalties.values():
            total_reward += penalty
        
        # === Auxiliary: Gait Coordination ===
        gait_rewards = self._compute_gait_coordination_rewards(leg_state)
        for reward in gait_rewards.values():
            total_reward += reward
        
        # === Survival Bonus ===
        rew_alive = torch.ones(self.num_envs, device=self.device) * REWARD_WEIGHTS.alive_bonus
        total_reward += rew_alive

        # Safety: prevent NaN/Inf rewards from corrupting training
        total_reward = torch.nan_to_num(total_reward, nan=0.0, posinf=10.0, neginf=-10.0)

        # Update tracking for TensorBoard visualization
        self._update_reward_tracking(rew_alive)
        
        # Store for next step comparison
        self.last_main_drive_vel = self.joint_vel[:, self._main_drive_indices].clone()

        return total_reward

    def _compute_leg_state_features(self) -> Dict[str, torch.Tensor]:
        """
        Extract features about leg state used by multiple reward components.
        
        Returns dict with:
        - effective_vel: Velocity with direction correction [N, 6]
        - vel_magnitude: Absolute velocities [N, 6]
        - mean_vel: Mean velocity across legs [N]
        - min_vel: Minimum leg velocity [N]
        - num_active: Count of legs moving > threshold [N]
        - leg_phase: Current phase of each leg [N, 6]
        """
        main_drive_vel = self.joint_vel[:, self._main_drive_indices]
        main_drive_pos = self.joint_pos[:, self._main_drive_indices]
        
        # Apply direction multiplier so positive = forward propulsion
        effective_vel = main_drive_vel * self._direction_multiplier
        vel_magnitude = torch.abs(effective_vel)
        
        return {
            "effective_vel": effective_vel,
            "vel_magnitude": vel_magnitude,
            "mean_vel": vel_magnitude.mean(dim=1),
            "min_vel": vel_magnitude.min(dim=1).values,
            "num_active": (vel_magnitude > 0.3).float().sum(dim=1),
            "leg_phase": torch.remainder(main_drive_pos * self._direction_multiplier, 2 * math.pi),
        }

    def _compute_forward_velocity_reward(self) -> torch.Tensor:
        """
        Reward for moving forward in the X direction.
        
        Simple and direct: positive X velocity = reward, negative = penalty.
        This creates a strong gradient toward forward motion.
        """
        forward_vel = self.base_lin_vel[:, 0]
        reward = forward_vel * REWARD_WEIGHTS.forward_velocity
        
        # Track for diagnostics
        self.episode_sums["rew_forward_vel"] += reward
        self.episode_sums["diag_forward_vel"] += forward_vel
        
        return reward

    def _compute_velocity_tracking_reward(self) -> torch.Tensor:
        """
        Reward for matching the commanded velocity.
        
        Uses exponential decay with velocity error - perfect tracking gives
        full reward, larger errors decay exponentially toward zero.
        """
        forward_vel = self.base_lin_vel[:, 0]
        target_vel = self.commands[:, 0]
        vel_error = torch.abs(forward_vel - target_vel)
        
        reward = torch.exp(-vel_error * 2.0) * REWARD_WEIGHTS.velocity_tracking
        self.episode_sums["rew_vel_tracking"] += reward
        
        return reward

    def _compute_leg_rotation_rewards(self, leg_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute rewards for proper leg rotation behavior.
        
        Components:
        1. Correct direction: Legs rotating to propel forward
        2. All legs active: Preventing any leg from "going on strike"
        3. Minimum velocity: Even the slowest leg should be moving
        4. Mean velocity: Encourage good average rotation speed
        """
        effective_vel = leg_state["effective_vel"]
        vel_magnitude = leg_state["vel_magnitude"]
        
        # Correct direction: each leg rotating the right way
        correct_direction = (effective_vel > 0.5).float()
        rew_rotation_dir = correct_direction.sum(dim=1) * REWARD_WEIGHTS.correct_rotation_direction
        
        # All legs active
        rew_all_legs = leg_state["num_active"] * REWARD_WEIGHTS.active_legs
        
        # Minimum velocity (prevent lazy legs)
        rew_min_vel = torch.clamp(leg_state["min_vel"], max=3.0) * REWARD_WEIGHTS.minimum_velocity
        
        # Mean velocity (encourage good overall speed)
        rew_mean_vel = torch.clamp(leg_state["mean_vel"], max=5.0) * REWARD_WEIGHTS.mean_velocity
        
        # Update tracking
        self.episode_sums["rew_rotation_dir"] += rew_rotation_dir
        self.episode_sums["rew_all_legs"] += rew_all_legs
        self.episode_sums["rew_min_vel"] += rew_min_vel
        self.episode_sums["rew_mean_vel"] += rew_mean_vel
        self.episode_sums["rew_correct_dir"] += rew_mean_vel  # Legacy compatibility
        self.episode_sums["diag_drive_vel_mean"] += leg_state["mean_vel"]
        self.episode_sums["diag_rotating_legs"] += leg_state["num_active"]
        self.episode_sums["diag_min_leg_vel"] += leg_state["min_vel"]
        
        return {
            "rotation_dir": rew_rotation_dir,
            "all_legs": rew_all_legs,
            "min_vel": rew_min_vel,
            "mean_vel": rew_mean_vel,
        }

    def _compute_stability_penalties(self) -> Dict[str, torch.Tensor]:
        """
        Compute mild penalties for unstable behavior.
        
        DESIGN PRINCIPLE: Penalties are intentionally mild to allow exploration.
        We only want to discourage catastrophic instability, not prevent all
        dynamic motion which might be necessary for efficient locomotion.
        """
        # Orientation penalty (tilt from upright)
        grav_xy = self.projected_gravity[:, :2]
        tilt = torch.norm(grav_xy, dim=1)
        rew_orientation = -tilt * REWARD_WEIGHTS.orientation_penalty
        
        # Height maintenance
        base_height = self.robot.data.root_pos_w[:, 2]
        target_height = 0.12  # meters
        height_error = torch.abs(base_height - target_height)
        rew_base_height = -height_error * REWARD_WEIGHTS.height_penalty
        
        # Vertical velocity penalty (discourage bouncing)
        z_vel = self.base_lin_vel[:, 2]
        rew_lin_vel_z = -torch.abs(z_vel) * REWARD_WEIGHTS.vertical_velocity_penalty
        
        # Angular velocity penalty in pitch/roll (discourage wobbling)
        ang_vel_xy = self.base_ang_vel[:, :2]
        rew_ang_vel_xy = -torch.norm(ang_vel_xy, dim=1) * REWARD_WEIGHTS.angular_velocity_penalty
        
        # Update tracking
        self.episode_sums["rew_orientation"] += rew_orientation
        self.episode_sums["rew_base_height"] += rew_base_height
        self.episode_sums["rew_lin_vel_z"] += rew_lin_vel_z
        self.episode_sums["rew_ang_vel_xy"] += rew_ang_vel_xy
        self.episode_sums["diag_base_height"] += base_height
        self.episode_sums["diag_tilt"] += tilt
        
        return {
            "orientation": rew_orientation,
            "base_height": rew_base_height,
            "lin_vel_z": rew_lin_vel_z,
            "ang_vel_xy": rew_ang_vel_xy,
        }

    def _compute_gait_coordination_rewards(self, leg_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute rewards for proper tripod gait coordination.
        
        These have LOW WEIGHT intentionally - we want to encourage the tripod
        pattern but not force it. The RL agent might discover variations that
        work better for this specific robot.
        
        Components:
        1. Tripod sync: Legs within each tripod group moving together
        2. Inter-tripod phase: 180° phase difference between groups
        3. Continuous support: At least one tripod always has ground contact
        """
        leg_phase = leg_state["leg_phase"]
        
        # Extract phases for each tripod group
        phase_a = leg_phase[:, self._tripod_a_indices]
        phase_b = leg_phase[:, self._tripod_b_indices]
        
        # Intra-tripod synchronization (legs in same group together)
        coherence_a = compute_phase_coherence(phase_a)
        coherence_b = compute_phase_coherence(phase_b)
        rew_tripod_sync = (coherence_a + coherence_b) * REWARD_WEIGHTS.tripod_synchronization
        
        # Inter-tripod phase difference (should be ~π)
        mean_phase_a = compute_circular_mean(phase_a)
        mean_phase_b = compute_circular_mean(phase_b)
        phase_diff = torch.abs(mean_phase_a - mean_phase_b)
        phase_diff = torch.min(phase_diff, 2 * math.pi - phase_diff)  # Handle wraparound
        phase_diff_error = torch.abs(phase_diff - math.pi)
        rew_gait_sync = torch.exp(-phase_diff_error) * REWARD_WEIGHTS.gait_phase_sync
        
        # Continuous support (stance phase is 0 to π)
        in_stance = leg_phase < math.pi
        stance_a = in_stance[:, self._tripod_a_indices].float().sum(dim=1)
        stance_b = in_stance[:, self._tripod_b_indices].float().sum(dim=1)
        has_support = ((stance_a >= 1) | (stance_b >= 1)).float()
        rew_continuous_support = has_support * REWARD_WEIGHTS.continuous_support
        
        # Update tracking
        self.episode_sums["rew_tripod_sync"] += rew_tripod_sync
        self.episode_sums["rew_gait_sync"] += rew_gait_sync
        self.episode_sums["rew_continuous_support"] += rew_continuous_support
        
        return {
            "tripod_sync": rew_tripod_sync,
            "gait_sync": rew_gait_sync,
            "continuous_support": rew_continuous_support,
        }

    def _update_reward_tracking(self, rew_alive: torch.Tensor) -> None:
        """
        Update episode sum tracking for rewards not tracked in component methods.
        
        This ensures all reward components are logged for TensorBoard visualization.
        """
        self.episode_sums["rew_alive"] += rew_alive
        
        # Placeholder zeros for unused reward components (TensorBoard compatibility)
        zeros = torch.zeros(self.num_envs, device=self.device)
        self.episode_sums["rew_abad_action"] += zeros
        self.episode_sums["rew_abad_stability"] += zeros
        self.episode_sums["rew_action_rate"] += zeros
        self.episode_sums["rew_smooth_rotation"] += zeros

    # =========================================================================
    # TERMINATION CONDITIONS
    # =========================================================================

    def _get_dones(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Determine which environments should be reset.
        
        Returns:
            terminated: Episodes that ended due to failure conditions
            time_out: Episodes that ended due to reaching max length
        
        DESIGN PHILOSOPHY:
        ------------------
        Termination conditions are intentionally PERMISSIVE to allow exploration.
        We only terminate when something is truly broken:
        - Physics simulation has failed (NaN/Inf values)
        - Robot has left the simulation bounds
        - Robot has completely flipped over
        
        We DO NOT terminate for:
        - Minor instability or wobbling
        - Slow forward progress
        - Suboptimal gait patterns
        
        This allows the RL agent to recover from mistakes and learn robustness.
        """
        # Time-based termination
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # Failure-based termination (accumulate conditions)
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        root_pos = self.robot.data.root_pos_w
        root_vel = self.robot.data.root_lin_vel_w
        
        # === Physics Validity Checks ===
        # If simulation produces NaN/Inf, something has gone catastrophically wrong
        pos_invalid = torch.any(torch.isnan(root_pos) | torch.isinf(root_pos), dim=1)
        vel_invalid = torch.any(torch.isnan(root_vel) | torch.isinf(root_vel), dim=1)
        terminated = terminated | pos_invalid | vel_invalid
        
        # === Boundary Checks ===
        # Robot wandered too far from origin (simulation bounds)
        pos_too_far = torch.any(torch.abs(root_pos[:, :2]) > _MAX_POSITION_BOUND, dim=1)
        terminated = terminated | pos_too_far
        
        # Velocity too high (physics explosion)
        vel_too_fast = torch.any(torch.abs(root_vel) > _MAX_ROOT_VELOCITY, dim=1)
        terminated = terminated | vel_too_fast

        # === Orientation Check ===
        # Only terminate if completely flipped (> ~60° from upright)
        # projected_gravity.z: -1 = upright, +1 = inverted
        flipped_over = self.projected_gravity[:, 2] > 0.5
        terminated = terminated | flipped_over

        # === Height Check ===
        # Only terminate at extreme heights (underground or flying)
        base_height = root_pos[:, 2]
        too_low = base_height < 0.01   # Below ground
        too_high = base_height > 1.0    # Flying
        terminated = terminated | too_low | too_high

        return terminated, time_out

    # =========================================================================
    # ENVIRONMENT RESET
    # =========================================================================

    def _reset_idx(self, env_ids: Optional[Sequence[int]]) -> None:
        """
        Reset specified environments to initial state.
        
        This is called when environments terminate or time out. It:
        1. Logs episode statistics to TensorBoard
        2. Resets robot pose and joint states
        3. Samples new velocity commands
        4. Clears internal buffers
        
        Args:
            env_ids: Indices of environments to reset, or None for all
        """
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        num_reset = len(env_ids)
        
        # === Log Episode Statistics ===
        self._log_episode_statistics(env_ids)

        # === Reset Robot State ===
        self._reset_robot_state(env_ids, num_reset)

        # === Reset Internal Buffers ===
        self._reset_buffers(env_ids, num_reset)

        # === Sample New Commands ===
        self._resample_commands(env_ids)

        # === Clear Reward Tracking ===
        for key in self.episode_sums:
            self.episode_sums[key][env_ids] = 0.0

    def _log_episode_statistics(self, env_ids: Sequence[int]) -> None:
        """
        Log episode reward statistics to TensorBoard via extras dict.
        
        The RSL-RL logger reads from extras["log"] and writes to TensorBoard.
        Using "/" in keys creates hierarchical grouping in TensorBoard.
        """
        extras = {}
        
        # Log average reward per second for each component
        for key, sums in self.episode_sums.items():
            avg_per_second = torch.mean(sums[env_ids]) / self.max_episode_length_s
            extras[f"Episode_Reward/{key}"] = avg_per_second
        
        # Log termination statistics
        extras["Episode_Termination/terminated"] = torch.count_nonzero(
            self.reset_terminated[env_ids]
        ).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(
            self.reset_time_outs[env_ids]
        ).item()
        
        self.extras["log"] = extras

    def _reset_robot_state(self, env_ids: Sequence[int], num_reset: int) -> None:
        """
        Reset robot pose and joint states to initial configuration.
        
        Includes small random perturbations to initial state for robustness.
        """
        # Get default joint positions from config
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = torch.zeros((num_reset, self.robot.num_joints), device=self.device)
        
        # Debug: print initial positions on first reset
        if not hasattr(self, '_printed_init_pos'):
            self._printed_init_pos = True
            self._log_initial_joint_positions(joint_pos)

        # Add small random perturbation for exploration robustness
        joint_pos += sample_uniform(-0.02, 0.02, joint_pos.shape, device=self.device)

        # Reset root state (position and orientation)
        default_root_state = self.robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        
        # Small position variation
        default_root_state[:, 0] += sample_uniform(-0.1, 0.1, (num_reset,), device=self.device)
        default_root_state[:, 1] += sample_uniform(-0.1, 0.1, (num_reset,), device=self.device)

        # Write to simulation
        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # Update cached state
        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel

    def _log_initial_joint_positions(self, joint_pos: torch.Tensor) -> None:
        """Log initial joint positions for debugging on first reset."""
        print("\n[DEBUG] Initial joint positions from config:")
        for i, name in enumerate(self.robot.data.joint_names):
            pos_rad = joint_pos[0, i].item()
            pos_deg = pos_rad * 180 / math.pi
            print(f"  {name}: {pos_rad:.3f} rad ({pos_deg:.1f}°)")
        print("")

    def _reset_buffers(self, env_ids: Sequence[int], num_reset: int) -> None:
        """Reset internal tracking buffers for reset environments."""
        self.actions[env_ids] = 0.0
        self.last_actions[env_ids] = 0.0
        self.last_main_drive_vel[env_ids] = 0.0
        
        # Randomize gait phase for variety in initial conditions
        self.gait_phase[env_ids] = sample_uniform(
            0, 2 * math.pi, (num_reset,), device=self.device
        )

    def _resample_commands(self, env_ids: torch.Tensor) -> None:
        """
        Sample new velocity commands for specified environments.
        
        Commands are sampled uniformly from configured ranges:
        - Linear X: Forward/backward velocity
        - Linear Y: Lateral velocity (typically small range)
        - Angular Z: Yaw rate (typically small range)
        """
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
