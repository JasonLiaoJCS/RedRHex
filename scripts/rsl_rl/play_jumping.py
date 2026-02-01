# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to PLAY a trained JUMPING robot with RSL-RL.

Usage:
    cd IsaacLab
    python scripts/reinforcement_learning/rsl_rl/play_jumping.py --checkpoint "logs/rsl_rl/jumping_go2/2025-12-26_10-42-02_jump_high/model_1499.pt" --num_envs 1
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import math
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a trained JUMPING robot with RSL-RL.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import torch

from rsl_rl.runners import OnPolicyRunner

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg, RslRlVecEnvWrapper

# Import mdp functions
import isaaclab.envs.mdp as mdp

# Import robot config (using Go2 as example)
from isaaclab_assets import UNITREE_GO2_CFG

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


##############################################################################
# CUSTOM REWARD FUNCTIONS FOR JUMPING (same as training)
##############################################################################

def reward_jump_height(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward for jumping high - the higher the base, the more reward."""
    asset = env.scene[asset_cfg.name]
    base_height = asset.data.root_pos_w[:, 2]
    standing_height = 0.34
    return torch.clamp(base_height - standing_height, min=0.0)


def reward_vertical_velocity(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward for upward velocity."""
    asset = env.scene[asset_cfg.name]
    vertical_vel = asset.data.root_lin_vel_w[:, 2]
    return torch.clamp(vertical_vel, min=0.0)


def penalty_horizontal_drift(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalty for horizontal movement."""
    asset = env.scene[asset_cfg.name]
    horizontal_vel = asset.data.root_lin_vel_w[:, :2]
    return torch.sum(torch.square(horizontal_vel), dim=1)


##############################################################################
# SCENE CONFIGURATION (same as training)
##############################################################################

@configclass
class JumpingSceneCfg(InteractiveSceneCfg):
    """Scene configuration for jumping."""

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    robot: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )


##############################################################################
# ACTION CONFIGURATION
##############################################################################

@configclass
class ActionsCfg:
    """Action specification for jumping."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.25,
        use_default_offset=True,
    )


##############################################################################
# OBSERVATION CONFIGURATION
##############################################################################

@configclass
class ObservationsCfg:
    """Observation specification for jumping."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy - MUST match training!"""

        # Base state
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))

        # Joint state
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))

        # Previous action
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False  # No noise during play
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


##############################################################################
# EVENT CONFIGURATION
##############################################################################

@configclass
class EventCfg:
    """Event configuration for jumping."""

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "yaw": (-0.1, 0.1)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
        },
    )


##############################################################################
# REWARDS CONFIGURATION
##############################################################################

@configclass
class RewardsCfg:
    """Reward configuration for jumping."""

    jump_height = RewTerm(
        func=reward_jump_height,
        weight=10.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    vertical_velocity = RewTerm(
        func=reward_vertical_velocity,
        weight=5.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    horizontal_drift = RewTerm(
        func=penalty_horizontal_drift,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    action_rate_l2 = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.01,
    )

    flat_orientation_l2 = RewTerm(
        func=mdp.flat_orientation_l2,
        weight=-1.0,
    )

    joint_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-1.0e-5,
    )

    joint_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-2.5e-7,
    )


##############################################################################
# TERMINATION CONFIGURATION
##############################################################################

@configclass
class TerminationsCfg:
    """Termination conditions for jumping."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    bad_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": math.pi / 3},
    )


##############################################################################
# ENVIRONMENT CONFIGURATION
##############################################################################

@configclass
class JumpingEnvCfg(ManagerBasedRLEnvCfg):
    """Jumping environment configuration."""

    scene: JumpingSceneCfg = JumpingSceneCfg(num_envs=1, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 10.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        )


##############################################################################
# AGENT CONFIGURATION (must match training!)
##############################################################################

@configclass
class JumpingAgentCfg(RslRlOnPolicyRunnerCfg):
    """Agent configuration for jumping - MUST MATCH TRAINING CONFIG!"""

    seed = 42
    device = "cuda:0"
    num_steps_per_env = 24
    max_iterations = 1500
    experiment_name = "jumping_go2"
    run_name = "jump_high"

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],  # MUST MATCH CHECKPOINT!
        critic_hidden_dims=[512, 256, 128],  # MUST MATCH CHECKPOINT!
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


##############################################################################
# MAIN PLAY FUNCTION
##############################################################################

def main():
    """Play the trained jumping robot!"""

    print("=" * 60)
    print("🐕 JUMPING ROBOT PLAYBACK 🎬")
    print("=" * 60)

    # Create environment config
    env_cfg = JumpingEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device else "cuda:0"

    # Create agent config
    agent_cfg = JumpingAgentCfg()
    agent_cfg.device = env_cfg.sim.device

    if args_cli.seed is not None:
        agent_cfg.seed = args_cli.seed
        env_cfg.seed = args_cli.seed

    # Create environment
    env = ManagerBasedRLEnv(cfg=env_cfg)

    # Wrap for RSL-RL
    env = RslRlVecEnvWrapper(env)

    # Create runner (no log_dir needed for play)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)

    # Load checkpoint
    checkpoint_path = os.path.abspath(args_cli.checkpoint)
    print(f"[INFO] Loading checkpoint: {checkpoint_path}")
    runner.load(checkpoint_path)

    print("=" * 60)
    print("🎮 Press Ctrl+C to exit")
    print("=" * 60)

    # Get policy for inference
    policy = runner.get_inference_policy(device=agent_cfg.device)

    # Reset environment and get initial observations (cross-version compatible)
    out = env.get_observations()
    if isinstance(out, tuple):
        obs, obs_info = out
    else:
        obs, obs_info = out, {}

    # IsaacLab sometimes returns obs as dict (e.g., {"policy": tensor, ...})
    if isinstance(obs, dict):
        obs = obs.get("policy", next(iter(obs.values())))

    # Run simulation loop
    while simulation_app.is_running():
        with torch.inference_mode():
            # Get action from policy
            actions = policy(obs)

            # Step environment
            step_out = env.step(actions)
            # Handle different return formats
            if len(step_out) == 5:
                obs, _, _, _, _ = step_out
            else:
                obs, _, _, _ = step_out

            # Handle dict observations
            if isinstance(obs, dict):
                obs = obs.get("policy", next(iter(obs.values())))

    # Cleanup
    env.close()
    print("\n✅ Playback complete!")


if __name__ == "__main__":
    main()
    simulation_app.close()
