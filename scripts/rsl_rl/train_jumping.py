# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to train a JUMPING robot with RSL-RL.
Goal: Train the robot to jump as HIGH as possible!

Usage:
    cd IsaacLab
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train_jumping.py --num_envs 4096
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import math
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train a JUMPING robot with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=4096, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=1500, help="RL Policy training iterations.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
from datetime import datetime

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
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg, RslRlVecEnvWrapper

# Import mdp functions
import isaaclab.envs.mdp as mdp
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as loco_mdp

# Import robot config (using Go2 as example, change to your robot)
from isaaclab_assets import UNITREE_GO2_CFG

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


##############################################################################
# CUSTOM REWARD FUNCTIONS FOR JUMPING
##############################################################################

def reward_jump_height(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward for jumping high - the higher the base, the more reward."""
    asset = env.scene[asset_cfg.name]
    # Get current base height (z position)
    base_height = asset.data.root_pos_w[:, 2]
    # Reward = height above some baseline (e.g., 0.3m is standing height for Go2)
    standing_height = 0.34  # approximate standing height
    height_above_standing = base_height - standing_height
    # Only reward positive height gains
    return torch.clamp(height_above_standing, min=0.0)


def reward_vertical_velocity(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward for upward velocity - encourages jumping motion."""
    asset = env.scene[asset_cfg.name]
    # Get vertical velocity (z component)
    vertical_vel = asset.data.root_lin_vel_w[:, 2]
    # Only reward positive (upward) velocity
    return torch.clamp(vertical_vel, min=0.0)


def reward_feet_contact_force(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward for strong foot contact (for push-off)."""
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    # Get contact forces magnitude
    forces = contact_sensor.data.net_forces_w_history[:, 0, :, 2]  # z-component
    # Sum across all feet
    total_force = torch.sum(torch.abs(forces), dim=1)
    return total_force / 1000.0  # normalize


def penalty_horizontal_drift(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize horizontal movement - we want vertical jump, not running."""
    asset = env.scene[asset_cfg.name]
    # Get horizontal velocity magnitude
    horizontal_vel = torch.norm(asset.data.root_lin_vel_w[:, :2], dim=1)
    return horizontal_vel


##############################################################################
# SCENE CONFIGURATION
##############################################################################

@configclass
class JumpingSceneCfg(InteractiveSceneCfg):
    """Scene configuration for jumping training."""

    # Ground plane
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

    # Robot - Using Unitree Go2 (change this to your robot)
    robot: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # Contact sensor for feet
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )

    # Lights - 必須用 AssetBaseCfg 包起來，不能直接用 DomeLightCfg
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)),
    )


##############################################################################
# MDP CONFIGURATION
##############################################################################

@configclass
class ActionsCfg:
    """Action specifications for jumping."""
    joint_pos = loco_mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.5,
        use_default_offset=True
    )


@configclass
class ObservationsCfg:
    """Observation specifications for jumping."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy."""

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
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Event configuration for jumping."""

    # Reset robot to standing position
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
            "position_range": (1.0, 1.0),  # Start at default position
            "velocity_range": (0.0, 0.0),
        },
    )


@configclass
class RewardsCfg:
    """
    🎯 JUMPING REWARDS - This is where the magic happens!

    Positive rewards (encourage):
    - jump_height: Higher = more reward
    - vertical_velocity: Upward motion = reward
    - feet_air_time: Being in the air = reward

    Negative rewards (penalize):
    - horizontal_drift: Moving sideways = penalty
    - action_rate: Smooth actions preferred
    - orientation: Stay upright
    """

    # ===== MAIN JUMPING REWARDS (POSITIVE) =====

    # 🚀 Jump Height - THE MAIN GOAL!
    jump_height = RewTerm(
        func=reward_jump_height,
        weight=10.0,  # High weight - this is our main objective!
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # ⬆️ Vertical Velocity - Encourage upward motion
    vertical_velocity = RewTerm(
        func=reward_vertical_velocity,
        weight=5.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # 🦶 Feet Air Time - 移除！因為這個 reward 需要 velocity command
    #    跳躍任務不需要 command，用 jump_height + vertical_velocity 就夠了
    # feet_air_time = RewTerm(
    #     func=loco_mdp.feet_air_time,
    #     weight=2.0,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*foot"),
    #         "command_name": "base_velocity",  # 需要 command manager，但跳躍沒有
    #         "threshold": 0.1,
    #     },
    # )

    # ===== PENALTIES (NEGATIVE) =====

    # ❌ Horizontal Drift - Stay in place, don't run away
    horizontal_drift = RewTerm(
        func=penalty_horizontal_drift,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # 🔄 Action Rate - Smooth actions
    action_rate_l2 = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.01,
    )

    # 📐 Flat Orientation - Stay upright (don't flip)
    flat_orientation_l2 = RewTerm(
        func=mdp.flat_orientation_l2,
        weight=-1.0,
    )

    # ⚡ Joint Torques - Don't use excessive force
    joint_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-1.0e-5,
    )

    # 🦵 Joint Acceleration - Smooth joint motion
    joint_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-2.5e-7,
    )


@configclass
class TerminationsCfg:
    """Termination conditions for jumping."""

    # Episode timeout
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # Terminate if robot flips over (bad orientation)
    bad_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": math.pi / 3},  # 60 degrees
    )


##############################################################################
# ENVIRONMENT CONFIGURATION
##############################################################################

@configclass
class JumpingEnvCfg(ManagerBasedRLEnvCfg):
    """
    🐕 JUMPING ENVIRONMENT CONFIGURATION

    Goal: Train the robot to JUMP AS HIGH AS POSSIBLE!
    """

    # Scene
    scene: JumpingSceneCfg = JumpingSceneCfg(num_envs=4096, env_spacing=2.5)

    # MDP
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        """Post initialization."""
        # General settings
        self.decimation = 4
        self.episode_length_s = 10.0  # 10 seconds per episode

        # Simulation settings
        self.sim.dt = 0.005  # 200 Hz physics
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material

        # Viewer
        self.viewer.eye = (3.0, 3.0, 2.0)
        self.viewer.lookat = (0.0, 0.0, 0.5)


##############################################################################
# AGENT CONFIGURATION
##############################################################################

@configclass
class JumpingAgentCfg(RslRlOnPolicyRunnerCfg):
    """RSL-RL PPO agent configuration for jumping."""

    seed = 42
    device = "cuda:0"
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 100
    experiment_name = "jumping_go2"
    run_name = "jump_high"
    logger = "tensorboard"

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
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
# MAIN TRAINING FUNCTION
##############################################################################

def main():
    """Train the jumping robot!"""

    print("=" * 60)
    print("🐕 JUMPING ROBOT TRAINING 🚀")
    print("Goal: Jump as HIGH as possible!")
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
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations

    # Set seeds
    env_cfg.seed = agent_cfg.seed

    # Setup logging directory
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    print(f"[INFO] Logging to: {log_dir}")

    # Set log directory for environment
    env_cfg.log_dir = log_dir

    # Create environment
    env = ManagerBasedRLEnv(cfg=env_cfg)

    # Wrap for video recording
    if args_cli.video:
        import gymnasium as gym
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # Wrap for RSL-RL
    env = RslRlVecEnvWrapper(env)

    # Create runner
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)

    # Save configurations
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    print("\n" + "=" * 60)
    print("🎯 REWARD CONFIGURATION:")
    print("-" * 60)
    print(f"  jump_height:        weight = {env_cfg.rewards.jump_height.weight}")
    print(f"  vertical_velocity:  weight = {env_cfg.rewards.vertical_velocity.weight}")
    print(f"  horizontal_drift:   weight = {env_cfg.rewards.horizontal_drift.weight}")
    print(f"  flat_orientation:   weight = {env_cfg.rewards.flat_orientation_l2.weight}")
    print("=" * 60 + "\n")

    # Start training!
    print("🚀 Starting training...")
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # Cleanup
    env.close()
    print("\n✅ Training complete!")


if __name__ == "__main__":
    main()
    simulation_app.close()
