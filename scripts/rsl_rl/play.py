# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
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
import time
import torch
import threading

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import RedRhex.tasks  # noqa: F401


# =============================================================================
# 鍵盤控制器 - 用 WASD + QE 控制機器人
# =============================================================================
class KeyboardController:
    """
    鍵盤控制器：讓你用鍵盤控制機器人的移動方向！
    
    控制方式：
    ┌─────────────────────────────────────┐
    │         Q (逆時針)   W (前進)   E (順時針)          │
    │                      ↑                             │
    │         A (左移) ←   ·   → D (右移)               │
    │                      ↓                             │
    │                    S (後退)                         │
    │                                                     │
    │         Space: 停止所有移動                         │
    │         ESC: 退出                                   │
    └─────────────────────────────────────┘
    
    組合按鍵：
    - W+D: 右前方移動
    - W+A: 左前方移動
    - W+E: 前進 + 順時針旋轉
    - 等等...
    """
    
    def __init__(self, velocity_scale: float = 1.0, angular_scale: float = 1.0):
        """
        初始化鍵盤控制器
        
        參數：
            velocity_scale: 線速度縮放（預設 1.0 m/s）
            angular_scale: 角速度縮放（預設 1.0 rad/s）
        """
        self.velocity_scale = velocity_scale
        self.angular_scale = angular_scale
        
        # 當前按下的按鍵狀態
        self.keys_pressed = {
            'w': False, 's': False,
            'a': False, 'd': False,
            'q': False, 'e': False,
        }
        
        # 目標速度命令
        self.target_vx = 0.0  # 前後速度
        self.target_vy = 0.0  # 左右速度
        self.target_wz = 0.0  # 旋轉速度
        
        # 控制執行緒
        self._running = False
        self._thread = None
        
    def start(self):
        """啟動鍵盤監聽（在背景執行緒中）"""
        try:
            import keyboard
            self._running = True
            
            # 註冊按鍵事件
            for key in self.keys_pressed.keys():
                keyboard.on_press_key(key, lambda e, k=key: self._on_key_press(k))
                keyboard.on_release_key(key, lambda e, k=key: self._on_key_release(k))
            
            # 空白鍵：停止
            keyboard.on_press_key('space', lambda e: self._stop_all())
            
            print("\n" + "="*60)
            print("🎮 鍵盤控制已啟用！")
            print("="*60)
            print("  W: 前進    S: 後退")
            print("  A: 左移    D: 右移")
            print("  Q: 逆時針  E: 順時針")
            print("  Space: 停止")
            print("="*60 + "\n")
            
        except ImportError:
            print("\n" + "="*60)
            print("⚠️  警告：無法載入 keyboard 模組")
            print("   請執行: pip install keyboard")
            print("   鍵盤控制功能將被停用")
            print("="*60 + "\n")
            self._running = False
    
    def _on_key_press(self, key: str):
        """按鍵按下事件"""
        self.keys_pressed[key] = True
        self._update_commands()
        
    def _on_key_release(self, key: str):
        """按鍵釋放事件"""
        self.keys_pressed[key] = False
        self._update_commands()
        
    def _stop_all(self):
        """停止所有移動"""
        for key in self.keys_pressed:
            self.keys_pressed[key] = False
        self._update_commands()
        print("[鍵盤] 停止所有移動")
        
    def _update_commands(self):
        """根據當前按鍵狀態更新速度命令"""
        # 前後速度 (vx)
        vx = 0.0
        if self.keys_pressed['w']:
            vx += 1.0
        if self.keys_pressed['s']:
            vx -= 1.0
            
        # 左右速度 (vy)
        # 注意：本體座標系中，正 Y 是左邊
        vy = 0.0
        if self.keys_pressed['a']:
            vy += 1.0  # 向左
        if self.keys_pressed['d']:
            vy -= 1.0  # 向右
            
        # 旋轉速度 (wz)
        # 正值 = 逆時針，負值 = 順時針
        wz = 0.0
        if self.keys_pressed['q']:
            wz += 1.0  # 逆時針
        if self.keys_pressed['e']:
            wz -= 1.0  # 順時針
            
        # 正規化對角移動（讓對角線速度不會超過直線速度）
        linear_speed = (vx**2 + vy**2)**0.5
        if linear_speed > 1.0:
            vx /= linear_speed
            vy /= linear_speed
            
        # 套用縮放
        self.target_vx = vx * self.velocity_scale
        self.target_vy = vy * self.velocity_scale
        self.target_wz = wz * self.angular_scale
        
        # 顯示當前命令（只在有變化時）
        if vx != 0 or vy != 0 or wz != 0:
            direction = []
            if vx > 0: direction.append("前進")
            if vx < 0: direction.append("後退")
            if vy > 0: direction.append("左移")
            if vy < 0: direction.append("右移")
            if wz > 0: direction.append("逆時針")
            if wz < 0: direction.append("順時針")
            print(f"[鍵盤] {'+'.join(direction)} | vx={self.target_vx:.2f}, vy={self.target_vy:.2f}, wz={self.target_wz:.2f}")
    
    def get_commands(self, num_envs: int, device: torch.device) -> torch.Tensor:
        """
        獲取當前的速度命令（用於覆蓋環境的 commands）
        
        參數：
            num_envs: 環境數量
            device: PyTorch 設備
            
        返回：
            commands: [num_envs, 3] 的速度命令張量
        """
        commands = torch.zeros(num_envs, 3, device=device)
        commands[:, 0] = self.target_vx
        commands[:, 1] = self.target_vy
        commands[:, 2] = self.target_wz
        return commands
    
    def stop(self):
        """停止鍵盤監聽"""
        self._running = False
        try:
            import keyboard
            keyboard.unhook_all()
        except:
            pass


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    # =========================================================================
    # 鍵盤控制初始化
    # =========================================================================
    # 從環境配置獲取速度範圍（用於縮放）
    unwrapped_env = env.unwrapped
    try:
        velocity_scale = getattr(unwrapped_env.cfg, 'vel_x_max', 1.0)
        angular_scale = getattr(unwrapped_env.cfg, 'ang_vel_max', 1.0)
    except:
        velocity_scale = 1.0
        angular_scale = 1.0
    
    # 創建鍵盤控制器
    keyboard_ctrl = KeyboardController(
        velocity_scale=velocity_scale,
        angular_scale=angular_scale
    )
    keyboard_ctrl.start()
    
    # 獲取設備和環境數量
    device = unwrapped_env.device
    num_envs = unwrapped_env.num_envs

    # reset environment
    obs = env.get_observations()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        
        # =====================================================================
        # 鍵盤控制：覆蓋環境的速度命令
        # =====================================================================
        if keyboard_ctrl._running:
            # 獲取鍵盤輸入的命令
            keyboard_commands = keyboard_ctrl.get_commands(num_envs, device)
            # 覆蓋環境的命令
            unwrapped_env.commands[:] = keyboard_commands
        
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, dones, _ = env.step(actions)
            # reset recurrent states for episodes that have terminated
            policy_nn.reset(dones)
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # 清理鍵盤控制器
    keyboard_ctrl.stop()
    
    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
