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
# 鍵盤控制器 - 切換模式（按一下維持，直到按下一個新按鍵）
# =============================================================================
class KeyboardController:
    """
    鍵盤控制器：按一下就維持那個命令，直到按下另一個按鍵！
    
    控制方式（單鍵切換）：
    ┌─────────────────────────────────────────────────────┐
    │                                                     │
    │     T (左前)     W (前進)     R (右前)              │
    │                    ↑                                │
    │     A (左移)   ←  ·  →   D (右移)                  │
    │                    ↓                                │
    │     F (左後)     S (後退)     G (右後)              │
    │                                                     │
    │     Q: 逆時針旋轉    E: 順時針旋轉                  │
    │     Space: 停止                                     │
    │                                                     │
    └─────────────────────────────────────────────────────┘
    
    ★ 切換模式：按一下就維持，直到按下另一個按鍵 ★
    """
    
    def __init__(self, velocity_scale: float = 0.4, lateral_scale: float = 0.3, angular_scale: float = 0.8):
        """
        初始化鍵盤控制器
        
        參數：
            velocity_scale: 前後線速度縮放（預設 0.4 m/s）
            lateral_scale: 左右線速度縮放（預設 0.3 m/s）
            angular_scale: 角速度縮放（預設 0.8 rad/s）
        """
        self.velocity_scale = velocity_scale
        self.lateral_scale = lateral_scale
        self.angular_scale = angular_scale
        
        # ★★★ 預定義的命令（切換模式用）★★★
        # 格式: (vx, vy, wz, 名稱)
        # 對角線命令使用更大的側移分量，讓方向更明顯（約 53 度）
        self.command_presets = {
            'w': (velocity_scale, 0.0, 0.0, "前進"),
            's': (-velocity_scale, 0.0, 0.0, "後退"),
            'a': (0.0, lateral_scale, 0.0, "左移"),
            'd': (0.0, -lateral_scale, 0.0, "右移"),
            'q': (0.0, 0.0, angular_scale, "逆時針"),
            'e': (0.0, 0.0, -angular_scale, "順時針"),
            # 對角線：vx=0.25, vy=±0.33 → 角度 ≈ 53° (更明顯的斜向)
            'r': (velocity_scale * 0.6, -lateral_scale * 1.1, 0.0, "右前"),
            't': (velocity_scale * 0.6, lateral_scale * 1.1, 0.0, "左前"),
            'g': (-velocity_scale * 0.6, -lateral_scale * 1.1, 0.0, "右後"),
            'f': (-velocity_scale * 0.6, lateral_scale * 1.1, 0.0, "左後"),
            'space': (0.0, 0.0, 0.0, "停止"),
        }
        
        # 當前命令（切換模式：維持直到下一個按鍵）
        self.target_vx = 0.0
        self.target_vy = 0.0
        self.target_wz = 0.0
        self.current_command_name = "停止"
        
        # 上一幀的按鍵狀態（用於檢測按鍵「剛按下」的瞬間）
        self._last_key_states = {}
        
        # 控制執行緒
        self._running = False
        self._use_carb = False
        
    def start(self):
        """啟動鍵盤監聽"""
        try:
            import carb.input
            import omni.appwindow
            self._use_carb = True
            self._running = True
            print("\n" + "="*60)
            print("🎮 鍵盤控制已啟用！【切換模式】")
            print("="*60)
            print("  按一下就維持那個命令，直到按下另一個按鍵")
            print("")
            print("  ┌─────────────────────────────────────┐")
            print("  │  T(左前)   W(前進)   R(右前)        │")
            print("  │  A(左移)     ·       D(右移)        │")
            print("  │  F(左後)   S(後退)   G(右後)        │")
            print("  │                                     │")
            print("  │  Q: 逆時針    E: 順時針             │")
            print("  │  Space: 停止                        │")
            print("  └─────────────────────────────────────┘")
            print("")
            print(f"  速度: 前進={self.velocity_scale:.2f} m/s, 側移={self.lateral_scale:.2f} m/s")
            print(f"        旋轉={self.angular_scale:.2f} rad/s")
            print("="*60)
            print("  ⚠️  請確保 Isaac Sim 視窗是焦點視窗！")
            print("="*60 + "\n")
            return
        except Exception as e:
            print(f"[鍵盤] carb input 不可用: {e}")
            self._running = False
    
    def update_from_carb(self):
        """從 carb input 讀取按鍵狀態（每幀調用）- 切換模式"""
        if not self._use_carb:
            return
            
        try:
            import carb.input
            import omni.appwindow
            
            app_window = omni.appwindow.get_default_app_window()
            keyboard = app_window.get_keyboard()
            input_iface = carb.input.acquire_input_interface()
            
            # 按鍵對應
            key_map = {
                'w': carb.input.KeyboardInput.W,
                's': carb.input.KeyboardInput.S,
                'a': carb.input.KeyboardInput.A,
                'd': carb.input.KeyboardInput.D,
                'q': carb.input.KeyboardInput.Q,
                'e': carb.input.KeyboardInput.E,
                'r': carb.input.KeyboardInput.R,
                't': carb.input.KeyboardInput.T,
                'f': carb.input.KeyboardInput.F,
                'g': carb.input.KeyboardInput.G,
                'space': carb.input.KeyboardInput.SPACE,
            }
            
            # 檢測「剛按下」的按鍵（上升沿觸發）
            for key_name, key_code in key_map.items():
                value = input_iface.get_keyboard_value(keyboard, key_code)
                is_pressed = value > 0.5
                was_pressed = self._last_key_states.get(key_name, False)
                
                # 檢測上升沿（從未按到按下）
                if is_pressed and not was_pressed:
                    self._on_key_triggered(key_name)
                
                self._last_key_states[key_name] = is_pressed
                
        except Exception as e:
            if not hasattr(self, '_carb_error_printed'):
                print(f"[鍵盤] carb input 錯誤: {e}")
                self._carb_error_printed = True
    
    def _on_key_triggered(self, key: str):
        """按鍵觸發事件（切換模式：設置並維持命令）"""
        if key in self.command_presets:
            vx, vy, wz, name = self.command_presets[key]
            self.target_vx = vx
            self.target_vy = vy
            self.target_wz = wz
            self.current_command_name = name
            print(f"[命令切換] → {name} (vx={vx:.2f}, vy={vy:.2f}, wz={wz:.2f})")
    
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
    
    # 使用與 discrete_directions 中相同的速度值
    # 前進: 0.4 m/s, 側移: 0.3 m/s, 旋轉: 0.8 rad/s
    velocity_scale = 0.4   # 與環境中的前進速度一致
    lateral_scale = 0.3    # 與環境中的側移速度一致
    angular_scale = 0.8    # 與環境中的旋轉速度一致
    
    # 創建鍵盤控制器（切換模式）
    keyboard_ctrl = KeyboardController(
        velocity_scale=velocity_scale,
        lateral_scale=lateral_scale,
        angular_scale=angular_scale
    )
    keyboard_ctrl.start()
    
    # ★★★ 啟用外部控制模式，禁用環境的自動命令重採樣 ★★★
    if hasattr(unwrapped_env, 'external_control'):
        unwrapped_env.external_control = True
        print("[INFO] 已啟用外部控制模式，環境不會自動切換命令")
    
    # 獲取設備和環境數量
    device = unwrapped_env.device
    num_envs = unwrapped_env.num_envs

    # reset environment
    obs = env.get_observations()
    timestep = 0
    frame_count = 0
    
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        frame_count += 1
        
        # =====================================================================
        # 鍵盤控制（切換模式）
        # =====================================================================
        if keyboard_ctrl._running:
            keyboard_ctrl.update_from_carb()
            keyboard_commands = keyboard_ctrl.get_commands(num_envs, device)
            if hasattr(unwrapped_env, 'commands'):
                unwrapped_env.commands[:] = keyboard_commands
        
        # run everything in inference mode
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            policy_nn.reset(dones)
        
        # 再次設置命令（確保 reset 後也正確）
        if keyboard_ctrl._running and hasattr(unwrapped_env, 'commands'):
            unwrapped_env.commands[:] = keyboard_commands
        
        if args_cli.video:
            timestep += 1
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    keyboard_ctrl.stop()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
