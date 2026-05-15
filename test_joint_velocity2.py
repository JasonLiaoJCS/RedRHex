"""
测试关节速度控制 - 使用现有环境
"""
import torch
import argparse
import sys

# 添加 source 路径
sys.path.insert(0, '/home/jasonliao/RedRhex/RedRhex/source/RedRhex')

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils

# 导入环境
import RedRhex  # 这会注册任务
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
import gymnasium as gym

def main():
    # 创建环境
    env_cfg = parse_env_cfg(
        "Template-Redrhex-Direct-v0",
        device="cuda:0",
        num_envs=1,
    )
    env = gym.make("Template-Redrhex-Direct-v0", cfg=env_cfg)
    
    obs, info = env.reset()
    
    print(f"\n关节名称: {env.unwrapped.robot.data.joint_names}")
    print(f"主驱动索引: {env.unwrapped._main_drive_indices.tolist()}")
    
    # 设置固定动作：所有主驱动速度设为最大
    action = torch.zeros(1, 12, device="cuda:0")
    action[:, :6] = 1.0  # 主驱动设为最大速度
    
    print(f"\n开始测试，动作 = {action[0].cpu().numpy()}")
    print("-" * 60)
    
    for step in range(100):
        obs, rew, terminated, truncated, info = env.step(action)
        
        # 获取关节速度
        joint_vel = env.unwrapped.robot.data.joint_vel[0]
        main_drive_vel = joint_vel[env.unwrapped._main_drive_indices]
        
        if step % 10 == 0:
            print(f"Step {step:3d}: 主驱动速度 = {main_drive_vel.cpu().numpy()}")
            print(f"          平均速度 = {main_drive_vel.abs().mean().item():.2f} rad/s")
            print(f"          基础高度 = {env.unwrapped.robot.data.root_pos_w[0, 2].item():.4f} m")
    
    env.close()
    print("\n测试完成!")
    simulation_app.close()

if __name__ == "__main__":
    main()
