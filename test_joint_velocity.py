"""
测试关节速度控制是否工作
"""
import torch
import argparse
from isaaclab.app import AppLauncher

# 启动 Isaac Sim
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass

@configclass
class TestSceneCfg(InteractiveSceneCfg):
    """场景配置"""
    ground = sim_utils.GroundPlaneCfg()
    
    robot = ArticulationCfg(
        prim_path="/World/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/jasonliao/RedRhex/RedRhex/RedRhex.usd",
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.5),
        ),
        actuators={
            "main_drive": ImplicitActuatorCfg(
                joint_names_expr=["Revolute_15", "Revolute_7", "Revolute_12", 
                                  "Revolute_18", "Revolute_23", "Revolute_24"],
                effort_limit=100.0,
                velocity_limit=30.0,
                stiffness=0.0,
                damping=50.0,
            ),
        },
    )

def main():
    # 创建场景
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device="cuda:0")
    sim = sim_utils.SimulationContext(sim_cfg)
    
    scene_cfg = TestSceneCfg(num_envs=1, env_spacing=2.5)
    scene = InteractiveScene(scene_cfg)
    
    sim.reset()
    
    # 获取机器人
    robot = scene["robot"]
    joint_names = robot.data.joint_names
    print(f"\n关节名称: {joint_names}")
    
    # 找主驱动关节索引
    main_drive_names = ["Revolute_15", "Revolute_7", "Revolute_12", 
                        "Revolute_18", "Revolute_23", "Revolute_24"]
    main_drive_indices = [joint_names.index(n) for n in main_drive_names]
    print(f"主驱动关节索引: {main_drive_indices}")
    
    # 设置目标速度
    target_vel = 6.28  # rad/s
    
    print(f"\n开始测试，目标速度: {target_vel} rad/s")
    print("-" * 60)
    
    for step in range(200):
        # 设置速度目标
        vel_target = torch.zeros(1, len(joint_names), device="cuda:0")
        for idx in main_drive_indices:
            vel_target[0, idx] = target_vel
        
        robot.set_joint_velocity_target(vel_target)
        
        # 模拟一步
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_cfg.dt)
        
        # 每 20 步打印一次
        if step % 20 == 0:
            joint_vel = robot.data.joint_vel[0]
            main_drive_vel = joint_vel[main_drive_indices]
            print(f"Step {step:3d}: 主驱动速度 = {main_drive_vel.cpu().numpy()}")
            print(f"          目标 = {target_vel:.2f}, 实际平均 = {main_drive_vel.mean().item():.2f}")
    
    print("\n测试完成!")
    simulation_app.close()

if __name__ == "__main__":
    main()
