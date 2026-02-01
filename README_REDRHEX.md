# RedRhex Hexapod Robot - Isaac Lab RL Training Project

A complete reinforcement learning training environment for the RedRhex hexapod robot, built on NVIDIA Isaac Lab and IsaacSim.

## Overview

RedRhex is a 6-legged hexapod robot with 18 degrees of freedom (3 joints per leg). This project implements:

- **Tripod Gait Locomotion**: Trains the robot to walk using an efficient 3-leg stance pattern
- **Reinforcement Learning**: PPO (Proximal Policy Optimization) training using RSL-RL framework
- **ABAD Joint Optimization**: Utilizes hip (ABAD) joints for dynamic balance and terrain adaptation
- **Physics Simulation**: High-fidelity simulation at 250 Hz using NVIDIA PhysX

## Project Structure

```
RedRhex/
├── source/RedRhex/                    # Main package
│   ├── RedRhex/
│   │   ├── tasks/
│   │   │   └── direct/redrhex/
│   │   │       ├── redrhex_env.py     # Environment implementation
│   │   │       ├── redrhex_env_cfg.py # Environment configuration
│   │   │       └── agents/            # RL agent configs (PPO, SKRL)
│   │   └── ui_extension_example.py
│   ├── setup.py
│   ├── pyproject.toml
│   └── config/
├── scripts/
│   ├── rsl_rl/
│   │   ├── train.py                   # PPO training script
│   │   ├── play.py                    # Play trained models
│   │   ├── train_jumping.py           # Jumping task training
│   │   └── cli_args.py
│   └── skrl/                          # SKRL framework scripts
├── RedRhex.usd                        # Robot model (USD format)
├── logs/                              # Training logs and checkpoints
└── README.md
```

## Prerequisites

- **Isaac Lab** v0.48.0+ ([Installation Guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html))
- **Isaac Sim** v5.1+
- **Python** 3.10+
- **CUDA** 11.8+
- **GPU**: NVIDIA RTX 4090 (or equivalent with ≥24GB VRAM recommended)

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd RedRhex
```

### 2. Install RedRhex Package

```bash
# If using IsaacLab installed via conda:
python -m pip install -e source/RedRhex

# Or use the IsaacLab launcher:
cd /path/to/IsaacLab
./isaaclab.sh -p /path/to/RedRhex/source/RedRhex/setup.py install
```

### 3. Verify Installation

```bash
# List available RedRhex tasks
python scripts/list_envs.py

# Expected output should include:
# Template-Redrhex-Direct-v0
```

## Quick Start

### Training a New Policy

```bash
# Basic training with 4 environments
./isaaclab.sh -p scripts/rsl_rl/train.py \
  --task Template-Redrhex-Direct-v0 \
  --num_envs 4 \
  --max_iterations 100

# Full training with 4096 environments (recommended for production)
./isaaclab.sh -p scripts/rsl_rl/train.py \
  --task Template-Redrhex-Direct-v0 \
  --num_envs 4096 \
  --max_iterations 1500
```

### Playing a Trained Model

```bash
# Run trained policy with visualization
python scripts/rsl_rl/play.py \
  --task Template-Redrhex-Direct-v0 \
  --num_envs 4 \
  --checkpoint logs/rsl_rl/redrhex_tripod_gait/<timestamp>/model_<iteration>.pt
```

### Testing with Dummy Agents

```bash
# Zero action agent (robot does nothing)
python scripts/zero_agent.py --task Template-Redrhex-Direct-v0

# Random action agent (baseline)
python scripts/random_agent.py --task Template-Redrhex-Direct-v0
```

## Configuration

### Main Environment Configuration

Edit `source/RedRhex/RedRhex/tasks/direct/redrhex/redrhex_env_cfg.py`:

**Key Physics Parameters:**
```python
# Gravity (standard Earth gravity)
gravity = (0.0, 0.0, -9.81)

# Robot damping (prevents instability)
linear_damping = 0.0        # Air resistance
angular_damping = 0.05      # Rotation damping

# Joint actuators
effort_limit = 10.0         # Max torque per joint
velocity_limit = 5.0        # Max joint speed
stiffness = 30.0            # Joint stiffness
damping = 3.0               # Joint damping
```

**Reward Scaling:**
```python
rew_scale_alive = 0.5                    # Survival reward
rew_scale_lin_vel_xy = 1.0              # Forward motion reward
rew_scale_forward_progress = 0.5        # Extra progress bonus
rew_scale_ang_vel_z = 0.3               # Turning penalty
rew_scale_base_height = -1.0            # Height tracking
rew_scale_action_rate = -0.01           # Smoothness penalty
```

### Training Configuration

Edit `source/RedRhex/RedRhex/tasks/direct/redrhex/agents/rsl_rl_ppo_cfg.py`:

```python
# Learning parameters
learning_rate = 1e-3
num_steps_per_env = 24        # Rollout length
num_mini_batches = 4          # Batch division
clip_ratio = 0.2              # PPO clip parameter

# Network architecture
policy_hidden_sizes = [256, 256, 128]
value_hidden_sizes = [256, 256, 128]
```

## Robot Specifications

- **DOF**: 18 (6 legs × 3 joints each)
- **Joint Types per Leg**:
  1. **ABAD (Hip)**: Abduction/Adduction (±30°)
  2. **Knee**: Flexion/Extension
  3. **Foot**: Toe joint
- **Gait Pattern**: Tripod gait alternating between two groups of 3 legs
- **Target Speed**: 0.15-0.3 m/s forward locomotion

## Training Tips

1. **Start Small**: Begin with 4-8 environments to debug issues, then scale to 4096
2. **Monitor Rewards**: Use tensorboard to track progress:
   ```bash
   tensorboard --logdir logs/rsl_rl/
   ```
3. **Adjust Damping**: If physics is unstable, increase joint damping gradually
4. **Curriculum Learning**: Start with simple tasks, progress to complex terrains
5. **Sample Efficiency**: Use 24-step rollouts to balance stability and efficiency

## Troubleshooting

### Physics Explosion
**Symptom**: Robot position jumps to extreme values

**Solution**: Increase joint damping in `redrhex_env_cfg.py`:
```python
damping = 3.0  # Increase value
```

### Slow Convergence
**Symptom**: Rewards not improving after many iterations

**Solution**: 
- Check learning rate (try 1e-3 to 1e-4)
- Verify reward scaling makes sense for your environment
- Ensure robot isn't getting stuck in local minima

### Memory Issues with Large num_envs
**Symptom**: CUDA out of memory error

**Solution**:
- Reduce `num_envs` (4096 → 2048)
- Reduce `num_steps_per_env` (24 → 16)
- Enable environment clustering

### Import Errors
**Symptom**: `ModuleNotFoundError: No module named 'isaaclab'`

**Solution**:
```bash
# Reinstall package in editable mode
python -m pip install -e source/RedRhex

# Or use IsaacLab launcher
cd /path/to/IsaacLab
./isaaclab.sh -p <your-script>.py
```

## Performance Metrics

### Expected Training Results
- **Initial Reward**: ~0 (random policy)
- **After 100 iterations**: ~45-65
- **After 500 iterations**: ~100-150
- **After 1500 iterations**: ~150-200+

### Convergence Speed
- ~4000 steps/second per environment (4 envs)
- ~60k steps/second total (4096 envs)
- Full training: ~8-12 hours on RTX 4090

## Contributing

1. Create a feature branch: `git checkout -b feature/my-feature`
2. Make changes and test thoroughly
3. Format code: `pre-commit run --all-files`
4. Commit with descriptive messages
5. Push and create a Pull Request

## Code Style

This project uses:
- **Pre-commit hooks** for automatic formatting
- **Flake8** for linting (config in `.flake8`)
- **Black** for code formatting

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

## References

- [Isaac Lab Documentation](https://isaac-sim.github.io/IsaacLab/)
- [RSL-RL PPO Implementation](https://github.com/leggedrobotics/rsl_rl)
- [PhysX Documentation](https://docs.omniverse.nvidia.com/sim/latest/index.html)

## Team

Built by: [Team Members]

## License

[Add appropriate license]

## Contact

For questions or issues, please contact: [contact info]

---

Last Updated: 2026-02-01
IsaacLab Version: 0.48.0
IsaacSim Version: 5.1
