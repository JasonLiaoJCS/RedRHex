from __future__ import annotations

import re
from pathlib import Path


REWARD_SCALE_RE = re.compile(r"^\s*(rew_scale_[A-Za-z0-9_]+)\s*=\s*([^#\n]+)(?:#\s*(.*))?")


TWEAKABLE_FILES = [
    {
        "title": "Reward scales and environment constants",
        "path": "source/RedRhex/RedRhex/tasks/direct/redrhex/redrhex_env_cfg.py",
        "why": "Primary place to inspect reward weights, command ranges, episode limits, and joint mappings.",
    },
    {
        "title": "Reward calculation logic",
        "path": "source/RedRhex/RedRhex/tasks/direct/redrhex/redrhex_env.py",
        "why": "Shows how each reward component is computed and logged during training.",
    },
    {
        "title": "PPO training settings",
        "path": "source/RedRhex/RedRhex/tasks/direct/redrhex/agents/rsl_rl_ppo_cfg.py",
        "why": "Controls PPO network size, learning rate, entropy, batch settings, and save interval.",
    },
]


def scan_reward_scales(repo_root: Path) -> list[dict]:
    cfg_path = repo_root / TWEAKABLE_FILES[0]["path"]
    if not cfg_path.exists():
        return []
    scales = []
    for line_no, line in enumerate(cfg_path.read_text(encoding="utf-8").splitlines(), start=1):
        match = REWARD_SCALE_RE.match(line)
        if not match:
            continue
        name, value, comment = match.groups()
        scales.append(
            {
                "name": name,
                "value": value.strip(),
                "comment": (comment or "").strip(),
                "path": str(cfg_path),
                "relative_path": TWEAKABLE_FILES[0]["path"],
                "line": line_no,
            }
        )
    return scales


def reward_file_index(repo_root: Path) -> dict:
    files = []
    for item in TWEAKABLE_FILES:
        path = repo_root / item["path"]
        files.append({**item, "absolute_path": str(path), "exists": path.exists()})
    return {
        "files": files,
        "reward_scales": scan_reward_scales(repo_root),
        "mode": "read-only",
    }

