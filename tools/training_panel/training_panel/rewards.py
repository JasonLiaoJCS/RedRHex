from __future__ import annotations

import re
from pathlib import Path


REWARD_SCALE_RE = re.compile(r"^\s*(rew_scale_[A-Za-z0-9_]+)\s*=\s*([^#\n]+)(?:#\s*(.*))?")
_YAML_SCALE_RE = re.compile(r"^\s*(rew_scale_[A-Za-z0-9_]+)\s*:\s*([^\s#]+)")

_DIFF_TOLERANCE = 1e-9  # float comparison tolerance


def read_reward_scales_from_yaml(env_yaml_path: Path) -> dict[str, float]:
    """Parse rew_scale_* entries from a saved params/env.yaml without requiring PyYAML."""
    if not env_yaml_path.exists():
        return {}
    scales: dict[str, float] = {}
    for line in env_yaml_path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = _YAML_SCALE_RE.match(line)
        if m:
            try:
                scales[m.group(1)] = float(m.group(2))
            except ValueError:
                pass
    return scales


def reward_diff(
    yaml_scales: dict[str, float],
    defaults: dict[str, float],
) -> dict:
    """Compare yaml_scales against defaults. Returns changed/same/missing lists."""
    changed = []
    same = []
    for name, default_val in defaults.items():
        if name not in yaml_scales:
            continue
        yaml_val = yaml_scales[name]
        if abs(yaml_val - default_val) > _DIFF_TOLERANCE:
            delta_pct = round((yaml_val - default_val) / (abs(default_val) + 1e-12) * 100, 1)
            changed.append({
                "name": name,
                "yaml_value": yaml_val,
                "default_value": default_val,
                "delta_pct": delta_pct,
            })
        else:
            same.append(name)
    # Also flag values in YAML that have no matching default (new fields)
    for name, yaml_val in yaml_scales.items():
        if name not in defaults:
            changed.append({
                "name": name,
                "yaml_value": yaml_val,
                "default_value": None,
                "delta_pct": None,
            })
    return {"changed": changed, "same": same}



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


def reward_defaults(repo_root: Path) -> dict[str, float]:
    """Return current rew_scale_* defaults as {name: float} from the env config."""
    scales = {}
    for item in scan_reward_scales(repo_root):
        try:
            scales[item["name"]] = float(item["value"])
        except ValueError:
            pass
    return scales


def reward_file_index(repo_root: Path) -> dict:
    files = []
    for item in TWEAKABLE_FILES:
        path = repo_root / item["path"]
        files.append({**item, "absolute_path": str(path), "exists": path.exists()})
    return {
        "files": files,
        "reward_scales": scan_reward_scales(repo_root),
        "reward_defaults": reward_defaults(repo_root),
        "mode": "read-only",
    }

