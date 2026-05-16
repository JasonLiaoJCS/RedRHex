from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


def default_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def timestamp_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


@dataclass(frozen=True)
class PanelPaths:
    repo_root: Path
    isaaclab_root: Path
    isaacsim_root: Path
    conda_sh: Path
    conda_env: str

    @classmethod
    def from_env(cls) -> "PanelPaths":
        repo_root = Path(os.environ.get("REDRHEX_ROOT", default_repo_root())).resolve()
        return cls(
            repo_root=repo_root,
            isaaclab_root=Path(os.environ.get("ISAACLAB_ROOT", "/home/lab_user1/isaac_lab_ws/IsaacLab")),
            isaacsim_root=Path(os.environ.get("ISAACSIM_ROOT", "/home/lab_user1/isaacsim")),
            conda_sh=Path(os.environ.get("CONDA_SH", "/home/lab_user1/miniconda3/etc/profile.d/conda.sh")),
            conda_env=os.environ.get("REDRHEX_CONDA_ENV", "env_isaaclab_bin"),
        )

    @property
    def isaaclab_launcher(self) -> Path:
        return self.isaaclab_root / "isaaclab.sh"

    @property
    def conda_root(self) -> Path:
        return self.conda_sh.parents[2]

    @property
    def conda_prefix(self) -> Path:
        return self.conda_root / "envs" / self.conda_env

    @property
    def panel_log_root(self) -> Path:
        return self.repo_root / "logs" / "training_panel"

    @property
    def process_log_dir(self) -> Path:
        return self.panel_log_root / "process_logs"

    @property
    def notes_dir(self) -> Path:
        return self.panel_log_root / "notes"

    @property
    def history_file(self) -> Path:
        return self.panel_log_root / "runs.json"

    @property
    def remote_state_file(self) -> Path:
        return self.panel_log_root / "remote_state.json"

    @property
    def convergence_config_file(self) -> Path:
        return self.panel_log_root / "convergence_config.json"

    @property
    def rsl_rl_log_root(self) -> Path:
        return self.repo_root / "logs" / "rsl_rl" / "redrhex_wheg"

    @property
    def reward_override_file(self) -> Path:
        return self.repo_root / "tools" / "training_panel" / "active_reward_override.json"

    def ensure_dirs(self) -> None:
        self.panel_log_root.mkdir(parents=True, exist_ok=True)
        self.process_log_dir.mkdir(parents=True, exist_ok=True)
        self.notes_dir.mkdir(parents=True, exist_ok=True)
