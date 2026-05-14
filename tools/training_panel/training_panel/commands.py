from __future__ import annotations

import shlex
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import PanelPaths


DEFAULT_TASK = "Template-Redrhex-Direct-v0"


@dataclass
class TrainingParams:
    task: str = DEFAULT_TASK
    num_envs: int = 4
    max_iterations: int = 1
    device: str = "cuda:0"
    headless: bool = True
    seed: int | None = None
    resume: bool = False
    checkpoint: str | None = None

    @classmethod
    def from_dict(cls, data: dict) -> "TrainingParams":
        params = cls(
            task=str(data.get("task") or DEFAULT_TASK),
            num_envs=int(data.get("num_envs") or 4),
            max_iterations=int(data.get("max_iterations") or 1),
            device=str(data.get("device") or "cuda:0"),
            headless=bool(data.get("headless", True)),
            seed=int(data["seed"]) if data.get("seed") not in (None, "") else None,
            resume=bool(data.get("resume", False)),
            checkpoint=str(data["checkpoint"]) if data.get("checkpoint") else None,
        )
        params.validate()
        return params

    def validate(self) -> None:
        if not self.task:
            raise ValueError("task is required")
        if self.num_envs < 1 or self.num_envs > 8192:
            raise ValueError("num_envs must be between 1 and 8192")
        if self.max_iterations < 1 or self.max_iterations > 100000:
            raise ValueError("max_iterations must be between 1 and 100000")
        if not (self.device == "cpu" or self.device.startswith("cuda")):
            raise ValueError("device must be cpu or cuda[:index]")
        if self.resume and not self.checkpoint:
            raise ValueError("checkpoint is required when resume is enabled")

    def to_dict(self) -> dict:
        return asdict(self)


def training_argv(params: TrainingParams) -> list[str]:
    argv = [
        "scripts/rsl_rl/train.py",
        "--task",
        params.task,
        "--num_envs",
        str(params.num_envs),
        "--max_iterations",
        str(params.max_iterations),
        "--device",
        params.device,
    ]
    if params.headless:
        argv.append("--headless")
    if params.seed is not None:
        argv.extend(["--seed", str(params.seed)])
    if params.resume:
        argv.append("--resume")
        argv.extend(["--checkpoint", params.checkpoint or ""])
    return argv


def play_argv(checkpoint: str, task: str = DEFAULT_TASK, num_envs: int = 1, device: str = "cuda:0") -> list[str]:
    return [
        "scripts/rsl_rl/play.py",
        "--task",
        task,
        "--num_envs",
        str(num_envs),
        "--device",
        device,
        "--checkpoint",
        checkpoint,
    ]


def tensorboard_argv(logdir: Path, host: str, port: int) -> list[str]:
    return ["tensorboard", "--logdir", str(logdir), "--host", host, "--port", str(port)]


def shell_for_isaaclab(paths: PanelPaths, script_argv: list[str]) -> str:
    quoted_launcher = shlex.quote(str(paths.isaaclab_launcher))
    quoted_args = " ".join(shlex.quote(arg) for arg in script_argv)
    return "\n".join(
        [
            f"export REDRHEX_ROOT={shlex.quote(str(paths.repo_root))}",
            f"export ISAACLAB_ROOT={shlex.quote(str(paths.isaaclab_root))}",
            f"export ISAACSIM_ROOT={shlex.quote(str(paths.isaacsim_root))}",
            f"source {shlex.quote(str(paths.conda_sh))}",
            f"conda activate {shlex.quote(paths.conda_env)}",
            "export TERM=xterm",
            f"cd {shlex.quote(str(paths.repo_root))}",
            f"exec {quoted_launcher} -p {quoted_args}",
        ]
    )


def shell_for_command(paths: PanelPaths, argv: list[str]) -> str:
    quoted = " ".join(shlex.quote(arg) for arg in argv)
    return "\n".join(
        [
            f"source {shlex.quote(str(paths.conda_sh))}",
            f"conda activate {shlex.quote(paths.conda_env)}",
            f"cd {shlex.quote(str(paths.repo_root))}",
            f"exec {quoted}",
        ]
    )


def display_isaaclab_command(paths: PanelPaths, script_argv: list[str]) -> str:
    quoted_args = " ".join(shlex.quote(arg) for arg in script_argv)
    return f"{paths.isaaclab_launcher} -p {quoted_args}"

