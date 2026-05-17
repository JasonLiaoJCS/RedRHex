from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable


SUMMARY_DIR_NAME = "training_panel"
SUMMARY_FILE_NAME = "tensorboard_summary.png"

PRIORITY_SCALARS: tuple[tuple[str, str], ...] = (
    ("Train/mean_reward", "Mean Reward"),
    ("Train/mean_episode_length", "Episode Length"),
    ("Loss/value_function", "Value Loss"),
    ("Loss/surrogate", "Surrogate Loss"),
    ("Episode_Reward/diag_forward_vel", "Forward Velocity"),
    ("Episode_Reward/diag_lateral_vel", "Lateral Velocity"),
    ("Episode_Reward/diag_actual_wz", "Yaw Rate"),
    ("Episode_Reward/diag_base_height", "Base Height"),
    ("Episode_Reward/diag_tilt", "Tilt"),
    ("Episode_Reward/diag_cost_of_transport", "Cost of Transport"),
    ("Episode_Reward/diag_mech_power_total", "Mechanical Power"),
    ("Policy/mean_noise_std", "Policy Noise"),
)


def tensorboard_summary_path(log_dir: Path) -> Path:
    return log_dir / SUMMARY_DIR_NAME / SUMMARY_FILE_NAME


def _event_files(log_dir: Path) -> list[Path]:
    return [path for path in log_dir.glob("events.out.tfevents.*") if path.is_file()]


def _newest_mtime(paths: Iterable[Path]) -> float:
    newest = 0.0
    for path in paths:
        try:
            newest = max(newest, path.stat().st_mtime)
        except OSError:
            continue
    return newest


def _needs_regeneration(log_dir: Path, output: Path) -> bool:
    events = _event_files(log_dir)
    if not events:
        return False
    if not output.is_file():
        return True
    try:
        return output.stat().st_mtime < _newest_mtime(events)
    except OSError:
        return True


def _downsample(values: list[tuple[int, float]], max_points: int = 800) -> list[tuple[int, float]]:
    if len(values) <= max_points:
        return values
    stride = max(1, math.ceil(len(values) / max_points))
    sampled = values[::stride]
    if sampled[-1] != values[-1]:
        sampled.append(values[-1])
    return sampled


def _select_scalar_tags(tags: list[str], max_plots: int = 8) -> list[tuple[str, str]]:
    selected: list[tuple[str, str]] = []
    used: set[str] = set()
    for tag, label in PRIORITY_SCALARS:
        if tag in tags and tag not in used:
            selected.append((tag, label))
            used.add(tag)
        if len(selected) >= max_plots:
            return selected
    for tag in tags:
        if tag in used:
            continue
        if tag.startswith("Episode_Reward/diag_") or tag.startswith("Episode_Reward/rew_"):
            selected.append((tag, tag.removeprefix("Episode_Reward/")))
            used.add(tag)
        if len(selected) >= max_plots:
            return selected
    for tag in tags:
        if tag not in used:
            selected.append((tag, tag))
            used.add(tag)
        if len(selected) >= max_plots:
            break
    return selected


def ensure_tensorboard_summary(log_dir: Path, *, title: str = "") -> Path | None:
    """Create or refresh a compact TensorBoard scalar PNG for a run log dir."""
    log_dir = Path(log_dir)
    if not log_dir.is_dir():
        return None
    output = tensorboard_summary_path(log_dir)
    if not _needs_regeneration(log_dir, output):
        return output if output.is_file() else None

    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    accumulator = EventAccumulator(str(log_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    tags = list(accumulator.Tags().get("scalars", []))
    selected = _select_scalar_tags(tags)
    if not selected:
        return None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = math.ceil(len(selected) / 2)
    fig, axes = plt.subplots(rows, 2, figsize=(13.5, max(3.8, rows * 3.1)), squeeze=False)
    fig.patch.set_facecolor("#f7faf9")
    fig.suptitle(title or log_dir.name, fontsize=15, fontweight="bold", y=0.995)
    for axis in axes.flat:
        axis.set_visible(False)

    for axis, (tag, label) in zip(axes.flat, selected):
        values = [(int(event.step), float(event.value)) for event in accumulator.Scalars(tag)]
        values = [(step, value) for step, value in values if math.isfinite(value)]
        if not values:
            continue
        values = _downsample(values)
        xs = [step for step, _ in values]
        ys = [value for _, value in values]
        axis.set_visible(True)
        axis.plot(xs, ys, color="#277f76", linewidth=1.7)
        axis.set_title(label, fontsize=11, loc="left")
        axis.grid(True, color="#d8e4e1", linewidth=0.8)
        axis.tick_params(labelsize=8)
        axis.text(
            0.98,
            0.92,
            f"latest {ys[-1]:.4g}",
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            color="#415150",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "#ffffff", "edgecolor": "#d8e4e1"},
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output, dpi=145, facecolor=fig.get_facecolor())
    plt.close(fig)
    return output if output.is_file() else None
