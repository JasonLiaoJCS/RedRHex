# RedRHex Training Panel

Local web panel for launching small RSL-RL training runs, finding reward tuning files, viewing run history, and keeping notes.

**Version:** 1.0.0  
**Published by:** BioRoLa ABAD RHex Team  
**Credits:** Jason Liao and Jacob Yang

## Run

Local-only:

```bash
python -m tools.training_panel --host 127.0.0.1 --port 8080
```

LAN access:

```bash
python -m tools.training_panel --host 0.0.0.0 --port 8080
```

SSH tunnel from another machine:

```bash
ssh -L 8080:127.0.0.1:8080 user@host
```

Then open:

```text
http://127.0.0.1:8080
```

## Scope

V1 is intentionally read-only for reward/config source files. It shows the main tweakable files and reward scales with path and line information, but does not edit source code.

History actions:

- `Rename` is handled in the details panel after selecting a run.
- `TensorBoard` opens a pending browser tab immediately, starts TensorBoard for that run's log directory, then points the tab at the launched port.
- `Play` starts `scripts/rsl_rl/play.py` with the latest checkpoint and selects that process in the Process Console.
- `Recorded Result` embeds the latest MP4 and records one high-quality default video. After a successful panel-launched training run, the panel automatically records the same high-quality result.
- `Resume` sends the latest checkpoint back to the Train form so you can choose new env/iteration values before continuing training.
- `Process Console` shows the exact panel-launched command, live output, diagnosis text, and attach metadata. When `tmux` is installed, panel processes run in detached tmux sessions so you can attach from SSH with the copied command and use `Ctrl+C` directly.
- `Pop Out` opens the same Process Console in a dedicated browser tab, which works over local, LAN, and SSH-tunneled panel sessions.
- `Copy Output` and `Copy Attach Command` copy the visible process log or real terminal attach command.
- `Open Run Folder`, `Open Video Folder`, and `Open Log Folder` try to open repo-owned paths on the host and always return copyable paths for remote sessions.
- `Stop Process` sends a Ctrl-C style interrupt to the currently selected training/play/video/TensorBoard process, then escalates if Isaac Sim does not close.
- `Delete Run` first previews the exact repo-owned log/note paths that will be removed, then requires typing the exact run id. It refuses to delete while a related process is still running.

Video default:

- `High`: 1920x1080, 1200 steps, 30 FPS, quality rendering.

Training history and notes are stored under:

```text
logs/training_panel/
```

RSL-RL logs continue to live under:

```text
logs/rsl_rl/redrhex_wheg/
```
