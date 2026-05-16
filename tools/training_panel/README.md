# RedRHex Training Panel

Local admin panel and V2.0 remote-control foundation for launching RSL-RL training runs, finding reward tuning files, viewing run history, keeping notes, and coordinating team access.

**Version:** 2.0.0
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

## V2.0 Remote Team Mode

V2.0 keeps this local panel as the admin/control-center experience and adds a remote worker architecture for team use outside the lab network.

Remote architecture:

- GitHub Pages hosts the public static web UI.
- Supabase stores team login, roles, run/job state, artifacts, proxy sessions, and notification events.
- This training PC runs a worker that polls Supabase and executes queued jobs locally.
- Cloudflare Tunnel provides secure live console and TensorBoard access.
- Discord and email notifications are generated from completion/failure events.

Worker command:

```bash
python -m tools.training_panel.remote_worker
```

Required remote environment variables:

```bash
export REDRHEX_SUPABASE_URL="https://<project>.supabase.co"
export REDRHEX_SUPABASE_ANON_KEY="<anon-key>"
export REDRHEX_SUPABASE_MACHINE_TOKEN="<machine-token>"
export REDRHEX_MACHINE_ID="biorolapc2-ubuntu"
export REDRHEX_REMOTE_ACCEPT_JOBS="false"
export REDRHEX_CLOUDFLARE_TUNNEL_HOST="https://<tunnel-host>"
export REDRHEX_DISCORD_WEBHOOK_URL="<discord-webhook>"
export REDRHEX_RESEND_API_KEY="<resend-key>"
```

Apply the Supabase schema from:

```text
tools/training_panel/supabase/schema.sql
```

Use the local panel's `Control Center` tab to inspect remote configuration, copy worker/tunnel commands, and enable or disable remote job acceptance.

## Scope

The local reward/config source files are still protected by default. The panel shows the main tweakable files and reward scales with path and line information, but does not directly edit source code.

History actions:

- `Rename` is handled in the details panel after selecting a run.
- `TensorBoard` opens a pending browser tab immediately, starts TensorBoard for that run's log directory, then points the tab at the launched port.
- `Play` starts `scripts/rsl_rl/play.py` with the latest checkpoint and selects that process in the Process Console.
- `Export ONNX` exports the latest checkpoint to `exported/policy.onnx` and selects that export process in the Process Console.
- `Compact Run` deletes old top-level `model_*.pt` checkpoints after confirmation, keeping the highest-iteration checkpoint and preserving videos, TensorBoard logs, params, notes, and exported policy files.
- `Recorded Result` embeds the latest MP4 and records one high-quality default video. After a successful panel-launched training run, the panel automatically records the same high-quality result.
- `Resume` sends the latest checkpoint back to the Train form so you can choose new env/iteration values before continuing training.
- `Process Console` shows the exact panel-launched command, live output, diagnosis text, and attach metadata. When `tmux` is installed, panel processes run in detached tmux sessions so you can attach from SSH with the copied command and use `Ctrl+C` directly.
- `Pop Out` opens the same Process Console in a dedicated browser tab, which works over local, LAN, and SSH-tunneled panel sessions.
- `Copy Output` and `Copy Attach Command` copy the visible process log or real terminal attach command.
- `Open Run Folder`, `Open Video Folder`, and `Open Log Folder` try to open repo-owned paths on the host and always return copyable paths for remote sessions.
- `Stop Process` sends a Ctrl-C style interrupt to the currently selected training/play/video/TensorBoard process, then escalates if Isaac Sim does not close.
- `Delete Run` first previews the exact repo-owned log/note paths that will be removed, then requires typing the exact run id. It refuses to delete while a related process is still running.

Remote roles:

- `viewer`: inspect runs, logs, artifacts, and proxy links.
- `operator`: viewer permissions plus launch, stop, video recording, and ONNX export.
- `admin`: operator permissions plus delete, compact, access, and settings.

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
