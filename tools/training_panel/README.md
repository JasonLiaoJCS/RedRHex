# RedRHex Training Panel

Local admin panel and V3.0 remote-control system for launching RSL-RL training runs, tuning rewards/terrain, viewing run history, keeping notes, coordinating team access, and sending requester-scoped notifications.

**Version:** 3.0.0
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

## V3.0 Remote Team Mode

V3.0 keeps this local panel as the admin/control-center experience and adds requester-scoped Discord notifications to the remote worker architecture.

V3.0 highlights:

- `RedRHex To Go`, the phone-friendly child GitHub Pages UI for dashboard, training launch, reward tuning, history, notes/folders, safe remote actions, and signed team video playback.
- Local Control Center worker management: start, stop, restart, tmux/child mode, auto-start, accept/pause jobs, status tail, and setup checks.
- Faster, safer worker sync with heartbeat polling, metadata convergence between mother and child, non-fatal artifact sync, and private video upload records.
- Child auto-update without full-page rebuilds, so video playback, scrolling, and in-progress edits stay stable.
- Per-user notification settings from the child Connection page, using each requester's Discord webhook.

Remote architecture:

- GitHub Pages hosts the public static web UI.
- Supabase stores team login, roles, run/job state, artifacts, proxy sessions, and notification events.
- This training PC runs a worker that polls Supabase and executes queued jobs locally.
- Cloudflare Tunnel provides secure live console and TensorBoard access.
- Requester-only Discord notifications are dispatched by the Supabase `notify` Edge Function for convergence, completion, failure/interruption, and video-ready events.

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
```

Notification delivery secrets live on the Supabase Edge Function, not in the child page:

```bash
supabase secrets set REDRHEX_SUPABASE_MACHINE_TOKEN="<machine-token-if-not-service-role>"
supabase functions deploy notify --project-ref <project-ref> --no-verify-jwt
```

### Start The Remote Worker

Use this on the training PC when you want the GitHub Pages remote UI to see this machine and launch jobs.

Preferred flow:

```bash
python -m tools.training_panel --host 127.0.0.1 --port 8080
```

Open the local panel, go to `Control Center`, then use:

- `Start Worker` / `Stop Worker` / `Restart`
- `Persist in tmux` for a worker that survives local panel restarts
- `Child process` for a worker owned by the current panel session
- `Auto-start worker when the local panel starts` to launch it automatically next time
- `Accept Jobs` / `Pause Jobs` to control whether remote queued jobs are executed

The panel reads secrets from `~/.redrhex_remote.env` and never shows the machine token in the browser.

Manual fallback:

1. Create a private environment file once:

```bash
nano ~/.redrhex_remote.env
```

Paste the remote settings:

```bash
export REDRHEX_SUPABASE_URL="https://<project>.supabase.co"
export REDRHEX_SUPABASE_ANON_KEY="<anon-or-publishable-key>"
export REDRHEX_SUPABASE_MACHINE_TOKEN="<service-role-key>"
export REDRHEX_MACHINE_ID="biorolapc2-ubuntu"
export REDRHEX_REMOTE_ACCEPT_JOBS="true"
export REDRHEX_CLOUDFLARE_TUNNEL_HOST=""
```

Save it, then lock down the file:

```bash
chmod 600 ~/.redrhex_remote.env
```

Do not commit this file or paste the service-role key into GitHub Pages. The service-role key belongs only on the training PC.

2. Test the Supabase connection once:

```bash
cd /home/lab_user1/Py/RedRHex
source ~/.redrhex_remote.env
python -m tools.training_panel.remote_worker --once
```

Expected output should include:

```text
'machine_id': 'biorolapc2-ubuntu'
'online': True
```

If it says `status: disabled`, set `REDRHEX_REMOTE_ACCEPT_JOBS="true"` in `~/.redrhex_remote.env` when you are ready to accept remote jobs.

3. Run the worker continuously in `tmux`:

```bash
cd /home/lab_user1/Py/RedRHex
tmux new-session -d -s redrhex_remote_worker 'source ~/.redrhex_remote.env && python -m tools.training_panel.remote_worker'
```

Check it:

```bash
tmux ls
tmux attach -t redrhex_remote_worker
```

Detach without stopping it:

```text
Ctrl+B, then D
```

Stop it:

```bash
tmux send-keys -t redrhex_remote_worker C-c
```

If `tmux` is not installed, use a log file instead:

```bash
cd /home/lab_user1/Py/RedRHex
mkdir -p logs/training_panel
nohup bash -lc 'source ~/.redrhex_remote.env && python -m tools.training_panel.remote_worker' \
  > logs/training_panel/remote_worker.log 2>&1 &
tail -f logs/training_panel/remote_worker.log
```

Verify from Supabase:

```text
Supabase -> Table Editor -> machines -> biorolapc2-ubuntu
```

The row should show `online = true`, `accept_jobs = true`, and a fresh `heartbeat_at`.

Phone page:

```text
https://popcorn-volcano.github.io/redrhex-training-remote/
```

The phone page is the **child** panel: a simplified, team-friendly version of the local **mother** panel. It supports Dashboard, Train, Rewards, Terrain, History, and Connection views. Team members can check worker health, queue training, tune shared reward and terrain presets, review history, edit notes/folders, start TensorBoard, compact runs, request checkpoint-specific videos, and play private uploaded result videos. Terminal, tmux attach, local file opening, ONNX export, stop-process controls, deletion, and worker administration stay mother-only.

Apply the Supabase schema from:

```text
tools/training_panel/supabase/schema.sql
```

Re-apply the schema after pulling V3.0.0 updates. It adds `reward_presets`, `terrain_presets`, run `notes`/`folder` metadata, updated-at triggers, queue filtering helpers, requester-scoped notification settings, run event delivery status, and the private `redrhex-videos` Storage bucket used for signed team-only MP4 playback.

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
- `operator`: viewer permissions plus launch, video recording, TensorBoard, and compaction.
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
