# RedRHex Training Panel Manual

Version line: V3.4 First Release, with Mother-side training queue support.

Published by: BioRoLa ABAD RHex Team  
Credits: Jason Liao and Jacob Yang

## 1. What This Is

RedRHex Training Panel is the lab control room for training, replaying, recording, exporting, and reviewing RedRHex RSL-RL experiments.

There are two faces:

- **Mother**: the local admin panel running on the training PC. It has the full tool belt.
- **RedRHex To Go**: the child phone/web app for team members. It is simpler, safer, and better for remote use.

Mother owns the machine. Child asks politely.

## 2. Start Mother

From the repo root:

```bash
python -m tools.training_panel --host 127.0.0.1 --port 8080
```

Open:

```text
http://127.0.0.1:8080
```

For LAN access:

```bash
python -m tools.training_panel --host 0.0.0.0 --port 8080
```

For SSH tunnel access:

```bash
ssh -L 8080:127.0.0.1:8080 user@host
```

Then open `http://127.0.0.1:8080` on your local browser.

## 3. Normal Training Workflow

1. Open `Train`.
2. Pick task, env count, iteration count, device, reward preset, and terrain preset.
3. Press `Train`.
4. If the GPU is free, Mother starts immediately.
5. If Isaac is busy, Mother creates a `queued` run. The queue is calm. The GPU is not.
6. Watch progress in `History` and `Process Console`.
7. Use `TensorBoard`, `Play`, `Record Video`, `Export ONNX`, or `Resume to Train` after checkpoints appear.

### Queue Behavior

Mother serializes Isaac/GPU actions:

- training
- play
- video recording
- ONNX export

If one is active, new training requests wait as `queued`. Use `Cancel Queue` if you change your mind.

This prevents the classic “five Isaac windows enter, zero GPU memory leaves” event.

## 4. Rewards

Use `Rewards` to tune reward scales through presets.

Rules:

- Built-in presets are read-only.
- Duplicate a preset before editing.
- The selected reward preset is used for new training.
- History can compare reward differences between runs.

Good habit: name presets like a lab note, not like a midnight mystery. `stairs_stable_v2` beats `asdf_final_real_final`.

## 5. Terrain

Use `Terrain` to tune terrain generator settings, curriculum, and sub-terrain mix.

Rules:

- Built-in terrain profiles are read-only.
- Duplicate before editing.
- The selected terrain preset is used for training.
- Play and video replay use the terrain saved with the run when available.

If playback looks wrong, check the run’s `params/env.yaml` first. That file is the time capsule.

## 6. History

History is the experiment library.

Each card shows:

- run status
- task/envs/iterations
- reward/terrain differences
- checkpoint state
- video and ONNX state
- notes/folder state

Useful actions:

- `TensorBoard`: metrics.
- `Play`: open Isaac playback for the checkpoint.
- `Record Video`: save a high-quality MP4.
- `Export ONNX`: export `exported/policy.onnx`.
- `Resume to Train`: continue from latest checkpoint.
- `Compact Run`: delete old top-level `model_*.pt`, keep the newest.
- `Console`: exact command and captured output.
- `Compare`: inspect reward/terrain differences between runs.

Folders are for human sanity. Use them early.

## 7. Process Console

The Process Console has two main boxes:

- **Launch Command**: what Mother asked the machine to run.
- **Output**: what Isaac/RSL-RL actually said back.

Buttons:

- `Copy Command`
- `Copy Output`
- `Pop Out`
- `Open Log Folder`
- `Stop Process`

When `tmux` is available, Mother runs Isaac jobs in detached sessions. That means SSH users can attach and press `Ctrl+C` like civilized engineers.

## 8. Videos

Mother records high-quality result videos by default:

- 1920x1080
- 1200 steps
- 30 FPS
- quality rendering

Videos are linked to checkpoint iteration when possible, so you know what model you are watching.

If video fails:

1. Open `Console`.
2. Read the output diagnosis.
3. Check GPU memory.
4. Retry after other Isaac jobs are stopped.

## 9. ONNX Export

Use `Export ONNX` from History after a checkpoint exists.

Output:

```text
<run log dir>/exported/policy.onnx
```

If ONNX export fails, check:

- checkpoint exists
- Isaac starts cleanly
- no other GPU action is active

## 10. Compact Run

`Compact Run` deletes only old top-level checkpoint files:

```text
model_0.pt
model_100.pt
model_200.pt
```

It keeps the highest numeric `model_N.pt`.

It preserves:

- videos
- TensorBoard logs
- params
- notes
- exported policy files

It asks for confirmation because deletion should not be a surprise party.

## 11. RedRHex To Go

RedRHex To Go is the child web app, intended for phone-first team use.

It supports:

- connection health
- training queue
- reward tuning
- terrain tuning
- history
- folders and notes
- signed team video playback
- safe remote actions

It does not expose:

- local file opening
- raw tmux control
- full terminal debugging
- worker secrets

Phone page:

```text
https://popcorn-volcano.github.io/redrhex-training-remote/
```

## 12. Remote Worker

Mother’s `Control Center` manages the remote worker.

Use it to:

- start worker
- stop worker
- restart worker
- choose `tmux` or child-process mode
- enable auto-start
- accept/pause remote jobs
- inspect setup status

Secrets live in:

```text
~/.redrhex_remote.env
```

Never paste the machine token into GitHub Pages or browser code.

## 13. Roles

- `viewer`: can inspect runs and artifacts.
- `operator`: can queue training and safe actions.
- `admin`: can manage access and destructive actions.

Configure roles in Supabase `profiles`.

## 14. Notifications

Requester-scoped notifications can go to Discord.

Typical notification events:

- convergence detected
- training completed
- training failed/interrupted
- video ready

If notifications are quiet, check:

- child Connection page
- Supabase Edge Function secrets
- `run_events`
- requester profile/settings

## 15. Activity

Mother’s `Activity` tab is team mission control.

It shows:

- contribution score
- training counts
- success/failure mix
- videos/ONNX
- active members
- recent failures
- member-grouped activity logs

Use it for team review, not blame. Data is a flashlight, not a hammer.

## 16. Troubleshooting

### Port Already In Use

Something is already on `8080`.

```bash
lsof -i :8080
```

Stop the old panel or run on another port.

### GPU Out Of Memory

Symptoms:

```text
CUDA error: out of memory
PhysX failed to allocate GPU memory
Failed to get DOF velocities from backend
```

Check:

```bash
nvidia-smi
```

Stop old Isaac jobs, close extra play/video windows, then retry. The queue helps prevent this from happening again.

### No Checkpoint

Training did not get far enough to save `model_*.pt`, or the run failed before logging.

Open `Console` and inspect the output.

### Video Missing

Possible causes:

- checkpoint missing
- GPU action collided
- movie/encoder issue
- Isaac failed during playback

Use `Console`, then retry when GPU is free.

### Child Says Offline

Check Mother `Control Center`:

- env file exists
- worker running
- accepting jobs enabled
- Supabase heartbeat fresh
- Cloudflare tunnel if using remote console/TensorBoard

## 17. Maintenance Checklist

Before a demo:

- Pull latest repo.
- Restart Mother.
- Start remote worker from `Control Center`.
- Confirm Child shows machine online.
- Run one tiny smoke training.
- Confirm History updates.
- Confirm video can be played.

Before a long training day:

- Check `nvidia-smi`.
- Compact old runs if disk is getting dramatic.
- Put new experiments into folders.
- Write notes while your brain still remembers why the run exists.
