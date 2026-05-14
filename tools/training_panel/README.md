# RedRHex Training Panel

Local web panel for launching small RSL-RL training runs, finding reward tuning files, viewing run history, and keeping notes.

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
- `TensorBoard` starts TensorBoard for that run's log directory and opens it.
- `Play` starts `scripts/rsl_rl/play.py` with the latest checkpoint.
- `Resume` sends the latest checkpoint back to the Train form so you can choose new env/iteration values before continuing training.
- `Debug` shows the exact panel-launched command and captured process log. Runs discovered from existing RSL-RL logs may not have a panel process log. TensorBoard/play startup failures also return their log tail in the panel so missing environment packages are visible immediately.

Training history and notes are stored under:

```text
logs/training_panel/
```

RSL-RL logs continue to live under:

```text
logs/rsl_rl/redrhex_wheg/
```
