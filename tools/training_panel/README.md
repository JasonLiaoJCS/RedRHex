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

Training history and notes are stored under:

```text
logs/training_panel/
```

RSL-RL logs continue to live under:

```text
logs/rsl_rl/redrhex_wheg/
```

