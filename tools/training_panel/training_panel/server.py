from __future__ import annotations

import argparse
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .commands import DEFAULT_TASK, TrainingParams
from .config import PanelPaths
from .history import HistoryStore
from .processes import ProcessRegistry
from .rewards import reward_file_index


STATIC_DIR = Path(__file__).resolve().parents[1] / "static"


class PanelState:
    def __init__(self, paths: PanelPaths):
        self.paths = paths
        self.history = HistoryStore(paths)
        self.processes = ProcessRegistry(paths, self.history)


class PanelHandler(BaseHTTPRequestHandler):
    state: PanelState

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            return self._send_static("index.html")
        if parsed.path.startswith("/static/"):
            return self._send_static(parsed.path.removeprefix("/static/"))
        if parsed.path == "/api/system":
            return self._json(
                {
                    "repo_root": str(self.state.paths.repo_root),
                    "rsl_rl_log_root": str(self.state.paths.rsl_rl_log_root),
                    "default_task": DEFAULT_TASK,
                    "local_url_hint": "http://127.0.0.1:8080",
                    "lan_hint": "Run with --host 0.0.0.0 and open http://<machine-ip>:8080",
                    "ssh_tunnel_hint": "ssh -L 8080:127.0.0.1:8080 user@host",
                }
            )
        if parsed.path == "/api/training/defaults":
            return self._json(TrainingParams().to_dict())
        if parsed.path == "/api/runs":
            return self._json({"runs": self.state.history.list_runs()})
        if parsed.path.startswith("/api/runs/") and parsed.path.endswith("/notes"):
            run_id = parsed.path.split("/")[3]
            return self._json({"run_id": run_id, "notes": self.state.history.get_note(run_id)})
        if parsed.path == "/api/tweakables":
            return self._json(reward_file_index(self.state.paths.repo_root))
        if parsed.path == "/api/processes":
            return self._json({"processes": self.state.processes.list_processes()})
        self._not_found()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            payload = self._payload()
            if parsed.path == "/api/training/start":
                params = TrainingParams.from_dict(payload)
                return self._json(self.state.processes.start_training(params), status=201)
            if parsed.path == "/api/training/stop":
                run_id = str(payload.get("run_id") or "")
                return self._json({"stopped": self.state.processes.stop(run_id)})
            if parsed.path == "/api/tensorboard/start":
                host = str(payload.get("host") or "127.0.0.1")
                port = int(payload.get("port") or 6006)
                return self._json(self.state.processes.start_tensorboard(host=host, port=port))
            if parsed.path.startswith("/api/runs/") and parsed.path.endswith("/notes"):
                run_id = parsed.path.split("/")[3]
                self.state.history.set_note(run_id, str(payload.get("notes") or ""))
                return self._json({"saved": True, "run_id": run_id})
            if parsed.path.startswith("/api/runs/") and parsed.path.endswith("/play"):
                run_id = parsed.path.split("/")[3]
                run = self.state.history.get_run(run_id)
                if not run or not run.get("latest_checkpoint"):
                    return self._json({"error": "No checkpoint found for run"}, status=404)
                return self._json(
                    self.state.processes.start_play(
                        run_id=run_id,
                        checkpoint=str(run["latest_checkpoint"]),
                        device=str(payload.get("device") or "cuda:0"),
                    ),
                    status=201,
                )
        except ValueError as exc:
            return self._json({"error": str(exc)}, status=400)
        self._not_found()

    def log_message(self, fmt: str, *args) -> None:
        print(f"[training-panel] {self.address_string()} - {fmt % args}")

    def _payload(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8"))

    def _json(self, data: dict, status: int = 200) -> None:
        body = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_static(self, relative: str) -> None:
        path = (STATIC_DIR / relative).resolve()
        if not path.is_file() or STATIC_DIR not in path.parents:
            return self._not_found()
        body = path.read_bytes()
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _not_found(self) -> None:
        self._json({"error": "not found"}, status=404)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the RedRHex local training panel.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host. Use 0.0.0.0 for LAN access.")
    parser.add_argument("--port", type=int, default=8080, help="Bind port.")
    args = parser.parse_args()

    paths = PanelPaths.from_env()
    paths.ensure_dirs()
    PanelHandler.state = PanelState(paths)
    server = ThreadingHTTPServer((args.host, args.port), PanelHandler)
    print(f"RedRHex training panel: http://{args.host}:{args.port}")
    print("For SSH tunnel: ssh -L 8080:127.0.0.1:8080 user@host")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping RedRHex training panel.")
    finally:
        server.server_close()
