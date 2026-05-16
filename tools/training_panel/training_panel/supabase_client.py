from __future__ import annotations

import json
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .remote_config import RemoteConfig


class RemoteAPIError(RuntimeError):
    pass


class SupabaseClient:
    """Small PostgREST/RPC client used by the local worker.

    The browser-facing GitHub Pages app uses the public anon key directly.
    This worker client uses the machine token as the bearer token and never
    exposes it to static frontend assets.
    """

    def __init__(self, config: RemoteConfig, timeout: float = 20.0):
        if not config.configured:
            raise ValueError("Supabase remote config is incomplete")
        self.config = config
        self.timeout = timeout

    def _headers(self, prefer: str | None = None) -> dict[str, str]:
        headers = {
            "apikey": self.config.supabase_anon_key,
            "Authorization": f"Bearer {self.config.machine_token}",
            "Content-Type": "application/json",
        }
        if prefer:
            headers["Prefer"] = prefer
        return headers

    def request(
        self,
        method: str,
        path: str,
        body: dict | list | None = None,
        query: dict | None = None,
        prefer: str | None = None,
    ):
        base = f"{self.config.supabase_url}/rest/v1/{path.lstrip('/')}"
        if query:
            base = f"{base}?{urlencode(query, doseq=True)}"
        data = json.dumps(body).encode("utf-8") if body is not None else None
        request = Request(base, data=data, method=method.upper(), headers=self._headers(prefer=prefer))
        try:
            with urlopen(request, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8")
        except Exception as exc:  # urllib exposes several network/HTTP exception types.
            raise RemoteAPIError(str(exc)) from exc
        return json.loads(raw) if raw else None

    def select(self, table: str, query: dict | None = None):
        return self.request("GET", table, query=query)

    def insert(self, table: str, payload: dict | list, prefer: str = "return=representation"):
        return self.request("POST", table, body=payload, prefer=prefer)

    def update(self, table: str, payload: dict, query: dict, prefer: str = "return=representation"):
        return self.request("PATCH", table, body=payload, query=query, prefer=prefer)

    def upsert(
        self,
        table: str,
        payload: dict | list,
        prefer: str = "resolution=merge-duplicates,return=representation",
        query: dict | None = None,
    ):
        return self.request("POST", table, body=payload, prefer=prefer, query=query)

    def rpc(self, name: str, payload: dict | None = None):
        url = f"{self.config.supabase_url}/rest/v1/rpc/{name}"
        data = json.dumps(payload or {}).encode("utf-8")
        request = Request(url, data=data, method="POST", headers=self._headers())
        try:
            with urlopen(request, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8")
        except Exception as exc:
            raise RemoteAPIError(str(exc)) from exc
        return json.loads(raw) if raw else None

    def heartbeat(self, payload: dict):
        return self.upsert("machines", payload)

    def claim_next_job(self, machine_id: str):
        claimed = self.rpc("claim_next_job_for_machine", {"p_machine_id": machine_id})
        if isinstance(claimed, list):
            return claimed[0] if claimed else None
        return claimed

    def complete_job(self, job_id: str, result: dict):
        return self.update(
            "jobs",
            {"status": "completed", "result": result},
            {"id": f"eq.{job_id}"},
        )

    def fail_job(self, job_id: str, message: str, result: dict | None = None):
        return self.update(
            "jobs",
            {"status": "failed", "error": message, "result": result or {}},
            {"id": f"eq.{job_id}"},
        )
