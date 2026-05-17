from __future__ import annotations

import json
from urllib.error import HTTPError
from urllib.parse import quote, urlencode
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
            missing = ", ".join(config.missing_required_env)
            raise ValueError(f"Supabase remote config is incomplete. Missing: {missing}")
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
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RemoteAPIError(f"HTTP {exc.code} for {method.upper()} {base}: {detail}") from exc
        except Exception as exc:  # urllib exposes several network/HTTP exception types.
            raise RemoteAPIError(str(exc)) from exc
        return json.loads(raw) if raw else None

    def storage_request(
        self,
        method: str,
        path: str,
        body: bytes | None = None,
        *,
        content_type: str = "application/octet-stream",
        extra_headers: dict[str, str] | None = None,
    ):
        base = f"{self.config.supabase_url}/storage/v1/{path.lstrip('/')}"
        headers = {
            "apikey": self.config.supabase_anon_key,
            "Authorization": f"Bearer {self.config.machine_token}",
            "Content-Type": content_type,
            **(extra_headers or {}),
        }
        request = Request(base, data=body, method=method.upper(), headers=headers)
        try:
            with urlopen(request, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RemoteAPIError(f"HTTP {exc.code} for {method.upper()} {base}: {detail}") from exc
        except Exception as exc:
            raise RemoteAPIError(str(exc)) from exc
        return json.loads(raw) if raw else None

    def function_request(self, name: str, payload: dict | None = None):
        url = f"{self.config.supabase_url}/functions/v1/{quote(name.strip('/'), safe='')}"
        data = json.dumps(payload or {}).encode("utf-8")
        request = Request(url, data=data, method="POST", headers=self._headers())
        try:
            with urlopen(request, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RemoteAPIError(f"HTTP {exc.code} for POST {url}: {detail}") from exc
        except Exception as exc:
            raise RemoteAPIError(str(exc)) from exc
        return json.loads(raw) if raw else None

    def upload_storage_object(
        self,
        bucket: str,
        object_path: str,
        file_path,
        *,
        content_type: str = "video/mp4",
        upsert: bool = True,
    ):
        data = file_path.read_bytes()
        encoded_bucket = quote(str(bucket).strip("/"), safe="")
        encoded_path = quote(str(object_path).strip("/"), safe="/")
        return self.storage_request(
            "POST",
            f"object/{encoded_bucket}/{encoded_path}",
            body=data,
            content_type=content_type,
            extra_headers={"x-upsert": "true" if upsert else "false"},
        )

    def select(self, table: str, query: dict | None = None):
        return self.request("GET", table, query=query)

    def insert(self, table: str, payload: dict | list, prefer: str = "return=representation"):
        return self.request("POST", table, body=payload, prefer=prefer)

    def update(self, table: str, payload: dict, query: dict, prefer: str = "return=representation"):
        return self.request("PATCH", table, body=payload, query=query, prefer=prefer)

    def delete(self, table: str, query: dict, prefer: str = "return=minimal"):
        return self.request("DELETE", table, query=query, prefer=prefer)

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
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RemoteAPIError(f"HTTP {exc.code} for POST {url}: {detail}") from exc
        except Exception as exc:
            raise RemoteAPIError(str(exc)) from exc
        return json.loads(raw) if raw else None

    def heartbeat(self, payload: dict):
        return self.upsert("machines", payload)

    def claim_next_job(self, machine_id: str, gpu_locked: bool = False):
        try:
            claimed = self.rpc(
                "claim_next_job_for_machine",
                {"p_machine_id": machine_id, "p_gpu_locked": bool(gpu_locked)},
            )
        except RemoteAPIError as exc:
            # Compatibility for projects that have not applied the V2.1 RPC yet.
            # When the GPU is busy we cannot safely use the old unfiltered RPC,
            # because it may claim another GPU job instead of leaving it queued.
            if gpu_locked:
                return None
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

    def mark_job_running(self, job_id: str):
        return self.update(
            "jobs",
            {"status": "running"},
            {"id": f"eq.{job_id}"},
        )

    def fail_job(self, job_id: str, message: str, result: dict | None = None):
        return self.update(
            "jobs",
            {"status": "failed", "error": message, "result": result or {}},
            {"id": f"eq.{job_id}"},
        )
