"""Thin client for the LiteLLM 1.92.0 admin API (guardrails, team/key attachment).

Endpoints (verified in litellm/proxy/guardrails/guardrail_endpoints.py and
management_endpoints/{team,key_management}_endpoints.py):

    GET    /v2/guardrails/list            DB guardrails (api_key masked); /guardrails/list = config-file only
    GET    /guardrails/{id}
    POST   /guardrails                    {"guardrail": {...}}  PROXY_ADMIN; initialised in memory immediately
    PUT    /guardrails/{id}               {"guardrail": {...}}  full replace
    PATCH  /guardrails/{id}               merge litellm_params
    DELETE /guardrails/{id}
    POST   /guardrails/apply_guardrail    {"guardrail_name", "text", "input_type"}  test without an LLM call
    GET    /team/info?team_id=            POST /team/update {"team_id", "guardrails": [...]}  (replaces the list)
    GET    /key/info?key=                 POST /key/update  {"key", "guardrails": [...]}

DB-backed guardrails are picked up by *other* proxy pods on their next DB poll
(``store_model_in_db``), not instantly — verify on every pod before relying on it.
"""

from __future__ import annotations

from typing import Any

import httpx


class LiteLLMAdminError(RuntimeError):
    def __init__(self, method: str, path: str, resp: httpx.Response):
        super().__init__(f"{method} {path} -> HTTP {resp.status_code}: {resp.text[:500]}")
        self.status_code = resp.status_code
        self.body = resp.text


class LiteLLMAdmin:
    def __init__(self, base_url: str, api_key: str, timeout: float = 30.0, verify: bool | str = True,
                 transport: httpx.BaseTransport | None = None):
        self.client = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout, verify=verify, transport=transport,
                                   headers={"Authorization": f"Bearer {api_key}"})

    def close(self) -> None:
        self.client.close()

    def _req(self, method: str, path: str, **kw: Any) -> Any:
        r = self.client.request(method, path, **kw)
        if r.status_code >= 400:
            raise LiteLLMAdminError(method, path, r)
        return r.json() if r.content else None

    # -- guardrails ------------------------------------------------------- #
    def list_guardrails(self) -> list[dict]:
        return (self._req("GET", "/v2/guardrails/list") or {}).get("guardrails", [])

    def get_guardrail(self, guardrail_id: str) -> dict:
        return self._req("GET", f"/guardrails/{guardrail_id}")

    def create_guardrail(self, guardrail: dict) -> dict:
        return self._req("POST", "/guardrails", json={"guardrail": guardrail})

    def update_guardrail(self, guardrail_id: str, guardrail: dict) -> dict:
        return self._req("PUT", f"/guardrails/{guardrail_id}", json={"guardrail": guardrail})

    def delete_guardrail(self, guardrail_id: str) -> Any:
        return self._req("DELETE", f"/guardrails/{guardrail_id}")

    def apply_guardrail(self, name: str, text: str, input_type: str = "request") -> httpx.Response:
        """Returns the raw response: a block surfaces as an HTTP error whose body names the guardrail."""
        return self.client.post("/guardrails/apply_guardrail",
                                json={"guardrail_name": name, "text": text, "input_type": input_type})

    # -- attachment ------------------------------------------------------- #
    def team_guardrails(self, team_id: str) -> list[str]:
        info = self._req("GET", "/team/info", params={"team_id": team_id}) or {}
        meta = (info.get("team_info") or info).get("metadata") or {}
        return list(meta.get("guardrails") or [])

    def set_team_guardrails(self, team_id: str, guardrails: list[str]) -> Any:
        return self._req("POST", "/team/update", json={"team_id": team_id, "guardrails": guardrails})

    def key_guardrails(self, key: str) -> list[str]:
        info = self._req("GET", "/key/info", params={"key": key}) or {}
        meta = (info.get("info") or info).get("metadata") or {}
        return list(meta.get("guardrails") or [])

    def set_key_guardrails(self, key: str, guardrails: list[str]) -> Any:
        return self._req("POST", "/key/update", json={"key": key, "guardrails": guardrails})
