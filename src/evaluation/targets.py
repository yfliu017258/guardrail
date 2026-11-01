"""Evaluation targets — *what* is being evaluated.

=========================  =====================================================
``inprocess``              build ``GuardPipeline`` from a server config, no HTTP.
                           Fastest loop for model / prompt / threshold sweeps.
``guardrail_server``       our uvicorn service, ``POST /v1/classify`` (per-stage detail).
``litellm_apply_guardrail`` the LiteLLM proxy's ``POST /guardrails/apply_guardrail``:
                           exercises the real proxy guardrail config, no LLM call.
``litellm_chat``           end-to-end ``/v1/chat/completions`` through the proxy with
                           ``guardrails=[...]``; response rows use ``mock_response`` so the
                           post_call guard sees the dataset's response. Measures real overhead.
=========================  =====================================================

All targets return ``Prediction``; transport failures raise ``TargetError`` (retried by the runner).
"""

from __future__ import annotations

import abc
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import httpx

from ..config import env, get_by_path, load_config
from ..data_schema import GuardrailSample


class TargetError(Exception):
    """Retryable transport / server failure (not a guardrail decision)."""


@dataclass
class Prediction:
    sample_id: str
    unsafe: bool | None
    score: float | None = None
    categories: list[str] = field(default_factory=list)
    blocked_by: list[str] = field(default_factory=list)
    action: str | None = None
    stage_unsafe: dict[str, bool] = field(default_factory=dict)
    stage_scores: dict[str, float | None] = field(default_factory=dict)
    latency_ms: float = 0.0           # client-observed
    server_latency_ms: float | None = None
    parse_ok: bool = True
    error: str | None = None
    raw: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _pred_from_pipeline_dict(sample_id: str, d: dict, latency_ms: float, save_raw: bool) -> Prediction:
    verdicts = d.get("verdicts", [])
    stage_unsafe: dict[str, bool] = {}
    stage_scores: dict[str, float | None] = {}
    raws, parse_ok = [], True
    for v in verdicts:  # several verdicts per stage when a response has multiple choices -> any()
        stage_unsafe[v["stage"]] = stage_unsafe.get(v["stage"], False) or bool(v["unsafe"])
        sc = v.get("score")
        if sc is not None:
            stage_scores[v["stage"]] = max(sc, stage_scores.get(v["stage"]) or 0.0)
        else:
            stage_scores.setdefault(v["stage"], None)
        parse_ok = parse_ok and v.get("extra", {}).get("parse_ok", True)
        if save_raw and v.get("raw"):
            raws.append(f"[{v['stage']}] {v['raw']}")
    errors = d.get("errors") or []
    return Prediction(
        sample_id=sample_id, unsafe=bool(d["unsafe"]), score=d.get("score"), categories=d.get("categories", []),
        blocked_by=d.get("blocked_by", []), action=d.get("action"), stage_unsafe=stage_unsafe,
        stage_scores=stage_scores, latency_ms=latency_ms, server_latency_ms=d.get("latency_ms"),
        parse_ok=parse_ok, error="; ".join(errors) or None, raw="\n".join(raws)[:4000] if raws else None,
    )


class Target(abc.ABC):
    name = "base"

    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        self.save_raw = bool(get_by_path(cfg, "eval.save_raw", True))
        self.timeout = float(get_by_path(cfg, "eval.timeout", 60))

    async def startup(self) -> None: ...

    async def shutdown(self) -> None: ...

    def describe(self) -> dict[str, Any]:
        return {"type": self.name}

    @abc.abstractmethod
    async def predict(self, s: GuardrailSample) -> Prediction: ...


class InProcessTarget(Target):
    name = "inprocess"

    def __init__(self, cfg: dict[str, Any]):
        super().__init__(cfg)
        from ..server.pipeline import GuardPipeline

        tcfg = get_by_path(cfg, "target.inprocess", {})
        self.server_cfg = load_config(
            tcfg.get("server_config", ["src/config/server/mock.yaml"]),
            overrides=tcfg.get("overrides", []),
            env_file=None,
            run_kind="server",
        )
        self.server_cfg["envs"] = cfg.get("envs", {})  # share the evaluation's .env
        self.pipeline = GuardPipeline(self.server_cfg)
        self.run_all = bool(tcfg.get("run_all", True))

    async def startup(self) -> None:
        await self.pipeline.startup()

    async def shutdown(self) -> None:
        await self.pipeline.shutdown()

    def describe(self) -> dict[str, Any]:
        from ..config import config_hash

        return {"type": self.name, "stages": self.pipeline.describe(), "server_config_hash": config_hash(self.server_cfg),
                "server_config": {k: v for k, v in self.server_cfg.items() if k not in ("envs", "meta")}}

    async def predict(self, s: GuardrailSample) -> Prediction:
        if s.input_type == "response":
            ctx = [{"role": "user", "content": s.prompt}] if s.prompt else []
            inputs = self.pipeline.build_inputs("response", [s.response or ""], ctx)
        else:
            inputs = self.pipeline.build_inputs("request", [s.prompt], [{"role": "user", "content": s.prompt}])
        t0 = time.perf_counter()
        res = await self.pipeline.run(inputs, run_all=self.run_all)
        return _pred_from_pipeline_dict(s.sample_id, res.to_dict(), (time.perf_counter() - t0) * 1000, self.save_raw)


class _HTTPTarget(Target):
    section = ""

    def __init__(self, cfg: dict[str, Any]):
        super().__init__(cfg)
        self.tcfg = get_by_path(cfg, f"target.{self.section}", {})
        self.base_url = (env(cfg, self.tcfg.get("base_url_env")) or self.tcfg.get("base_url", "")).rstrip("/")
        if not self.base_url:
            raise ValueError(f"target.{self.section}.base_url (or base_url_env) is required")
        self.api_key = env(cfg, self.tcfg.get("api_key_env"))
        self.client: httpx.AsyncClient | None = None

    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    async def startup(self) -> None:
        self.client = httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout, headers=self.headers(),
                                        verify=self.tcfg.get("verify_ssl", True))

    async def shutdown(self) -> None:
        if self.client:
            await self.client.aclose()

    def describe(self) -> dict[str, Any]:
        return {"type": self.name, "base_url": self.base_url, **{k: v for k, v in self.tcfg.items() if "key" not in k}}

    async def _post(self, path: str, payload: dict) -> tuple[httpx.Response, float]:
        assert self.client is not None
        t0 = time.perf_counter()
        try:
            r = await self.client.post(path, json=payload)
        except httpx.TransportError as e:
            raise TargetError(f"{type(e).__name__}: {e}") from e
        return r, (time.perf_counter() - t0) * 1000


class GuardrailServerTarget(_HTTPTarget):
    name = "guardrail_server"
    section = "guardrail_server"

    def headers(self) -> dict[str, str]:
        return {"x-api-key": self.api_key} if self.api_key else {}

    async def predict(self, s: GuardrailSample) -> Prediction:
        payload: dict[str, Any] = {"input_type": s.input_type, "run_all": self.tcfg.get("run_all", True),
                                   "include_raw": self.save_raw, "stages": self.tcfg.get("stages")}
        if s.input_type == "response":
            payload.update(text=s.response, prompt=s.prompt or None)
        else:
            payload["text"] = s.prompt
        r, ms = await self._post("/v1/classify", payload)
        if r.status_code >= 500 or r.status_code in (408, 429):
            raise TargetError(f"HTTP {r.status_code}: {r.text[:200]}")
        r.raise_for_status()
        return _pred_from_pipeline_dict(s.sample_id, r.json(), ms, self.save_raw)


class _LiteLLMBlockMixin:
    tcfg: dict

    def _is_block(self, r: httpx.Response) -> bool:
        codes = set(self.tcfg.get("blocked_status_codes", [400]))
        markers = [m.lower() for m in self.tcfg.get("blocked_markers", ["guardrail", "blocked", "violat", "攔截"])]
        return r.status_code in codes and any(m in r.text.lower() for m in markers)

    @staticmethod
    def _retryable(r: httpx.Response) -> bool:
        return r.status_code >= 500 or r.status_code in (408, 429)


class LiteLLMApplyGuardrailTarget(_LiteLLMBlockMixin, _HTTPTarget):
    name = "litellm_apply_guardrail"
    section = "litellm"

    async def predict(self, s: GuardrailSample) -> Prediction:
        is_resp = s.input_type == "response"
        name = self.tcfg.get("response_guardrail_name" if is_resp else "guardrail_name") or self.tcfg["guardrail_name"]
        payload = {"guardrail_name": name, "text": s.text, "input_type": s.input_type}
        r, ms = await self._post("/guardrails/apply_guardrail", payload)
        if self._is_block(r):
            return Prediction(s.sample_id, unsafe=True, action="BLOCKED", latency_ms=ms, raw=r.text[:1000] if self.save_raw else None)
        if self._retryable(r):
            raise TargetError(f"HTTP {r.status_code}: {r.text[:200]}")
        if r.status_code != 200:
            return Prediction(s.sample_id, unsafe=None, latency_ms=ms, error=f"HTTP {r.status_code}: {r.text[:200]}")
        out = r.json().get("response_text")
        action = "GUARDRAIL_INTERVENED" if (out is not None and out != s.text) else "NONE"
        return Prediction(s.sample_id, unsafe=False, action=action, latency_ms=ms)


class LiteLLMChatTarget(_LiteLLMBlockMixin, _HTTPTarget):
    name = "litellm_chat"
    section = "litellm"

    async def predict(self, s: GuardrailSample) -> Prediction:
        is_resp = s.input_type == "response"
        guardrails = self.tcfg.get("response_guardrails" if is_resp else "request_guardrails") or []
        payload: dict[str, Any] = {
            "model": self.tcfg["model"],
            "messages": [{"role": "user", "content": s.prompt or "."}],
            "max_tokens": int(self.tcfg.get("max_tokens", 1)),
            "temperature": 0,
        }
        if guardrails:
            payload["guardrails"] = guardrails
        if is_resp:  # make the "LLM" return the dataset response so the post_call guard judges it
            payload["mock_response"] = s.response
        r, ms = await self._post("/v1/chat/completions", payload)
        if self._is_block(r):
            return Prediction(s.sample_id, unsafe=True, action="BLOCKED", latency_ms=ms, raw=r.text[:1000] if self.save_raw else None)
        if self._retryable(r):
            raise TargetError(f"HTTP {r.status_code}: {r.text[:200]}")
        if r.status_code != 200:
            return Prediction(s.sample_id, unsafe=None, latency_ms=ms, error=f"HTTP {r.status_code}: {r.text[:200]}")
        return Prediction(s.sample_id, unsafe=False, action="NONE", latency_ms=ms)


TARGETS: dict[str, type[Target]] = {
    t.name: t for t in (InProcessTarget, GuardrailServerTarget, LiteLLMApplyGuardrailTarget, LiteLLMChatTarget)
}


def build_target(cfg: dict[str, Any]) -> Target:
    t = get_by_path(cfg, "target.type", "inprocess")
    if t not in TARGETS:
        raise ValueError(f"unknown target.type {t!r}; available: {sorted(TARGETS)}")
    return TARGETS[t](cfg)
