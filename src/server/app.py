"""FastAPI guardrail service.

* ``POST /beta/litellm_basic_guardrail_api`` — called by LiteLLM's built-in
  ``generic_guardrail_api`` guardrail (pre_call / post_call / during_call).
* ``POST /v1/classify`` — same pipeline, detailed per-stage output (evaluation & debugging).
* ``GET /health``, ``GET /v1/info``, ``GET /v1/stats``.
"""

from __future__ import annotations

import hmac
import json
import logging
from collections import Counter
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException

from ..config import env, get_by_path
from .pipeline import ACTION_BLOCKED, GuardPipeline, PipelineResult
from .schemas import ClassifyRequest, ClassifyResponse, LiteLLMGuardrailRequest, LiteLLMGuardrailResponse

log = logging.getLogger("guardrail")

DEFAULT_BLOCK_MESSAGE = (
    "此請求可能違反公司 AI 使用規範（{categories}），已被系統攔截。"
    "如認為誤判，請附上 request id {call_id} 聯繫 AI 平台團隊。"
)


def create_app(cfg: dict[str, Any], pipeline: GuardPipeline | None = None) -> FastAPI:
    pipeline = pipeline or GuardPipeline(cfg)
    expected_key = env(cfg, get_by_path(cfg, "server.api_key_env"))
    block_template = get_by_path(cfg, "server.block_message", DEFAULT_BLOCK_MESSAGE)
    expose_categories = bool(get_by_path(cfg, "server.expose_categories", True))
    log_text = bool(get_by_path(cfg, "server.log_text", False))  # prompts may contain confidential data
    stats: Counter[str] = Counter()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        await pipeline.startup()
        log.info("guardrail pipeline ready: %s", json.dumps(pipeline.describe(), ensure_ascii=False))
        yield
        await pipeline.shutdown()

    app = FastAPI(title="OJT Guardrail Service", version="0.1.0", lifespan=lifespan)
    app.state.pipeline = pipeline
    app.state.cfg = cfg

    async def auth(x_api_key: str | None = Header(default=None), authorization: str | None = Header(default=None)) -> None:
        if not expected_key:
            return
        supplied = x_api_key or (authorization[7:] if authorization and authorization.lower().startswith("bearer ") else None)
        if not supplied or not hmac.compare_digest(supplied, expected_key):
            raise HTTPException(status_code=401, detail="invalid guardrail api key")

    def _log_decision(route: str, call_id: str | None, res: PipelineResult, input_type: str, text: str = "") -> None:
        stats[f"{input_type}:{res.action}"] += 1
        if res.errors:
            stats["errors"] += 1
        record = {
            "route": route, "call_id": call_id, "input_type": input_type, "action": res.action,
            "categories": res.categories, "blocked_by": res.blocked_by, "score": res.score,
            "latency_ms": round(res.latency_ms, 1), "errors": res.errors,
            "stages": {v.stage: {"unsafe": v.unsafe, "score": v.score, "ms": round(v.latency_ms, 1)} for v in res.verdicts},
        }
        if log_text:
            record["text"] = text[:500]
        log.info(json.dumps(record, ensure_ascii=False))

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/info", dependencies=[Depends(auth)])
    async def info() -> dict[str, Any]:
        return {"stages": pipeline.describe(), "strategy": pipeline.strategy, "request_scope": pipeline.request_scope,
                "config_hash": cfg.get("meta", {}).get("config_hash")}

    @app.get("/v1/stats", dependencies=[Depends(auth)])
    async def get_stats() -> dict[str, int]:
        return dict(stats)

    @app.post("/beta/litellm_basic_guardrail_api", response_model=LiteLLMGuardrailResponse,
              response_model_exclude_none=True, dependencies=[Depends(auth)])
    async def litellm_guardrail(req: LiteLLMGuardrailRequest) -> LiteLLMGuardrailResponse:
        inputs = pipeline.build_inputs(req.input_type, req.texts, req.structured_messages)
        res = await pipeline.run(inputs)
        _log_decision("litellm", req.litellm_call_id, res, req.input_type, inputs[-1].target_text)
        if res.action == ACTION_BLOCKED:
            cats = ", ".join(res.categories) if (expose_categories and res.categories) else "policy"
            return LiteLLMGuardrailResponse(
                action="BLOCKED",
                blocked_reason=block_template.format(categories=cats, call_id=req.litellm_call_id or "-"),
            )
        if res.masked_texts is not None and len(res.masked_texts) == len(req.texts or []):
            return LiteLLMGuardrailResponse(action="GUARDRAIL_INTERVENED", texts=res.masked_texts)
        return LiteLLMGuardrailResponse(action="NONE")

    @app.post("/v1/classify", response_model=ClassifyResponse, dependencies=[Depends(auth)])
    async def classify(req: ClassifyRequest) -> ClassifyResponse:
        if req.messages:
            msgs = req.messages
            texts = [m.get("content", "") for m in msgs if isinstance(m.get("content"), str)]
            if req.input_type == "response":
                inputs = pipeline.build_inputs("response", [msgs[-1].get("content", "")], msgs[:-1])
            else:
                inputs = pipeline.build_inputs("request", texts, msgs)
        elif req.text is not None:
            if req.input_type == "response":
                ctx = [{"role": "user", "content": req.prompt}] if req.prompt else []
                inputs = pipeline.build_inputs("response", [req.text], ctx)
            else:
                inputs = pipeline.build_inputs("request", [req.text], [{"role": "user", "content": req.text}])
        else:
            raise HTTPException(status_code=422, detail="provide `text` or `messages`")
        res = await pipeline.run(inputs, run_all=req.run_all, only=req.stages)
        _log_decision("classify", None, res, req.input_type)
        d = res.to_dict(include_raw=req.include_raw)
        return ClassifyResponse(**{k: d[k] for k in ClassifyResponse.model_fields if k in d})

    return app
