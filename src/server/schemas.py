"""HTTP contracts.

``LiteLLMGuardrailRequest/Response`` mirror LiteLLM 1.92.0
``litellm/types/proxy/guardrails/guardrail_hooks/generic_guardrail_api.py`` — we do NOT
import litellm here (the guard service should not depend on the proxy package). If the
proxy is upgraded, diff that file and update these models; ``extra="allow"`` keeps us
forward-compatible with new optional fields.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class LiteLLMGuardrailRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    input_type: Literal["request", "response"]
    litellm_call_id: Optional[str] = None
    litellm_trace_id: Optional[str] = None
    structured_messages: Optional[list[dict[str, Any]]] = None
    images: Optional[list[str]] = None
    tools: Optional[list[dict[str, Any]]] = None
    texts: Optional[list[str]] = None
    request_data: dict[str, Any] = Field(default_factory=dict)   # user_api_key_* metadata
    request_headers: Optional[dict[str, str]] = None
    litellm_version: Optional[str] = None
    additional_provider_specific_params: Optional[dict[str, Any]] = None
    tool_calls: Optional[list[dict[str, Any]]] = None
    model: Optional[str] = None


class LiteLLMGuardrailResponse(BaseModel):
    """``action``: NONE (pass) | BLOCKED (proxy raises, client gets an error) |
    GUARDRAIL_INTERVENED (proxy replaces texts with ``texts``)."""

    action: Literal["NONE", "BLOCKED", "GUARDRAIL_INTERVENED"]
    blocked_reason: Optional[str] = None
    texts: Optional[list[str]] = None


class ClassifyRequest(BaseModel):
    """Internal endpoint used by the evaluator and for debugging (returns per-stage detail)."""

    input_type: Literal["request", "response"] = "request"
    text: Optional[str] = None                       # the judged text
    prompt: Optional[str] = None                     # context for response classification
    messages: Optional[list[dict[str, Any]]] = None  # alternative: full conversation
    tool_calls: Optional[list[dict[str, Any]]] = None  # OpenAI tool calls to judge (function name + arguments)
    run_all: bool = True                             # no short-circuit, shadow stages awaited: every stage's verdict
    stages: Optional[list[str]] = None               # restrict to these stage names
    include_raw: bool = True


class ClassifyResponse(BaseModel):
    action: str
    unsafe: bool
    score: Optional[float]
    categories: list[str]
    blocked_by: list[str]
    latency_ms: float
    verdicts: list[dict[str, Any]]
    errors: list[str]
    masked_texts: Optional[list[str]] = None
