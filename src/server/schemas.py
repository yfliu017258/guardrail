"""HTTP contracts.

``LiteLLMGuardrailRequest/Response`` mirror LiteLLM 1.98.1
``litellm/types/proxy/guardrails/guardrail_hooks/generic_guardrail_api.py`` (the generic guardrail
API wire format) — the server does NOT depend on litellm. If callers move to a newer protocol version,
diff that file and update these models; ``extra="allow"`` keeps us
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
    """``action``: NONE (pass) | BLOCKED (the caller rejects the request) |
    GUARDRAIL_INTERVENED (the caller replaces its texts with ``texts``, positionally)."""

    action: Literal["NONE", "BLOCKED", "GUARDRAIL_INTERVENED"]
    blocked_reason: Optional[str] = None
    texts: Optional[list[str]] = None
    # accepted by LiteLLM >= 1.98 (unused by this server; responses are serialized without nulls)
    images: Optional[list[str]] = None
    tools: Optional[list[dict[str, Any]]] = None
    stream_holdback_chars: Optional[list[int]] = None


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
    shadow_stages: list[str] = Field(default_factory=list)  # stages whose verdicts are logged but never decide
