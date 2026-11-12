"""Guardrail service adapters (registry ``ADAPTERS``) — the only code a new guardrail service needs.

An adapter translates between one service's HTTP API and the normalized result every metric and the
analysis app understand, which is the LiteLLM generic guardrail API response (checked against LiteLLM
1.98.1, ``litellm/types/proxy/guardrails/guardrail_hooks/generic_guardrail_api.py``)::

    action: NONE | BLOCKED | GUARDRAIL_INTERVENED,  texts,  blocked_reason

Adapters do no I/O: the ``http_guardrail`` target owns the connection, timeouts, retries and timing, so
an adapter is two pure functions and can be unit-tested with recorded JSON::

    @ADAPTERS.register("my_service")
    class MyServiceAdapter(GuardrailAdapter):
        def build_request(self, sample, input_type):
            return AdapterRequest(path="/v2/moderate", json={"input": self.judged_text(sample, input_type)})

        def parse_response(self, status, headers, body):
            return GuardrailResult(action="BLOCKED" if body["flagged"] else "NONE", score=body.get("prob"))

======================  ==========================================================================
``generic_guardrail_api``  a service that already speaks the LiteLLM generic guardrail API
                           (``POST /beta/litellm_basic_guardrail_api``) — nothing to write.
======================  ==========================================================================
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Mapping

from ..common import Registry
from .dataset import EvalSample

__all__ = ["ADAPTERS", "ACTIONS", "INPUT_TYPES", "AdapterParseError", "AdapterRequest", "GuardrailAdapter",
           "GuardrailResult", "GenericGuardrailAPIAdapter", "build_adapter"]

ACTIONS = ("NONE", "BLOCKED", "GUARDRAIL_INTERVENED")
INPUT_TYPES = ("request", "response")
#: the path LiteLLM appends to a generic guardrail's ``api_base``
GENERIC_GUARDRAIL_PATH = "/beta/litellm_basic_guardrail_api"


class AdapterParseError(ValueError):
    """The service answered, but not with something the adapter can turn into a decision. The row is
    recorded as an error (never scored as 'pass') and is not retried."""


@dataclass
class AdapterRequest:
    path: str
    json: dict[str, Any]
    method: str = "POST"
    headers: dict[str, str] = field(default_factory=dict)


@dataclass
class GuardrailResult:
    """Normalized decision. ``action`` / ``texts`` / ``blocked_reason`` are the LiteLLM contract; the rest
    is optional detail an adapter passes on when its service has it."""

    action: str
    texts: list[str] | None = None        # GUARDRAIL_INTERVENED: replacement texts, aligned with the request's
    blocked_reason: str | None = None
    score: float | None = None            # P(unsafe) in [0, 1]; enables ROC / threshold analysis
    categories: list[str] = field(default_factory=list)
    raw: Any = None                       # the service's own answer, kept for debugging

    def __post_init__(self) -> None:
        if self.action not in ACTIONS:
            raise AdapterParseError(f"unknown action {self.action!r} (expected one of {ACTIONS})")


class GuardrailAdapter(abc.ABC):
    """Translation layer for one guardrail service. Subclasses register in ``ADAPTERS``."""

    registry_name = "base"

    def __init__(self, cfg: dict[str, Any] | None = None):
        self.cfg = dict(cfg or {})

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "GuardrailAdapter":
        return cls(cfg)

    def describe(self) -> dict[str, Any]:
        return {"type": self.registry_name, **{k: v for k, v in self.cfg.items() if k != "type"}}

    @staticmethod
    def judged_text(sample: EvalSample, input_type: str) -> str:
        """The text the guardrail decides on: the user prompt, or the model answer in response direction."""
        if input_type == "response":
            if sample.response is None:
                raise ValueError(f"{sample.sample_id}: input_type=response needs a `response` column in the data")
            return sample.response
        return sample.prompt

    def auth_headers(self, api_key: str | None) -> dict[str, str]:
        """How this service takes its API key (default: Bearer)."""
        return {"Authorization": f"Bearer {api_key}"} if api_key else {}

    @abc.abstractmethod
    def build_request(self, sample: EvalSample, input_type: str) -> AdapterRequest: ...

    @abc.abstractmethod
    def parse_response(self, status: int, headers: Mapping[str, str], body: Any) -> GuardrailResult:
        """Called for 2xx answers only. Raise ``AdapterParseError`` when no decision can be read."""

    def is_retryable(self, status: int, body: Any) -> bool:
        """Transient failures the runner retries (anything else non-2xx is recorded as an error)."""
        return status >= 500 or status in (408, 429)


ADAPTERS: Registry[GuardrailAdapter] = Registry("guardrail service adapter", base=GuardrailAdapter)


@ADAPTERS.register("generic_guardrail_api", aliases=["litellm"])
class GenericGuardrailAPIAdapter(GuardrailAdapter):
    """A service that already speaks the LiteLLM (1.98) generic guardrail API — no code needed."""

    registry_name = "generic_guardrail_api"

    def auth_headers(self, api_key: str | None) -> dict[str, str]:
        return {"x-api-key": api_key} if api_key else {}  # what LiteLLM sends for `api_key`

    def build_request(self, sample: EvalSample, input_type: str) -> AdapterRequest:
        """The body LiteLLM builds for one chat turn (fields LiteLLM leaves null are omitted)."""
        messages: list[dict[str, Any]] = [{"role": "user", "content": sample.prompt}]
        body: dict[str, Any] = {
            "input_type": input_type,
            "texts": [self.judged_text(sample, input_type)],
            "structured_messages": messages,
            "request_data": dict(self.cfg.get("request_data") or {}),
            "litellm_call_id": f"eval-{sample.sample_id}",
            "litellm_version": str(self.cfg.get("litellm_version", "1.98.1")),
            "additional_provider_specific_params": dict(self.cfg.get("additional_provider_specific_params") or {}),
        }
        if self.cfg.get("model"):
            body["model"] = self.cfg["model"]
        return AdapterRequest(path=self.cfg.get("path", GENERIC_GUARDRAIL_PATH), json=body)

    def parse_response(self, status: int, headers: Mapping[str, str], body: Any) -> GuardrailResult:
        if not isinstance(body, dict):
            raise AdapterParseError(f"expected a JSON object, got {type(body).__name__}")
        if "action" not in body:
            # LiteLLM itself would read this as NONE; an evaluation must not score it as a real 'pass'
            raise AdapterParseError("response has no `action` field")
        texts = body.get("texts")
        if texts is not None and not (isinstance(texts, list) and all(isinstance(t, str) for t in texts)):
            raise AdapterParseError("`texts` must be a list of strings")
        # not part of the LiteLLM contract, but passed on when a service adds them
        score = body.get("score")
        cats = body.get("categories")
        return GuardrailResult(
            action=body["action"], texts=texts, blocked_reason=body.get("blocked_reason"),
            score=float(score) if isinstance(score, (int, float)) and not isinstance(score, bool) else None,
            categories=[str(c) for c in cats] if isinstance(cats, list) else [], raw=body,
        )


def build_adapter(cfg: dict[str, Any] | str | None) -> GuardrailAdapter:
    """``adapter: generic_guardrail_api`` or ``adapter: {type: my_service, ...options}``."""
    if cfg is None:
        cfg = {"type": "generic_guardrail_api"}
    if isinstance(cfg, str):
        cfg = {"type": cfg}
    return ADAPTERS.build(cfg)
