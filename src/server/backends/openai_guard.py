"""Guard model served behind an OpenAI-compatible API (vLLM on the MIG slice, or any internal endpoint).

The backend is generic; what differs per model family is composed from two registries:

* ``prompt_mode`` -> a ``PromptBuilder`` (``prompts.py``): how the request is shaped and read back.
* ``parser`` + ``parser_options`` -> an ``OutputParser`` (``parsers.py``): text -> verdict, logprobs -> score.

The backend adds the model / sampling params, ``extra_body`` (e.g. ``chat_template_kwargs`` for
Nemotron-3.5 custom policies, with ``policy_as_template_kwarg`` injecting our policy there), retries,
and turns the parsed output + score into a ``Verdict``.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx

from ...config import env
from .base import BACKENDS, GuardBackend, GuardInput, Verdict
from .parsers import PARSERS, OutputParser
from .prompts import PROMPT_BUILDERS, PromptBuilder


@BACKENDS.register("openai_guard")
class OpenAIGuardBackend(GuardBackend):
    """Generative guard (Llama Guard, Qwen3Guard, Nemotron, LLM-as-judge ...) over ``/chat/completions`` or ``/completions``."""

    def __init__(self, name: str, cfg: dict[str, Any], root_cfg: dict[str, Any]):
        super().__init__(name, cfg, root_cfg)
        self.base_url = cfg["base_url"].rstrip("/")
        self.model = cfg["model"]
        self.api_key = env(root_cfg, cfg.get("api_key_env"), "EMPTY")
        self.parser: OutputParser = PARSERS.build(cfg, type_key="parser", default="llama_guard")
        self.prompt: PromptBuilder = PROMPT_BUILDERS.build(cfg, type_key="prompt_mode", default="chat_template")
        self.use_score = bool(cfg.get("use_score", True))
        self.request_logprobs = bool(cfg.get("logprobs", True))
        self.top_logprobs = int(cfg.get("top_logprobs", 5))
        self.gen_params = {
            "temperature": cfg.get("temperature", 0.0),
            "max_tokens": cfg.get("max_tokens", 64),
            **({"seed": cfg["seed"]} if "seed" in cfg else {}),
        }
        self.extra_body = self._extra_body()
        self.timeout = float(cfg.get("timeout", 15))
        self.retries = int(cfg.get("retries", 1))
        self._client: httpx.AsyncClient | None = None

    # -- lifecycle -------------------------------------------------------- #
    async def startup(self) -> None:
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout,
            headers={"Authorization": f"Bearer {self.api_key}"},
            limits=httpx.Limits(max_connections=int(self.cfg.get("max_connections", 64))),
        )

    async def shutdown(self) -> None:
        if self._client:
            await self._client.aclose()

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "model": self.model, "parser": self.parser.registry_name,
                "prompt_mode": self.prompt.registry_name, "base_url": self.base_url}

    # -- request ---------------------------------------------------------- #
    def _extra_body(self) -> dict[str, Any]:
        body = dict(self.cfg.get("extra_body", {}))
        # custom policy for models that take it as a chat-template kwarg (Nemotron-3.5-Content-Safety)
        kwarg = self.cfg.get("policy_as_template_kwarg")
        if kwarg and self.prompt.policy:
            body["chat_template_kwargs"] = {**body.get("chat_template_kwargs", {}), kwarg: self.prompt.policy}
        return body

    def build_payload(self, inp: GuardInput) -> tuple[str, dict[str, Any]]:
        """``(route, json body)`` for one judged turn; logprobs only when the parser can score them."""
        payload = {"model": self.model, **self.gen_params, **self.extra_body, **self.prompt.build(inp)}
        if self.request_logprobs and self.parser.score_spec(inp.input_type) is not None:
            payload.update(self.prompt.logprob_params(self.top_logprobs))
        return self.prompt.route, payload

    async def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        assert self._client is not None, "startup() not called"
        last: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                r = await self._client.post(path, json=payload)
                if r.status_code >= 500 and attempt < self.retries:
                    raise httpx.HTTPStatusError("retryable", request=r.request, response=r)
                r.raise_for_status()
                return r.json()
            except (httpx.TransportError, httpx.HTTPStatusError) as e:
                last = e
                if attempt < self.retries:
                    await asyncio.sleep(0.2 * 2**attempt)
        raise last  # type: ignore[misc]

    # -- verdict ---------------------------------------------------------- #
    async def _classify(self, inp: GuardInput) -> Verdict:
        route, payload = self.build_payload(inp)
        data = await self._post(route, payload)
        out = self.prompt.read_choice(data["choices"][0])
        parsed = self.parser.parse(out.text, inp.input_type)
        score = self.parser.score(out.logprobs, inp.input_type) if self.use_score else None
        # unparsable output is not a transport error: it is decided by `unparsable_as_unsafe`
        # and counted separately by the evaluator (parse_error_rate)
        extra: dict[str, Any] = {"hard_label_unsafe": parsed.unsafe, "parse_ok": parsed.parse_ok, **parsed.extra}
        if score is not None and parsed.label_overridden:
            # the score measures the model's label, which an option overrode (e.g. ignored category):
            # keep it for analysis, but let the score agree with the verdict so the threshold cannot undo it
            extra["label_score"] = score
            score = float(parsed.unsafe)
        # If we have a calibrated score, the threshold decides; the hard label is kept in `extra`.
        unsafe = parsed.unsafe if score is None else score >= self.threshold
        return Verdict(
            stage=self.name,
            unsafe=unsafe,
            score=score,
            categories=(parsed.categories or ["other"]) if unsafe else [],
            native_categories=parsed.native_categories,
            reason=parsed.reason,
            raw=out.text[:2000],
            extra={**extra, "usage": data.get("usage", {}), "reasoning": out.reasoning},
        )
