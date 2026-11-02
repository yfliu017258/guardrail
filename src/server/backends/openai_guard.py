"""Guard model served behind an OpenAI-compatible API (vLLM on the MIG slice, or any internal endpoint).

The backend is generic; what differs per model family is composed from two registries:

* ``prompt_mode`` -> a ``PromptBuilder`` (``prompts.py``): how the request is shaped and read back.
* ``parser`` + ``parser_options`` -> an ``OutputParser`` (``parsers.py``): text -> verdict, logprobs -> score.

The backend adds the model / sampling params, ``extra_body`` (e.g. ``chat_template_kwargs`` for
Nemotron-3.5 custom policies, with ``policy_as_template_kwarg`` injecting our policy there), retries,
and turns the parsed output + score into a ``Verdict``.

Errors: transport errors, 5xx and 429 are retried (``retries``); any other 4xx is a request the server
will reject again and is raised at once. A 400 saying the prompt exceeds the context (the char-based
token estimate was too low for this tokenizer) is retried once with a smaller budget; if that fails
too the verdict is an error (logged), so the stage's ``on_error`` decides — never a silent pass.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

import httpx

from ...config import env, scrub_url
from .base import BACKENDS, GuardBackend, GuardInput, Verdict
from .parsers import PARSERS, OutputParser
from .prompts import DEFAULT_MAX_TOKENS, PROMPT_BUILDERS, FittedContext, PromptBuilder

log = logging.getLogger("guardrail.openai_guard")

# vLLM / OpenAI wordings: "This model's maximum context length is 8192 tokens. However, you requested 9941
# tokens ...", "... your request has 9941 input tokens", "The decoder prompt (length 9941) is longer than
# the maximum model length of 8192"
_CONTEXT_ERROR = re.compile(r"context length|context window|maximum model length|max_model_len|"
                            r"longer than the maximum|prompt is too long", re.IGNORECASE)
_MAX_LEN = re.compile(r"maximum (?:context|model) length (?:is |of )?(\d+)", re.IGNORECASE)
_REQUESTED = re.compile(r"(?:you requested|request has|prompt \(length)\s*(\d+)", re.IGNORECASE)


class ContextLengthError(RuntimeError):
    """The server rejected the prompt as longer than the model context (HTTP 400)."""

    def __init__(self, detail: str):
        super().__init__(f"guard prompt exceeds the model's context length: {detail[:300]}")
        self.detail = detail

    @property
    def shrink(self) -> float:
        """Budget factor for the retry: at most half, less when the server says by how much we are over."""
        m, n = _MAX_LEN.search(self.detail), _REQUESTED.search(self.detail)
        if m and n and int(n.group(1)) > 0:
            return max(0.05, min(0.5, 0.9 * int(m.group(1)) / int(n.group(1))))
        return 0.5


def _error_text(r: httpx.Response) -> str:
    try:
        body = r.json()
    except ValueError:
        return r.text
    if isinstance(body, dict):
        err = body.get("error")
        msg = body.get("message") or (err.get("message") if isinstance(err, dict) else err)
        if msg:
            return str(msg)
    return r.text


@BACKENDS.register("openai_guard")
class OpenAIGuardBackend(GuardBackend):
    """Generative guard (Llama Guard, Qwen3Guard, Nemotron, LLM-as-judge ...) over ``/chat/completions`` or ``/completions``."""

    def __init__(self, name: str, cfg: dict[str, Any], root_cfg: dict[str, Any]):
        super().__init__(name, cfg, root_cfg)
        # deploy-time override: the env var (e.g. GUARD_MODEL_BASE_URL) wins over the preset's default base_url
        url = env(root_cfg, cfg.get("base_url_env")) or cfg.get("base_url")
        if not url:
            raise ValueError(f"stage {name!r}: base_url (or base_url_env) is required")
        self.base_url = url.rstrip("/")
        self.model = cfg["model"]
        self.api_key = env(root_cfg, cfg.get("api_key_env"), "EMPTY")
        self.parser: OutputParser = PARSERS.build(cfg, type_key="parser", default="llama_guard")
        self.prompt: PromptBuilder = PROMPT_BUILDERS.build(cfg, type_key="prompt_mode", default="chat_template")
        self.use_score = bool(cfg.get("use_score", True))
        self.request_logprobs = bool(cfg.get("logprobs", True))
        self.top_logprobs = int(cfg.get("top_logprobs", 5))
        self.gen_params = {
            "temperature": cfg.get("temperature", 0.0),
            "max_tokens": cfg.get("max_tokens", DEFAULT_MAX_TOKENS),
            **({"seed": cfg["seed"]} if "seed" in cfg else {}),
        }
        self.extra_body = self._extra_body()
        self.timeout = float(cfg.get("timeout", 15))
        self.retries = int(cfg.get("retries", 1))
        self._client: httpx.AsyncClient | None = None
        self._probe: httpx.AsyncClient | None = None  # readiness probes, separate from the request pool

    # -- lifecycle -------------------------------------------------------- #
    async def startup(self) -> None:
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout,
            headers={"Authorization": f"Bearer {self.api_key}"},
            limits=httpx.Limits(max_connections=int(self.cfg.get("max_connections", 64))),
        )

    async def ready(self) -> tuple[bool, str]:
        """The guard model's server answers ``GET /models`` and serves ``self.model``. Uses its own small
        client: a request pool saturated by real traffic must not make a healthy vLLM look down."""
        if self._client is None:
            return False, "not started"
        if self._probe is None:
            self._probe = httpx.AsyncClient(base_url=self.base_url, timeout=2.0,
                                            headers={"Authorization": f"Bearer {self.api_key}"})
        try:
            r = await self._probe.get("/models")
        except httpx.HTTPError as e:
            return False, f"{scrub_url(self.base_url)} unreachable: {type(e).__name__}"
        if r.status_code != 200:
            return False, f"{scrub_url(self.base_url)}/models -> HTTP {r.status_code}"
        served = [m.get("id") for m in (r.json().get("data") or []) if isinstance(m, dict)]
        if not served:
            return False, f"{scrub_url(self.base_url)} serves no models"
        if self.model not in served:
            return False, f"model {self.model!r} not served (server has {served})"
        return True, f"{self.model} @ {scrub_url(self.base_url)}"

    async def shutdown(self) -> None:
        for c in (self._client, self._probe):
            if c is not None:
                await c.aclose()

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "model": self.model, "parser": self.parser.registry_name,
                "prompt_mode": self.prompt.registry_name, "base_url": scrub_url(self.base_url)}

    # -- request ---------------------------------------------------------- #
    def _extra_body(self) -> dict[str, Any]:
        body = dict(self.cfg.get("extra_body", {}))
        # custom policy for models that take it as a chat-template kwarg (Nemotron-3.5-Content-Safety)
        kwarg = self.cfg.get("policy_as_template_kwarg")
        if kwarg and self.prompt.policy:
            body["chat_template_kwargs"] = {**body.get("chat_template_kwargs", {}), kwarg: self.prompt.policy}
        return body

    def build_payload(self, inp: GuardInput, budget_scale: float = 1.0) -> tuple[str, dict[str, Any]]:
        """``(route, json body)`` for one judged turn; logprobs only when the parser can score them."""
        route, payload, _ = self._request(inp, budget_scale)
        return route, payload

    def _request(self, inp: GuardInput, budget_scale: float) -> tuple[str, dict[str, Any], FittedContext]:
        fitted = self.prompt.fit(inp, budget_scale)
        payload = {"model": self.model, **self.gen_params, **self.extra_body, **self.prompt.body(inp, fitted.messages)}
        if self.request_logprobs and self.parser.score_spec(inp.input_type) is not None:
            payload.update(self.prompt.logprob_params(self.top_logprobs))
        return self.prompt.route, payload, fitted

    async def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        """POST with retries for transport errors, 5xx and 429 only; a context-length 400 raises
        ``ContextLengthError``, any other 4xx raises at once (resending the same request cannot help)."""
        assert self._client is not None, "startup() not called"
        for attempt in range(self.retries + 1):
            final = attempt == self.retries
            try:
                r = await self._client.post(path, json=payload)
            except httpx.TransportError:
                if final:
                    raise
            else:
                if r.status_code < 400:
                    return r.json()
                if r.status_code == 400 and _CONTEXT_ERROR.search(detail := _error_text(r)):
                    raise ContextLengthError(detail)
                if final or not (r.status_code >= 500 or r.status_code == 429):
                    r.raise_for_status()
            await asyncio.sleep(0.2 * 2**attempt)
        raise AssertionError("unreachable")  # the last attempt returns or raises

    MAX_CONTEXT_RETRIES = 3

    async def _complete(self, inp: GuardInput) -> tuple[dict[str, Any], FittedContext, bool]:
        """One completion; on "prompt too long" retry with a budget derived from what was actually SENT and
        the server's own over-by ratio (the char-based estimate was wrong, so shrinking the nominal budget
        alone may not shrink the prompt at all). Still too long after the retries -> error verdict."""
        scale, retried = 1.0, False
        for attempt in range(self.MAX_CONTEXT_RETRIES + 1):
            route, payload, fitted = self._request(inp, scale)
            try:
                return await self._post(route, payload), fitted, retried
            except ContextLengthError as e:
                if attempt == self.MAX_CONTEXT_RETRIES:
                    log.warning("stage %s: prompt still too long after %d retries (last ~%d estimated tokens); "
                                "returning an error verdict (on_error decides) - check max_model_len / max_input_tokens",
                                self.name, attempt, fitted.tokens)
                    raise
                sent_fraction = min(1.0, fitted.tokens / max(fitted.budget, 1))
                scale = scale * sent_fraction * e.shrink
                retried = True
                log.warning("stage %s: prompt rejected as too long (~%d estimated tokens sent, budget %d): %s; "
                            "retry %d with budget x%.3f", self.name, fitted.tokens, fitted.budget, e.detail[:200],
                            attempt + 1, scale)
        raise AssertionError("unreachable")

    # -- verdict ---------------------------------------------------------- #
    async def _classify(self, inp: GuardInput) -> Verdict:
        data, fitted, retried = await self._complete(inp)
        out = self.prompt.read_choice(data["choices"][0])
        parsed = self.parser.parse(out.text, inp.input_type)
        score = self.parser.score(out.logprobs, inp.input_type) if self.use_score else None
        # unparsable output is not a transport error: it is decided by `unparsable_as_unsafe`
        # and counted separately by the evaluator (parse_error_rate)
        extra: dict[str, Any] = {"hard_label_unsafe": parsed.unsafe, "parse_ok": parsed.parse_ok, **parsed.extra,
                                 "truncated": fitted.truncated, "context_retry": retried}
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
