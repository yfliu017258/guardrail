"""Shared base for judges that call an OpenAI-compatible ``/chat/completions`` endpoint.

Both the LLM-as-judge (``llm``) and the guard-model judge (``qwen3guard_refusal``) talk to the same
kind of server (vLLM directly, or the LiteLLM proxy) and differ only in *what they send* and *how
they read the answer*. ``ChatCompletionJudge`` is a template method::

    _judge(prompt, response)
        -> build_messages(prompt, response)      # subclass
        -> chat(messages)                        # here: payload, retries, logprobs
        -> parse_reply(reply, prompt, response)  # subclass

Failures (transport errors, 5xx after retries, 4xx) are raised as ``JudgeHTTPError``;
``RefusalJudge.judge`` turns them into ``verdict.error`` so an eval run never crashes on them and
the task's error policy decides what an errored row means.
"""

from __future__ import annotations

import abc
import asyncio
import math
import re
from dataclasses import dataclass
from typing import Any, Mapping

import httpx

from ..config import env, scrub_url
from .base import RefusalJudge, RefusalVerdict

Message = dict[str, Any]
RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}


class JudgeHTTPError(RuntimeError):
    """The judge endpoint did not return a usable completion."""


@dataclass
class ChatReply:
    text: str
    finish_reason: str | None = None
    logprobs: list[dict[str, Any]] | None = None  # OpenAI ``choices[0].logprobs.content``


def truncate_middle(text: str, max_chars: int | None) -> str:
    """Keep head and tail: a refusal is usually at the head, a late "however, here is ..." at the tail."""
    if not max_chars or len(text) <= max_chars:
        return text
    half = max_chars // 2
    return f"{text[:half]}\n...[truncated {len(text) - 2 * half} chars]...\n{text[-half:]}"


def label_probs_from_logprobs(content: list[dict[str, Any]] | None, anchor: str,
                              label_words: Mapping[str, tuple[str, ...]]) -> dict[str, float] | None:
    """Distribution over labels read from the first label token after ``anchor``.

    Walks the generated tokens; at the first position where the text so far ends with ``anchor``,
    each top-k alternative counts toward the label whose word it is a prefix of (``"ref"`` ->
    refusal), so multi-token label words still work. ``None`` when the server returned no logprobs
    or the anchor never appeared — callers then fall back to the hard label.
    """
    if not content:
        return None
    pattern = re.compile(anchor + r"\Z", re.DOTALL)
    so_far = ""
    for pos in content:
        if pattern.search(so_far):
            alts = pos.get("top_logprobs") or [{"token": pos.get("token", ""), "logprob": pos.get("logprob", 0.0)}]
            probs: dict[str, float] = {}
            for alt in alts:
                tok = str(alt.get("token", "")).strip().strip("\"'").lower()
                if not tok:
                    continue
                for label, words in label_words.items():
                    if any(w.startswith(tok) for w in words):
                        probs[label] = probs.get(label, 0.0) + math.exp(float(alt.get("logprob", -math.inf)))
                        break
            if probs:
                return probs
        so_far += str(pos.get("token", ""))
    return None


class ChatCompletionJudge(RefusalJudge):
    """Abstract: connection settings, client lifecycle and retries for chat-completion judges.

    Config: ``base_url`` / ``base_url_env``, ``api_key_env`` (resolved with ``src.config.env``, so the
    key comes from ``.env`` or the process env, never from YAML), ``model``, ``temperature``,
    ``max_tokens``, ``seed``, ``timeout``, ``retries``, ``retry_backoff``, ``extra_body`` (e.g.
    ``chat_template_kwargs``), ``logprobs`` / ``top_logprobs``, ``max_connections``, ``verify_ssl``.
    """

    DEFAULT_MAX_TOKENS = 256

    #: Test seam: an ``httpx`` transport used instead of the network (``httpx.MockTransport``).
    transport: httpx.AsyncBaseTransport | None = None

    def __init__(self, cfg: dict[str, Any], root_cfg: dict[str, Any] | None = None):
        super().__init__(cfg, root_cfg)
        self.base_url = (env(self.root_cfg, self.cfg.get("base_url_env")) or self.cfg.get("base_url") or "").rstrip("/")
        if not self.base_url:
            raise ValueError(f"{self.registry_name} judge needs base_url or base_url_env")
        if not self.cfg.get("model"):
            raise ValueError(f"{self.registry_name} judge needs model")
        self.model: str = self.cfg["model"]
        self.api_key = env(self.root_cfg, self.cfg.get("api_key_env"), "EMPTY")
        self.timeout = float(self.cfg.get("timeout", 60))
        self.retries = int(self.cfg.get("retries", 2))
        self.retry_backoff = float(self.cfg.get("retry_backoff", 0.5))
        self.request_logprobs = bool(self.cfg.get("logprobs", False))
        self.top_logprobs = int(self.cfg.get("top_logprobs", 5))
        self.extra_body: dict[str, Any] = dict(self.cfg.get("extra_body") or {})
        self._client: httpx.AsyncClient | None = None

    # -- lifecycle -------------------------------------------------------- #
    async def startup(self) -> None:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url, timeout=self.timeout, transport=self.transport,
                headers={"Authorization": f"Bearer {self.api_key}"},
                limits=httpx.Limits(max_connections=int(self.cfg.get("max_connections", 64))),
                verify=self.cfg.get("verify_ssl", True),
            )

    async def shutdown(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "model": self.model, "base_url": scrub_url(self.base_url),
                "temperature": self.cfg.get("temperature", 0.0), "max_tokens": self.max_tokens,
                "logprobs": self.request_logprobs, "extra_body": self.extra_body}

    # -- request ---------------------------------------------------------- #
    @property
    def max_tokens(self) -> int:
        return int(self.cfg.get("max_tokens", self.DEFAULT_MAX_TOKENS))

    def payload(self, messages: list[Message]) -> dict[str, Any]:
        body: dict[str, Any] = {"model": self.model, "messages": messages,
                                "temperature": self.cfg.get("temperature", 0.0), "max_tokens": self.max_tokens}
        if self.cfg.get("seed") is not None:
            body["seed"] = self.cfg["seed"]
        if self.request_logprobs:
            body.update(logprobs=True, top_logprobs=self.top_logprobs)
        body.update(self.extra_body)
        return body

    async def chat(self, messages: list[Message]) -> ChatReply:
        if self._client is None:
            raise JudgeHTTPError("startup() not called")
        body = self.payload(messages)
        last = ""
        for attempt in range(self.retries + 1):
            if attempt:
                await asyncio.sleep(self.retry_backoff * 2 ** (attempt - 1))
            try:
                r = await self._client.post("/chat/completions", json=body)
            except httpx.TransportError as e:
                last = f"{type(e).__name__}: {e}"
                continue
            if r.status_code in RETRYABLE_STATUS:
                last = f"HTTP {r.status_code}: {r.text[:200]}"
                continue
            if r.status_code != 200:
                raise JudgeHTTPError(f"HTTP {r.status_code}: {r.text[:300]}")
            return self._reply(r)
        raise JudgeHTTPError(f"gave up after {self.retries + 1} attempts: {last}")

    @staticmethod
    def _reply(r: httpx.Response) -> ChatReply:
        try:
            choice = r.json()["choices"][0]
        except (ValueError, KeyError, IndexError, TypeError) as e:
            raise JudgeHTTPError(f"malformed completion: {r.text[:200]}") from e
        msg = choice.get("message") or {}
        return ChatReply(text=msg.get("content") or "", finish_reason=choice.get("finish_reason"),
                         logprobs=(choice.get("logprobs") or {}).get("content"))

    # -- template method -------------------------------------------------- #
    @abc.abstractmethod
    def build_messages(self, prompt: str, response: str) -> list[Message]: ...

    @abc.abstractmethod
    def parse_reply(self, reply: ChatReply, prompt: str, response: str) -> RefusalVerdict: ...

    async def _judge(self, prompt: str, response: str) -> RefusalVerdict:
        reply = await self.chat(self.build_messages(prompt, response))
        return self.parse_reply(reply, prompt, response)
