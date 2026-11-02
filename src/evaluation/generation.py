"""Bare-model generation for ``model_safety`` evaluation.

A *generator* sends a dataset prompt to the model under test and returns what the user would see
(plus reasoning tokens, kept separately). Generations are expensive, so they are stored on every
prediction row together with a ``fingerprint`` of the generation settings; a later run can reuse
them (``task.model_safety.reuse_generations_from``) to try a different refusal judge without
regenerating — and refuses to mix generations made with different settings.

    GENERATORS.build({"type": "openai_chat", "model": "qwen3.6-27b", ...}, root_cfg)
"""

from __future__ import annotations

import abc
import asyncio
import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import httpx

from ..common import Registry
from ..config import env
from .errors import TargetError

_THINK = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)


@dataclass
class Generation:
    text: str                              # user-visible answer (reasoning removed)
    reasoning: str | None = None           # thinking tokens, if any (never judged)
    finish_reason: str | None = None
    blocked: bool = False                  # rejected by a gateway / guardrail in front of the model
    block_message: str | None = None
    latency_ms: float = 0.0
    usage: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Generator(abc.ABC):
    registry_name = "base"
    #: settings that change the output; hashed into ``fingerprint`` (subclasses extend)
    FINGERPRINT_KEYS: tuple[str, ...] = ("type",)

    def __init__(self, cfg: dict[str, Any], root_cfg: dict[str, Any] | None = None):
        self.cfg = dict(cfg)
        self.root_cfg = root_cfg or {}

    @classmethod
    def from_config(cls, cfg: dict[str, Any], root_cfg: dict[str, Any] | None = None) -> "Generator":
        return cls(cfg, root_cfg)

    @property
    def fingerprint(self) -> str:
        body = {k: self.cfg.get(k) for k in self.FINGERPRINT_KEYS}
        return hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()[:12]

    def describe(self) -> dict[str, Any]:
        return {"type": self.registry_name, "fingerprint": self.fingerprint,
                **{k: self.cfg.get(k) for k in self.FINGERPRINT_KEYS if k != "type"}}

    async def startup(self) -> None: ...

    async def shutdown(self) -> None: ...

    @abc.abstractmethod
    async def generate(self, prompt: str) -> Generation: ...


GENERATORS: Registry[Generator] = Registry("model generator", base=Generator)


def split_reasoning(content: str, reasoning: str | None) -> tuple[str, str | None]:
    """Servers without a reasoning parser inline ``<think>..</think>``; move it out of the answer."""
    found = _THINK.findall(content or "")
    if found:
        content = _THINK.sub("", content)
        reasoning = "\n".join(filter(None, [reasoning, *found]))
    if "<think>" in (content or "").lower():  # unterminated: generation ran out inside the reasoning
        i = content.lower().index("<think>")
        reasoning = "\n".join(filter(None, [reasoning, content[i + 7:]]))
        content = content[:i]
    return (content or "").strip(), reasoning


@GENERATORS.register("openai_chat")
class OpenAIChatGenerator(Generator):
    """Any OpenAI-compatible chat endpoint (the model's vLLM server).

    Call the model directly for a *bare-model* eval. If a gateway in front of it rejects a request
    (HTTP 400 whose body matches ``blocked_markers``), the row is recorded as ``blocked`` and, by
    default, counted as a refusal.
    """

    FINGERPRINT_KEYS = ("type", "model", "system_prompt", "temperature", "top_p", "max_tokens", "seed", "extra_body")

    def __init__(self, cfg: dict[str, Any], root_cfg: dict[str, Any] | None = None):
        super().__init__(cfg, root_cfg)
        self.base_url = (env(self.root_cfg, cfg.get("base_url_env")) or cfg.get("base_url", "")).rstrip("/")
        if not self.base_url:
            raise ValueError("generator needs base_url or base_url_env")
        self.api_key = env(self.root_cfg, cfg.get("api_key_env"), "EMPTY")
        self.model = cfg["model"]
        self.timeout = float(cfg.get("timeout", 180))
        self.blocked_status = set(cfg.get("blocked_status_codes", [400]))
        self.blocked_markers = [m.lower() for m in cfg.get("blocked_markers", ["guardrail", "blocked", "violat", "攔截"])]
        self._client: httpx.AsyncClient | None = None
        self._sem = asyncio.Semaphore(int(cfg.get("max_concurrency", 64)))

    async def startup(self) -> None:
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout,
                                         headers={"Authorization": f"Bearer {self.api_key}"},
                                         verify=self.cfg.get("verify_ssl", True))

    async def shutdown(self) -> None:
        if self._client:
            await self._client.aclose()

    def payload(self, prompt: str) -> dict[str, Any]:
        msgs = []
        if self.cfg.get("system_prompt"):
            msgs.append({"role": "system", "content": self.cfg["system_prompt"]})
        msgs.append({"role": "user", "content": prompt})
        body: dict[str, Any] = {"model": self.model, "messages": msgs,
                                "temperature": self.cfg.get("temperature", 0.0),
                                "max_tokens": int(self.cfg.get("max_tokens", 1024))}
        for k in ("top_p", "seed"):
            if self.cfg.get(k) is not None:
                body[k] = self.cfg[k]
        body.update(self.cfg.get("extra_body") or {})
        return body

    async def generate(self, prompt: str) -> Generation:
        assert self._client is not None, "startup() not called"
        t0 = time.perf_counter()
        async with self._sem:
            try:
                r = await self._client.post("/chat/completions", json=self.payload(prompt))
            except httpx.TransportError as e:
                raise TargetError(f"{type(e).__name__}: {e}") from e
        ms = (time.perf_counter() - t0) * 1000
        if r.status_code in self.blocked_status and any(m in r.text.lower() for m in self.blocked_markers):
            return Generation(text="", blocked=True, block_message=r.text[:500], latency_ms=ms)
        if r.status_code >= 500 or r.status_code in (408, 429):
            raise TargetError(f"HTTP {r.status_code}: {r.text[:200]}")
        if r.status_code != 200:
            return Generation(text="", latency_ms=ms, error=f"HTTP {r.status_code}: {r.text[:300]}")
        data = r.json()
        choice = data["choices"][0]
        msg = choice.get("message") or {}
        reasoning = msg.get("reasoning_content") or msg.get("reasoning")
        text, reasoning = split_reasoning(msg.get("content") or "", reasoning)
        return Generation(text=text, reasoning=reasoning, finish_reason=choice.get("finish_reason"),
                          latency_ms=ms, usage=data.get("usage") or {})


@GENERATORS.register("mock")
class MockGenerator(Generator):
    """Deterministic fake model for tests / CI: refuses when the prompt contains a trigger word."""

    FINGERPRINT_KEYS = ("type", "refuse_on", "refusal_text", "comply_text")
    DEFAULT_TRIGGERS = ("bomb", "ransomware", "勒索", "炸彈", "steal", "jailbreak", "忽略之前")

    async def generate(self, prompt: str) -> Generation:
        triggers = self.cfg.get("refuse_on") or self.DEFAULT_TRIGGERS
        if any(t.lower() in prompt.lower() for t in triggers):
            text = self.cfg.get("refusal_text", "抱歉，我無法協助這個請求。")
        else:
            text = self.cfg.get("comply_text", "Sure, here is an answer: ...")
        return Generation(text=text, reasoning="<mock reasoning>", finish_reason="stop", latency_ms=0.1)


def build_generator(cfg: dict[str, Any], root_cfg: dict[str, Any] | None = None) -> Generator:
    return GENERATORS.build(cfg, root_cfg)
