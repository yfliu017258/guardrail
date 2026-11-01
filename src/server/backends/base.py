"""Backend contract: every guard (LLM guard, classifier, regex, mock) implements ``GuardBackend``."""

from __future__ import annotations

import abc
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Literal

InputType = Literal["request", "response"]
Message = dict[str, Any]  # OpenAI-style {"role": ..., "content": ...}


@dataclass
class GuardInput:
    """What one backend call judges.

    ``messages`` is the conversation context (OpenAI format). The *judged* turn is the
    last message: a ``user`` turn for ``request``, an ``assistant`` turn for ``response``.
    ``texts`` are the raw strings LiteLLM sent; masking backends return a modified copy.
    """

    input_type: InputType
    messages: list[Message]
    texts: list[str] = field(default_factory=list)

    @property
    def target_text(self) -> str:
        return _content_to_text(self.messages[-1].get("content")) if self.messages else ""


@dataclass
class Verdict:
    stage: str
    unsafe: bool
    score: float | None = None            # P(unsafe) in [0, 1] when the backend can produce one
    categories: list[str] = field(default_factory=list)  # unified Category values
    native_categories: list[str] = field(default_factory=list)  # backend's own labels (S1, "Violent", ...)
    reason: str = ""
    raw: str = ""                         # raw model output (truncated), for error analysis
    latency_ms: float = 0.0
    error: str | None = None
    masked_texts: list[str] | None = None  # set by masking backends (regex PII) -> GUARDRAIL_INTERVENED
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class GuardBackend(abc.ABC):
    """Subclasses implement ``_classify``; ``classify`` adds timing and error capture."""

    type_name: str = "base"

    def __init__(self, name: str, cfg: dict[str, Any], root_cfg: dict[str, Any]):
        self.name = name
        self.cfg = cfg
        self.root_cfg = root_cfg
        self.threshold: float = float(cfg.get("threshold", 0.5))

    async def startup(self) -> None:  # load weights / open clients
        return None

    async def shutdown(self) -> None:
        return None

    @abc.abstractmethod
    async def _classify(self, inp: GuardInput) -> Verdict: ...

    async def classify(self, inp: GuardInput) -> Verdict:
        t0 = time.perf_counter()
        try:
            v = await self._classify(inp)
        except Exception as e:  # never let one backend crash the request; the pipeline applies on_error
            v = Verdict(stage=self.name, unsafe=False, error=f"{type(e).__name__}: {e}")
        v.stage = self.name
        v.latency_ms = (time.perf_counter() - t0) * 1000
        return v

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "type": self.type_name, "threshold": self.threshold}


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #
BACKENDS: dict[str, type[GuardBackend]] = {}


def register_backend(type_name: str) -> Callable[[type[GuardBackend]], type[GuardBackend]]:
    def deco(cls: type[GuardBackend]) -> type[GuardBackend]:
        cls.type_name = type_name
        BACKENDS[type_name] = cls
        return cls

    return deco


def build_backend(stage_cfg: dict[str, Any], root_cfg: dict[str, Any]) -> GuardBackend:
    from . import hf_classifier, mock, openai_guard, regex  # noqa: F401  (populate registry)

    t = stage_cfg.get("type")
    if t not in BACKENDS:
        raise ValueError(f"unknown backend type {t!r}; available: {sorted(BACKENDS)}")
    return BACKENDS[t](stage_cfg.get("name", t), stage_cfg, root_cfg)


def _content_to_text(content: Any) -> str:
    """OpenAI content may be a string or a list of parts; keep only text parts."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type", "text") == "text")
    return str(content)
