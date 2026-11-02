"""Backend contract: every guard (LLM guard, classifier, regex, mock) implements ``GuardBackend``.

Backends are classes in the ``BACKENDS`` registry; a pipeline stage's ``type`` picks one::

    @BACKENDS.register("my_guard")
    class MyGuard(GuardBackend):
        async def _classify(self, inp: GuardInput) -> Verdict: ...

    backend = build_backend({"name": "x", "type": "my_guard", ...}, root_cfg)

A backend living outside this package registers itself on import; add its module to ``BACKENDS``'
``lazy_modules`` (or call ``BACKENDS.add_lazy_module``) so configs can name it without an import.
"""

from __future__ import annotations

import abc
import asyncio
import time
from dataclasses import asdict, dataclass, field
from typing import Any, ClassVar, Literal

from ...common import Registry

InputType = Literal["request", "response"]
Message = dict[str, Any]  # OpenAI-style {"role": ..., "content": ...}


#: What part of a LiteLLM request a ``GuardInput`` judges (``GuardPipeline.build_inputs``).
#: ``turn``: the judged user / assistant turn; ``system``: client system / developer messages;
#: ``tool_output``: tool / function results after the newest user turn; ``assistant``: assistant text
#: after the newest user turn; ``tool_call``: a tool call's function name + arguments.
Origin = Literal["turn", "system", "tool_output", "assistant", "tool_call"]


@dataclass
class GuardInput:
    """What one backend call judges.

    ``messages`` is the conversation context (OpenAI format). The *judged* turn is the
    last message: a ``user`` turn for ``request``, an ``assistant`` turn for ``response``.
    ``texts`` are the raw strings LiteLLM sent; masking backends return a modified copy. Only the
    primary input of a request carries them; the extra inputs (system, tool output, tool calls) judge
    text that is either already in ``texts`` or (tool calls) cannot be returned to LiteLLM at all.
    """

    input_type: InputType
    messages: list[Message]
    texts: list[str] = field(default_factory=list)
    origin: Origin = "turn"

    @property
    def target_text(self) -> str:
        return _content_to_text(self.messages[-1].get("content")) if self.messages else ""

    @property
    def maskable(self) -> bool:
        """False when a redaction could not be applied: LiteLLM 1.92 never copies tool calls back from the
        guard response, so a mask hit in one must block instead."""
        return self.origin != "tool_call"


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

    registry_name: ClassVar[str] = "base"  # set by @BACKENDS.register; reported as the stage "type"

    def __init__(self, name: str, cfg: dict[str, Any], root_cfg: dict[str, Any]):
        self.name = name
        self.cfg = cfg
        self.root_cfg = root_cfg
        self.threshold: float = float(cfg.get("threshold", 0.5))

    @classmethod
    def from_config(cls, stage_cfg: dict[str, Any], root_cfg: dict[str, Any]) -> GuardBackend:
        """Registry factory: the stage ``name`` defaults to its ``type`` (one stage per type is the common case)."""
        return cls(stage_cfg.get("name", cls.registry_name), stage_cfg, root_cfg)

    async def startup(self) -> None:  # load weights / open clients
        return None

    async def ready(self) -> tuple[bool, str]:
        """Readiness for ``GET /ready``: can this stage answer right now? (``/health`` only says the process
        is up.) Backends with a remote dependency override this; local ones are ready once started."""
        return True, "ok"

    async def shutdown(self) -> None:
        return None

    @abc.abstractmethod
    async def _classify(self, inp: GuardInput) -> Verdict: ...

    async def classify(self, inp: GuardInput) -> Verdict:
        t0 = time.perf_counter()
        try:
            v = await self._classify(inp)
        except _PROPAGATE:
            raise
        except BaseException as e:  # noqa: BLE001 - never let one backend crash the request; on_error applies
            # BaseException, not Exception: the Rust tokenizers raise pyo3 PanicException, which derives
            # from BaseException (e.g. a pair window whose budget is below `stride`)
            v = Verdict(stage=self.name, unsafe=False, error=f"{type(e).__name__}: {e}")
        v.stage = self.name
        v.latency_ms = (time.perf_counter() - t0) * 1000
        return v

    @property
    def masks(self) -> bool:
        """True when this backend may return ``masked_texts`` (``GuardPipeline`` allows one such live stage)."""
        return False

    @property
    def type_name(self) -> str:
        """Pre-registry spelling of ``registry_name``, kept for out-of-tree backends that read it."""
        return self.registry_name

    def describe(self) -> dict[str, Any]:
        out = {"name": self.name, "type": self.registry_name, "threshold": self.threshold}
        if self.cfg.get("trained_on"):  # training-data families: evaluation excludes them from "clean" metrics
            out["trained_on"] = list(self.cfg["trained_on"])
        return out


#: Not backend failures: shutdown, Ctrl-C and task cancellation (request timeouts) must propagate.
_PROPAGATE = (KeyboardInterrupt, SystemExit, GeneratorExit, asyncio.CancelledError)


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #
# Built-in backends are imported on first lookup: configs can name any of them without the caller
# importing its module, and there is no import cycle (each backend module imports this one).
BACKENDS: Registry[GuardBackend] = Registry(
    "guard backend",
    base=GuardBackend,
    lazy_modules=[f"{__package__}.{m}" for m in ("regex", "hf_classifier", "openai_guard", "mock")],
)

#: Backwards-compatible spelling of ``@BACKENDS.register(name)``.
register_backend = BACKENDS.register


def build_backend(stage_cfg: dict[str, Any], root_cfg: dict[str, Any]) -> GuardBackend:
    """One pipeline stage from its config (``type`` selects the class); unknown types list the available ones."""
    return BACKENDS.build(stage_cfg, root_cfg)


def _content_to_text(content: Any) -> str:
    """OpenAI content may be a string or a list of parts; keep only text parts."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type", "text") == "text")
    return str(content)
