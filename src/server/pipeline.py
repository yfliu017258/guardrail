"""Stage orchestration: turn a LiteLLM request into GuardInputs, run stages, aggregate a decision."""

from __future__ import annotations

import asyncio
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from .backends.base import GuardBackend, GuardInput, InputType, Message, Verdict, _content_to_text, build_backend

ACTION_NONE = "NONE"
ACTION_BLOCKED = "BLOCKED"
ACTION_INTERVENED = "GUARDRAIL_INTERVENED"


@dataclass
class Stage:
    backend: GuardBackend
    apply_to: tuple[str, ...] = ("request", "response")
    on_error: str = "fail_open"   # fail_open: treat as safe | fail_closed: treat as unsafe
    enabled: bool = True
    shadow: bool = False          # run + log, but never affects the decision (safe rollout of a new model)


@dataclass
class PipelineResult:
    action: str
    unsafe: bool
    score: float | None
    categories: list[str]
    blocked_by: list[str]
    verdicts: list[Verdict]
    latency_ms: float
    masked_texts: list[str] | None = None
    errors: list[str] = field(default_factory=list)

    def to_dict(self, include_raw: bool = True) -> dict[str, Any]:
        d = asdict(self)
        if not include_raw:
            for v in d["verdicts"]:
                v.pop("raw", None)
                v.get("extra", {}).pop("reasoning", None)
        return d


class GuardPipeline:
    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        pcfg = cfg.get("pipeline", {})
        self.strategy = pcfg.get("strategy", "sequential")  # sequential (short-circuit) | parallel
        self.request_scope = pcfg.get("request_scope", "last_user")  # last_user | all_user | all_texts
        self.stages: list[Stage] = []
        for scfg in pcfg.get("stages", []):
            if not scfg.get("enabled", True):
                continue
            self.stages.append(
                Stage(
                    backend=build_backend(scfg, cfg),
                    apply_to=tuple(scfg.get("apply_to", ["request", "response"])),
                    on_error=scfg.get("on_error", "fail_open"),
                    shadow=bool(scfg.get("shadow", False)),
                )
            )
        if not self.stages:
            raise ValueError("pipeline.stages has no enabled stage")

    async def startup(self) -> None:
        await asyncio.gather(*(s.backend.startup() for s in self.stages))

    async def shutdown(self) -> None:
        await asyncio.gather(*(s.backend.shutdown() for s in self.stages), return_exceptions=True)

    def describe(self) -> list[dict[str, Any]]:
        return [{**s.backend.describe(), "apply_to": list(s.apply_to), "on_error": s.on_error, "shadow": s.shadow}
                for s in self.stages]

    # ------------------------------------------------------------------ #
    # input shaping
    # ------------------------------------------------------------------ #
    def build_inputs(
        self,
        input_type: InputType,
        texts: list[str] | None,
        structured_messages: list[Message] | None,
    ) -> list[GuardInput]:
        """LiteLLM 1.92 semantics (see llms/openai/chat/guardrail_translation/handler.py):

        * request: ``texts`` = every message's text (incl. system/history unless the guardrail
          sets skip_system_message...), ``structured_messages`` = the conversation.
        * response: ``texts`` = generated choice texts; no prompt context is sent.
        """
        texts = [t for t in (texts or []) if isinstance(t, str)]
        msgs = [m for m in (structured_messages or []) if isinstance(m, dict)]
        if input_type == "response":
            ctx = [m for m in msgs if m.get("role") in ("user", "assistant")]
            return [GuardInput("response", ctx + [{"role": "assistant", "content": t}], texts) for t in texts if t.strip()] \
                or [GuardInput("response", [{"role": "assistant", "content": ""}], texts)]

        if not msgs:  # /guardrails/apply_guardrail and SDK paths only send texts
            joined = "\n".join(texts)
            return [GuardInput("request", [{"role": "user", "content": joined}], texts)]

        if self.request_scope == "all_texts":
            return [GuardInput("request", msgs[:-1] + [{"role": "user", "content": "\n".join(texts)}], texts)]
        user_idx = [i for i, m in enumerate(msgs) if m.get("role") == "user"]
        if not user_idx:
            return [GuardInput("request", [{"role": "user", "content": "\n".join(texts)}], texts)]
        if self.request_scope == "all_user":
            joined = "\n".join(_content_to_text(msgs[i].get("content")) for i in user_idx)
            return [GuardInput("request", msgs[: user_idx[-1]] + [{"role": "user", "content": joined}], texts)]
        last = user_idx[-1]  # last_user: judge the newest user turn with prior turns as context
        return [GuardInput("request", msgs[: last + 1], texts)]

    # ------------------------------------------------------------------ #
    # run
    # ------------------------------------------------------------------ #
    async def run(self, inputs: list[GuardInput], run_all: bool = False, only: list[str] | None = None) -> PipelineResult:
        t0 = time.perf_counter()
        input_type = inputs[0].input_type
        stages = [s for s in self.stages if input_type in s.apply_to and (not only or s.backend.name in only)]
        verdicts: list[Verdict] = []
        decided_unsafe = False
        sequential = self.strategy == "sequential" and not run_all

        async def run_stage(stage: Stage) -> list[Verdict]:
            return list(await asyncio.gather(*(stage.backend.classify(i) for i in inputs)))

        if sequential:
            for st in stages:
                vs = await run_stage(st)
                verdicts.extend(vs)
                if not st.shadow and any(self._effective_unsafe(v, st) for v in vs):
                    decided_unsafe = True
                    break
        else:
            for vs in await asyncio.gather(*(run_stage(st) for st in stages)):
                verdicts.extend(vs)

        stage_by_name = {s.backend.name: s for s in stages}
        live = [(v, stage_by_name[v.stage]) for v in verdicts if not stage_by_name[v.stage].shadow]
        blocked_by = sorted({v.stage for v, st in live if self._effective_unsafe(v, st)})
        decided_unsafe = decided_unsafe or bool(blocked_by)
        cats = sorted({c for v, st in live if self._effective_unsafe(v, st) for c in v.categories})
        scores = [v.score for v, _ in live if v.score is not None and v.error is None]
        masked = next((v.masked_texts for v, _ in live if v.masked_texts), None)

        action = ACTION_BLOCKED if decided_unsafe else (ACTION_INTERVENED if masked else ACTION_NONE)
        return PipelineResult(
            action=action,
            unsafe=decided_unsafe,
            score=max(scores) if scores else None,
            categories=cats,
            blocked_by=blocked_by,
            verdicts=verdicts,
            latency_ms=(time.perf_counter() - t0) * 1000,
            masked_texts=masked if action == ACTION_INTERVENED else None,
            errors=[f"{v.stage}: {v.error}" for v in verdicts if v.error],
        )

    @staticmethod
    def _effective_unsafe(v: Verdict, st: Stage) -> bool:
        if v.error:
            return st.on_error == "fail_closed"
        return v.unsafe
