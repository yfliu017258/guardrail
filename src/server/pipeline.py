"""Stage orchestration: turn a LiteLLM request into GuardInputs, run stages, aggregate a decision."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any

from .backends.base import GuardBackend, GuardInput, InputType, Message, Verdict, _content_to_text, build_backend

ACTION_NONE = "NONE"
ACTION_BLOCKED = "BLOCKED"
ACTION_INTERVENED = "GUARDRAIL_INTERVENED"

log = logging.getLogger("guardrail.pipeline")


@dataclass
class Stage:
    backend: GuardBackend
    apply_to: tuple[str, ...] = ("request", "response")
    on_error: str = "fail_open"   # fail_open: treat as safe | fail_closed: treat as unsafe
    enabled: bool = True
    shadow: bool = False          # run + log, but never affects the decision (safe rollout of a new model)

    @property
    def name(self) -> str:
        return self.backend.name


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
    """Stages from ``pipeline.stages``; ``build_inputs`` decides what is judged, ``run`` decides the action.

    Checked when the pipeline is built:

    * **Stage names are unique.** Verdicts are attributed to stages by name (shadow flag, ``blocked_by``,
      logs, ``/v1/classify`` ``stages``, the evaluator's per-stage metrics). A shadow copy of a stage that
      kept the live stage's name — or two unnamed stages of one type, since ``name`` defaults to ``type`` —
      would swap their roles: the candidate blocking live traffic, or a live block dropped.
    * **At most one live masking stage per input type.** LiteLLM replaces ``texts`` positionally with ONE
      list, and every masking stage redacts the original texts independently, so a second stage's
      redactions would be silently dropped (merging them would depend on stage order and overlapping
      spans). Put every mask rule in one regex stage (inline ``rules`` + ``rules_file``).

    Shadow stages never decide. On the request path (``run_all=False``: the LiteLLM endpoint) they run as
    background tasks whose verdicts are logged (``"route": "shadow"``), so a slow or unreachable candidate
    adds no latency, and they run on every request, also those a live stage blocked (unbiased stats).
    ``run_all=True`` (``/v1/classify``, evaluation) awaits them and returns their verdicts.
    """

    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        pcfg = cfg.get("pipeline", {})
        self.strategy = pcfg.get("strategy", "sequential")  # sequential (short-circuit) | parallel
        self.request_scope = pcfg.get("request_scope", "last_user")  # last_user | all_user | all_texts
        # client system / developer messages are client-written on a shared proxy: judge them too
        self.judge_system_messages = bool(pcfg.get("judge_system_messages", True))
        # tool results / assistant text / tool calls after the newest user turn (indirect prompt injection)
        self.judge_tool_messages = bool(pcfg.get("judge_tool_messages", True))
        self.shadow_max_pending = int(pcfg.get("shadow_max_pending", 256))
        self.shadow_drain_timeout = float(pcfg.get("shadow_drain_timeout", 5.0))
        self.shadow_dropped = 0
        self._shadow_tasks: set[asyncio.Task] = set()
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
        self._check_stages()

    def _check_stages(self) -> None:
        dups = sorted(n for n, c in Counter(s.name for s in self.stages).items() if c > 1)
        if dups:
            raise ValueError(f"duplicate stage name(s) {dups}: verdicts are attributed to stages by name, so the "
                             "stages would take each other's shadow flag; give each stage a unique `name` "
                             "(it defaults to the stage type)")
        for input_type in ("request", "response"):
            maskers = [s.name for s in self.stages if s.backend.masks and not s.shadow and input_type in s.apply_to]
            if len(maskers) > 1:
                raise ValueError(f"at most one masking stage per input type; {input_type}: {maskers}. LiteLLM "
                                 "replaces `texts` with one list and each stage masks the original texts, so the "
                                 "other stages' redactions would be dropped: put all mask rules in one regex stage "
                                 "(inline `rules` + `rules_file`)")

    async def startup(self) -> None:
        await asyncio.gather(*(s.backend.startup() for s in self.stages))

    async def shutdown(self) -> None:
        await self.drain_shadow(self.shadow_drain_timeout)  # before their clients are closed
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
        tool_calls: list[dict[str, Any]] | None = None,
    ) -> list[GuardInput]:
        """LiteLLM 1.92 semantics (see llms/openai/chat/guardrail_translation/handler.py):

        * request: ``texts`` = every message's text (incl. system/tool messages unless the guardrail
          sets skip_system_message... / skip_tool_message...), ``structured_messages`` = the conversation,
          ``tool_calls`` = every assistant tool call in it.
        * response: ``texts`` = generated choice texts, ``tool_calls`` = generated tool calls; no prompt
          context is sent.

        The first input is the judged turn (it carries ``texts``, for masking). Extra inputs follow, each
        judged as a single user turn (guard templates and classifiers only judge user / assistant turns):
        system / developer content (``judge_system_messages``); tool / function results and assistant
        text after the newest user turn (``judge_tool_messages``: indirect prompt injection — earlier turns
        stay context, so old history does not re-block); tool calls as ``name(arguments)`` (requests: those
        after the newest user turn, plus any not found in the messages, also under ``judge_tool_messages``;
        responses: always, they are the model's output). A mask hit in a tool call blocks: LiteLLM cannot
        apply a redaction there.
        """
        texts = [t for t in (texts or []) if isinstance(t, str)]
        msgs = [m for m in (structured_messages or []) if isinstance(m, dict)]
        calls = [c for c in (tool_calls or []) if isinstance(c, dict)]
        if input_type == "response":
            ctx = [m for m in msgs if m.get("role") in ("user", "assistant")]
            inputs = [GuardInput("response", ctx + [{"role": "assistant", "content": t}], texts) for t in texts if t.strip()]
            inputs += [GuardInput("response", ctx + [{"role": "assistant", "content": t}], [], origin="tool_call")
                       for t in map(_tool_call_text, calls) if t.strip()]
            return inputs or [GuardInput("response", [{"role": "assistant", "content": ""}], texts)]

        joined = [{"role": "user", "content": "\n".join(texts)}]
        user_idx = [i for i, m in enumerate(msgs) if m.get("role") == "user"]
        if not msgs or not user_idx or self.request_scope == "all_texts":
            # every text (system / tool messages included) is the judged turn: only tool calls are left over.
            # No messages: /guardrails/apply_guardrail and SDK paths only send texts.
            primary = GuardInput("request", (msgs[:-1] if self.request_scope == "all_texts" else []) + joined, texts)
            return [primary, *self._tool_call_inputs(calls)] if self.judge_tool_messages else [primary]

        last = user_idx[-1]
        if self.request_scope == "all_user":
            judged = "\n".join(_content_to_text(msgs[i].get("content")) for i in user_idx)
            inputs = [GuardInput("request", msgs[:last] + [{"role": "user", "content": judged}], texts)]
        else:  # last_user: judge the newest user turn with prior turns as context
            inputs = [GuardInput("request", msgs[: last + 1], texts)]
        if self.judge_system_messages:
            system = "\n".join(t for m in msgs if m.get("role") in ("system", "developer")
                               if (t := _content_to_text(m.get("content"))).strip())
            if system:
                inputs.append(GuardInput("request", [{"role": "user", "content": system}], [], origin="system"))
        if self.judge_tool_messages:
            for m in msgs[last + 1:]:
                origin = {"tool": "tool_output", "function": "tool_output", "assistant": "assistant"}.get(m.get("role"))
                text = _content_to_text(m.get("content"))
                if origin and text.strip():
                    inputs.append(GuardInput("request", [{"role": "user", "content": text}], [], origin=origin))
            history = [c for m in msgs for c in _message_tool_calls(m)]
            recent = [c for m in msgs[last + 1:] for c in _message_tool_calls(m)]
            inputs += self._tool_call_inputs(recent + [c for c in calls if c not in history])
        return inputs

    @staticmethod
    def _tool_call_inputs(calls: list[dict[str, Any]]) -> list[GuardInput]:
        return [GuardInput("request", [{"role": "user", "content": t}], [], origin="tool_call")
                for t in map(_tool_call_text, calls) if t.strip()]

    # ------------------------------------------------------------------ #
    # run
    # ------------------------------------------------------------------ #
    async def run(self, inputs: list[GuardInput], run_all: bool = False, only: list[str] | None = None,
                  call_id: str | None = None) -> PipelineResult:
        t0 = time.perf_counter()
        input_type = inputs[0].input_type
        stages = [s for s in self.stages if input_type in s.apply_to and (not only or s.name in only)]
        live = [s for s in stages if not s.shadow]
        shadow = [s for s in stages if s.shadow]
        pairs: list[tuple[Stage, Verdict]] = []
        decided_unsafe = False

        if shadow and not run_all:
            self._spawn_shadow(shadow, inputs, call_id)  # off the request path; verdicts are logged
        if self.strategy == "sequential" and not run_all:
            for st in live:
                vs = await self._run_stage(st, inputs)
                pairs += [(st, v) for v in vs]
                if any(self._effective_unsafe(v, st) for v in vs):
                    decided_unsafe = True
                    break
        else:
            awaited = stages if run_all else live
            for st, vs in zip(awaited, await asyncio.gather(*(self._run_stage(st, inputs) for st in awaited))):
                pairs += [(st, v) for v in vs]

        live_pairs = [(st, v) for st, v in pairs if not st.shadow]
        blocking = [(st, v) for st, v in live_pairs if self._effective_unsafe(v, st)]
        decided_unsafe = decided_unsafe or bool(blocking)
        # The score is a max over calibrated stage scores; rule stages emit none (a mask hit is not a
        # probability of blocking). A block by a stage without a score is a hard decision: the row's score
        # is 1.0, so re-thresholding the pipeline score (evaluation --rescore) cannot undo it.
        scores = [v.score for _, v in live_pairs if v.score is not None and v.error is None]
        hard_block = any(v.score is None and v.error is None for _, v in blocking)
        masked = next((v.masked_texts for _, v in live_pairs if v.masked_texts), None)
        verdicts = [v for _, v in pairs]

        action = ACTION_BLOCKED if decided_unsafe else (ACTION_INTERVENED if masked else ACTION_NONE)
        return PipelineResult(
            action=action,
            unsafe=decided_unsafe,
            score=1.0 if hard_block else (max(scores) if scores else None),
            categories=sorted({c for _, v in blocking for c in v.categories}),
            blocked_by=sorted({st.name for st, _ in blocking}),
            verdicts=verdicts,
            latency_ms=(time.perf_counter() - t0) * 1000,
            masked_texts=masked if action == ACTION_INTERVENED else None,
            errors=[f"{v.stage}: {v.error}" for v in verdicts if v.error],
        )

    @staticmethod
    async def _run_stage(stage: Stage, inputs: list[GuardInput]) -> list[Verdict]:
        verdicts = list(await asyncio.gather(*(stage.backend.classify(i) for i in inputs)))
        for inp, v in zip(inputs, verdicts):
            if inp.origin != "turn":  # which part of the request an extra verdict judged (logs, /v1/classify)
                v.extra["origin"] = inp.origin
        return verdicts

    @staticmethod
    def _effective_unsafe(v: Verdict, st: Stage) -> bool:
        if v.error:
            return st.on_error == "fail_closed"
        return v.unsafe

    # ------------------------------------------------------------------ #
    # shadow stages
    # ------------------------------------------------------------------ #
    @property
    def pending_shadow(self) -> int:
        """Background shadow runs not finished yet."""
        return len(self._shadow_tasks)

    def _spawn_shadow(self, stages: list[Stage], inputs: list[GuardInput], call_id: str | None) -> None:
        if len(self._shadow_tasks) >= self.shadow_max_pending:  # a hung candidate must not pile up tasks
            self.shadow_dropped += 1
            log.warning("shadow stages %s skipped: %d runs still pending (pipeline.shadow_max_pending)",
                        [s.name for s in stages], len(self._shadow_tasks))
            return
        task = asyncio.create_task(self._run_shadow(stages, inputs, call_id))
        self._shadow_tasks.add(task)  # keep a reference: the loop only holds weak ones
        task.add_done_callback(self._shadow_tasks.discard)

    async def _run_shadow(self, stages: list[Stage], inputs: list[GuardInput], call_id: str | None) -> None:
        try:
            results = await asyncio.gather(*(self._run_stage(st, inputs) for st in stages))
        except Exception:  # noqa: BLE001 - classify() already captures backend errors; never crash the loop
            log.exception("shadow run failed (call_id=%s)", call_id)
            return
        record = {"route": "shadow", "call_id": call_id, "input_type": inputs[0].input_type, "stages": {
            st.name: {"unsafe": any(v.unsafe for v in vs if not v.error),
                      "score": max((v.score for v in vs if v.score is not None and not v.error), default=None),
                      "categories": sorted({c for v in vs if v.unsafe for c in v.categories}),
                      "errors": [v.error for v in vs if v.error],
                      "ms": round(max(v.latency_ms for v in vs), 1) if vs else 0.0}
            for st, vs in zip(stages, results)}}
        log.info(json.dumps(record, ensure_ascii=False))

    async def drain_shadow(self, timeout: float | None = None) -> None:
        """Wait for background shadow runs (shutdown, tests); cancel those still running after ``timeout``."""
        if not self._shadow_tasks:
            return
        tasks = set(self._shadow_tasks)
        _, pending = await asyncio.wait(tasks, timeout=timeout)
        for t in pending:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self._shadow_tasks.difference_update(tasks)


def _message_tool_calls(m: Message) -> list[dict[str, Any]]:
    """Tool calls of one message: ``tool_calls`` and the legacy ``function_call``."""
    calls = [c for c in (m.get("tool_calls") or []) if isinstance(c, dict)]
    if isinstance(m.get("function_call"), dict):
        calls.append({"type": "function", "function": m["function_call"]})
    return calls


def _tool_call_text(call: dict[str, Any]) -> str:
    """``name(arguments)`` with JSON arguments decoded, so escaped non-ASCII (``\\u4f60``) is judged as text."""
    fn = call.get("function") if isinstance(call.get("function"), dict) else call
    name = str(fn.get("name") or "")
    args = fn.get("arguments")
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            pass
    if args is None:
        args = ""
    elif not isinstance(args, str):
        args = json.dumps(args, ensure_ascii=False)
    return f"{name}({args})" if name else args
