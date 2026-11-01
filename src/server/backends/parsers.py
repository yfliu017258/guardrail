"""Output parsers for generative guard models.

Every guard family answers in its own text format (``unsafe\\nS1``, ``Safety: Unsafe``, JSON,
``<score>yes</score>`` ...). An ``OutputParser`` turns that text into a ``ParsedOutput`` (hard label +
unified categories) and declares where P(unsafe) can be read from token logprobs (``score_spec``), so
the evaluator can sweep thresholds / draw PR curves instead of relying on a single hard label.

Parsers are classes in the ``PARSERS`` registry, selected and configured by a guard stage::

    parser: qwen3guard                                   # -> Qwen3GuardParser
    parser_options: {controversial_as_unsafe: false}     # -> Qwen3GuardParser(options)

Add a model family: subclass ``OutputParser``, implement ``_parse``, set the class-level knobs
(``SCORE_SPEC``, ``CATEGORY_MAP``, ``OPTIONS`` ...) and decorate it with ``@PARSERS.register("name")``.
The base already handles reasoning blocks, ``unparsable_as_unsafe``, ``category_map`` and
``ignore_categories``, and rejects unknown options. Build verdicts with ``flagged`` / ``cleared`` /
``unparsable`` so ignored categories behave the same in every family (they never block, score or not).
"""

from __future__ import annotations

import abc
import json
import math
import re
from dataclasses import dataclass, field
from itertools import accumulate
from typing import Any, ClassVar, Iterable, Mapping

from ...common import Registry
from ...data_schema.taxonomy import AEGIS_MAP, LLAMA_GUARD_MAP, QWEN3GUARD_MAP
from .base import InputType


@dataclass
class ParsedOutput:
    unsafe: bool
    native_categories: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    reason: str = ""
    parse_ok: bool = True
    # True when the verdict does not rest on the model's scored safe/unsafe label alone (an ignored
    # category cleared it, an adversarial flag raised it): the logprob score must not overturn it.
    label_overridden: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# logprob scoring
# --------------------------------------------------------------------------- #
def _matches(token: str, words: tuple[str, ...]) -> bool:
    t = token.strip().lower()
    return bool(t) and any(w.startswith(t) for w in words)


@dataclass(frozen=True)
class ScoreSpec:
    """Where to read P(unsafe) from logprobs.

    ``anchor``: regex that the *generated text so far* must match (at its end) for the next token to
    be the label token. ``unsafe``/``safe``: words whose leading token counts toward each side
    (prefix match, case-insensitive). Parsers return a spec per input type / options, so one class
    covers e.g. both the ``User Safety:`` and ``Response Safety:`` positions.
    """

    anchor: str
    unsafe: tuple[str, ...]
    safe: tuple[str, ...]

    def score(self, content_logprobs: list[dict[str, Any]] | None) -> float | None:
        """P(unsafe) from an OpenAI-style ``choices[0].logprobs.content`` list.

        Walks generated tokens; at the first position where the answer generated so far matches
        ``anchor`` and the top-k alternatives contain a label token, returns
        ``sum p(unsafe tokens) / (sum p(unsafe) + sum p(safe))``. ``None`` if not found (e.g. logprobs
        disabled) — the caller then falls back to the hard label. Anchoring matters: Qwen3Guard's
        first token "Saf" (of "Safety:") must not be read as the label "Safe". Two kinds of position
        are never the label:

        * a sampled token that is only whitespace (Llama Guard's leading ``"\\n\\n"``, the ``" "`` an
          o200k tokenizer emits after ``"violation":``) — anchors ending in ``\\s*$`` already match there,
          and tiny label alternatives at that step would decide the score;
        * anything inside a ``<think>`` block, which ``split_reasoning`` also keeps out of the verdict.
        """
        if not content_logprobs:
            return None
        tokens = [str(pos.get("token") or "") for pos in content_logprobs]
        starts = list(accumulate((len(t) for t in tokens), initial=0))
        full = "".join(tokens)
        lo, hi = _answer_span(full)
        anchor = re.compile(self.anchor + r"\Z", re.DOTALL)
        for pos, tok, start in zip(content_logprobs, tokens, starts):
            if start < lo or start >= hi or not tok.strip():
                continue
            if anchor.search(full[lo:start]):
                # read the distribution only where the model actually emitted a label token; at any other
                # anchored position (an opening quote, a markdown star) the label alternatives are tiny
                # and would decide the score by noise
                if not (_matches(tok, self.unsafe) or _matches(tok, self.safe)):
                    continue
                alts = pos.get("top_logprobs") or [{"token": tok, "logprob": pos.get("logprob", 0.0)}]
                p_u = sum(math.exp(a["logprob"]) for a in alts if _matches(a["token"], self.unsafe))
                p_s = sum(math.exp(a["logprob"]) for a in alts
                          if _matches(a["token"], self.safe) and not _matches(a["token"], self.unsafe))
                if p_u + p_s > 0:
                    return p_u / (p_u + p_s)
        return None


# --------------------------------------------------------------------------- #
# shared helpers
# --------------------------------------------------------------------------- #
_THINK_BLOCK = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)


def split_reasoning(text: str) -> tuple[str, str]:
    """``(answer, thinking)``: a ``<think>`` block is never the verdict, so parsers only see the answer.

    Handles the three shapes served models produce: full ``<think>..</think>`` blocks; only the closing
    tag (the chat template opened ``<think>`` inside the prompt); and an unterminated ``<think>`` (cut by
    ``max_tokens`` before any verdict — the answer is then empty, i.e. unparsable).
    """
    thinking = [m.strip() for m in _THINK_BLOCK.findall(text)]
    answer = _THINK_BLOCK.sub("", text)
    low = answer.lower()
    if "</think>" in low:
        i = low.rindex("</think>")
        thinking.append(answer[:i].strip())
        answer = answer[i + len("</think>"):]
    elif "<think>" in low:
        i = low.index("<think>")
        thinking.append(answer[i + len("<think>"):].strip())
        answer = answer[:i]
    return answer, "\n".join(t for t in thinking if t)


def _answer_span(text: str) -> tuple[int, int]:
    """Character span of ``text`` outside reasoning, mirroring ``split_reasoning``: after the last
    ``</think>`` (full blocks, or a block the chat template opened), or before an unterminated ``<think>``."""
    low = text.lower()
    if "</think>" in low:
        return low.rindex("</think>") + len("</think>"), len(text)
    if "<think>" in low:
        return 0, low.index("<think>")
    return 0, len(text)


def _first_json(text: str) -> dict | None:
    for m in re.finditer(r"\{.*?\}", text, re.DOTALL):
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
    try:  # greedy fallback for nested objects
        s, e = text.index("{"), text.rindex("}")
        return json.loads(text[s : e + 1])
    except (ValueError, json.JSONDecodeError):
        return None


def _labels(value: Any, sep: str = r",") -> list[str]:
    """Native labels from a comma-separated string or a JSON list (models emit both); never raises.

    Free-form judges also emit a bare number (``"category": 3``), mixed lists or objects: everything is
    coerced to strings, because a category-format problem must not turn a detected violation into an
    error (which fails open).
    """
    if not value:
        return []
    if isinstance(value, str):
        items = re.split(sep, value)
    elif isinstance(value, (list, tuple, set, dict)):
        items = [str(v) for v in value if v is not None]
    else:
        items = [str(value)]
    return [c.strip() for c in items if c.strip()]


# --------------------------------------------------------------------------- #
# base
# --------------------------------------------------------------------------- #
#: Options of parsers whose output carries native category labels.
TAXONOMY_OPTIONS = frozenset({"category_map", "ignore_categories"})


class OutputParser(abc.ABC):
    """Guard output text -> ``ParsedOutput``; subclasses implement ``_parse`` on the reasoning-free answer.

    Class-level knobs a subclass sets:

    * ``OPTIONS``: accepted ``parser_options`` keys. Unknown keys raise, because a typo in a
      safety-relevant option (``controversal_as_unsafe``) would otherwise silently change what is blocked.
    * ``CATEGORY_MAP``: native label (lower-case) -> unified ``Category``; ``category_map`` replaces it
      entirely, since a model that reuses this output format usually brings its own taxonomy.
    * ``DEFAULT_IGNORE``: native labels ignored unless ``ignore_categories`` says otherwise.
    * ``SCORE_SPEC``: logprob anchor; override ``score_spec`` when it depends on options / input type.
    * ``REASONING_AS_REASON``: surface the ``<think>`` text as ``reason`` (models whose rationale lives there).
    """

    OPTIONS: ClassVar[frozenset[str]] = frozenset({"unparsable_as_unsafe"})
    CATEGORY_MAP: ClassVar[Mapping[str, str]] = {}
    DEFAULT_IGNORE: ClassVar[tuple[str, ...]] = ()
    SCORE_SPEC: ClassVar[ScoreSpec | None] = None
    REASONING_AS_REASON: ClassVar[bool] = False
    registry_name: ClassVar[str] = "base"  # set by @PARSERS.register

    def __init__(self, options: Mapping[str, Any] | None = None):
        self.options: dict[str, Any] = dict(options or {})
        unknown = sorted(set(self.options) - self.OPTIONS)
        if unknown:
            raise ValueError(f"parser {self.registry_name!r}: unknown parser_options {unknown}; "
                             f"accepted: {sorted(self.OPTIONS)}")
        self.unparsable_as_unsafe = bool(self.options.get("unparsable_as_unsafe", False))
        cmap = self.options.get("category_map") or self.CATEGORY_MAP
        self.category_map: dict[str, str] = {str(k).lower(): v for k, v in cmap.items()}
        ignore = self.options.get("ignore_categories")
        if isinstance(ignore, str):  # one category, not its characters (that would drop the default silently)
            ignore = [ignore]
        self.ignore_categories = {str(c).lower() for c in (self.DEFAULT_IGNORE if ignore is None else ignore)}

    @classmethod
    def from_config(cls, stage_cfg: Mapping[str, Any]) -> OutputParser:
        """Build from a guard stage: ``parser`` picked the class, ``parser_options`` configure it."""
        return cls(stage_cfg.get("parser_options") or {})

    # -- contract --------------------------------------------------------- #
    def parse(self, text: str, input_type: InputType = "request") -> ParsedOutput:
        answer, thinking = split_reasoning(text or "")
        out = self._parse(answer, input_type)
        if self.REASONING_AS_REASON and thinking and not out.reason:
            out.reason = thinking[:500]
        return out

    @abc.abstractmethod
    def _parse(self, answer: str, input_type: InputType) -> ParsedOutput:
        """Parse the model's answer (reasoning already removed) for the judged side ``input_type``."""

    def score_spec(self, input_type: InputType = "request") -> ScoreSpec | None:
        """``None`` = this output format has no stable label-token position -> hard labels only."""
        return self.SCORE_SPEC

    def score(self, content_logprobs: list[dict[str, Any]] | None, input_type: InputType = "request") -> float | None:
        spec = self.score_spec(input_type)
        return spec.score(content_logprobs) if spec else None

    # -- building blocks for subclasses ----------------------------------- #
    def unparsable(self, reason: str) -> ParsedOutput:
        """The model answered off-format: ``unparsable_as_unsafe`` decides (fail-open by default)."""
        return ParsedOutput(unsafe=self.unparsable_as_unsafe, parse_ok=False, reason=reason)

    def map_categories(self, native: Iterable[str]) -> list[str]:
        return sorted({self.category_map.get(str(c).lower(), "other") for c in native})

    def flagged(self, native: list[str], extra: dict[str, Any] | None = None, reason: str = "") -> ParsedOutput:
        """Unsafe verdict for ``native`` labels, honouring ``ignore_categories``.

        When every fired label is ignored (e.g. Qwen3Guard "Politically Sensitive Topics") the verdict
        is safe and ``label_overridden`` is set, so a high P(unsafe) score cannot re-block it.
        """
        extra = dict(extra or {})
        kept = [c for c in native if str(c).lower() not in self.ignore_categories]
        if native and not kept:
            return self._ignored(native, extra, reason)
        return ParsedOutput(unsafe=True, native_categories=native, categories=self.map_categories(kept) or ["other"],
                            extra=extra, reason=reason)

    def cleared(self, native: list[str] | None = None, extra: dict[str, Any] | None = None,
                reason: str = "") -> ParsedOutput:
        """Safe verdict; the counterpart of ``flagged`` for a label the model (or an option) counts as safe.

        A safe label can still name categories (Qwen3Guard "Controversial" in loose mode). When every one of
        them is ignored, ``label_overridden`` is set as in ``flagged``: otherwise the logprob score — which
        measures the label, not the categories — could still block content flagged only for ignored ones.
        """
        native = list(native or [])
        if native and all(str(c).lower() in self.ignore_categories for c in native):
            return self._ignored(native, dict(extra or {}), reason)
        return ParsedOutput(unsafe=False, native_categories=native, extra=dict(extra or {}), reason=reason)

    @staticmethod
    def _ignored(native: list[str], extra: dict[str, Any], reason: str) -> ParsedOutput:
        return ParsedOutput(unsafe=False, native_categories=native, label_overridden=True,
                            extra={**extra, "ignored": native}, reason=reason)


PARSERS: Registry[OutputParser] = Registry("guard output parser", base=OutputParser)


# --------------------------------------------------------------------------- #
# Llama Guard 3 / 4  — "safe"  |  "unsafe\nS1,S10"
# --------------------------------------------------------------------------- #
@PARSERS.register("llama_guard")
class LlamaGuardParser(OutputParser):
    """Llama Guard 3/4 (and S-code look-alikes such as OpenGuardrails): ``safe`` | ``unsafe\\nS1,S10``."""

    OPTIONS = OutputParser.OPTIONS | TAXONOMY_OPTIONS
    CATEGORY_MAP = LLAMA_GUARD_MAP
    SCORE_SPEC = ScoreSpec(anchor=r"^\s*$", unsafe=("unsafe",), safe=("safe",))

    def _parse(self, answer: str, input_type: InputType) -> ParsedOutput:
        lines = [ln.strip() for ln in answer.strip().splitlines() if ln.strip()]
        if not lines:
            return self.unparsable("empty output")
        head = lines[0].lower()
        if head.startswith("safe"):
            return self.cleared()
        if head.startswith("unsafe"):
            return self.flagged(re.findall(r"S\d{1,2}", " ".join(lines[1:])))
        return self.unparsable(f"unparsable: {head[:50]}")


# --------------------------------------------------------------------------- #
# Qwen3Guard-Gen — "Safety: Safe|Unsafe|Controversial\nCategories: ...\n[Refusal: Yes|No]"
# --------------------------------------------------------------------------- #
@PARSERS.register("qwen3guard")
class Qwen3GuardParser(OutputParser):
    """Qwen3Guard-Gen: ``Safety: Safe|Unsafe|Controversial`` + ``Categories:`` (+ ``Refusal:`` on responses)."""

    OPTIONS = OutputParser.OPTIONS | TAXONOMY_OPTIONS | {"controversial_as_unsafe"}
    CATEGORY_MAP = QWEN3GUARD_MAP
    # The model was trained with a PRC-centric notion of political sensitivity, which over-flags ordinary
    # zh-TW / Taiwan-related content. Revisit with your own policy owners.
    DEFAULT_IGNORE = ("Politically Sensitive Topics",)

    def __init__(self, options: Mapping[str, Any] | None = None):
        super().__init__(options)
        self.controversial_as_unsafe = bool(self.options.get("controversial_as_unsafe", True))

    def score_spec(self, input_type: InputType = "request") -> ScoreSpec:
        # "Controversial" mass goes to the side the hard label puts it on, so the score measures the same
        # decision (loose mode: Controversial is allowed, i.e. safe — dropping it would inflate P(unsafe))
        if self.controversial_as_unsafe:
            return ScoreSpec(anchor=r"Safety:\s*$", unsafe=("unsafe", "controversial"), safe=("safe",))
        return ScoreSpec(anchor=r"Safety:\s*$", unsafe=("unsafe",), safe=("safe", "controversial"))

    def _parse(self, answer: str, input_type: InputType) -> ParsedOutput:
        m = re.search(r"Safety:\s*(Safe|Unsafe|Controversial)", answer, re.IGNORECASE)
        if not m:
            return self.unparsable("no 'Safety:' field")
        level = m.group(1).lower()
        cats_m = re.search(r"Categories:\s*(.+)", answer)
        native = [c for c in _labels(cats_m.group(1), r"[,，]") if c.lower() != "none"] if cats_m else []
        refusal = re.search(r"Refusal:\s*(Yes|No)", answer, re.IGNORECASE)
        extra = {"level": level, **({"refusal": refusal.group(1).lower()} if refusal else {})}
        if level == "unsafe" or (level == "controversial" and self.controversial_as_unsafe):
            return self.flagged(native, extra)
        return self.cleared(native, extra)


# --------------------------------------------------------------------------- #
# Granite Guardian 3.x/4.x, Breeze-Guard — "[<think>..</think>] <score> yes </score>"  (yes = risk present)
# --------------------------------------------------------------------------- #
@PARSERS.register("granite_guardian")
class GraniteGuardianParser(OutputParser):
    """Granite Guardian / Breeze-Guard: ``<score> yes|no </score>`` for one criterion (yes = risk present)."""

    OPTIONS = OutputParser.OPTIONS | {"criteria_id", "criteria_category"}
    SCORE_SPEC = ScoreSpec(anchor=r"<score>\s*$", unsafe=("yes",), safe=("no",))
    REASONING_AS_REASON = True

    def __init__(self, options: Mapping[str, Any] | None = None):
        super().__init__(options)
        # one criterion per call: the stage config names it and its unified category
        self.criteria_id = str(self.options.get("criteria_id", "harm"))
        self.criteria_category = str(self.options.get("criteria_category", "other"))

    def _parse(self, answer: str, input_type: InputType) -> ParsedOutput:
        m = re.search(r"<score>\s*(yes|no)\s*</score>", answer, re.IGNORECASE)
        word = m.group(1).lower() if m else (answer.strip().split() or [""])[0].lower().strip(".")
        if word not in ("yes", "no"):
            return self.unparsable("no <score>")
        if word == "no":
            return self.cleared()
        return ParsedOutput(unsafe=True, native_categories=[self.criteria_id], categories=[self.criteria_category])


# --------------------------------------------------------------------------- #
# WildGuard — "Harmful request: yes\nResponse refusal: no\nHarmful response: no"
# --------------------------------------------------------------------------- #
@PARSERS.register("wildguard")
class WildGuardParser(OutputParser):
    """WildGuard: ``Harmful request`` / ``Response refusal`` / ``Harmful response`` yes|no|n/a fields."""

    FIELDS: ClassVar[dict[str, str]] = {"request": "Harmful request", "response": "Harmful response"}

    @staticmethod
    def _field(answer: str, name: str) -> str | None:
        m = re.search(rf"{name}:\s*(yes|no|n/a)", answer, re.IGNORECASE)
        return m.group(1).lower() if m else None

    def _parse(self, answer: str, input_type: InputType) -> ParsedOutput:
        val = self._field(answer, self.FIELDS[input_type])
        if val is None:
            return self.unparsable("field missing")
        extra = {"refusal": self._field(answer, "Response refusal")}
        return self.flagged([], extra) if val == "yes" else self.cleared(extra=extra)


# --------------------------------------------------------------------------- #
# NVIDIA Nemotron safety guards (Aegis 2.0 taxonomy)
# --------------------------------------------------------------------------- #
class _AegisParser(OutputParser):
    """Shared by the Nemotron guards: per-side ``User Safety`` / ``Response Safety`` + Aegis categories."""

    OPTIONS = OutputParser.OPTIONS | TAXONOMY_OPTIONS
    CATEGORY_MAP = AEGIS_MAP
    SAFETY_FIELD: ClassVar[dict[str, str]] = {"request": "User Safety", "response": "Response Safety"}

    def verdict(self, unsafe: bool, native: list[str]) -> ParsedOutput:
        return self.flagged(native) if unsafe else self.cleared(native)


@PARSERS.register("nemotron")
class NemotronJSONParser(_AegisParser):
    """Llama-3.1-Nemotron-Safety-Guard-8B-v3: JSON ``{"User Safety", "Response Safety", "Safety Categories"}``."""

    def _parse(self, answer: str, input_type: InputType) -> ParsedOutput:
        obj = _first_json(answer)
        if obj is None:
            return self.unparsable("no JSON")
        unsafe = str(obj.get(self.SAFETY_FIELD[input_type], "")).lower() == "unsafe"
        return self.verdict(unsafe, _labels(obj.get("Safety Categories")))


@PARSERS.register("nemotron35")
class Nemotron35Parser(_AegisParser):
    """Nemotron-3.5-Content-Safety: ``User Safety: ..\\nResponse Safety: ..\\nSafety Categories: ..`` lines."""

    def score_spec(self, input_type: InputType = "request") -> ScoreSpec:
        # the label token of the judged side follows its own field name
        return ScoreSpec(anchor=rf"{self.SAFETY_FIELD[input_type]}:\s*$", unsafe=("unsafe",), safe=("safe",))

    def _parse(self, answer: str, input_type: InputType) -> ParsedOutput:
        name = self.SAFETY_FIELD[input_type]
        m = re.search(rf"{name}:\s*(safe|unsafe)", answer, re.IGNORECASE)
        if not m:
            return self.unparsable(f"no '{name}:'")
        cm = re.search(r"Safety Categories:\s*(.+)", answer)
        return self.verdict(m.group(1).lower() == "unsafe", _labels(cm.group(1)) if cm else [])


# --------------------------------------------------------------------------- #
# AprielGuard — "unsafe-O14,O12\nnon_adversarial"  |  "safe\nadversarial"
# --------------------------------------------------------------------------- #
@PARSERS.register("aprielguard")
class AprielGuardParser(OutputParser):
    """AprielGuard: ``safe|unsafe[-O1,O2]`` + ``adversarial|non_adversarial`` (adversarial = jailbreak attempt)."""

    OPTIONS = OutputParser.OPTIONS | TAXONOMY_OPTIONS | {"adversarial_as_unsafe"}
    SCORE_SPEC = ScoreSpec(anchor=r"^\s*$", unsafe=("unsafe",), safe=("safe",))

    def __init__(self, options: Mapping[str, Any] | None = None):
        super().__init__(options)
        self.adversarial_as_unsafe = bool(self.options.get("adversarial_as_unsafe", True))

    def _parse(self, answer: str, input_type: InputType) -> ParsedOutput:
        m = re.search(r"(safe|unsafe)-?([\w,]*)\s*\n\s*(adversarial|non_adversarial)", answer, re.IGNORECASE)
        if not m:
            return self.unparsable("no verdict")
        codes = [c for c in m.group(2).split(",") if c]
        out = self.flagged(codes) if m.group(1).lower() == "unsafe" else self.cleared(codes)
        adversarial = m.group(3).lower() == "adversarial"
        # adversarial-ness is a property of the request; a response is not "a jailbreak". The logprob
        # score only measures the harmful label, so it must not overturn an adversarial flag.
        if adversarial and self.adversarial_as_unsafe and input_type == "request":
            out.label_overridden = True
            out.unsafe = True
            out.categories = sorted({*out.categories, "jailbreak"})
        out.extra["adversarial"] = adversarial
        return out


# --------------------------------------------------------------------------- #
# Policy-as-prompt JSON (gpt-oss-safeguard, generic LLM-as-judge baseline)
#   expected: {"violation": 0|1|true|false, "categories": [...], "rationale": "..."}
# --------------------------------------------------------------------------- #
@PARSERS.register("json_policy")
class JSONPolicyParser(OutputParser):
    """Policy-as-prompt judges: ``{"violation": 0|1, "categories": [..], "rationale": ".."}`` (or a bare 0/1)."""

    OPTIONS = OutputParser.OPTIONS | TAXONOMY_OPTIONS | {"allowed_categories"}
    SCORE_SPEC = ScoreSpec(anchor=r"\"violation\"\s*:\s*\"?\s*$", unsafe=("1", "true"), safe=("0", "false"))
    _UNSAFE_VALUES: ClassVar[tuple[Any, ...]] = (1, True, "1", "true", "True", "yes", "unsafe")
    # off-format answer: a bare leading 0/1, or a "violation: 1"-style field — not any digit in prose
    # ("falls under section 1.2 but is allowed" is not a violation)
    # "1", "1.", "1. The request...", "Violation: 1", "**Violation:** 1" -> verdict; "1.2" / "10" are not
    _FREE_TEXT = re.compile(r"^\W*([01])(?!\w)(?!\.\d)|\bviolation\b\W{0,8}([01])(?!\w)(?!\.\d)", re.IGNORECASE)

    def __init__(self, options: Mapping[str, Any] | None = None):
        super().__init__(options)
        self.allowed = set(self.options.get("allowed_categories", []))

    def map_categories(self, native: Iterable[str]) -> list[str]:
        # The policy prompt asks for unified category names directly, so keep them as-is; ``category_map``
        # renames, ``allowed_categories`` bounds what a free-form judge may invent (rest -> "other").
        out = set()
        for c in native:
            c = self.category_map.get(str(c).lower(), str(c))
            out.add(c if not self.allowed or c in self.allowed else "other")
        return sorted(out)

    def _parse(self, answer: str, input_type: InputType) -> ParsedOutput:
        obj = _first_json(answer)
        if obj is None:
            m = self._FREE_TEXT.search(answer)
            if not m:
                return self.unparsable("no JSON/0/1")
            return self.flagged([]) if (m.group(1) or m.group(2)) == "1" else self.cleared()
        # the verdict first: nothing about the categories' format may drop a detected violation
        unsafe = obj.get("violation", obj.get("unsafe", obj.get("flagged"))) in self._UNSAFE_VALUES
        native = _labels(obj.get("categories") or obj.get("policy_category") or obj.get("category"))
        reason = str(obj.get("rationale", ""))[:500]
        return self.flagged(native, reason=reason) if unsafe else self.cleared(native, reason=reason)


# --------------------------------------------------------------------------- #
# Generic yes/no (ShieldGemma-style prompts, custom judges)
# --------------------------------------------------------------------------- #
@PARSERS.register("yes_no")
class YesNoParser(OutputParser):
    """Generic yes/no judge (ShieldGemma-style prompts); ``yes_means_unsafe: false`` for "is it safe?" prompts."""

    OPTIONS = OutputParser.OPTIONS | {"criteria_category", "yes_means_unsafe"}

    def __init__(self, options: Mapping[str, Any] | None = None):
        super().__init__(options)
        self.yes_means_unsafe = bool(self.options.get("yes_means_unsafe", True))
        self.criteria_category = str(self.options.get("criteria_category", "other"))

    def score_spec(self, input_type: InputType = "request") -> ScoreSpec:
        unsafe, safe = (("yes",), ("no",)) if self.yes_means_unsafe else (("no",), ("yes",))
        return ScoreSpec(anchor=r"^\s*$", unsafe=unsafe, safe=safe)

    def _parse(self, answer: str, input_type: InputType) -> ParsedOutput:
        w = (re.findall(r"[A-Za-z]+", answer) or [""])[0].lower()
        if w not in ("yes", "no"):
            return self.unparsable("not yes/no")
        unsafe = (w == "yes") == self.yes_means_unsafe
        return ParsedOutput(unsafe=True, categories=[self.criteria_category]) if unsafe else self.cleared()
