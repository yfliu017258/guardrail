"""Output parsers for generative guard models.

Each parser turns raw generated text into ``ParsedOutput`` and declares how to
read a probability from token logprobs (``score_spec``), so the evaluator can
sweep thresholds / draw PR curves instead of relying on a single hard label.

Add a new model family = add a function + a ``ScoreSpec`` and register it in ``PARSERS``.
Native-category -> unified-category maps live next to each parser.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from ...data_schema.taxonomy import AEGIS_MAP, LLAMA_GUARD_MAP, QWEN3GUARD_MAP


@dataclass
class ParsedOutput:
    unsafe: bool
    native_categories: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    reason: str = ""
    parse_ok: bool = True
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScoreSpec:
    """Where to read P(unsafe) from logprobs.

    ``anchor``: regex that the *generated text so far* must match (at its end) for the
    next token to be the label token. ``unsafe``/``safe``: words whose leading token
    counts toward each side (prefix match, case-insensitive).
    """

    anchor: str
    unsafe: tuple[str, ...]
    safe: tuple[str, ...]
    anchor_response: str | None = None   # different anchor when judging a response (e.g. "Response Safety:")


# --------------------------------------------------------------------------- #
# Llama Guard 3 / 4  — "safe"  |  "unsafe\nS1,S10"
# --------------------------------------------------------------------------- #
def parse_llama_guard(text: str, opts: dict) -> ParsedOutput:
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if not lines:
        return ParsedOutput(unsafe=False, parse_ok=False, reason="empty output")
    head = lines[0].lower()
    if head.startswith("safe"):
        return ParsedOutput(unsafe=False)
    if head.startswith("unsafe"):
        codes = re.findall(r"S\d{1,2}", " ".join(lines[1:]))
        cmap = {k.lower(): v for k, v in (opts.get("category_map") or LLAMA_GUARD_MAP).items()}
        ignore = {c.lower() for c in opts.get("ignore_categories", [])}
        kept = [c for c in codes if c.lower() not in ignore]
        if codes and not kept:  # only ignored codes fired (e.g. OpenGuardrails S2/S3 political)
            return ParsedOutput(unsafe=False, native_categories=codes, extra={"ignored": codes})
        return ParsedOutput(
            unsafe=True,
            native_categories=codes,
            categories=sorted({cmap.get(c.lower(), "other") for c in kept}) or ["other"],
        )
    return ParsedOutput(unsafe=opts.get("unparsable_as_unsafe", False), parse_ok=False, reason=f"unparsable: {head[:50]}")


# --------------------------------------------------------------------------- #
# Qwen3Guard-Gen — "Safety: Safe|Unsafe|Controversial\nCategories: ...\n[Refusal: Yes|No]"
# --------------------------------------------------------------------------- #
def parse_qwen3guard(text: str, opts: dict) -> ParsedOutput:
    m = re.search(r"Safety:\s*(Safe|Unsafe|Controversial)", text, re.IGNORECASE)
    if not m:
        return ParsedOutput(unsafe=opts.get("unparsable_as_unsafe", False), parse_ok=False, reason="no 'Safety:' field")
    level = m.group(1).lower()
    cats_m = re.search(r"Categories:\s*(.+)", text)
    native = [c.strip() for c in re.split(r"[,，]", cats_m.group(1))] if cats_m else []
    native = [c for c in native if c and c.lower() != "none"]
    # Ignore categories the policy does not care about. Default ignores "Politically Sensitive Topics":
    # the model was trained with a PRC-centric notion of political sensitivity, which over-flags
    # ordinary zh-TW / Taiwan-related content. Revisit with your own policy owners.
    ignore = {c.lower() for c in opts.get("ignore_categories", ["Politically Sensitive Topics"])}
    kept = [c for c in native if c.lower() not in ignore]
    controversial_unsafe = bool(opts.get("controversial_as_unsafe", True))
    unsafe = level == "unsafe" or (level == "controversial" and controversial_unsafe)
    if unsafe and native and not kept:  # only ignored categories fired
        unsafe = False
    refusal = re.search(r"Refusal:\s*(Yes|No)", text, re.IGNORECASE)
    return ParsedOutput(
        unsafe=unsafe,
        native_categories=native,
        categories=sorted({QWEN3GUARD_MAP.get(c.lower(), "other") for c in kept}) if unsafe else [],
        extra={"level": level, **({"refusal": refusal.group(1).lower()} if refusal else {})},
    )


# --------------------------------------------------------------------------- #
# Granite Guardian 3.x — "... <score> yes </score>"  (yes = risk present)
# --------------------------------------------------------------------------- #
def parse_granite_guardian(text: str, opts: dict) -> ParsedOutput:
    m = re.search(r"<score>\s*(yes|no)\s*</score>", text, re.IGNORECASE)
    word = m.group(1).lower() if m else (text.strip().split() or [""])[0].lower().strip(".")
    if word not in ("yes", "no"):
        return ParsedOutput(unsafe=opts.get("unparsable_as_unsafe", False), parse_ok=False, reason="no <score>")
    risk = opts.get("criteria_category", "other")
    think = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    return ParsedOutput(
        unsafe=word == "yes",
        native_categories=[opts.get("criteria_id", "harm")] if word == "yes" else [],
        categories=[risk] if word == "yes" else [],
        reason=(think.group(1).strip()[:500] if think else ""),
    )


# --------------------------------------------------------------------------- #
# WildGuard — "Harmful request: yes\nResponse refusal: no\nHarmful response: no"
# --------------------------------------------------------------------------- #
def parse_wildguard(text: str, opts: dict) -> ParsedOutput:
    def field_(name: str) -> str | None:
        m = re.search(rf"{name}:\s*(yes|no|n/a)", text, re.IGNORECASE)
        return m.group(1).lower() if m else None

    side = opts.get("input_type", "request")
    val = field_("Harmful request") if side == "request" else field_("Harmful response")
    if val is None:
        return ParsedOutput(unsafe=opts.get("unparsable_as_unsafe", False), parse_ok=False, reason="field missing")
    return ParsedOutput(unsafe=val == "yes", categories=["other"] if val == "yes" else [],
                        extra={"refusal": field_("Response refusal")})


# --------------------------------------------------------------------------- #
# Nemotron Safety Guard — JSON {"User Safety": "unsafe", "Response Safety": "safe", "Safety Categories": "..."}
# --------------------------------------------------------------------------- #
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


def parse_nemotron(text: str, opts: dict) -> ParsedOutput:
    obj = _first_json(text)
    if obj is None:
        return ParsedOutput(unsafe=opts.get("unparsable_as_unsafe", False), parse_ok=False, reason="no JSON")
    key = "User Safety" if opts.get("input_type", "request") == "request" else "Response Safety"
    val = str(obj.get(key, "")).lower()
    native = [c.strip() for c in str(obj.get("Safety Categories", "")).split(",") if c.strip()]
    unsafe = val == "unsafe"
    return ParsedOutput(unsafe=unsafe, native_categories=native,
                        categories=sorted({AEGIS_MAP.get(c.lower(), "other") for c in native}) if unsafe else [])


# --------------------------------------------------------------------------- #
# Nemotron-3.5-Content-Safety — "User Safety: unsafe\nResponse Safety: safe\nSafety Categories: Illegal Activity"
# (optionally preceded by <think>...</think>)
# --------------------------------------------------------------------------- #
def parse_nemotron35(text: str, opts: dict) -> ParsedOutput:
    body = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    field = "User Safety" if opts.get("input_type", "request") == "request" else "Response Safety"
    m = re.search(rf"{field}:\s*(safe|unsafe)", body, re.IGNORECASE)
    if not m:
        return ParsedOutput(unsafe=opts.get("unparsable_as_unsafe", False), parse_ok=False, reason=f"no '{field}:'")
    unsafe = m.group(1).lower() == "unsafe"
    cm = re.search(r"Safety Categories:\s*(.+)", body)
    native = [c.strip() for c in cm.group(1).split(",") if c.strip()] if cm else []
    return ParsedOutput(unsafe=unsafe, native_categories=native,
                        categories=sorted({AEGIS_MAP.get(c.lower(), "other") for c in native}) or ["other"] if unsafe else [])


# --------------------------------------------------------------------------- #
# AprielGuard — "unsafe-O14,O12\nnon_adversarial"  |  "safe\nadversarial"
# --------------------------------------------------------------------------- #
def parse_aprielguard(text: str, opts: dict) -> ParsedOutput:
    m = re.search(r"(safe|unsafe)-?([\w,]*)\s*\n\s*(adversarial|non_adversarial)", text, re.IGNORECASE)
    if not m:
        return ParsedOutput(unsafe=opts.get("unparsable_as_unsafe", False), parse_ok=False, reason="no verdict")
    harmful = m.group(1).lower() == "unsafe"
    adversarial = m.group(3).lower() == "adversarial"
    codes = [c for c in m.group(2).split(",") if c]
    count_adv = opts.get("adversarial_as_unsafe", True) and opts.get("input_type", "request") == "request"
    unsafe = harmful or (adversarial and count_adv)
    cats = (["other"] if harmful else []) + (["jailbreak"] if adversarial and count_adv else [])
    return ParsedOutput(unsafe=unsafe, native_categories=codes, categories=sorted(set(cats)),
                        extra={"adversarial": adversarial})


# --------------------------------------------------------------------------- #
# Policy-as-prompt JSON (gpt-oss-safeguard, generic LLM-as-judge baseline)
#   expected: {"violation": 0|1|true|false, "categories": [...], "rationale": "..."}
# --------------------------------------------------------------------------- #
def parse_json_policy(text: str, opts: dict) -> ParsedOutput:
    obj = _first_json(text)
    if obj is None:
        m = re.search(r"\b([01])\b", text)
        if m:
            return ParsedOutput(unsafe=m.group(1) == "1", categories=["other"] if m.group(1) == "1" else [])
        return ParsedOutput(unsafe=opts.get("unparsable_as_unsafe", False), parse_ok=False, reason="no JSON/0/1")
    v = obj.get("violation", obj.get("unsafe", obj.get("flagged")))
    unsafe = v in (1, True, "1", "true", "True", "yes", "unsafe")
    cats = obj.get("categories") or obj.get("policy_category") or obj.get("category") or []
    if isinstance(cats, str):
        cats = [c.strip() for c in cats.split(",") if c.strip()]
    allowed = set(opts.get("allowed_categories", []))
    unified = sorted({c if (not allowed or c in allowed) else "other" for c in cats}) if unsafe else []
    return ParsedOutput(unsafe=unsafe, native_categories=list(cats), categories=unified or (["other"] if unsafe else []),
                        reason=str(obj.get("rationale", ""))[:500])


# --------------------------------------------------------------------------- #
# Generic yes/no (ShieldGemma-style prompts, custom judges)
# --------------------------------------------------------------------------- #
def parse_yes_no(text: str, opts: dict) -> ParsedOutput:
    w = (re.findall(r"[A-Za-z]+", text) or [""])[0].lower()
    if w not in ("yes", "no"):
        return ParsedOutput(unsafe=opts.get("unparsable_as_unsafe", False), parse_ok=False, reason="not yes/no")
    yes_is_unsafe = opts.get("yes_means_unsafe", True)
    unsafe = (w == "yes") == yes_is_unsafe
    return ParsedOutput(unsafe=unsafe, categories=[opts.get("criteria_category", "other")] if unsafe else [])


PARSERS: dict[str, tuple[Callable[[str, dict], ParsedOutput], ScoreSpec | None]] = {
    "llama_guard": (parse_llama_guard, ScoreSpec(anchor=r"^\s*$", unsafe=("unsafe",), safe=("safe",))),
    "qwen3guard": (parse_qwen3guard, ScoreSpec(anchor=r"Safety:\s*$", unsafe=("unsafe", "controversial"), safe=("safe",))),
    "granite_guardian": (parse_granite_guardian, ScoreSpec(anchor=r"<score>\s*$", unsafe=("yes",), safe=("no",))),
    "wildguard": (parse_wildguard, None),
    "nemotron": (parse_nemotron, None),
    "nemotron35": (parse_nemotron35, ScoreSpec(anchor=r"User Safety:\s*$", unsafe=("unsafe",), safe=("safe",),
                                               anchor_response=r"Response Safety:\s*$")),
    "aprielguard": (parse_aprielguard, ScoreSpec(anchor=r"^\s*$", unsafe=("unsafe",), safe=("safe",))),
    "json_policy": (parse_json_policy, ScoreSpec(anchor=r"\"violation\"\s*:\s*$", unsafe=("1", "true"), safe=("0", "false"))),
    "yes_no": (parse_yes_no, ScoreSpec(anchor=r"^\s*$", unsafe=("yes",), safe=("no",))),
}


def get_parser(name: str) -> tuple[Callable[[str, dict], ParsedOutput], ScoreSpec | None]:
    if name not in PARSERS:
        raise ValueError(f"unknown parser {name!r}; available: {sorted(PARSERS)}")
    return PARSERS[name]


# --------------------------------------------------------------------------- #
# logprob scoring
# --------------------------------------------------------------------------- #
def _matches(token: str, words: tuple[str, ...]) -> bool:
    t = token.strip().lower()
    return bool(t) and any(w.startswith(t) for w in words)


def score_from_logprobs(content_logprobs: list[dict] | None, spec: ScoreSpec | None,
                        unsafe_words: tuple[str, ...] | None = None, input_type: str = "request") -> float | None:
    """P(unsafe) from an OpenAI-style ``choices[0].logprobs.content`` list.

    Walks generated tokens; at the first position where the text generated so far matches
    ``spec.anchor`` and the top-k alternatives contain a label token, returns
    ``sum p(unsafe tokens) / (sum p(unsafe) + sum p(safe))``. ``None`` if not found
    (e.g. logprobs disabled) — the evaluator then falls back to the hard label.
    """
    if not content_logprobs or spec is None:
        return None
    unsafe_words = unsafe_words or spec.unsafe
    pattern = spec.anchor_response if (input_type == "response" and spec.anchor_response) else spec.anchor
    anchor = re.compile(pattern + r"\Z", re.DOTALL)
    so_far = ""
    for pos in content_logprobs:
        if anchor.search(so_far):
            alts = pos.get("top_logprobs") or [{"token": pos.get("token", ""), "logprob": pos.get("logprob", 0.0)}]
            p_u = sum(math.exp(a["logprob"]) for a in alts if _matches(a["token"], unsafe_words))
            p_s = sum(math.exp(a["logprob"]) for a in alts if _matches(a["token"], spec.safe)
                      and not _matches(a["token"], unsafe_words))
            if p_u + p_s > 0:
                return p_u / (p_u + p_s)
        so_far += pos.get("token", "")
    return None
