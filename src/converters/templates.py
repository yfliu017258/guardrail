"""Mapping templates: HOW one raw record becomes ``GuardrailSample``(s).

Each template implements ``BaseConverter.convert`` for one record shape. A dataset is described
by class attributes; a hook is overridden only for what the attributes cannot express::

    TextLabelConverter        one text + one label column -> one request sample
      UnsafeOnlyConverter     every row unsafe   (recall-only sets: HarmBench, MultiJail)
      SafeOnlyConverter       every row safe     (FPR / over-defense sets: NotInject)
    PromptResponseConverter   prompt (+ response) per record -> a request and/or a response sample,
                              response rows optionally carrying ``refusal_label``

A label must never silently become ``safe``: a missing label or a value outside the declared
sets is a malformed row (counted ``invalid``), not a negative.

Hooks receive the raw record (plus the judged ``side`` for prompt/response sets). The source
split being converted is ``self.current_split`` — used by the few sets whose split is the label.
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Iterable, Mapping
from typing import Any, ClassVar

from ..data_schema import GuardrailSample, InputType, Label, RefusalLabel
from .base import BaseConverter, Record


def label_key(value: Any) -> str:
    """Normalise a raw label so ``1`` / ``1.0`` / ``True`` / ``"1"`` / ``"1.0"`` / ``"true"`` (and case /
    padding, numpy scalars) compare equal — CSV and pandas exports turn an int column into ``1.0``.

    A missing label (None / NaN / pd.NA / blank) raises: an unlabelled row must be counted invalid,
    not silently become safe."""
    if hasattr(value, "item") and type(value).__module__ == "numpy":  # numpy scalar (np.bool_ is not numbers.Real)
        value = value.item()
    if is_missing(value):
        raise ValueError("missing label")
    if isinstance(value, numbers.Real) and float(value).is_integer():  # bool and ints included
        return str(int(value))
    key = str(value).strip().lower()
    if key in ("true", "false"):
        return "1" if key == "true" else "0"
    try:
        number = float(key)
    except ValueError:
        return key
    if math.isnan(number):  # "nan" / "NaN" written by an export
        raise ValueError("missing label")
    return str(int(number)) if number.is_integer() else key


def is_missing(value: Any) -> bool:
    """None, float NaN, pandas NA / NaT, or a blank string."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return True
    if type(value).__name__ in ("NAType", "NaTType"):
        return True
    return isinstance(value, str) and not value.strip()


def label_set(values: Iterable[Any] | None) -> frozenset[str] | None:
    return None if values is None else frozenset(label_key(v) for v in values)


class TextLabelConverter(BaseConverter):
    """One text column + one label column -> one ``request`` sample per row.

    The label is unsafe when it is in ``positive_values`` and safe when it is in ``negative_values``
    (both compared through ``label_key``, so ``{"INJECTION"}`` matches ``"injection"`` and ``{"1"}``
    matches ``1.0``); any other value raises -> the row is counted ``invalid``. A set whose label is
    "violation vs anything else" opts out with ``negative_values = None``.

    Unsafe rows get ``[category]`` and ``attack_type``; safe rows get neither. By default the
    text is used verbatim, so a missing / blank text fails validation and is counted ``invalid``;
    ``skip_blank_text`` is for sources where blank rows are expected (chat logs) — they are
    stripped and counted ``skipped`` instead.
    """

    text_field: ClassVar[str] = "text"
    label_field: ClassVar[str] = "label"
    positive_values: ClassVar[frozenset[str]] = frozenset({"1"})          # label values meaning unsafe
    negative_values: ClassVar[frozenset[str] | None] = frozenset({"0"})   # label values meaning safe; None = all others
    category: ClassVar[str] = "other"           # category of unsafe rows
    attack_type: ClassVar[str | None] = None    # attack_type of unsafe rows
    language: ClassVar[str] = "auto"            # "auto" = detect from the text
    skip_blank_text: ClassVar[bool] = False

    def convert(self, record: Record, split: str, idx: int) -> list[GuardrailSample]:
        out = []
        for r in self.expand(record):
            if not self.keep(r):
                continue
            text = self.text(r)
            if self.skip_blank_text and not text:
                continue
            unsafe = self.is_unsafe(r)
            out.append(self.sample(prompt=text, label=Label.UNSAFE if unsafe else Label.SAFE,
                                   categories=self.categories(r) if unsafe else [], attack_type=self.attack(r, unsafe),
                                   language=self.lang(r), meta=self.meta(r)))
        return out

    # -- hooks ------------------------------------------------------------ #
    def expand(self, record: Record) -> Iterable[Record]:
        """Logical rows of one raw record; parallel-column sets (one column per language) yield several."""
        return (record,)

    def keep(self, r: Record) -> bool:
        """``False`` drops the row on purpose (counted ``skipped``), e.g. out-of-policy subjects."""
        return True

    def text(self, r: Record) -> str:
        value = r.get(self.text_field)
        return str(value or "").strip() if self.skip_blank_text else value

    def is_unsafe(self, r: Record) -> bool:
        value = r.get(self.label_field)
        key = label_key(value)
        if key in label_set(self.positive_values):
            return True
        negatives = label_set(self.negative_values)
        if negatives is None or key in negatives:
            return False
        raise ValueError(f"unknown {self.label_field} {value!r} (unsafe: {sorted(label_set(self.positive_values))}, "
                         f"safe: {sorted(negatives)})")

    def categories(self, r: Record) -> list[str]:
        """Categories of an unsafe row (never called for safe rows)."""
        return [self.category]

    def attack(self, r: Record, unsafe: bool) -> str | None:
        return self.attack_type if unsafe else None

    def lang(self, r: Record) -> str:
        return self.language

    def meta(self, r: Record) -> dict[str, Any]:
        return {}


class UnsafeOnlyConverter(TextLabelConverter):
    """Every row is unsafe: recall-only sets. Pair with a safe set before reading any FPR."""

    def is_unsafe(self, r: Record) -> bool:
        return True


class SafeOnlyConverter(TextLabelConverter):
    """Every row is safe: false-positive / over-defense sets. Measures FPR only."""

    def is_unsafe(self, r: Record) -> bool:
        return False


class PromptResponseConverter(BaseConverter):
    """One record -> a ``request`` sample (prompt judged) and/or a ``response`` sample (response judged).

    Labels come from ``prompt_label_field`` / ``response_label_field`` through ``label_values``
    (a missing or unmapped label means "no sample for that side", not an error — most sets only
    label some sides). Response samples carry ``refusal_label`` from ``refusal_field`` through
    ``refusal_values``; an unknown non-empty refusal value raises, because a silently missing
    ground truth would bias a refusal-judge eval. Every sample needs the prompt: a response
    without its prompt has no context to judge. Options ``include_prompts`` / ``include_responses``
    override the class defaults of the same name.

    The two sides are judged independently: ``keep_side`` drops one side on purpose (counted
    ``skipped_<side>``), and a side that fails (bad refusal value, invalid sample) is counted
    ``invalid_<side>`` while the other side is still emitted. Only when NO sample survives does the
    error propagate, so the record is counted ``invalid`` as a whole.
    """

    prompt_field: ClassVar[str] = "prompt"
    response_field: ClassVar[str] = "response"
    prompt_label_field: ClassVar[str | None] = None
    response_label_field: ClassVar[str | None] = None
    label_values: ClassVar[Mapping[Any, str]] = {"safe": "safe", "unsafe": "unsafe"}
    refusal_field: ClassVar[str | None] = None
    refusal_values: ClassVar[Mapping[Any, RefusalLabel]] = {
        "refusal": "refusal", "partial": "partial", "compliance": "compliance"}
    default_category: ClassVar[str] = "other"   # unsafe sample whose ``categories`` hook found nothing
    language: ClassVar[str] = "auto"
    strip_text: ClassVar[bool] = True
    include_prompts: ClassVar[bool] = True
    include_responses: ClassVar[bool] = True

    def convert(self, record: Record, split: str, idx: int) -> list[GuardrailSample] | None:
        if not self.keep(record):
            return None
        prompt = self.text(record, self.prompt_field)
        if not prompt:
            return None
        out, failed = [], []
        response = self.text(record, self.response_field)
        for side in ("request", "response"):
            if not self.wants(side) or (side == "response" and not response):
                continue
            if not self.keep_side(record, side):
                self.stats[f"skipped_{side}"] += 1
                continue
            try:
                sample = self._side(record, side, prompt, response)
            except Exception as e:  # noqa: BLE001 - one bad side must not cost the other side's sample
                failed.append((side, e))
                continue
            if sample is not None:
                out.append(sample)
        if failed and not out:
            raise failed[0][1]  # nothing survives -> the record is invalid
        for side, _ in failed:
            self.stats[f"invalid_{side}"] += 1
        return out

    def _side(self, r: Record, side: InputType, prompt: str, response: str) -> GuardrailSample | None:
        if side == "request":
            label = self.prompt_label(r)
            return None if label is None else self._side_sample(r, side, label, prompt=prompt)
        label = self.response_label(r)
        if label is None:
            return None
        return self._side_sample(r, side, label, prompt=prompt, response=response, refusal_label=self.refusal_label(r))

    def _side_sample(self, r: Record, side: InputType, label: str, **fields: Any) -> GuardrailSample:
        unsafe = label == Label.UNSAFE.value
        return self.sample(input_type=side, label=label,
                           categories=(self.categories(r, side) or [self.default_category]) if unsafe else [],
                           attack_type=self.attack(r, side), language=self.lang(r, side), meta=self.meta(r, side),
                           **fields)

    # -- hooks ------------------------------------------------------------ #
    def keep(self, r: Record) -> bool:
        """``False`` drops the whole record on purpose (counted ``skipped``)."""
        return True

    def keep_side(self, r: Record, side: InputType) -> bool:
        """``False`` drops one side's sample on purpose (counted ``skipped_<side>``), e.g. an artefact response."""
        return True

    def wants(self, side: InputType) -> bool:
        key = "include_prompts" if side == "request" else "include_responses"
        return bool(self.options.get(key, getattr(self, key)))

    def text(self, r: Record, field: str) -> str:
        value = r.get(field) or ""
        return value.strip() if self.strip_text else value

    def prompt_label(self, r: Record) -> str | None:
        return None if self.prompt_label_field is None else self.label_values.get(r.get(self.prompt_label_field))

    def response_label(self, r: Record) -> str | None:
        return None if self.response_label_field is None else self.label_values.get(r.get(self.response_label_field))

    def refusal_label(self, r: Record) -> RefusalLabel | None:
        if self.refusal_field is None or r.get(self.refusal_field) in (None, ""):
            return None
        value = r[self.refusal_field]
        if value not in self.refusal_values:
            raise ValueError(f"unknown {self.refusal_field} {value!r}")
        return self.refusal_values[value]

    def categories(self, r: Record, side: InputType) -> list[str]:
        """Categories of an unsafe sample (never called for safe ones); empty -> ``[default_category]``."""
        return []

    def attack(self, r: Record, side: InputType) -> str | None:
        return None

    def lang(self, r: Record, side: InputType) -> str:
        return self.language

    def meta(self, r: Record, side: InputType) -> dict[str, Any]:
        return {}
