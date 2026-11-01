"""Mapping templates: HOW one raw record becomes ``GuardrailSample``(s).

Each template implements ``BaseConverter.convert`` for one record shape. A dataset is described
by class attributes; a hook is overridden only for what the attributes cannot express::

    TextLabelConverter        one text + one label column -> one request sample
      UnsafeOnlyConverter     every row unsafe   (recall-only sets: HarmBench, MultiJail)
      SafeOnlyConverter       every row safe     (FPR / over-defense sets: NotInject)
    PromptResponseConverter   prompt (+ response) per record -> a request and/or a response sample,
                              response rows optionally carrying ``refusal_label``

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
    """Normalise a raw label so ``1`` / ``1.0`` / ``True`` / ``"1"`` (and case / padding) compare equal.

    A missing label raises: an unlabelled row must be counted invalid, not silently become safe."""
    if value is None or (isinstance(value, float) and math.isnan(value)):  # None / NaN from pandas
        raise ValueError("missing label")
    if isinstance(value, numbers.Real) and float(value).is_integer():  # bool and numpy ints included
        return str(int(value))
    return str(value).strip().lower()


class TextLabelConverter(BaseConverter):
    """One text column + one label column -> one ``request`` sample per row.

    Unsafe rows get ``[category]`` and ``attack_type``; safe rows get neither. By default the
    text is used verbatim, so a missing / blank text fails validation and is counted ``invalid``;
    ``skip_blank_text`` is for sources where blank rows are expected (chat logs) — they are
    stripped and counted ``skipped`` instead.
    """

    text_field: ClassVar[str] = "text"
    label_field: ClassVar[str] = "label"
    positive_values: ClassVar[frozenset[str]] = frozenset({"1"})  # ``label_key`` values meaning unsafe
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
        return label_key(r.get(self.label_field)) in self.positive_values

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
        out = []
        if self.wants("request") and (label := self.prompt_label(record)) is not None:
            out.append(self._side_sample(record, "request", label, prompt=prompt))
        response = self.text(record, self.response_field)
        if response and self.wants("response") and (label := self.response_label(record)) is not None:
            out.append(self._side_sample(record, "response", label, prompt=prompt, response=response,
                                         refusal_label=self.refusal_label(record)))
        return out

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
