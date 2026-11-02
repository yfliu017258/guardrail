"""Mapping templates: HOW one raw record becomes ``Sample(prompt, expected)`` rows.

Each template implements ``BaseConverter.convert`` for one record shape. A dataset is described
by class attributes; a hook is overridden only for what the attributes cannot express::

    TextLabelConverter        text column + label column -> one sample per row:
                              ``positive_values`` -> ``positive_expected`` (block),
                              ``negative_values`` -> ``negative_expected`` (pass)
      UnsafeOnlyConverter     every row ``block``   (recall-only sets: HarmBench, MultiJail)
      SafeOnlyConverter       every row ``pass``    (false-positive / over-defense sets: NotInject)
    PromptLabelConverter      prompt column + the PROMPT's own label column, for sets that label
                              prompts and responses separately (Aegis, WildGuardMix): ``label_values``
                              maps raw prompt labels -> block / pass; responses are never converted

A label must never silently become ``pass``: a value outside the declared sets is a malformed row
(counted ``invalid``), not a negative. ``TextLabelConverter`` also treats a missing label as
malformed; ``PromptLabelConverter`` skips an unlabelled prompt (those sets leave some sides
unannotated on purpose). Labels compare through ``label_key``, so ``1`` / ``1.0`` / ``"1"`` /
``True`` / ``" INJECTION "`` match the declared ``"1"`` / ``"injection"``.

The source split being converted is ``self.current_split`` — used by the few sets whose split
is the label.
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Iterable, Mapping
from typing import Any, ClassVar

from ..data_schema import Expected, Sample
from .base import BaseConverter, Record


def label_key(value: Any) -> str:
    """Normalise a raw label so ``1`` / ``1.0`` / ``True`` / ``"1"`` / ``"1.0"`` / ``"true"`` (and case /
    padding, numpy scalars) compare equal — CSV and pandas exports turn an int column into ``1.0``.

    A missing label (None / NaN / pd.NA / blank) raises: an unlabelled row must be counted invalid,
    not silently become ``pass``."""
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
    """One text column + one label column -> one sample per row.

    A label in ``positive_values`` gives ``positive_expected`` (``block``), one in ``negative_values``
    gives ``negative_expected`` (``pass``); any other value — or a missing one — raises, so the row is
    counted ``invalid``. A set whose label is "violation vs anything else" opts out with
    ``negative_values = None``.

    By default the text is used verbatim, so a missing / blank text fails validation and is counted
    ``invalid``; ``skip_blank_text`` is for sources where blank rows are expected (chat logs, parallel
    language columns) — texts are stripped and blank ones counted ``skipped`` instead.
    """

    text_field: ClassVar[str] = "text"
    label_field: ClassVar[str] = "label"
    positive_values: ClassVar[frozenset[str]] = frozenset({"1"})          # labels meaning harmful / attack / PII
    negative_values: ClassVar[frozenset[str] | None] = frozenset({"0"})   # labels meaning benign; None = all others
    positive_expected: ClassVar[Expected] = "block"
    negative_expected: ClassVar[Expected] = "pass"
    skip_blank_text: ClassVar[bool] = False

    def convert(self, record: Record, split: str, idx: int) -> list[Sample]:
        out = []
        for r in self.expand(record):
            if not self.keep(r):
                continue
            text = self.text(r)
            if self.skip_blank_text and not text:
                continue
            out.append(self.sample(text, self.expected(r)))
        return out

    # -- hooks ------------------------------------------------------------ #
    def expand(self, record: Record) -> Iterable[Record]:
        """Logical rows of one raw record; parallel-column sets (one column per language) yield several."""
        return (record,)

    def keep(self, r: Record) -> bool:
        """``False`` drops the row on purpose (counted ``skipped``), e.g. out-of-policy subjects."""
        return True

    def text(self, r: Record) -> Any:
        value = r.get(self.text_field)
        return str(value or "").strip() if self.skip_blank_text else value

    def is_positive(self, r: Record) -> bool:
        """Does the row have what the set labels (harmful / injection / PII)? Unknown labels raise."""
        value = r.get(self.label_field)
        key = label_key(value)
        if key in label_set(self.positive_values):
            return True
        negatives = label_set(self.negative_values)
        if negatives is None or key in negatives:
            return False
        raise ValueError(f"unknown {self.label_field} {value!r} ({self.positive_expected}: "
                         f"{sorted(label_set(self.positive_values))}, {self.negative_expected}: {sorted(negatives)})")

    def expected(self, r: Record) -> Expected:
        return self.positive_expected if self.is_positive(r) else self.negative_expected


class UnsafeOnlyConverter(TextLabelConverter):
    """Every row is ``block``: recall-only sets. Pair with a ``pass`` set before reading any FPR."""

    def is_positive(self, r: Record) -> bool:
        return True


class SafeOnlyConverter(TextLabelConverter):
    """Every row is ``pass``: false-positive / over-defense sets. Measures FPR only."""

    def is_positive(self, r: Record) -> bool:
        return False


class PromptLabelConverter(BaseConverter):
    """One record -> one sample from its prompt and the PROMPT's label (``prompt_label_field``).

    For sets that label prompts and responses separately (Aegis 2.0, WildGuardMix, RTP-LX): only the
    request side is converted. ``label_values`` maps raw prompt labels to ``block`` / ``pass``
    (compared through ``label_key``); an unmapped value raises (``invalid``), a missing label or prompt
    means "no sample" (``skipped``). The prompt is stripped (``strip_text``). Override ``expected`` when
    the label is derived (scores, flags) — return ``None`` for a prompt without ground truth.
    """

    prompt_field: ClassVar[str] = "prompt"
    prompt_label_field: ClassVar[str] = "prompt_label"
    label_values: ClassVar[Mapping[str, Expected]] = {"unsafe": "block", "safe": "pass"}
    strip_text: ClassVar[bool] = True

    def convert(self, record: Record, split: str, idx: int) -> Sample | None:
        if not self.keep(record):
            return None
        prompt = self.text(record)
        if not prompt:
            return None
        expected = self.expected(record)
        return None if expected is None else self.sample(prompt, expected)

    # -- hooks ------------------------------------------------------------ #
    def keep(self, r: Record) -> bool:
        """``False`` drops the record on purpose (counted ``skipped``), e.g. redacted or mistranslated prompts."""
        return True

    def text(self, r: Record) -> str:
        value = r.get(self.prompt_field) or ""
        return value.strip() if self.strip_text else value

    def expected(self, r: Record) -> Expected | None:
        value = r.get(self.prompt_label_field)
        if is_missing(value):
            return None  # this prompt was not annotated
        key = label_key(value)
        mapping = {label_key(k): v for k, v in self.label_values.items()}
        if key not in mapping:
            raise ValueError(f"unknown {self.prompt_label_field} {value!r} (known: {sorted(mapping)})")
        return mapping[key]
