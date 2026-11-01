"""Generic converter for local files (internal red-team sets, labelled production logs, SaaS exports).

Options (YAML or ``--set``)::

    path: data/raw/internal_redteam.jsonl      # .csv / .jsonl / .json / .parquet
    input_type: request                       # or `column:<name>` to read it per row
    columns: {prompt: question, response: answer, label: is_attack, category: type, language: lang}
    label_map: {"1": unsafe, "0": safe, attack: unsafe, benign: safe}
    category_map: {inj: prompt_injection, leak: confidential}
    default_unsafe_category: other

pandas / parquet / spreadsheet exports are handled: null cells (None, NaN, pd.NA, blank) are
missing values — a missing prompt or label makes the row ``invalid``, never the text ``"nan"`` —
labels compare through ``label_key`` (``1`` / ``1.0`` / ``"1.0"`` / ``True`` all match the
``label_map`` key ``"1"``), and a language column is mapped onto ``LANGUAGES`` (``zh_TW`` ->
``zh-TW``, ``de`` -> ``other``; empty -> detected).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, ClassVar

from ...data_schema import Category, GuardrailSample, Label
from ...data_schema.schema import normalize_language
from ..base import Record, register
from ..sources import LocalFileSource
from ..templates import is_missing, label_key

VALID_CATS = {c.value for c in Category}


def _value(v: Any) -> Any:
    """A cell as a plain Python value; ``None`` when missing."""
    if hasattr(v, "item") and type(v).__module__ == "numpy" and getattr(v, "ndim", 0) == 0:
        v = v.item()
    return None if is_missing(v) else v


@register("local_file")
class LocalFileConverter(LocalFileSource):
    """Column-mapped local file: the mapping lives in the options, so one class serves any internal export."""

    default_splits = ("test",)
    license = "internal"
    MAPPING_OPTIONS: ClassVar[frozenset[str]] = frozenset({"columns", "label_map", "category_map"})  # --set a.b=c

    def convert(self, r: Record, split: str, idx: int) -> GuardrailSample:
        cols = {"prompt": "prompt", "response": "response", "label": "label", "category": "category",
                "language": "language", "attack_type": "attack_type", **self.options.get("columns", {})}

        def get(field: str) -> Any:
            return _value(r.get(cols[field]))

        label = self._label(get("label"))
        it = self.options.get("input_type", "request")
        input_type = str(_value(r.get(it.split(":", 1)[1]))) if it.startswith("column:") else it
        response = get("response")
        attack_type = get("attack_type")
        return self.sample(
            input_type=input_type,
            prompt=str(get("prompt") or ""),
            response=None if response is None else str(response),
            label=label,
            categories=self._categories(get("category")) if label == Label.UNSAFE.value else [],
            language=normalize_language(get("language")) or "auto",
            attack_type=None if attack_type is None else str(attack_type),
            meta={"row": idx, "file": self.path().name},
        )

    def _label(self, value: Any) -> str:
        label_map = {label_key(k): str(v).strip().lower() for k, v in (self.options.get("label_map") or {}).items()}
        raw = label_key(value)  # raises "missing label" for None / NaN / blank
        label = label_map.get(raw, raw)
        if label not in (Label.SAFE.value, Label.UNSAFE.value):
            raise ValueError(f"unmapped label {value!r}; add {raw!r} to label_map")
        return label

    def _categories(self, raw: Any) -> list[str]:
        cmap = self.options.get("category_map", {})
        fallback = self.options.get("default_unsafe_category", "other")
        cats = []
        items = raw if isinstance(raw, Iterable) and not isinstance(raw, str) else str(raw or "").split("|")
        for c in items:  # a list / numpy array (parquet list column) or a "|"-joined string
            c = str(c).strip()
            if c:
                c = cmap.get(c, c)
                cats.append(c if c in VALID_CATS else fallback)
        if not cats and self.options.get("default_unsafe_category"):
            cats = [self.options["default_unsafe_category"]]
        return sorted(set(cats))
