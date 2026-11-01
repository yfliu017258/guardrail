"""Generic converter for local files (internal red-team sets, labelled production logs, SaaS exports).

Options (YAML or ``--set``)::

    path: data/raw/internal_redteam.jsonl      # .csv / .jsonl / .json / .parquet
    input_type: request                       # or `column:<name>` to read it per row
    columns: {prompt: question, response: answer, label: is_attack, category: type, language: lang}
    label_map: {"1": unsafe, "0": safe, attack: unsafe, benign: safe}
    category_map: {inj: prompt_injection, leak: confidential}
    default_unsafe_category: other
"""

from __future__ import annotations

from typing import Any

from ...data_schema import Category, GuardrailSample, Label
from ..base import Record, register
from ..sources import LocalFileSource

VALID_CATS = {c.value for c in Category}


@register("local_file")
class LocalFileConverter(LocalFileSource):
    """Column-mapped local file: the mapping lives in the options, so one class serves any internal export."""

    default_splits = ("test",)
    license = "internal"

    def convert(self, r: Record, split: str, idx: int) -> GuardrailSample:
        cols = {"prompt": "prompt", "response": "response", "label": "label", "category": "category",
                "language": "language", "attack_type": "attack_type", **self.options.get("columns", {})}
        label = self._label(r.get(cols["label"], ""))
        it = self.options.get("input_type", "request")
        input_type = str(r.get(it.split(":", 1)[1])) if it.startswith("column:") else it
        return self.sample(
            input_type=input_type,
            prompt=str(r.get(cols["prompt"]) or ""),
            response=(str(r[cols["response"]]) if r.get(cols["response"]) else None),
            label=label,
            categories=self._categories(r.get(cols["category"])) if label == Label.UNSAFE.value else [],
            language=r.get(cols["language"]) or "auto",
            attack_type=r.get(cols["attack_type"]) or None,
            meta={"row": idx, "file": self.path().name},
        )

    def _label(self, value: Any) -> str:
        label_map = {str(k).lower(): v for k, v in self.options.get("label_map", {}).items()}
        raw = str(value).strip().lower()
        label = label_map.get(raw, raw)
        if label not in (Label.SAFE.value, Label.UNSAFE.value):
            raise ValueError(f"unmapped label {raw!r}; add it to label_map")
        return label

    def _categories(self, raw: Any) -> list[str]:
        cmap = self.options.get("category_map", {})
        fallback = self.options.get("default_unsafe_category", "other")
        cats = []
        for c in (raw if isinstance(raw, list) else str(raw or "").split("|")):
            c = str(c).strip()
            if c:
                c = cmap.get(c, c)
                cats.append(c if c in VALID_CATS else fallback)
        if not cats and self.options.get("default_unsafe_category"):
            cats = [self.options["default_unsafe_category"]]
        return sorted(set(cats))
