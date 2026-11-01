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

import csv
import json
from pathlib import Path
from typing import Any, Iterable

from ...config import PROJECT_ROOT
from ...data_schema import Category, Label
from ..base import BaseConverter, register

VALID_CATS = {c.value for c in Category}


def read_records(path: Path) -> Iterable[dict[str, Any]]:
    suf = path.suffix.lower()
    if suf == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as f:
            yield from csv.DictReader(f)
    elif suf == ".jsonl":
        with path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)
    elif suf == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        yield from (data if isinstance(data, list) else data.get("data", []))
    elif suf == ".parquet":
        import pandas as pd

        yield from pd.read_parquet(path).to_dict("records")
    else:
        raise ValueError(f"unsupported file type: {path}")


@register("local_file")
class LocalFileConverter(BaseConverter):
    default_splits = ("test",)
    license = "internal"

    def load_raw(self, split: str) -> Iterable[dict[str, Any]]:
        p = Path(self.options["path"])
        p = p if p.is_absolute() else PROJECT_ROOT / p
        return read_records(p)

    def convert(self, r: dict[str, Any], split: str, idx: int):
        cols = {"prompt": "prompt", "response": "response", "label": "label", "category": "category",
                "language": "language", "attack_type": "attack_type", **self.options.get("columns", {})}
        label_map = {str(k).lower(): v for k, v in self.options.get("label_map", {}).items()}
        raw_label = str(r.get(cols["label"], "")).strip().lower()
        label = label_map.get(raw_label, raw_label)
        if label not in (Label.SAFE.value, Label.UNSAFE.value):
            raise ValueError(f"unmapped label {raw_label!r}; add it to label_map")

        it = self.options.get("input_type", "request")
        input_type = str(r.get(it.split(":", 1)[1])) if it.startswith("column:") else it

        cats: list[str] = []
        if label == "unsafe":
            cmap = self.options.get("category_map", {})
            raw_cats = r.get(cols["category"]) or ""
            for c in (raw_cats if isinstance(raw_cats, list) else str(raw_cats).split("|")):
                c = str(c).strip()
                if not c:
                    continue
                c = cmap.get(c, c)
                cats.append(c if c in VALID_CATS else self.options.get("default_unsafe_category", "other"))
            if not cats and self.options.get("default_unsafe_category"):
                cats = [self.options["default_unsafe_category"]]
        return self.sample(
            split=split,
            input_type=input_type,
            prompt=str(r.get(cols["prompt"]) or ""),
            response=(str(r[cols["response"]]) if r.get(cols["response"]) else None),
            label=label,
            categories=sorted(set(cats)),
            language=r.get(cols["language"]) or "auto",
            attack_type=r.get(cols["attack_type"]) or None,
            meta={"row": idx, "file": Path(self.options["path"]).name},
        )
