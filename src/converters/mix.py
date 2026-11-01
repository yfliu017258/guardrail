"""Freeze versioned evaluation mixes from processed CSVs.

Model comparisons are only meaningful on a *fixed* eval set. A mix is defined in
YAML, built once with a seed, and written with a manifest of every input file's
sha256 — bump the version (core_v1 -> core_v2) instead of editing a mix in place.

    python -m src.converters.mix --config src/config/data.yaml [--only core_v1]

YAML::

    mixes:
      core_v1:
        seed: 42
        parts:
          - {file: processed_data/toxic_chat/test.csv, n: 800, strategy: stratified}
          - {file: processed_data/xstest/test.csv}            # all rows
          - {file: processed_data/deepset_prompt_injections/test.csv, n: 300, filter: {language: [en]}}
          - {file: processed_data/tw_pii_bench/test.csv, filter: {categories: [privacy_pii], attack_type: [null]}}

``filter`` keys are ``GuardrailSample`` fields (anything else raises). A row passes when its value is in
the allowed list; a list field (``categories``) passes when ANY element is; ``null`` matches an empty
value. A part that ends up with 0 rows is logged as a warning.

``sample_id`` includes the source, so the same text from two converters is two rows. When those rows
disagree on the gold label (e.g. an injection-axis "safe" vs a content-safety "unsafe" for one
prompt), the mix keeps both but lists them under ``label_conflicts`` in its manifest and logs a
warning: fix the parts (drop a config, filter a source) and bump the mix version.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from ..config import PROJECT_ROOT, get_by_path, load_config
from ..data_schema import GuardrailSample, file_sha256, read_csv, write_csv
from ..evaluation.dataset import sample_rows

log = logging.getLogger("converters.mix")


def _key(value: Any) -> str | None:
    """Comparable form of a field value or an allowed value: enums by value, empty -> None."""
    if isinstance(value, Enum):
        value = value.value
    return None if value is None or value == "" else str(value)


def label_conflicts(rows: list[GuardrailSample], max_examples: int = 20) -> dict[str, Any]:
    """Texts (input_type + prompt + response, source ignored) that appear with different gold labels."""
    seen: dict[tuple[str, str, str], dict[str, list]] = {}
    for r in rows:
        entry = seen.setdefault((r.input_type, r.prompt, r.response or ""), {"labels": [], "sources": [], "ids": []})
        gold = r.label.value if r.refusal_label is None else f"{r.label.value}/{r.refusal_label}"
        for key, value in (("labels", gold), ("sources", r.source), ("ids", r.sample_id)):
            if value not in entry[key]:
                entry[key].append(value)
    bad = [{"input_type": k[0], "text": (k[2] or k[1])[:120], "labels": sorted(v["labels"]),
            "sources": sorted(v["sources"]), "sample_ids": v["ids"]}
           for k, v in seen.items() if len(v["labels"]) > 1]
    return {"count": len(bad), "examples": bad[:max_examples]}


def filter_rows(rows: list[GuardrailSample], spec: dict[str, Any] | None) -> list[GuardrailSample]:
    for key, allowed in (spec or {}).items():
        if key not in GuardrailSample.model_fields:
            raise ValueError(f"mix filter key {key!r} is not a GuardrailSample field ({', '.join(GuardrailSample.model_fields)})")
        wanted = {_key(a) for a in (allowed if isinstance(allowed, list) else [allowed])}

        def ok(r: GuardrailSample, key: str = key, wanted: set = wanted) -> bool:
            value = getattr(r, key)
            if isinstance(value, list):  # categories: membership, an empty list is "null"
                return any(_key(v) in wanted for v in value) if value else None in wanted
            return _key(value) in wanted

        rows = [r for r in rows if ok(r)]
    return rows


def build_mix(name: str, spec: dict, out_root: Path) -> Path:
    seed = int(spec.get("seed", 42))
    rows, parts_manifest = [], []
    for i, part in enumerate(spec["parts"]):
        path = Path(part["file"])
        path = path if path.is_absolute() else PROJECT_ROOT / path
        data = filter_rows(read_csv(path), part.get("filter"))
        if not data:
            log.warning("mix %s part %d (%s): 0 rows after filter %s", name, i, part["file"], part.get("filter"))
        picked = sample_rows(data, part.get("n"), part.get("strategy", "stratified"), seed + i)
        rows.extend(picked)
        parts_manifest.append({"file": part["file"], "sha256": file_sha256(path), "available": len(data),
                               "picked": len(picked), "strategy": part.get("strategy", "stratified")})
    uniq, seen = [], set()
    for r in rows:
        if r.sample_id not in seen:
            seen.add(r.sample_id)
            uniq.append(r)
    out = out_root / "mixes" / f"{name}.csv"
    n = write_csv(uniq, out)
    conflicts = label_conflicts(uniq)
    if conflicts["count"]:
        log.warning("mix %s: %d texts appear with conflicting labels across sources, e.g. %s", name,
                    conflicts["count"], conflicts["examples"][:2])
    manifest = {"name": name, "seed": seed, "rows": n, "sha256": file_sha256(out), "parts": parts_manifest,
                "label_conflicts": conflicts, "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    out.with_suffix(".manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("mix %s -> %s (%d rows)", name, out, n)
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", nargs="+", default=["src/config/data.yaml"])
    p.add_argument("--only", nargs="+", default=None)
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    cfg = load_config(a.config, env_file=None, run_kind="mix")
    out_root = PROJECT_ROOT / get_by_path(cfg, "processed_dir", "processed_data")
    for name, spec in (get_by_path(cfg, "mixes", {}) or {}).items():
        if a.only and name not in a.only:
            continue
        build_mix(name, spec, out_root)


if __name__ == "__main__":
    main()
