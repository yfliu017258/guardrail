"""Freeze versioned evaluation mixes from processed CSVs.

Model comparisons are only meaningful on a *fixed* eval set. A mix is defined in YAML, built once
with a seed, and written with a manifest of every input file's sha256 — bump the version
(core_v1 -> core_v2) instead of editing a mix in place.

    python -m src.converters.mix --config src/config/data.yaml [--only core_v1]

YAML::

    mixes:
      core_v1:
        seed: 42
        parts:
          - {file: processed_data/toxic_chat.csv, n: 800, strategy: stratified}
          - {file: processed_data/xstest.csv}                                   # all rows
          - {file: processed_data/deepset_prompt_injections.csv, n: 300, filter: {language: [en]}}
          - {file: processed_data/tw_pii_bench.csv, filter: {expected: [mask]}}

A mix is a DIRECTORY, one CSV per source plus a manifest (evaluation loads every ``*.csv`` in it,
and the file name keeps each row's source)::

    processed_data/mixes/core_v1/
        aegis2.csv            rows picked from processed_data/aegis2.csv (source = the part file's name)
        xstest.csv
        manifest.json         seed, parts (file, sha256, available, picked, ...), files, conflicts

Parts from the same source merge into one file (a prompt picked twice is kept once). Rebuilding a
mix replaces every CSV in its directory, so no stale source survives.

``filter`` keys: ``expected`` (block / pass / mask) and ``language`` — detected from the prompt at
mix time with ``detect_language`` (``en`` / ``zh-TW`` / ``zh-CN`` / ``zh`` / ``ja`` / ``ko`` / ``mixed`` /
``other`` / ``unknown``, exact match). Anything else raises. ``n`` (default: all) rows are picked per
part with ``strategy`` ``stratified`` (default; keeps the ``expected`` ratio) / ``balanced`` /
``random`` / ``head``, seeded with ``seed + part index``: the same inputs always give the same mix.
A part that ends up with 0 rows is logged as a warning.

The same prompt with a different ``expected`` in two sources (e.g. an injection-axis ``pass`` vs a
content-safety ``block``) is kept in both files but listed under ``conflicts`` in the manifest and
logged as a warning: fix the parts (drop a config, filter a source) and bump the mix version. The same
prompt with the same ``expected`` in several sources is kept in each (it belongs to each source's
slice) and counted in ``cross_source_duplicates``.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config import get_by_path, load_config
from ..data_schema import (
    EXPECTED,
    Sample,
    detect_language,
    file_sha256,
    read_csv,
    source_of,
    write_csv,
)
from .base import resolve
from .sampling import sample_rows

log = logging.getLogger("converters.mix")

FILTER_KEYS = ("expected", "language")
MANIFEST = "manifest.json"
MAX_LISTED_CONFLICTS = 20


def _allowed(values: Any) -> set[str]:
    return {str(v).strip() for v in (values if isinstance(values, (list, tuple, set)) else [values])}


def filter_rows(rows: list[Sample], spec: dict[str, Any] | None) -> list[Sample]:
    """Rows whose ``expected`` / detected ``language`` is in the allowed list of each ``spec`` key."""
    for key, allowed in (spec or {}).items():
        if key not in FILTER_KEYS:
            raise ValueError(f"mix filter key {key!r} is not one of {FILTER_KEYS}")
        wanted = _allowed(allowed)
        if key == "expected":
            if bad := sorted(wanted - set(EXPECTED)):
                raise ValueError(f"mix filter expected: {bad} not in {EXPECTED}")
            rows = [r for r in rows if r.expected in wanted]
        else:
            rows = [r for r in rows if detect_language(r.prompt) in wanted]
    return rows


def cross_source_conflicts(files: dict[str, list[Sample]]) -> tuple[dict[str, Any], int]:
    """Prompts that appear with different ``expected`` values in different sources (``{count, examples}``),
    and the number of prompts that appear in several sources with the same ``expected``."""
    seen: dict[str, tuple[Sample, dict[str, str]]] = {}
    for source, rows in files.items():
        for r in rows:
            seen.setdefault(r.prompt, (r, {}))[1][source] = r.expected
    bad = [{"prompt": row.prompt[:120], "key": row.key, "expected": dict(sorted(by.items()))}
           for row, by in seen.values() if len(set(by.values())) > 1]
    duplicates = sum(1 for _, by in seen.values() if len(by) > 1 and len(set(by.values())) == 1)
    return {"count": len(bad), "examples": bad[:MAX_LISTED_CONFLICTS]}, duplicates


def build_mix(name: str, spec: dict[str, Any], out_root: str | Path) -> Path:
    """Build mix ``name`` into ``<out_root>/mixes/<name>/`` and return that directory."""
    seed = int(spec.get("seed", 42))
    by_source: dict[str, list[Sample]] = {}
    seen: dict[str, set[str]] = {}
    parts_manifest = []
    for i, part in enumerate(spec["parts"]):
        path = resolve(part["file"])
        strategy = part.get("strategy", "stratified")
        data = filter_rows(read_csv(path), part.get("filter"))
        if not data:
            log.warning("mix %s part %d (%s): 0 rows after filter %s", name, i, part["file"], part.get("filter"))
        picked = sample_rows(data, part.get("n"), strategy, seed + i)
        source = source_of(path)
        rows, prompts = by_source.setdefault(source, []), seen.setdefault(source, set())
        added = 0
        for r in picked:  # parts from one source merge; a prompt picked by two of them is kept once
            if r.prompt not in prompts:
                prompts.add(r.prompt)
                rows.append(r)
                added += 1
        parts_manifest.append({"file": str(part["file"]), "source": source, "sha256": file_sha256(path),
                               "n": part.get("n"), "strategy": strategy, "filter": part.get("filter"),
                               "available": len(data), "picked": len(picked), "added": added})

    out_dir = resolve(out_root) / "mixes" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("*.csv"):  # evaluation loads every CSV here: none may outlive a rebuild
        stale.unlink()
    files = {}
    for source, rows in by_source.items():
        path = out_dir / f"{source}.csv"
        write_csv(rows, path)
        files[path.name] = {"source": source, "rows": len(rows), "sha256": file_sha256(path),
                            "expected": {e: sum(r.expected == e for r in rows) for e in EXPECTED}}
    conflicts, duplicates = cross_source_conflicts(by_source)
    if conflicts["count"]:
        log.warning("mix %s: %d prompts expect different results in different sources, e.g. %s", name,
                    conflicts["count"], conflicts["examples"][:2])
    total = sum(f["rows"] for f in files.values())
    manifest = {"name": name, "seed": seed, "rows": total, "files": files, "parts": parts_manifest,
                "conflicts": conflicts, "cross_source_duplicates": duplicates,
                "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    (out_dir / MANIFEST).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("mix %s -> %s (%d rows in %d files)", name, out_dir, total, len(files))
    return out_dir


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", nargs="+", default=["src/config/data.yaml"])
    p.add_argument("--only", nargs="+", default=None)
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    cfg = load_config(a.config, env_file=None, run_kind="mix")
    out_root = get_by_path(cfg, "processed_dir", "processed_data")
    for name, spec in (get_by_path(cfg, "mixes", {}) or {}).items():
        if a.only and name not in a.only:
            continue
        build_mix(name, spec, out_root)


if __name__ == "__main__":
    main()
