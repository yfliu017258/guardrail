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
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from ..config import PROJECT_ROOT, get_by_path, load_config
from ..data_schema import file_sha256, read_csv, write_csv
from ..evaluation.dataset import sample_rows

log = logging.getLogger("converters.mix")


def build_mix(name: str, spec: dict, out_root: Path) -> Path:
    seed = int(spec.get("seed", 42))
    rows, parts_manifest = [], []
    for i, part in enumerate(spec["parts"]):
        path = Path(part["file"])
        path = path if path.is_absolute() else PROJECT_ROOT / path
        data = read_csv(path)
        for key, allowed in (part.get("filter") or {}).items():
            allowed = set(allowed if isinstance(allowed, list) else [allowed])
            data = [r for r in data if str(getattr(r, key) if not key == "label" else r.label.value) in allowed]
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
    manifest = {"name": name, "seed": seed, "rows": n, "sha256": file_sha256(out), "parts": parts_manifest,
                "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
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
