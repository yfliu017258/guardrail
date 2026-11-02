"""Convert raw datasets into the two-column CSV format (``prompt,expected``).

    python -m src.converters --list
    python -m src.converters --dataset toxic_chat                       # -> processed_data/toxic_chat.csv
    python -m src.converters --dataset aegis2 --splits test train       # -> aegis2.csv + aegis2.train.csv
    python -m src.converters --dataset toxic_chat aegis2 --limit 5000
    python -m src.converters --config src/config/data.yaml            # everything listed there
    python -m src.converters --dataset local_file --set path=data/raw/internal_redteam.jsonl \
        --set columns.prompt=question --set columns.label=is_attack --set label_map.1=block --set label_map.0=pass
    python -m src.converters --dataset local_file ... --set 'label_map={"1": block, "0": pass}'   # same, one flag
    python -m src.converters.mix --config src/config/data.yaml        # build frozen eval mixes

Each converter writes ``<processed_dir>/<name>.csv`` (+ ``<name>.<split>.csv`` for non-eval splits)
and ``<name>.manifest.json``; see src/converters/base.py.

``--set a.1=x`` normally means "list index 1"; for the mapping options a converter declares in
``MAPPING_OPTIONS`` (local_file: columns / label_map) a missing mapping is created first, so
``label_map.1`` is the key ``"1"``.

``--list`` exits non-zero (after listing what did load) when a converter module failed to import.
"""

from __future__ import annotations

import argparse
import logging
import sys

from ..config import get_by_path, load_config, parse_overrides, set_by_path
from .base import CONVERTERS


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", nargs="+", default=None, help="converter names (see --list)")
    p.add_argument("--splits", nargs="+", default=None)
    p.add_argument("--limit", type=int, default=None, help="max rows per split (debugging)")
    p.add_argument("--raw-dir", default=None)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--revision", default=None, help="pin HF dataset revision (commit sha) for reproducibility")
    p.add_argument("--config", nargs="+", default=None, help="data YAML with a `datasets:` mapping")
    p.add_argument("--set", dest="overrides", action="append", default=[], help="converter option key=value")
    p.add_argument("--list", action="store_true")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    if a.list:
        for name, cls in CONVERTERS.items():
            print(f"{name:28s} {cls.hf_id or '(local)':55s} license={cls.license} gated={cls.gated}")
        if CONVERTERS.import_errors:  # their converters are missing from the list above
            for module, err in CONVERTERS.import_errors.items():
                print(f"error: converter module {module} failed to import: {err}", file=sys.stderr)
            sys.exit(1)
        return

    cfg = load_config(a.config, env_file=".env", run_kind="convert") if a.config else {}
    raw_dir = a.raw_dir or get_by_path(cfg, "raw_dir", "data/raw")
    out_dir = a.out_dir or get_by_path(cfg, "processed_dir", "processed_data")
    jobs: dict[str, dict] = {}
    if a.dataset:
        for name in a.dataset:
            jobs[name] = dict(get_by_path(cfg, f"datasets.{name}", {}) or {})
    else:
        jobs = {k: dict(v or {}) for k, v in (get_by_path(cfg, "datasets", {}) or {}).items() if (v or {}).get("enabled", True)}
    if not jobs:
        p.error("nothing to do: pass --dataset or --config with a `datasets:` mapping")

    for name, job in jobs.items():
        mapping_options = CONVERTERS.get(job.get("converter", name)).MAPPING_OPTIONS
        for k, v in parse_overrides(a.overrides):  # parsed per job: YAML values must not be shared between jobs
            head = k.split(".", 1)[0]
            if "." in k and head in mapping_options and not isinstance(job.get(head), dict):
                job[head] = {}  # `label_map.1=block`: "1" is a key here, not a list index
            set_by_path(job, k, v)
        # `converter:` lets one class produce several named outputs (nemotron_safety_v3_zh / _zhtw)
        conv = CONVERTERS.build(job, type_key="converter", default=name, name=name, raw_dir=raw_dir,
                                revision=a.revision)
        conv.run(a.splits or job.get("splits"), out_dir, a.limit or job.get("limit"))


if __name__ == "__main__":
    main()
