"""Convert raw datasets into the unified CSV format.

    python -m src.converters --list
    python -m src.converters --dataset toxic_chat --splits test
    python -m src.converters --dataset toxic_chat aegis2 wildguardmix --limit 5000
    python -m src.converters --config src/config/data.yaml            # everything listed there
    python -m src.converters --dataset local_file --set path=data/raw/internal_redteam.jsonl \
        --set columns.prompt=question --set columns.label=is_attack --set label_map.1=unsafe
    python -m src.converters.mix --config src/config/data.yaml        # build fixed eval mixes
"""

from __future__ import annotations

import argparse
import logging

from ..config import get_by_path, load_config, parse_overrides, set_by_path
from .base import CONVERTERS, get_converter


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

    from . import datasets as _  # noqa: F401  (populate registry)
    if a.list:
        for name, cls in sorted(CONVERTERS.items()):
            print(f"{name:28s} {cls.hf_id or '(local)':55s} license={cls.license} gated={cls.gated}")
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

    for name, opts in jobs.items():
        for k, v in parse_overrides(a.overrides):
            set_by_path(opts, k, v)
        cls = get_converter(opts.pop("converter", name))
        splits = a.splits or opts.pop("splits", None)
        limit = a.limit or opts.pop("limit", None)
        opts.pop("enabled", None)
        conv = cls(options=opts, raw_dir=raw_dir, revision=a.revision or opts.get("revision"))
        conv.name = opts.get("name", name)  # same converter class can produce several named outputs
        conv.run(splits, out_dir, limit)


if __name__ == "__main__":
    main()
