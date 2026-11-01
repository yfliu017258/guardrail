"""Run an evaluation.

    # GPU-free smoke test (mock backend, bundled sample data)
    python -m src.evaluation --config src/config/evaluation/smoke.yaml

    # evaluate a guard model in-process against processed datasets
    python -m src.evaluation --config src/config/evaluation/default.yaml \
        --server-config src/config/server/qwen3guard_gen_4b.yaml \
        --data "processed_data/*/test.csv" --run-name qwen3guard4b --max-samples 2000

    # evaluate the real LiteLLM proxy guardrail
    python -m src.evaluation --config src/config/evaluation/default.yaml --target litellm_apply_guardrail

    # resume an interrupted run / recompute metrics only
    python -m src.evaluation --resume results/20261101-120000_qwen3guard4b_ab12cd34ef56
    python -m src.evaluation --rescore results/... --set metrics.threshold=0.3

Every run writes to ``results/<timestamp>_<name>_<config_hash>/``:
config.yaml (resolved, secrets redacted) · data_manifest.json · environment.json ·
predictions.jsonl · metrics.json · slices.csv · errors.csv · report.md
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

import yaml

from ..config import dump_config, get_by_path, load_config, make_run_dir, set_by_path
from .dataset import load_eval_set
from .metrics import evaluate_rows
from .report import write_report
from .runner import load_done, run_predictions
from .targets import build_target

ARG_TO_KEY = {
    "run_name": "run.name",
    "seed": "run.seed",
    "output_dir": "run.output_dir",
    "data": "data.files",
    "max_samples": "data.max_samples_per_file",
    "target": "target.type",
    "server_config": "target.inprocess.server_config",
    "concurrency": "eval.concurrency",
    "limit": "eval.limit",
    "threshold": "metrics.threshold",
}

log = logging.getLogger("evaluation")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reproducible guardrail evaluation")
    p.add_argument("--config", nargs="+", default=["src/config/evaluation/default.yaml"])
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--env-file", default=".env")
    p.add_argument("--run-name", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--data", nargs="+", default=None, help="processed CSV paths or globs")
    p.add_argument("--max-samples", type=int, default=None, help="per file")
    p.add_argument("--limit", type=int, default=None, help="total cap after merging files")
    p.add_argument("--target", default=None, choices=["inprocess", "guardrail_server", "litellm_apply_guardrail", "litellm_chat"])
    p.add_argument("--server-config", nargs="+", default=None, help="server YAML(s) for the inprocess target")
    p.add_argument("--concurrency", type=int, default=None)
    p.add_argument("--threshold", type=float, default=None, help="re-threshold the pipeline score")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--resume", default=None, metavar="RUN_DIR", help="continue an interrupted run with its saved config")
    g.add_argument("--rescore", default=None, metavar="RUN_DIR", help="recompute metrics from predictions.jsonl only")
    return p.parse_args(argv)


def _metrics(rows: list[dict], cfg: dict) -> dict:
    return evaluate_rows(
        rows,
        threshold=get_by_path(cfg, "metrics.threshold"),
        error_policy=get_by_path(cfg, "eval.error_policy", "exclude"),
        n_boot=int(get_by_path(cfg, "metrics.bootstrap", 1000)),
        ci=float(get_by_path(cfg, "metrics.ci", 0.95)),
        seed=int(get_by_path(cfg, "run.seed", 42)),
        fpr_targets=get_by_path(cfg, "metrics.fpr_targets", [0.01, 0.05]),
        slice_by=get_by_path(cfg, "metrics.slice_by", ["source", "language", "input_type"]),
        min_slice_size=int(get_by_path(cfg, "metrics.min_slice_size", 1)),
    )


def _environment() -> dict:
    import importlib.metadata as md

    return {"python": sys.version, "packages": sorted(f"{d.metadata['Name']}=={d.version}" for d in md.distributions())}


async def _run(cfg: dict, run_dir: Path) -> None:
    samples, manifest = load_eval_set(cfg)
    (run_dir / "data_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    target = build_target(cfg)
    await target.startup()
    try:
        desc = target.describe()
        (run_dir / "target.json").write_text(json.dumps(desc, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        rows = await run_predictions(target, samples, cfg, run_dir / "predictions.jsonl")
    finally:
        await target.shutdown()
    metrics = _metrics(rows, cfg)
    write_report(run_dir, metrics, cfg, manifest, desc, rows)
    o = metrics["overall"]
    log.info("done: n=%s precision=%s recall=%s f1=%s fpr=%s -> %s", o["n"], o["precision"], o["recall"], o["f1"], o["fpr"], run_dir)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    if args.rescore or args.resume:
        run_dir = Path(args.rescore or args.resume)
        cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
        from ..config import load_envs, parse_overrides

        cfg["envs"] = load_envs(args.env_file, cfg.get("env_passthrough", []))  # secrets were redacted in the snapshot
        for k, v in parse_overrides(args.overrides):
            set_by_path(cfg, k, v)
        if args.rescore:
            rows = list(load_done(run_dir / "predictions.jsonl").values())
            manifest = json.loads((run_dir / "data_manifest.json").read_text(encoding="utf-8"))
            desc = json.loads((run_dir / "target.json").read_text(encoding="utf-8")) if (run_dir / "target.json").exists() else {}
            sub = run_dir / f"rescore_{get_by_path(cfg, 'metrics.threshold', 'default')}"
            sub.mkdir(exist_ok=True)
            write_report(sub, _metrics(rows, cfg), cfg, manifest, desc, rows)
            log.info("rescored -> %s", sub)
            return
        asyncio.run(_run(cfg, run_dir))
        return

    cfg = load_config(args.config, vars(args), ARG_TO_KEY, args.overrides, args.env_file, run_kind="evaluation")
    run_dir = make_run_dir(cfg)
    dump_config(cfg, run_dir / "config.yaml")
    (run_dir / "environment.json").write_text(json.dumps(_environment(), indent=2), encoding="utf-8")
    asyncio.run(_run(cfg, run_dir))


if __name__ == "__main__":
    main()
