"""Run every missing evaluation of a suite: guardrails x datasets, and models x datasets.

One YAML lists what you have; one command produces the runs that do not exist yet. Adding a guardrail,
a model or a dataset = add one line and run the same command again — finished runs are skipped.

    cp src/config/evaluation/suite.example.yaml suite.yaml      # edit: your services, models, datasets
    python -m src.evaluation.suite suite.yaml                   # run what is missing
    python -m src.evaluation.suite suite.yaml --dry-run         # only show what would run
    python -m src.evaluation.suite suite.yaml --limit 50        # quick check on 50 samples per dataset
    python -m src.evaluation.suite suite.yaml --only my-guardrail --force
    streamlit run src/analysis/app.py -- --results results/     # look at the results

A run is "done" when ``results/`` holds a completed run with the same name (``<subject>@<dataset>``) on
the same data (same files, same sample limit). Changed data or ``--force`` runs it again.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ..config import PROJECT_ROOT
from .__main__ import _dataset_summary, main as run_evaluation
from .dataset import load_eval_set

GUARDRAIL_CONFIG = "src/config/evaluation/guardrail_service.yaml"
MODEL_CONFIG = "src/config/evaluation/model_safety_litellm.yaml"


@dataclass
class Job:
    kind: str                 # guardrail | model
    subject: str
    dataset: str
    data: list[str]
    argv: list[str] = field(default_factory=list)
    status: str = "pending"   # pending | done (skipped) | ok | failed
    detail: str = ""

    @property
    def name(self) -> str:
        return f"{self.subject}@{self.dataset}"


def _as_list(v: Any) -> list[str]:
    return [str(x) for x in v] if isinstance(v, (list, tuple)) else [str(v)]


def _sets(prefix: str, options: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for k, v in options.items():
        out += ["--set", f"{prefix}.{k}={json.dumps(v, ensure_ascii=False)}"]
    return out


def build_jobs(suite: dict[str, Any]) -> list[Job]:
    datasets = {name: _as_list(d["path"] if isinstance(d, dict) else d) for name, d in (suite.get("datasets") or {}).items()}
    directions = {name: (d.get("input_type", "request") if isinstance(d, dict) else "request")
                  for name, d in (suite.get("datasets") or {}).items()}
    if not datasets:
        raise SystemExit("suite: `datasets` is empty")
    jobs: list[Job] = []
    for gname, g in (suite.get("guardrails") or {}).items():
        g = dict(g or {})
        url = g.pop("url", None)
        only = g.pop("datasets", None)
        if not url:
            raise SystemExit(f"suite: guardrails.{gname} needs `url`")
        g.setdefault("api_key_env", None)   # no key unless one is named
        for dname, files in datasets.items():
            if only and dname not in only:
                continue
            opts = dict(g)
            if directions[dname] == "response":
                opts.setdefault("canary", False)   # the default canary is a request-style injection
            jobs.append(Job("guardrail", gname, dname, files, [
                "--config", GUARDRAIL_CONFIG, "--service-url", url, "--input-type", directions[dname],
                *_sets("target.http_guardrail", opts)]))
    judge = suite.get("judge") or {}
    for mname, m in (suite.get("models") or {}).items():
        m = dict(m or {})
        only = m.pop("datasets", None)
        on_blocked = m.pop("on_blocked", None)
        m.setdefault("model", mname)
        for dname, files in datasets.items():
            if directions[dname] != "request" or (only and dname not in only):
                continue                          # response data carries its own answers: nothing to generate
            argv = ["--config", MODEL_CONFIG, *_sets("task.model_safety.generator", m),
                    *_sets("task.model_safety.judge", judge)]
            if on_blocked:
                argv += ["--set", f"task.model_safety.on_blocked={on_blocked}"]
            jobs.append(Job("model", mname, dname, files, argv))
    return jobs


def dataset_id(files: list[str], limit: int | None) -> str:
    _, manifest = load_eval_set({"data": {"files": files}, "eval": {"limit": limit}})
    return _dataset_summary(manifest)["sha256"]


def finished_runs(results_dir: Path) -> dict[tuple[str, str], str]:
    """(run name, dataset id) -> run directory, for completed runs."""
    done: dict[tuple[str, str], str] = {}
    if results_dir.is_dir():
        for p in sorted(results_dir.iterdir()):
            try:
                info = json.loads((p / "run.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if info.get("status") == "completed":
                done[(str(info.get("name")), str((info.get("dataset") or {}).get("sha256")))] = p.name
    return done


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run the missing evaluations of a suite (see the module docstring).")
    p.add_argument("suite", help="suite YAML (see src/config/evaluation/suite.example.yaml)")
    p.add_argument("--results", default=None, help="results directory (default: the suite's `results`, else results/)")
    p.add_argument("--only", nargs="+", default=None, help="only these guardrails / models / datasets")
    p.add_argument("--limit", type=int, default=None, help="cap the samples per dataset (quick functional check)")
    p.add_argument("--concurrency", type=int, default=None)
    p.add_argument("--force", action="store_true", help="run even if a completed run exists")
    p.add_argument("--dry-run", action="store_true", help="show what would run, run nothing")
    p.add_argument("--env-file", default=".env")
    args = p.parse_args(argv)

    suite = yaml.safe_load(Path(args.suite).read_text(encoding="utf-8")) or {}
    results = Path(args.results or suite.get("results") or "results")
    results = results if results.is_absolute() else PROJECT_ROOT / results
    limit = args.limit if args.limit is not None else suite.get("limit")
    concurrency = args.concurrency if args.concurrency is not None else suite.get("concurrency")
    jobs = build_jobs(suite)
    if args.only:
        jobs = [j for j in jobs if j.subject in args.only or j.dataset in args.only]
    done = finished_runs(results)

    for j in jobs:
        try:
            key = (j.name, dataset_id(j.data, limit))
        except Exception as e:  # noqa: BLE001 - a missing data file must not stop the other jobs
            j.status, j.detail = "failed", f"data: {e}"
            continue
        if key in done and not args.force:
            j.status, j.detail = "done", done[key]
            continue
        if args.dry_run:
            continue
        cmd = [*j.argv, "--data", *j.data, "--run-name", j.name, "--output-dir", str(results), "--env-file", args.env_file]
        if limit:
            cmd += ["--limit", str(limit)]
        if concurrency:
            cmd += ["--concurrency", str(concurrency)]
        print(f"\n=== {j.kind}: {j.name} ===", flush=True)
        try:
            run_evaluation(cmd)
            j.status = "ok"
        except (Exception, SystemExit) as e:  # noqa: BLE001 - keep going: one broken service must not block the rest
            j.status, j.detail = "failed", f"{type(e).__name__}: {e}"[:300]
            traceback.print_exc()

    label = {"pending": "would run", "done": "already done", "ok": "ran", "failed": "FAILED"}
    print("\n" + "-" * 78)
    for j in jobs:
        print(f"{label[j.status]:13s} {j.kind:10s} {j.name:40s} {j.detail}")
    print("-" * 78)
    print(f"results: {results}\nview:    streamlit run src/analysis/app.py -- --results {results}")
    return 1 if any(j.status == "failed" for j in jobs) else 0


if __name__ == "__main__":
    sys.exit(main())
