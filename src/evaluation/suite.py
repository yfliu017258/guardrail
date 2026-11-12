"""Run every missing evaluation of a suite: guardrails x datasets, and models x datasets.

One YAML lists what you have; one command produces the runs that do not exist yet. Adding a guardrail,
a model or a dataset = add one line and run the same command again — finished runs are skipped.

    cp src/config/evaluation/suite.example.yaml suite.yaml      # edit: your services, models, datasets
    python -m src.evaluation.suite suite.yaml                   # run what is missing
    python -m src.evaluation.suite suite.yaml --dry-run         # only show what would run
    python -m src.evaluation.suite suite.yaml --max-samples 50  # at most 50 samples per dataset
    python -m src.evaluation.suite suite.yaml --only my-guardrail qwen@core_v1 --force
    python -m src.evaluation.suite suite.yaml --retries 5 --retry-wait 300   # unattended: keep trying what failed
    streamlit run src/analysis/app.py -- --results results/     # look at the results

Each run is ``<subject>@<dataset>`` on a fixed sample (same files, same ``max_samples``):

* a completed run of it exists      -> skipped (``--force`` runs it again; ``--retry-errors`` re-sends only the
                                       rows that failed with a connection error)
* an interrupted / failed run exists -> RESUMED in its directory: finished rows are kept, the rest is sent
* otherwise                          -> a new run

``max_samples`` (suite-wide, per dataset, or ``--max-samples``) caps the samples of a dataset; the sample is
seeded and stratified by ``expected``, so every guardrail and model gets the SAME rows. scripts/run_experiments.sh
runs this in the background with a log file.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
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
    max_samples: int | None = None
    status: str = "pending"   # pending | done (skipped) | ok | resumed | failed
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


def build_jobs(suite: dict[str, Any], max_samples: int | None = None) -> list[Job]:
    """``max_samples``: the command-line cap; it wins over the suite-wide and per-dataset values."""
    raw = suite.get("datasets") or {}
    if not raw:
        raise SystemExit("suite: `datasets` is empty")
    datasets = {name: _as_list(d["path"] if isinstance(d, dict) else d) for name, d in raw.items()}
    directions = {name: (d.get("input_type", "request") if isinstance(d, dict) else "request") for name, d in raw.items()}
    default_cap = suite.get("max_samples", suite.get("limit"))
    caps = {name: max_samples if max_samples is not None
            else (d.get("max_samples", default_cap) if isinstance(d, dict) else default_cap) for name, d in raw.items()}
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
                *_sets("target.http_guardrail", opts)], caps[dname]))
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
            jobs.append(Job("model", mname, dname, files, argv, caps[dname]))
    return jobs


def dataset_id(files: list[str], limit: int | None) -> str:
    _, manifest = load_eval_set({"data": {"files": files}, "eval": {"limit": limit}})
    return _dataset_summary(manifest)["sha256"]


def _connection_errors(run_dir: Path) -> int:
    """Rows of a run whose last attempt failed in transit (they are re-sent by a resume)."""
    from .runner import load_latest

    return sum(1 for r in load_latest(run_dir / "predictions.jsonl") if r.get("_retry"))


def existing_runs(results_dir: Path) -> dict[tuple[str, str], dict[str, Any]]:
    """(run name, dataset id) -> the run to build on: the newest completed one, else the newest unfinished one."""
    found: dict[tuple[str, str], dict[str, Any]] = {}
    if results_dir.is_dir():
        for p in sorted(results_dir.iterdir()):   # directory names start with the UTC time: oldest first
            try:
                info = json.loads((p / "run.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            key = (str(info.get("name")), str((info.get("dataset") or {}).get("sha256")))
            entry = {"dir": p, "completed": info.get("status") == "completed"}
            if entry["completed"] or not found.get(key, {}).get("completed"):
                found[key] = entry
    return found


def _selected(job: Job, only: list[str] | None) -> bool:
    return not only or bool({job.subject, job.dataset, job.name} & set(only))


def run_job(job: Job, key: tuple[str, str] | None, known: dict, args: argparse.Namespace, results: Path,
            concurrency: int | None) -> None:
    prev = known.get(key) if key else None
    if prev and prev["completed"] and not args.force:
        retry = _connection_errors(prev["dir"]) if args.retry_errors else 0
        if not retry:
            job.status, job.detail = "done", prev["dir"].name
            return
    resume = prev is not None and not args.force and (prev["dir"] / "config.yaml").exists()
    if args.dry_run:
        job.status, job.detail = "pending", (f"resume {prev['dir'].name}" if resume else "new run")
        return
    print(f"\n=== {job.kind}: {job.name} ({'resume ' + prev['dir'].name if resume else 'new run'}) ===", flush=True)
    if resume:
        try:
            run_evaluation(["--resume", str(prev["dir"]), "--env-file", args.env_file])
            job.status, job.detail = "resumed", prev["dir"].name
            return
        except SystemExit as e:   # its settings changed since (edited URL / options): cannot continue it
            if "refusing to --resume" not in str(e):
                raise
            print(f"cannot resume ({e}); starting a new run", flush=True)
    cmd = [*job.argv, "--data", *job.data, "--run-name", job.name, "--output-dir", str(results), "--env-file", args.env_file]
    if job.max_samples:
        cmd += ["--limit", str(job.max_samples)]
    if concurrency:
        cmd += ["--concurrency", str(concurrency)]
    before = {d.name for d in results.iterdir()} if results.is_dir() else set()
    try:
        run_evaluation(cmd)
    except BaseException:
        # a run that died before its first prediction (service down, bad key) holds nothing to resume or
        # analyse: drop its directory so retries do not pile up empty runs. The error is in the log.
        for d in (results.iterdir() if results.is_dir() else []):
            if d.name not in before and not (d / "predictions.jsonl").exists():
                shutil.rmtree(d, ignore_errors=True)
        raise
    job.status, job.detail = "ok", ""


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run the missing evaluations of a suite (see the module docstring).")
    p.add_argument("suite", help="suite YAML (see src/config/evaluation/suite.example.yaml)")
    p.add_argument("--results", default=None, help="results directory (default: the suite's `results`, else results/)")
    p.add_argument("--only", nargs="+", default=None, metavar="NAME",
                   help="only these: a guardrail / model name, a dataset name, or <subject>@<dataset>")
    p.add_argument("--max-samples", "--limit", dest="max_samples", type=int, default=None,
                   help="at most this many samples per dataset (seeded, stratified by expected)")
    p.add_argument("--concurrency", type=int, default=None)
    p.add_argument("--force", action="store_true", help="start a new run even if one exists")
    p.add_argument("--retry-errors", action="store_true",
                   help="also re-send the rows of COMPLETED runs that failed with a connection error")
    p.add_argument("--retries", type=int, default=0, help="extra rounds for runs that failed (they are resumed)")
    p.add_argument("--retry-wait", type=float, default=60, help="seconds to wait before each extra round")
    p.add_argument("--dry-run", action="store_true", help="show what would run, run nothing")
    p.add_argument("--env-file", default=".env")
    args = p.parse_args(argv)

    suite = yaml.safe_load(Path(args.suite).read_text(encoding="utf-8")) or {}
    results = Path(args.results or suite.get("results") or "results")
    results = results if results.is_absolute() else PROJECT_ROOT / results
    concurrency = args.concurrency if args.concurrency is not None else suite.get("concurrency")
    jobs = [j for j in build_jobs(suite, args.max_samples) if _selected(j, args.only)]
    if not jobs:
        raise SystemExit(f"nothing selected (--only {args.only}); check the names in {args.suite}")

    for attempt in range(args.retries + 1):
        todo = [j for j in jobs if j.status in ("pending", "failed")]
        if attempt:
            if not todo:
                break
            print(f"\n### round {attempt + 1}: {len(todo)} run(s) failed, retrying in {args.retry_wait:.0f}s", flush=True)
            time.sleep(args.retry_wait)
        known = existing_runs(results)
        for j in todo:
            try:
                run_job(j, (j.name, dataset_id(j.data, j.max_samples)), known, args, results, concurrency)
            except (Exception, SystemExit) as e:  # noqa: BLE001 - one broken service must not block the rest
                j.status, j.detail = "failed", f"{type(e).__name__}: {e}"[:300]
                if not isinstance(e, FileNotFoundError):
                    traceback.print_exc()
        if args.dry_run:
            break

    label = {"pending": "would run", "done": "already done", "ok": "ran", "resumed": "resumed", "failed": "FAILED"}
    print("\n" + "-" * 78)
    for j in jobs:
        cap = f" (max {j.max_samples})" if j.max_samples else ""
        print(f"{label[j.status]:13s} {j.kind:10s} {j.name + cap:40s} {j.detail}")
    print("-" * 78)
    failed = sum(j.status == "failed" for j in jobs)
    print(f"{len(jobs) - failed}/{len(jobs)} runs available" + (f", {failed} FAILED" if failed else "") +
          f"\nresults: {results}\nview:    streamlit run src/analysis/app.py -- --results {results}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
