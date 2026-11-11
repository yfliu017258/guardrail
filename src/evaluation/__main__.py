"""Run an evaluation task.

Tasks (``task.type``, registry ``TASKS``):
  guardrail      is the guardrail's block decision right?              (default)
  model_safety   does a BARE model refuse unsafe prompts / answer safe ones?   (generator + refusal judge)

    # GPU-free smoke tests
    python -m src.evaluation --config src/config/evaluation/smoke.yaml
    python -m src.evaluation --config src/config/evaluation/smoke_model_safety.yaml

Data: CSV files with the columns ``prompt,expected`` (block | pass | mask; optionally ``response``); the file
name is the source. ``--data`` takes files, globs or directories (a mix).

    # guardrail: evaluate a RUNNING guardrail server (python -m src.server --config ...)
    python -m src.evaluation --server-url http://localhost:8080 --data processed_data/mixes/core_v1/ \
        --run-name qwen3guard4b
    python -m src.evaluation --server-url http://localhost:8080 --endpoint guardrail_api ...   # production endpoint

    # guardrail: ANY guardrail service through an adapter (default: LiteLLM generic guardrail API)
    python -m src.evaluation --config src/config/evaluation/guardrail_service.yaml --service-url http://svc:8080
    python -m src.evaluation --config src/config/evaluation/guardrail_service.yaml --input-type response --data ...

    # bare model: Qwen3.6-27B with a cascade refusal judge
    python -m src.evaluation --config src/config/evaluation/model_safety.yaml src/config/models/qwen3_6_27b.yaml

    # resume an interrupted run / recompute metrics only
    python -m src.evaluation --resume results/<run_dir>
    python -m src.evaluation --rescore results/<run_dir> --set metrics.threshold=0.3

Every run writes ``results/<UTC>_<name>_<config_hash>/``: config.yaml (secrets redacted) · data_manifest.json ·
environment.json · run.json · task.json · predictions.jsonl · metrics.json · slices.csv · errors.csv · report.md
"""

from __future__ import annotations

import argparse
import asyncio
import fnmatch
import json
import logging
import sys
from pathlib import Path

import yaml

from ..config import (config_hash, dump_config, get_by_path, load_config, load_envs, make_run_dir, parse_overrides,
                      set_by_path)
from .dataset import load_eval_set
from .report import write_report
from .runner import load_latest, run_predictions
from .tasks import EvalTask, build_task

ARG_TO_KEY = {
    "run_name": "run.name",
    "seed": "run.seed",
    "output_dir": "run.output_dir",
    "data": "data.files",
    "max_samples": "data.max_samples_per_file",
    "task": "task.type",
    "target": "target.type",
    "server_config": "target.inprocess.server_config",
    "server_url": "target.guardrail_server.base_url",
    "endpoint": "target.guardrail_server.endpoint",
    "service_url": "target.http_guardrail.base_url",
    "input_type": "target.http_guardrail.input_type",
    "concurrency": "eval.concurrency",
    "limit": "eval.limit",
    "threshold": "metrics.threshold",
}

log = logging.getLogger("evaluation")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    from .targets import TARGETS
    from .tasks import TASKS

    p = argparse.ArgumentParser(description="Reproducible guardrail / model-safety evaluation")
    p.add_argument("--config", nargs="+", default=["src/config/evaluation/default.yaml"])
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--env-file", default=".env")
    p.add_argument("--run-name", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--data", nargs="+", default=None, help="processed CSV paths or globs")
    p.add_argument("--max-samples", type=int, default=None, help="per file")
    p.add_argument("--limit", type=int, default=None, help="total cap after merging files")
    p.add_argument("--task", default=None, choices=TASKS.names())
    p.add_argument("--target", default=None, choices=TARGETS.names(), help="guardrail task only")
    p.add_argument("--server-url", default=None, help="guardrail server base URL (guardrail_server target)")
    p.add_argument("--endpoint", default=None, choices=["classify", "guardrail_api"],
                   help="guardrail_server: /v1/classify (per-stage detail) or the production endpoint")
    p.add_argument("--service-url", default=None, help="guardrail service base URL (http_guardrail target)")
    p.add_argument("--input-type", default=None, choices=["request", "response"],
                   help="http_guardrail: judge the user prompt, or the model answer (needs a `response` column)")
    p.add_argument("--server-config", nargs="+", default=None, help="server YAML(s) for the inprocess target")
    p.add_argument("--concurrency", type=int, default=None)
    p.add_argument("--threshold", type=float, default=None, help="re-threshold the score")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--resume", default=None, metavar="RUN_DIR", help="continue an interrupted run with its saved config")
    g.add_argument("--rescore", default=None, metavar="RUN_DIR", help="recompute metrics from predictions.jsonl only")
    g.add_argument("--list", action="store_true", help="list registered tasks / targets / generators / judges")
    p.add_argument("--force", action="store_true", help="--resume even if data files or the task changed")
    return p.parse_args(argv)


def compute_metrics(task: EvalTask, rows: list[dict]) -> dict:
    """Task metrics + the same metrics on the contamination-free subset (if the model declares training data)."""
    metrics = task.compute_metrics(rows)
    pats = task.contaminated_patterns()
    if pats:
        dirty = {r["source"] for r in rows if any(fnmatch.fnmatch(str(r["source"]), p) for p in pats)}
        clean_rows = [r for r in rows if r["source"] not in dirty]
        metrics["contamination"] = {"patterns": pats, "excluded_sources": sorted(dirty), "n_clean": len(clean_rows)}
        if dirty and clean_rows:
            m = task.compute_metrics(clean_rows)
            metrics["clean"] = {k: m[k] for k in ("n_total", "n_scored", "overall", "ci", "score_metrics") if k in m}
    return metrics


def _environment() -> dict:
    import importlib.metadata as md

    return {"python": sys.version, "packages": sorted(f"{d.metadata['Name']}=={d.version}" for d in md.distributions())}


def _write_json(path: Path, obj) -> None:
    from ..config import redact

    path.write_text(json.dumps(redact(obj), ensure_ascii=False, indent=2, default=str), encoding="utf-8")


#: describe() keys that legitimately differ between the start of a run and a resume of it
#: (server_identity stays: a resume must talk to the same guardrail)
_VOLATILE_DESC_KEYS = ("server_info", "server_info_end", "server_changed_during_run", "server_config_hash",
                       "server_segments")


def _merge_segments(old: dict | None, desc: dict) -> dict:
    """A resumed run keeps the history of every server segment it talked to, and stays flagged as changed
    once any segment differed (a resume must not erase that). ``old`` = task.json before this segment."""
    if not old or "server_identity" not in desc:
        return desc
    segments = list(old.get("server_segments") or [{"start": old.get("server_info"), "end": old.get("server_info_end"),
                                                    "identity": old.get("server_identity")}])
    segments.append({"start": desc.get("server_info"), "end": desc.get("server_info_end"),
                     "identity": desc.get("server_identity")})
    changed = old.get("server_changed_during_run") is True or desc.get("server_changed_during_run") is True \
        or len({json.dumps(seg.get("identity"), sort_keys=True) for seg in segments}) > 1
    return {**desc, "server_segments": segments, "server_changed_during_run": True if changed
            else desc.get("server_changed_during_run")}


def _target_warnings(desc: dict) -> list[str]:
    """Warnings derived from what the target observed (shared by a run and a rescore of it)."""
    out = []
    if desc.get("server_changed_during_run") is True:
        out.append("the guardrail server changed during the run (pipeline_hash / code differ between start and end, "
                   "or between resumed segments): predictions mix two configurations")
    elif desc.get("server_changed_during_run") is None and "server_info_end" in desc:
        out.append("could not verify the guardrail server at the end of the run "
                   f"({(desc.get('server_info_end') or {}).get('error', 'no /v1/info')})")
    return out


def _stable(desc: dict) -> dict:
    return {k: v for k, v in desc.items() if k not in _VOLATILE_DESC_KEYS}


def _check_resume(run_dir: Path, manifest: dict, desc: dict, force: bool) -> None:
    """A resumed run must use the data and task settings of the original one, or the run mixes two configs."""
    problems = []
    old_m = run_dir / "data_manifest.json"
    if old_m.exists():
        old = {f["path"]: f["sha256"] for f in json.loads(old_m.read_text(encoding="utf-8")).get("files", [])}
        new = {f["path"]: f["sha256"] for f in manifest.get("files", [])}
        if old != new:
            problems.append(f"data files changed: {sorted(set(old.items()) ^ set(new.items()))}")
    old_t = next((p for p in (run_dir / "task.json", run_dir / "target.json") if p.exists()), None)
    if old_t is not None:
        from ..config import redact

        old_desc = _stable(json.loads(old_t.read_text(encoding="utf-8")))
        if config_hash(old_desc) != config_hash(_stable(json.loads(json.dumps(redact(desc), default=str)))):
            problems.append(f"task/target settings differ from {old_t.name} (e.g. an edited server preset)")
    if problems and not force:
        raise SystemExit("refusing to --resume: " + "; ".join(problems) + ". Start a new run, or pass --force.")


#: version of the results contract (run.json + predictions.jsonl + metrics.json) the analysis app reads
RESULTS_SCHEMA_VERSION = 1


def _now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _dataset_summary(manifest: dict) -> dict:
    """One identity for the eval set: runs are comparable sample-by-sample only when ``sha256`` matches."""
    import hashlib

    files = manifest.get("files", [])
    ident = json.dumps([[f["source"], f["sha256"], f["rows_used"]] for f in files] + [manifest.get("n_samples")])
    parents = {str(Path(f["path"]).parent.name) for f in files}
    name = parents.pop() if len(parents) == 1 and len(files) > 1 else "+".join(f["source"] for f in files)
    return {"name": name, "n": manifest.get("n_samples"), "sha256": hashlib.sha256(ident.encode()).hexdigest()[:16],
            "sources": [f["source"] for f in files], "expected_counts": manifest.get("expected_counts")}


def write_run_json(run_dir: Path, cfg: dict, status: str, **fields) -> dict:
    """``run.json`` — the index entry the analysis app lists runs by. Rewritten as the run progresses
    (``running`` -> ``completed`` | ``failed``); earlier fields are kept."""
    path = run_dir / "run.json"
    old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    meta = cfg.get("meta", {}) or {}
    doc = {
        "schema_version": RESULTS_SCHEMA_VERSION,
        "run_id": run_dir.name,
        "name": get_by_path(cfg, "run.name"),
        "task": get_by_path(cfg, "task.type", "guardrail"),
        "started_at": _now(),
        "git_sha": (meta.get("git") or {}).get("commit"),
        "config_hash": meta.get("config_hash"),
        **old, **fields, "status": status,
    }
    if status != "running":
        doc["finished_at"] = _now()
    _write_json(path, doc)
    return doc


async def _run(cfg: dict, run_dir: Path, resume: bool = False, force: bool = False) -> None:
    write_run_json(run_dir, cfg, "running")
    try:
        await _run_inner(cfg, run_dir, resume, force)
    except BaseException as e:
        write_run_json(run_dir, cfg, "failed", error=f"{type(e).__name__}: {e}"[:500])
        raise


async def _run_inner(cfg: dict, run_dir: Path, resume: bool, force: bool) -> None:
    samples, manifest = load_eval_set(cfg)
    task = build_task(cfg)
    n_before = len(samples)
    samples = task.select(samples)
    manifest["task_selected"] = {"task": task.registry_name, "kept": len(samples), "dropped": n_before - len(samples)}
    try:
        await task.startup()  # before describe(): a server target only knows what it evaluates once connected
    except BaseException:
        await task.shutdown()  # release what startup opened (HTTP clients, models) before reporting the failure
        raise
    previous = None
    if resume and (run_dir / "task.json").exists():  # read before this segment overwrites it
        previous = json.loads((run_dir / "task.json").read_text(encoding="utf-8"))
    try:
        desc = task.describe()
        if resume:
            _check_resume(run_dir, manifest, desc, force)
        _write_json(run_dir / "data_manifest.json", manifest)
        _write_json(run_dir / "task.json", desc)
        write_run_json(run_dir, cfg, "running", subject=task.subject(), dataset=_dataset_summary(manifest),
                       judge=desc.get("judge"), error=None)
        rows = await run_predictions(task, samples, cfg, run_dir / "predictions.jsonl")
    finally:
        await task.shutdown()
    desc = task.describe()  # again: targets add what they observed at the end (e.g. the server's identity)
    if resume:
        desc = _merge_segments(previous, desc)
    _write_json(run_dir / "task.json", desc)
    metrics = compute_metrics(task, rows)
    for msg in _target_warnings(desc):
        log.error(msg)
        metrics.setdefault("warnings", []).append(msg)
    write_report(run_dir, metrics, cfg, manifest, desc, rows)
    o = metrics["overall"]
    write_run_json(run_dir, cfg, "completed", n_scored=metrics.get("n_scored"), n_total=metrics.get("n_total"),
                   has_scores=any(r.get("score") is not None for r in rows), warnings=metrics.get("warnings") or [])
    log.info("done [%s]: n=%s precision=%s recall=%s f1=%s fpr=%s -> %s", task.registry_name, o["n"], o["precision"],
             o["recall"], o["f1"], o["fpr"], run_dir)


def _list() -> None:
    from ..refuse_judge import REFUSAL_JUDGES
    from .generation import GENERATORS
    from .targets import TARGETS
    from .tasks import TASKS

    from .adapters import ADAPTERS

    for title, reg in (("tasks", TASKS), ("targets (guardrail task)", TARGETS),
                       ("adapters (http_guardrail target)", ADAPTERS), ("generators (model_safety)", GENERATORS),
                       ("refusal judges", REFUSAL_JUDGES)):
        print(f"{title}:")
        for d in reg.describe():
            print(f"  {d['name']:26s} {d['doc']}")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # it logs every URL (incl. user:pw@) at INFO
    if args.list:
        _list()
        return

    if args.rescore or args.resume:
        run_dir = Path(args.rescore or args.resume)
        cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
        cfg["envs"] = load_envs(args.env_file, cfg.get("env_passthrough", []))  # secrets were redacted in the snapshot
        mapped = {a: v for a, v in vars(args).items() if a in ARG_TO_KEY and v is not None}
        if args.resume:
            if args.overrides or mapped:  # a resumed run must keep the settings recorded in its config.yaml
                raise SystemExit("--resume takes no --set / config flags (they would not be recorded); "
                                 "use --rescore for metric settings or start a new run")
            asyncio.run(_run(cfg, run_dir, resume=True, force=args.force))
            return
        for a, v in mapped.items():
            set_by_path(cfg, ARG_TO_KEY[a], v)
        for k, v in parse_overrides(args.overrides):
            set_by_path(cfg, k, v)
        rows = load_latest(run_dir / "predictions.jsonl")  # transport failures stay in as errors
        manifest = json.loads((run_dir / "data_manifest.json").read_text(encoding="utf-8"))
        desc_path = next((p for p in (run_dir / "task.json", run_dir / "target.json") if p.exists()), None)
        desc = json.loads(desc_path.read_text(encoding="utf-8")) if desc_path else {}
        task = build_task(cfg)  # constructors never touch the network; only startup() does
        target = getattr(task, "target", None)
        if target is not None and hasattr(target, "server_info"):  # what the run saw (stages, trained_on)
            target.server_info = desc.get("server_info") or {}
        settings = {k: cfg.get(k) for k in ("metrics", "eval")}
        th = get_by_path(cfg, "metrics.threshold")
        sub = run_dir / f"rescore_{'default' if th is None else th}_{config_hash(settings)[:8]}"
        sub.mkdir(exist_ok=True)
        dump_config({**cfg, "meta": {**cfg.get("meta", {}), "rescore_of": str(run_dir),
                                     "rescore_overrides": args.overrides, "rescore_args": mapped}}, sub / "config.yaml")
        metrics = compute_metrics(task, rows)
        if _target_warnings(desc):
            metrics.setdefault("warnings", []).extend(_target_warnings(desc))
        write_report(sub, metrics, cfg, manifest, desc, rows)
        log.info("rescored -> %s", sub)
        return

    cfg = load_config(args.config, vars(args), ARG_TO_KEY, args.overrides, args.env_file, run_kind="evaluation")
    run_dir = make_run_dir(cfg)
    dump_config(cfg, run_dir / "config.yaml")
    _write_json(run_dir / "environment.json", _environment())
    asyncio.run(_run(cfg, run_dir))


if __name__ == "__main__":
    main()
