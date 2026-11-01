"""Write run artifacts: metrics.json, slices.csv, errors (FP/FN) and a human-readable report.md."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def _fmt(v: Any) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def write_metrics(run_dir: Path, metrics: dict[str, Any]) -> None:
    (run_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")


def write_slices_csv(run_dir: Path, metrics: dict[str, Any]) -> None:
    cols = ["slice", "value", "n", "n_unsafe", "n_safe", "precision", "recall", "f1", "fpr"]
    with (run_dir / "slices.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for key, groups in metrics.get("slices", {}).items():
            for val, m in groups.items():
                w.writerow({"slice": key, "value": val, **m})


def write_errors(run_dir: Path, rows: list[dict], n: int) -> None:
    """False positives / negatives for manual review — the most useful file in the run directory."""
    cols = ["kind", "sample_id", "source", "language", "input_type", "gold_categories", "categories",
            "score", "blocked_by", "text_preview", "raw", "error"]
    fps = [r for r in rows if r.get("unsafe") is True and r["gold_label"] == "safe"]
    fns = [r for r in rows if r.get("unsafe") is False and r["gold_label"] == "unsafe"]
    errs = [r for r in rows if r.get("unsafe") is None]
    # most confident mistakes first
    fps.sort(key=lambda r: -(r.get("score") or 0))
    fns.sort(key=lambda r: (r.get("score") or 0))
    with (run_dir / "errors.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for kind, group in (("false_positive", fps), ("false_negative", fns), ("error", errs)):
            for r in group[:n] if n else group:
                w.writerow({**r, "kind": kind, "gold_categories": "|".join(r.get("gold_categories") or []),
                            "categories": "|".join(r.get("categories") or []),
                            "blocked_by": "|".join(r.get("blocked_by") or [])})


def render_markdown(metrics: dict[str, Any], cfg: dict[str, Any], manifest: dict[str, Any], target_desc: dict) -> str:
    o = metrics["overall"]
    ci = metrics.get("ci", {})
    lat = metrics.get("latency_ms", {})

    def with_ci(k: str) -> str:
        lo, hi = (ci.get(k) or [None, None])
        return f"{_fmt(o.get(k))} [{_fmt(lo)}, {_fmt(hi)}]" if lo is not None else _fmt(o.get(k))

    meta = cfg.get("meta", {})
    lines = [
        f"# Guardrail eval — {cfg.get('run', {}).get('name', 'run')}",
        "",
        f"- target: `{target_desc.get('type')}`  stages: {', '.join(s.get('name', '?') for s in target_desc.get('stages', [])) or '-'}",
        f"- config_hash: `{meta.get('config_hash', '-')}`  git: `{(meta.get('git') or {}).get('commit', '-')}`"
        f"{' (dirty)' if (meta.get('git') or {}).get('dirty') else ''}",
        f"- samples: {metrics['n_scored']}/{metrics['n_total']} scored  "
        f"(error_rate {_fmt(metrics['error_rate'])}, parse_error_rate {_fmt(metrics['parse_error_rate'])}, policy `{metrics['error_policy']}`)",
        f"- labels: {manifest.get('label_counts')}  languages: {manifest.get('language_counts')}",
        "",
        "## Overall",
        "",
        "| metric | value (95% CI) |",
        "|---|---|",
        f"| precision | {with_ci('precision')} |",
        f"| recall (block rate on unsafe) | {with_ci('recall')} |",
        f"| F1 | {with_ci('f1')} |",
        f"| FPR (false block rate on safe) | {with_ci('fpr')} |",
        f"| balanced accuracy | {_fmt(o.get('balanced_accuracy'))} |",
        f"| TP / FP / FN / TN | {o['tp']} / {o['fp']} / {o['fn']} / {o['tn']} |",
        f"| latency p50 / p95 / p99 (ms) | {_fmt(lat.get('p50'))} / {_fmt(lat.get('p95'))} / {_fmt(lat.get('p99'))} |",
    ]
    sm = metrics.get("score_metrics")
    if sm and "auroc" in sm:
        lines += [
            f"| AUROC / AUPRC | {_fmt(sm['auroc'])} / {_fmt(sm['auprc'])} |",
            f"| best-F1 threshold | {_fmt(sm['best_f1']['threshold'])} (F1 {_fmt(sm['best_f1']['f1'])}) |",
        ]
        for k, v in sm["recall_at_fpr"].items():
            lines.append(f"| recall @ FPR≤{k} | {_fmt(v['recall'])} (th {_fmt(v['threshold'])}) |")
    if metrics.get("contamination"):
        c = metrics["contamination"]
        lines += ["", "## Contamination", "",
                  f"Model(s) trained on `{', '.join(c['patterns'])}` -> excluded sources: "
                  f"{', '.join(c['excluded_sources']) or 'none present'}"]
        if metrics.get("clean"):
            co = metrics["clean"]["overall"]
            lines += ["", "| clean subset | n | precision | recall | F1 | FPR |", "|---|---|---|---|---|---|",
                      f"| without contaminated sources | {co['n']} | {_fmt(co['precision'])} | {_fmt(co['recall'])} | "
                      f"{_fmt(co['f1'])} | {_fmt(co['fpr'])} |"]
    if metrics.get("per_stage"):
        lines += ["", "## Per stage (each stage alone)", "", "| stage | precision | recall | F1 | FPR | AUROC |", "|---|---|---|---|---|---|"]
        for st, m in metrics["per_stage"].items():
            lines.append(f"| {st} | {_fmt(m['precision'])} | {_fmt(m['recall'])} | {_fmt(m['f1'])} | {_fmt(m['fpr'])} | {_fmt(m.get('auroc'))} |")
    for key, groups in metrics.get("slices", {}).items():
        if not groups:
            continue
        lines += ["", f"## By {key}", ""]
        if key == "gold_category":
            lines += ["| category | n | recall |", "|---|---|---|"]
            lines += [f"| {g} | {m['n']} | {_fmt(m['recall'])} |" for g, m in groups.items()]
        else:
            lines += ["| value | n | unsafe/safe | precision | recall | F1 | FPR |", "|---|---|---|---|---|---|---|"]
            lines += [f"| {g} | {m['n']} | {m['n_unsafe']}/{m['n_safe']} | {_fmt(m['precision'])} | {_fmt(m['recall'])} | "
                      f"{_fmt(m['f1'])} | {_fmt(m['fpr'])} |" for g, m in groups.items()]
    lines += ["", "## Data", ""] + [f"- `{f['path']}` sha256 `{f['sha256'][:12]}` used {f['rows_used']}/{f['rows_in_file']}"
                                    for f in manifest.get("files", [])]
    return "\n".join(lines) + "\n"


def write_report(run_dir: Path, metrics: dict, cfg: dict, manifest: dict, target_desc: dict, rows: list[dict]) -> None:
    write_metrics(run_dir, metrics)
    write_slices_csv(run_dir, metrics)
    write_errors(run_dir, rows, int(cfg.get("report", {}).get("n_error_examples", 100)))
    (run_dir / "report.md").write_text(render_markdown(metrics, cfg, manifest, target_desc), encoding="utf-8")
