"""Write run artifacts: metrics.json, slices.csv, errors (FP/FN) and a human-readable report.md."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .metrics import final_decisions, gold_of, pred_of


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


def write_errors(run_dir: Path, rows: list[dict], n: int, cfg: dict | None = None) -> None:
    """False positives / negatives for manual review — the most useful file in the run directory."""
    cols = ["kind", "sample_id", "source", "language", "input_type", "gold_label", "gold_categories",
            "gold_refusal_label", "refusal_label", "categories", "score", "blocked_by", "judge", "judge_reason",
            "text_preview", "response", "raw", "error"]
    m = (cfg or {}).get("metrics", {}) or {}
    decided = final_decisions(rows, m.get("threshold"), "exclude", m.get("score_stage"))  # same rule as the metrics
    fps = [r for r in rows if decided.get(r.get("sample_id")) is True and not gold_of(r)]
    fns = [r for r in rows if decided.get(r.get("sample_id")) is False and gold_of(r)]
    errs = [r for r in rows if pred_of(r) is None]
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


DEFAULT_LABELS = {"precision": "precision", "recall": "recall", "fpr": "FPR"}


def _header(desc: dict) -> str:
    task = desc.get("task", "guardrail")
    if task == "model_safety":
        g, j = desc.get("generator") or {}, desc.get("judge") or {}
        return f"- task: `model_safety`  model: `{g.get('model')}` (gen fingerprint `{g.get('fingerprint')}`)  judge: `{j.get('type')}`"
    if task == "refusal_judge":
        return f"- task: `refusal_judge`  judge: `{(desc.get('judge') or {}).get('type', desc.get('judge_config', {}).get('type'))}`"
    stages = ", ".join(s.get("name", "?") for s in desc.get("stages", [])) or "-"
    return f"- task: `guardrail`  target: `{desc.get('type')}`  stages: {stages}"


def render_markdown(metrics: dict[str, Any], cfg: dict[str, Any], manifest: dict[str, Any], target_desc: dict) -> str:
    o = metrics["overall"]
    labels = {**DEFAULT_LABELS, **(metrics.get("metric_labels") or {})}
    ci = metrics.get("ci", {})
    lat = metrics.get("latency_ms", {})

    def with_ci(k: str) -> str:
        lo, hi = (ci.get(k) or [None, None])
        return f"{_fmt(o.get(k))} [{_fmt(lo)}, {_fmt(hi)}]" if lo is not None else _fmt(o.get(k))

    meta = cfg.get("meta", {})
    lines = [
        f"# Eval [{metrics.get('task', 'guardrail')}] — {cfg.get('run', {}).get('name', 'run')}",
        "",
        _header(target_desc),
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
        f"| {labels['precision']} | {with_ci('precision')} |",
        f"| {labels['recall']} | {with_ci('recall')} |",
        f"| F1 | {with_ci('f1')} |",
        f"| {labels['fpr']} | {with_ci('fpr')} |",
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
    ms = metrics.get("model_safety")
    if ms:
        lines += ["", "## Model safety", "", "| metric | value |", "|---|---|"]
        for k in ("n_unsafe_prompts", "n_safe_prompts", "refusal_rate_unsafe", "attack_success_rate", "partial_rate_unsafe",
                  "over_refusal_rate_safe", "harmful_response_rate_unsafe", "harmful_given_not_refused", "blocked_rate",
                  "empty_response_rate", "truncated_rate"):
            if k in ms:
                lines.append(f"| {k} | {_fmt(ms[k])} |")
        lines += ["", f"label distribution: `{json.dumps(ms.get('label_distribution'), ensure_ascii=False)}`"]
        if ms.get("refusal_rate_by_category"):
            lines += ["", "| unsafe category | n | refusal rate |", "|---|---|---|"]
            lines += [f"| {c} | {v['n']} | {_fmt(v['refusal_rate'])} |" for c, v in ms["refusal_rate_by_category"].items()]
    if metrics.get("label_confusion"):
        conf = metrics["label_confusion"]
        cols = list(next(iter(conf.values())).keys())
        lines += ["", f"## Judge label confusion (rows = gold, cols = judged; kappa {_fmt(metrics.get('label_kappa'))})", "",
                  "| gold \\ judged | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
        lines += [f"| {g} | " + " | ".join(str(row[c]) for c in cols) + " |" for g, row in conf.items()]
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
        full = all((m.get("coverage") or 0) >= 0.999 for m in metrics["per_stage"].values())
        title = "each stage alone" if full else "CONDITIONAL on reaching the stage (short-circuit) — use run_all for standalone numbers"
        lines += ["", f"## Per stage ({title})", "", "| stage | n | coverage | precision | recall | F1 | FPR | AUROC |",
                  "|---|---|---|---|---|---|---|---|"]
        for st, m in metrics["per_stage"].items():
            lines.append(f"| {st} | {m['n']} | {_fmt(m.get('coverage'))} | {_fmt(m['precision'])} | {_fmt(m['recall'])} | "
                         f"{_fmt(m['f1'])} | {_fmt(m['fpr'])} | {_fmt(m.get('auroc'))} |")
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
    write_errors(run_dir, rows, int(cfg.get("report", {}).get("n_error_examples", 100)), cfg)
    (run_dir / "report.md").write_text(render_markdown(metrics, cfg, manifest, target_desc), encoding="utf-8")
