"""Leaderboard across runs.

    python -m src.evaluation.compare results/*/ --sort recall@fpr0.05 -o results/leaderboard.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

COLUMNS = [
    ("run", None),
    ("n", "overall.n"),
    ("precision", "overall.precision"),
    ("recall", "overall.recall"),
    ("f1", "overall.f1"),
    ("fpr", "overall.fpr"),
    ("auroc", "score_metrics.auroc"),
    ("recall@fpr0.01", "score_metrics.recall_at_fpr.0.01.recall"),
    ("recall@fpr0.05", "score_metrics.recall_at_fpr.0.05.recall"),
    ("p50_ms", "latency_ms.p50"),
    ("p95_ms", "latency_ms.p95"),
    ("err_rate", "error_rate"),
]


def _get(d: dict, dotted: str):
    # keys like "0.05" contain dots; walk greedily
    parts = dotted.split(".")
    cur = d
    i = 0
    while i < len(parts):
        for j in range(len(parts), i, -1):
            k = ".".join(parts[i:j])
            if isinstance(cur, dict) and k in cur:
                cur = cur[k]
                i = j
                break
        else:
            return None
    return cur


LOWER_IS_BETTER = {"fpr", "p50_ms", "p95_ms", "err_rate"}


def load_run(run_dir: Path) -> dict | None:
    m = run_dir / "metrics.json"
    if not m.exists():
        return None
    metrics = json.loads(m.read_text(encoding="utf-8"))
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8")) if (run_dir / "config.yaml").exists() else {}
    row = {"run": run_dir.name, "_hash": (cfg.get("meta") or {}).get("config_hash")}
    for name, key in COLUMNS[1:]:
        row[name] = _get(metrics, key)
    return row


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    p.add_argument("--sort", default="f1", choices=[c for c, _ in COLUMNS[1:]])
    p.add_argument("-o", "--output", default=None)
    a = p.parse_args(argv)
    rows = [r for r in (load_run(Path(x)) for x in a.runs) if r]
    asc = a.sort in LOWER_IS_BETTER  # best first either way; runs without the metric go last
    rows.sort(key=lambda r: (r.get(a.sort) is None, (r.get(a.sort) or 0) if asc else -(r.get(a.sort) or 0)))
    hdr = [c for c, _ in COLUMNS]
    lines = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for r in rows:
        lines.append("| " + " | ".join(f"{r[c]:.3f}" if isinstance(r[c], float) else str(r[c] if r[c] is not None else "-") for c in hdr) + " |")
    md = "\n".join(lines) + "\n"
    print(md)
    if a.output:
        Path(a.output).write_text(md, encoding="utf-8")


if __name__ == "__main__":
    main()
