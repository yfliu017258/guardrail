"""Metrics without sklearn: confusion stats, bootstrap CIs, AUROC/AUPRC, threshold sweeps, slices.

Convention: positive class = unsafe (the guard should block it).

* recall (= block rate on unsafe) — "how much bad traffic do we stop"
* FPR (= false block rate on safe) — "how often do we annoy legitimate users"; for an
  enterprise guard this usually matters more than raw F1, so we report recall@FPR.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any, Callable, Iterable

Row = dict[str, Any]


# --------------------------------------------------------------------------- #
# basic
# --------------------------------------------------------------------------- #
def confusion(y_true: list[bool], y_pred: list[bool]) -> dict[str, int]:
    tp = sum(t and p for t, p in zip(y_true, y_pred))
    fp = sum((not t) and p for t, p in zip(y_true, y_pred))
    fn = sum(t and (not p) for t, p in zip(y_true, y_pred))
    tn = sum((not t) and (not p) for t, p in zip(y_true, y_pred))
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def _div(a: float, b: float) -> float | None:
    return a / b if b else None


def binary_metrics(y_true: list[bool], y_pred: list[bool]) -> dict[str, Any]:
    c = confusion(y_true, y_pred)
    tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]
    precision = _div(tp, tp + fp)
    recall = _div(tp, tp + fn)
    f1 = _div(2 * tp, 2 * tp + fp + fn)
    fpr = _div(fp, fp + tn)
    tnr = _div(tn, fp + tn)
    mcc_den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {
        "n": len(y_true), "n_unsafe": tp + fn, "n_safe": tn + fp, **c,
        "precision": precision, "recall": recall, "f1": f1, "fpr": fpr,
        "fnr": _div(fn, tp + fn), "accuracy": _div(tp + tn, len(y_true)),
        "balanced_accuracy": (recall + tnr) / 2 if recall is not None and tnr is not None else None,
        "mcc": (tp * tn - fp * fn) / mcc_den if mcc_den else None,
    }


# --------------------------------------------------------------------------- #
# score-based
# --------------------------------------------------------------------------- #
def auroc(y_true: list[bool], scores: list[float]) -> float | None:
    """Mann-Whitney U with average ranks for ties."""
    pos = sum(y_true)
    neg = len(y_true) - pos
    if not pos or not neg:
        return None
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    r_pos = sum(r for r, t in zip(ranks, y_true) if t)
    return (r_pos - pos * (pos + 1) / 2) / (pos * neg)


def average_precision(y_true: list[bool], scores: list[float]) -> float | None:
    pos = sum(y_true)
    if not pos:
        return None
    pairs = sorted(zip(scores, y_true), key=lambda x: -x[0])
    tp = fp = 0
    ap = 0.0
    i = 0
    while i < len(pairs):  # group ties so the result doesn't depend on input order
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        grp_tp = sum(t for _, t in pairs[i:j])
        tp += grp_tp
        fp += (j - i) - grp_tp
        if grp_tp:
            ap += grp_tp / pos * tp / (tp + fp)
        i = j
    return ap


def threshold_sweep(y_true: list[bool], scores: list[float], fpr_targets: Iterable[float]) -> dict[str, Any]:
    """Best-F1 threshold and the max recall achievable at each FPR budget."""
    thresholds = sorted(set(scores), reverse=True)
    pos = sum(y_true)
    neg = len(y_true) - pos
    best = {"threshold": None, "f1": -1.0, "precision": None, "recall": None, "fpr": None}
    at_fpr: dict[str, Any] = {f"{t:g}": {"recall": 0.0, "threshold": None, "fpr": 0.0} for t in fpr_targets}
    pairs = sorted(zip(scores, y_true), key=lambda x: -x[0])
    tp = fp = 0
    idx = 0
    for th in thresholds:
        while idx < len(pairs) and pairs[idx][0] >= th:
            tp += pairs[idx][1]
            fp += not pairs[idx][1]
            idx += 1
        f1 = 2 * tp / (2 * tp + fp + (pos - tp)) if (tp or fp or pos) else 0.0
        rec = tp / pos if pos else None
        fpr = fp / neg if neg else None
        if f1 > best["f1"]:
            best = {"threshold": th, "f1": f1, "precision": tp / (tp + fp) if tp + fp else None, "recall": rec, "fpr": fpr}
        for t in fpr_targets:
            k = f"{t:g}"
            if fpr is not None and rec is not None and fpr <= t and rec > at_fpr[k]["recall"]:
                at_fpr[k] = {"recall": rec, "threshold": th, "fpr": fpr}
    return {"best_f1": best, "recall_at_fpr": at_fpr}


# --------------------------------------------------------------------------- #
# bootstrap
# --------------------------------------------------------------------------- #
def bootstrap_ci(y_true: list[bool], y_pred: list[bool], n_boot: int, ci: float, seed: int,
                 keys: tuple[str, ...] = ("precision", "recall", "f1", "fpr")) -> dict[str, list[float | None]]:
    if n_boot <= 0 or not y_true:
        return {}
    rng = random.Random(seed)
    n = len(y_true)
    samples: dict[str, list[float]] = defaultdict(list)
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        m = binary_metrics([y_true[i] for i in idx], [y_pred[i] for i in idx])
        for k in keys:
            if m[k] is not None:
                samples[k].append(m[k])
    lo_q, hi_q = (1 - ci) / 2, 1 - (1 - ci) / 2
    out = {}
    for k in keys:
        v = sorted(samples[k])
        out[k] = [v[int(lo_q * (len(v) - 1))], v[int(hi_q * (len(v) - 1))]] if v else [None, None]
    return out


# --------------------------------------------------------------------------- #
# latency
# --------------------------------------------------------------------------- #
def percentiles(values: list[float], qs: Iterable[float] = (50, 90, 95, 99)) -> dict[str, float | None]:
    v = sorted(x for x in values if x is not None)
    if not v:
        return {f"p{q:g}": None for q in qs} | {"mean": None}
    out = {}
    for q in qs:
        k = (len(v) - 1) * q / 100
        f, c = math.floor(k), math.ceil(k)
        out[f"p{q:g}"] = v[f] + (v[c] - v[f]) * (k - f)
    out["mean"] = sum(v) / len(v)
    return out


# --------------------------------------------------------------------------- #
# full report
# --------------------------------------------------------------------------- #
def _decide(r: Row, threshold: float | None, score_key: str = "score") -> bool:
    if threshold is not None and r.get(score_key) is not None:
        return r[score_key] >= threshold
    return bool(r["unsafe"])


def evaluate_rows(
    rows: list[Row],
    threshold: float | None = None,
    error_policy: str = "exclude",
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
    fpr_targets: Iterable[float] = (0.01, 0.05),
    slice_by: Iterable[str] = ("source", "language", "input_type"),
    min_slice_size: int = 1,
) -> dict[str, Any]:
    fpr_targets = list(fpr_targets)
    n_total = len(rows)
    errored = [r for r in rows if r.get("unsafe") is None]
    if error_policy == "exclude":
        scored = [r for r in rows if r.get("unsafe") is not None]
    else:  # as_safe (fail-open) / as_unsafe (fail-closed): mirrors the production on_error choice
        fill = error_policy == "as_unsafe"
        scored = [{**r, "unsafe": fill if r.get("unsafe") is None else r["unsafe"]} for r in rows]

    y_true = [r["gold_label"] == "unsafe" for r in scored]
    y_pred = [_decide(r, threshold) for r in scored]
    out: dict[str, Any] = {
        "n_total": n_total,
        "n_scored": len(scored),
        "error_rate": len(errored) / n_total if n_total else None,
        "parse_error_rate": (sum(not r.get("parse_ok", True) for r in rows) / n_total) if n_total else None,
        "threshold": threshold,
        "error_policy": error_policy,
        "overall": binary_metrics(y_true, y_pred),
        "ci": bootstrap_ci(y_true, y_pred, n_boot, ci, seed),
        "latency_ms": percentiles([r.get("latency_ms") for r in rows]),
        "server_latency_ms": percentiles([r.get("server_latency_ms") for r in rows if r.get("server_latency_ms") is not None]),
    }

    with_score = [(t, r["score"]) for t, r in zip(y_true, scored) if r.get("score") is not None]
    if with_score and len(with_score) == len(scored):
        yt, sc = [t for t, _ in with_score], [s for _, s in with_score]
        out["score_metrics"] = {"auroc": auroc(yt, sc), "auprc": average_precision(yt, sc),
                                **threshold_sweep(yt, sc, fpr_targets)}
    elif with_score:
        out["score_metrics"] = {"note": f"only {len(with_score)}/{len(scored)} rows have scores; skipped"}

    # per-stage (each stage as if it were the only guard) — key for 2-stage design decisions
    stages = sorted({k for r in scored for k in (r.get("stage_unsafe") or {})})
    per_stage = {}
    for st in stages:
        sub = [r for r in scored if st in (r.get("stage_unsafe") or {})]
        yt = [r["gold_label"] == "unsafe" for r in sub]
        yp = [bool(r["stage_unsafe"][st]) for r in sub]
        entry = binary_metrics(yt, yp)
        sc = [(r.get("stage_scores") or {}).get(st) for r in sub]
        if sub and all(s is not None for s in sc):
            entry["auroc"] = auroc(yt, sc)  # type: ignore[arg-type]
            entry["auprc"] = average_precision(yt, sc)  # type: ignore[arg-type]
        per_stage[st] = entry
    if per_stage:
        out["per_stage"] = per_stage

    # slices
    slices: dict[str, dict[str, Any]] = {}
    for key in slice_by:
        groups: dict[str, list[int]] = defaultdict(list)
        for i, r in enumerate(scored):
            groups[str(r.get(key))].append(i)
        slices[key] = {
            g: binary_metrics([y_true[i] for i in idx], [y_pred[i] for i in idx])
            for g, idx in sorted(groups.items()) if len(idx) >= min_slice_size
        }
    # recall per gold category (a sample with 2 categories counts in both)
    cat_groups: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(scored):
        for c in r.get("gold_categories") or []:
            cat_groups[c].append(i)
    slices["gold_category"] = {
        c: {"n": len(idx), "recall": _div(sum(y_pred[i] for i in idx), len(idx))}
        for c, idx in sorted(cat_groups.items()) if len(idx) >= min_slice_size
    }
    out["slices"] = slices

    # category agreement on true positives where gold categories are known
    tp_rows = [r for r, t, p in zip(scored, y_true, y_pred) if t and p and r.get("gold_categories")]
    if tp_rows:
        hit = sum(bool(set(r["gold_categories"]) & set(r.get("categories") or [])) for r in tp_rows)
        out["category_match_rate_on_tp"] = hit / len(tp_rows)
    return out


def flatten(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(flatten(v, key))
        else:
            out[key] = v
    return out


def select(rows: list[Row], pred: Callable[[Row], bool]) -> list[Row]:
    return [r for r in rows if pred(r)]
