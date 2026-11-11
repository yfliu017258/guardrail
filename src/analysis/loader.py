"""Load evaluation runs and derive what the app shows. Pure pandas, no Streamlit — unit-testable.

Vocabulary: every run has a binary decision per sample, ``pred_positive`` — a guardrail run's "blocked",
a model run's "refused" — and a ground truth ``gold_positive`` ("this should be stopped").
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

SUPPORTED_SCHEMA = 1
OUTCOMES = ("TP", "FN", "FP", "TN", "error")
OUTCOME_LABELS = {
    "guardrail": {"TP": "正確擋下", "FN": "漏擋", "FP": "誤擋", "TN": "正確放行", "error": "錯誤"},
    "model_safety": {"TP": "正確拒絕", "FN": "未拒絕（有害請求）", "FP": "過度拒絕", "TN": "正常回答", "error": "錯誤"},
}
#: who handled a request when a guardrail sits in front of a model
HANDLERS = ("guardrail", "model", "answered")
HANDLER_LABELS = {"guardrail": "Guardrail 擋下", "model": "模型拒絕", "answered": "模型回答"}


# --------------------------------------------------------------------------- files
def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def read_predictions(run_dir: str | Path) -> pd.DataFrame:
    """Latest row per sample (a resumed run appends retries); tolerates a truncated last line."""
    rows: dict[str, dict] = {}
    path = Path(run_dir) / "predictions.jsonl"
    if path.exists():
        with path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                rows[rec["sample_id"]] = rec
    df = pd.DataFrame(list(rows.values()))
    if df.empty:
        return df
    for col in ("prompt", "response", "action", "refusal_label", "error", "source", "language", "expected"):
        if col not in df:
            df[col] = None
    df["prompt"] = df["prompt"].fillna(df.get("text_preview"))
    df["outcome"] = [outcome(g, p) for g, p in zip(df["gold_positive"], df["pred_positive"])]
    return df


def outcome(gold: Any, pred: Any) -> str:
    if pred is None or (isinstance(pred, float) and math.isnan(pred)) or gold is None:
        return "error"
    return ("TP" if pred else "FN") if gold else ("FP" if pred else "TN")


def latency_column(task: str) -> str:
    """Model runs: the generation time alone (``latency_ms`` there also includes the judge)."""
    return "gen_latency_ms" if task == "model_safety" else "latency_ms"


# --------------------------------------------------------------------------- statistics
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float | None, float | None]:
    """95% Wilson interval for a proportion — honest on the small samples of a quick check."""
    if not n:
        return None, None
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, mid - half), min(1.0, mid + half)


def _rate(k: int, n: int) -> float | None:
    return k / n if n else None


def latency_stats(values: Iterable[float]) -> dict[str, float | None]:
    s = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna()
    if s.empty:
        return {k: None for k in ("p50", "p90", "p95", "p99", "mean", "max")}
    return {"p50": float(s.quantile(0.5)), "p90": float(s.quantile(0.9)), "p95": float(s.quantile(0.95)),
            "p99": float(s.quantile(0.99)), "mean": float(s.mean()), "max": float(s.max())}


def summarize(df: pd.DataFrame, task: str = "guardrail") -> dict[str, Any]:
    """Headline numbers of one run (or of any filtered subset of its rows)."""
    if df.empty:
        return {"n": 0}
    c = df["outcome"].value_counts()
    tp, fn, fp, tn, err = (int(c.get(k, 0)) for k in OUTCOMES)
    prec, rec = _rate(tp, tp + fp), _rate(tp, tp + fn)
    lat = latency_stats(df[latency_column(task)]) if latency_column(task) in df else latency_stats([])
    return {
        "n": len(df), "n_unsafe": tp + fn, "n_safe": fp + tn, "tp": tp, "fn": fn, "fp": fp, "tn": tn, "errors": err,
        "recall": rec, "recall_ci": wilson(tp, tp + fn), "fpr": _rate(fp, fp + tn), "fpr_ci": wilson(fp, fp + tn),
        "precision": prec, "f1": 2 * prec * rec / (prec + rec) if prec and rec else (0.0 if prec is not None and rec is not None else None),
        "error_rate": _rate(err, len(df)), **{f"latency_{k}": v for k, v in lat.items()},
    }


def slice_table(df: pd.DataFrame, by: str, task: str = "guardrail") -> pd.DataFrame:
    out = []
    for value, grp in df.groupby(by, dropna=False):
        s = summarize(grp, task)
        out.append({by: value, "n": s["n"], "該擋": s["n_unsafe"], "不該擋": s["n_safe"], "recall": s["recall"],
                    "fpr": s["fpr"], "errors": s["errors"], "p50 ms": s["latency_p50"], "p95 ms": s["latency_p95"]})
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- runs
class Run:
    """One results directory. ``rows`` are loaded lazily."""

    def __init__(self, path: Path):
        self.path = path
        self.info = _read_json(path / "run.json")
        self.metrics = _read_json(path / "metrics.json")
        self._rows: pd.DataFrame | None = None

    # identity ------------------------------------------------------------
    @property
    def run_id(self) -> str:
        return self.info.get("run_id") or self.path.name

    @property
    def task(self) -> str:
        return self.info.get("task") or self.metrics.get("task") or "guardrail"

    @property
    def subject(self) -> dict[str, Any]:
        return self.info.get("subject") or {}

    @property
    def name(self) -> str:
        return str(self.info.get("name") or self.subject.get("name") or self.path.name)

    @property
    def label(self) -> str:
        """Unique, readable: the run name plus its start time."""
        return f"{self.name} · {str(self.info.get('started_at') or self.path.name[:15])[5:16].replace('T', ' ')}"

    @property
    def input_type(self) -> str:
        return self.subject.get("input_type") or "request"

    @property
    def dataset_id(self) -> str | None:
        return (self.info.get("dataset") or {}).get("sha256")

    @property
    def supported(self) -> bool:
        return self.info.get("schema_version") == SUPPORTED_SCHEMA

    @property
    def rows(self) -> pd.DataFrame:
        if self._rows is None:
            self._rows = read_predictions(self.path)
        return self._rows

    def summary(self) -> dict[str, Any]:
        s = summarize(self.rows, self.task)
        ds = self.info.get("dataset") or {}
        return {"run": self.label, "run_id": self.run_id, "名稱": self.name,
                "類型": {"guardrail": "Guardrail", "model_safety": "模型"}.get(self.task, self.task),
                "方向": self.input_type if self.task == "guardrail" else "-",
                "狀態": self.info.get("status", "?"), "資料集": ds.get("name"), "n": s.get("n", 0),
                "recall": s.get("recall"), "FPR": s.get("fpr"), "precision": s.get("precision"), "F1": s.get("f1"),
                "錯誤率": s.get("error_rate"), "p50 ms": s.get("latency_p50"), "p95 ms": s.get("latency_p95"),
                "開始時間": self.info.get("started_at"), "dataset_id": self.dataset_id}


def list_runs(results_dir: str | Path) -> list[Run]:
    """Every directory with a ``run.json`` (newest first). Rescore sub-directories are not runs."""
    root = Path(results_dir)
    if not root.is_dir():
        return []
    runs = [Run(p) for p in root.iterdir() if p.is_dir() and (p / "run.json").exists()]
    return sorted(runs, key=lambda r: str(r.info.get("started_at") or r.path.name), reverse=True)


def results_fingerprint(results_dir: str | Path) -> float:
    """Changes whenever a run is added or updated — the app's cache key."""
    root = Path(results_dir)
    if not root.is_dir():
        return 0.0
    return sum(f.stat().st_mtime for p in root.iterdir() if p.is_dir()
               for f in (p / "run.json", p / "predictions.jsonl") if f.exists())


# --------------------------------------------------------------------------- comparisons
def align(runs: list[Run]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Decisions of several runs side by side on the samples ALL of them scored.

    Columns: sample fields, then ``<label>`` (decision) and ``<label>__ms`` (latency) per run."""
    if not runs:
        return pd.DataFrame(), {"common": 0}
    base = None
    for r in runs:
        d = r.rows
        if d.empty:
            return pd.DataFrame(), {"common": 0}
        d = d[d["outcome"] != "error"]
        part = d[["sample_id", "pred_positive", latency_column(r.task)]].rename(
            columns={"pred_positive": r.label, latency_column(r.task): f"{r.label}__ms"})
        if base is None:
            meta = [c for c in ("sample_id", "prompt", "response", "expected", "source", "language", "gold_positive")
                    if c in d and not (c == "response" and r.task == "model_safety")]
            base = d[meta].merge(part, on="sample_id")
        else:
            base = base.merge(part, on="sample_id")
    info = {"common": len(base), "per_run": {r.label: int((r.rows["outcome"] != "error").sum()) for r in runs},
            "same_dataset": len({r.dataset_id for r in runs}) == 1}
    for r in runs:
        base[r.label] = base[r.label].astype(bool)
    return base, info


def disagreements(aligned: pd.DataFrame, a: str, b: str) -> dict[str, pd.DataFrame]:
    """Where two runs decide differently, split by who is right."""
    gold = aligned["gold_positive"].astype(bool)
    return {
        f"只有「{a}」擋下（該擋）": aligned[gold & aligned[a] & ~aligned[b]],
        f"只有「{b}」擋下（該擋）": aligned[gold & ~aligned[a] & aligned[b]],
        f"只有「{a}」誤擋（不該擋）": aligned[~gold & aligned[a] & ~aligned[b]],
        f"只有「{b}」誤擋（不該擋）": aligned[~gold & ~aligned[a] & aligned[b]],
        "兩者都漏擋": aligned[gold & ~aligned[a] & ~aligned[b]],
    }


def combine(model: Run, guard: Run | None) -> pd.DataFrame:
    """A request-direction guardrail in front of a model, replayed offline from the two runs.

    Per sample: ``handler`` = who dealt with the request (guardrail blocked it -> the model never saw it;
    otherwise the model refused or answered) and ``latency_ms`` = what the caller waits (the guardrail
    check, plus the generation unless the request was blocked). ``guard=None`` is the bare model."""
    m = model.rows
    m = m[m["outcome"] != "error"][["sample_id", "prompt", "response", "expected", "source", "language",
                                    "gold_positive", "pred_positive", "gen_latency_ms"]].rename(
        columns={"pred_positive": "model_refused", "gen_latency_ms": "model_ms"})
    if guard is None:
        df = m.assign(guard_blocked=False, guard_ms=0.0)
    else:
        g = guard.rows
        g = g[g["outcome"] != "error"][["sample_id", "pred_positive", "latency_ms", "action"]].rename(
            columns={"pred_positive": "guard_blocked", "latency_ms": "guard_ms", "action": "guard_action"})
        df = m.merge(g, on="sample_id")
    if df.empty:
        return df
    df["guard_blocked"] = df["guard_blocked"].astype(bool)
    df["model_refused"] = df["model_refused"].astype(bool)
    df["handler"] = ["guardrail" if gb else ("model" if mr else "answered")
                     for gb, mr in zip(df["guard_blocked"], df["model_refused"])]
    df["latency_ms"] = df["guard_ms"].fillna(0) + df["model_ms"].fillna(0).where(~df["guard_blocked"], 0.0)
    return df


def combo_summary(df: pd.DataFrame) -> dict[str, Any]:
    """Benefit and cost of the combination: residual risk on harmful prompts, service on benign ones, latency."""
    if df.empty:
        return {"n": 0}
    unsafe, safe = df[df["gold_positive"].astype(bool)], df[~df["gold_positive"].astype(bool)]
    cnt_u = unsafe["handler"].value_counts()
    cnt_s = safe["handler"].value_counts()
    nu, ns = len(unsafe), len(safe)
    residual, served = int(cnt_u.get("answered", 0)), int(cnt_s.get("answered", 0))
    lat = latency_stats(df["latency_ms"])
    return {
        "n": len(df), "n_unsafe": nu, "n_safe": ns,
        "unsafe": {h: int(cnt_u.get(h, 0)) for h in HANDLERS}, "safe": {h: int(cnt_s.get(h, 0)) for h in HANDLERS},
        "residual_rate": _rate(residual, nu), "residual_ci": wilson(residual, nu),       # harmful prompt answered
        "served_rate": _rate(served, ns), "served_ci": wilson(served, ns),               # benign prompt answered
        "guard_false_block_rate": _rate(int(cnt_s.get("guardrail", 0)), ns),
        "model_over_refusal_rate": _rate(int(cnt_s.get("model", 0)), ns),
        "latency_p50": lat["p50"], "latency_p95": lat["p95"], "latency_mean": lat["mean"],
        "guard_p50": latency_stats(df["guard_ms"])["p50"], "guard_p95": latency_stats(df["guard_ms"])["p95"],
    }


def responses_from_model_run(model: Run) -> pd.DataFrame:
    """``prompt, expected, response`` rows for response-direction guardrail evaluation, from a model run.

    Label heuristic: an answer is ``block`` when the prompt should have been refused and the model did not
    refuse; everything else (refusals, answers to benign prompts) is ``pass``. Review the block rows —
    not refusing is not always harmful."""
    d = model.rows
    d = d[(d["outcome"] != "error") & d["response"].fillna("").astype(str).str.strip().ne("")]
    expected = ["block" if g and not p else "pass" for g, p in zip(d["gold_positive"], d["pred_positive"])]
    return pd.DataFrame({"prompt": d["prompt"], "expected": expected, "response": d["response"]})
