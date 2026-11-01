"""Load processed CSVs into an eval set, deterministically."""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from ..config import PROJECT_ROOT, get_by_path
from ..data_schema import GuardrailSample, file_sha256, read_csv


def _resolve(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _expand(files: list[str]) -> list[Path]:
    out: list[Path] = []
    for f in files:
        if any(ch in f for ch in "*?["):
            matches = sorted(PROJECT_ROOT.glob(f))
            if not matches:
                raise FileNotFoundError(f"glob matched nothing: {f}")
            out.extend(matches)
        else:
            p = _resolve(f)
            if not p.exists():
                raise FileNotFoundError(f"{p} (run the converter first: python -m src.converters ...)")
            out.append(p)
    return out


def sample_rows(rows: list[GuardrailSample], n: int | None, strategy: str, seed: int) -> list[GuardrailSample]:
    """``stratified`` keeps the label (and input_type) ratio; ``random``; ``head`` = first n."""
    if not n or n >= len(rows):
        return rows
    rng = random.Random(seed)
    if strategy == "head":
        return rows[:n]
    if strategy == "random":
        return rng.sample(rows, n)
    if strategy == "balanced":  # equal safe/unsafe (as far as available) — for recall/FPR at small n
        groups = defaultdict(list)
        for r in rows:
            groups[r.label.value].append(r)
        per = n // max(len(groups), 1)
        out = []
        for k in sorted(groups):
            g = groups[k]
            out += rng.sample(g, min(per, len(g)))
        return out
    groups: dict[tuple, list[GuardrailSample]] = defaultdict(list)
    for r in rows:
        groups[(r.label.value, r.input_type)].append(r)
    out = []
    for key in sorted(groups):
        g = groups[key]
        k = max(1, round(n * len(g) / len(rows)))
        out += rng.sample(g, min(k, len(g)))
    rng.shuffle(out)
    return out[:n]


def load_eval_set(cfg: dict[str, Any]) -> tuple[list[GuardrailSample], dict[str, Any]]:
    """Returns (samples, manifest). The manifest (file hashes, counts) is saved with results."""
    files = get_by_path(cfg, "data.files", [])
    if not files:
        raise ValueError("data.files is empty")
    seed = int(get_by_path(cfg, "run.seed", 42))
    max_n = get_by_path(cfg, "data.max_samples_per_file")
    strategy = get_by_path(cfg, "data.sampling", "stratified")
    input_types = set(get_by_path(cfg, "data.input_types") or ["request", "response"])
    languages = get_by_path(cfg, "data.languages")
    sources = get_by_path(cfg, "data.sources")

    samples: list[GuardrailSample] = []
    manifest: dict[str, Any] = {"files": [], "seed": seed, "sampling": strategy, "max_samples_per_file": max_n}
    for path in _expand(files):
        rows = read_csv(path)
        n_raw = len(rows)
        rows = [r for r in rows if r.input_type in input_types]
        if languages:
            rows = [r for r in rows if r.language in languages]
        if sources:
            rows = [r for r in rows if r.source in sources]
        rows = sample_rows(rows, max_n, strategy, seed)
        samples.extend(rows)
        manifest["files"].append({
            "path": str(path.relative_to(PROJECT_ROOT)) if path.is_relative_to(PROJECT_ROOT) else str(path),
            "sha256": file_sha256(path), "rows_in_file": n_raw, "rows_used": len(rows),
        })

    if get_by_path(cfg, "data.dedupe", True):
        seen, uniq = set(), []
        for s in samples:
            if s.sample_id not in seen:
                seen.add(s.sample_id)
                uniq.append(s)
        manifest["duplicates_dropped"] = len(samples) - len(uniq)
        samples = uniq

    limit = get_by_path(cfg, "eval.limit")
    if limit:
        samples = sample_rows(samples, int(limit), strategy, seed)

    manifest["n_samples"] = len(samples)
    manifest["label_counts"] = _count(samples, lambda s: s.label.value)
    manifest["input_type_counts"] = _count(samples, lambda s: s.input_type)
    manifest["language_counts"] = _count(samples, lambda s: s.language)
    manifest["source_counts"] = _count(samples, lambda s: s.source)
    return samples, manifest


def _count(samples, key) -> dict[str, int]:
    c: dict[str, int] = defaultdict(int)
    for s in samples:
        c[key(s)] += 1
    return dict(sorted(c.items()))
