"""Load processed CSVs into an eval set, deterministically."""

from __future__ import annotations

import glob as _glob
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
            matches = sorted(Path(m) for m in _glob.glob(f)) if Path(f).is_absolute() else sorted(PROJECT_ROOT.glob(f))
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
        quotas = _allocate(n, {k: len(v) for k, v in groups.items()}, equal=True)
    else:
        groups = defaultdict(list)
        for r in rows:
            groups[(r.label.value, r.input_type)].append(r)
        quotas = _allocate(n, {k: len(v) for k, v in groups.items()})
    out = []
    for key in sorted(groups):
        out += rng.sample(groups[key], quotas[key])
    rng.shuffle(out)  # a later `limit` / partial run must not see one class only
    return out


def _allocate(n: int, sizes: dict, equal: bool = False) -> dict:
    """Largest-remainder allocation that sums to exactly n (capped by availability), >=1 per stratum when n allows."""
    keys = sorted(sizes)
    total = sum(sizes.values())
    if equal:
        want = {k: n / len(keys) for k in keys}
    else:
        want = {k: n * sizes[k] / total for k in keys}
    quota = {k: min(sizes[k], int(want[k])) for k in keys}
    if n >= len(keys):
        for k in keys:
            quota[k] = max(quota[k], 1)
    # hand out what is left by largest remainder, then to any stratum with spare rows
    order = sorted(keys, key=lambda k: (want[k] - int(want[k])), reverse=True)
    while sum(quota.values()) < n:
        spare = [k for k in order if quota[k] < sizes[k]]
        if not spare:
            break
        quota[spare[0]] += 1
        order = order[1:] + order[:1]
    while sum(quota.values()) > n:  # the >=1 floor can overshoot when n is tiny
        k = max(keys, key=lambda k: quota[k])
        quota[k] -= 1
    return quota


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

    ids: dict[str, str] = {}
    for smp in samples:  # results are keyed by sample_id: the same id with two labels would be silently merged
        if ids.setdefault(smp.sample_id, smp.label.value) != smp.label.value:
            raise ValueError(f"sample_id {smp.sample_id} appears with conflicting labels across data files")
    # Always dedupe: predictions are keyed by sample_id, so duplicates could not be scored separately anyway.
    seen: set[str] = set()
    uniq = []
    for smp in samples:
        if smp.sample_id not in seen:
            seen.add(smp.sample_id)
            uniq.append(smp)
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
