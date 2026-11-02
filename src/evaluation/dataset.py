"""Load ``prompt,expected`` CSVs into an eval set, deterministically.

``data.files`` takes CSV paths, globs, or directories (a mix: every ``*.csv`` in it). Each file's name is
its source (``processed_data/aegis2.csv`` -> ``aegis2``); the language is detected from the prompt.
"""

from __future__ import annotations

import glob as _glob
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, Sequence, TypeVar

from ..config import PROJECT_ROOT, get_by_path
from ..data_schema import detect_language, file_sha256, read_csv, source_of


@dataclass(frozen=True)
class EvalSample:
    """A data row plus what the evaluation derives from where and what it is."""

    prompt: str
    expected: str
    source: str
    sample_id: str = ""
    language: str = field(default="", compare=False)

    @classmethod
    def build(cls, prompt: str, expected: str, source: str, key: str) -> "EvalSample":
        return cls(prompt=prompt, expected=expected, source=source, sample_id=f"{source}:{key}",
                   language=detect_language(prompt))

    @property
    def should_block(self) -> bool:
        return self.expected == "block"


class _HasExpected(Protocol):
    expected: str


T = TypeVar("T", bound=_HasExpected)


def _resolve(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else PROJECT_ROOT / path


def expand_files(files: Sequence[str]) -> list[Path]:
    """CSV paths, globs and directories (-> their ``*.csv``), in a stable order."""
    out: list[Path] = []
    for f in files:
        f = str(f)
        if any(ch in f for ch in "*?["):
            matches = sorted(Path(m) for m in _glob.glob(f)) if Path(f).is_absolute() else sorted(PROJECT_ROOT.glob(f))
            matches = [m for m in matches if m.suffix == ".csv"]
            if not matches:
                raise FileNotFoundError(f"glob matched no .csv file: {f}")
            out.extend(matches)
            continue
        p = _resolve(f)
        if p.is_dir():
            csvs = sorted(p.glob("*.csv"))
            if not csvs:
                raise FileNotFoundError(f"{p} contains no .csv file")
            out.extend(csvs)
        elif p.exists():
            out.append(p)
        else:
            raise FileNotFoundError(f"{p} (run the converter / mix first: python -m src.converters ...)")
    return out


def sample_rows(rows: list[T], n: int | None, strategy: str, seed: int) -> list[T]:
    """``stratified`` keeps the ``expected`` ratio; ``balanced`` = equal per expected value; ``random``;
    ``head`` = first n. Deterministic for a given seed."""
    if not n or n >= len(rows):
        return list(rows)
    rng = random.Random(seed)
    if strategy == "head":
        return list(rows[:n])
    if strategy == "random":
        return rng.sample(list(rows), n)
    if strategy not in ("stratified", "balanced"):
        raise ValueError(f"unknown sampling strategy {strategy!r} (stratified | balanced | random | head)")
    groups: dict[str, list[T]] = defaultdict(list)
    for r in rows:
        groups[r.expected].append(r)
    quotas = _allocate(n, {k: len(v) for k, v in groups.items()}, equal=strategy == "balanced")
    out: list[T] = []
    for key in sorted(groups):
        out += rng.sample(groups[key], quotas[key])
    rng.shuffle(out)  # a later `limit` / partial run must not see one class only
    return out


def _allocate(n: int, sizes: dict, equal: bool = False) -> dict:
    """Largest-remainder allocation that sums to exactly n (capped by availability), >=1 per stratum when n allows."""
    keys = sorted(sizes)
    total = sum(sizes.values())
    want = {k: n / len(keys) for k in keys} if equal else {k: n * sizes[k] / total for k in keys}
    quota = {k: min(sizes[k], int(want[k])) for k in keys}
    if n >= len(keys):
        for k in keys:
            quota[k] = max(quota[k], 1)
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


def load_eval_set(cfg: dict[str, Any]) -> tuple[list[EvalSample], dict[str, Any]]:
    """Returns (samples, manifest). The manifest (file hashes, counts) is saved with results."""
    files = get_by_path(cfg, "data.files", [])
    if not files:
        raise ValueError("data.files is empty")
    seed = int(get_by_path(cfg, "run.seed", 42))
    max_n = get_by_path(cfg, "data.max_samples_per_file")
    strategy = get_by_path(cfg, "data.sampling", "stratified")
    expected = set(get_by_path(cfg, "data.expected") or [])
    languages = set(get_by_path(cfg, "data.languages") or [])
    sources = set(get_by_path(cfg, "data.sources") or [])

    samples: list[EvalSample] = []
    manifest: dict[str, Any] = {"files": [], "seed": seed, "sampling": strategy, "max_samples_per_file": max_n}
    seen: dict[str, str] = {}
    for path in expand_files(files):
        source = source_of(path)
        if sources and source not in sources:
            continue
        raw = read_csv(path)
        rows = [EvalSample.build(s.prompt, s.expected, source, s.key) for s in raw]
        for r in rows:  # predictions are keyed by sample_id: one prompt, one expectation per source
            if seen.setdefault(r.sample_id, r.expected) != r.expected:
                raise ValueError(f"{path}: the same prompt appears with expected={seen[r.sample_id]} and {r.expected}")
        rows = list({r.sample_id: r for r in rows}.values())
        if expected:
            rows = [r for r in rows if r.expected in expected]
        if languages:
            rows = [r for r in rows if r.language in languages]
        rows = sample_rows(rows, max_n, strategy, seed)
        samples.extend(rows)
        manifest["files"].append({
            "path": str(path.relative_to(PROJECT_ROOT)) if path.is_relative_to(PROJECT_ROOT) else str(path),
            "source": source, "sha256": file_sha256(path), "rows_in_file": len(raw), "rows_used": len(rows),
        })
    if not samples:
        raise ValueError("no samples left after data.* filters")

    uniq = list({s.sample_id: s for s in samples}.values())
    manifest["duplicates_dropped"] = len(samples) - len(uniq)
    samples = uniq

    limit = get_by_path(cfg, "eval.limit")
    if limit:
        samples = sample_rows(samples, int(limit), strategy, seed)

    manifest["n_samples"] = len(samples)
    for name, key in (("expected_counts", "expected"), ("source_counts", "source"), ("language_counts", "language")):
        manifest[name] = dict(sorted(Counter(getattr(s, key) for s in samples).items()))
    return samples, manifest
