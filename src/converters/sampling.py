"""Seeded, deterministic row sampling shared by ``options.sample`` (converters) and eval mixes.

    stratified   keep the ratio of ``expected`` values (block / pass / mask) of the input
    balanced     equal rows per ``expected`` value, as far as each has rows (recall / FPR at small n)
    random       uniform sample
    head         the first n rows (stream order; debugging only)

``n`` of ``None`` / 0 or ``n >= len(rows)`` returns every row unchanged. The same rows, ``n``,
strategy and seed always give the same picks in the same order.
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Callable, Hashable
from typing import Any, TypeVar

T = TypeVar("T")
STRATEGIES = ("stratified", "balanced", "random", "head")


def _expected(row: Any) -> str:
    return row.expected


def sample_rows(rows: list[T], n: int | None, strategy: str = "stratified", seed: int = 42,
                key: Callable[[T], Hashable] = _expected) -> list[T]:
    """Pick ``n`` of ``rows``; ``key`` is the stratum of a row (default: its ``expected`` value)."""
    if strategy not in STRATEGIES:
        raise ValueError(f"sampling strategy must be one of {STRATEGIES}, got {strategy!r}")
    if not n or n >= len(rows):
        return list(rows)
    if strategy == "head":
        return list(rows[:n])
    rng = random.Random(seed)
    if strategy == "random":
        return rng.sample(rows, n)
    groups: dict[Hashable, list[T]] = defaultdict(list)
    for r in rows:
        groups[key(r)].append(r)
    quotas = allocate(n, {k: len(v) for k, v in groups.items()}, equal=strategy == "balanced")
    out: list[T] = []
    for k in sorted(groups, key=str):
        out += rng.sample(groups[k], quotas[k])
    rng.shuffle(out)  # a later `limit` / partial run must not see one class only
    return out


def allocate(n: int, sizes: dict[Hashable, int], equal: bool = False) -> dict[Hashable, int]:
    """Largest-remainder quotas summing to exactly ``n`` (capped by availability), >= 1 per stratum when n allows."""
    keys = sorted(sizes, key=str)
    total = sum(sizes.values())
    want = {k: (n / len(keys) if equal else n * sizes[k] / total) for k in keys}
    quota = {k: min(sizes[k], int(want[k])) for k in keys}
    if n >= len(keys):
        for k in keys:
            quota[k] = max(quota[k], min(1, sizes[k]))
    # hand out what is left by largest remainder, then to any stratum with spare rows
    order = sorted(keys, key=lambda k: want[k] - int(want[k]), reverse=True)
    while sum(quota.values()) < n:
        spare = [k for k in order if quota[k] < sizes[k]]
        if not spare:
            break
        quota[spare[0]] += 1
        order.remove(spare[0])  # round robin: the next extra row goes to another stratum
        order.append(spare[0])
    while sum(quota.values()) > n:  # the >= 1 floor can overshoot when n is tiny
        quota[max(keys, key=lambda k: quota[k])] -= 1
    return quota
