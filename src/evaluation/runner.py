"""Async runner shared by every task: bounded concurrency, retries on ``TargetError``,
incremental + resumable predictions.jsonl. Rows = sample fields + whatever ``task.predict`` returns."""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from pathlib import Path
from typing import Any

from tqdm.asyncio import tqdm

from ..config import get_by_path
from ..data_schema import GuardrailSample
from .errors import TargetError

log = logging.getLogger("evaluation")


def read_records(pred_path: Path, repair: bool = False) -> list[dict]:
    """All records in file order. A truncated final line (process killed mid-write) is skipped and,
    with ``repair``, cut off so that appending continues on a clean line boundary."""
    if not pred_path.exists():
        return []
    raw = pred_path.read_bytes()
    out, good_end, pos = [], 0, 0
    for line in raw.split(b"\n"):
        end = pos + len(line) + 1
        if line.strip():
            try:
                out.append(json.loads(line))
                good_end = min(end, len(raw))
            except json.JSONDecodeError:
                if end < len(raw):  # corruption in the middle is not a crash artefact -> fail loudly
                    raise
                log.warning("%s: dropping truncated final record (%d bytes)", pred_path, len(line))
                if repair:
                    with pred_path.open("r+b") as f:
                        f.truncate(good_end)
        pos = end
    return out


def load_done(pred_path: Path, repair: bool = False) -> dict[str, dict]:
    """Finished predictions (for --resume): latest record per sample, transport-error rows excluded."""
    done: dict[str, dict] = {}
    for rec in read_records(pred_path, repair):
        if rec.get("_retry"):
            done.pop(rec["sample_id"], None)
        else:
            done[rec["sample_id"]] = rec
    return done


def load_latest(pred_path: Path) -> list[dict]:
    """Latest record per sample INCLUDING transport failures (for --rescore: they are errors, not missing)."""
    latest: dict[str, dict] = {}
    for rec in read_records(pred_path):
        latest[rec["sample_id"]] = rec
    return list(latest.values())


async def run_predictions(task: Any, samples: list[GuardrailSample], cfg: dict[str, Any], pred_path: Path) -> list[dict]:
    """``task`` is an ``EvalTask`` (anything with ``async predict(sample) -> dict`` and ``registry_name``)."""
    concurrency = int(get_by_path(cfg, "eval.concurrency", 16))
    retries = int(get_by_path(cfg, "eval.retries", 3))
    seed = int(get_by_path(cfg, "run.seed", 42))
    rng = random.Random(seed)

    done = load_done(pred_path, repair=True)
    todo = [s for s in samples if s.sample_id not in done]
    if done:
        log.info("resume: %d done, %d remaining", len(done), len(todo))

    sem = asyncio.Semaphore(concurrency)
    write_lock = asyncio.Lock()
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    fh = pred_path.open("a", encoding="utf-8")

    async def one(s: GuardrailSample) -> dict:
        async with sem:
            t_start = time.perf_counter()
            pred: dict | None = None
            err: str | None = None
            retry_flag = False
            for attempt in range(retries + 1):
                try:
                    pred = await task.predict(s)
                    break
                except TargetError as e:
                    err = str(e)
                    retry_flag = True
                    if attempt < retries:
                        await asyncio.sleep(min(30.0, 0.5 * 2**attempt) * (0.5 + rng.random()))
                except Exception as e:  # non-retryable (4xx, bad payload, bug) -> record and move on
                    err = f"{type(e).__name__}: {e}"
                    retry_flag = False
                    break
            if pred is None:
                pred = {"gold_positive": None, "pred_positive": None, "error": err,
                        "latency_ms": (time.perf_counter() - t_start) * 1000}
            rec = {**_sample_fields(s), **pred, "sample_id": s.sample_id}
            if rec.get("gold_positive") is None:  # failed rows still need ground truth for error_policy fills
                rec["gold_positive"] = task.gold(s) if hasattr(task, "gold") else s.is_unsafe
            if pred.get("pred_positive") is None and retry_flag and pred.get("error") == err:
                rec["_retry"] = True  # transport failure: --resume will try this row again
            async with write_lock:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
            return rec

    try:
        new = await tqdm.gather(*(one(s) for s in todo), desc=f"eval[{getattr(task, 'registry_name', 'task')}]", total=len(todo))
    finally:
        fh.close()

    by_id = {**done, **{r["sample_id"]: r for r in new}}
    return [by_id[s.sample_id] for s in samples if s.sample_id in by_id]


def _sample_fields(s: GuardrailSample) -> dict[str, Any]:
    return {
        "source": s.source,
        "split": s.split,
        "input_type": s.input_type,
        "language": s.language,
        "attack_type": s.attack_type,
        "gold_label": s.label.value,
        "gold_categories": [c.value for c in s.categories],
        "gold_refusal_label": s.refusal_label,
        "text_preview": s.text[:300],
    }
