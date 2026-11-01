"""Async runner: bounded concurrency, retries, incremental + resumable predictions.jsonl."""

from __future__ import annotations

import asyncio
import json
import logging
import random
from pathlib import Path
from typing import Any

from tqdm.asyncio import tqdm

from ..config import get_by_path
from ..data_schema import GuardrailSample
from .targets import Prediction, Target, TargetError

log = logging.getLogger("evaluation")


def load_done(pred_path: Path) -> dict[str, dict]:
    """Already-finished predictions (for --resume). Rows with transport errors are re-run."""
    done: dict[str, dict] = {}
    if pred_path.exists():
        with pred_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if not rec.get("_retry"):
                    done[rec["sample_id"]] = rec
    return done


async def run_predictions(
    target: Target, samples: list[GuardrailSample], cfg: dict[str, Any], pred_path: Path
) -> list[dict]:
    concurrency = int(get_by_path(cfg, "eval.concurrency", 16))
    retries = int(get_by_path(cfg, "eval.retries", 3))
    seed = int(get_by_path(cfg, "run.seed", 42))
    rng = random.Random(seed)

    done = load_done(pred_path)
    todo = [s for s in samples if s.sample_id not in done]
    if done:
        log.info("resume: %d done, %d remaining", len(done), len(todo))

    sem = asyncio.Semaphore(concurrency)
    write_lock = asyncio.Lock()
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    fh = pred_path.open("a", encoding="utf-8")

    async def one(s: GuardrailSample) -> dict:
        async with sem:
            pred: Prediction | None = None
            err: str | None = None
            retry_flag = False
            for attempt in range(retries + 1):
                try:
                    pred = await target.predict(s)
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
                pred = Prediction(sample_id=s.sample_id, unsafe=None, error=err)
            rec = {**_sample_fields(s), **pred.to_dict()}
            if pred.unsafe is None and retry_flag:
                rec["_retry"] = True  # transport failure: --resume will try this row again
            async with write_lock:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
            return rec

    try:
        new = await tqdm.gather(*(one(s) for s in todo), desc=f"eval[{target.name}]", total=len(todo))
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
        "text_preview": s.text[:300],
    }
