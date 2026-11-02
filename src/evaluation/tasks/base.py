"""Evaluation task contract.

A *task* decides which samples it can score, how one sample is predicted, what "positive" means,
and which extra metrics it reports. The runner, metrics, report and CLI are shared::

    task = TASKS.build(cfg["task"], cfg)          # task.type: guardrail | model_safety
    samples = task.select(samples)
    await task.startup()
    row = await task.predict(sample)              # -> dict with gold_positive / pred_positive / score / ...
    metrics = task.compute_metrics(rows)

Adding a task = subclass ``EvalTask``, register it, give it a config section ``task.<name>``.
"""

from __future__ import annotations

import abc
from typing import Any, ClassVar

from ...common import Registry
from ...config import get_by_path
from ..dataset import EvalSample
from ..metrics import evaluate_rows


class EvalTask(abc.ABC):
    registry_name: ClassVar[str] = "base"
    #: report wording for the binary metrics, e.g. {"recall": "block rate on unsafe"}
    METRIC_LABELS: ClassVar[dict[str, str]] = {}
    POSITIVE: ClassVar[str] = "positive"

    def __init__(self, task_cfg: dict[str, Any], cfg: dict[str, Any]):
        self.task_cfg = dict(task_cfg)
        self.cfg = cfg  # the whole evaluation config (metrics.*, eval.*, envs ...)
        self.section: dict[str, Any] = dict(get_by_path(cfg, f"task.{self.registry_name}", {}) or {})

    @classmethod
    def from_config(cls, task_cfg: dict[str, Any], cfg: dict[str, Any]) -> "EvalTask":
        return cls(task_cfg, cfg)

    # -- lifecycle -------------------------------------------------------- #
    async def startup(self) -> None: ...

    async def shutdown(self) -> None: ...

    def describe(self) -> dict[str, Any]:
        return {"task": self.registry_name, "positive": self.POSITIVE}

    # -- per sample ------------------------------------------------------- #
    def gold(self, sample: EvalSample) -> bool:
        """Ground-truth positive for a sample (default: the server should block it)."""
        return sample.should_block

    def select(self, samples: list[EvalSample]) -> list[EvalSample]:
        """Samples this task can score (default: all)."""
        return samples

    @abc.abstractmethod
    async def predict(self, sample: EvalSample) -> dict[str, Any]:
        """Return row fields. Must include ``gold_positive`` (bool) and ``pred_positive`` (bool, or None
        on a non-retryable error together with ``error``). Raise ``TargetError`` for retryable failures."""

    # -- metrics ---------------------------------------------------------- #
    def compute_metrics(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        c = self.cfg
        m = evaluate_rows(
            rows,
            threshold=get_by_path(c, "metrics.threshold"),
            error_policy=get_by_path(c, "eval.error_policy", "exclude"),
            n_boot=int(get_by_path(c, "metrics.bootstrap", 1000)),
            ci=float(get_by_path(c, "metrics.ci", 0.95)),
            seed=int(get_by_path(c, "run.seed", 42)),
            fpr_targets=get_by_path(c, "metrics.fpr_targets", [0.01, 0.05]),
            slice_by=get_by_path(c, "metrics.slice_by", ["source", "language"]),
            min_slice_size=int(get_by_path(c, "metrics.min_slice_size", 1)),
            score_stage=get_by_path(c, "metrics.score_stage"),
        )
        m["task"] = self.registry_name
        m["metric_labels"] = dict(self.METRIC_LABELS)
        m.update(self.extra_metrics(rows))
        return m

    def extra_metrics(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {}

    def contaminated_patterns(self) -> list[str]:
        """Source globs the evaluated model was trained on (excluded from the 'clean' metrics)."""
        return list(get_by_path(self.cfg, "metrics.contaminated_sources") or [])


TASKS: Registry[EvalTask] = Registry(
    "evaluation task",
    base=EvalTask,
    lazy_modules=["src.evaluation.tasks.guardrail", "src.evaluation.tasks.refusal"],
)


def build_task(cfg: dict[str, Any]) -> EvalTask:
    task_cfg = dict(get_by_path(cfg, "task", {}) or {})
    task_cfg.setdefault("type", "guardrail")
    return TASKS.build(task_cfg, cfg)
