"""``guardrail`` task: is the guardrail's block decision right? positive = unsafe."""

from __future__ import annotations

from typing import Any

from ...config import get_by_path
from ...data_schema import GuardrailSample
from ..targets import Target, build_target
from .base import TASKS, EvalTask


@TASKS.register("guardrail")
class GuardrailTask(EvalTask):
    """Evaluate a guardrail (``target.type``: guardrail_server = a running server, or inprocess)."""

    POSITIVE = "unsafe"
    METRIC_LABELS = {
        "precision": "precision (blocked that were unsafe)",
        "recall": "recall (block rate on unsafe)",
        "fpr": "FPR (false block rate on safe)",
    }

    def __init__(self, task_cfg: dict[str, Any], cfg: dict[str, Any]):
        super().__init__(task_cfg, cfg)
        self.target: Target = build_target(cfg)

    async def startup(self) -> None:
        await self.target.startup()

    async def shutdown(self) -> None:
        await self.target.shutdown()

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), **self.target.describe()}

    async def predict(self, sample: GuardrailSample) -> dict[str, Any]:
        pred = await self.target.predict(sample)  # TargetError propagates -> runner retries
        row = pred.to_dict()
        row["gold_positive"] = self.gold(sample)
        row["pred_positive"] = row.pop("unsafe")
        return row

    def contaminated_patterns(self) -> list[str]:
        pats = super().contaminated_patterns()
        desc = self.target.describe()
        # inprocess: the server YAML; guardrail_server: the stages reported by GET /v1/info
        stages = (get_by_path(desc, "server_config.pipeline.stages", []) or []) + (desc.get("stages") or [])
        for st in stages:
            if st.get("enabled", True):
                pats += st.get("trained_on") or []
        return sorted(set(pats))
