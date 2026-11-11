"""``guardrail`` task: is the guardrail's block decision right? positive = unsafe."""

from __future__ import annotations

from typing import Any

from ...config import get_by_path
from ...data_schema import EXPECTED, EXPECTED_ACTION
from ..dataset import EvalSample
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

    def select(self, samples: list[EvalSample]) -> list[EvalSample]:
        select = getattr(self.target, "select", None)  # e.g. response direction keeps rows with a response
        return select(samples) if select else samples

    def subject(self) -> dict[str, Any]:
        d = self.target.describe()
        adapter = (d.get("adapter") or {}).get("type")
        return {"kind": "guardrail_service", "name": get_by_path(self.cfg, "run.name", "guardrail"),
                "target": d.get("type"), "adapter": adapter, "input_type": d.get("input_type", "request"),
                "base_url": d.get("base_url")}

    async def predict(self, sample: EvalSample) -> dict[str, Any]:
        pred = await self.target.predict(sample)  # TargetError propagates -> runner retries
        row = pred.to_dict()
        row["gold_positive"] = self.gold(sample)
        row["pred_positive"] = row.pop("unsafe")
        # exact expectation: block -> BLOCKED, pass -> NONE, mask -> GUARDRAIL_INTERVENED
        row["match"] = None if row.get("action") is None else row["action"] == EXPECTED_ACTION[sample.expected]
        return row

    def extra_metrics(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        """Beyond block / not-block: did the server do exactly what ``expected`` says? (mask rows must be
        masked — passing them unmasked leaks the data, blocking them over-blocks)."""
        ok = [r for r in rows if r.get("pred_positive") is not None and r.get("action")]
        by_expected: dict[str, Any] = {}
        for exp in EXPECTED:
            grp = [r for r in ok if r.get("expected") == exp]
            if not grp:
                continue
            actions: dict[str, int] = {}
            for r in grp:
                actions[r["action"]] = actions.get(r["action"], 0) + 1
            by_expected[exp] = {"n": len(grp), "match_rate": sum(bool(r["match"]) for r in grp) / len(grp),
                                "actions": dict(sorted(actions.items()))}
        out: dict[str, Any] = {"expected_vs_action": by_expected}
        if ok:
            out["exact_match_rate"] = sum(bool(r["match"]) for r in ok) / len(ok)
        return out

    def contaminated_patterns(self) -> list[str]:
        pats = super().contaminated_patterns()
        desc = self.target.describe()
        # inprocess: the server YAML; guardrail_server: the stages reported by GET /v1/info
        stages = (get_by_path(desc, "server_config.pipeline.stages", []) or []) + (desc.get("stages") or [])
        for st in stages:
            if st.get("enabled", True):
                pats += st.get("trained_on") or []
        return sorted(set(pats))
