"""Keyword mock — lets the whole server + evaluation loop run without a GPU (tests, CI, demos)."""

from __future__ import annotations

import asyncio
from typing import Any

from .base import BACKENDS, GuardBackend, GuardInput, Verdict

DEFAULT_KEYWORDS = {
    "prompt_injection": ["ignore previous instructions", "ignore all previous", "忽略之前的指令", "忽略之前的所有指令", "忽略以上", "system prompt"],
    "jailbreak": ["dan mode", "developer mode", "jailbreak", "越獄"],
    "weapons_cbrn": ["make a bomb", "炸彈", "nerve agent"],
    "crime": ["steal credit card", "malware", "ransomware", "勒索軟體"],
    "confidential": ["internal only", "機密", "confidential"],
}


@BACKENDS.register("mock")
class MockBackend(GuardBackend):
    """Keyword matcher with a fake score (tests / CI / demos); ``delay_ms`` and ``fail_on`` simulate latency and errors."""

    def __init__(self, name: str, cfg: dict[str, Any], root_cfg: dict[str, Any]):
        super().__init__(name, cfg, root_cfg)
        self.keywords: dict[str, list[str]] = cfg.get("keywords", DEFAULT_KEYWORDS)
        self.delay_ms = float(cfg.get("delay_ms", 0))
        self.fail_on = cfg.get("fail_on")  # substring that triggers a simulated backend error

    async def _classify(self, inp: GuardInput) -> Verdict:
        if self.delay_ms:
            await asyncio.sleep(self.delay_ms / 1000)
        text = inp.target_text.lower()
        if self.fail_on and self.fail_on in text:
            raise RuntimeError("simulated backend failure")
        cats = sorted(c for c, kws in self.keywords.items() if any(k.lower() in text for k in kws))
        score = min(1.0, 0.45 * len(cats) + (0.1 if "?" in text else 0.0)) if cats else 0.05
        return Verdict(stage=self.name, unsafe=score >= self.threshold, score=score,
                       categories=cats if score >= self.threshold else [])
