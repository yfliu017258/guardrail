"""Deterministic rules: secrets, PII formats, internal codenames / confidential markers.

Each rule has ``action: block`` (reject the request) or ``action: mask`` (replace the
match and let the request through -> LiteLLM ``GUARDRAIL_INTERVENED``).
Rules come from ``rules:`` inline and/or ``rules_file:`` (YAML list). Keep the real
codename list in an internal file that is NOT committed to git.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from ...config import PROJECT_ROOT
from .base import GuardBackend, GuardInput, Verdict, register_backend

BUILTIN_RULES: list[dict[str, Any]] = [
    {"name": "private_key", "pattern": r"-----BEGIN (?:RSA |EC |OPENSSH |)PRIVATE KEY-----", "category": "confidential", "action": "block"},
    {"name": "aws_access_key", "pattern": r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b", "category": "confidential", "action": "mask"},
    {"name": "generic_api_key", "pattern": r"\b(?:sk|pk|rk)-[A-Za-z0-9_-]{20,}\b", "category": "confidential", "action": "mask"},
    {"name": "github_token", "pattern": r"\bgh[pousr]_[A-Za-z0-9]{36,}\b", "category": "confidential", "action": "mask"},
    {"name": "tw_national_id", "pattern": r"\b[A-Z][12]\d{8}\b", "category": "privacy_pii", "action": "mask"},
    {"name": "tw_mobile", "pattern": r"(?<!\d)09\d{2}-?\d{3}-?\d{3}(?!\d)", "category": "privacy_pii", "action": "mask"},
    {"name": "email", "pattern": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", "category": "privacy_pii", "action": "mask"},
]


@register_backend("regex")
class RegexBackend(GuardBackend):
    def __init__(self, name: str, cfg: dict[str, Any], root_cfg: dict[str, Any]):
        super().__init__(name, cfg, root_cfg)
        rules: list[dict] = list(BUILTIN_RULES) if cfg.get("use_builtin", True) else []
        disabled = set(cfg.get("disable", []))
        rules = [r for r in rules if r["name"] not in disabled]
        rules += cfg.get("rules", [])
        if cfg.get("rules_file"):
            p = Path(cfg["rules_file"])
            p = p if p.is_absolute() else PROJECT_ROOT / p
            if p.exists():
                rules += yaml.safe_load(p.read_text(encoding="utf-8")) or []
            elif cfg.get("rules_file_required", False):
                raise FileNotFoundError(p)
        self.rules = [
            {**r, "_re": re.compile(r["pattern"], re.IGNORECASE if r.get("ignore_case", False) else 0)}
            for r in rules
        ]
        self.mask_template = cfg.get("mask_template", "[REDACTED_{name}]")
        self.scan_all_texts = bool(cfg.get("scan_all_texts", True))

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "rules": [r["name"] for r in self.rules]}

    async def _classify(self, inp: GuardInput) -> Verdict:
        texts = list(inp.texts) if (self.scan_all_texts and inp.texts) else [inp.target_text]
        hits_block, hits_mask, cats = [], [], set()
        masked = []
        for t in texts:
            new = t
            for r in self.rules:
                if r["_re"].search(new):
                    cats.add(r.get("category", "other"))
                    if r.get("action", "block") == "block":
                        hits_block.append(r["name"])
                    else:
                        hits_mask.append(r["name"])
                        new = r["_re"].sub(self.mask_template.format(name=r["name"].upper()), new)
            masked.append(new)
        unsafe = bool(hits_block)
        return Verdict(
            stage=self.name,
            unsafe=unsafe,
            score=1.0 if unsafe else (0.5 if hits_mask else 0.0),
            categories=sorted(cats),
            native_categories=sorted(set(hits_block + hits_mask)),
            masked_texts=masked if (hits_mask and not unsafe and inp.texts) else None,
            reason=",".join(sorted(set(hits_block + hits_mask))),
        )
