r"""Deterministic rules: secrets, PII formats, internal codenames / confidential markers.

Each rule has ``action: block`` (reject the request) or ``action: mask`` (replace the
match and let the request through -> LiteLLM ``GUARDRAIL_INTERVENED``).
Rules come from ``rules:`` inline and/or ``rules_file:`` (YAML list). Keep the real
codename list in an internal file that is NOT committed to git.

Word boundaries: Python's ``re`` counts CJK characters as ``\w``, so ``\b`` never matches between
``證`` and ``A`` and ``\bA123456789\b`` misses ``身分證A123456789``. The built-in rules use the ASCII
lookarounds ``(?<![A-Za-z0-9_])`` / ``(?![A-Za-z0-9_])`` instead; write ``rules_file`` patterns the
same way (``(?<![A-Za-z0-9_])PROJECT_ALPHA(?![A-Za-z0-9_])`` also matches ``專案PROJECT_ALPHA進度``).

Mask hits that cannot be applied block instead: a tool call (LiteLLM 1.92 never copies tool calls
back from the guard response) and, with ``response_mask_action: block``, any response — LiteLLM 1.92
ignores ``GUARDRAIL_INTERVENED`` on streamed output, so a streaming deployment must block PII there.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import yaml

from ...config.config import _resolve_path
from .base import BACKENDS, GuardBackend, GuardInput, Verdict

log = logging.getLogger("guardrail.regex")

#: ASCII-only word boundaries (``\b`` treats CJK as word characters, see the module docstring).
_B = r"(?<![A-Za-z0-9_])"
_E = r"(?![A-Za-z0-9_])"

BUILTIN_RULES: list[dict[str, Any]] = [
    {"name": "private_key", "pattern": r"-----BEGIN (?:RSA |EC |OPENSSH |)PRIVATE KEY-----", "category": "confidential", "action": "block"},
    {"name": "aws_access_key", "pattern": rf"{_B}(?:AKIA|ASIA)[0-9A-Z]{{16}}{_E}", "category": "confidential", "action": "mask"},
    {"name": "generic_api_key", "pattern": rf"{_B}(?:sk|pk|rk)-[A-Za-z0-9_-]{{20,}}", "category": "confidential", "action": "mask"},
    {"name": "github_token", "pattern": rf"{_B}gh[pousr]_[A-Za-z0-9]{{36,}}{_E}", "category": "confidential", "action": "mask"},
    {"name": "tw_national_id", "pattern": rf"{_B}[A-Z][12]\d{{8}}{_E}", "category": "privacy_pii", "action": "mask"},
    {"name": "tw_mobile", "pattern": r"(?<!\d)09\d{2}-?\d{3}-?\d{3}(?!\d)", "category": "privacy_pii", "action": "mask"},
    {"name": "email", "pattern": rf"(?<![A-Za-z0-9._%+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{{2,}}{_E}",
     "category": "privacy_pii", "action": "mask"},
]

RESPONSE_MASK_ACTIONS = ("mask", "block")


@BACKENDS.register("regex")
class RegexBackend(GuardBackend):
    """Regex rules for secrets / PII / codenames: ``block`` rejects, ``mask`` redacts (hard decisions, no score)."""

    def __init__(self, name: str, cfg: dict[str, Any], root_cfg: dict[str, Any]):
        super().__init__(name, cfg, root_cfg)
        rules: list[dict] = list(BUILTIN_RULES) if cfg.get("use_builtin", True) else []
        disabled = set(cfg.get("disable", []))
        rules = [r for r in rules if r["name"] not in disabled]
        rules += cfg.get("rules", [])
        self.rules_file_loaded: bool | None = None  # None = no rules_file configured
        if cfg.get("rules_file"):
            try:  # same resolution as the config loader (cwd, repo root, config dir), so hashes and stage agree
                p = _resolve_path(cfg["rules_file"])
            except FileNotFoundError:
                p = None
            if p is not None:
                rules += yaml.safe_load(p.read_text(encoding="utf-8")) or []
                self.rules_file_loaded = True
            elif cfg.get("rules_file_required", False):
                raise FileNotFoundError(f"stage {name!r}: rules_file {cfg['rules_file']!r} not found")
            else:
                self.rules_file_loaded = False
                log.warning("stage %s: rules_file %s not found - running with built-in rules only "
                            "(set rules_file_required: true to make this fatal)", name, cfg["rules_file"])
        self.rules = [
            {**r, "_re": re.compile(r["pattern"], re.IGNORECASE if r.get("ignore_case", False) else 0)}
            for r in rules
        ]
        self.mask_template = cfg.get("mask_template", "[REDACTED_{name}]")
        self.scan_all_texts = bool(cfg.get("scan_all_texts", True))
        # mask (default) | block: LiteLLM 1.92 does not apply GUARDRAIL_INTERVENED to streamed output, so a
        # deployment whose clients stream must block response-side mask hits or the PII reaches the client
        self.response_mask_action = str(cfg.get("response_mask_action", "mask"))
        if self.response_mask_action not in RESPONSE_MASK_ACTIONS:
            raise ValueError(f"stage {name!r}: response_mask_action must be one of {list(RESPONSE_MASK_ACTIONS)}, "
                             f"got {self.response_mask_action!r}")

    @property
    def masks(self) -> bool:
        return any(r.get("action", "block") == "mask" for r in self.rules)

    def describe(self) -> dict[str, Any]:
        out = {**super().describe(), "rules": [r["name"] for r in self.rules],
               "response_mask_action": self.response_mask_action}
        if self.rules_file_loaded is not None:
            out["rules_file"] = self.cfg.get("rules_file")
            out["rules_file_loaded"] = self.rules_file_loaded
        return out

    def _mask_blocks(self, inp: GuardInput) -> bool:
        """True when a mask hit on ``inp`` must block: the redaction could not be applied."""
        return not inp.maskable or (inp.input_type == "response" and self.response_mask_action == "block")

    async def _classify(self, inp: GuardInput) -> Verdict:
        texts = list(inp.texts) if (self.scan_all_texts and inp.texts) else [inp.target_text]
        mask_blocks = self._mask_blocks(inp)
        hits_block, hits_mask, unmaskable, cats = [], [], [], set()
        masked = []
        for t in texts:
            new = t
            for r in self.rules:
                if r["_re"].search(new):
                    cats.add(r.get("category", "other"))
                    if r.get("action", "block") == "block":
                        hits_block.append(r["name"])
                    elif mask_blocks:
                        hits_block.append(r["name"])
                        unmaskable.append(r["name"])
                    else:
                        hits_mask.append(r["name"])
                        new = r["_re"].sub(self.mask_template.format(name=r["name"].upper()), new)
            masked.append(new)
        unsafe = bool(hits_block)
        return Verdict(
            stage=self.name,
            unsafe=unsafe,
            score=None,  # rules are hard decisions, not probabilities: they must not rank rows in the metrics
            categories=sorted(cats),
            native_categories=sorted(set(hits_block + hits_mask)),
            masked_texts=masked if (hits_mask and not unsafe and inp.texts) else None,
            reason=",".join(sorted(set(hits_block + hits_mask))),
            extra={"unmaskable": sorted(set(unmaskable))} if unmaskable else {},
        )
