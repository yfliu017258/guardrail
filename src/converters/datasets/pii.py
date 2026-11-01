"""PII / secrets datasets.

These measure *detection* (should the rules / masking stage fire?), not "block the request":
whether PII in a prompt is blocked, masked or allowed is a policy decision. Rows with any
in-policy span are ``unsafe`` with category ``privacy_pii`` (or ``confidential`` for secrets);
the span labels are kept in ``meta`` for span-level scoring later.
"""

from __future__ import annotations

from ..base import BaseConverter, register

SECRET_LABELS = {"secret", "api_key", "password", "pin", "cvv", "http_cookie"}


@register("tw_pii_bench")
class TwPIIBench(BaseConverter):
    """lianghsun/tw-PII-bench: the only verified zh-TW PII set (身分證 / 健保卡 / 統編 / 車牌 ... + 40 hard negatives).
    LLM-generated, ~5-10% human checked — test-only. Option ``ignore_labels`` (e.g. [private_date])."""

    output_split = "test"
    hf_id = "lianghsun/tw-PII-bench"
    license = "apache-2.0"
    default_splits = ("short", "mid", "long")

    def convert(self, r, split, idx):
        ignore = set(self.options.get("ignore_labels", []))
        spans = [s for s in (r.get("spans") or []) if s.get("label") not in ignore]
        unsafe = bool(spans) and not r.get("is_negative", False)
        labels = sorted({s.get("label") for s in spans})
        cats = sorted({"confidential" if lbl in SECRET_LABELS else "privacy_pii" for lbl in labels}) if unsafe else []
        return self.sample(split="test", prompt=r["text"], label="unsafe" if unsafe else "safe", categories=cats,
                           language="zh-TW", meta={"id": r.get("id"), "length_bucket": split, "category": r.get("category"),
                                                   "labels": labels, "spans": spans})


@register("seatbelt_pii_heldout")
class SeatbeltPIIHeldout(BaseConverter):
    """cjhwong/seatbelt-pii-tagger-data held-out files (zh-Hant / zh-Hans / en + regex-generated secrets).
    Viewer is broken -> files loaded directly. Option ``files`` overrides the default held-out files."""

    hf_id = "cjhwong/seatbelt-pii-tagger-data"
    license = "apache-2.0"
    default_splits = ("test",)

    def load_raw(self, split):
        from datasets import load_dataset

        files = self.options.get("files", ["synthetic/heldout.jsonl", "synthetic/rule_secrets/heldout.jsonl"])
        return load_dataset(self.hf_id, data_files={"test": files}, split="test", revision=self.revision)

    def convert(self, r, split, idx):
        spans = r.get("spans") or []
        labels = sorted({s.get("label") for s in spans})
        unsafe = bool(spans)
        cats = sorted({"confidential" if lbl in SECRET_LABELS else "privacy_pii" for lbl in labels}) if unsafe else []
        return self.sample(split="test", prompt=r["text"], label="unsafe" if unsafe else "safe", categories=cats,
                           meta={"labels": labels, "rule": r.get("rule"), "script": r.get("script")})
