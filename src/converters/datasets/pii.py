"""PII / secrets datasets.

These measure *detection* (should the rules / masking stage fire?), not "block the request":
whether PII in a prompt is blocked, masked or allowed is a policy decision. Rows with any
in-policy span are ``unsafe`` with category ``privacy_pii`` (or ``confidential`` for secrets);
the span labels are kept in ``meta`` for span-level scoring later.
"""

from __future__ import annotations

from typing import Any

from ..base import Record, register
from ..sources import HFDatasetSource, HFFilesSource
from ..templates import TextLabelConverter

SECRET_LABELS = {"secret", "api_key", "password", "pin", "cvv", "http_cookie"}


class _SpanPIIConverter(TextLabelConverter):
    """Span-annotated text: unsafe iff any span survives ``options.ignore_labels`` (and the row is not a
    marked hard negative); categories follow the span labels."""

    def spans(self, r: Record) -> list[dict[str, Any]]:
        ignore = set(self.options.get("ignore_labels", []))
        return [s for s in (r.get("spans") or []) if s.get("label") not in ignore]

    def span_labels(self, r: Record) -> list[str]:
        return sorted({s.get("label") for s in self.spans(r)})

    def is_unsafe(self, r: Record) -> bool:
        return bool(self.spans(r)) and not r.get("is_negative", False)

    def categories(self, r: Record) -> list[str]:
        return sorted({"confidential" if lbl in SECRET_LABELS else "privacy_pii" for lbl in self.span_labels(r)})


@register("tw_pii_bench")
class TwPIIBench(HFDatasetSource, _SpanPIIConverter):
    """lianghsun/tw-PII-bench: the only verified zh-TW PII set (身分證 / 健保卡 / 統編 / 車牌 ... + 40 hard negatives).
    LLM-generated, ~5-10% human checked — test-only. Option ``ignore_labels`` (e.g. [private_date])."""

    output_split = "test"
    hf_id = "lianghsun/tw-PII-bench"
    license = "apache-2.0"
    default_splits = ("short", "mid", "long")
    language = "zh-TW"

    def meta(self, r: Record) -> dict[str, Any]:
        return {"id": r.get("id"), "length_bucket": self.current_split, "category": r.get("category"),
                "labels": self.span_labels(r), "spans": self.spans(r)}


@register("seatbelt_pii_heldout")
class SeatbeltPIIHeldout(HFFilesSource, _SpanPIIConverter):
    """cjhwong/seatbelt-pii-tagger-data held-out files (zh-Hant / zh-Hans / en + regex-generated secrets).
    Viewer is broken -> files loaded directly. Option ``files`` overrides the default held-out files."""

    hf_id = "cjhwong/seatbelt-pii-tagger-data"
    license = "apache-2.0"
    default_splits = ("test",)
    data_files = ("synthetic/heldout.jsonl", "synthetic/rule_secrets/heldout.jsonl")

    def meta(self, r: Record) -> dict[str, Any]:
        return {"labels": self.span_labels(r), "rule": r.get("rule"), "script": r.get("script")}
