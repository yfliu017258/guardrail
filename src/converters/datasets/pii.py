"""PII / secrets datasets: a prompt with in-policy spans -> ``mask``, a clean prompt -> ``pass``.

The guardrail server lets such a request through with the sensitive data masked
(``GUARDRAIL_INTERVENED``), so the default expectation is ``mask``. Option
``pii_expected: block`` is for a policy that rejects such prompts instead. Spans whose label is
in ``ignore_labels`` (e.g. ``private_date``) are out of policy and do not count; marked hard
negatives are ``pass`` whatever their spans say.
"""

from __future__ import annotations

from typing import Any, ClassVar

from ...data_schema import Expected
from ..base import Record, register
from ..sources import HFDatasetSource, HFFilesSource
from ..templates import TextLabelConverter

PII_EXPECTED = ("mask", "block")


class _SpanPIIConverter(TextLabelConverter):
    """Span-annotated text: ``pii_expected`` (mask) iff any span survives ``options.ignore_labels`` and the
    row is not a marked hard negative (``is_negative``); ``pass`` otherwise."""

    positive_expected: ClassVar[Expected] = "mask"

    def check_options(self) -> None:
        super().check_options()
        if self.options.get("pii_expected", "mask") not in PII_EXPECTED:
            raise ValueError(f"{self.name}: pii_expected must be one of {PII_EXPECTED}, "
                             f"got {self.options['pii_expected']!r}")

    def spans(self, r: Record) -> list[dict[str, Any]]:
        ignore = set(self.options.get("ignore_labels", []))
        return [s for s in (r.get("spans") or []) if s.get("label") not in ignore]

    def is_positive(self, r: Record) -> bool:
        return bool(self.spans(r)) and not r.get("is_negative", False)

    def expected(self, r: Record) -> Expected:
        return self.options.get("pii_expected", self.positive_expected) if self.is_positive(r) else "pass"


@register("tw_pii_bench")
class TwPIIBench(HFDatasetSource, _SpanPIIConverter):
    """lianghsun/tw-PII-bench: the only verified zh-TW PII set (身分證 / 健保卡 / 統編 / 車牌 ... + 40 hard negatives).
    LLM-generated, ~5-10% human checked — test-only. Option ``ignore_labels`` (e.g. [private_date])."""

    output_split = "test"
    hf_id = "lianghsun/tw-PII-bench"
    license = "apache-2.0"
    default_splits = ("short", "mid", "long")


@register("seatbelt_pii_heldout")
class SeatbeltPIIHeldout(HFFilesSource, _SpanPIIConverter):
    """cjhwong/seatbelt-pii-tagger-data held-out files (zh-Hant / zh-Hans / en + regex-generated secrets).
    Viewer is broken -> files loaded directly, ONE ``load_dataset`` per file: the rule_secrets file has an
    extra ``rule`` column, and one multi-file split is cast to the first file's columns (CastError).
    Option ``files`` overrides the default held-out files; rows are tagged ``_file``."""

    hf_id = "cjhwong/seatbelt-pii-tagger-data"
    license = "apache-2.0"
    default_splits = ("test",)
    data_files = ("{file}",)
    group_key = "file"
    group_option = "files"
    default_groups = ("synthetic/heldout.jsonl", "synthetic/rule_secrets/heldout.jsonl")

    def files_for(self, split: str, group: str | None) -> list[str]:
        return [str(group)]  # ``options.files`` lists the groups here, it does not replace the pattern
