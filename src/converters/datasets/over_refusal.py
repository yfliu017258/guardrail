"""Over-refusal / false-positive benchmarks, refusal-judge references and multilingual recall sets."""

from __future__ import annotations

from typing import Any, ClassVar

from ...data_schema import InputType, RefusalLabel
from ..base import Record, register
from ..sources import HFDatasetSource, HFFilesSource, MultiConfigHFSource
from ..templates import PromptResponseConverter, TextLabelConverter, UnsafeOnlyConverter


@register("xstest")
class XSTest(HFDatasetSource, TextLabelConverter):
    """250 safe prompts that *look* unsafe (homonyms, figurative language...) + 200 unsafe contrasts."""

    output_split = "test"
    hf_id = "Paul/XSTest"
    license = "cc-by-4.0"
    default_splits = ("train",)
    text_field = "prompt"
    positive_values = frozenset({"unsafe"})
    NOTE_CATEGORY: ClassVar[dict[str, str]] = {"violence": "violence", "drugs": "crime", "privacy": "privacy_pii"}

    def categories(self, r: Record) -> list[str]:
        return [self.NOTE_CATEGORY.get(str(r.get("note") or ""), "other")]

    def meta(self, r: Record) -> dict[str, Any]:
        return {"type": r.get("type"), "focus": r.get("focus"), "note": r.get("note")}


@register("xstest_responses")
class XSTestResponses(HFDatasetSource, PromptResponseConverter):
    """XSTest v2 completions of 5 models (GPT-4, Llama-2 x2, Mistral x2), each human-labelled full
    compliance / full refusal / partial refusal -> ``refusal_label``: the reference set for validating
    refusal judges. ``label`` is the PROMPT's harm (``contrast_*`` types are the unsafe contrasts):
    XSTest annotates refusal, not response harm, so do not use it for response-guard recall.
    Only response rows by default — the prompts themselves are the ``xstest`` converter."""

    output_split = "test"
    hf_id = "natolambert/xstest-v2-copy"
    license = "cc-by-4.0 (prompts); completions under the generating model's terms (Meta / Mistral / OpenAI)"
    default_splits = ("gpt4", "llama2new", "llama2orig", "mistralguard", "mistralinstruct")
    response_field = "completion"
    refusal_field = "final_label"
    refusal_values: ClassVar[dict[str, RefusalLabel]] = {
        "1_full_compliance": "compliance", "2_full_refusal": "refusal", "3_partial_refusal": "partial"}
    language = "en"
    include_prompts = False
    TYPE_CATEGORY: ClassVar[dict[str, str]] = {"contrast_discr": "hate_harassment", "contrast_privacy": "privacy_pii"}

    @staticmethod
    def _prompt_harm(r: Record) -> str:
        return "unsafe" if str(r["type"]).startswith("contrast_") else "safe"

    def keep(self, r: Record) -> bool:
        return bool(r.get("final_label"))  # unannotated rows are useless as a judge reference

    def prompt_label(self, r: Record) -> str | None:
        return self._prompt_harm(r)

    def response_label(self, r: Record) -> str | None:
        return self._prompt_harm(r)

    def categories(self, r: Record, side: InputType) -> list[str]:
        return [self.TYPE_CATEGORY[r["type"]]] if r["type"] in self.TYPE_CATEGORY else []

    def meta(self, r: Record, side: InputType) -> dict[str, Any]:
        return {"id": r.get("id"), "type": r.get("type"), "model": self.current_split, "agreement": r.get("agreement"),
                "annotation_1": r.get("annotation_1"), "annotation_2": r.get("annotation_2")}


@register("or_bench")
class ORBench(MultiConfigHFSource, TextLabelConverter):
    """``or-bench-hard-1k`` (seemingly toxic, actually benign) + ``or-bench-toxic`` (unsafe).
    The config is the label; option ``configs`` restricts to one of them."""

    output_split = "test"
    hf_id = "bench-llm/or-bench"
    license = "cc-by-4.0"
    default_splits = ("train",)
    default_configs = ("or-bench-hard-1k", "or-bench-toxic")
    text_field = "prompt"
    UNSAFE_CONFIG: ClassVar[str] = "or-bench-toxic"
    CAT: ClassVar[dict[str, str]] = {
        "deception": "crime", "harassment": "hate_harassment", "hate": "hate_harassment", "illegal": "crime",
        "privacy": "privacy_pii", "self-harm": "self_harm", "sexual": "sexual", "violence": "violence",
        "unethical": "other"}

    def configs(self) -> list[str]:
        # only the two known configs carry a label; keep their canonical order so the CSV (and its
        # sha256) does not depend on how the option lists them
        wanted = self.options.get("configs", self.default_configs)
        return [c for c in self.default_configs if c in wanted]

    def is_unsafe(self, r: Record) -> bool:
        return r["_config"] == self.UNSAFE_CONFIG

    def categories(self, r: Record) -> list[str]:
        return [self.CAT.get(str(r.get("category", "")).lower(), "other")]

    def meta(self, r: Record) -> dict[str, Any]:
        return {"config": r["_config"], "category": r.get("category")}


@register("multijail")
class MultiJail(HFDatasetSource, UnsafeOnlyConverter):
    """315 human-translated harmful prompts, parallel columns per language (unsafe-only -> recall).
    Option ``languages: [zh, en]``; zh is Simplified (use ``opencc: s2twp`` for zh-TW)."""

    output_split = "test"
    hf_id = "DAMO-NLP-SG/MultiJail"
    license = "mit"
    default_splits = ("train",)
    text_field = "_text"
    skip_blank_text = True  # not every prompt exists in every language

    def expand(self, record: Record) -> list[Record]:
        return [{**record, "_col": col, "_text": record.get(col)} for col in self.options.get("languages", ["zh", "en"])]

    def lang(self, r: Record) -> str:
        return "zh-CN" if r["_col"] == "zh" else "auto"

    def meta(self, r: Record) -> dict[str, Any]:
        return {"id": r.get("id"), "tags": r.get("tags"), "col": r["_col"]}


@register("rtp_lx")
class RTPLX(HFFilesSource, PromptResponseConverter):
    """ToxicityPrompts/RTP-LX — human-annotated, ODC-BY, with a NATIVE Traditional Chinese file (ZH_Hant).
    Annotations are 1-5 means of 3 annotators per dimension; a row is unsafe when any dimension >= threshold.
    RealToxicityPrompts-style continuations (not chat requests) and skewed toxic -> report as its own slice.
    Options: ``locales: [ZH_Hant, ZH_Hans, EN]``, ``threshold: 3.0``, ``include_completions: true``.
    Check the ZH_Hant rows are Taiwan (not HK) usage before relying on them."""

    hf_id = "ToxicityPrompts/RTP-LX"
    license = "odc-by"
    output_split = "test"
    data_files = ("data/{locale}-00000-of-00001.parquet",)
    group_key = "locale"
    group_option = "locales"
    default_groups = ("ZH_Hant",)
    prompt_field = "Prompt"
    response_field = "Completion"
    DIM_MAP: ClassVar[dict[str, str]] = {
        "bias": "hate_harassment", "identityattack": "hate_harassment", "microaggression": "hate_harassment",
        "insult": "hate_harassment", "selfharm": "self_harm", "sexualcontent": "sexual",
        "violence": "violence", "toxicity": "hate_harassment"}
    LANG: ClassVar[dict[str, str]] = {"ZH_Hant": "zh-TW", "ZH_Hans": "zh-CN", "EN": "en"}
    ANNOTATIONS: ClassVar[dict[str, str]] = {"request": "PromptAnnotations", "response": "CompletionAnnotations"}

    def _scores(self, r: Record, side: InputType) -> dict[str, float]:
        ann = r.get(self.ANNOTATIONS[side]) or {}
        return {str(k).lower(): v for k, v in ann.items() if isinstance(v, (int, float))}

    def _hits(self, r: Record, side: InputType) -> list[str]:
        thr = float(self.options.get("threshold", 3.0))
        return sorted({self.DIM_MAP.get(k, "other") for k, v in self._scores(r, side).items() if v >= thr})

    def wants(self, side: InputType) -> bool:
        if side == "response" and "include_completions" in self.options:  # this set's documented option name
            return bool(self.options["include_completions"])
        return super().wants(side)

    def prompt_label(self, r: Record) -> str | None:
        return "unsafe" if self._hits(r, "request") else "safe"

    def response_label(self, r: Record) -> str | None:
        return "unsafe" if self._hits(r, "response") else "safe"

    def categories(self, r: Record, side: InputType) -> list[str]:
        return self._hits(r, side)

    def lang(self, r: Record, side: InputType) -> str:
        return self.LANG.get(r["_locale"], "auto")

    def meta(self, r: Record, side: InputType) -> dict[str, Any]:
        meta = {"locale": r["_locale"], "max_score": max(self._scores(r, side).values(), default=0.0)}
        return {**meta, "index": r.get("Index")} if side == "request" else meta
