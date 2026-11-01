"""Harmful-content datasets (prompt and/or response labels).

Schemas verified against the HF Hub cards / viewer on 2026-11-01 — see docs/datasets.md.
Pin ``revision`` for anything you report on: several of these repos were updated in 2026.
"""

from __future__ import annotations

from typing import Any, ClassVar

from ...data_schema import InputType
from ...data_schema.taxonomy import (
    AEGIS_MAP,
    AEGIS_NON_VIOLATION,
    BEAVERTAILS_MAP,
    CHINESESAFE_DROP,
    CHINESESAFE_MAP,
    LLAMA_GUARD_MAP,
    OPENAI_MODERATION_MAP,
    map_categories,
)
from ..base import Record, register
from ..sources import HFDatasetSource, HFFilesSource
from ..templates import PromptResponseConverter, TextLabelConverter, is_missing


# --------------------------------------------------------------------------- #
# Aegis 2.0 schema (Aegis 2.0, Nemotron Safety Guard v3)
# --------------------------------------------------------------------------- #
def _aegis_cats(violated: str | None) -> list[str]:
    names = [c.strip() for c in str(violated or "").split(",")]
    names = [n for n in names if n.lower() not in AEGIS_NON_VIOLATION]
    return map_categories(names, AEGIS_MAP)


class _AegisLike(PromptResponseConverter):
    """Aegis 2.0 columns: ``prompt_label`` / ``response_label`` (safe|unsafe) + comma-separated
    ``violated_categories`` shared by both sides. ``REDACTED`` prompts (Suicide-Detection rows,
    not redistributed) are dropped."""

    prompt_label_field = "prompt_label"
    response_label_field = "response_label"

    def keep(self, r: Record) -> bool:
        return self.text(r, self.prompt_field) != "REDACTED"

    def categories(self, r: Record, side: InputType) -> list[str]:
        return _aegis_cats(r.get("violated_categories"))

    def attack(self, r: Record, side: InputType) -> str | None:
        return "jailbreak" if side == "request" and r.get("tag") == "jailbreaking" else None

    def meta(self, r: Record, side: InputType) -> dict[str, Any]:
        prefix = "prompt" if side == "request" else "response"
        return {"id": r.get("id"), "violated_categories": r.get("violated_categories"),
                "label_source": r.get(f"{prefix}_label_source")}


@register("aegis2")
class Aegis2(HFDatasetSource, _AegisLike):
    """NVIDIA Aegis 2.0: human-labelled prompts and responses (en)."""

    hf_id = "nvidia/Aegis-AI-Content-Safety-Dataset-2.0"
    license = "cc-by-4.0"
    default_splits = ("test",)


@register("nemotron_safety_v3")
class NemotronSafetyV3(HFFilesSource, _AegisLike):
    """Multilingual Aegis 2.0 (12 langs). zh is Simplified (``zh-CN``) — combine with ``opencc: s2twp``
    for a machine-converted zh-TW slice. Options: ``languages: [zh, en]`` (repo directory names)."""

    hf_id = "nvidia/Nemotron-Safety-Guard-Dataset-v3"
    license = "cc-by-4.0"
    default_splits = ("test",)
    data_files = ("{lang}/{split}.jsonl",)
    file_split_names: ClassVar[dict[str, str]] = {"validation": "valid"}
    group_key = "lang"
    group_option = "languages"
    default_groups = ("zh",)

    def lang(self, r: Record, side: InputType) -> str:
        # with `opencc` set, BaseConverter.sample() converts the text and re-detects the language
        lang = str(r.get("language") or "")
        return "zh-CN" if lang.startswith("zh") else ("en" if lang == "en" else "other")

    def meta(self, r: Record, side: InputType) -> dict[str, Any]:
        return {**super().meta(r, side), "orig_language": r.get("language")}


# --------------------------------------------------------------------------- #
# WildGuardMix schema (WildGuardTest, PolyGuardPrompts)
# --------------------------------------------------------------------------- #
class _WildGuardLike(PromptResponseConverter):
    """WildGuardMix columns: ``prompt_harm_label`` / ``response_harm_label`` (harmful|unharmful),
    ``response_refusal_label`` (refusal|compliance) -> ``refusal_label``, ``adversarial`` flag."""

    prompt_label_field = "prompt_harm_label"
    response_label_field = "response_harm_label"
    label_values: ClassVar[dict[str, str]] = {"harmful": "unsafe", "unharmful": "safe"}
    refusal_field = "response_refusal_label"

    def attack(self, r: Record, side: InputType) -> str | None:
        return "adversarial" if side == "request" and r.get("adversarial") else None


@register("wildguardtest")
class WildGuardTest(HFDatasetSource, _WildGuardLike):
    """Gated (AI2 form). Prompt-harm + response-harm + response-refusal labels with an adversarial flag.
    Config names follow the repo layout (wildguardtrain / wildguardtest) — re-check after access."""

    hf_id = "allenai/wildguardmix"
    hf_config = "wildguardtest"
    license = "odc-by"
    gated = True
    default_splits = ("test",)

    def categories(self, r: Record, side: InputType) -> list[str]:
        return ["jailbreak"] if side == "request" and r.get("adversarial") else []  # -> "other" otherwise

    def meta(self, r: Record, side: InputType) -> dict[str, Any]:
        meta = {"subcategory": r.get("subcategory"), "adversarial": bool(r.get("adversarial"))}
        return {**meta, "refusal": r.get("response_refusal_label")} if side == "response" else meta


@register("polyguard_prompts")
class PolyGuardPrompts(HFDatasetSource, _WildGuardLike):
    """WildGuardTest translated to 17 languages (zh = Simplified MT). Options: ``languages: [Chinese, English]``.

    Machine-translation artefacts in the TRANSLATED rows are filtered (English rows are the original
    WildGuardTest text and are never filtered — adversarial prompts often quote "I'm sorry, but I can't"):
    a record whose prompt leaked the translator's tags or IS a translator refusal is dropped (it is
    mislabelled harmful); a response sample is dropped (``skipped_response``, the request is kept) when it
    leaked the tags, or reads as a refusal while labelled ``compliance`` — the translator refused, the
    original response did not, and the refusal ground truth would be wrong."""

    hf_id = "ToxicityPrompts/PolyGuardPrompts"
    license = "cc-by-4.0"
    default_splits = ("test",)
    ORIGINAL_LANGUAGE: ClassVar[str] = "English"
    ARTEFACT_MARKERS: ClassVar[tuple[str, ...]] = ("<SOURCE_TEXT>", "</SOURCE_TEXT>", "<TRANSLATION>", "</TRANSLATION>")
    REFUSAL_MARKERS: ClassVar[tuple[str, ...]] = ("抱歉，我无法", "我无法满足", "我无法翻译", "I'm sorry, but I can't",
                                                  "I cannot translate", "I can't translate")
    REFUSAL_WINDOW: ClassVar[int] = 30  # a translator refusal is the whole output: look at its start only
    LANG: ClassVar[dict[str, str]] = {"Chinese": "zh-CN", "English": "en"}

    def _artefact(self, text: str) -> bool:
        return any(m in text for m in self.ARTEFACT_MARKERS)

    def _refusal(self, text: str) -> bool:
        """The text STARTS with a translator refusal (a prompt that merely quotes one later is an attack)."""
        head = text.strip().lstrip("\"'“「").replace("’", "'")[: self.REFUSAL_WINDOW]
        return any(head.startswith(m) for m in self.REFUSAL_MARKERS)

    def keep(self, r: Record) -> bool:
        if r.get("language") not in self.options.get("languages", ["Chinese", "English"]):
            return False
        if r.get("language") == self.ORIGINAL_LANGUAGE:
            return True
        prompt = self.text(r, self.prompt_field)
        return not (self._artefact(prompt) or self._refusal(prompt))

    def keep_side(self, r: Record, side: InputType) -> bool:
        if side == "request" or r.get("language") == self.ORIGINAL_LANGUAGE:
            return True
        response = self.text(r, self.response_field)
        if self._artefact(response):
            return False
        return not (self._refusal(response) and r.get(self.refusal_field) == "compliance")

    def categories(self, r: Record, side: InputType) -> list[str]:
        column = "prompt_categories" if side == "request" else "response_categories"
        return map_categories(str(r.get(column) or "").split(), LLAMA_GUARD_MAP)

    def lang(self, r: Record, side: InputType) -> str:
        return self.LANG.get(r["language"], "other")

    def meta(self, r: Record, side: InputType) -> dict[str, Any]:
        return {"id": r.get("id"), "subcategory": r.get("subcategory")}


# --------------------------------------------------------------------------- #
# other prompt / response sets
# --------------------------------------------------------------------------- #
@register("toxic_chat")
class ToxicChat(HFDatasetSource, TextLabelConverter):
    """Real user prompts (Vicuna demo); low toxic rate -> realistic FPR. License CC-BY-NC-4.0: check with legal."""

    hf_id = "lmsys/toxic-chat"
    hf_config = "toxicchat0124"
    license = "cc-by-nc-4.0"
    default_splits = ("test",)
    text_field = "user_input"
    skip_blank_text = True

    @staticmethod
    def _flag(r: Record, key: str) -> bool:
        return int(r.get(key, 0)) == 1

    def is_unsafe(self, r: Record) -> bool:
        return self._flag(r, "toxicity") or self._flag(r, "jailbreaking")

    def categories(self, r: Record) -> list[str]:
        return ["jailbreak"] if self._flag(r, "jailbreaking") else ["other"]

    def attack(self, r: Record, unsafe: bool) -> str | None:
        return "jailbreak" if self._flag(r, "jailbreaking") else None

    def meta(self, r: Record) -> dict[str, Any]:
        return {"conv_id": r.get("conv_id"), "human_annotation": r.get("human_annotation")}


@register("openai_moderation")
class OpenAIModeration(HFDatasetSource, TextLabelConverter):
    """OpenAI moderation eval set: one 0/1 column per category; a null flag means "not annotated"
    (per the dataset README), never 0. A row is ``unsafe`` when any flag is 1 and ``safe`` only when
    EVERY flag is annotated 0; a row with no 1 and some null flags has no ground truth and is
    skipped. ``keep_partial: false`` also drops partially annotated unsafe rows. Unannotated
    categories are listed in ``meta.unannotated``."""

    output_split = "test"
    hf_id = "mmathys/openai-moderation-api-evaluation"
    license = "mit"
    default_splits = ("train",)
    text_field = "prompt"

    @staticmethod
    def _flags(r: Record) -> dict[str, Any]:
        return {k: r.get(k.upper()) for k in OPENAI_MODERATION_MAP}

    @classmethod
    def _unannotated(cls, r: Record) -> list[str]:
        return [k.upper() for k, v in cls._flags(r).items() if is_missing(v)]

    def keep(self, r: Record) -> bool:
        if not self._unannotated(r):
            return True
        return self.is_unsafe(r) and bool(self.options.get("keep_partial", True))

    def is_unsafe(self, r: Record) -> bool:
        return any(v == 1 for v in self._flags(r).values())

    def categories(self, r: Record) -> list[str]:
        return map_categories([k for k, v in self._flags(r).items() if v == 1], OPENAI_MODERATION_MAP)

    def meta(self, r: Record) -> dict[str, Any]:
        meta: dict[str, Any] = {"flags": {k: v for k, v in self._flags(r).items() if not is_missing(v)}}
        return {**meta, "unannotated": self._unannotated(r)} if self._unannotated(r) else meta


@register("beavertails")
class BeaverTails(HFDatasetSource, PromptResponseConverter):
    """QA-pair (response) labels only. License CC-BY-NC-4.0."""

    output_split = "test"
    hf_id = "PKU-Alignment/BeaverTails"
    license = "cc-by-nc-4.0"
    default_splits = ("30k_test",)
    strip_text = False  # keep the text byte-identical to the source (sample ids hash it)

    @staticmethod
    def _hits(r: Record) -> list[str]:
        return [k for k, v in (r.get("category") or {}).items() if v]

    def response_label(self, r: Record) -> str | None:
        return "safe" if r.get("is_safe", True) else "unsafe"

    def categories(self, r: Record, side: InputType) -> list[str]:
        return map_categories(self._hits(r), BEAVERTAILS_MAP)

    def meta(self, r: Record, side: InputType) -> dict[str, Any]:
        return {"orig_categories": self._hits(r)}


@register("chinesesafe")
class ChineseSafe(HFDatasetSource, TextLabelConverter):
    """zh-CN binary set. ``政治错误`` rows are DROPPED (PRC political norms); noisy labels — sample & review.
    ``违规`` is unsafe and any other non-empty label safe (``negative_values = None``). License CC-BY-NC-4.0."""

    hf_id = "SUSTech/ChineseSafe"
    license = "cc-by-nc-4.0"
    default_splits = ("test",)
    positive_values = frozenset({"违规"})
    negative_values = None
    language = "zh-CN"

    def keep(self, r: Record) -> bool:
        return (r.get("subject") or "") not in CHINESESAFE_DROP

    def categories(self, r: Record) -> list[str]:
        return [CHINESESAFE_MAP.get(r.get("subject") or "", "other")]

    def attack(self, r: Record, unsafe: bool) -> str | None:
        return "obfuscation" if r.get("subject") == "变体词" else None  # variant spellings, whatever the label

    def meta(self, r: Record) -> dict[str, Any]:
        return {"subject": r.get("subject") or ""}
