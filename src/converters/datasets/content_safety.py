"""Harmful-content datasets: harmful prompt -> ``block``, benign prompt -> ``pass``.

Only the prompt and the PROMPT's label are converted; sets that also label model responses
(Aegis 2.0, WildGuardMix, PolyGuard) contribute their request side only.

Schemas verified against the HF Hub cards / viewer on 2026-11-01 — see docs/datasets.md.
Pin ``revision`` for anything you report on: several of these repos were updated in 2026.
"""

from __future__ import annotations

from typing import Any, ClassVar

from ...data_schema import Expected
from ..base import Record, register
from ..sources import HFDatasetSource, HFFilesSource
from ..templates import PromptLabelConverter, TextLabelConverter, is_missing


# --------------------------------------------------------------------------- #
# Aegis 2.0 schema (Aegis 2.0, Nemotron Safety Guard v3)
# --------------------------------------------------------------------------- #
class _AegisLike(PromptLabelConverter):
    """Aegis 2.0 columns: ``prompt`` + ``prompt_label`` (safe|unsafe). ``REDACTED`` prompts
    (Suicide-Detection rows, not redistributed) are dropped."""

    prompt_label_field = "prompt_label"
    label_values: ClassVar[dict[str, Expected]] = {"unsafe": "block", "safe": "pass"}

    def keep(self, r: Record) -> bool:
        return self.text(r) != "REDACTED"


@register("aegis2")
class Aegis2(HFDatasetSource, _AegisLike):
    """NVIDIA Aegis 2.0: human-labelled prompts (en)."""

    hf_id = "nvidia/Aegis-AI-Content-Safety-Dataset-2.0"
    license = "cc-by-4.0"
    default_splits = ("test",)


@register("nemotron_safety_v3")
class NemotronSafetyV3(HFFilesSource, _AegisLike):
    """Multilingual Aegis 2.0 (12 langs). zh is Simplified — combine with ``opencc: s2twp`` for a
    machine-converted zh-TW set. Options: ``languages: [zh, en]`` (repo directory names)."""

    hf_id = "nvidia/Nemotron-Safety-Guard-Dataset-v3"
    license = "cc-by-4.0"
    default_splits = ("test",)
    data_files = ("{lang}/{split}.jsonl",)
    file_split_names: ClassVar[dict[str, str]] = {"validation": "valid"}
    group_key = "lang"
    group_option = "languages"
    default_groups = ("zh",)


# --------------------------------------------------------------------------- #
# WildGuardMix schema (WildGuardTest, PolyGuardPrompts)
# --------------------------------------------------------------------------- #
class _WildGuardLike(PromptLabelConverter):
    """WildGuardMix columns: ``prompt`` + ``prompt_harm_label`` (harmful|unharmful); adversarial
    (jailbreak-wrapped) prompts carry the harm label of the wrapped request."""

    prompt_label_field = "prompt_harm_label"
    label_values: ClassVar[dict[str, Expected]] = {"harmful": "block", "unharmful": "pass"}


@register("wildguardtest")
class WildGuardTest(HFDatasetSource, _WildGuardLike):
    """Gated (AI2 form). Human prompt-harm labels incl. adversarial prompts. Config names follow the
    repo layout (wildguardtrain / wildguardtest) — re-check after access."""

    hf_id = "allenai/wildguardmix"
    hf_config = "wildguardtest"
    license = "odc-by"
    gated = True
    default_splits = ("test",)


@register("polyguard_prompts")
class PolyGuardPrompts(HFDatasetSource, _WildGuardLike):
    """WildGuardTest translated to 17 languages (zh = Simplified MT). Option ``languages: [Chinese, English]``.

    Machine-translation artefacts in TRANSLATED prompts are filtered (English rows are the original
    WildGuardTest text and are never filtered — adversarial prompts often quote "I'm sorry, but I can't"):
    a prompt that leaked the translator's tags or IS a translator refusal is dropped (``skipped``), since
    its harm label describes a text it no longer contains."""

    hf_id = "ToxicityPrompts/PolyGuardPrompts"
    license = "cc-by-4.0"
    default_splits = ("test",)
    ORIGINAL_LANGUAGE: ClassVar[str] = "English"
    ARTEFACT_MARKERS: ClassVar[tuple[str, ...]] = ("<SOURCE_TEXT>", "</SOURCE_TEXT>", "<TRANSLATION>", "</TRANSLATION>")
    REFUSAL_MARKERS: ClassVar[tuple[str, ...]] = ("抱歉，我无法", "我无法满足", "我无法翻译", "I'm sorry, but I can't",
                                                  "I cannot translate", "I can't translate")
    REFUSAL_WINDOW: ClassVar[int] = 30  # a translator refusal is the whole output: look at its start only

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
        prompt = self.text(r)
        return not (self._artefact(prompt) or self._refusal(prompt))


# --------------------------------------------------------------------------- #
# single-label prompt sets
# --------------------------------------------------------------------------- #
@register("toxic_chat")
class ToxicChat(HFDatasetSource, TextLabelConverter):
    """Real user prompts (Vicuna demo): toxic OR jailbreaking -> block. Low toxic rate -> realistic FPR.
    License CC-BY-NC-4.0: check with legal."""

    hf_id = "lmsys/toxic-chat"
    hf_config = "toxicchat0124"
    license = "cc-by-nc-4.0"
    default_splits = ("test",)
    text_field = "user_input"
    skip_blank_text = True

    def is_positive(self, r: Record) -> bool:
        return int(r.get("toxicity", 0)) == 1 or int(r.get("jailbreaking", 0)) == 1


@register("openai_moderation")
class OpenAIModeration(HFDatasetSource, TextLabelConverter):
    """OpenAI moderation eval set: one 0/1 column per category; a null flag means "not annotated"
    (per the dataset README), never 0. A row is ``block`` when any flag is 1 and ``pass`` only when
    EVERY flag is annotated 0; a row with no 1 and some null flags has no ground truth and is
    skipped. ``keep_partial: false`` also drops partially annotated ``block`` rows."""

    output_split = "test"
    hf_id = "mmathys/openai-moderation-api-evaluation"
    license = "mit"
    default_splits = ("train",)
    text_field = "prompt"
    FLAGS: ClassVar[tuple[str, ...]] = ("S", "H", "V", "HR", "SH", "S3", "H2", "V2")

    def _flags(self, r: Record) -> dict[str, Any]:
        return {k: r.get(k) for k in self.FLAGS}

    def keep(self, r: Record) -> bool:
        if not any(is_missing(v) for v in self._flags(r).values()):
            return True
        return self.is_positive(r) and bool(self.options.get("keep_partial", True))

    def is_positive(self, r: Record) -> bool:
        return any(v == 1 for v in self._flags(r).values())


@register("chinesesafe")
class ChineseSafe(HFDatasetSource, TextLabelConverter):
    """zh-CN binary set: ``违规`` -> block, any other non-empty label -> pass (``negative_values = None``).
    ``政治错误`` rows are DROPPED (PRC political norms); noisy labels — sample & review. License CC-BY-NC-4.0."""

    hf_id = "SUSTech/ChineseSafe"
    license = "cc-by-nc-4.0"
    default_splits = ("test",)
    positive_values = frozenset({"违规"})
    negative_values = None
    DROP_SUBJECTS: ClassVar[frozenset[str]] = frozenset({"政治错误"})

    def keep(self, r: Record) -> bool:
        return (r.get("subject") or "") not in self.DROP_SUBJECTS
