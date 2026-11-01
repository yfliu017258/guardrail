"""Harmful-content datasets (prompt and/or response labels).

Schemas verified against the HF Hub cards / viewer on 2026-11-01 — see docs/datasets.md.
Pin ``revision`` for anything you report on: several of these repos were updated in 2026.
"""

from __future__ import annotations

from typing import Any, Iterable

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
from ..base import BaseConverter, register


def _aegis_cats(violated: str | None) -> list[str]:
    names = [c.strip() for c in str(violated or "").split(",")]
    names = [n for n in names if n.lower() not in AEGIS_NON_VIOLATION]
    return map_categories(names, AEGIS_MAP)


class _AegisLike(BaseConverter):
    """One raw row -> up to two samples: the prompt (request) and the response (response)."""

    def _rows(self, r: dict[str, Any], split: str, idx: int, extra_meta: dict | None = None) -> list:
        prompt = (r.get("prompt") or "").strip()
        if not prompt or prompt == "REDACTED":  # Suicide-Detection rows are not redistributed
            return []
        out = []
        cats = _aegis_cats(r.get("violated_categories"))
        meta = {"id": r.get("id"), "violated_categories": r.get("violated_categories"), **(extra_meta or {})}
        attack = "jailbreak" if r.get("tag") == "jailbreaking" else None
        if self.options.get("include_prompts", True) and r.get("prompt_label") in ("safe", "unsafe"):
            unsafe = r["prompt_label"] == "unsafe"
            out.append(self.sample(split=split, input_type="request", prompt=prompt, label=r["prompt_label"],
                                   categories=(cats or ["other"]) if unsafe else [], attack_type=attack,
                                   language=self._lang(r), meta={**meta, "label_source": r.get("prompt_label_source")}))
        resp = (r.get("response") or "").strip()
        if self.options.get("include_responses", True) and resp and r.get("response_label") in ("safe", "unsafe"):
            unsafe = r["response_label"] == "unsafe"
            out.append(self.sample(split=split, input_type="response", prompt=prompt, response=resp,
                                   label=r["response_label"], categories=(cats or ["other"]) if unsafe else [],
                                   language=self._lang(r), meta={**meta, "label_source": r.get("response_label_source")}))
        return out

    def _lang(self, r: dict) -> str:
        return "auto"


@register("aegis2")
class Aegis2(_AegisLike):
    hf_id = "nvidia/Aegis-AI-Content-Safety-Dataset-2.0"
    license = "cc-by-4.0"
    default_splits = ("test",)

    def convert(self, r, split, idx):
        return self._rows(r, split, idx)


@register("nemotron_safety_v3")
class NemotronSafetyV3(_AegisLike):
    """Multilingual Aegis 2.0 (12 langs). zh is Simplified (``zh-CN``) — combine with ``opencc: s2twp``
    for a machine-converted zh-TW slice. Options: ``languages: [zh, en]`` (repo directory names)."""

    hf_id = "nvidia/Nemotron-Safety-Guard-Dataset-v3"
    license = "cc-by-4.0"
    default_splits = ("test",)

    def load_raw(self, split: str) -> Iterable[dict[str, Any]]:
        from datasets import load_dataset

        fname = {"test": "test", "validation": "valid", "train": "train"}[split]
        langs = self.options.get("languages", ["zh"])
        files = [f"{lang}/{fname}.jsonl" for lang in langs]
        return load_dataset(self.hf_id, data_files={split: files}, split=split, revision=self.revision)

    def _lang(self, r: dict) -> str:
        lang = str(r.get("language") or "")
        return "zh-CN" if lang.startswith("zh") else ("en" if lang == "en" else "other")

    def convert(self, r, split, idx):
        # with `opencc` set, BaseConverter.sample() converts the text and re-detects the language
        return self._rows(r, split, idx, {"orig_language": r.get("language")})


@register("toxic_chat")
class ToxicChat(BaseConverter):
    """Real user prompts (Vicuna demo); low toxic rate -> realistic FPR. License CC-BY-NC-4.0: check with legal."""

    hf_id = "lmsys/toxic-chat"
    hf_config = "toxicchat0124"
    license = "cc-by-nc-4.0"
    default_splits = ("test",)

    def convert(self, r, split, idx):
        text = (r.get("user_input") or "").strip()
        if not text:
            return None
        toxic, jb = int(r.get("toxicity", 0)) == 1, int(r.get("jailbreaking", 0)) == 1
        cats = (["jailbreak"] if jb else []) + (["other"] if toxic and not jb else [])
        return self.sample(split=split, prompt=text, label="unsafe" if (toxic or jb) else "safe",
                           categories=cats, attack_type="jailbreak" if jb else None,
                           meta={"conv_id": r.get("conv_id"), "human_annotation": r.get("human_annotation")})


@register("wildguardtest")
class WildGuardTest(BaseConverter):
    """Gated (AI2 form). Prompt-harm + response-harm labels with an adversarial flag.
    Config names follow the repo layout (wildguardtrain / wildguardtest) — re-check after access."""

    hf_id = "allenai/wildguardmix"
    hf_config = "wildguardtest"
    license = "odc-by"
    gated = True
    default_splits = ("test",)

    def convert(self, r, split, idx):
        out = []
        prompt = (r.get("prompt") or "").strip()
        adv = bool(r.get("adversarial"))
        sub = r.get("subcategory")
        cat = "jailbreak" if adv else "other"
        meta = {"subcategory": sub, "adversarial": adv}
        if r.get("prompt_harm_label") in ("harmful", "unharmful") and prompt:
            unsafe = r["prompt_harm_label"] == "harmful"
            out.append(self.sample(split=split, prompt=prompt, label="unsafe" if unsafe else "safe",
                                   categories=[cat] if unsafe else [], attack_type="adversarial" if adv else None, meta=meta))
        resp = (r.get("response") or "").strip()
        if self.options.get("include_responses", True) and resp and r.get("response_harm_label") in ("harmful", "unharmful"):
            unsafe = r["response_harm_label"] == "harmful"
            out.append(self.sample(split=split, input_type="response", prompt=prompt, response=resp,
                                   label="unsafe" if unsafe else "safe", categories=["other"] if unsafe else [],
                                   meta={**meta, "refusal": r.get("response_refusal_label")}))
        return out or None


@register("polyguard_prompts")
class PolyGuardPrompts(BaseConverter):
    """WildGuardTest translated to 17 languages (zh = Simplified MT). Options: ``languages: [Chinese, English]``.
    Drops rows where the *translator* refused (prompt is a refusal) — they are mislabelled harmful."""

    hf_id = "ToxicityPrompts/PolyGuardPrompts"
    license = "cc-by-4.0"
    default_splits = ("test",)
    REFUSAL_MARKERS = ("抱歉，我无法", "我无法满足", "I'm sorry, but I can't", "<SOURCE_TEXT>", "<TRANSLATION>")

    def convert(self, r, split, idx):
        langs = self.options.get("languages", ["Chinese", "English"])
        if r.get("language") not in langs:
            return None
        prompt = (r.get("prompt") or "").strip()
        if not prompt or any(m in prompt for m in self.REFUSAL_MARKERS):
            return None
        lang = {"Chinese": "zh-CN", "English": "en"}.get(r["language"], "other")
        meta = {"id": r.get("id"), "subcategory": r.get("subcategory")}
        out = []
        if r.get("prompt_harm_label") in ("harmful", "unharmful"):
            unsafe = r["prompt_harm_label"] == "harmful"
            cats = map_categories(str(r.get("prompt_categories") or "").split(), LLAMA_GUARD_MAP) if unsafe else []
            out.append(self.sample(split=split, prompt=prompt, label="unsafe" if unsafe else "safe",
                                   categories=(cats or ["other"]) if unsafe else [], language=lang,
                                   attack_type="adversarial" if r.get("adversarial") else None, meta=meta))
        resp = (r.get("response") or "").strip()
        if self.options.get("include_responses", True) and resp and r.get("response_harm_label") in ("harmful", "unharmful"):
            unsafe = r["response_harm_label"] == "harmful"
            cats = map_categories(str(r.get("response_categories") or "").split(), LLAMA_GUARD_MAP) if unsafe else []
            out.append(self.sample(split=split, input_type="response", prompt=prompt, response=resp,
                                   label="unsafe" if unsafe else "safe", categories=(cats or ["other"]) if unsafe else [],
                                   language="auto" if self.options.get("opencc") else lang, meta=meta))
        return out or None


@register("openai_moderation")
class OpenAIModeration(BaseConverter):
    output_split = "test"
    hf_id = "mmathys/openai-moderation-api-evaluation"
    license = "mit"
    default_splits = ("train",)

    def convert(self, r, split, idx):
        flags = {k: r.get(k.upper()) for k in OPENAI_MODERATION_MAP}
        if any(v is None for v in flags.values()) and not self.options.get("keep_partial", True):
            return None
        hit = [k for k, v in flags.items() if v == 1]
        return self.sample(split="test", prompt=r["prompt"], label="unsafe" if hit else "safe",
                           categories=map_categories(hit, OPENAI_MODERATION_MAP) if hit else [],
                           meta={"flags": {k: v for k, v in flags.items() if v is not None}})


@register("beavertails")
class BeaverTails(BaseConverter):
    """QA-pair (response) labels. License CC-BY-NC-4.0."""

    output_split = "test"
    hf_id = "PKU-Alignment/BeaverTails"
    license = "cc-by-nc-4.0"
    default_splits = ("30k_test",)

    def convert(self, r, split, idx):
        cats = r.get("category") or {}
        hit = [k for k, v in cats.items() if v]
        unsafe = not r.get("is_safe", True)
        return self.sample(split="test", input_type="response", prompt=r["prompt"], response=r["response"],
                           label="unsafe" if unsafe else "safe",
                           categories=(map_categories(hit, BEAVERTAILS_MAP) or ["other"]) if unsafe else [],
                           meta={"orig_categories": hit})


@register("chinesesafe")
class ChineseSafe(BaseConverter):
    """zh-CN binary set. ``政治错误`` rows are DROPPED (PRC political norms); noisy labels — sample & review.
    License CC-BY-NC-4.0."""

    hf_id = "SUSTech/ChineseSafe"
    license = "cc-by-nc-4.0"
    default_splits = ("test",)

    def convert(self, r, split, idx):
        subject = r.get("subject") or ""
        if subject in CHINESESAFE_DROP:
            return None
        unsafe = r.get("label") == "违规"
        return self.sample(split=split, prompt=r["text"], label="unsafe" if unsafe else "safe",
                           categories=([CHINESESAFE_MAP.get(subject, "other")] if unsafe else []),
                           attack_type="obfuscation" if subject == "变体词" else None,
                           language="auto" if self.options.get("opencc") else "zh-CN", meta={"subject": subject})

