"""Over-refusal / false-positive benchmarks and multilingual recall sets."""

from __future__ import annotations

from typing import Any, Iterable

from ..base import BaseConverter, register


@register("xstest")
class XSTest(BaseConverter):
    """250 safe prompts that *look* unsafe (homonyms, figurative language...) + 200 unsafe contrasts."""

    output_split = "test"
    hf_id = "Paul/XSTest"
    license = "cc-by-4.0"
    default_splits = ("train",)

    def convert(self, r, split, idx):
        unsafe = r.get("label") == "unsafe"
        cat_hint = {"violence": "violence", "drugs": "crime", "privacy": "privacy_pii"}.get(str(r.get("note") or ""), "other")
        return self.sample(split="test", prompt=r["prompt"], label="unsafe" if unsafe else "safe",
                           categories=[cat_hint] if unsafe else [],
                           meta={"type": r.get("type"), "focus": r.get("focus"), "note": r.get("note")})


@register("or_bench")
class ORBench(BaseConverter):
    """``or-bench-hard-1k`` (seemingly toxic, actually benign) + ``or-bench-toxic`` (unsafe)."""

    output_split = "test"
    hf_id = "bench-llm/or-bench"
    license = "cc-by-4.0"
    default_splits = ("train",)
    CAT = {"deception": "crime", "harassment": "hate_harassment", "hate": "hate_harassment", "illegal": "crime",
           "privacy": "privacy_pii", "self-harm": "self_harm", "sexual": "sexual", "violence": "violence",
           "unethical": "other"}

    def load_raw(self, split: str) -> Iterable[dict[str, Any]]:
        from datasets import load_dataset

        for cfg, unsafe in (("or-bench-hard-1k", False), ("or-bench-toxic", True)):
            if cfg in self.options.get("configs", ["or-bench-hard-1k", "or-bench-toxic"]):
                for r in load_dataset(self.hf_id, cfg, split=split, revision=self.revision):
                    yield {**r, "_unsafe": unsafe, "_config": cfg}

    def convert(self, r, split, idx):
        cat = self.CAT.get(str(r.get("category", "")).lower(), "other")
        cat = cat if cat in ("crime", "hate_harassment", "privacy_pii", "self_harm", "sexual", "violence") else "other"
        return self.sample(split="test", prompt=r["prompt"], label="unsafe" if r["_unsafe"] else "safe",
                           categories=[cat] if r["_unsafe"] else [], meta={"config": r["_config"], "category": r.get("category")})


@register("multijail")
class MultiJail(BaseConverter):
    """315 human-translated harmful prompts, parallel columns per language (unsafe-only -> recall).
    Option ``languages: [zh, en]``; zh is Simplified (use ``opencc: s2twp`` for zh-TW)."""

    output_split = "test"
    hf_id = "DAMO-NLP-SG/MultiJail"
    license = "mit"
    default_splits = ("train",)

    def convert(self, r, split, idx):
        out = []
        for lang in self.options.get("languages", ["zh", "en"]):
            text = (r.get(lang) or "").strip()
            if text:
                out.append(self.sample(split="test", prompt=text, label="unsafe", categories=["other"],
                                       language="auto" if (self.options.get("opencc") or lang != "zh") else "zh-CN",
                                       meta={"id": r.get("id"), "tags": r.get("tags"), "col": lang}))
        return out or None


@register("rtp_lx")
class RTPLX(BaseConverter):
    """ToxicityPrompts/RTP-LX — human-annotated, ODC-BY, with a NATIVE Traditional Chinese file (ZH_Hant).
    Annotations are 1-5 means of 3 annotators per dimension; a row is unsafe when any dimension >= threshold.
    RealToxicityPrompts-style continuations (not chat requests) and skewed toxic -> report as its own slice.
    Options: ``locales: [ZH_Hant, ZH_Hans, EN]``, ``threshold: 3.0``, ``include_completions: true``.
    Check the ZH_Hant rows are Taiwan (not HK) usage before relying on them."""

    hf_id = "ToxicityPrompts/RTP-LX"
    license = "odc-by"
    output_split = "test"
    DIM_MAP = {"bias": "hate_harassment", "identityattack": "hate_harassment", "microaggression": "hate_harassment",
               "insult": "hate_harassment", "selfharm": "self_harm", "sexualcontent": "sexual",
               "violence": "violence", "toxicity": "hate_harassment"}
    LANG = {"ZH_Hant": "zh-TW", "ZH_Hans": "zh-CN", "EN": "en"}

    def load_raw(self, split):
        from datasets import load_dataset

        for loc in self.options.get("locales", ["ZH_Hant"]):
            for r in load_dataset(self.hf_id, data_files={"test": f"data/{loc}-00000-of-00001.parquet"},
                                  split="test", revision=self.revision):
                yield {**r, "_locale": loc}

    def _label(self, ann: dict | None) -> tuple[str, list[str], float]:
        thr = float(self.options.get("threshold", 3.0))
        ann = {str(k).lower(): v for k, v in (ann or {}).items() if isinstance(v, (int, float))}
        hits = sorted({self.DIM_MAP.get(k, "other") for k, v in ann.items() if v >= thr})
        return ("unsafe" if hits else "safe"), hits, max(ann.values(), default=0.0)

    def convert(self, r, split, idx):
        lang = self.LANG.get(r["_locale"], "auto")
        out = []
        prompt = (r.get("Prompt") or "").strip()
        label, cats, score = self._label(r.get("PromptAnnotations"))
        if prompt:
            out.append(self.sample(split="test", prompt=prompt, label=label, categories=cats, language=lang,
                                   meta={"locale": r["_locale"], "max_score": score, "index": r.get("Index")}))
        comp = (r.get("Completion") or "").strip()
        if comp and prompt and self.options.get("include_completions", True):
            label, cats, score = self._label(r.get("CompletionAnnotations"))
            out.append(self.sample(split="test", input_type="response", prompt=prompt, response=comp, label=label,
                                   categories=cats, language=lang, meta={"locale": r["_locale"], "max_score": score}))
        return out or None
