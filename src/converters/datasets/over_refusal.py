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
