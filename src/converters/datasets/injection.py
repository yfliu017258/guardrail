"""Prompt-injection / jailbreak datasets.

Keep two label axes apart (see docs/datasets.md): *attack* (prompt_injection / jailbreak,
recorded in ``attack_type``) vs *harm category*. Several PI datasets label a plain harmful
request without an override attempt as benign — those rows are ``safe`` here only with
respect to injection, so do not mix them into content-safety recall numbers.
"""

from __future__ import annotations

from typing import Any, Iterable

from ...data_schema.taxonomy import HARMBENCH_MAP, JBB_MAP
from ..base import BaseConverter, register


@register("pi_eval_suite")
class PIEvalSuite(BaseConverter):
    """Horizon-Labs/prompt-injection-eval-suite: 16 configs, unified schema (text, label, source, kind, lang).
    Option ``configs`` picks subsets; ``synthetic_*`` are in-distribution for the publisher's own model."""

    hf_id = "Horizon-Labs/prompt-injection-eval-suite"
    license = "mixed (per config: MIT / Apache-2.0 / CC-BY-4.0 / CC-BY-SA-4.0)"
    default_splits = ("test",)
    DEFAULT_CONFIGS = ["notinject", "xstest", "orbench_hard", "deepset_test", "jackhhao_test", "boundary_pairs_test",
                       "neuralchemy_test", "bipia", "piarena", "llmail_phase2"]

    def load_raw(self, split: str) -> Iterable[dict[str, Any]]:
        from datasets import load_dataset

        for cfg in self.options.get("configs", self.DEFAULT_CONFIGS):
            for r in load_dataset(self.hf_id, cfg, split=split, revision=self.revision):
                yield {**r, "_config": cfg}

    def convert(self, r, split, idx):
        text = (r.get("text") or "").strip()
        if not text:
            return None
        unsafe = int(r.get("label", 0)) == 1
        kind = r.get("kind") or "direct"
        src = str(r.get("source") or r["_config"])
        is_jb = "jailbreak" in src or r["_config"] in ("jackhhao_test", "simsonsun_jailbreaks")
        cat = "jailbreak" if is_jb else "prompt_injection"
        return self.sample(split=split, prompt=text, label="unsafe" if unsafe else "safe",
                           categories=[cat] if unsafe else [],
                           attack_type=(("indirect" if kind == "indirect" else ("jailbreak" if is_jb else "direct")) if unsafe else None),
                           language=r.get("lang") if r.get("lang") in ("en", "zh-TW", "zh-CN") else "auto",
                           meta={"config": r["_config"], "source": src, "kind": kind})


@register("notinject")
class NotInject(BaseConverter):
    """Over-defense benchmark: benign prompts full of injection trigger words. All safe -> measures FPR."""

    output_split = "test"
    hf_id = "leolee99/NotInject"
    license = "mit"
    default_splits = ("NotInject_one", "NotInject_two", "NotInject_three")

    def convert(self, r, split, idx):
        return self.sample(split="test", prompt=r["prompt"], label="safe",
                           meta={"category": r.get("category"), "trigger_words": r.get("word_list"), "orig_split": split})


@register("boundary_pairs")
class BoundaryPairs(BaseConverter):
    """Minimal benign/attack pairs sharing vocabulary — the sharpest keyword-FP test."""

    hf_id = "3nesdeniz/agentic-prompt-injection-boundary-pairs"
    license = "cc-by-4.0"
    default_splits = ("test",)

    def convert(self, r, split, idx):
        unsafe = int(r.get("label", 0)) == 1
        fam = r.get("attack_family") or ""
        cat = "jailbreak" if "jailbreak" in fam else "prompt_injection"
        return self.sample(split=split, prompt=r["text"], label="unsafe" if unsafe else "safe",
                           categories=[cat] if unsafe else [],
                           attack_type=("indirect" if ("indirect" in fam or "rag" in fam) else fam or "direct") if unsafe else None,
                           meta={"pair_id": r.get("pair_id"), "attack_family": fam, "risk_domain": r.get("risk_domain")})


@register("neuralchemy_pi")
class NeuralchemyPI(BaseConverter):
    hf_id = "neuralchemy/Prompt-injection-dataset"
    hf_config = "core"
    license = "apache-2.0 (mixed upstream)"
    default_splits = ("test",)
    JB = {"jailbreak", "persona_replacement", "many_shot", "crescendo"}

    def convert(self, r, split, idx):
        unsafe = int(r.get("label", 0)) == 1
        c = r.get("category") or ""
        return self.sample(split=split, prompt=r["text"], label="unsafe" if unsafe else "safe",
                           categories=(["jailbreak" if c in self.JB else "prompt_injection"] if unsafe else []),
                           attack_type=(c or "direct") if unsafe else None,
                           meta={"category": c, "source": r.get("source"), "severity": r.get("severity")})


@register("deepset_prompt_injections")
class DeepsetPI(BaseConverter):
    """Classic small baseline (en/de). Noisy labels and widely trained on — sanity check only."""

    hf_id = "deepset/prompt-injections"
    license = "apache-2.0"
    default_splits = ("test",)

    def convert(self, r, split, idx):
        unsafe = int(r["label"]) == 1
        return self.sample(split=split, prompt=r["text"], label="unsafe" if unsafe else "safe",
                           categories=["prompt_injection"] if unsafe else [], attack_type="direct" if unsafe else None)


@register("jackhhao_jailbreak")
class JackhhaoJailbreak(BaseConverter):
    hf_id = "jackhhao/jailbreak-classification"
    license = "apache-2.0"
    default_splits = ("test",)

    def convert(self, r, split, idx):
        unsafe = r.get("type") == "jailbreak"
        return self.sample(split=split, prompt=r["prompt"], label="unsafe" if unsafe else "safe",
                           categories=["jailbreak"] if unsafe else [], attack_type="jailbreak" if unsafe else None)


@register("in_the_wild_jailbreak")
class InTheWildJailbreak(BaseConverter):
    """TrustAIRLab: jailbreak_* configs -> unsafe, regular_* -> safe role-play hard negatives. Dedupe advised."""

    output_split = "test"
    hf_id = "TrustAIRLab/in-the-wild-jailbreak-prompts"
    license = "mit"
    default_splits = ("train",)

    def load_raw(self, split: str) -> Iterable[dict[str, Any]]:
        from datasets import load_dataset

        date = self.options.get("date", "2023_12_25")
        for kind in ("jailbreak", "regular"):
            for r in load_dataset(self.hf_id, f"{kind}_{date}", split=split, revision=self.revision):
                yield {**r, "_kind": kind}

    def convert(self, r, split, idx):
        unsafe = r["_kind"] == "jailbreak" and bool(r.get("jailbreak", True))
        return self.sample(split="test", prompt=r["prompt"], label="unsafe" if unsafe else "safe",
                           categories=["jailbreak"] if unsafe else [], attack_type="roleplay" if unsafe else None,
                           meta={"platform": r.get("platform"), "source": r.get("source")})


@register("jbb_behaviors")
class JBBBehaviors(BaseConverter):
    """100 harmful goals + 100 topic-matched benign goals (content safety, not jailbreak wrappers)."""

    output_split = "test"
    hf_id = "JailbreakBench/JBB-Behaviors"
    hf_config = "behaviors"
    license = "mit"
    default_splits = ("harmful", "benign")

    def convert(self, r, split, idx):
        unsafe = split == "harmful"
        return self.sample(split="test", prompt=r["Goal"], label="unsafe" if unsafe else "safe",
                           categories=[JBB_MAP.get(str(r.get("Category", "")).lower(), "other")] if unsafe else [],
                           meta={"behavior": r.get("Behavior"), "category": r.get("Category"), "orig_split": split})


@register("harmbench")
class HarmBench(BaseConverter):
    """Ungated mirror of HarmBench behaviours (unsafe-only -> recall). Copyright rows dropped by default."""

    hf_id = "swiss-ai/harmbench"
    hf_config = "DirectRequest"
    license = "mit (upstream)"
    default_splits = ("test",)

    def convert(self, r, split, idx):
        if r.get("FunctionalCategory") == "copyright" and not self.options.get("keep_copyright", False):
            return None
        ctx = (r.get("ContextString") or "").strip()
        prompt = f"{ctx}\n\n{r['Behavior']}" if ctx else r["Behavior"]
        return self.sample(split=split, prompt=prompt, label="unsafe",
                           categories=[HARMBENCH_MAP.get(str(r.get("SemanticCategory", "")).lower(), "other")],
                           attack_type="contextual" if ctx else None,
                           meta={"behavior_id": r.get("BehaviorID"), "functional": r.get("FunctionalCategory")})
