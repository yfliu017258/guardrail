"""Prompt-injection / jailbreak datasets.

Keep two label axes apart (see docs/datasets.md): *attack* (prompt_injection / jailbreak,
recorded in ``attack_type``) vs *harm category*. Several PI datasets label a plain harmful
request without an override attempt as benign — those rows are ``safe`` here only with
respect to injection, so do not mix them into content-safety recall numbers.
"""

from __future__ import annotations

from typing import Any, ClassVar

from ...data_schema.taxonomy import HARMBENCH_MAP, JBB_MAP
from ..base import Record, register
from ..sources import HFDatasetSource, MultiConfigHFSource
from ..templates import SafeOnlyConverter, TextLabelConverter, UnsafeOnlyConverter


@register("pi_eval_suite")
class PIEvalSuite(MultiConfigHFSource, TextLabelConverter):
    """Horizon-Labs/prompt-injection-eval-suite: 16 configs, unified schema (text, label, source, kind, lang).
    Option ``configs`` picks subsets; ``synthetic_*`` are in-distribution for the publisher's own model."""

    hf_id = "Horizon-Labs/prompt-injection-eval-suite"
    license = "mixed (per config: MIT / Apache-2.0 / CC-BY-4.0 / CC-BY-SA-4.0)"
    default_splits = ("test",)
    default_configs = ("notinject", "xstest", "orbench_hard", "deepset_test", "jackhhao_test", "boundary_pairs_test",
                       "neuralchemy_test", "bipia", "piarena", "llmail_phase2")
    skip_blank_text = True
    JAILBREAK_CONFIGS: ClassVar[frozenset[str]] = frozenset({"jackhhao_test", "simsonsun_jailbreaks"})

    @staticmethod
    def _source(r: Record) -> str:
        return str(r.get("source") or r["_config"])

    def _is_jailbreak(self, r: Record) -> bool:
        return "jailbreak" in self._source(r) or r["_config"] in self.JAILBREAK_CONFIGS

    def categories(self, r: Record) -> list[str]:
        return ["jailbreak" if self._is_jailbreak(r) else "prompt_injection"]

    def attack(self, r: Record, unsafe: bool) -> str | None:
        if not unsafe:
            return None
        if (r.get("kind") or "direct") == "indirect":
            return "indirect"
        return "jailbreak" if self._is_jailbreak(r) else "direct"

    def lang(self, r: Record) -> str:
        return r["lang"] if r.get("lang") in ("en", "zh-TW", "zh-CN") else "auto"

    def meta(self, r: Record) -> dict[str, Any]:
        return {"config": r["_config"], "source": self._source(r), "kind": r.get("kind") or "direct"}


@register("notinject")
class NotInject(HFDatasetSource, SafeOnlyConverter):
    """Over-defense benchmark: benign prompts full of injection trigger words. All safe -> measures FPR."""

    output_split = "test"
    hf_id = "leolee99/NotInject"
    license = "mit"
    default_splits = ("NotInject_one", "NotInject_two", "NotInject_three")
    text_field = "prompt"

    def meta(self, r: Record) -> dict[str, Any]:
        return {"category": r.get("category"), "trigger_words": r.get("word_list"), "orig_split": self.current_split}


@register("boundary_pairs")
class BoundaryPairs(HFDatasetSource, TextLabelConverter):
    """Minimal benign/attack pairs sharing vocabulary — the sharpest keyword-FP test."""

    hf_id = "3nesdeniz/agentic-prompt-injection-boundary-pairs"
    license = "cc-by-4.0"
    default_splits = ("test",)

    @staticmethod
    def _family(r: Record) -> str:
        return r.get("attack_family") or ""

    def categories(self, r: Record) -> list[str]:
        return ["jailbreak" if "jailbreak" in self._family(r) else "prompt_injection"]

    def attack(self, r: Record, unsafe: bool) -> str | None:
        if not unsafe:
            return None
        fam = self._family(r)
        return "indirect" if ("indirect" in fam or "rag" in fam) else (fam or "direct")

    def meta(self, r: Record) -> dict[str, Any]:
        return {"pair_id": r.get("pair_id"), "attack_family": self._family(r), "risk_domain": r.get("risk_domain")}


@register("neuralchemy_pi")
class NeuralchemyPI(HFDatasetSource, TextLabelConverter):
    """neuralchemy prompt-injection set (``core`` config); its ``category`` names the attack technique."""

    hf_id = "neuralchemy/Prompt-injection-dataset"
    hf_config = "core"
    license = "apache-2.0 (mixed upstream)"
    default_splits = ("test",)
    JB: ClassVar[frozenset[str]] = frozenset({"jailbreak", "persona_replacement", "many_shot", "crescendo"})

    def categories(self, r: Record) -> list[str]:
        return ["jailbreak" if (r.get("category") or "") in self.JB else "prompt_injection"]

    def attack(self, r: Record, unsafe: bool) -> str | None:
        return (r.get("category") or "direct") if unsafe else None

    def meta(self, r: Record) -> dict[str, Any]:
        return {"category": r.get("category") or "", "source": r.get("source"), "severity": r.get("severity")}


@register("deepset_prompt_injections")
class DeepsetPI(HFDatasetSource, TextLabelConverter):
    """Classic small baseline (en/de). Noisy labels and widely trained on — sanity check only."""

    hf_id = "deepset/prompt-injections"
    license = "apache-2.0"
    default_splits = ("test",)
    category = "prompt_injection"
    attack_type = "direct"


@register("jackhhao_jailbreak")
class JackhhaoJailbreak(HFDatasetSource, TextLabelConverter):
    """jackhhao/jailbreak-classification: ``type`` jailbreak vs benign prompts."""

    hf_id = "jackhhao/jailbreak-classification"
    license = "apache-2.0"
    default_splits = ("test",)
    text_field = "prompt"
    label_field = "type"
    positive_values = frozenset({"jailbreak"})
    category = "jailbreak"
    attack_type = "jailbreak"


@register("in_the_wild_jailbreak")
class InTheWildJailbreak(MultiConfigHFSource, TextLabelConverter):
    """TrustAIRLab: jailbreak_* configs -> unsafe, regular_* -> safe role-play hard negatives. Dedupe advised.
    Option ``date`` picks the snapshot (config suffix)."""

    output_split = "test"
    hf_id = "TrustAIRLab/in-the-wild-jailbreak-prompts"
    license = "mit"
    default_splits = ("train",)
    text_field = "prompt"
    category = "jailbreak"
    attack_type = "roleplay"

    def configs(self) -> list[str]:
        date = self.options.get("date", "2023_12_25")
        return [f"{kind}_{date}" for kind in ("jailbreak", "regular")]

    def is_unsafe(self, r: Record) -> bool:
        return r["_config"].startswith("jailbreak_") and bool(r.get("jailbreak", True))

    def meta(self, r: Record) -> dict[str, Any]:
        return {"platform": r.get("platform"), "source": r.get("source")}


@register("jbb_behaviors")
class JBBBehaviors(HFDatasetSource, TextLabelConverter):
    """100 harmful goals + 100 topic-matched benign goals (content safety, not jailbreak wrappers).
    The source split is the label."""

    output_split = "test"
    hf_id = "JailbreakBench/JBB-Behaviors"
    hf_config = "behaviors"
    license = "mit"
    default_splits = ("harmful", "benign")
    text_field = "Goal"

    def is_unsafe(self, r: Record) -> bool:
        return self.current_split == "harmful"

    def categories(self, r: Record) -> list[str]:
        return [JBB_MAP.get(str(r.get("Category", "")).lower(), "other")]

    def meta(self, r: Record) -> dict[str, Any]:
        return {"behavior": r.get("Behavior"), "category": r.get("Category"), "orig_split": self.current_split}


@register("harmbench")
class HarmBench(HFDatasetSource, UnsafeOnlyConverter):
    """Ungated mirror of HarmBench behaviours (unsafe-only -> recall). Copyright rows dropped by default
    (``keep_copyright: true`` keeps them); contextual behaviours get their context prepended."""

    hf_id = "swiss-ai/harmbench"
    hf_config = "DirectRequest"
    license = "mit (upstream)"
    default_splits = ("test",)

    @staticmethod
    def _context(r: Record) -> str:
        return (r.get("ContextString") or "").strip()

    def keep(self, r: Record) -> bool:
        return r.get("FunctionalCategory") != "copyright" or bool(self.options.get("keep_copyright", False))

    def text(self, r: Record) -> str:
        ctx = self._context(r)
        return f"{ctx}\n\n{r['Behavior']}" if ctx else r["Behavior"]

    def categories(self, r: Record) -> list[str]:
        return [HARMBENCH_MAP.get(str(r.get("SemanticCategory", "")).lower(), "other")]

    def attack(self, r: Record, unsafe: bool) -> str | None:
        return "contextual" if self._context(r) else None

    def meta(self, r: Record) -> dict[str, Any]:
        return {"behavior_id": r.get("BehaviorID"), "functional": r.get("FunctionalCategory")}
