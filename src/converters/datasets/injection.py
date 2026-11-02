"""Prompt-injection / jailbreak datasets: attack -> ``block``, benign -> ``pass``.

Mind the label axis (see docs/datasets.md): several PI datasets label a plain harmful request
without an override attempt as benign (``pass`` here), i.e. ``pass`` only with respect to
injection. Do not mix such rows with content-safety sets that label the same prompt ``block``:
the same prompt would then expect both (a cross-source conflict in the mix manifest).
"""

from __future__ import annotations

import logging
from typing import ClassVar

from ..base import Record, register
from ..sources import HFDatasetSource, MultiConfigHFSource
from ..templates import SafeOnlyConverter, TextLabelConverter, UnsafeOnlyConverter

log = logging.getLogger("converters")


@register("pi_eval_suite")
class PIEvalSuite(MultiConfigHFSource, TextLabelConverter):
    """Horizon-Labs/prompt-injection-eval-suite: 16 configs, unified schema (text, label, source, kind, lang).
    Option ``configs`` picks subsets; ``synthetic_*`` are in-distribution for the publisher's own model.

    Labels are on the INJECTION axis only (1 = injection / jailbreak -> block, 0 = everything else ->
    pass, including plain harmful requests). The suite's ``xstest``, ``orbench_hard`` and ``notinject``
    configs are therefore NOT in ``default_configs``: their rows are all label 0 here, while our
    dedicated ``xstest`` / ``or_bench`` / ``notinject`` converters label the same texts on the
    content-safety axis (XSTest's 200 contrast prompts and OR-Bench-toxic are ``block`` there). With
    both in one mix (core_v1) the same prompt would expect pass AND block. They stay available through
    ``options.configs`` for an injection-only eval."""

    hf_id = "Horizon-Labs/prompt-injection-eval-suite"
    license = "mixed (per config: MIT / Apache-2.0 / CC-BY-4.0 / CC-BY-SA-4.0)"
    default_splits = ("test",)
    default_configs = ("deepset_test", "jackhhao_test", "boundary_pairs_test", "neuralchemy_test", "bipia", "piarena",
                       "llmail_phase2")
    CONTENT_SAFETY_OVERLAP: ClassVar[frozenset[str]] = frozenset({"xstest", "orbench_hard", "notinject"})
    skip_blank_text = True

    def configs(self) -> list[str]:
        configs = super().configs()
        if overlap := sorted(self.CONTENT_SAFETY_OVERLAP & set(configs)):
            log.warning("%s: configs %s carry injection-axis labels (their unsafe-looking prompts are label 0); "
                        "keep them out of mixes that also use the xstest / or_bench / notinject converters",
                        self.name, overlap)
        return configs


@register("notinject")
class NotInject(HFDatasetSource, SafeOnlyConverter):
    """Over-defense benchmark: benign prompts full of injection trigger words. All pass -> measures FPR."""

    output_split = "test"
    hf_id = "leolee99/NotInject"
    license = "mit"
    default_splits = ("NotInject_one", "NotInject_two", "NotInject_three")
    text_field = "prompt"


@register("boundary_pairs")
class BoundaryPairs(HFDatasetSource, TextLabelConverter):
    """Minimal benign/attack pairs sharing vocabulary — the sharpest keyword-FP test (label 1 -> block)."""

    hf_id = "3nesdeniz/agentic-prompt-injection-boundary-pairs"
    license = "cc-by-4.0"
    default_splits = ("test",)


@register("neuralchemy_pi")
class NeuralchemyPI(HFDatasetSource, TextLabelConverter):
    """neuralchemy prompt-injection set (``core`` config): label 1 -> block, 0 -> pass."""

    hf_id = "neuralchemy/Prompt-injection-dataset"
    hf_config = "core"
    license = "apache-2.0 (mixed upstream)"
    default_splits = ("test",)


@register("deepset_prompt_injections")
class DeepsetPI(HFDatasetSource, TextLabelConverter):
    """Classic small baseline (en/de). Noisy labels and widely trained on — sanity check only."""

    hf_id = "deepset/prompt-injections"
    license = "apache-2.0"
    default_splits = ("test",)


@register("jackhhao_jailbreak")
class JackhhaoJailbreak(HFDatasetSource, TextLabelConverter):
    """jackhhao/jailbreak-classification: ``type`` jailbreak -> block, benign -> pass."""

    hf_id = "jackhhao/jailbreak-classification"
    license = "apache-2.0"
    default_splits = ("test",)
    text_field = "prompt"
    label_field = "type"
    positive_values = frozenset({"jailbreak"})
    negative_values = frozenset({"benign"})


@register("in_the_wild_jailbreak")
class InTheWildJailbreak(MultiConfigHFSource, TextLabelConverter):
    """TrustAIRLab: jailbreak_* configs -> block, regular_* -> pass (role-play hard negatives).
    Option ``date`` picks the snapshot (config suffix). Cap the size with ``sample: {n, seed}`` (seeded,
    stratified by expected), NOT ``limit``: jailbreak_* is read first, so ``limit`` keeps every jailbreak
    plus the first regular rows in file order (~47% block instead of the source's ~9%)."""

    output_split = "test"
    hf_id = "TrustAIRLab/in-the-wild-jailbreak-prompts"
    license = "mit"
    default_splits = ("train",)
    text_field = "prompt"

    def configs(self) -> list[str]:
        date = self.options.get("date", "2023_12_25")
        return [f"{kind}_{date}" for kind in ("jailbreak", "regular")]

    def is_positive(self, r: Record) -> bool:
        return r["_config"].startswith("jailbreak_") and bool(r.get("jailbreak", True))


@register("jbb_behaviors")
class JBBBehaviors(HFDatasetSource, TextLabelConverter):
    """100 harmful goals (block) + 100 topic-matched benign goals (pass). The source split is the label."""

    output_split = "test"
    hf_id = "JailbreakBench/JBB-Behaviors"
    hf_config = "behaviors"
    license = "mit"
    default_splits = ("harmful", "benign")
    text_field = "Goal"

    def is_positive(self, r: Record) -> bool:
        return self.current_split == "harmful"


@register("harmbench")
class HarmBench(HFDatasetSource, UnsafeOnlyConverter):
    """Ungated mirror of HarmBench behaviours (all block -> recall). Copyright rows dropped by default
    (``keep_copyright: true`` keeps them); contextual behaviours get their context prepended."""

    hf_id = "swiss-ai/harmbench"
    hf_config = "DirectRequest"
    license = "mit (upstream)"
    default_splits = ("test",)

    def keep(self, r: Record) -> bool:
        return r.get("FunctionalCategory") != "copyright" or bool(self.options.get("keep_copyright", False))

    def text(self, r: Record) -> str:
        ctx = (r.get("ContextString") or "").strip()
        return f"{ctx}\n\n{r['Behavior']}" if ctx else r["Behavior"]
