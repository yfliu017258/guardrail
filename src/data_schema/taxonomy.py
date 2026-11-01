"""Native label sets -> unified ``Category`` values. Shared by converters (gold labels)
and server parsers (model outputs), so dataset and model taxonomies line up.

Lower-case keys; look up with ``map_categories(names, MAP)``.
"""

from __future__ import annotations

from typing import Iterable

# Llama Guard 3/4 hazard codes (also used by PolyGuard S-codes)
LLAMA_GUARD_MAP = {
    "s1": "violence",          # Violent Crimes
    "s2": "crime",             # Non-Violent Crimes
    "s3": "sexual",            # Sex-Related Crimes
    "s4": "child_safety",      # Child Sexual Exploitation
    "s5": "misinformation",    # Defamation
    "s6": "misinformation",    # Specialized Advice
    "s7": "privacy_pii",       # Privacy
    "s8": "confidential",      # Intellectual Property (closest enterprise bucket)
    "s9": "weapons_cbrn",      # Indiscriminate Weapons
    "s10": "hate_harassment",  # Hate
    "s11": "self_harm",        # Suicide & Self-Harm
    "s12": "sexual",           # Sexual Content
    "s13": "misinformation",   # Elections
    "s14": "crime",            # Code Interpreter Abuse
}

# Aegis 2.0 / Nemotron Safety Guard (core + extended categories)
AEGIS_MAP = {
    "violence": "violence",
    "threat": "violence",
    "criminal planning/confessions": "crime",
    "guns and illegal weapons": "weapons_cbrn",
    "controlled/regulated substances": "crime",
    "sexual": "sexual",
    "sexual (minor)": "child_safety",
    "hate/identity hate": "hate_harassment",
    "harassment": "hate_harassment",
    "profanity": "hate_harassment",
    "suicide and self harm": "self_harm",
    "pii/privacy": "privacy_pii",
    "fraud/deception": "crime",
    "malware": "crime",
    "illegal activity": "crime",
    "copyright/trademark/plagiarism": "confidential",
    "unauthorized advice": "misinformation",
    "political/misinformation/conspiracy": "misinformation",
    "high risk gov decision making": "other",
    "immoral/unethical": "other",
    "manipulation": "jailbreak",
    "other": "other",
}
AEGIS_NON_VIOLATION = {"needs caution", "safe", ""}

# Qwen3Guard-Gen
QWEN3GUARD_MAP = {
    "violent": "violence",
    "non-violent illegal acts": "crime",
    "sexual content or sexual acts": "sexual",
    "pii": "privacy_pii",
    "suicide & self-harm": "self_harm",
    "unethical acts": "hate_harassment",
    "politically sensitive topics": "misinformation",
    "copyright violation": "confidential",
    "jailbreak": "jailbreak",
}

# OpenAI moderation eval set columns
OPENAI_MODERATION_MAP = {
    "s": "sexual", "s3": "child_safety", "h": "hate_harassment", "h2": "hate_harassment",
    "hr": "hate_harassment", "v": "violence", "v2": "violence", "sh": "self_harm",
}

# BeaverTails 14 harm flags
BEAVERTAILS_MAP = {
    "animal_abuse": "violence",
    "child_abuse": "child_safety",
    "controversial_topics,politics": "misinformation",
    "discrimination,stereotype,injustice": "hate_harassment",
    "drug_abuse,weapons,banned_substance": "crime",
    "financial_crime,property_crime,theft": "crime",
    "hate_speech,offensive_language": "hate_harassment",
    "misinformation_regarding_ethics,laws_and_safety": "misinformation",
    "non_violent_unethical_behavior": "other",
    "privacy_violation": "privacy_pii",
    "self_harm": "self_harm",
    "sexually_explicit,adult_content": "sexual",
    "terrorism,organized_crime": "violence",
    "violence,aiding_and_abetting,incitement": "violence",
}

# JailbreakBench JBB-Behaviors "Category"
JBB_MAP = {
    "harassment/discrimination": "hate_harassment",
    "malware/hacking": "crime",
    "physical harm": "violence",
    "economic harm": "crime",
    "fraud/deception": "crime",
    "disinformation": "misinformation",
    "sexual/adult content": "sexual",
    "privacy": "privacy_pii",
    "expert advice": "misinformation",
    "government decision-making": "other",
}

# HarmBench SemanticCategory
HARMBENCH_MAP = {
    "chemical_biological": "weapons_cbrn",
    "cybercrime_intrusion": "crime",
    "harassment_bullying": "hate_harassment",
    "illegal": "crime",
    "misinformation_disinformation": "misinformation",
    "harmful": "other",
    "copyright": "confidential",
}

# ChineseSafe subjects. 政治错误 is deliberately absent: it encodes PRC political norms
# (e.g. June-4th references labelled as violations) and must not be used for a Taiwan deployment.
CHINESESAFE_MAP = {
    "偏见歧视": "hate_harassment", "淫秽色情": "sexual", "财产隐私": "privacy_pii", "脏话侮辱": "hate_harassment",
    "违法犯罪": "crime", "身体伤害": "violence", "心理健康": "self_harm", "变体词": "crime", "道德伦理": "other",
}
CHINESESAFE_DROP = {"政治错误"}


def map_categories(names: Iterable[str], mapping: dict[str, str], default: str = "other") -> list[str]:
    out = set()
    for n in names:
        key = str(n).strip().lower()
        if not key:
            continue
        out.add(mapping.get(key, default))
    return sorted(out)
