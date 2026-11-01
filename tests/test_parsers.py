import math

from src.server.backends.parsers import get_parser, score_from_logprobs


def test_llama_guard():
    parse, _ = get_parser("llama_guard")
    assert parse("\n\nsafe", {}).unsafe is False
    out = parse("unsafe\nS1,S10", {})
    assert out.unsafe and out.native_categories == ["S1", "S10"] and out.categories == ["hate_harassment", "violence"]


def test_qwen3guard_controversial_and_ignored_category():
    parse, _ = get_parser("qwen3guard")
    assert parse("Safety: Unsafe\nCategories: Violent", {}).categories == ["violence"]
    assert parse("Safety: Controversial\nCategories: Jailbreak", {"controversial_as_unsafe": False}).unsafe is False
    # politically-sensitive-only flags are ignored by default (zh-TW over-blocking)
    assert parse("Safety: Unsafe\nCategories: Politically Sensitive Topics", {}).unsafe is False
    assert parse("garbage", {}).parse_ok is False


def test_granite_and_json_and_wildguard():
    assert get_parser("granite_guardian")[0]("<think>x</think> <score> yes </score>", {}).unsafe
    assert not get_parser("granite_guardian")[0]("<score>no</score>", {}).unsafe
    j = get_parser("json_policy")[0]('```json\n{"violation": 1, "categories": ["jailbreak"]}\n```', {})
    assert j.unsafe and j.categories == ["jailbreak"]
    w = get_parser("wildguard")[0]("Harmful request: yes\nResponse refusal: n/a\nHarmful response: n/a", {"input_type": "request"})
    assert w.unsafe


def _tok(token, alts):
    return {"token": token, "logprob": math.log(alts[0][1]), "top_logprobs": [{"token": t, "logprob": math.log(p)} for t, p in alts]}


def test_logprob_score_llama_guard():
    _, spec = get_parser("llama_guard")
    lp = [_tok("\n\n", [("\n\n", 0.99)]), _tok("unsafe", [("unsafe", 0.7), ("safe", 0.3)]), _tok("\n", [("\n", 1.0)])]
    assert abs(score_from_logprobs(lp, spec) - 0.7) < 1e-6


def test_logprob_score_qwen_anchor():
    _, spec = get_parser("qwen3guard")
    # first token "Saf" must NOT be read as the label: anchor requires "Safety:" before it
    lp = [_tok("Saf", [("Saf", 0.9), ("Uns", 0.1)]), _tok("ety", [("ety", 1.0)]), _tok(":", [(":", 1.0)]),
          _tok(" Safe", [(" Safe", 0.8), (" Unsafe", 0.15), (" Controversial", 0.05)])]
    assert abs(score_from_logprobs(lp, spec) - 0.2) < 1e-6
    assert abs(score_from_logprobs(lp, spec, ("unsafe",)) - 0.15 / 0.95) < 1e-6
