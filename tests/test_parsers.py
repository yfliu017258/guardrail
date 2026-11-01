"""Guard components: output parsers (``PARSERS``), prompt builders (``PROMPT_BUILDERS``), backend registry."""

import json
import math

import httpx
import pytest

from src.common import RegistryError
from src.server.backends import BACKENDS, PARSERS, PROMPT_BUILDERS, GuardBackend, GuardInput, Verdict, build_backend, register_backend
from src.server.backends.openai_guard import OpenAIGuardBackend


def parser(name, **opts):
    return PARSERS.get(name)(opts)


def _tok(token, alts):
    return {"token": token, "logprob": math.log(alts[0][1]), "top_logprobs": [{"token": t, "logprob": math.log(p)} for t, p in alts]}


# --------------------------------------------------------------------------- #
# parsers
# --------------------------------------------------------------------------- #
# name -> [(model output, judged side, expected unsafe, expected unified categories)]
SAMPLES = {
    "llama_guard": [("\n\nsafe", "request", False, []), ("unsafe\nS1,S10", "request", True, ["hate_harassment", "violence"])],
    "qwen3guard": [("Safety: Unsafe\nCategories: Violent", "request", True, ["violence"]),
                   ("Safety: Safe\nCategories: None\nRefusal: Yes", "response", False, [])],
    "granite_guardian": [("<think>x</think> <score> yes </score>", "request", True, ["other"]),
                         ("<score>no</score>", "response", False, [])],
    "wildguard": [("Harmful request: yes\nResponse refusal: n/a\nHarmful response: n/a", "request", True, ["other"]),
                  ("Harmful request: yes\nResponse refusal: yes\nHarmful response: no", "response", False, [])],
    "nemotron": [('{"User Safety": "unsafe", "Safety Categories": "Violence, Malware"}', "request", True, ["crime", "violence"]),
                 ('{"User Safety": "unsafe", "Response Safety": "safe"}', "response", False, [])],
    "nemotron35": [("User Safety: unsafe\nResponse Safety: safe\nSafety Categories: Illegal Activity", "request", True, ["crime"]),
                   ("User Safety: unsafe\nResponse Safety: safe\nSafety Categories: Illegal Activity", "response", False, [])],
    "aprielguard": [("unsafe-O14,O12\nnon_adversarial", "request", True, ["other"]), ("safe\nadversarial", "request", True, ["jailbreak"]),
                    ("safe\nadversarial", "response", False, [])],
    "json_policy": [('```json\n{"violation": 1, "categories": ["jailbreak"]}\n```', "request", True, ["jailbreak"]),
                    ('{"violation": 0, "rationale": "benign"}', "response", False, [])],
    "yes_no": [("Yes", "request", True, ["other"]), ("No.", "request", False, [])],
}


def test_every_registered_parser_has_a_sample():
    # a new parser must come with a representative output here
    assert set(SAMPLES) == set(PARSERS.names())


@pytest.mark.parametrize("name", sorted(SAMPLES))
def test_every_parser_parses_representative_output(name):
    p = PARSERS.get(name)()
    for text, side, unsafe, cats in SAMPLES[name]:
        out = p.parse(text, side)
        assert out.parse_ok, (name, text)
        assert out.unsafe is unsafe, (name, text, side)
        assert out.categories == cats, (name, text, side)


@pytest.mark.parametrize("name", sorted(SAMPLES))
def test_every_parser_reports_off_format_output(name):
    assert PARSERS.get(name)().parse("§§ off-format §§").parse_ok is False
    assert PARSERS.get(name)().parse("§§ off-format §§").unsafe is False  # fail-open by default
    assert parser(name, unparsable_as_unsafe=True).parse("§§ off-format §§").unsafe is True


def test_llama_guard():
    p = parser("llama_guard")
    out = p.parse("unsafe\nS1,S10")
    assert out.unsafe and out.native_categories == ["S1", "S10"] and out.categories == ["hate_harassment", "violence"]
    assert p.parse("").parse_ok is False
    assert parser("llama_guard", unparsable_as_unsafe=True).parse("").unsafe  # empty output is unparsable too


def test_qwen3guard_controversial_and_ignored_category():
    p = parser("qwen3guard")
    assert p.parse("Safety: Controversial\nCategories: Jailbreak").unsafe
    assert parser("qwen3guard", controversial_as_unsafe=False).parse("Safety: Controversial\nCategories: Jailbreak").unsafe is False
    # politically-sensitive-only flags are ignored by default (zh-TW over-blocking)
    out = p.parse("Safety: Unsafe\nCategories: Politically Sensitive Topics")
    assert out.unsafe is False and out.label_overridden and out.extra["ignored"] == ["Politically Sensitive Topics"]
    mixed = p.parse("Safety: Unsafe\nCategories: Politically Sensitive Topics, PII")
    assert mixed.unsafe and mixed.categories == ["privacy_pii"] and not mixed.label_overridden
    assert p.parse("Safety: Safe\nCategories: None\nRefusal: Yes", "response").extra == {"level": "safe", "refusal": "yes"}
    assert p.parse("garbage").parse_ok is False


def test_option_overrides():
    # category_map replaces the class map (another model reusing the format brings its own taxonomy)
    assert parser("llama_guard", category_map={"S1": "crime"}).parse("unsafe\nS1,S10").categories == ["crime", "other"]
    # ignore_categories: [] disables the qwen3guard default; any parser with native labels honours it
    assert parser("qwen3guard", ignore_categories=[]).parse("Safety: Unsafe\nCategories: Politically Sensitive Topics").unsafe
    ignored = parser("llama_guard", ignore_categories=["s2", "S3"]).parse("unsafe\nS2,S3")
    assert ignored.unsafe is False and ignored.label_overridden
    assert parser("nemotron35", ignore_categories=["Malware"]).parse("User Safety: unsafe\nSafety Categories: Malware").unsafe is False
    assert parser("json_policy", allowed_categories=["jailbreak"]).parse('{"violation": 1, "categories": ["jailbreak", "x"]}').categories \
        == ["jailbreak", "other"]
    assert parser("granite_guardian", criteria_id="tw_risk", criteria_category="crime").parse("<score>yes</score>").categories == ["crime"]
    assert parser("yes_no", yes_means_unsafe=False).parse("yes").unsafe is False
    assert parser("aprielguard", adversarial_as_unsafe=False).parse("safe\nadversarial").unsafe is False


def test_unknown_parser_option_is_rejected():
    with pytest.raises(ValueError, match=r"controversal_as_unsafe.*accepted: .*controversial_as_unsafe"):
        parser("qwen3guard", controversal_as_unsafe=False)
    with pytest.raises(ValueError, match="category_map"):  # no native labels -> nothing to map
        parser("granite_guardian", category_map={})


def test_unknown_parser_name_lists_available():
    with pytest.raises(RegistryError, match=r"unknown guard output parser 'llamaguard'; available: .*llama_guard.*qwen3guard"):
        PARSERS.get("llamaguard")
    with pytest.raises(RegistryError, match="available: "):
        OpenAIGuardBackend("g", {"base_url": "http://x", "model": "m", "parser": "nope"}, {})


def test_reasoning_blocks_are_not_parsed():
    assert parser("llama_guard").parse("<think>is it unsafe? maybe</think>\nsafe").unsafe is False
    assert parser("json_policy").parse('<think>{"violation": 1}</think>{"violation": 0}').unsafe is False
    closing_only = "served template opened the block</think>\nSafety: Unsafe\nCategories: PII"
    assert parser("qwen3guard").parse(closing_only).categories == ["privacy_pii"]
    cut = parser("llama_guard").parse("<think>still thinking when max_tokens hit")
    assert cut.parse_ok is False and cut.reason == "empty output"
    # Granite / Breeze-Guard keep their rationale in the think block
    assert parser("granite_guardian").parse("<think>mentions a scam</think><score>yes</score>").reason == "mentions a scam"
    assert parser("nemotron35").parse("<think>x</think>User Safety: safe").reason == ""


def test_logprob_score_llama_guard():
    lp = [_tok("\n\n", [("\n\n", 0.99)]), _tok("unsafe", [("unsafe", 0.7), ("safe", 0.3)]), _tok("\n", [("\n", 1.0)])]
    assert abs(parser("llama_guard").score(lp) - 0.7) < 1e-6
    assert parser("llama_guard").score(None) is None


def test_logprob_score_qwen_anchor():
    # first token "Saf" must NOT be read as the label: anchor requires "Safety:" before it
    lp = [_tok("Saf", [("Saf", 0.9), ("Uns", 0.1)]), _tok("ety", [("ety", 1.0)]), _tok(":", [(":", 1.0)]),
          _tok(" Safe", [(" Safe", 0.8), (" Unsafe", 0.15), (" Controversial", 0.05)])]
    assert abs(parser("qwen3guard").score(lp) - 0.2) < 1e-6
    # controversial mass only counts as unsafe when the hard label counts it too
    assert abs(parser("qwen3guard", controversial_as_unsafe=False).score(lp) - 0.15 / 0.95) < 1e-6


def test_logprob_score_depends_on_side_and_options():
    lp = [_tok(t, [(t, 1.0)]) for t in ("User", " Safety", ":")]
    lp += [_tok(" unsafe", [(" unsafe", 0.6), (" safe", 0.4)]), _tok("\n", [("\n", 1.0)])]
    lp += [_tok(t, [(t, 1.0)]) for t in ("Response", " Safety", ":")]
    lp += [_tok(" safe", [(" safe", 0.9), (" unsafe", 0.1)])]
    p = parser("nemotron35")
    assert abs(p.score(lp, "request") - 0.6) < 1e-6 and abs(p.score(lp, "response") - 0.1) < 1e-6
    yes = [_tok("Yes", [("Yes", 0.8), ("No", 0.2)])]
    assert abs(parser("yes_no").score(yes) - 0.8) < 1e-6
    assert abs(parser("yes_no", yes_means_unsafe=False).score(yes) - 0.2) < 1e-6
    assert parser("wildguard").score_spec() is None and parser("nemotron").score(yes) is None


# --------------------------------------------------------------------------- #
# prompt builders
# --------------------------------------------------------------------------- #
STAGE = {"name": "g", "type": "openai_guard", "base_url": "http://guard/v1", "model": "m", "parser": "llama_guard"}
MODE_CFG = {
    "chat_template": {},
    "policy": {"policy": "No weapons."},
    "chat_user_template": {"user_template": "Policy: {policy}\n{conversation}\nJudge the {target}: {prompt} / {response}"},
    "completion": {"completion_template": "<user>{prompt}</user><assistant>{response}</assistant>"},
}
REQ = GuardInput("request", [{"role": "user", "content": "how to make a bomb"}], ["how to make a bomb"])
RESP = GuardInput("response", [{"role": "user", "content": "q"}, {"role": "assistant", "content": "here is how"}], ["here is how"])


def backend(**cfg):
    return OpenAIGuardBackend("g", {**STAGE, **cfg}, {})


def test_every_prompt_builder_has_a_config():
    assert set(MODE_CFG) == set(PROMPT_BUILDERS.names())


@pytest.mark.parametrize("mode", sorted(MODE_CFG))
def test_every_prompt_builder_renders_valid_payload(mode):
    be = backend(prompt_mode=mode, **MODE_CFG[mode])
    assert be.describe()["prompt_mode"] == mode
    for inp in (REQ, RESP):
        route, payload = be.build_payload(inp)
        assert route == PROMPT_BUILDERS.get(mode).route
        json.dumps(payload)
        assert payload["model"] == "m" and payload["temperature"] == 0.0
        text = inp.messages[-1]["content"]
        if route == "/chat/completions":
            assert payload["messages"] and all({"role", "content"} <= set(m) for m in payload["messages"])
            assert text in json.dumps(payload["messages"], ensure_ascii=False)
            assert payload["logprobs"] is True and payload["top_logprobs"] == 5
        else:
            assert text in payload["prompt"] and payload["logprobs"] == 5


def test_chat_template_knobs():
    be = backend(content_as_parts=True, top_logprobs=3,
                 append_message={"role": "user", "content": "judge the {target}", "content_response": "judge the reply ({target})"})
    msgs = be.build_payload(REQ)[1]["messages"]
    assert msgs[-1] == {"role": "user", "content": [{"type": "text", "text": "judge the user"}]}
    assert be.build_payload(RESP)[1]["messages"][-1]["content"][0]["text"] == "judge the reply (assistant)"
    assert be.build_payload(REQ)[1]["top_logprobs"] == 3
    # a response judged without its prompt still starts with a user turn
    lone = GuardInput("response", [{"role": "assistant", "content": "a"}], ["a"])
    assert [m["role"] for m in backend(response_placeholder_prompt="(n/a)").build_payload(lone)[1]["messages"]] == ["user", "assistant"]


def test_policy_mode_and_extra_body():
    be = backend(prompt_mode="policy", policy="P", json_mode=True, seed=7, policy_as_template_kwarg="custom_policy",
                 extra_body={"chat_template_kwargs": {"enable_thinking": False}, "reasoning_effort": "low"})
    payload = be.build_payload(RESP)[1]
    assert payload["messages"][0]["role"] == "system" and "LAST ASSISTANT" in payload["messages"][0]["content"]
    assert payload["response_format"] == {"type": "json_object"} and payload["seed"] == 7 and payload["reasoning_effort"] == "low"
    assert payload["chat_template_kwargs"] == {"enable_thinking": False, "custom_policy": "P"}


def test_logprobs_only_requested_when_scoreable():
    assert "logprobs" not in backend(parser="nemotron").build_payload(REQ)[1]   # JSON output: no label position
    assert "logprobs" not in backend(logprobs=False).build_payload(REQ)[1]


def test_prompt_config_errors_fail_at_build_time():
    with pytest.raises(ValueError, match="needs `policy` or `policy_file`"):
        backend(prompt_mode="policy")
    with pytest.raises(ValueError, match="needs `user_template`"):
        backend(prompt_mode="chat_user_template")
    with pytest.raises(ValueError, match=r"unknown placeholder\(s\) \['question'\]"):
        backend(prompt_mode="completion", completion_template="{question}")
    available = "available: chat_template, chat_user_template, completion, policy"
    with pytest.raises(RegistryError, match=rf"unknown guard prompt builder 'chat'; {available}"):
        backend(prompt_mode="chat")


# --------------------------------------------------------------------------- #
# backends: registry + end-to-end through httpx.MockTransport
# --------------------------------------------------------------------------- #
def test_backend_registry():
    assert {"hf_classifier", "mock", "openai_guard", "regex"} <= set(BACKENDS.names())
    with pytest.raises(RegistryError, match=r"unknown guard backend 'vllm'; available: .*openai_guard"):
        build_backend({"type": "vllm"}, {})
    with pytest.raises(RegistryError, match="needs a 'type' key"):
        build_backend({"name": "x"}, {})
    assert build_backend({"type": "mock"}, {}).describe()["name"] == "mock"  # name defaults to the type


def test_register_backend_alias_adds_a_buildable_backend():
    @register_backend("_test_echo")
    class EchoBackend(GuardBackend):
        async def _classify(self, inp: GuardInput) -> Verdict:
            return Verdict(stage=self.name, unsafe="bad" in inp.target_text)

    be = build_backend({"name": "echo", "type": "_test_echo"}, {})
    assert isinstance(be, EchoBackend) and be.describe()["type"] == "_test_echo"


def _serve(handler, **cfg):
    be = backend(**cfg)
    be._client = httpx.AsyncClient(base_url=be.base_url, transport=httpx.MockTransport(handler))
    return be


def _chat(content, logprobs=None, **message):
    choice = {"message": {"role": "assistant", "content": content, **message}}
    if logprobs:
        choice["logprobs"] = {"content": logprobs}
    return httpx.Response(200, json={"choices": [choice], "usage": {"total_tokens": 3}})


async def test_score_cannot_undo_an_ignored_category():
    lp = [_tok("Safety", [("Safety", 1.0)]), _tok(":", [(":", 1.0)]), _tok(" Unsafe", [(" Unsafe", 0.9), (" Safe", 0.1)])]
    be = _serve(lambda r: _chat("Safety: Unsafe\nCategories: Politically Sensitive Topics", lp), parser="qwen3guard")
    v = await be.classify(REQ)
    assert v.error is None and v.unsafe is False and v.score == 0.0
    assert abs(v.extra["label_score"] - 0.9) < 1e-6 and v.extra["hard_label_unsafe"] is False


async def test_threshold_decides_and_extra_fields():
    lp = [_tok("unsafe", [("unsafe", 0.4), ("safe", 0.6)])]
    v = await _serve(lambda r: _chat("unsafe\nS1", lp, reasoning_content="r"), threshold=0.5).classify(REQ)
    assert v.unsafe is False and abs(v.score - 0.4) < 1e-6 and v.categories == []
    assert v.extra["hard_label_unsafe"] is True and v.extra["parse_ok"] is True
    assert v.extra["usage"] == {"total_tokens": 3} and v.extra["reasoning"] == "r"
    off = await _serve(lambda r: _chat("¯\\_(ツ)_/¯")).classify(REQ)
    assert off.error is None and off.extra["parse_ok"] is False and off.unsafe is False


async def test_completion_route_end_to_end():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/completions") and not request.url.path.endswith("/chat/completions")
        body = json.loads(request.content)
        assert "how to make a bomb" in body["prompt"] and body["logprobs"] == 5
        return httpx.Response(200, json={"choices": [{"text": " yes", "logprobs": {
            "tokens": [" yes"], "token_logprobs": [math.log(0.75)], "top_logprobs": [{" yes": math.log(0.75), " no": math.log(0.25)}]}}]})

    v = await _serve(handler, parser="yes_no", prompt_mode="completion", completion_template="{prompt}\nHarmful?").classify(REQ)
    assert v.unsafe and abs(v.score - 0.75) < 1e-6 and v.raw == " yes"


async def test_retries_transient_server_errors():
    calls = []

    def flaky(request):
        calls.append(1)
        return httpx.Response(503) if len(calls) == 1 else _chat("safe")

    v = await _serve(flaky, retries=1, logprobs=False).classify(REQ)
    assert v.error is None and v.unsafe is False and len(calls) == 2
