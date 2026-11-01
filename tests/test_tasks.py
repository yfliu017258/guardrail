"""model_safety / refusal_judge tasks + generators, independent of the concrete judge implementations."""

import json

import httpx
import pytest

from src.evaluation.__main__ import main
from src.evaluation.errors import TargetError
from src.evaluation.generation import GENERATORS, OpenAIChatGenerator, split_reasoning
from src.evaluation.tasks import TASKS
from src.refuse_judge.base import REFUSAL_JUDGES, RefusalJudge


@REFUSAL_JUDGES.register("test_stub")
class StubJudge(RefusalJudge):
    """Refusal iff the answer starts with a refusal marker (test only)."""

    async def _judge(self, prompt, response):
        if response.startswith(("抱歉", "I can't", "Sorry")):
            return self.verdict("refusal", score=0.95)
        if "however" in response.lower():
            return self.verdict("partial", score=0.5)
        return self.verdict("compliance", score=0.05)


def test_registries_list_everything():
    assert {"guardrail", "model_safety", "refusal_judge"} <= set(TASKS.names())
    assert {"openai_chat", "mock"} <= set(GENERATORS.names())
    with pytest.raises(KeyError, match="available"):
        TASKS.get("nope")


def test_split_reasoning():
    assert split_reasoning("<think>plan</think>\n答案", None) == ("答案", "plan")
    assert split_reasoning("<think>never finished", None) == ("", "never finished")
    assert split_reasoning("plain", "server-side") == ("plain", "server-side")


def _gen(handler, **cfg):
    g = OpenAIChatGenerator({"type": "openai_chat", "base_url": "http://m/v1", "model": "qwen3.6-27b", **cfg}, {})
    g._client = httpx.AsyncClient(base_url=g.base_url, transport=httpx.MockTransport(handler))
    return g


async def test_openai_chat_generator_paths():
    def ok(req):
        body = json.loads(req.content)
        assert body["messages"][0] == {"role": "system", "content": "SYS"} and body["chat_template_kwargs"] == {"enable_thinking": False}
        return httpx.Response(200, json={"choices": [{"message": {"content": "<think>x</think>Hello", "reasoning_content": None},
                                                       "finish_reason": "stop"}], "usage": {"completion_tokens": 3}})
    g = _gen(ok, system_prompt="SYS", extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    out = await g.generate("hi")
    assert out.text == "Hello" and out.reasoning == "x" and out.usage["completion_tokens"] == 3

    blocked = await _gen(lambda r: httpx.Response(400, text='{"error":"Guardrail blocked: 此請求可能違反"}')).generate("x")
    assert blocked.blocked and blocked.text == ""
    with pytest.raises(TargetError):
        await _gen(lambda r: httpx.Response(503, text="busy")).generate("x")
    bad = await _gen(lambda r: httpx.Response(404, text="no model")).generate("x")
    assert bad.error and not bad.blocked


def test_fingerprint_tracks_generation_settings():
    a = GENERATORS.build({"type": "mock"}).fingerprint
    b = GENERATORS.build({"type": "mock", "refusal_text": "no"}).fingerprint
    c = _gen(lambda r: None, temperature=0.0).fingerprint
    d = _gen(lambda r: None, temperature=0.7).fingerprint
    assert a != b and c != d


ARGS = ["--env-file", "/nonexistent"]


def _run(tmp_path, config, *extra):
    main(["--config", config, "--output-dir", str(tmp_path), *ARGS, *extra])
    return sorted(tmp_path.iterdir())[-1]


def test_model_safety_smoke(tmp_path):
    run = _run(tmp_path, "src/config/evaluation/smoke_model_safety.yaml", "--set", "task.model_safety.judge={type: test_stub}")
    m = json.loads((run / "metrics.json").read_text())
    ms = m["model_safety"]
    assert m["task"] == "model_safety" and ms["n_unsafe_prompts"] + ms["n_safe_prompts"] == 24  # request rows only
    rows = [json.loads(line) for line in (run / "predictions.jsonl").open()]
    assert all(r["input_type"] == "request" and r["gen_fingerprint"] for r in rows)
    assert all(r["reasoning"] == "<mock reasoning>" for r in rows)            # kept, never judged
    report = (run / "report.md").read_text()
    assert "## Model safety" in report and "over-refusal rate on safe prompts" in report


def test_model_safety_reuse_generations(tmp_path):
    first = _run(tmp_path / "a", "src/config/evaluation/smoke_model_safety.yaml", "--set", "task.model_safety.judge={type: test_stub}")
    second = _run(tmp_path / "b", "src/config/evaluation/smoke_model_safety.yaml", "--set", "task.model_safety.judge={type: test_stub}",
                  "--set", f"task.model_safety.reuse_generations_from={first}")
    rows = [json.loads(line) for line in (second / "predictions.jsonl").open()]
    assert rows and all(r["gen_cached"] for r in rows)
    with pytest.raises(ValueError, match="fingerprint"):  # different generator settings -> refuse to mix
        _run(tmp_path / "c", "src/config/evaluation/smoke_model_safety.yaml", "--set", "task.model_safety.judge={type: test_stub}",
             "--set", "task.model_safety.generator.refusal_text=nope",
             "--set", f"task.model_safety.reuse_generations_from={first}")


def test_refusal_judge_smoke(tmp_path):
    run = _run(tmp_path, "src/config/evaluation/smoke_refusal_judge.yaml", "--set", "task.refusal_judge.judge={type: test_stub}")
    m = json.loads((run / "metrics.json").read_text())
    assert m["n_total"] == 6 and m["overall"]["recall"] == 1.0 and m["overall"]["fpr"] == 0.0
    assert m["label_confusion"]["refusal"]["refusal"] == 2 and m["label_kappa"] == 1.0
    assert "Judge label confusion" in (run / "report.md").read_text()


def test_rescore_model_safety(tmp_path):
    run = _run(tmp_path, "src/config/evaluation/smoke_model_safety.yaml", "--set", "task.model_safety.judge={type: test_stub}")
    main(["--rescore", str(run), *ARGS])
    (sub,) = [p for p in run.iterdir() if p.name.startswith("rescore_default_")]
    assert json.loads((sub / "metrics.json").read_text())["model_safety"]


def test_judge_config_deep_merge(tmp_path):
    from src.evaluation.tasks.refusal import load_judge_cfg

    base = tmp_path / "base.yaml"
    base.write_text("judge: {type: cascade, hi: 0.85, first: {type: keyword}, then: {type: llm, model: X, timeout: 30}}\n")
    child = tmp_path / "child.yaml"
    child.write_text(f"_base_: {base.name}\njudge: {{hi: 0.9}}\n")
    cfg = load_judge_cfg({"judge_config": str(child), "judge": {"then": {"model": "glm-5.3-flash"}}})
    assert cfg["hi"] == 0.9 and cfg["then"] == {"type": "llm", "model": "glm-5.3-flash", "timeout": 30}
