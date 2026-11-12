"""Analysis layer: loader maths on real run directories, and every app page renders without an exception."""

import sys

import pytest

pd = pytest.importorskip("pandas")

from src.analysis import loader as L
from src.evaluation.__main__ import main

NOENV = ["--env-file", "/nonexistent"]


@pytest.fixture(scope="module")
def results(mock_server, tmp_path_factory):
    """Two guardrail runs (the second one misses weapons / crime) and a model run, on the same data."""
    out = tmp_path_factory.mktemp("results")
    common = ["--output-dir", str(out), "--set", "metrics.bootstrap=20", *NOENV]
    guard = ["--config", "src/config/evaluation/guardrail_service.yaml", "--data", "processed_data/samples/smoke.csv",
             "--set", "target.http_guardrail.api_key_env=null"]
    main([*guard, "--service-url", mock_server.url, "--run-name", "guard-full", *common])
    main(["--config", "src/config/evaluation/smoke.yaml", "--target", "inprocess", "--run-name", "guard-weak",
          "--server-config", "src/config/server/mock.yaml",
          "--set", "target.inprocess.overrides=['pipeline.stages.1.keywords={jailbreak: [dan mode]}']", *common])
    main(["--config", "src/config/evaluation/smoke_model_safety.yaml", "--run-name", "mock-model", *common])
    return out


def _by_name(results):
    return {r.name: r for r in L.list_runs(results)}


def test_wilson_and_latency_stats():
    assert L.wilson(0, 0) == (None, None)
    lo, hi = L.wilson(5, 10)
    assert 0.2 < lo < 0.5 < hi < 0.8
    assert L.wilson(10, 10)[1] == 1.0 and L.wilson(0, 10)[0] == 0.0
    assert L.latency_stats([])["p95"] is None and L.latency_stats([10, 20, 30])["p50"] == 20


def test_runs_are_listed_and_summarised(results):
    runs = _by_name(results)
    assert set(runs) == {"guard-full", "guard-weak", "mock-model"} and all(r.supported for r in runs.values())
    assert runs["mock-model"].task == "model_safety" and runs["guard-full"].input_type == "request"
    assert len({r.dataset_id for r in runs.values()}) == 1
    s = L.summarize(runs["guard-full"].rows)
    assert s["n"] == 27 and s["tp"] + s["fn"] == 12 and s["latency_p95"] >= s["latency_p50"] > 0
    assert s["recall"] > L.summarize(runs["guard-weak"].rows)["recall"]
    assert L.slice_table(runs["guard-full"].rows, "language")["n"].sum() == 27
    assert runs["guard-full"].summary()["recall"] == s["recall"]


def test_compare_guardrails(results):
    runs = _by_name(results)
    a, b = runs["guard-full"], runs["guard-weak"]
    aligned, info = L.align([a, b])
    assert info["common"] == 27 and info["same_dataset"]
    groups = L.disagreements(aligned, a.label, b.label)
    only_a = next(v for k, v in groups.items() if k.startswith(f"只有「{a.label}」擋下"))
    assert len(only_a) == L.summarize(a.rows)["tp"] - L.summarize(b.rows)["tp"] > 0
    assert sum(len(v) for v in groups.values()) <= 27


def test_model_plus_guardrail(results):
    runs = _by_name(results)
    model = runs["mock-model"]
    bare = L.combo_summary(L.combine(model, None))
    full = L.combo_summary(L.combine(model, runs["guard-full"]))
    weak = L.combo_summary(L.combine(model, runs["guard-weak"]))
    assert bare["unsafe"]["guardrail"] == 0 and bare["residual_rate"] == 1 - L.summarize(model.rows, "model_safety")["recall"]
    assert full["residual_rate"] <= weak["residual_rate"] <= bare["residual_rate"]      # a guardrail never adds risk
    for s in (bare, full, weak):
        assert sum(s["unsafe"].values()) == 12 and sum(s["safe"].values()) == 15
    df = L.combine(model, runs["guard-full"])
    blocked = df[df["guard_blocked"]]
    assert (blocked["latency_ms"] == blocked["guard_ms"]).all()                        # blocked: the model is never called
    passed = df[~df["guard_blocked"]]
    assert ((passed["latency_ms"] - passed["guard_ms"] - passed["model_ms"]).abs() < 1e-6).all()


def test_responses_from_a_model_run_feed_response_direction(results, mock_server, tmp_path):
    from src.analysis.responses import main as to_responses

    model = _by_name(results)["mock-model"]
    out = tmp_path / "answers.csv"
    to_responses([str(model.path), "-o", str(out)])
    df = pd.read_csv(out)
    assert list(df.columns) == ["prompt", "expected", "response"] and len(df) == 27
    assert (df["expected"] == "block").sum() == L.summarize(model.rows, "model_safety")["fn"]
    main(["--config", "src/config/evaluation/guardrail_service.yaml", "--service-url", mock_server.url, "--data", str(out),
          "--input-type", "response", "--set", "target.http_guardrail.api_key_env=null",
          "--set", "target.http_guardrail.canary=false", "--set", "metrics.bootstrap=20",
          "--output-dir", str(tmp_path / "r"), *NOENV])
    (run,) = L.list_runs(tmp_path / "r")
    assert run.input_type == "response" and len(run.rows) == 27


def test_every_app_page_renders(results, monkeypatch):
    pytest.importorskip("streamlit")
    pytest.importorskip("plotly")
    from streamlit.testing.v1 import AppTest

    monkeypatch.setattr(sys, "argv", ["app.py", "--results", str(results)])
    at = AppTest.from_file(str(L.__file__).replace("loader.py", "app.py"), default_timeout=60).run()
    pages = at.sidebar.radio[0].options
    assert len(pages) == 5
    for page in pages:
        at.sidebar.radio[0].set_value(page).run()
        assert not at.exception, f"{page}: {[e.value for e in at.exception]}"
        assert not at.error, f"{page}: {[e.value for e in at.error]}"


def test_service_info_table():
    class R:
        def __init__(self, label, info):
            self.label, self.service_info = label, info

    a = R("a", {"version": "1.2", "model": "guard-4b", "stages": [{"name": "llm", "threshold": 0.5}]})
    b = R("b", {"version": "1.2", "model": "guard-8b", "stages": [{"name": "llm", "threshold": 0.7}], "extra": True})
    diff = L.info_table([a, b])
    assert list(diff["設定"]) == ["extra", "model", "stages.0.threshold"] and list(diff.columns) == ["設定", "a", "b"]
    assert len(L.info_table([a, b], only_differences=False)) == 5
    assert L.flatten({"x": {"y": [1, 2]}}) == {"x.y": [1, 2]}
