"""End-to-end evaluation against a REAL guardrail server (uvicorn on a free port, mock backend)."""

import json

import pytest

from src.evaluation.__main__ import main
from src.server.testing import BackgroundServer

NOENV = ["--env-file", "/nonexistent"]


def _eval(mock_server, tmp_path, *extra):
    main(["--config", "src/config/evaluation/smoke.yaml", "--server-url", mock_server.url,
          "--output-dir", str(tmp_path), *NOENV, *extra])
    return sorted(p for p in tmp_path.iterdir() if p.is_dir())[-1]


def test_smoke_run_against_server_and_resume(mock_server, tmp_path):
    run_dir = _eval(mock_server, tmp_path)
    for f in ("config.yaml", "data_manifest.json", "task.json", "predictions.jsonl", "metrics.json", "report.md",
              "errors.csv", "slices.csv"):
        assert (run_dir / f).exists(), f
    m = json.loads((run_dir / "metrics.json").read_text())
    assert m["n_total"] == 27 and m["overall"]["recall"] is not None and m["per_stage"]
    task = json.loads((run_dir / "task.json").read_text())
    assert task["type"] == "guardrail_server" and task["base_url"] == mock_server.url
    assert task["server_config_hash"] and {s["name"] for s in task["stages"]} == {"rules", "mock_guard"}
    assert task["server_changed_during_run"] is False
    assert mock_server.url in (run_dir / "report.md").read_text()
    # resume is a no-op on a finished run and keeps the same predictions
    before = (run_dir / "predictions.jsonl").read_text()
    main(["--resume", str(run_dir), *NOENV])
    assert (run_dir / "predictions.jsonl").read_text() == before
    main(["--rescore", str(run_dir), "--set", "metrics.threshold=0.9"])
    (sub,) = [p for p in run_dir.iterdir() if p.name.startswith("rescore_0.9_")]
    assert (sub / "metrics.json").exists() and (sub / "config.yaml").exists()
    main(["--rescore", str(run_dir), "--threshold", "0.2"])          # mapped flags work on rescore too
    assert any(p.name.startswith("rescore_0.2_") for p in run_dir.iterdir())


def test_production_endpoint_gives_the_same_decisions(mock_server, tmp_path):
    a = _eval(mock_server, tmp_path / "classify")
    b = _eval(mock_server, tmp_path / "api", "--endpoint", "guardrail_api")
    load = lambda d: {json.loads(l)["sample_id"]: json.loads(l)["pred_positive"] for l in (d / "predictions.jsonl").open()}
    assert load(a) == load(b)                                      # same server, same decisions
    rows = [json.loads(l) for l in (b / "predictions.jsonl").open()]
    assert {r["action"] for r in rows} <= {"NONE", "BLOCKED", "GUARDRAIL_INTERVENED"} and not any(r["stage_unsafe"] for r in rows)


def test_reproducible_sampling(mock_server, tmp_path):
    ids = []
    for d in ("a", "b"):
        run = _eval(mock_server, tmp_path / d, "--limit", "10")
        ids.append(sorted(json.loads(l)["sample_id"] for l in (run / "predictions.jsonl").open()))
    assert ids[0] == ids[1] and len(ids[0]) == 10


def test_contamination_clean_metrics(mock_server, tmp_path):
    run_dir = _eval(mock_server, tmp_path, "--set", "metrics.contaminated_sources=[smok*]")
    m = json.loads((run_dir / "metrics.json").read_text())
    assert m["contamination"]["excluded_sources"] == ["smoke"] and m["contamination"]["n_clean"] == 0
    assert "## Contamination" in (run_dir / "report.md").read_text()


def test_unreachable_server_fails_fast(tmp_path):
    with pytest.raises(RuntimeError, match="not reachable"):
        main(["--config", "src/config/evaluation/smoke.yaml", "--server-url", "http://127.0.0.1:9",
              "--output-dir", str(tmp_path), *NOENV])


def test_canary_aborts_when_the_server_blocks_nothing(tmp_path):
    with BackgroundServer(["src/config/server/mock.yaml"],
                          overrides=["pipeline.stages.1.keywords={}"]) as srv, pytest.raises(RuntimeError, match="canary"):
        _eval(srv, tmp_path)


def test_api_key_is_required_and_used(tmp_path):
    with BackgroundServer(["src/config/server/mock.yaml"], overrides=["server.api_key_env=GUARDRAIL_API_KEY"],
                          envs={"GUARDRAIL_API_KEY": "k"}) as srv:
        with pytest.raises(RuntimeError, match="API key"):
            _eval(srv, tmp_path / "nokey")
        envf = tmp_path / ".env"
        envf.write_text("GUARDRAIL_API_KEY=k\n")
        main(["--config", "src/config/evaluation/smoke.yaml", "--server-url", srv.url, "--output-dir", str(tmp_path / "ok"),
              "--env-file", str(envf), "--set", "target.guardrail_server.api_key_env=GUARDRAIL_API_KEY"])
        assert any((tmp_path / "ok").iterdir())


def test_inprocess_target_still_available(tmp_path):
    main(["--config", "src/config/evaluation/smoke.yaml", "--target", "inprocess", "--output-dir", str(tmp_path), *NOENV,
          "--set", "target.inprocess.server_config=[src/config/server/mock.yaml]"])
    (run_dir,) = list(tmp_path.iterdir())
    assert json.loads((run_dir / "metrics.json").read_text())["n_total"] == 27


def _rows(run):
    return [json.loads(l) for l in (run / "predictions.jsonl").open()]


@pytest.mark.parametrize("endpoint", ["classify", "guardrail_api"])
def test_live_stage_errors_are_errors_not_safe_on_both_endpoints(tmp_path, endpoint):
    with BackgroundServer(["src/config/server/mock.yaml"], overrides=["pipeline.stages.1.fail_on=ransomware"]) as srv:
        run = _eval(srv, tmp_path, "--endpoint", endpoint, "--set", "eval.retries=0")
    m = json.loads((run / "metrics.json").read_text())
    errored = [r for r in _rows(run) if r["pred_positive"] is None]
    assert m["n_errors"] == len(errored) >= 1 and all("ransomware" in r["text_preview"].lower() for r in errored)


def test_shadow_stage_errors_and_votes_never_decide(tmp_path):
    shadow = ["pipeline.stages.2={name: candidate, type: mock, shadow: true, threshold: 0.0, fail_on: bomb}"]
    with BackgroundServer(["src/config/server/mock.yaml"], overrides=shadow) as srv:
        run = _eval(srv, tmp_path)
    m = json.loads((run / "metrics.json").read_text())
    assert m["n_errors"] == 0 and m["n_scored"] == 27                       # a broken candidate breaks no row
    assert all(r["shadow_stages"] == ["candidate"] or "candidate" not in r["stage_unsafe"] for r in _rows(run))
    main(["--rescore", str(run), *NOENV, "--set", "metrics.score_stage=mock_guard", "--set", "metrics.threshold=0.99"])
    (sub,) = [p for p in run.iterdir() if p.name.startswith("rescore_0.99_")]
    o = json.loads((sub / "metrics.json").read_text())["overall"]
    assert o["fpr"] == 0.0                     # the candidate flags everything, but it is shadow: it does not block


def test_score_stage_and_threshold_warnings(mock_server, tmp_path):
    run = _eval(mock_server, tmp_path / "api", "--endpoint", "guardrail_api")
    main(["--rescore", str(run), *NOENV, "--set", "metrics.score_stage=mock_guard", "--threshold", "0.3"])
    (sub,) = [p for p in run.iterdir() if p.name.startswith("rescore_0.3_")]
    warnings = json.loads((sub / "metrics.json").read_text())["warnings"]
    assert any("score_stage" in w for w in warnings) and any("no row has a score" in w for w in warnings)


def test_rescore_keeps_contamination_from_the_server(tmp_path):
    with BackgroundServer(["src/config/server/mock.yaml"], overrides=["pipeline.stages.1.trained_on=[smok*]"]) as srv:
        run = _eval(srv, tmp_path)
    assert json.loads((run / "metrics.json").read_text())["contamination"]["excluded_sources"] == ["smoke"]
    main(["--rescore", str(run), *NOENV])
    (sub,) = [p for p in run.iterdir() if p.name.startswith("rescore_")]
    assert json.loads((sub / "metrics.json").read_text())["contamination"]["excluded_sources"] == ["smoke"]


def test_resume_against_a_different_server_is_refused_and_recorded(tmp_path):
    with BackgroundServer(["src/config/server/mock.yaml"]) as srv:
        run = _eval(srv, tmp_path, "--limit", "10")
    task = json.loads((run / "task.json").read_text())
    assert task["server_identity"]["pipeline_hash"]
    with BackgroundServer(["src/config/server/mock.yaml"], overrides=["pipeline.strategy=sequential"]) as other:
        cfg = (run / "config.yaml").read_text().replace(task["base_url"], other.url)
        (run / "config.yaml").write_text(cfg)
        with pytest.raises(SystemExit, match="refusing to --resume"):
            main(["--resume", str(run), *NOENV])
        main(["--resume", str(run), "--force", *NOENV])
    task2 = json.loads((run / "task.json").read_text())
    assert task2["server_changed_during_run"] is True and len(task2["server_segments"]) == 2
    assert "changed during the run" in (run / "report.md").read_text()


def test_identical_replicas_on_other_ports_are_the_same_guardrail(tmp_path):
    with BackgroundServer(["src/config/server/mock.yaml"]) as a, \
         BackgroundServer(["src/config/server/mock.yaml"], overrides=["server.port=9999", "server.log_level=debug"]) as b:
        ia = __import__("httpx").get(f"{a.url}/v1/info").json()
        ib = __import__("httpx").get(f"{b.url}/v1/info").json()
    assert ia["pipeline_hash"] == ib["pipeline_hash"] and ia["config_hash"] != ib["config_hash"]



def test_stage_filter_is_what_task_json_reports(mock_server, tmp_path):
    run = _eval(mock_server, tmp_path, "--set", "target.guardrail_server.stages=[mock_guard]")
    task = json.loads((run / "task.json").read_text())
    assert [s["name"] for s in task["stages"]] == ["mock_guard"] and len(task["server_stages"]) == 2


def test_expected_vs_action_including_mask(mock_server, tmp_path):
    run = _eval(mock_server, tmp_path)
    m = json.loads((run / "metrics.json").read_text())
    mask = m["expected_vs_action"]["mask"]
    assert mask["n"] == 3 and mask["match_rate"] == 1.0 and mask["actions"] == {"GUARDRAIL_INTERVENED": 3}
    assert m["overall"]["n_unsafe"] == 12 and m["overall"]["n_safe"] == 15     # block vs pass+mask
    assert "Expected vs server action" in (run / "report.md").read_text()
