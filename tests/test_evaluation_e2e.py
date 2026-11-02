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
    assert m["n_total"] == 30 and m["overall"]["recall"] is not None and m["per_stage"]
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
    assert json.loads((run_dir / "metrics.json").read_text())["n_total"] == 30
