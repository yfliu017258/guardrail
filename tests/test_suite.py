"""The suite runner: runs what is missing, skips what is done, keeps going after a failure."""

import json

import yaml

from src.evaluation.suite import main


def _suite(tmp_path, mock_server, **extra):
    doc = {"results": str(tmp_path / "results"), "datasets": {"smoke": "processed_data/samples/smoke.csv"},
           "guardrails": {"g1": {"url": mock_server.url}}, **extra}
    path = tmp_path / "suite.yaml"
    path.write_text(yaml.safe_dump(doc), encoding="utf-8")
    return path


def _runs(tmp_path):
    out = {}
    for p in (tmp_path / "results").iterdir():
        info = json.loads((p / "run.json").read_text())
        out[info["name"]] = info
    return out


def test_suite_runs_missing_and_skips_done(mock_server, tmp_path, capsys):
    args = ["--env-file", "/nonexistent", "--limit", "10"]
    path = _suite(tmp_path, mock_server)
    assert main([str(path), "--dry-run", *args]) == 0 and not (tmp_path / "results").exists()
    assert "would run" in capsys.readouterr().out

    assert main([str(path), *args]) == 0
    assert set(_runs(tmp_path)) == {"g1@smoke"} and _runs(tmp_path)["g1@smoke"]["dataset"]["n"] == 10

    # add a guardrail (unreachable) and one more: only the new ones run; the broken one does not stop the rest
    path = _suite(tmp_path, mock_server, guardrails={"g1": {"url": mock_server.url}, "down": {"url": "http://127.0.0.1:9"},
                                                     "g2": {"url": mock_server.url}})
    capsys.readouterr()
    assert main([str(path), *args]) == 1
    out = capsys.readouterr().out
    assert "already done  guardrail  g1@smoke" in out and "FAILED" in out and "ran           guardrail  g2@smoke" in out
    runs = _runs(tmp_path)
    assert runs["g2@smoke"]["status"] == "completed" and runs["down@smoke"]["status"] == "failed"
    assert sum(1 for p in (tmp_path / "results").iterdir()) == 3            # g1 was not run again

    # another sample limit is other data -> runs again; --only restricts
    assert main([str(path), "--env-file", "/nonexistent", "--limit", "12", "--only", "g2"]) == 0
    assert sum(1 for p in (tmp_path / "results").iterdir()) == 4
