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
    assert runs["g2@smoke"]["status"] == "completed" and "down@smoke" not in runs   # nothing to keep from a dead start
    assert sum(1 for p in (tmp_path / "results").iterdir()) == 2            # g1 was not run again

    # another sample cap is other data -> runs again; --only takes <subject>@<dataset>
    assert main([str(path), "--env-file", "/nonexistent", "--max-samples", "12", "--only", "g2@smoke"]) == 0
    assert sum(1 for p in (tmp_path / "results").iterdir()) == 3
    # extra rounds for what failed: still down, still reported, no leftovers
    assert main([str(path), *args, "--only", "down", "--retries", "2", "--retry-wait", "0"]) == 1
    assert sum(1 for p in (tmp_path / "results").iterdir()) == 3


def test_suite_resumes_an_interrupted_run(mock_server, tmp_path, capsys):
    args = ["--env-file", "/nonexistent"]
    path = _suite(tmp_path, mock_server, datasets={"smoke": {"path": "processed_data/samples/smoke.csv", "max_samples": 12}})
    assert main([str(path), *args]) == 0
    (run_dir,) = (tmp_path / "results").iterdir()
    full = (run_dir / "predictions.jsonl").read_text().splitlines()
    assert len(full) == 12                                                  # the per-dataset cap

    # the process died after 5 rows (the last one half-written)
    (run_dir / "predictions.jsonl").write_text("\n".join(full[:5]) + "\n" + full[5][:40])
    info = json.loads((run_dir / "run.json").read_text())
    (run_dir / "run.json").write_text(json.dumps({**info, "status": "running"}))
    capsys.readouterr()
    assert main([str(path), "--dry-run", *args]) == 0 and f"resume {run_dir.name}" in capsys.readouterr().out
    assert main([str(path), *args]) == 0
    assert "resumed" in capsys.readouterr().out
    assert [p.name for p in (tmp_path / "results").iterdir()] == [run_dir.name]          # continued in place
    rows = [json.loads(l) for l in (run_dir / "predictions.jsonl").read_text().splitlines()]
    assert len(rows) == 12 and len({r["sample_id"] for r in rows}) == 12                 # 5 kept, 7 sent, none twice
    assert [json.dumps(r, ensure_ascii=False) for r in rows[:5]] == full[:5]
    assert json.loads((run_dir / "run.json").read_text())["status"] == "completed"

    # rows that failed in transit on a COMPLETED run are re-sent only on request
    rows[0] = {**rows[0], "pred_positive": None, "error": "ConnectError", "_retry": True}
    (run_dir / "predictions.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    assert main([str(path), *args]) == 0 and "already done" in capsys.readouterr().out
    assert main([str(path), "--retry-errors", *args]) == 0 and "resumed" in capsys.readouterr().out
    latest = {}
    for l in (run_dir / "predictions.jsonl").read_text().splitlines():
        latest[json.loads(l)["sample_id"]] = json.loads(l)
    assert len(latest) == 12 and not any(r.get("_retry") for r in latest.values())
