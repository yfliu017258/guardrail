import json

from src.evaluation.__main__ import main


def test_smoke_run_and_resume(tmp_path):
    main(["--config", "src/config/evaluation/smoke.yaml", "--output-dir", str(tmp_path), "--env-file", "/nonexistent"])
    (run_dir,) = list(tmp_path.iterdir())
    for f in ("config.yaml", "data_manifest.json", "predictions.jsonl", "metrics.json", "report.md", "errors.csv", "slices.csv"):
        assert (run_dir / f).exists(), f
    m = json.loads((run_dir / "metrics.json").read_text())
    assert m["n_total"] == 30 and m["overall"]["recall"] is not None
    # resume is a no-op on a finished run and keeps the same predictions
    before = (run_dir / "predictions.jsonl").read_text()
    main(["--resume", str(run_dir), "--env-file", "/nonexistent"])
    assert (run_dir / "predictions.jsonl").read_text() == before
    main(["--rescore", str(run_dir), "--set", "metrics.threshold=0.9"])
    assert (run_dir / "rescore_0.9" / "metrics.json").exists()


def test_reproducible_sampling(tmp_path):
    args = ["--config", "src/config/evaluation/smoke.yaml", "--limit", "10", "--env-file", "/nonexistent"]
    main(args + ["--output-dir", str(tmp_path / "a")])
    main(args + ["--output-dir", str(tmp_path / "b")])
    ids = [[json.loads(l)["sample_id"] for l in next((tmp_path / d).iterdir()).joinpath("predictions.jsonl").open()]
           for d in ("a", "b")]
    assert sorted(ids[0]) == sorted(ids[1]) and len(ids[0]) == 10


def test_contamination_clean_metrics(tmp_path):
    main(["--config", "src/config/evaluation/smoke.yaml", "--output-dir", str(tmp_path), "--env-file", "/nonexistent",
          "--set", "metrics.contaminated_sources=[smok*]"])
    (run_dir,) = list(tmp_path.iterdir())
    m = json.loads((run_dir / "metrics.json").read_text())
    assert m["contamination"]["excluded_sources"] == ["smoke"] and m["contamination"]["n_clean"] == 0
    assert "## Contamination" in (run_dir / "report.md").read_text()
