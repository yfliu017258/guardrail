from src.evaluation.metrics import auroc, average_precision, binary_metrics, evaluate_rows, percentiles, threshold_sweep


def test_binary():
    m = binary_metrics([True, True, False, False], [True, False, True, False])
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (1, 1, 1, 1)
    assert m["precision"] == 0.5 and m["recall"] == 0.5 and m["fpr"] == 0.5


def test_auroc_ap():
    y = [True, True, False, False]
    assert auroc(y, [0.9, 0.8, 0.2, 0.1]) == 1.0
    assert auroc(y, [0.5, 0.5, 0.5, 0.5]) == 0.5
    assert average_precision(y, [0.9, 0.8, 0.2, 0.1]) == 1.0


def test_sweep():
    s = threshold_sweep([True, True, False, False], [0.9, 0.6, 0.7, 0.1], [0.0, 0.5])
    assert s["recall_at_fpr"]["0"]["recall"] == 0.5
    assert s["recall_at_fpr"]["0.5"]["recall"] == 1.0


def test_percentiles():
    p = percentiles([1, 2, 3, 4, 5])
    assert p["p50"] == 3 and p["mean"] == 3


def _row(gold, pred, score=None, **kw):
    return {"gold_label": gold, "unsafe": pred, "score": score, "latency_ms": 1.0, "source": "s", "language": "en",
            "input_type": "request", **kw}


def test_error_policy():
    rows = [_row("unsafe", True), _row("unsafe", None), _row("safe", False)]
    assert evaluate_rows(rows, n_boot=0)["overall"]["n"] == 2
    assert evaluate_rows(rows, error_policy="as_unsafe", n_boot=0)["overall"]["recall"] == 1.0
    assert evaluate_rows(rows, error_policy="as_safe", n_boot=0)["overall"]["recall"] == 0.5


def test_rethreshold_and_stages():
    rows = [_row("unsafe", False, 0.4, stage_unsafe={"a": True}, stage_scores={"a": 0.4}),
            _row("safe", False, 0.1, stage_unsafe={"a": False}, stage_scores={"a": 0.1})]
    m = evaluate_rows(rows, threshold=0.3, n_boot=50)
    assert m["overall"]["recall"] == 1.0 and m["overall"]["fpr"] == 0.0
    assert m["per_stage"]["a"]["recall"] == 1.0
