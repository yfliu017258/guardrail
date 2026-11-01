import pytest

from src.data_schema import GuardrailSample, detect_language, read_csv, write_csv


def test_roundtrip(tmp_path):
    rows = [
        GuardrailSample(source="t", prompt='含 "引號", 逗號\n換行', label="unsafe", categories=["jailbreak", "crime"],
                        language="zh-TW", meta={"orig": 1}),
        GuardrailSample(source="t", input_type="response", prompt="q", response="a", label="safe", language="en"),
    ]
    p = tmp_path / "x.csv"
    write_csv(rows, p)
    assert read_csv(p) == rows


def test_validation():
    with pytest.raises(ValueError):
        GuardrailSample(source="t", prompt="x", label="safe", categories=["crime"])
    with pytest.raises(ValueError):
        GuardrailSample(source="t", input_type="response", prompt="x", label="safe")
    with pytest.raises(ValueError):
        GuardrailSample(source="t", prompt="x", label="unsafe", categories=["not_a_category"])


def test_stable_ids():
    a = GuardrailSample(source="t", prompt="hello", label="safe")
    b = GuardrailSample(source="t", prompt="hello", label="safe")
    assert a.sample_id == b.sample_id and a.sample_id.startswith("t:")


@pytest.mark.parametrize("text,lang", [("請問這個怎麼設定", "zh-TW"), ("请问这个怎么设置", "zh-CN"), ("hello world", "en"), ("", "unknown")])
def test_language(text, lang):
    assert detect_language(text) == lang
