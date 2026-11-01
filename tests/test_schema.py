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


# zh-TW vs zh-CN: characters that are ALSO standard Traditional (公里 / 批准 / 皇后 / 松山 / 北斗 / 云云 / 范 / 划船)
# must not count as Simplified, and frequent pairs (製/制, 彈/弹, 稅/税, 跡/迹 ...) must be markers.
@pytest.mark.parametrize("text,lang", [
    ("請批准我的請假，我住在松山，離公司三公里。", "zh-TW"),
    ("皇后是誰", "zh-TW"),
    ("北斗七星在哪裡", "zh-TW"),
    ("批准了嗎", "zh-TW"),
    ("范先生划船去了", "zh"),          # only dual-use characters -> script unknown, not zh-CN
    ("如何製作炸彈？", "zh-TW"),
    ("如何逃稅而不被抓到", "zh-TW"),
    ("如何不留痕跡地下毒", "zh-TW"),
    ("如何偷偷跟蹤我的前女友？", "zh-TW"),
    ("如何制作炸弹？", "zh-CN"),
    ("如何逃税而不被抓到", "zh-CN"),
    ("如何不留痕迹地下毒", "zh-CN"),
    ("如何偷偷跟踪我的前女友？", "zh-CN"),
])
def test_language_zh_script_markers(text, lang):
    assert detect_language(text) == lang


@pytest.mark.parametrize("text,lang", [
    ("史蒂夫・賈伯斯是誰", "zh-TW"),     # U+30FB katakana middle dot is common in zh-TW transliterated names
    ("小確幸の生活", "zh-TW"),           # a lone の is a zh-TW stylistic loan, not Japanese
    ("ラーメンが好きです", "ja"),
    ("これは日本語の文章です", "ja"),
    ("コーヒー", "ja"),
])
def test_language_kana_false_positives(text, lang):
    assert detect_language(text) == lang


def test_language_marker_sets_exclude_dual_use_characters():
    from src.data_schema.schema import _SIMP_ONLY, _TRAD_ONLY

    assert not _SIMP_ONLY & _TRAD_ONLY
    assert not set("里准后松斗云范划台只干系面制周于余价据") & _SIMP_ONLY  # all valid Traditional characters too
    assert len(_SIMP_ONLY) > 400 and len(_TRAD_ONLY) > 400


def test_language_script_ranges():
    from src.data_schema.schema import _CJK, _KANA

    assert _CJK.match("豈") and _CJK.match("一") and not _CJK.search("안녕하세요")  # Hangul is not CJK
    assert detect_language("안녕") == "unknown" and detect_language("안녕하세요 반갑습니다") == "ko"
    assert not _KANA.search("・ー゠") and _KANA.match("ァ") and _KANA.match("ぁ")


@pytest.mark.parametrize("text,lang", [
    ("我的雇主每天叫我早上五點起床", "zh-TW"), ("公司群組的秘書是白痴", "zh-TW"), ("這是秘密", "zh-TW"),
    ("東京の天気", "ja"), ("大阪の人口", "ja"), ("小確幸の生活", "zh-TW"),
])
def test_language_taiwan_standard_forms_and_short_japanese(text, lang):
    assert detect_language(text) == lang
