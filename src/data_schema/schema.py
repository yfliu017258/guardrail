"""Unified sample format shared by converters, the server and the evaluator.

Every converter emits ``GuardrailSample`` rows; ``write_csv``/``read_csv`` are the
only functions that touch the CSV layout, so the on-disk format is defined here
and nowhere else.

CSV columns (``CSV_COLUMNS``)::

    sample_id, source, split, input_type, prompt, response, label, categories,
    language, attack_type, meta

* ``input_type`` mirrors LiteLLM's ``GenericGuardrailAPIRequest.input_type``:
  ``request`` = judge the user prompt (pre_call), ``response`` = judge the model
  output (post_call). For ``response`` rows the ``prompt`` is context only.
* ``label`` is the *ground truth for the judged side*.
* ``categories`` is ``|``-joined ``Category`` values (empty for safe rows).
* ``meta`` is a JSON object string holding anything source-specific
  (original label, sub-category, toxicity score, ...).
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Label(str, Enum):
    SAFE = "safe"
    UNSAFE = "unsafe"


class Category(str, Enum):
    """Unified taxonomy. Roughly MLCommons/Llama-Guard hazards + enterprise-specific risks.

    Converters / parsers map their native labels onto these. Keep the list short:
    every extra category is another slice that needs enough samples to be meaningful.
    """

    VIOLENCE = "violence"                    # violent crimes, threats, terrorism
    CRIME = "crime"                          # non-violent crimes, fraud, hacking, drugs
    WEAPONS_CBRN = "weapons_cbrn"            # weapons, chemical/biological/nuclear
    SEXUAL = "sexual"                        # adult sexual content
    CHILD_SAFETY = "child_safety"            # sexual content involving minors / CSAM
    HATE_HARASSMENT = "hate_harassment"      # hate speech, harassment, discrimination
    SELF_HARM = "self_harm"                  # suicide / self-harm / eating disorders
    PRIVACY_PII = "privacy_pii"              # personal data, doxxing
    CONFIDENTIAL = "confidential"            # company secrets / IP / internal-only data (enterprise)
    PROMPT_INJECTION = "prompt_injection"    # instructions hidden in input that try to hijack the system
    JAILBREAK = "jailbreak"                  # role-play / DAN-style policy-bypass attempts
    MISINFORMATION = "misinformation"        # elections, defamation, specialised advice
    OTHER = "other"                          # unsafe but not mappable


InputType = Literal["request", "response"]
LANGUAGES = ("en", "zh-TW", "zh-CN", "zh", "ja", "ko", "mixed", "other", "unknown")

CSV_COLUMNS = [
    "sample_id",
    "source",
    "split",
    "input_type",
    "prompt",
    "response",
    "label",
    "categories",
    "language",
    "attack_type",
    "meta",
]


class GuardrailSample(BaseModel):
    model_config = ConfigDict(use_enum_values=False, extra="forbid")

    sample_id: str = ""
    source: str
    split: str = "test"
    input_type: InputType = "request"
    prompt: str = ""
    response: str | None = None
    label: Label
    categories: list[Category] = Field(default_factory=list)
    language: str = "unknown"
    attack_type: str | None = None   # e.g. direct / indirect / roleplay / obfuscation / multi_turn
    meta: dict[str, Any] = Field(default_factory=dict)

    # -- validation ------------------------------------------------------- #
    @field_validator("categories", mode="before")
    @classmethod
    def _split_categories(cls, v: Any) -> Any:
        if v is None or (isinstance(v, float) and v != v):  # NaN from pandas
            return []
        if isinstance(v, str):
            return [c for c in (s.strip() for s in v.split("|")) if c]
        return v

    @field_validator("meta", mode="before")
    @classmethod
    def _parse_meta(cls, v: Any) -> Any:
        if v in (None, ""):
            return {}
        if isinstance(v, str):
            return json.loads(v)
        return v

    @field_validator("language")
    @classmethod
    def _check_language(cls, v: str) -> str:
        if v not in LANGUAGES:
            raise ValueError(f"language must be one of {LANGUAGES}, got {v!r}")
        return v

    @model_validator(mode="after")
    def _consistency(self) -> "GuardrailSample":
        if self.input_type == "response" and not self.response:
            raise ValueError("input_type=response requires a non-empty response")
        if self.input_type == "request" and not self.prompt:
            raise ValueError("input_type=request requires a non-empty prompt")
        if self.label == Label.SAFE and self.categories:
            raise ValueError("safe samples must not carry unsafe categories")
        if not self.sample_id:
            self.sample_id = self.content_hash()
        return self

    # -- helpers ---------------------------------------------------------- #
    @property
    def text(self) -> str:
        """The text that is actually being judged."""
        return self.response if self.input_type == "response" else self.prompt  # type: ignore[return-value]

    @property
    def is_unsafe(self) -> bool:
        return self.label == Label.UNSAFE

    def content_hash(self) -> str:
        h = hashlib.sha256()
        for part in (self.source, self.input_type, self.prompt, self.response or ""):
            h.update(part.encode("utf-8"))
            h.update(b"\x00")
        return f"{self.source}:{h.hexdigest()[:16]}"

    def to_row(self) -> dict[str, str]:
        return {
            "sample_id": self.sample_id,
            "source": self.source,
            "split": self.split,
            "input_type": self.input_type,
            "prompt": self.prompt,
            "response": self.response or "",
            "label": self.label.value,
            "categories": "|".join(c.value for c in self.categories),
            "language": self.language,
            "attack_type": self.attack_type or "",
            "meta": json.dumps(self.meta, ensure_ascii=False, sort_keys=True) if self.meta else "",
        }

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "GuardrailSample":
        row = {k: (None if v == "" else v) for k, v in row.items() if k in CSV_COLUMNS}
        for k in ("prompt",):
            row[k] = row.get(k) or ""
        return cls(**{k: v for k, v in row.items() if v is not None or k == "response"})


# --------------------------------------------------------------------------- #
# CSV IO
# --------------------------------------------------------------------------- #
def write_csv(samples: Iterable[GuardrailSample], path: str | Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, quoting=csv.QUOTE_ALL)
        w.writeheader()
        for s in samples:
            w.writerow(s.to_row())
            n += 1
    return n


def iter_csv(path: str | Path) -> Iterator[GuardrailSample]:
    csv.field_size_limit(1 << 30)
    with Path(path).open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        missing = set(CSV_COLUMNS) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        for i, row in enumerate(reader):
            try:
                yield GuardrailSample.from_row(row)
            except Exception as e:  # surface the row number — debugging 50k-row CSVs is otherwise miserable
                raise ValueError(f"{path}: row {i + 2}: {e}") from e


def read_csv(path: str | Path) -> list[GuardrailSample]:
    return list(iter_csv(path))


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# language heuristics (no extra deps; swap for opencc / fasttext-langid if needed)
# --------------------------------------------------------------------------- #
_CJK = re.compile(r"[㐀-䶿一-鿿豈-﫿]")
_KANA = re.compile(r"[぀-ヿ]")
_HANGUL = re.compile(r"[가-힯]")
_LATIN = re.compile(r"[A-Za-z]")
# frequent characters that differ between Traditional and Simplified Chinese
_TRAD = set("這個們來時說為對會過還麼開問學國發後裡體點樣機與電話實經關見長應頭區動讓無氣當號將認從現進書買賣請錢車門員專業務資產權處統輸給標戶報單種識語類準確際變計訊網線錯寫讀層難歡邊傳測試據環範歷價選擇構級盡導參響觀視飛雲華總兒筆鐘階壓鬆嗎碼腦檔運遠達違規則細紀錄議該設備記憶驗證圖庫數驟寫據換隨屬歸擊殺惡軟碼爾啟礙難劃慮態戲獎鬥雙優隊鍵價簡歲醫藥療護練習節義靈興舉歐齡燈聽覺誤謝詞課談證識讚贊轉輕載邊鄉錄鏈閱隱須顯風飯馬魚鳥麗齒龍")
_SIMP = set("这个们来时说为对会过还么开问学国发后里体点样机与电话实经关见长应头区动让无气当号将认从现进书买卖请钱车门员专业务资产权处统输给标户报单种识语类准确际变计讯网线错写读层难欢边传测试据环范历价选择构级尽导参响观视飞云华总儿笔钟阶压松吗码脑档运远达违规则细纪录议该设备记忆验证图库数骤写据换随属归击杀恶软码尔启碍难划虑态戏奖斗双优队键价简岁医药疗护练习节义灵兴举欧龄灯听觉误谢词课谈证识赞赞转轻载边乡录链阅隐须显风饭马鱼鸟丽齿龙")
_TRAD_ONLY = _TRAD - _SIMP
_SIMP_ONLY = _SIMP - _TRAD


def detect_language(text: str | None) -> str:
    if not text:
        return "unknown"
    cjk = len(_CJK.findall(text))
    latin = len(_LATIN.findall(text))
    if _KANA.search(text) and len(_KANA.findall(text)) > 0.1 * max(cjk, 1):
        return "ja"
    if len(_HANGUL.findall(text)) > 5:
        return "ko"
    if cjk == 0:
        return "en" if latin else "unknown"
    if latin > 3 * cjk:
        return "mixed" if cjk >= 5 else "en"
    trad = sum(ch in _TRAD_ONLY for ch in text)
    simp = sum(ch in _SIMP_ONLY for ch in text)
    if trad > simp:
        return "zh-TW"
    if simp > trad:
        return "zh-CN"
    return "zh"
