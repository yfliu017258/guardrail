"""Generic converter for local files (internal red-team sets, labelled production logs, SaaS exports).

Options (YAML or ``--set``)::

    path: data/raw/internal_redteam.jsonl          # .csv / .jsonl / .json / .parquet
    columns: {prompt: question, expected: verdict}   # a column that already holds block / pass / mask
    # or a raw label column plus a mapping onto block / pass / mask:
    columns: {prompt: question, label: is_attack}
    label_map: {"1": block, "0": pass, pii: mask}

Without ``columns`` the file's own ``prompt`` column is used, with ``expected`` (or else ``label``)
as the label column. A raw value is looked up in ``label_map`` and must end up as ``block`` /
``pass`` / ``mask``; anything else makes the row ``invalid`` (never a silent ``pass``).

pandas / parquet / spreadsheet exports are handled: null cells (None, NaN, pd.NA, blank) are
missing values — a missing prompt or label makes the row ``invalid``, never the text ``"nan"`` —
and labels compare through ``label_key`` (``1`` / ``1.0`` / ``"1.0"`` / ``True`` / ``np.int64(1)``
all match the ``label_map`` key ``"1"``; ``" Block "`` matches ``block``).
"""

from __future__ import annotations

from typing import Any, ClassVar

from ...data_schema import EXPECTED, Expected, Sample
from ..base import Record, register
from ..sources import LocalFileSource
from ..templates import is_missing, label_key

COLUMN_KEYS = ("prompt", "expected", "label")


def _value(v: Any) -> Any:
    """A cell as a plain Python value; ``None`` when missing."""
    if hasattr(v, "item") and type(v).__module__ == "numpy" and getattr(v, "ndim", 0) == 0:
        v = v.item()
    return None if is_missing(v) else v


@register("local_file")
class LocalFileConverter(LocalFileSource):
    """Column-mapped local file: the mapping lives in the options, so one class serves any internal export."""

    default_splits = ("test",)
    license = "internal"
    MAPPING_OPTIONS: ClassVar[frozenset[str]] = frozenset({"columns", "label_map"})  # --set label_map.1=block

    def check_options(self) -> None:
        super().check_options()
        cols = self.options.get("columns") or {}
        if unknown := sorted(set(cols) - set(COLUMN_KEYS)):
            raise ValueError(f"{self.name}: columns maps only {COLUMN_KEYS}, got {unknown}")
        if "expected" in cols and "label" in cols:
            raise ValueError(f"{self.name}: columns: give either `expected` or `label`, not both")
        for raw, value in (self.options.get("label_map") or {}).items():
            if str(value).strip().lower() not in EXPECTED:
                raise ValueError(f"{self.name}: label_map[{raw!r}] = {value!r}; values must be one of {EXPECTED}")

    def _columns(self, r: Record) -> tuple[str, str]:
        cols = self.options.get("columns") or {}
        label = cols.get("expected") or cols.get("label") or ("expected" if "expected" in r else "label")
        return cols.get("prompt", "prompt"), label

    def convert(self, r: Record, split: str, idx: int) -> Sample:
        prompt_col, label_col = self._columns(r)
        prompt = _value(r.get(prompt_col))
        if prompt is None:
            raise ValueError(f"missing prompt (column {prompt_col!r})")
        return self.sample(str(prompt), self._expected(r.get(label_col), label_col))

    def _expected(self, value: Any, column: str) -> Expected:
        label_map = {label_key(k): str(v).strip().lower() for k, v in (self.options.get("label_map") or {}).items()}
        raw = label_key(value)  # raises "missing label" for None / NaN / blank
        expected = label_map.get(raw, raw)
        if expected not in EXPECTED:
            raise ValueError(f"unmapped {column} value {value!r}; add {raw!r}: block|pass|mask to label_map")
        return expected  # type: ignore[return-value]
