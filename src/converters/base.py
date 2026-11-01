"""Converter framework: raw dataset -> ``GuardrailSample`` rows -> ``processed_data/<name>/<split>.csv``.

``BaseConverter.run`` is a template method; a concrete converter only supplies the two
dataset-specific steps, each usually by inheriting a ready-made base::

    records(split)            WHERE the raw rows come from    -> src/converters/sources.py
    convert(record, ...)      HOW one row maps to our schema   -> src/converters/templates.py

    @register("my_set")
    class MySet(HFDatasetSource, TextLabelConverter):
        hf_id = "org/my-set"
        text_field, label_field, positive_values = "prompt", "is_attack", frozenset({"1"})

Everything else is shared here so it cannot drift between datasets: validation into
``GuardrailSample``, dedupe, row accounting (raw / kept / skipped / invalid / duplicate),
language tagging, OpenCC conversion, CSV writing and the ``manifest.json`` that pins
provenance (HF repo + revision, row counts, label distribution, CSV sha256), so a processed
file can always be traced back and rebuilt.
"""

from __future__ import annotations

import abc
import json
import logging
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar

from ..common import Registry
from ..config import PROJECT_ROOT
from ..data_schema import GuardrailSample, detect_language, file_sha256, write_csv

log = logging.getLogger("converters")

Record = dict[str, Any]
Converted = GuardrailSample | list[GuardrailSample] | None

# keys of a `datasets:` YAML entry that steer the job, not the converter -> never stored as options
JOB_KEYS = frozenset({"converter", "enabled", "splits", "limit"})


class BaseConverter(abc.ABC):
    """Template method: ``run`` -> ``iter_samples`` -> ``records`` (source) -> ``convert`` (mapping)."""

    registry_name: ClassVar[str] = "base"   # set by CONVERTERS.register
    hf_id: ClassVar[str | None] = None          # HF dataset repo, if any (provenance + ``--list``)
    hf_config: ClassVar[str | None] = None      # dataset config / subset
    default_splits: ClassVar[tuple[str, ...]] = ("test",)
    output_split: ClassVar[str | None] = None   # write every source split into <output_split>.csv (eval-only sets)
    license: ClassVar[str] = "unknown"          # copy from the dataset card; check before internal use
    gated: ClassVar[bool] = False

    def __init__(self, options: Mapping[str, Any] | None = None, raw_dir: str | Path = "data/raw",
                 revision: str | None = None, name: str | None = None):
        self.options: dict[str, Any] = dict(options or {})
        # output name; differs from registry_name when one class feeds several named outputs (nemotron_safety_v3_zhtw)
        self.name: str = name or self.registry_name
        self.raw_dir = Path(raw_dir) if Path(raw_dir).is_absolute() else PROJECT_ROOT / raw_dir
        self.revision = revision or self.options.get("revision")
        self.stats: Counter[str] = Counter()
        # source split being converted; lets mapping hooks see it without threading it through every signature
        self.current_split: str | None = None

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any], *, name: str | None = None, raw_dir: str | Path = "data/raw",
                    revision: str | None = None) -> BaseConverter:
        """Build from one ``datasets:`` YAML entry (``CONVERTERS.build(entry, type_key="converter")``)."""
        options = {k: v for k, v in cfg.items() if k not in JOB_KEYS}
        return cls(options=options, raw_dir=raw_dir, revision=revision, name=options.get("name", name))

    # -- the two dataset-specific steps ------------------------------------ #
    @abc.abstractmethod
    def records(self, split: str) -> Iterable[Mapping[str, Any]]:
        """Yield the raw rows of one source split (provided by a ``sources`` base)."""

    @abc.abstractmethod
    def convert(self, record: Record, split: str, idx: int) -> Converted:
        """Map one raw row to zero or more samples (provided by a ``templates`` base).

        Return ``None`` / ``[]`` for rows that are deliberately not used (counted ``skipped``);
        raise for malformed rows (counted ``invalid`` and logged, the conversion continues)."""

    # -- shared ----------------------------------------------------------- #
    def output_name(self, split: str) -> str:
        """Output CSV (and ``GuardrailSample.split``) for a source split: ``options.split_rename`` >
        ``output_split`` > the split itself."""
        return self.options.get("split_rename", {}).get(split) or self.output_split or split

    def sample(self, **kw: Any) -> GuardrailSample:
        """Build a validated sample with ``source`` / ``split`` filled and ``language`` auto-detected.

        ``options.opencc: s2twp`` converts Simplified -> Taiwan Traditional (with Taiwan
        phrasing). Machine conversion is a stop-gap for building zh-TW eval data: rows are
        tagged ``meta.converted`` and still need human review for vocabulary / cultural fit.
        """
        kw.setdefault("source", self.name)
        if self.current_split is not None:
            kw.setdefault("split", self.output_name(self.current_split))
        conv = self._opencc()
        if conv is not None:
            for k in ("prompt", "response"):
                if kw.get(k):
                    kw[k] = conv.convert(kw[k])
            kw["meta"] = {**(kw.get("meta") or {}), "converted": f"opencc_{self.options['opencc']}"}
            kw["language"] = "auto"  # the source language no longer applies to converted text
        if kw.get("language") in (None, "", "auto"):
            kw["language"] = detect_language(kw.get("response") if kw.get("input_type") == "response" else kw.get("prompt"))
        return GuardrailSample(**kw)

    def _opencc(self) -> Any:
        if not self.options.get("opencc"):
            return None
        if not hasattr(self, "_cc"):
            import opencc  # pip install opencc-python-reimplemented (pure python) or opencc

            self._cc = opencc.OpenCC(self.options["opencc"])
        return self._cc

    def iter_samples(self, split: str, limit: int | None = None) -> Iterator[GuardrailSample]:
        """Validated, de-duplicated samples of one source split; ``self.stats`` accounts for every raw row."""
        self.current_split = split
        seen: set[str] = set()
        n = 0
        for idx, rec in enumerate(self.records(split)):
            self.stats["raw"] += 1
            try:
                out = self.convert(dict(rec), split, idx)
            except Exception as e:  # keep going; a few malformed rows shouldn't kill a 100k conversion
                self.stats["invalid"] += 1
                if self.stats["invalid"] <= 5:
                    log.warning("%s[%s] row %d skipped: %s", self.name, split, idx, e)
                continue
            produced = out if isinstance(out, list) else ([] if out is None else [out])
            if not produced:
                self.stats["skipped"] += 1
                continue
            for s in produced:
                if s.sample_id in seen:
                    self.stats["duplicate"] += 1
                    continue
                seen.add(s.sample_id)
                self.stats["kept"] += 1
                yield s
                n += 1
                if limit and n >= limit:
                    return

    def run(self, splits: Iterable[str] | None, out_root: str | Path = "processed_data",
            limit: int | None = None) -> list[Path]:
        """Convert ``splits`` (default ``default_splits``), write one CSV per output split + ``manifest.json``."""
        out_root = Path(out_root) if Path(out_root).is_absolute() else PROJECT_ROOT / out_root
        out_dir = out_root / self.name
        written = []
        manifest: dict[str, Any] = {
            "converter": self.name, "hf_id": self.hf_id, "hf_config": self.hf_config, "revision": self.revision,
            "license": self.license, "gated": self.gated, "options": self.options, "limit": limit,
            "converted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "splits": {},
        }
        # several source splits may land in one output file (e.g. NotInject_one/two/three -> test.csv)
        groups: dict[str, list[str]] = {}
        for split in splits or self.default_splits:
            groups.setdefault(self.output_name(split), []).append(split)
        for out_name, src_splits in groups.items():
            samples, stats, seen = [], Counter(), set()
            for split in src_splits:
                self.stats = Counter()
                for s in self.iter_samples(split, limit):
                    if s.sample_id not in seen:
                        seen.add(s.sample_id)
                        samples.append(s)
                stats.update(self.stats)
            path = out_dir / f"{out_name}.csv"
            write_csv(samples, path)
            manifest["splits"][out_name] = self._split_manifest(samples, src_splits, path, stats)
            log.info("%s%s -> %s (%d rows, %s)", self.name, src_splits, path, len(samples), dict(stats))
            written.append(path)
        mpath = out_dir / "manifest.json"
        if mpath.exists():  # keep entries for splits converted in earlier invocations
            old = json.loads(mpath.read_text(encoding="utf-8"))
            manifest["splits"] = {**old.get("splits", {}), **manifest["splits"]}
        mpath.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return written

    @staticmethod
    def _split_manifest(samples: list[GuardrailSample], src_splits: list[str], path: Path,
                        stats: Counter[str]) -> dict[str, Any]:
        return {
            "source_splits": src_splits, "rows": len(samples), "sha256": file_sha256(path),
            "labels": dict(Counter(s.label.value for s in samples)),
            "input_types": dict(Counter(s.input_type for s in samples)),
            "languages": dict(Counter(s.language for s in samples)),
            "categories": dict(Counter(c.value for s in samples for c in s.categories)),
            "stats": dict(stats),
        }


CONVERTERS: Registry[BaseConverter] = Registry(
    "dataset converter", base=BaseConverter,
    lazy_modules=[f"{__package__}.datasets.{m}"
                  for m in ("content_safety", "injection", "over_refusal", "pii", "local_file")],
)
register = CONVERTERS.register  # ``@register("name")`` kept for existing converter modules / external code


def get_converter(name: str) -> type[BaseConverter]:
    return CONVERTERS.get(name)
