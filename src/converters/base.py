"""Converter framework: raw dataset -> ``GuardrailSample`` rows -> ``processed_data/<name>/<split>.csv``.

A converter only has to implement ``load_raw`` (yield raw records) and ``convert``
(raw record -> sample(s) or None). The base class handles validation, dedupe,
language tagging, writing and the ``manifest.json`` that pins provenance
(HF repo + revision, row counts, label distribution, CSV sha256), so a processed
file can always be traced back and rebuilt.
"""

from __future__ import annotations

import abc
import json
import logging
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

from ..config import PROJECT_ROOT
from ..data_schema import GuardrailSample, detect_language, file_sha256, write_csv

log = logging.getLogger("converters")

CONVERTERS: dict[str, type["BaseConverter"]] = {}


def register(name: str) -> Callable[[type["BaseConverter"]], type["BaseConverter"]]:
    def deco(cls: type[BaseConverter]) -> type[BaseConverter]:
        cls.name = name
        CONVERTERS[name] = cls
        return cls

    return deco


class BaseConverter(abc.ABC):
    name: str = "base"
    hf_id: str | None = None          # HF dataset repo, if any
    hf_config: str | None = None      # dataset config / subset
    default_splits: tuple[str, ...] = ("test",)
    output_split: str | None = None   # write every source split into <output_split>.csv (eval-only sets)
    license: str = "unknown"          # copy from the dataset card; check before internal use
    gated: bool = False

    def __init__(self, options: dict[str, Any] | None = None, raw_dir: str | Path = "data/raw",
                 revision: str | None = None):
        self.options = options or {}
        self.raw_dir = Path(raw_dir) if Path(raw_dir).is_absolute() else PROJECT_ROOT / raw_dir
        self.revision = revision or self.options.get("revision")
        self.stats: Counter[str] = Counter()

    # -- to implement ----------------------------------------------------- #
    @abc.abstractmethod
    def convert(self, record: dict[str, Any], split: str, idx: int) -> GuardrailSample | list[GuardrailSample] | None:
        ...

    def load_raw(self, split: str) -> Iterable[dict[str, Any]]:
        """Default: ``datasets.load_dataset(hf_id, hf_config, split=split, revision=...)``.

        Offline boxes: set ``HF_DATASETS_OFFLINE=1`` and point ``HF_HOME`` at a pre-synced
        cache, or override this to read ``self.raw_dir / <file>`` (parquet/jsonl/csv).
        """
        if not self.hf_id:
            raise NotImplementedError(f"{self.name}: no hf_id; override load_raw")
        from datasets import load_dataset  # optional dependency: pip install .[data]

        ds = load_dataset(self.hf_id, self.hf_config, split=split, revision=self.revision,
                          cache_dir=self.options.get("cache_dir"))
        return ds

    # -- shared ----------------------------------------------------------- #
    def sample(self, **kw: Any) -> GuardrailSample:
        """Build a sample with ``source`` filled and ``language`` auto-detected when not given.

        ``options.opencc: s2twp`` converts Simplified -> Taiwan Traditional (with Taiwan
        phrasing). Machine conversion is a stop-gap for building zh-TW eval data: rows are
        tagged ``meta.converted`` and still need human review for vocabulary / cultural fit.
        """
        kw.setdefault("source", self.name)
        conv = self._opencc()
        if conv is not None:
            for k in ("prompt", "response"):
                if kw.get(k):
                    kw[k] = conv.convert(kw[k])
            kw.setdefault("meta", {})["converted"] = f"opencc_{self.options['opencc']}"
            kw["language"] = "auto"
        if kw.get("language") in (None, "", "auto"):
            kw["language"] = detect_language(kw.get("response") if kw.get("input_type") == "response" else kw.get("prompt"))
        return GuardrailSample(**kw)

    def _opencc(self):
        if not self.options.get("opencc"):
            return None
        if not hasattr(self, "_cc"):
            import opencc  # pip install opencc-python-reimplemented (pure python) or opencc

            self._cc = opencc.OpenCC(self.options["opencc"])
        return self._cc

    def iter_samples(self, split: str, limit: int | None = None) -> Iterator[GuardrailSample]:
        seen: set[str] = set()
        n = 0
        for idx, rec in enumerate(self.load_raw(split)):
            self.stats["raw"] += 1
            try:
                out = self.convert(dict(rec), split, idx)
            except Exception as e:  # keep going; a few malformed rows shouldn't kill a 100k conversion
                self.stats["invalid"] += 1
                if self.stats["invalid"] <= 5:
                    log.warning("%s[%s] row %d skipped: %s", self.name, split, idx, e)
                continue
            if out is None:
                self.stats["skipped"] += 1
                continue
            for s in out if isinstance(out, list) else [out]:
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
            out_name = self.options.get("split_rename", {}).get(split) or self.output_split or split
            groups.setdefault(out_name, []).append(split)
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
            manifest["splits"][out_name] = {
                "source_splits": src_splits, "rows": len(samples), "sha256": file_sha256(path),
                "labels": dict(Counter(s.label.value for s in samples)),
                "input_types": dict(Counter(s.input_type for s in samples)),
                "languages": dict(Counter(s.language for s in samples)),
                "categories": dict(Counter(c.value for s in samples for c in s.categories)),
                "stats": dict(stats),
            }
            log.info("%s%s -> %s (%d rows, %s)", self.name, src_splits, path, len(samples), dict(stats))
            written.append(path)
        mpath = out_dir / "manifest.json"
        if mpath.exists():  # keep entries for splits converted in earlier invocations
            old = json.loads(mpath.read_text(encoding="utf-8"))
            manifest["splits"] = {**old.get("splits", {}), **manifest["splits"]}
        mpath.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return written


def get_converter(name: str) -> type[BaseConverter]:
    from . import datasets as _  # noqa: F401  (populate registry)

    if name not in CONVERTERS:
        raise ValueError(f"unknown converter {name!r}; available: {sorted(CONVERTERS)}")
    return CONVERTERS[name]
