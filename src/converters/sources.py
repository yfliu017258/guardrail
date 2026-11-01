"""Data sources: WHERE a converter's raw records come from (one loading strategy per class).

Each class implements only ``BaseConverter.records(split)``; combine it with a mapping
template from ``templates.py`` (put the source first in the bases)::

    class Aegis2(HFDatasetSource, PromptResponseConverter): ...

    HFDatasetSource       load_dataset(hf_id, hf_config, split=split)        the common case
      HFFilesSource       load_dataset(hf_id, data_files=<repo paths>)       per-language dirs, broken viewers
      MultiConfigHFSource one load_dataset per config, rows tagged _config   suites split into configs
    LocalFileSource       options.path: .csv / .jsonl / .json / .parquet    internal / exported data

Sources tag rows with underscore keys (``_config``, ``_<group_key>``) so the mapping can tell
where a row came from without the source and the mapping knowing about each other.

``datasets`` is imported lazily (``pip install .[data]``): the evaluation side and the tests
never need it. Offline boxes: set ``HF_DATASETS_OFFLINE=1`` and point ``HF_HOME`` at a
pre-synced cache (``options.cache_dir`` is passed through).
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any, ClassVar

from ..config import PROJECT_ROOT
from .base import BaseConverter, Record


class HFDatasetSource(BaseConverter):
    """``records(split)`` = ``datasets.load_dataset(hf_id, hf_config, split=split, revision=...)``."""

    def records(self, split: str) -> Iterable[Record]:
        return self.load_hf(self.hf_config, split=split)

    def load_hf(self, config: str | None = None, **kwargs: Any) -> Iterable[Record]:
        """The single place that calls ``load_dataset``, so revision pinning and caching apply to every HF source."""
        if not self.hf_id:
            raise NotImplementedError(f"{type(self).__name__}: no hf_id")
        from datasets import load_dataset  # optional dependency: pip install .[data]

        return load_dataset(self.hf_id, config, revision=self.revision, cache_dir=self.options.get("cache_dir"), **kwargs)


class HFFilesSource(HFDatasetSource):
    """Load raw files of the repo via ``data_files`` instead of a dataset config.

    Needed when the files are laid out per language (Nemotron v3 ``{lang}/{split}.jsonl``,
    RTP-LX ``data/{locale}-*.parquet``) or the Hub viewer / builder is broken. ``data_files``
    patterns may use ``{split}`` (renamed through ``file_split_names``) and ``{<group_key>}``;
    the latter is expanded over ``options[group_option]`` (default ``default_groups``) and
    each row is tagged ``_<group_key>``. ``options.files`` replaces ``data_files`` entirely.
    """

    data_files: ClassVar[tuple[str, ...]] = ()
    file_split_names: ClassVar[Mapping[str, str]] = {}   # source split -> name used in file paths
    group_key: ClassVar[str | None] = None                # placeholder expanded over option values
    group_option: ClassVar[str] = ""                      # option listing the values, e.g. "languages"
    default_groups: ClassVar[tuple[str, ...]] = ()

    def groups(self) -> list[str | None]:
        if self.group_key is None:
            return [None]
        return list(self.options.get(self.group_option, self.default_groups))

    def files_for(self, split: str, group: str | None) -> list[str]:
        patterns = self.options.get("files") or self.data_files
        fields = {"split": self.file_split_names.get(split, split)}
        if self.group_key is not None:
            fields[self.group_key] = group
        return [p.format(**fields) for p in patterns]

    def records(self, split: str) -> Iterator[Record]:
        for group in self.groups():
            for r in self.load_hf(data_files={split: self.files_for(split, group)}, split=split):
                yield r if group is None else {**r, f"_{self.group_key}": group}


class MultiConfigHFSource(HFDatasetSource):
    """One ``load_dataset`` per config (``options.configs``, default ``default_configs``); rows tagged ``_config``.

    For suites published as many configs (PI eval suite, OR-Bench hard/toxic) where the
    config itself often carries the label or the provenance."""

    default_configs: ClassVar[tuple[str, ...]] = ()

    def configs(self) -> list[str]:
        return list(self.options.get("configs", self.default_configs))

    def records(self, split: str) -> Iterator[Record]:
        for cfg in self.configs():
            for r in self.load_hf(cfg, split=split):
                yield {**r, "_config": cfg}


class LocalFileSource(BaseConverter):
    """``options.path`` (relative to the project root): ``.csv`` / ``.jsonl`` / ``.json`` / ``.parquet``.

    One file holds the whole dataset, so the requested split only names the output."""

    def path(self) -> Path:
        p = Path(self.options["path"])
        return p if p.is_absolute() else PROJECT_ROOT / p

    def records(self, split: str) -> Iterable[Record]:
        return read_records(self.path())


def read_records(path: Path) -> Iterator[Record]:
    suf = path.suffix.lower()
    if suf == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as f:  # utf-8-sig: Excel exports start with a BOM
            yield from csv.DictReader(f)
    elif suf == ".jsonl":
        with path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)
    elif suf == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        yield from (data if isinstance(data, list) else data.get("data", []))
    elif suf == ".parquet":
        import pandas as pd

        yield from pd.read_parquet(path).to_dict("records")
    else:
        raise ValueError(f"unsupported file type: {path}")
