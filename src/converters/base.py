"""Converter framework: raw dataset -> ``Sample(prompt, expected)`` rows -> ``processed_data/<name>.csv``.

``BaseConverter.run`` is a template method; a concrete converter only supplies the two
dataset-specific steps, each usually by inheriting a ready-made base::

    records(split)            WHERE the raw rows come from          -> src/converters/sources.py
    convert(record, ...)      HOW one row maps to (prompt, expected) -> src/converters/templates.py

    @register("my_set")
    class MySet(HFDatasetSource, TextLabelConverter):
        hf_id = "org/my-set"
        text_field, label_field, positive_values = "prompt", "is_attack", frozenset({"1"})

Only the REQUEST side is converted: ``prompt`` is what a user sends to the guardrail server and
``expected`` is what the server must do with it (``block`` / ``pass`` / ``mask``, see
``src/data_schema/schema.py``). Everything else is shared here so it cannot drift between
datasets: validation into ``Sample``, OpenCC conversion of the prompt, dedupe by prompt text,
row accounting, seeded subsampling, CSV writing and the provenance manifest.

Output layout (``out_root`` = ``processed_dir``, default ``processed_data``)::

    <name>.csv              the eval output split (``eval_split``, "test") -> source name ``<name>``
    <name>.<split>.csv      any other output split (e.g. ``--splits train`` of a set with a test split)
    <name>.manifest.json    provenance: hf_id, revision, license, options, converted_at and, per output
                            file, its rows, sha256, counts per expected value and the row accounting

Several source splits may feed one output file: ``output_split`` (class) or ``options.split_rename``
(``{validation: test}``) name the output split of a source split; NotInject_one/two/three all land
in ``notinject.csv``.

Row accounting (``manifest.files.<file>.stats``)::

    raw          records read                       = records with >=1 sample + skipped + invalid
    skipped      records deliberately unused (``convert`` returned nothing: filters, unlabelled rows)
    invalid      malformed records (``convert`` raised, e.g. an unknown label; first 5 logged)
    duplicate    samples dropped: same prompt AND same expected as an earlier one (any source split)
    conflict     samples dropped: same prompt, DIFFERENT expected (listed in ``conflicts``); by default
                 every copy goes, ``options.keep_first: true`` keeps the first one
    sampled_out  samples dropped by ``options.sample``
    kept         samples written                    = rows
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
from ..data_schema import EXPECTED, Sample, file_sha256, write_csv
from .sampling import STRATEGIES, sample_rows

log = logging.getLogger("converters")

Record = dict[str, Any]
Converted = Sample | list[Sample] | None

# keys of a `datasets:` YAML entry that steer the job, not the converter -> never stored as options
JOB_KEYS = frozenset({"converter", "enabled", "splits", "limit"})
MAX_LISTED_CONFLICTS = 50  # the manifest lists this many; stats.conflict counts all
EVAL_SPLIT = "test"


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


class BaseConverter(abc.ABC):
    """Template method: ``run`` -> ``collect`` -> ``records`` (source) -> ``convert`` (mapping) -> ``Sample``."""

    registry_name: ClassVar[str] = "base"   # set by CONVERTERS.register
    hf_id: ClassVar[str | None] = None          # HF dataset repo, if any (provenance + ``--list``)
    hf_config: ClassVar[str | None] = None      # dataset config / subset
    default_splits: ClassVar[tuple[str, ...]] = ("test",)
    output_split: ClassVar[str | None] = None   # write every source split into this output split (eval-only sets)
    eval_split: ClassVar[str] = EVAL_SPLIT      # the output split written to ``<name>.csv``
    license: ClassVar[str] = "unknown"          # copy from the dataset card; check before internal use
    gated: ClassVar[bool] = False
    MAPPING_OPTIONS: ClassVar[frozenset[str]] = frozenset()  # dict-valued options: CLI `--set opt.1=x` sets key "1"

    def __init__(self, options: Mapping[str, Any] | None = None, raw_dir: str | Path = "data/raw",
                 revision: str | None = None, name: str | None = None):
        self.options: dict[str, Any] = dict(options or {})
        # output name; differs from registry_name when one class feeds several named outputs (nemotron_safety_v3_zhtw)
        self.name: str = name or self.registry_name
        self.raw_dir = resolve(raw_dir)
        self.revision = revision or self.options.get("revision")
        self.stats: Counter[str] = Counter()
        self.conflicts: list[dict[str, Any]] = []  # of the last ``collect`` / output file
        # source split being converted; lets mapping hooks see it without threading it through every signature
        self.current_split: str | None = None
        self.check_options()

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any], *, name: str | None = None, raw_dir: str | Path = "data/raw",
                    revision: str | None = None) -> BaseConverter:
        """Build from one ``datasets:`` YAML entry (``CONVERTERS.build(entry, type_key="converter")``)."""
        options = {k: v for k, v in cfg.items() if k not in JOB_KEYS}
        return cls(options=options, raw_dir=raw_dir, revision=revision, name=options.get("name", name))

    def check_options(self) -> None:
        """Reject a misconfigured job up front (a bad option would otherwise make every row ``invalid``)."""
        if not isinstance(self.options.get("keep_first", False), bool):
            raise TypeError(f"{self.name}: keep_first must be true / false, got {self.options['keep_first']!r}")
        spec = self.options.get("sample")
        if isinstance(spec, Mapping) and spec.get("strategy", "stratified") not in STRATEGIES:
            raise ValueError(f"{self.name}: sample.strategy must be one of {STRATEGIES}, got {spec['strategy']!r}")

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
        """Output split of a source split: ``options.split_rename`` > ``output_split`` > the split itself."""
        return (self.options.get("split_rename") or {}).get(split) or self.output_split or split

    def output_path(self, out_root: str | Path, out_split: str) -> Path:
        """``<name>.csv`` for the eval split (its source name is ``<name>``), ``<name>.<split>.csv`` otherwise."""
        suffix = "" if out_split == self.eval_split else f".{out_split}"
        return resolve(out_root) / f"{self.name}{suffix}.csv"

    def manifest_path(self, out_root: str | Path) -> Path:
        return resolve(out_root) / f"{self.name}.manifest.json"

    def sample(self, prompt: Any, expected: str) -> Sample:
        """A validated sample (a blank prompt or an expected value outside block/pass/mask raises).

        ``options.opencc: s2twp`` converts the prompt Simplified -> Taiwan Traditional (with Taiwan
        phrasing). Machine conversion is a stop-gap for building zh-TW eval data: the manifest records
        the option, and the rows still need human review for vocabulary / cultural fit."""
        conv = self._opencc()
        if conv is not None and isinstance(prompt, str):
            prompt = conv.convert(prompt)
        return Sample(prompt=prompt, expected=expected)

    def _opencc(self) -> Any:
        if not self.options.get("opencc"):
            return None
        if not hasattr(self, "_cc"):
            import opencc  # pip install opencc-python-reimplemented (pure python) or opencc

            self._cc = opencc.OpenCC(self.options["opencc"])
        return self._cc

    def iter_samples(self, split: str, limit: int | None = None) -> Iterator[Sample]:
        """Validated, de-duplicated samples of one source split; ``self.stats`` accounts for every raw row."""
        return iter(self.collect([split], limit))

    def collect(self, splits: Iterable[str], limit: int | None = None) -> list[Sample]:
        """Validated samples of ``splits`` (the source splits feeding one output file), de-duplicated by prompt.

        A repeated prompt with the same ``expected`` is a ``duplicate``. With a different ``expected``
        the prompt has no single ground truth: it is a ``conflict`` and EVERY copy is dropped
        (``options.keep_first: true`` keeps the first one instead). Conflicts land in
        ``self.conflicts`` (-> manifest) either way. ``limit`` caps the unique samples taken from each split.
        """
        splits = list(splits)
        keep_first = bool(self.options.get("keep_first", False))
        first: dict[str, tuple[Sample, str]] = {}
        conflicts: dict[str, dict[str, list[str]]] = {}
        for split in splits:
            n = 0
            for s in self._validated(split):
                prev = first.get(s.prompt)
                if prev is None:
                    first[s.prompt] = (s, split)
                    n += 1
                    if limit and n >= limit:
                        break
                elif prev[0].expected == s.expected:
                    self.stats["duplicate"] += 1
                else:
                    c = conflicts.setdefault(s.prompt, {"expected": [prev[0].expected], "splits": [prev[1]]})
                    for key, value in (("expected", s.expected), ("splits", split)):
                        if value not in c[key]:
                            c[key].append(value)
                    self.stats["conflict"] += 1
        if conflicts and not keep_first:
            self.stats["conflict"] += len(conflicts)  # the first copy goes too
        samples = [s for prompt, (s, _) in first.items() if keep_first or prompt not in conflicts]
        self.stats["kept"] += len(samples)
        self.conflicts = [{"prompt": prompt[:200], "key": first[prompt][0].key, **c, "kept_first": keep_first}
                          for prompt, c in conflicts.items()]
        if conflicts:
            log.warning("%s%s: %d prompts with conflicting expected values (%s), e.g. %s", self.name, splits,
                        len(conflicts), "first kept" if keep_first else "all copies dropped", self.conflicts[:3])
        return samples

    def _validated(self, split: str) -> Iterator[Sample]:
        """Every sample ``convert`` produces for one source split (counts raw / invalid / skipped records)."""
        self.current_split = split
        for idx, rec in enumerate(self.records(split)):
            self.stats["raw"] += 1
            try:
                out = self.convert(dict(rec), split, idx)
            except Exception as e:  # noqa: BLE001 - keep going; a few malformed rows shouldn't kill a 100k conversion
                self.stats["invalid"] += 1
                if self.stats["invalid"] <= 5:
                    log.warning("%s[%s] row %d skipped: %s", self.name, split, idx, e)
                continue
            produced = out if isinstance(out, list) else ([] if out is None else [out])
            if not produced:
                self.stats["skipped"] += 1
                continue
            yield from produced

    def subsample(self, samples: list[Sample]) -> list[Sample]:
        """``options.sample: {n, seed: 42, strategy: stratified}`` (or just ``sample: n``): a seeded sample of
        one output file, taken after dedupe, instead of ``limit``'s stream head (``limit`` stops reading
        after N rows — for debugging only, it skews whatever the source happens to list first).
        ``stratified`` keeps the ratio of ``expected`` values; ``balanced`` / ``random`` / ``head`` work as
        in eval mixes (src/converters/sampling.py). Picked rows keep their source order."""
        spec = self.options.get("sample")
        if not spec:
            return samples
        spec = {"n": spec} if isinstance(spec, int) else dict(spec)
        picked = {s.prompt for s in sample_rows(samples, int(spec["n"]), spec.get("strategy", "stratified"),
                                                int(spec.get("seed", 42)))}
        out = [s for s in samples if s.prompt in picked]  # prompts are unique after ``collect``
        dropped = len(samples) - len(out)
        if dropped:
            self.stats["sampled_out"] += dropped
            self.stats["kept"] -= dropped
        return out

    def run(self, splits: Iterable[str] | None, out_root: str | Path = "processed_data",
            limit: int | None = None) -> list[Path]:
        """Convert ``splits`` (default ``default_splits``): one CSV per output split + ``<name>.manifest.json``."""
        written = []
        manifest: dict[str, Any] = {
            "converter": self.registry_name, "name": self.name, "hf_id": self.hf_id, "hf_config": self.hf_config,
            "revision": self.revision, "license": self.license, "gated": self.gated, "options": self.options,
            "limit": limit, "converted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "files": {},
        }
        groups: dict[str, list[str]] = {}
        for split in splits or self.default_splits:
            groups.setdefault(self.output_name(split), []).append(split)
        # each file entry carries its own provenance: a later invocation may convert other splits with
        # another revision / options / limit, and the top-level fields only describe that last invocation
        provenance = {k: manifest[k] for k in ("revision", "hf_config", "options", "limit", "converted_at")}
        for out_split, src_splits in groups.items():
            self.stats = Counter()
            samples = self.subsample(self.collect(src_splits, limit))
            stats = +self.stats  # drop zero counters
            path = self.output_path(out_root, out_split)
            write_csv(samples, path)
            entry: dict[str, Any] = {
                **provenance, "split": out_split, "source_splits": src_splits, "rows": len(samples),
                "sha256": file_sha256(path),
                "expected": {e: sum(s.expected == e for s in samples) for e in EXPECTED},
                "stats": dict(stats),
            }
            if self.conflicts:
                entry["conflicts"] = self.conflicts[:MAX_LISTED_CONFLICTS]
            manifest["files"][path.name] = entry
            log.info("%s%s -> %s (%d rows, %s)", self.name, src_splits, path, len(samples), dict(stats))
            written.append(path)
        mpath = self.manifest_path(out_root)
        if mpath.exists():  # keep entries (with their own provenance) for files converted in earlier invocations
            old = json.loads(mpath.read_text(encoding="utf-8"))
            manifest["files"] = {**old.get("files", {}), **manifest["files"]}
        mpath.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return written


CONVERTERS: Registry[BaseConverter] = Registry(
    "dataset converter", base=BaseConverter,
    lazy_modules=[f"{__package__}.datasets.{m}"
                  for m in ("content_safety", "injection", "over_refusal", "pii", "local_file")],
)
register = CONVERTERS.register  # ``@register("name")`` kept for existing converter modules / external code


def get_converter(name: str) -> type[BaseConverter]:
    return CONVERTERS.get(name)
