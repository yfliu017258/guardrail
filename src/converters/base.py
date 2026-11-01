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
``GuardrailSample``, dedupe, row accounting, language tagging, OpenCC conversion, seeded
subsampling, CSV writing and the ``manifest.json`` that pins provenance (HF repo + revision,
options, limit, row counts, label distribution, CSV sha256) per output split, so a processed
file can always be traced back and rebuilt.

Row accounting (``manifest.splits.<out>.stats``)::

    raw              records read                        = records with >=1 sample + skipped + invalid
    skipped          records deliberately unused (``convert`` returned nothing)
    invalid          malformed records (``convert`` raised; first 5 logged)
    duplicate        samples dropped: same sample_id AND same gold labels as an earlier one (any source split)
    label_conflict   samples dropped: same sample_id, DIFFERENT label / refusal_label (listed in ``label_conflicts``)
    sampled_out      samples dropped by ``options.sample``
    kept             samples written                     = rows
    skipped_<side> / invalid_<side>   one side of a prompt/response record dropped (the other side kept)
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
from ..data_schema.schema import ZH_LANGUAGES

log = logging.getLogger("converters")

Record = dict[str, Any]
Converted = GuardrailSample | list[GuardrailSample] | None

# keys of a `datasets:` YAML entry that steer the job, not the converter -> never stored as options
JOB_KEYS = frozenset({"converter", "enabled", "splits", "limit"})
# OpenCC configs with a known output script: converted Chinese text is tagged with it, not re-detected
OPENCC_TARGET_LANGUAGE = {"s2tw": "zh-TW", "s2twp": "zh-TW", "t2tw": "zh-TW",
                          "t2s": "zh-CN", "tw2s": "zh-CN", "tw2sp": "zh-CN", "hk2s": "zh-CN"}
LABEL_CONFLICT_POLICIES = ("drop", "keep_first")
MAX_LISTED_CONFLICTS = 50  # manifest lists this many; stats.label_conflict counts all


class BaseConverter(abc.ABC):
    """Template method: ``run`` -> ``iter_samples`` -> ``records`` (source) -> ``convert`` (mapping)."""

    registry_name: ClassVar[str] = "base"   # set by CONVERTERS.register
    hf_id: ClassVar[str | None] = None          # HF dataset repo, if any (provenance + ``--list``)
    hf_config: ClassVar[str | None] = None      # dataset config / subset
    default_splits: ClassVar[tuple[str, ...]] = ("test",)
    output_split: ClassVar[str | None] = None   # write every source split into <output_split>.csv (eval-only sets)
    license: ClassVar[str] = "unknown"          # copy from the dataset card; check before internal use
    gated: ClassVar[bool] = False
    MAPPING_OPTIONS: ClassVar[frozenset[str]] = frozenset()  # dict-valued options: CLI `--set opt.1=x` sets key "1"

    def __init__(self, options: Mapping[str, Any] | None = None, raw_dir: str | Path = "data/raw",
                 revision: str | None = None, name: str | None = None):
        self.options: dict[str, Any] = dict(options or {})
        # output name; differs from registry_name when one class feeds several named outputs (nemotron_safety_v3_zhtw)
        self.name: str = name or self.registry_name
        self.raw_dir = Path(raw_dir) if Path(raw_dir).is_absolute() else PROJECT_ROOT / raw_dir
        self.revision = revision or self.options.get("revision")
        self.stats: Counter[str] = Counter()
        self.label_conflicts: list[dict[str, Any]] = []  # of the last ``iter_samples`` / output split
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
        """Build a validated sample with ``source`` / ``split`` filled and ``language`` auto-detected
        (only when the mapping passes ``"auto"``: a language the source states is kept).

        ``options.opencc: s2twp`` converts Simplified -> Taiwan Traditional (with Taiwan
        phrasing). Machine conversion is a stop-gap for building zh-TW eval data: rows are
        tagged ``meta.converted`` and still need human review for vocabulary / cultural fit.
        Converted Chinese text is tagged with the config's target script (``s2tw*`` -> zh-TW,
        ``*2s`` -> zh-CN, see ``OPENCC_TARGET_LANGUAGE``) instead of being re-detected: the
        detector returns plain ``zh`` for a sentence without script-specific characters.
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
            kw["language"] = self._converted_language(self._judged_text(kw))  # the source language no longer applies
        if kw.get("language") in (None, "", "auto"):
            kw["language"] = detect_language(self._judged_text(kw))
        return GuardrailSample(**kw)

    @staticmethod
    def _judged_text(kw: Mapping[str, Any]) -> str | None:
        return kw.get("response") if kw.get("input_type") == "response" else kw.get("prompt")

    def _converted_language(self, text: str | None) -> str:
        detected = detect_language(text)
        target = OPENCC_TARGET_LANGUAGE.get(str(self.options["opencc"]).removesuffix(".json"))
        return target if target and detected in ZH_LANGUAGES else detected  # an `en` column stays `en`

    def _opencc(self) -> Any:
        if not self.options.get("opencc"):
            return None
        if not hasattr(self, "_cc"):
            import opencc  # pip install opencc-python-reimplemented (pure python) or opencc

            self._cc = opencc.OpenCC(self.options["opencc"])
        return self._cc

    def iter_samples(self, split: str, limit: int | None = None) -> Iterator[GuardrailSample]:
        """Validated, de-duplicated samples of one source split; ``self.stats`` accounts for every raw row."""
        return iter(self.collect([split], limit))

    def collect(self, splits: Iterable[str], limit: int | None = None) -> list[GuardrailSample]:
        """Validated samples of ``splits`` (the source splits feeding one output file), de-duplicated together.

        A repeated ``sample_id`` with the same gold labels (``label`` + ``refusal_label``) is a
        ``duplicate``. With different labels the text has no single ground truth: it is a
        ``label_conflict`` and, by default, EVERY copy is dropped (``options.on_label_conflict:
        keep_first`` keeps the first one instead). Conflicts land in ``self.label_conflicts``
        (-> manifest) either way. ``limit`` caps the unique samples taken from each split.
        """
        policy = self.options.get("on_label_conflict", "drop")
        if policy not in LABEL_CONFLICT_POLICIES:
            raise ValueError(f"on_label_conflict must be one of {LABEL_CONFLICT_POLICIES}, got {policy!r}")
        first: dict[str, tuple[GuardrailSample, str]] = {}
        conflicts: dict[str, dict[str, list]] = {}
        for split in splits:
            n = 0
            for s in self._validated(split):
                prev = first.get(s.sample_id)
                if prev is None:
                    first[s.sample_id] = (s, split)
                    n += 1
                    if limit and n >= limit:
                        break
                elif self._gold(prev[0]) == self._gold(s):
                    self.stats["duplicate"] += 1
                else:
                    c = conflicts.setdefault(s.sample_id, {"labels": [self._gold(prev[0])], "splits": [prev[1]]})
                    for key, value in (("labels", self._gold(s)), ("splits", split)):
                        if value not in c[key]:
                            c[key].append(value)
                    self.stats["label_conflict"] += 1
        keep_first = policy == "keep_first"
        if conflicts and not keep_first:
            self.stats["label_conflict"] += len(conflicts)  # the first copy goes too
        samples = [s for sid, (s, _) in first.items() if keep_first or sid not in conflicts]
        self.stats["kept"] += len(samples)
        self.label_conflicts = [{"sample_id": sid, "labels": [list(g) for g in c["labels"]], "splits": c["splits"],
                                 "kept_first": keep_first} for sid, c in conflicts.items()]
        if conflicts:
            log.warning("%s%s: %d texts with conflicting labels (%s), e.g. %s", self.name, list(splits), len(conflicts),
                        "first kept" if keep_first else "all copies dropped", self.label_conflicts[:3])
        return samples

    @staticmethod
    def _gold(s: GuardrailSample) -> tuple[str, str | None]:
        return s.label.value, s.refusal_label

    def _validated(self, split: str) -> Iterator[GuardrailSample]:
        """Every sample ``convert`` produces for one source split (counts raw / invalid / skipped records)."""
        self.current_split = split
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
            yield from produced

    def subsample(self, samples: list[GuardrailSample]) -> list[GuardrailSample]:
        """``options.sample: {n, seed: 42, strategy: stratified}`` (or just ``sample: n``): a seeded sample of
        one output split, taken after dedupe, instead of ``limit``'s stream head (``limit`` stops reading
        after N rows — for debugging only, it skews whatever the source happens to list first).
        ``stratified`` keeps the label / input_type ratio of the converted rows; ``balanced`` and
        ``random`` work as in eval mixes. Picked rows keep their source order."""
        spec = self.options.get("sample")
        if not spec:
            return samples
        spec = {"n": spec} if isinstance(spec, int) else dict(spec)
        # lazy: the converters package must not import the evaluator at import time
        from ..evaluation.dataset import sample_rows

        picked = {s.sample_id for s in sample_rows(samples, int(spec["n"]), spec.get("strategy", "stratified"),
                                                   int(spec.get("seed", 42)))}
        out = [s for s in samples if s.sample_id in picked]
        dropped = len(samples) - len(out)
        if dropped:
            self.stats["sampled_out"] += dropped
            self.stats["kept"] -= dropped
        return out

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
        # each split entry carries its own provenance: a later invocation may convert other splits with
        # another revision / options / limit, and the top-level fields only describe that last invocation
        provenance = {k: manifest[k] for k in ("revision", "hf_config", "options", "limit", "converted_at")}
        for out_name, src_splits in groups.items():
            self.stats = Counter()
            samples = self.subsample(self.collect(src_splits, limit))
            stats = +self.stats  # drop zero counters
            path = out_dir / f"{out_name}.csv"
            write_csv(samples, path)
            entry = {**provenance, **self._split_manifest(samples, src_splits, path, stats)}
            if self.label_conflicts:
                entry["label_conflicts"] = self.label_conflicts[:MAX_LISTED_CONFLICTS]
            manifest["splits"][out_name] = entry
            log.info("%s%s -> %s (%d rows, %s)", self.name, src_splits, path, len(samples), dict(stats))
            written.append(path)
        mpath = out_dir / "manifest.json"
        if mpath.exists():  # keep entries (with their own provenance) for splits converted in earlier invocations
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
