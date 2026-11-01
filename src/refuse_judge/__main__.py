"""Judge whether model responses are refusals, outside an eval run (spot checks, re-labelling files).

    python -m src.refuse_judge --list                          # registered judges
    python -m src.refuse_judge --patterns                      # keyword judge's built-in patterns (for disable_builtin)
    python -m src.refuse_judge --prompt "..." --response "..." # default: src/config/refuse_judge/keyword.yaml
    python -m src.refuse_judge --config src/config/refuse_judge/cascade.yaml \\
        --input responses.jsonl --output judged.jsonl --concurrency 16
    python -m src.refuse_judge --config src/config/refuse_judge/llm_judge.yaml --set judge.model=glm-5.3-flash ...

Configs are loaded with ``src.config.load_config`` (YAML merge, ``--set`` overrides, ``.env``) and need
a top-level ``judge: {type: ...}``. Input JSONL rows need ``response`` (a string, or null for an empty
response; ``prompt`` and ``reasoning`` optional; reasoning is never judged); a row without one, or
with a non-string value, stops the run with its line number instead of being mislabelled. Every
input field is kept and the verdict is added as ``verdict``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any, TextIO

from ..config import load_config
from .base import REFUSAL_JUDGES, RefusalJudge, build_judge
from .ensemble import resolve_judge_cfg

DEFAULT_CONFIG = "src/config/refuse_judge/keyword.yaml"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m src.refuse_judge", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--list", action="store_true", help="list registered judges")
    p.add_argument("--patterns", action="store_true", help="list the keyword judge's built-in patterns")
    p.add_argument("--config", nargs="+", default=[DEFAULT_CONFIG], help="judge YAML(s), merged in order")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                   help="dotted override, e.g. judge.model=glm-5.3-flash")
    p.add_argument("--env-file", default=".env")
    p.add_argument("--prompt", default="", help="single-shot: the user prompt")
    p.add_argument("--response", default=None, help="single-shot: the model response")
    p.add_argument("--reasoning", default=None, help="single-shot: reasoning tokens (recorded, never judged)")
    p.add_argument("--input", type=Path, help="JSONL with prompt / response [/ reasoning]")
    p.add_argument("--output", default="-", help="JSONL output path ('-' = stdout)")
    p.add_argument("--concurrency", type=int, default=8)
    return p.parse_args(argv)


def build_from_config(args: argparse.Namespace) -> RefusalJudge:
    cfg = load_config(args.config, overrides=args.overrides, env_file=args.env_file, run_kind="refuse_judge")
    judge_cfg = cfg.get("judge")
    if not isinstance(judge_cfg, dict):
        raise SystemExit(f"{args.config}: needs a top-level `judge: {{type: ...}}` (available: {REFUSAL_JUDGES.names()})")
    return build_judge(resolve_judge_cfg(judge_cfg), cfg)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise SystemExit(f"{path}:{n}: invalid JSON: {e}") from e
            if not isinstance(row, dict):
                raise SystemExit(f"{path}:{n}: expected a JSON object")
            if "response" not in row:  # e.g. "completion"/"output": would be judged as an empty response
                raise SystemExit(f"{path}:{n}: missing `response` (got keys {sorted(row)}); "
                                 f"rows need response [, prompt, reasoning]")
            for key in ("response", "prompt"):
                if row.get(key) is not None and not isinstance(row[key], str):
                    raise SystemExit(f"{path}:{n}: `{key}` must be a string or null, got {type(row[key]).__name__}")
            rows.append(row)
    return rows


async def judge_rows(judge: RefusalJudge, rows: list[dict[str, Any]], concurrency: int) -> list[dict[str, Any]]:
    """Bounded concurrency; output order = input order (so files can be diffed / zipped back)."""
    sem = asyncio.Semaphore(max(1, concurrency))

    async def one(row: dict[str, Any]) -> dict[str, Any]:
        async with sem:
            v = await judge.judge(str(row.get("prompt") or ""), row.get("response"), row.get("reasoning"))
        return {**row, "verdict": v.to_dict()}

    return await asyncio.gather(*(one(r) for r in rows))


def summary(rows: list[dict[str, Any]]) -> str:
    verdicts = [r["verdict"] for r in rows]
    labels = Counter(v["label"] for v in verdicts if not v["error"])
    errors = sum(1 for v in verdicts if v["error"])
    refused = sum(1 for v in verdicts if v["refused"] and not v["error"])
    return (f"judged {len(rows)} rows: {dict(labels)}; refused={refused}; errors={errors}; "
            f"mean latency {sum(v['latency_ms'] for v in verdicts) / max(1, len(verdicts)):.1f} ms")


async def run(judge: RefusalJudge, args: argparse.Namespace,
              rows: list[dict[str, Any]] | None) -> list[dict[str, Any]] | dict[str, Any]:
    """Judged ``rows`` (batch) or one verdict dict (single-shot); file I/O stays outside the event loop."""
    await judge.startup()
    try:
        if rows is not None:
            return await judge_rows(judge, rows, args.concurrency)
        return (await judge.judge(args.prompt, args.response, args.reasoning)).to_dict()
    finally:
        await judge.shutdown()


def _write_jsonl(rows: list[dict[str, Any]], f: TextIO) -> None:
    f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)


def _print_list(out: TextIO) -> None:
    out.writelines(f"  {d['name']:22s} {d['doc']}\n" for d in REFUSAL_JUDGES.describe())


def _print_patterns(out: TextIO) -> None:
    from .keyword import BUILTIN_PATTERNS

    out.writelines(f"  {name:26s} {kind:10s} {weight:<5} {lang:3s} {rx}\n"
                   for name, kind, weight, lang, rx in BUILTIN_PATTERNS)


def main(argv: list[str] | None = None, out: TextIO | None = None) -> None:
    out = out or sys.stdout
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s", stream=sys.stderr)
    logging.getLogger("httpx").setLevel(logging.WARNING)  # it logs every URL (incl. user:pw@) at INFO
    if args.list:
        _print_list(out)
        return
    if args.patterns:
        _print_patterns(out)
        return
    if args.input is None and args.response is None:
        raise SystemExit("give --response (single-shot) or --input file.jsonl (batch); see --help")
    rows = read_jsonl(args.input) if args.input else None
    result = asyncio.run(run(build_from_config(args), args, rows))
    if isinstance(result, dict):
        out.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        return
    if args.output == "-":
        _write_jsonl(result, out)
    else:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            _write_jsonl(result, f)
    print(summary(result), file=sys.stderr)


if __name__ == "__main__":
    main()
