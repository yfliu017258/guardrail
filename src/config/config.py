"""Config loading shared by the server and the evaluation CLI.

Resolution order (later wins):

    1. YAML files given with ``--config`` (deep-merged left to right; a file may
       declare ``_base_: other.yaml`` to inherit from another file)
    2. CLI args that map to a config key (see ``ARG_TO_KEY`` in each ``__main__``)
    3. ``--set a.b.c=value`` dotted overrides (values parsed as YAML: ``true``, ``3``, ``[a,b]``)

Then:

    * ``.env`` (and the process env for keys listed in ``env_passthrough``) is
      loaded into ``config["envs"]``
    * ``config["meta"]`` gets run provenance (git sha, dirty flag, argv, versions ...)

``dump_config`` writes the resolved dict to ``results/<run>/config.yaml`` with
secret-looking values redacted, so the run is reproducible without leaking keys.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import os
import platform
import re
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml
from dotenv import dotenv_values

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = Path(__file__).resolve().parent

SECRET_PATTERN = re.compile(r"(KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|AUTH)", re.IGNORECASE)
REDACTED = "***REDACTED***"
_MISSING = object()


# --------------------------------------------------------------------------- #
# merge helpers
# --------------------------------------------------------------------------- #
def deep_merge(base: dict, override: Mapping) -> dict:
    """Recursively merge ``override`` into a copy of ``base``. Lists are replaced, not appended."""
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, Mapping) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def get_by_path(cfg: Mapping, dotted: str, default: Any = None) -> Any:
    cur: Any = cfg
    for part in dotted.split("."):
        if isinstance(cur, Mapping) and part in cur:
            cur = cur[part]
        elif isinstance(cur, list) and part.isdigit() and int(part) < len(cur):
            cur = cur[int(part)]
        else:
            return default
    return cur


def set_by_path(cfg: dict, dotted: str, value: Any) -> None:
    """``set_by_path(cfg, "pipeline.stages.0.threshold", 0.7)`` — list indices are supported."""
    parts = dotted.split(".")
    cur: Any = cfg
    for i, part in enumerate(parts[:-1]):
        nxt_is_index = parts[i + 1].isdigit()
        if isinstance(cur, list):
            idx = int(part)
            cur = cur[idx]
            continue
        if part not in cur or cur[part] is None:
            cur[part] = [] if nxt_is_index else {}
        cur = cur[part]
    last = parts[-1]
    if isinstance(cur, list):
        cur[int(last)] = value
    else:
        cur[last] = value


def parse_overrides(overrides: Iterable[str] | None) -> list[tuple[str, Any]]:
    """Parse ``["a.b=1", "c=[x, y]"]`` into ``[("a.b", 1), ("c", ["x", "y"])]`` (values via YAML)."""
    parsed = []
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"--set expects key=value, got {item!r}")
        key, raw = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"empty key in override {item!r}")
        try:
            value = yaml.safe_load(raw) if raw != "" else ""
        except yaml.YAMLError:
            value = raw
        parsed.append((key, value))
    return parsed


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def _resolve_path(p: str | Path, relative_to: Path | None = None) -> Path:
    p = Path(p)
    if p.is_absolute() and p.exists():
        return p
    for base in filter(None, [relative_to, Path.cwd(), PROJECT_ROOT, CONFIG_DIR]):
        cand = (base / p).resolve()
        if cand.exists():
            return cand
    raise FileNotFoundError(f"config file not found: {p}")


def load_yaml(path: str | Path, _seen: set[Path] | None = None) -> dict:
    """Load one YAML file, following ``_base_`` (str or list) inheritance."""
    path = _resolve_path(path)
    _seen = _seen or set()
    if path in _seen:
        raise ValueError(f"circular _base_ include: {path}")
    _seen.add(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a mapping at top level")
    bases = data.pop("_base_", None)
    if bases:
        merged: dict = {}
        for b in [bases] if isinstance(bases, str) else bases:
            merged = deep_merge(merged, load_yaml(_resolve_path(b, path.parent), _seen))
        data = deep_merge(merged, data)
    return data


def load_envs(env_file: str | Path | None = ".env", passthrough: Iterable[str] = ()) -> dict[str, str]:
    """``.env`` values, then selected process env vars (process env wins for passthrough keys)."""
    envs: dict[str, str] = {}
    if env_file:
        p = Path(env_file)
        if not p.is_absolute() and not p.exists():
            p = PROJECT_ROOT / p
        if p.exists():
            envs.update({k: v for k, v in dotenv_values(p).items() if v is not None})
    for k in passthrough:
        if k in os.environ:
            envs[k] = os.environ[k]
    return envs


def load_config(
    config_paths: Iterable[str | Path] | str | Path,
    args: Mapping[str, Any] | None = None,
    arg_to_key: Mapping[str, str] | None = None,
    overrides: Iterable[str] | None = None,
    env_file: str | Path | None = ".env",
    run_kind: str = "run",
) -> dict:
    """Build the resolved config dict.

    Args:
        config_paths: one or more YAML files, merged in order.
        args: parsed argparse namespace as a dict. Only keys present in ``arg_to_key``
            and whose value is not ``None`` override the config (so argparse defaults
            never silently clobber YAML values — keep CLI defaults ``None``).
        arg_to_key: ``{"concurrency": "eval.concurrency", ...}``.
        overrides: ``--set a.b=c`` strings, applied last.
        env_file: path to ``.env``; ``None`` disables.
        run_kind: written to ``meta.kind`` ("server" / "evaluation" / "convert").
    """
    if isinstance(config_paths, (str, Path)):
        config_paths = [config_paths]
    cfg: dict = {}
    sources = []
    for p in config_paths:
        resolved = _resolve_path(p)
        cfg = deep_merge(cfg, load_yaml(resolved))
        sources.append(str(resolved.relative_to(PROJECT_ROOT)) if resolved.is_relative_to(PROJECT_ROOT) else str(resolved))

    applied_args = {}
    for arg_name, key in (arg_to_key or {}).items():
        val = (args or {}).get(arg_name)
        if val is not None:
            set_by_path(cfg, key, val)
            applied_args[arg_name] = val

    applied_overrides = parse_overrides(overrides)
    for key, val in applied_overrides:
        set_by_path(cfg, key, val)

    cfg["envs"] = load_envs(env_file, passthrough=cfg.get("env_passthrough", []))
    cfg["meta"] = build_meta(run_kind, sources, applied_args, applied_overrides)
    cfg["meta"]["config_hash"] = config_hash(cfg)
    return cfg


def env(cfg: Mapping, name: str | None, default: str | None = None) -> str | None:
    """Look up ``name`` in ``config["envs"]`` then the process env."""
    if not name:
        return default
    return cfg.get("envs", {}).get(name) or os.environ.get(name) or default


# --------------------------------------------------------------------------- #
# provenance & dumping
# --------------------------------------------------------------------------- #
def _git(*cmd: str) -> str | None:
    try:
        out = subprocess.run(["git", *cmd], cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=5)
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:  # git missing / not a repo
        return None


def _pkg_versions(names: Iterable[str]) -> dict[str, str | None]:
    out = {}
    for n in names:
        try:
            out[n] = importlib.metadata.version(n)
        except importlib.metadata.PackageNotFoundError:
            out[n] = None
    return out


def build_meta(kind: str, sources: list[str], applied_args: dict, applied_overrides: list) -> dict:
    status = _git("status", "--porcelain")
    return {
        "kind": kind,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config_sources": sources,
        "cli_args": applied_args,
        "cli_overrides": [f"{k}={v}" for k, v in applied_overrides],
        "argv": sys.argv,
        "git": {
            "commit": _git("rev-parse", "HEAD"),
            "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(status) if status is not None else None,
        },
        "host": socket.gethostname(),
        "python": platform.python_version(),
        "packages": _pkg_versions(
            ["pydantic", "fastapi", "httpx", "pandas", "datasets", "transformers", "torch", "vllm", "litellm"]
        ),
    }


def redact(obj: Any, key: str = "") -> Any:
    """Replace values whose key looks secret. ``*_env`` keys hold env-var *names*, so they are kept."""
    if isinstance(obj, Mapping):
        return {k: redact(v, str(k)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [redact(v, key) for v in obj]
    if key and SECRET_PATTERN.search(key) and not key.endswith("_env") and obj not in (None, "", "EMPTY"):
        return REDACTED
    return obj


def config_hash(cfg: Mapping) -> str:
    """Stable hash of the config, excluding ``meta`` and ``envs`` — two runs with equal hash used equal settings."""
    body = {k: v for k, v in cfg.items() if k not in ("meta", "envs")}
    return hashlib.sha256(yaml.safe_dump(body, sort_keys=True, allow_unicode=True).encode()).hexdigest()[:12]


def dump_config(cfg: Mapping, path: str | Path, redact_secrets: bool = True) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = redact(cfg) if redact_secrets else cfg
    data = {**data, "meta": {**data.get("meta", {}), "config_hash": config_hash(cfg)}}
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def make_run_dir(cfg: dict, root: str | Path | None = None, name: str | None = None) -> Path:
    """``results/<UTC timestamp>_<name>_<config hash>/`` — also stored in ``cfg["meta"]["run_dir"]``."""
    root = Path(root or get_by_path(cfg, "run.output_dir", "results"))
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    name = name or get_by_path(cfg, "run.name", "run")
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(name)).strip("-") or "run"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_dir = root / f"{stamp}_{safe}_{config_hash(cfg)}"
    run_dir.mkdir(parents=True, exist_ok=False)
    cfg.setdefault("meta", {})["run_dir"] = str(run_dir)
    return run_dir


def get(cfg: Mapping, dotted: str, default: Any = _MISSING) -> Any:
    """Like ``get_by_path`` but raises a helpful KeyError when the key is required."""
    val = get_by_path(cfg, dotted, _MISSING)
    if val is _MISSING:
        if default is _MISSING:
            raise KeyError(f"missing required config key: {dotted}")
        return default
    return val
