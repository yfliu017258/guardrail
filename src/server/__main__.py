"""Run the guardrail service.

    python -m src.server --config src/config/server/default.yaml
    python -m src.server --config src/config/server/default.yaml src/config/server/qwen3guard.yaml --port 8081
    python -m src.server --config src/config/server/mock.yaml --set pipeline.strategy=parallel
"""

from __future__ import annotations

import argparse
import logging

import uvicorn

from ..config import dump_config, get_by_path, load_config, make_run_dir
from .app import create_app

ARG_TO_KEY = {
    "host": "server.host",
    "port": "server.port",
    "log_level": "server.log_level",
    "strategy": "pipeline.strategy",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OJT guardrail server (LiteLLM generic_guardrail_api backend)")
    p.add_argument("--config", nargs="+", default=["src/config/server/default.yaml"], help="YAML files, merged in order")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE", help="dotted override, repeatable")
    p.add_argument("--env-file", default=".env")
    # keep defaults None so they only override YAML when explicitly passed
    p.add_argument("--host", default=None)
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--log-level", default=None)
    p.add_argument("--strategy", choices=["sequential", "parallel"], default=None)
    p.add_argument("--no-snapshot", action="store_true", help="do not write the resolved config to results/server/")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = load_config(args.config, vars(args), ARG_TO_KEY, args.overrides, args.env_file, run_kind="server")
    level = str(get_by_path(cfg, "server.log_level", "info")).upper()
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # it logs every URL (incl. user:pw@) at INFO
    if not args.no_snapshot:
        run_dir = make_run_dir(cfg, root=get_by_path(cfg, "run.output_dir", "results/server"),
                               name=get_by_path(cfg, "run.name", "server"))
        dump_config(cfg, run_dir / "config.yaml")
        logging.getLogger("guardrail").info("resolved config -> %s", run_dir / "config.yaml")
    app = create_app(cfg)
    uvicorn.run(
        app,
        host=get_by_path(cfg, "server.host", "0.0.0.0"),
        port=int(get_by_path(cfg, "server.port", 8080)),
        log_level=level.lower(),
        workers=1,  # model clients/state live in-process; scale with replicas, not uvicorn workers
    )


if __name__ == "__main__":
    main()
