"""Run a real guardrail server in a background thread (tests, notebooks, local experiments).

    with BackgroundServer(["src/config/server/mock.yaml"]) as srv:
        httpx.post(f"{srv.url}/v1/classify", json={"text": "..."})

The server is the same ``create_app`` + uvicorn stack as ``python -m src.server`` (HTTP, auth,
startup/shutdown hooks), just bound to a free localhost port.
"""

from __future__ import annotations

import socket
import threading
import time
from typing import Any, Iterable

import uvicorn

from ..config import load_config
from .app import create_app


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class BackgroundServer:
    def __init__(self, config: Iterable[str] | str, overrides: Iterable[str] = (), envs: dict[str, str] | None = None,
                 port: int | None = None, startup_timeout: float = 30.0):
        self.cfg: dict[str, Any] = load_config(config, overrides=list(overrides), env_file=None, run_kind="server")
        if envs is not None:
            self.cfg["envs"] = dict(envs)
        self.port = port or free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        self.startup_timeout = startup_timeout
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> "BackgroundServer":
        app = create_app(self.cfg)
        self._server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=self.port, log_level="warning",
                                                     lifespan="on"))
        self._thread = threading.Thread(target=self._server.run, name=f"guardrail-server-{self.port}", daemon=True)
        self._thread.start()
        deadline = time.monotonic() + self.startup_timeout
        while not self._server.started:
            if not self._thread.is_alive():
                raise RuntimeError("guardrail server exited during startup (see the log above)")
            if time.monotonic() > deadline:
                raise TimeoutError(f"guardrail server did not start within {self.startup_timeout}s")
            time.sleep(0.02)
        return self

    def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=10)

    def __enter__(self) -> "BackgroundServer":
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()
