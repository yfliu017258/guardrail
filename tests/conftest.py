import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def mock_server():
    """A real guardrail server (mock backend, no auth) on a free localhost port, shared by the session."""
    from src.server.testing import BackgroundServer

    with BackgroundServer(["src/config/server/mock.yaml"]) as srv:
        yield srv
