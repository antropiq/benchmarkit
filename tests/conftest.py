from typing import Iterator

import pytest

from src.core.job import RunJob


@pytest.fixture(autouse=True)
def _no_mcp_processes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never spawn real MCP connector subprocesses from app lifespans in tests."""
    monkeypatch.setenv("BENCHMARKIT_DISABLE_MCP", "1")


@pytest.fixture
def job() -> Iterator[RunJob]:
    """Provide a fresh, empty benchmark run job."""
    yield RunJob()
