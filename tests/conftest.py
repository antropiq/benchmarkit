from typing import Iterator

import pytest

from src.core.job import RunJob


@pytest.fixture(autouse=True)
def _no_mcp_processes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never spawn real MCP connector subprocesses from app lifespans in tests."""
    monkeypatch.setenv("BENCHMARKIT_DISABLE_MCP", "1")


@pytest.fixture(autouse=True)
def _isolated_agent_doc(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Pin the chat agent document to an empty per-test XDG dir.

    Without this, a developer's own ``~/.config/benchmarkit/AGENTS.md`` would
    leak into every chat request made through the app under test, making the
    suite depend on the machine it runs on. Tests that exercise the document
    point ``AGENT_DOC`` at a file of their own.
    """
    monkeypatch.delenv("AGENT_DOC", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))


@pytest.fixture
def job() -> Iterator[RunJob]:
    """Provide a fresh, empty benchmark run job."""
    yield RunJob()
