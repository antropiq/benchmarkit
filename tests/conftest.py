from typing import Iterator

import pytest

from src.core.job import RunJob


@pytest.fixture(autouse=True)
def _no_mcp_processes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never spawn real MCP connector subprocesses from app lifespans in tests."""
    monkeypatch.setenv("BENCHMARKIT_DISABLE_MCP", "1")


@pytest.fixture(autouse=True)
def _no_lsp_processes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never spawn real language servers from app lifespans in tests."""
    monkeypatch.setenv("BENCHMARKIT_DISABLE_LSP", "1")


@pytest.fixture(autouse=True)
def _no_legacy_tool_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the tool-loop cap deterministic.

    ``create_app()`` loads the developer's own ``.env`` once, so a legacy
    ``MAX_MCP_ITERATION`` (still honoured as a fallback by
    ``config.max_tool_iteration()``) or an ``LSP_ENABLED`` list would
    otherwise leak into every test and depend on the machine it runs on.
    """
    monkeypatch.delenv("MAX_TOOL_ITERATION", raising=False)
    monkeypatch.delenv("MAX_MCP_ITERATION", raising=False)
    monkeypatch.delenv("LSP_ENABLED", raising=False)


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
