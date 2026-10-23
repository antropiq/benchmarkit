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
    ``MCP_ENABLED`` is gone from the config entirely (connectors activate only
    through their ``enable`` flag in ``externals.json``); clearing it keeps a
    stale entry in a developer's ``.env`` from confusing the suite.
    """
    monkeypatch.delenv("MAX_TOOL_ITERATION", raising=False)
    monkeypatch.delenv("MAX_MCP_ITERATION", raising=False)
    monkeypatch.delenv("LSP_ENABLED", raising=False)
    monkeypatch.delenv("MCP_ENABLED", raising=False)


@pytest.fixture
def job() -> Iterator[RunJob]:
    """Provide a fresh, empty benchmark run job."""
    yield RunJob()
