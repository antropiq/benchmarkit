"""Tests for the Editor tab's terminal surface: the WebSocket PTY bridge and the
session registry.

The workspace is a throw-away ``tmp_path`` pointed at by
``BENCHMARKIT_PATH_LOCATION`` (the ``test_editorfs``/``test_batch`` pattern), so
the shell always starts in a clean directory. Process tests are gated on the
shell actually existing on the machine — the guards and the registry tests run
unconditionally (ed-06's own rule), and the teardown grace is shortened through
``termhub.constants`` so a tree-kill assertion stays quick.
"""

import os
import shutil
import time
from pathlib import Path
from typing import Optional

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from src.api.server import create_app
from src.core import termhub

WS_URL = "/api/terminal"
SHELL = os.environ.get("SHELL") or "/bin/sh"
needs_shell = pytest.mark.skipif(
    not shutil.which(SHELL), reason=f"no shell at {SHELL} to spawn"
)


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("BENCHMARKIT_PATH_LOCATION", str(tmp_path))
    return tmp_path


@pytest.fixture
def client(workspace: Path):
    with TestClient(create_app()) as test_client:
        yield test_client


@pytest.fixture
def fast_grace(monkeypatch: pytest.MonkeyPatch) -> None:
    """Shorten the SIGTERM→SIGKILL grace so teardown tests stay quick."""
    monkeypatch.setattr(termhub.constants, "STOP_GRACE_SECONDS", 1.0)


def open_session(client: TestClient, **kwargs):
    """Connect and read the ``open`` frame, returning the live session."""
    session = client.websocket_connect(WS_URL, **kwargs)
    session.__enter__()
    frame = session.receive_json()
    assert frame["type"] == "open", frame
    return session


def wait_until(predicate, timeout=4.0, step=0.05) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(step)
    return predicate()


def pid_gone(pid: Optional[int]) -> bool:
    if not pid:
        return True
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return True
    # A zombie is still "there" to signal 0 but is dead; treat it as gone.
    state = stat.rsplit(")", 1)[1].split()[0]
    return state == "Z"


class TestGuards:
    """Refusals that must happen before any process is spawned."""

    def test_foreign_origin_is_refused_without_spawning(self, client):
        # The refusal is a close before accept, which surfaces from __enter__.
        with pytest.raises(WebSocketDisconnect) as caught:
            with client.websocket_connect(
                WS_URL, headers={"Origin": "https://evil.example"}
            ):
                pass
        assert caught.value.code == termhub.CLOSE_CODE_POLICY
        assert client.get(WS_URL).json() == {"sessions": []}

    @pytest.mark.parametrize(
        "origin", ["http://127.0.0.1:5173", "http://localhost:8000", "http://[::1]:8000"]
    )
    def test_local_origin_is_allowed(self, client, origin, fast_grace):
        session = open_session(client, headers={"Origin": origin})
        try:
            assert session is not None
        finally:
            session.__exit__(None, None, None)

    def test_no_origin_is_allowed(self, client, fast_grace):
        # A deliberately-launched local client (wscat) sends no Origin. The
        # session must be closed, or its portal thread outlives the test and
        # hangs the suite.
        session = open_session(client)
        try:
            assert session is not None
        finally:
            session.__exit__(None, None, None)


class TestRegistry:
    def test_live_session_is_listed_and_empties_after_teardown(self, client, fast_grace):
        assert client.get(WS_URL).json() == {"sessions": []}
        session = open_session(client)
        listed = client.get(WS_URL).json()["sessions"]
        assert len(listed) == 1
        entry = listed[0]
        assert entry["pid"] and entry["cwd"]
        assert entry["shell"] and isinstance(entry["startedAt"], float)
        session.__exit__(None, None, None)
        assert wait_until(lambda: client.get(WS_URL).json() == {"sessions": []})

    def test_session_cap_refuses_the_next_upgrade(self, client, monkeypatch, fast_grace):
        monkeypatch.setattr(termhub.constants, "MAX_SESSIONS", 1)
        first = open_session(client)
        # The second upgrade is refused before accept: the close surfaces from
        # __enter__, and no shell is spawned for it.
        with pytest.raises(WebSocketDisconnect) as caught:
            with client.websocket_connect(WS_URL):
                pass
        assert caught.value.code == termhub.CLOSE_CODE_CAPACITY
        # Only the first session is registered — the refused one spawned nothing.
        assert len(client.get(WS_URL).json()["sessions"]) == 1
        first.__exit__(None, None, None)


@needs_shell
class TestSession:
    def test_input_produces_output_and_shell_stays_open(self, client, fast_grace):
        session = open_session(client)
        session.send_json({"type": "input", "data": "echo benchmarkit-marker\r"})
        # Read output frames until the marker shows up (the shell echoes and
        # re-prompts, so scan across frames rather than expecting one).
        found = False
        for _ in range(60):
            frame = session.receive_json()
            if frame["type"] == "output" and "benchmarkit-marker" in frame["data"]:
                found = True
                break
        assert found, "no output frame carried the echoed marker"
        # Still open: the registry still lists it.
        assert len(client.get(WS_URL).json()["sessions"]) == 1
        session.__exit__(None, None, None)

    def test_exit_frame_reports_the_code(self, client, fast_grace):
        session = open_session(client)
        pid = next(s["pid"] for s in client.get(WS_URL).json()["sessions"])
        session.send_json({"type": "input", "data": "exit 7\r"})
        frame = None
        for _ in range(60):
            frame = session.receive_json()
            if frame["type"] == "exit":
                break
        assert frame and frame["type"] == "exit"
        assert frame["code"] == 7
        assert wait_until(lambda: pid_gone(pid))
        session.__exit__(None, None, None)

    def test_resize_updates_the_window_size(self, client, fast_grace):
        session = open_session(client)
        # Read the open's first output, then resize and read the shell's echo.
        session.send_json({"type": "resize", "cols": 132, "rows": 41})
        session.send_json({"type": "input", "data": "stty size\r"})
        seen = ""
        for _ in range(80):
            frame = session.receive_json()
            if frame["type"] == "output":
                seen += frame["data"]
                if "41 132" in seen:  # `stty size` prints "rows cols"
                    break
        assert "41 132" in seen, f"resize not seen by the shell: {seen[-200:]!r}"
        session.__exit__(None, None, None)

    def test_kill_frame_closes_session_and_kills_pid(self, client, fast_grace):
        session = open_session(client)
        pid = next(s["pid"] for s in client.get(WS_URL).json()["sessions"])
        session.send_json({"type": "kill"})
        assert wait_until(lambda: pid_gone(pid)), f"shell {pid} survived the kill frame"
        assert wait_until(lambda: client.get(WS_URL).json() == {"sessions": []})
        session.__exit__(None, None, None)

    def test_disconnect_tears_down_the_whole_session_tree(self, client, fast_grace):
        # The tree-kill assertion, same shape as the hub teardown tests: start a
        # *backgrounded* job, drop the socket, and require the child to die too.
        session = open_session(client)
        session.send_json({"type": "input", "data": "sleep 40 &\r"})
        time.sleep(0.4)
        shell_pid = next(s["pid"] for s in client.get(WS_URL).json()["sessions"])
        session.__exit__(None, None, None)  # client disconnect

        assert wait_until(lambda: pid_gone(shell_pid)), "shell survived disconnect"

        # Find any descendant still in the shell's session and require it gone.
        def session_empty() -> bool:
            return not termhub_session_members(shell_pid)

        assert wait_until(session_empty, timeout=3.0), (
            "a child of the disconnected shell survived — the teardown killed "
            "the shell but not its tree"
        )


def termhub_session_members(sid: int):
    """Live members of process session ``sid`` (test-local, mirrors proctree)."""
    from src.core import proctree

    return [p for p in proctree.session_members(sid) if not pid_gone(p)]
