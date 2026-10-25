"""Tests for batch mode's backend surface: the lot listing, ``batchContext``
and the ``POST /run`` exclusion of a streaming chat reply.

The workspace is always a throw-away ``tmp_path`` pointed at by
``BENCHMARKIT_PATH_LOCATION`` (the pattern ``tests/test_api.py`` uses for the
hub endpoints), so nothing here reads the repository's own ``todos/``.
"""

import json
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
from fastapi.testclient import TestClient

from src.api import routes
from src.api.server import create_app
from src.core import batch as batch_core
from src.core import chat as chat_core


def make_lot(
    root: Path,
    names: List[str],
    dir: str = "todos",
    done: bool = True,
) -> Path:
    """Create ``dir`` under ``root`` holding ``names`` (headings included)."""
    folder = root / dir
    folder.mkdir(parents=True, exist_ok=True)
    for name in names:
        path = folder / name
        if path.parent != folder:  # nested name → subfolder file
            path.parent.mkdir(parents=True, exist_ok=True)
        title = path.stem.replace("-", " ")
        path.write_text(f"# {title}\n\nBody of {name}.\n", encoding="utf-8")
    if done:
        (folder / "done").mkdir(exist_ok=True)
    return folder


@pytest.fixture
def batch_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """App whose working location (and so its lot folder) is ``tmp_path``."""
    monkeypatch.setenv("BENCHMARKIT_PATH_LOCATION", str(tmp_path))
    with TestClient(create_app()) as client:
        yield client


def listing(client: TestClient, prefix: str, dir: Optional[str] = None):
    params: Dict[str, str] = {"prefix": prefix}
    if dir is not None:
        params["dir"] = dir
    return client.get("/api/batch/tasks", params=params)


class TestTaskNumbering:
    def test_parses_int_normalised_and_refuses_glued_digits(self):
        assert batch_core.parse_task_number("ed-01-workspace", "ed-") == 1
        assert batch_core.parse_task_number("ed-007", "ed-") == 7
        assert batch_core.parse_task_number("ed-00", "ed-") == 0
        assert batch_core.parse_task_number("ed-10", "ed-") == 10
        # No number, and a digit run that is not its own token.
        assert batch_core.parse_task_number("ed-foo", "ed-") is None
        assert batch_core.parse_task_number("ed-1x", "ed-") is None
        assert batch_core.parse_task_number("ed", "ed-") is None
        assert batch_core.skipped_reason("ed-foo", "ed-") == (
            "no task number after the prefix"
        )
        assert "not a separate token" in batch_core.skipped_reason("ed-1x", "ed-")

    def test_lot_block_frames_the_text_as_context(self):
        block = chat_core.batch_instructions_block("  build the editor  ")
        assert block.endswith("build the editor")
        assert "one task at a time" in block


class TestTaskListingEndpoint:
    def test_full_lot_counts_numbers_and_carries_the_big_picture(
        self, batch_client: TestClient, tmp_path: Path
    ):
        names = ["ed-00-big-picture.md"] + [
            f"ed-{n:02d}-task.md" for n in range(1, 9)
        ]
        make_lot(tmp_path, names)
        body = listing(batch_client, "ed-").json()

        assert body["dir"] == "todos"
        assert body["workspace"] == str(tmp_path.resolve())
        assert body["prefix"] == "ed-"
        assert body["taskCount"] == 8
        assert body["doneFolder"] is True
        assert body["truncated"] is False
        assert body["lotNote"] is None
        assert [task["number"] for task in body["tasks"]] == list(range(1, 9))
        assert body["tasks"][0]["name"] == "ed-01-task.md"
        assert body["tasks"][0]["size"] > 0
        assert body["tasks"][0]["heading"] == "ed 01 task"
        # The exact string the per-task message interpolates.
        assert all(
            task["taskLocation"] == f"todos/{task['name']}"
            for task in body["tasks"]
        )
        assert body["lot"]["name"] == "ed-00-big-picture.md"
        assert body["lot"]["path"] == "todos/ed-00-big-picture.md"
        assert "Body of ed-00-big-picture.md." in body["lot"]["text"]

    def test_gaps_leave_no_phantom_entries(self, batch_client, tmp_path):
        make_lot(tmp_path, ["ed-01-a.md", "ed-05-b.md"])
        body = listing(batch_client, "ed-").json()
        assert body["taskCount"] == 5
        assert [task["number"] for task in body["tasks"]] == [1, 5]
        assert body["skipped"] == []
        assert body["lot"] is None

    def test_numberless_matches_are_skipped_others_absent(
        self, batch_client, tmp_path
    ):
        make_lot(
            tmp_path,
            [
                "ed-1x-weird.md",
                "ed-notes.md",
                "opt-orphan.md",
                "ed-03-task.txt",
            ],
        )
        body = listing(batch_client, "ed-").json()
        assert body["taskCount"] == 0
        assert body["tasks"] == []
        reasons = {item["name"]: item["reason"] for item in body["skipped"]}
        assert reasons["ed-notes.md"] == "no task number after the prefix"
        assert "not a separate token" in reasons["ed-1x-weird.md"]
        # Outside the lot by definition: never mentioned at all.
        assert "opt-orphan.md" not in reasons
        assert "ed-03-task.txt" not in reasons

    def test_two_and_three_digit_numbers_sort_numerically(
        self, batch_client, tmp_path
    ):
        make_lot(
            tmp_path,
            ["ed-09-a.md", "ed-10-b.md", "ed-100-c.md", "ed-2-d.md"],
        )
        body = listing(batch_client, "ed-").json()
        assert [task["number"] for task in body["tasks"]] == [2, 9, 10, 100]
        assert body["taskCount"] == 100

    def test_task_cap_truncates_and_reports_the_overflow(
        self, batch_client, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(batch_core, "MAX_BATCH_TASKS", 3)
        make_lot(tmp_path, [f"ed-{n:02d}-t.md" for n in range(1, 9)])
        body = listing(batch_client, "ed-").json()
        assert body["truncated"] is True
        assert len(body["tasks"]) == 3
        assert [task["number"] for task in body["tasks"]] == [1, 2, 3]
        # taskCount stays max(number) of the *reported* tasks, and every lost
        # file is explained rather than silently dropped.
        assert body["taskCount"] == 3
        assert [item["name"] for item in body["skipped"]] == [
            f"ed-{n:02d}-t.md" for n in range(4, 9)
        ]
        assert "MAX_BATCH_TASKS" in body["skipped"][0]["reason"]

    def test_lot_text_over_the_cap_is_dropped_not_rejected(
        self, batch_client, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(batch_core, "MAX_BATCH_LOT_BYTES", 64)
        folder = make_lot(tmp_path, ["ed-00-big-picture.md", "ed-01-a.md"])
        (folder / "ed-00-big-picture.md").write_text("x" * 200, encoding="utf-8")
        response = listing(batch_client, "ed-")
        assert response.status_code == 200
        body = response.json()
        assert body["lot"]["text"] is None
        assert body["lot"]["size"] == 200
        assert "too large to inject" in body["lotNote"]
        assert body["taskCount"] == 1

    def test_no_big_picture_file_means_lot_null(self, batch_client, tmp_path):
        make_lot(tmp_path, ["ed-02-a.md", "ed-07-b.md"])
        body = listing(batch_client, "ed-").json()
        assert body["lot"] is None
        assert body["lotNote"] is None
        assert body["taskCount"] == 7

    def test_subfolders_are_never_listed(self, batch_client, tmp_path):
        make_lot(
            tmp_path,
            ["ed-04-live.md", "done/ed-04-live.md", "sub/ed-99-x.md"],
        )
        body = listing(batch_client, "ed-").json()
        assert [task["number"] for task in body["tasks"]] == [4]
        assert body["skipped"] == []
        assert body["taskCount"] == 4
        # The done/ folder is observed (the instruction's mv target) but its
        # contents stay invisible.
        assert body["doneFolder"] is True

    def test_missing_done_folder_is_reported(self, batch_client, tmp_path):
        make_lot(tmp_path, ["ed-01-a.md"], done=False)
        body = listing(batch_client, "ed-").json()
        assert body["doneFolder"] is False

    def test_empty_lot_is_200_with_zero_tasks(self, batch_client, tmp_path):
        make_lot(tmp_path, ["nothing-here.md"])
        response = listing(batch_client, "ed-")
        assert response.status_code == 200
        body = response.json()
        assert body["taskCount"] == 0
        assert body["tasks"] == []
        assert body["skipped"] == []
        assert body["truncated"] is False


class TestListingPathGuards:
    @pytest.mark.parametrize(
        "prefix",
        [
            "ed-;rm -rf",
            "*",
            "../",
            "ed/",
            "ed ",
            "",
            "é",
            ".hidden",
        ],
    )
    def test_illegal_prefix_returns_422(self, batch_client, tmp_path, prefix):
        make_lot(tmp_path, ["ed-01-a.md"])
        response = listing(batch_client, prefix)
        assert response.status_code == 422
        assert "^[A-Za-z0-9][A-Za-z0-9_-]*$" in response.json()["detail"]

    def test_legal_prefixes_are_accepted(self, batch_client, tmp_path):
        make_lot(tmp_path, ["bt_1-a.md", "x2.md"])
        assert listing(batch_client, "bt_").status_code == 200
        assert listing(batch_client, "x2").status_code == 200

    def test_dir_above_the_workspace_returns_422(self, batch_client, tmp_path):
        make_lot(tmp_path, ["ed-01-a.md"])
        response = listing(batch_client, "ed-", dir="../.ssh")
        assert response.status_code == 422
        assert "must not contain '..'" in response.json()["detail"]

    def test_dir_on_the_connector_deny_list_returns_422(
        self, batch_client, tmp_path
    ):
        # The third lock is what a `..` check cannot do: a secret store *named*
        # inside the workspace stays unreachable, so batch mode never becomes a
        # GUI for reading a tree the connectors may not read either.
        make_lot(tmp_path, ["ed-01-a.md"])
        (tmp_path / ".ssh").mkdir()
        response = listing(batch_client, "ed-", dir=".ssh")
        assert response.status_code == 422
        assert "path policy" in response.json()["detail"]

    def test_dir_traversal_and_absolute_are_422(self, batch_client, tmp_path):
        make_lot(tmp_path, ["ed-01-a.md"])
        for bad in ("..", "todos/../../etc", "/etc", "~/"):
            response = listing(batch_client, "ed-", dir=bad)
            assert response.status_code == 422, bad

    def test_missing_dir_returns_404(self, batch_client, tmp_path):
        response = listing(batch_client, "ed-", dir="nope")
        assert response.status_code == 404
        assert "does not exist" in response.json()["detail"]

    def test_dir_naming_a_file_returns_422(self, batch_client, tmp_path):
        (tmp_path / "todos").write_text("not a folder", encoding="utf-8")
        response = listing(batch_client, "ed-", dir="todos")
        assert response.status_code == 422
        assert "not a directory" in response.json()["detail"]

    def test_dir_default_is_todos(self, batch_client, tmp_path):
        make_lot(tmp_path, ["ed-01-a.md"])
        body = listing(batch_client, "ed-", dir="").json()
        assert body["dir"] == "todos"
        assert len(body["tasks"]) == 1

    def test_dot_dir_is_refused(self, batch_client, tmp_path):
        # A named folder keeps every taskLocation a clean root-relative path.
        make_lot(tmp_path, ["ed-01-a.md"], dir="")
        response = listing(batch_client, "ed-", dir=".")
        assert response.status_code == 422

    def test_nested_lot_dir_is_listed(self, batch_client, tmp_path):
        make_lot(tmp_path, ["ed-01-a.md"], dir="plans/next")
        body = listing(batch_client, "ed-", dir="plans/next").json()
        assert body["dir"] == "plans/next"
        assert body["tasks"][0]["taskLocation"] == "plans/next/ed-01-a.md"

    def test_a_symlinked_task_file_is_skipped_not_read(
        self, batch_client, tmp_path
    ):
        outside = tmp_path.parent / f"{tmp_path.name}-elsewhere"
        outside.mkdir(exist_ok=True)
        secret = outside / "secret.md"
        secret.write_text("# the heading of a file outside the lot\n", encoding="utf-8")
        folder = make_lot(tmp_path, ["ed-01-a.md"])
        (folder / "ed-02-link.md").symlink_to(secret)
        body = listing(batch_client, "ed-").json()
        assert [task["number"] for task in body["tasks"]] == [1]
        assert "ed-02-link.md" in [item["name"] for item in body["skipped"]]
        assert "secret" not in json.dumps(body)

    def test_a_symlinked_dir_cannot_escape_the_workspace(
        self, batch_client, tmp_path
    ):
        outside = tmp_path.parent / f"{tmp_path.name}-outside"
        outside.mkdir(exist_ok=True)
        (outside / "ed-01-secret.md").write_text("# no\n", encoding="utf-8")
        (tmp_path / "todos").symlink_to(outside, target_is_directory=True)
        response = listing(batch_client, "ed-")
        assert response.status_code == 422
        assert "outside the working location" in response.json()["detail"]


def _chat_stream_stub(monkeypatch: pytest.MonkeyPatch, seen: Dict[str, Any]):
    """A ``stream_chat`` double that records the payload it was handed."""

    def fake(messages, model, server_url, emit, sampling=None, **kwargs):
        seen["messages"] = messages
        seen.update(kwargs)
        emit({"event": "done", "took": 0.0, "ttft": 0.0, "tokens": 0,
              "tokensEstimated": False})

    monkeypatch.setattr(routes, "stream_chat", fake)


def _post_chat(client: TestClient, extra: Optional[Dict[str, Any]] = None):
    body: Dict[str, Any] = {"messages": [{"role": "user", "content": "task 1"}]}
    body.update(extra or {})
    return client.post("/api/chat", json=body)


@pytest.fixture
def chat_client(batch_client, tmp_path, monkeypatch):
    """``batch_client`` with a fixed config and an AGENTS.md at the location."""
    from src.core.models import EndpointsConfig

    (tmp_path / "AGENTS.md").write_text("# Project rules\n\nUse pytest.\n",
                                        encoding="utf-8")
    config = EndpointsConfig(serverUrl="http://localhost:8080/v1",
                             endpoint="model-a", sysprompt="Base prompt.")
    monkeypatch.setattr(routes, "load_endpoints_config", lambda: (config, None))
    # The run endpoint reads the real question set; keep it hermetic.
    monkeypatch.setattr(
        routes,
        "load_questions",
        lambda: [{"uuid": "q1", "question": "one?", "expected": "yes"}],
    )
    # A finished run writes results.json: redirect it away from the repo.
    results = tmp_path / "results.json"
    monkeypatch.setattr(routes, "RESULTS_PATH", results)
    monkeypatch.setattr("src.core.utils.RESULTS_PATH", results)
    return batch_client


class TestBatchContextOnChat:
    def test_batch_context_is_forwarded_to_stream_chat(self, chat_client,
                                                       monkeypatch):
        seen: Dict[str, Any] = {}
        _chat_stream_stub(monkeypatch, seen)
        response = _post_chat(chat_client, {"batchContext": "The lot: 8 tasks."})
        assert response.status_code == 200
        assert seen["batch_instructions"] == "The lot: 8 tasks."
        # Sent alongside the project instructions, never merged into sysprompt.
        assert "The lot" not in (seen["sysprompt"] or "")
        assert seen["sysprompt"] == "Base prompt."

    def test_call_shape_without_batch_context_is_unchanged(self, chat_client,
                                                           monkeypatch):
        seen: Dict[str, Any] = {}
        _chat_stream_stub(monkeypatch, seen)
        _post_chat(chat_client)
        assert "batch_instructions" not in seen

    def test_batch_context_over_the_cap_returns_422(self, chat_client,
                                                    monkeypatch):
        monkeypatch.setattr(batch_core, "MAX_BATCH_LOT_BYTES", 32)
        response = _post_chat(chat_client, {"batchContext": "x" * 33})
        assert response.status_code == 422
        assert "batchContext is too large" in response.json()["detail"][0]["msg"]


class TestSystemPromptOrdering:
    """sysprompt → TOOL_NOTICE → AGENTS.md → big picture (spec §1.4)."""

    def _system_message(self, monkeypatch, **kwargs) -> str:
        # The stubbed completion receives the final message list, which is
        # exactly what would be sent upstream.
        captured: Dict[str, Any] = {}
        monkeypatch.setattr(
            "src.core.chat.completion",
            lambda **kwargs: captured.update(kwargs) or iter([]),
        )
        chat_core.stream_chat(
            [{"role": "user", "content": "task 1"}],
            "model-a",
            "http://s:8080/v1",
            lambda event: None,
            **kwargs,
        )
        return captured["messages"][0]["content"]

    def test_big_picture_comes_last_after_agent_instructions(self, monkeypatch):
        content = self._system_message(
            monkeypatch,
            sysprompt="Base prompt.",
            agent_instructions="The current project root is /x\n\nUse pytest.",
            batch_instructions="The lot: tasks 1 to 8.",
            tools=[{"type": "function", "function": {"name": "read_file"}}],
            tool_executor=lambda name, args: "ok",
        )
        order = [
            content.index("Base prompt."),
            content.index(chat_core.TOOL_NOTICE.strip()),
            content.index("The current project root is /x"),
            content.index("The lot: tasks 1 to 8."),
        ]
        assert order == sorted(order)
        assert "one task at a time" in content

    def test_without_batch_instructions_nothing_is_appended(self, monkeypatch):
        # The field must be a no-op when absent or blank: the system message
        # stays byte-identical to what a chat request produced before batch
        # mode existed.
        plain = self._system_message(
            monkeypatch, sysprompt="Base prompt.",
            agent_instructions="Project root /x",
        )
        blank = self._system_message(
            monkeypatch, sysprompt="Base prompt.",
            agent_instructions="Project root /x",
            batch_instructions="   ",
        )
        assert plain == blank
        assert "big-picture" not in plain


class TestRunRefusedWhileChatStreams:
    """``POST /run`` must refuse under a streaming reply (spec §1.5).

    ``POST /chat`` already refuses a running benchmark; without the twin guard
    a batch turn (streaming for minutes) and a benchmark run would generate at
    the same time on one endpoint, with a single _CHAT_HANDLE to pause either.
    """

    def test_run_returns_409_while_a_reply_streams(
        self, chat_client, monkeypatch
    ):
        routes._CHAT_TASKS.add(object())
        try:
            response = chat_client.post("/api/run")
        finally:
            routes._CHAT_TASKS.clear()
        assert response.status_code == 409
        assert response.json()["detail"] == (
            "Cannot start a run while a chat reply is streaming"
        )

    def test_run_still_starts_when_no_reply_streams(
        self, chat_client, monkeypatch
    ):
        from tests.test_api import patch_bench

        patch_bench(monkeypatch)
        assert chat_client.post("/api/run").status_code == 202

    def test_409_while_a_real_reply_is_streaming(self, chat_client, monkeypatch):
        """Same race, driven through an actual streaming reply.

        The TestClient buffers an SSE body before returning, so the second
        request has to come from another thread — exactly as a second browser
        tab or a ``curl`` would do it.
        """
        from tests.test_api import patch_bench

        patch_bench(monkeypatch)
        streaming = threading.Event()
        release = threading.Event()
        results: Dict[str, Any] = {}

        def fake(messages, model, server_url, emit, sampling=None, **kwargs):
            streaming.set()
            release.wait(timeout=5)
            emit({"event": "done", "took": 0.0, "ttft": 0.0, "tokens": 0,
                  "tokensEstimated": False})

        monkeypatch.setattr(routes, "stream_chat", fake)

        def try_run():
            streaming.wait(timeout=5)
            results["run"] = chat_client.post("/api/run")
            results["chat_still_running"] = bool(routes._CHAT_TASKS)
            release.set()

        poster = threading.Thread(target=try_run)
        poster.start()
        chat = chat_client.post("/api/chat",
                                json={"messages": [{"role": "user",
                                                    "content": "task 1"}]})
        poster.join(timeout=5)

        assert chat.status_code == 200
        assert results["run"].status_code == 409
        assert results["chat_still_running"] is True
        # The reply ended normally, and the slot is free again for a run.
        assert "event: done" in chat.text
        assert not routes._CHAT_TASKS
        assert chat_client.post("/api/run").status_code == 202
