"""ws-03 — the leak the reads cutover creates, stated as a test.

Group 1 (server, key, model id, prompt, sampling, ``lspEnabled``, the tool-loop
cap) left ``.env`` for ``<working location>/.nojae/nojae-settings.json``. That
move silently removes the protection of the suite's six
``for key in config_module.SETTING_KEYS`` wipe loops (``test_api.py``,
``test_lsp.py`` twice, ``test_mcp.py``, ``test_mcp_settings_seed.py``,
``test_lsp_settings_seed.py``): those loops wipe *environment* keys, and the
group-1 values no longer live in the environment, so nothing in them stands
between a test and the developer's own settings file — a live ``serverUrl`` and
``serverApiKey``, plus a real language-server list and tool-loop cap. The
replacement guard is the *location*: ``tests/conftest.py``'s autouse
``_no_workspace_settings_leak`` points ``BENCHMARKIT_PATH_LOCATION`` at a
throw-away root, so the lookup can only find a file the test itself wrote.

This file is that invariant rather than a trust in the fixture (spec section
10: "no test can read group-1 values from a location it did not create"). It
asserts it through the module's **own** attributes —
``workspace_settings.workspace_settings_file_path()``,
``WORKSPACE_SETTINGS_TEMPLATE`` — the way every other test in the folder points
at a hub's declared file, so a rename cannot quietly stop these cases exercising
the real path. Nothing here reads the values of, or writes to, this checkout's
own ``.nojae``.

The group-2/3 half of the story is asserted too, and it is what keeps this file
honest through **ws-04**: that task shrinks ``SETTING_KEYS`` to the verifier
block, ``QUESTION_SET``, ``LLM_DEBUG`` and ``BENCHMARKIT_PYTHON_ROOT``, and the
wipe loops must keep protecting exactly those, because those keys *stay* in the
environment. Drop one from the tuple over there and the ambient environment
answers it here, which turns one of the cases below red.
"""

import os
from pathlib import Path

import pytest

from src import config as config_module
from src.config import DEFAULT_MAX_TOOL_ITERATION, load_endpoints_config
from src.core import workspace_settings
from tests.conftest import (
    neutralise_workspace_settings,
    workspace_settings_guard_root,
)


def install_settings_path() -> Path:
    """This checkout's own settings file — the thing a leak would read.

    Built through ``workspace_settings``'s own naming rules against
    ``config.PROJECT_ROOT``, because "the guard points at the new file's own
    constant, not a re-derived path" is what this file exists to show.
    """
    return Path(
        workspace_settings.workspace_settings_file_path(str(config_module.PROJECT_ROOT))
    )


class TestTheGuardIsPointedAtTheNewFilesConstant:
    """The autouse guard neutralises *this* store, by the module's own naming."""

    def test_the_active_lookup_is_the_neutralised_root_not_this_checkout(
        self, tmp_path: Path
    ):
        guard_root = workspace_settings_guard_root(tmp_path)

        # The guard moved the working location off the install directory.
        assert os.path.normcase(config_module.path_location()) == os.path.normcase(
            str(guard_root)
        )
        assert os.path.normcase(str(guard_root)) != os.path.normcase(
            str(config_module.PROJECT_ROOT)
        )
        # And the settings lookup followed it: the file it consulted is under the
        # guard root, not this checkout's — whatever that one holds. A leak
        # reports the install path here.
        state = workspace_settings.workspace_settings_state()
        answered = os.path.normcase(state["path"])
        assert answered.startswith(os.path.normcase(str(guard_root)))
        assert answered != os.path.normcase(str(install_settings_path()))

    def test_the_install_own_file_exists_yet_answers_nothing_here(
        self, tmp_path: Path
    ):
        """The leak case: this checkout has a real file, and it is not read.

        ``WORKSPACE_SETTINGS_TEMPLATE`` lives at the install root, so an install
        that has ever booted also has a real
        ``<install>/.nojae/nojae-settings.json`` — a developer's own model
        connection. On a machine where that file does not exist yet the case
        still proves the invariant (the lookup landed on the guard root), and it
        never prints the file's values: a failure names the guard that broke
        rather than echoing a live API key into the runner's output.
        """
        leaked = install_settings_path()
        state = workspace_settings.workspace_settings_state()

        if leaked.is_file():
            if os.path.normcase(state["path"]) == os.path.normcase(str(leaked)):
                pytest.fail(
                    "the workspace settings guard leaked: the lookup answered "
                    f"{leaked}, i.e. this checkout's own file, not the test's "
                    "throw-away location"
                )
            # The install's file holds a real connection on a used machine; it
            # must not reach this test's mapping either.
            assert workspace_settings.load_workspace_settings() == {}

        # No group-1 value reaches the config either way, so there is nothing to
        # dial and the refusal names the guard root's own path.
        config, error = load_endpoints_config()
        assert config is None
        assert isinstance(error, str)
        assert os.path.normcase(state["path"]) in os.path.normcase(error)


class TestNoTestReadsValuesFromALocationItDidNotCreate:
    """No file of its own → nothing answers; its own file → its values answer."""

    def test_without_a_file_of_its_own_nothing_is_configured(self, tmp_path: Path):
        state = workspace_settings.workspace_settings_state()
        assert state["state"] == "absent"
        assert state["configured"] is False
        assert workspace_settings.load_workspace_settings() == {}

        config, error = load_endpoints_config()
        assert config is None
        assert isinstance(error, str) and state["path"] in error

    def test_a_seeded_file_answers_not_configured_not_a_default(self, tmp_path: Path):
        """A boot seed at the guard root is still no usable connection (D8)."""
        workspace_settings.ensure_workspace_settings_file(
            str(workspace_settings_guard_root(tmp_path))
        )
        state = workspace_settings.workspace_settings_state()
        assert state["state"] == "ok"
        assert state["configured"] is False

        config, error = load_endpoints_config()
        assert config is None
        assert isinstance(error, str) and state["path"] in error

    def test_after_the_test_writes_its_own_file_the_values_answer(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        """The guard is a neutralisation, not a wall: an owned file is read."""
        neutralise_workspace_settings(
            monkeypatch,
            tmp_path,
            serverUrl="http://mine:9090/v1",
            serverApiKey="sk-mine",
            endpoint="my-model",
        )
        assert workspace_settings.workspace_settings_state()["state"] == "ok"

        config, error = load_endpoints_config()
        assert error is None
        assert config is not None
        assert config.serverUrl == "http://mine:9090/v1"
        assert config.apiKey == "sk-mine"
        assert config.model_list() == ["my-model"]


class TestTheWorkspaceSourcedHelpersAreNeutralToo:
    """``lsp_enabled_ids()`` / ``max_tool_iteration()`` answer the code default."""

    def test_no_language_server_is_active_from_an_ambient_file(self):
        # A developer with servers enabled in this checkout's file would see them
        # reach every test — including the two-gate LSP cases — if the guard did
        # not move the location.
        assert config_module.lsp_enabled_ids() == []

    def test_the_tool_loop_cap_is_the_code_default(self):
        # This checkout's file carries the template's 300, so a leak answers 300
        # and every tool-loop assertion in the suite changes meaning.
        assert config_module.max_tool_iteration() == DEFAULT_MAX_TOOL_ITERATION

    def test_an_env_name_cannot_reopen_either_one(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """Both are read from the file only (D2): no ``os.environ`` fallback."""
        monkeypatch.setenv("LSP_ENABLED", "pyright,typescript")
        monkeypatch.setenv("MAX_TOOL_ITERATION", "3")
        monkeypatch.setenv("MAX_MCP_ITERATION", "3")
        assert config_module.lsp_enabled_ids() == []
        assert config_module.max_tool_iteration() == DEFAULT_MAX_TOOL_ITERATION


class TestTheEnvWipeLoopsStillCoverWhatStaysInTheEnv:
    """Group 2/3 is still the environment's, so the loops must keep its keys."""

    def test_the_verifier_block_answers_nothing_after_the_wipe_loop(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        """The six fixtures' loop, run here: group 2 stays neutralised.

        Group 1 is neutralised by the *location*; group 2 (the verifier) is
        still an environment store, so it is neutralised by wiping
        ``SETTING_KEYS`` — which is why ws-04 must keep the four
        ``VERIFIER_AGENT_*`` names inside that tuple while it shrinks it.
        """
        for key in config_module.SETTING_KEYS:
            monkeypatch.delenv(key, raising=False)
            assert key not in os.environ

        # A file of this test's own, so the config is built at all: with the env
        # wiped, the verifier half must be absent rather than ambient.
        neutralise_workspace_settings(monkeypatch, tmp_path)
        config, error = load_endpoints_config()
        assert error is None
        assert config is not None
        assert config.verifierAgent is None

    def test_the_still_env_sourced_keys_are_named_by_the_tuple(self):
        """ws-04 shrinks ``SETTING_KEYS`` to these; it may not drop them.

        Groups 2 and 3 as the spec names them: the verifier block, the question
        set and the two technical keys. They stay in ``.env``, so they stay
        wiped by the loops this file's first class is about.
        """
        for key in (
            "VERIFIER_AGENT_SYSPROMPT",
            "VERIFIER_AGENT_SERVERURL",
            "VERIFIER_AGENT_SERVERURL_API_KEY",
            "VERIFIER_AGENT_ENDPOINT",
            "QUESTION_SET",
            config_module.LLM_DEBUG_KEY,
            "BENCHMARKIT_PYTHON_ROOT",
        ):
            assert key in config_module.SETTING_KEYS


class TestTheTestSettingsSwitchCannotLeakIntoTheSuite:
    """T-D7: the e2e switch is pinned off, so it cannot redirect these reads.

    ``BENCHMARKIT_TEST_SETTINGS`` makes a process read the suite's own
    ``.nojae/nojae-test-settings.json`` — a hand-authored fixture the app never
    seeds. An exported ``1`` in the runner's shell would leave every test here
    reading a file that is not there, failing en masse for a reason no traceback
    explains, so ``tests/conftest.py`` clears the key and these cases pin that.
    """

    @pytest.fixture(autouse=True, scope="module")
    def ambient_switch_is_exported(self):
        """Put the switch in the environment for this module, as a shell would.

        A module-scoped autouse fixture runs *before* the function-scoped guard
        this file is about, so every test here starts with the key exported —
        the exact leak ``tests/conftest.py`` exists to close. Drop that guard and
        the cases below go red on this machine rather than passing quietly, which
        is what makes this a test of the guard and not of the ambient shell.
        """
        saved = os.environ.get(workspace_settings.TEST_SETTINGS_KEY)
        os.environ[workspace_settings.TEST_SETTINGS_KEY] = "1"
        yield
        if saved is None:
            os.environ.pop(workspace_settings.TEST_SETTINGS_KEY, None)
        else:
            os.environ[workspace_settings.TEST_SETTINGS_KEY] = saved

    def test_the_guard_clears_an_ambient_value(self):
        """The exported ``1`` above is gone by the time a test body runs."""
        assert workspace_settings.TEST_SETTINGS_KEY not in os.environ
        assert workspace_settings.test_settings_mode() is False
        assert workspace_settings.active_settings_filename() == (
            workspace_settings.WORKSPACE_SETTINGS_FILENAME
        )
        # And the read side answers the production name, so a group-1 value can
        # only come from a file this test wrote.
        assert workspace_settings.workspace_settings_state()["path"].endswith(
            workspace_settings.WORKSPACE_SETTINGS_FILENAME
        )

    def test_with_the_switch_on_no_test_reads_the_production_file(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        """The mode is reachable per test — and then the workspace file is dead."""
        neutralise_workspace_settings(monkeypatch, tmp_path,
                                      serverUrl="http://dev-live:9/v1")
        monkeypatch.setenv(workspace_settings.TEST_SETTINGS_KEY, "1")
        workspace_settings.invalidate_workspace_settings()

        state = workspace_settings.workspace_settings_state()
        assert state["state"] == "absent"
        assert state["path"].endswith(
            workspace_settings.WORKSPACE_TEST_SETTINGS_FILENAME
        )
        assert workspace_settings.load_workspace_settings() == {}
        assert workspace_settings.workspace_setting("serverUrl") is None

        config, error = load_endpoints_config()
        assert config is None
        assert isinstance(error, str)
        assert workspace_settings.WORKSPACE_TEST_SETTINGS_FILENAME in error

    def test_the_two_names_are_named_by_the_module_not_by_a_literal(self):
        """These cases exercise the real lookup only if the names differ."""
        assert workspace_settings.WORKSPACE_TEST_SETTINGS_FILENAME != (
            workspace_settings.WORKSPACE_SETTINGS_FILENAME
        )
        assert workspace_settings.TEST_SETTINGS_KEY == "BENCHMARKIT_TEST_SETTINGS"
