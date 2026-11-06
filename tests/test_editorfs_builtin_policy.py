"""Tests for pol-03: level A — the rules benchmarkit has about **itself**.

Pure ``mode_for`` calls, plus the two checks that make the task a proven
non-event: the **agreement and the measured deltas** against the predicates that
are still live (the old path stays the live one until pol-06 wires the verdict),
and a route-level pin that the live path still refuses the app's own ``.env`` —
this task must not open anything.

The workspace is always pointed at by ``BENCHMARKIT_PATH_LOCATION`` — never by
patching ``path_location`` itself. Stubbing that function would stub away the
single comparison this task is about (:func:`is_self_workspace`), so a suite that
patched it could not tell a conditional level A from an unconditional one.

Both sides of that comparison are read at call time (the location from the
environment, the install dir from ``editorfs.constants``), so the shapes the task
names — the app's parent, a sibling copy — are built as real paths rather than
mocked, which is what keeps a live level A and a dormant one tellable apart.
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple

import pytest
from fastapi.testclient import TestClient

from mcp_connectors import policy_shared as shared

from src.api.server import create_app
from src.core import editorfs
from src.core import prompt_helpers
from src.core.editorfs import builtin_policy
from src.core.editorfs import policy_file, policy_seed
from src.core.editorfs import policy_modes

APP_ROOT = os.path.realpath(str(editorfs.constants.PROJECT_ROOT))


@pytest.fixture
def app_workspace(monkeypatch: pytest.MonkeyPatch) -> None:
    """The workspace *is* this installation — level A is live."""
    monkeypatch.setenv("BENCHMARKIT_PATH_LOCATION", APP_ROOT)


@pytest.fixture
def foreign_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    """The workspace is somewhere else — level A is dormant."""
    monkeypatch.setenv("BENCHMARKIT_PATH_LOCATION", str(tmp_path))
    return tmp_path


def _root_names(client: TestClient) -> List[str]:
    """The root listing's entry names — the tree half of the D8 pair."""
    return [
        entry["name"] for entry in client.get("/api/fs/tree").json()["entries"]
    ]


def copy_of_the_repo(root: Path) -> Path:
    """A directory that looks like the app, somewhere else.

    The comparison is on paths, not contents, so the stand-in needs the app's
    *shape* rather than its weight: the same leaf name, and the names level A is
    written about inside it. A rule that had been left unconditional — or written
    as a basename or prefix test — would find ``.env`` hidden here as well.
    """
    (root / "src" / "core").mkdir(parents=True)
    (root / ".env").write_text("ENDPOINT=x\n", encoding="utf-8")
    (root / "logs" / "trash").mkdir(parents=True)
    (root / "web" / "dist").mkdir(parents=True)
    return root


class TestIsSelfWorkspace:
    def test_the_install_directory_itself_is_self(self, app_workspace):
        assert builtin_policy.is_self_workspace() is True

    @pytest.mark.requires_symlink(
        "is_self_workspace compares realpaths, so a working location spelled "
        "through a link to the install tree is still the app itself"
    )
    def test_a_symlinked_location_for_the_same_tree_is_still_self(
        self, tmp_path: Path, monkeypatch
    ):
        """The location may be spelled through a link; both sides resolve.

        This is why the comparison uses ``workspace_root()`` (a ``realpath``)
        rather than the raw environment value."""
        link = tmp_path / "benchmarkit-via-link"
        link.symlink_to(APP_ROOT, target_is_directory=True)
        monkeypatch.setenv("BENCHMARKIT_PATH_LOCATION", str(link))
        assert builtin_policy.is_self_workspace() is True

    @pytest.mark.parametrize("which", ["parent", "copy", "near", "tmp"])
    def test_everything_else_is_not_self(
        self, which: str, monkeypatch, tmp_path: Path
    ):
        """Exact, not a prefix — and not merely because the path is far away."""
        candidates: List[Tuple[str, Optional[Path]]] = [
            ("parent", Path(APP_ROOT).parent),
            ("copy", copy_of_the_repo(tmp_path / "benchmarkit-copy")),
            # A directory *inside* the app is not the app either: the comparison
            # is equality, so the parent rule cannot leak into a sub-folder.
            ("near", Path(APP_ROOT) / "src"),
            ("tmp", tmp_path),
        ]
        location = dict(candidates)[which]
        assert location is not None
        monkeypatch.setenv("BENCHMARKIT_PATH_LOCATION", str(location))
        assert builtin_policy.is_self_workspace() is False

    def test_an_unset_location_lands_on_the_app(self, monkeypatch):
        """``path_location()`` falls back to the install dir, so level A is live
        from the first request — the mixed state is the normal one."""
        monkeypatch.delenv("BENCHMARKIT_PATH_LOCATION", raising=False)
        assert builtin_policy.is_self_workspace() is True


class TestSelfWorkspaceRules:
    """The rules that only apply while benchmarkit edits itself."""

    @pytest.mark.parametrize(
        "rel,pattern",
        [
            (".env", "**/.env**"),
            (".env.test", "**/.env**"),
            (".env.example", "**/.env**"),
            ("src/.env", "**/.env**"),
            ("web/.env.local", "**/.env**"),
            ("logs/trash/a/b.txt", "logs/**"),
            ("logs/editor_backups/x.orig", "logs/**"),
            ("web/dist/index.html", "web/dist/**"),
            ("web/dist/assets/app.js", "web/dist/**"),
        ],
    )
    def test_an_app_owned_path_carries_its_rule_verbatim(
        self, app_workspace, rel: str, pattern: str
    ):
        verdict = policy_modes.mode_for(rel)
        assert verdict.level == shared.LEVEL_BUILTIN
        assert verdict.pattern == pattern

    @pytest.mark.parametrize(
        "rel,mode",
        [
            (".env", shared.HIDDEN),
            (".env.test", shared.HIDDEN),
            ("web/.env.local", shared.HIDDEN),
            ("logs/trash/x", shared.HIDDEN),
            ("web/dist/index.html", shared.READ_ONLY),
        ],
    )
    def test_the_mode_each_rule_gives(self, app_workspace, rel: str, mode: str):
        assert policy_modes.mode_for(rel).mode == mode

    def test_the_reason_is_the_authors_own_sentence(self, app_workspace):
        verdict = policy_modes.mode_for(".env")
        assert verdict.reason
        message = policy_modes.refusal_message(verdict, for_write=True)
        assert "benchmarkit's own rules for this workspace" in message
        assert verdict.reason in message

    @pytest.mark.parametrize("rel", ["notes.env", "src/environment.py", "a/env/x"])
    def test_a_name_only_shaped_like_env_is_not_app_owned(self, app_workspace, rel: str):
        """``.env`` is a name, not a substring — and nothing here hides ``logs``
        in someone else's tree, so an unrelated ``.env``-ish file is left alone."""
        assert policy_modes.mode_for(rel) == shared.Verdict(
            shared.READ_WRITE, None, None, None
        )

    @pytest.mark.parametrize("rel", [".environment", "src/.envsh", "a/.envrc/x"])
    def test_a_name_starting_with_the_letters_env_is_taken_as_one_of_them(
        self, app_workspace, rel: str
    ):
        """The prefix shape, kept on purpose: today's floor is
        ``name.startswith(".env")`` (``path_policy.py:73``), so ``.envrc`` is
        already refused. ``.env**`` says the same thing in the pattern language —
        the ``**`` is the subtree rule, not a separator."""
        assert policy_modes.mode_for(rel).mode == shared.HIDDEN

    def test_an_extension_sounding_env_file_is_not_app_owned(self, app_workspace):
        """What the prefix does *not* reach: a name that only ends in ``.env``."""
        assert policy_modes.mode_for("notes.env").mode == shared.READ_WRITE

    def test_the_rules_apply_to_the_folder_they_name(self, app_workspace):
        """``logs`` itself, not only what is under it — the app owns the folder."""
        assert policy_modes.mode_for("logs").mode == shared.HIDDEN
        assert policy_modes.mode_for("web/dist").mode == shared.READ_ONLY

    def test_there_is_no_git_rule_at_this_level(self, app_workspace):
        """D6: ``.git`` is a workspace-policy concern, present in every seeded
        file — not an app rule. Level A is about files *this app writes*."""
        assert not any(".git" in rule[0] for rule in builtin_policy.BUILTIN_RULES)
        assert builtin_policy.builtin_rules(".git/config") == []

    def test_a_git_rule_arrives_from_the_seeded_file_not_from_here(
        self, app_workspace
    ):
        """The other half of D6, and what pol-05 made observable.

        ``.git`` is hidden in this workspace — but by level B, the seeded file,
        not by code. A test that only asserted "level A says nothing" would pass
        even if the workspace had no policy at all, so the winning level is
        named: if a ``.git`` rule ever moves into ``BUILTIN_RULES`` (making it
        unconfigurable, which D6 forbids), the level changes and this fails.
        """
        assert policy_seed.ensure_policy_file(APP_ROOT)["state"] in (
            "created",
            "existing",
        )
        policy_file.invalidate_policy_cache()
        verdict = policy_modes.mode_for(".git/config")
        assert verdict.mode == shared.HIDDEN
        assert verdict.level == shared.LEVEL_WORKSPACE
        assert verdict.pattern == ".git"

    def test_level_a_is_narrow(self, app_workspace):
        assert policy_modes.mode_for("src/config.py").mode == shared.READ_WRITE

    def test_the_old_helpers_path_is_ordinary_here_too(self, app_workspace):
        """The behaviour change this task makes (D7), on benchmarkit's own tree.

        It used to be ``ro``: browsable, and refused writes. Now no level-A rule
        mentions it at all, because what protects the helpers is the **reserved
        folder** the file moved into, not a pattern for one name. A leftover copy
        of the old path is therefore an ordinary file (the app never reads it
        again, and removing it is the user's own business).
        """
        old = "resources/memory_registration.json"
        verdict = policy_modes.mode_for(old)
        assert verdict.mode == shared.READ_WRITE
        assert verdict.level is None
        assert builtin_policy.builtin_rules(old) == []


class TestForeignWorkspace:
    """The point of the task: another project's files are its own."""

    @pytest.mark.parametrize("rel", [".env", ".env.test", "src/.env.local"])
    def test_a_foreign_env_is_an_ordinary_file(self, foreign_workspace, rel: str):
        verdict = policy_modes.mode_for(rel)
        assert verdict == shared.Verdict(shared.READ_WRITE, None, None, None)
        assert builtin_policy.builtin_rules(rel) == []

    @pytest.mark.parametrize("rel", ["logs/x", "web/dist/app.js"])
    def test_a_foreign_folder_of_the_same_name_is_ordinary(
        self, foreign_workspace, rel: str
    ):
        assert policy_modes.mode_for(rel).mode == shared.READ_WRITE

    def test_the_app_owned_names_are_ordinary_too(self, foreign_workspace):
        """``resources/memory_registration.json`` is the *old* app file: here it
        is a stale leftover of somebody else's tree, so nothing about it is a
        built-in verdict. The prompt helpers moved into the reserved folder,
        which level A answers in **every** workspace (D7)."""
        for rel in ("resources/memory_registration.json", "web/dist"):
            assert policy_modes.mode_for(rel).level != shared.LEVEL_BUILTIN
        assert (
            policy_modes.mode_for("resources/memory_registration.json").mode
            == shared.READ_WRITE
        )

    def test_the_prompt_helpers_file_is_hidden_in_a_foreign_workspace(
        self, foreign_workspace
    ):
        """D7: the replacement for the deleted level-A rule is the reservation,
        which is unconditional — so the file is invisible in a foreign project's
        tree exactly as it is in benchmarkit's own.

        Built from the module's own constants, not a literal: a rename of the
        folder or the file must move this assertion with it rather than leave it
        testing a path that no longer exists.
        """
        rel = (
            f"{prompt_helpers.POLICY_DIRNAME}/"
            f"{prompt_helpers.PROMPT_HELPERS_FILENAME}"
        )
        verdict = policy_modes.mode_for(rel)
        assert verdict.mode == shared.HIDDEN
        assert verdict.level == shared.LEVEL_BUILTIN
        assert verdict.reason == builtin_policy.reserved_reason()

    def test_a_copy_of_the_repo_is_not_the_app(self, tmp_path: Path, monkeypatch):
        """The same file names, in a checkout this process did not boot from:
        dormant. The failing test if level A ever moves back to unconditional."""
        copy = copy_of_the_repo(tmp_path / "benchmarkit-copy")
        monkeypatch.setenv("BENCHMARKIT_PATH_LOCATION", str(copy))
        assert builtin_policy.is_self_workspace() is False
        for rel in (".env", "logs/trash/x", "web/dist/index.html"):
            assert policy_modes.mode_for(rel).mode == shared.READ_WRITE


class TestReservedFolderIsUnconditional:
    """The one level-A rule that does not care which workspace this is (D5)."""

    PATHS = [
        ".nojae",
        ".nojae/read-write-policy.json",
        "a/.nojae/b",
        "src/deep/.nojae/x/y",
    ]

    @pytest.mark.parametrize("workspace", ["app_workspace", "foreign_workspace"])
    @pytest.mark.parametrize("rel", PATHS)
    def test_hidden_in_every_workspace(self, workspace: str, rel: str, request):
        request.getfixturevalue(workspace)
        verdict = policy_modes.mode_for(rel)
        assert verdict.mode == shared.HIDDEN
        assert verdict.level == shared.LEVEL_BUILTIN
        assert verdict.pattern == editorfs.constants.POLICY_DIRNAME

    @pytest.mark.parametrize("rel", [".nojaeer", "nojae/x", "x.nojae/y"])
    def test_a_name_only_similar_to_it_is_untouched(self, rel: str, foreign_workspace):
        assert policy_modes.mode_for(rel).mode == shared.READ_WRITE

    def test_it_holds_without_a_policy_file_on_disk(self, foreign_workspace: Path):
        """The answer is about the name, so an absent folder is still hidden —
        and ``mode_for`` never asks whether the file loaded."""
        assert not (foreign_workspace / ".nojae").exists()
        assert policy_modes.mode_for(".nojae/x").mode == shared.HIDDEN

    def test_the_reserved_name_is_patchable_through_constants(
        self, foreign_workspace, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(editorfs.constants, "POLICY_DIRNAME", ".policydir")
        assert policy_modes.mode_for(".policydir/x").mode == shared.HIDDEN
        assert policy_modes.mode_for(".nojae/x").mode == shared.READ_WRITE

    def test_a_looser_workspace_rule_cannot_lift_it(
        self, foreign_workspace, monkeypatch: pytest.MonkeyPatch
    ):
        """D5 through the merge: a ``rw`` rule for the reserved folder loses to
        the unconditional ``hidden``, whatever its level or order."""
        monkeypatch.setattr(
            policy_modes,
            "list_workspace_rules",
            lambda rel, *, fold=False: [
                shared.Rule(
                    mode=shared.READ_WRITE,
                    pattern=".nojae",
                    reason=None,
                    index=99,
                    level=shared.LEVEL_WORKSPACE,
                )
            ],
        )
        verdict = policy_modes.mode_for(".nojae/read-write-policy.json")
        assert verdict.mode == shared.HIDDEN
        assert verdict.level == shared.LEVEL_BUILTIN
        assert verdict.pattern == ".nojae"


class TestRulesAreWrittenInThePatternLanguage:
    """The list itself: order, spelling and the shape of each entry."""

    def test_it_is_a_tuple_of_pattern_mode_reason_triples(self):
        assert isinstance(builtin_policy.BUILTIN_RULES, tuple)
        for rule in builtin_policy.BUILTIN_RULES:
            assert isinstance(rule, tuple) and len(rule) == 3
            pattern, mode, reason = rule
            assert isinstance(pattern, str) and pattern
            assert mode in shared.MODES
            assert isinstance(reason, str) and reason.strip()

    def test_the_index_that_merges_them_is_their_position(self, app_workspace):
        # ``app_workspace`` because level A exists only while the working
        # location *is* the install: with the location elsewhere the list is
        # empty by design, and the index below would be a fact about nothing.
        assert [rule.index for rule in builtin_policy.builtin_rules(".env")] == [0]
        matched = [
            rule.pattern for rule in builtin_policy.builtin_rules(".env")
        ]
        assert matched == [builtin_policy.BUILTIN_RULES[0][0]]

    def test_no_rule_is_a_pattern_the_matcher_rejects(self):
        from mcp_connectors import policy_glob

        for pattern, _, _ in builtin_policy.BUILTIN_RULES:
            assert policy_glob.pattern_error(pattern) is None, pattern

    def test_no_rule_names_the_reserved_folder(self):
        reserved = editorfs.constants.POLICY_DIRNAME
        for pattern, _, _ in builtin_policy.BUILTIN_RULES:
            assert reserved not in pattern.split("/")


class TestTheVerdictIsWhatTheRoutesNowAnswerWith:
    """pol-06 landed: the predicates this class used to compare against are gone.

    What remains worth pinning is the same split, stated against the **routes**
    rather than against two booleans that no longer exist: on the app's own repo
    the pane answers with the verdict, and the paths that used to be refused by
    the connector floor instead of by level A are the interesting ones — some of
    them opened on purpose (this task's visible change), and the ones that stayed
    closed did so because a rule says so, not because code does.
    """

    # Ordinary files of this checkout: unchanged, and listed, and writable.
    ORDINARY: List[str] = [
        "src/config.py",
        "src/core/editorfs/policy.py",
        "tests/test_api.py",
        "resources/122qcm.json",
    ]

    @pytest.mark.parametrize("rel", ORDINARY)
    def test_an_ordinary_path_of_the_app_is_still_an_ordinary_file(
        self, app_workspace, rel: str
    ):
        """The refactor's cost, checked: nothing that was editable got locked."""
        assert policy_modes.mode_for(rel).mode == shared.READ_WRITE
        with TestClient(create_app()) as client:
            body = client.get("/api/fs/file", params={"path": rel})
            assert body.status_code == 200, rel
            listed = client.get("/api/fs/tree").json()
            assert rel.split("/")[0] in [
                entry["name"] for entry in listed["entries"]
            ], rel

    # Level A's rows, now live at the routes: hidden answers a look with absence
    # (404) and a touch with the rule (422).
    @pytest.mark.parametrize(
        "rel,pattern",
        [
            (".env", "**/.env**"),
            (".env.test", "**/.env**"),
            (".env.example", "**/.env**"),
            ("logs/trash/x", "logs/**"),
            ("logs/editor_backups/y", "logs/**"),
        ],
    )
    def test_an_app_owned_path_is_absent_to_a_read_and_named_to_a_write(
        self, app_workspace, rel: str, pattern: str
    ):
        """The D8 asymmetry, on benchmarkit's own workspace, at the routes."""
        with TestClient(create_app()) as client:
            read = client.get("/api/fs/file", params={"path": rel})
            assert read.status_code == 404, rel
            assert pattern not in read.text
            if "/" not in rel:
                assert rel not in _root_names(client), rel

            write = client.put(
                "/api/fs/file", json={"path": rel, "text": "overwritten\n"}
            )
            assert write.status_code == 422, rel
            detail = write.json()["detail"]
            assert f'"{pattern}"' in detail, rel
            assert "benchmarkit's own rules" in detail, rel

    def test_the_prompt_helpers_file_is_absent_to_a_read_and_named_to_a_write(
        self, app_workspace
    ):
        """D7 at the routes: the helpers file has **no level-A pattern** — the
        reserved folder is what hides it, so the verdict comes from the folder.

        This is deliberately not another row of the list above: that list asserts
        the matched *pattern* is quoted in the 422 and never appears in the 404,
        and neither holds for a reserved path. The 422 quotes the folder name
        (``.nojae``), and the 404 is the ordinary absence sentence, which echoes
        the requested path — so a hidden name inside a reserved folder is
        indistinguishable from absence all the same, exactly as
        ``tests/test_editorfs_write.py`` pins for a batch under ``.nojae/trash``.
        """
        rel = (
            f"{prompt_helpers.POLICY_DIRNAME}/"
            f"{prompt_helpers.PROMPT_HELPERS_FILENAME}"
        )
        reserved = f'"{prompt_helpers.POLICY_DIRNAME}"'
        with TestClient(create_app()) as client:
            read = client.get("/api/fs/file", params={"path": rel})
            assert read.status_code == 404
            assert "does not exist" in read.json()["detail"]
            assert prompt_helpers.POLICY_DIRNAME not in _root_names(client)

            write = client.put(
                "/api/fs/file", json={"path": rel, "text": "hand-edited\n"}
            )
            assert write.status_code == 422
            detail = write.json()["detail"]
            assert reserved in detail
            assert "benchmarkit's own rules" in detail
            assert "reserved by benchmarkit" in detail

    @pytest.mark.parametrize("rel", ["web/dist/index.html"])
    def test_an_app_owned_build_or_data_file_is_readable_and_locked_from_writes(
        self, app_workspace, rel: str
    ):
        """``ro`` at the routes: the tree and the read keep working, the save does not."""
        verdict = policy_modes.mode_for(rel)
        assert verdict.level == shared.LEVEL_BUILTIN
        assert verdict.mode == shared.READ_ONLY
        with TestClient(create_app()) as client:
            assert client.get("/api/fs/file", params={"path": rel}).status_code == 200
            write = client.put(
                "/api/fs/file", json={"path": rel, "text": "hand-edited\n"}
            )
            assert write.status_code == 422
            assert "benchmarkit's own rules" in write.json()["detail"]

    @pytest.mark.parametrize(
        "rel", [".git/config", ".ssh/id_rsa", "web/src/id_rsa", "secrets/aws.pem"]
    )
    def test_what_the_connector_floor_used_to_close_is_now_a_workspaces_business(
        self, app_workspace, rel: str
    ):
        """The listed deltas, restated now that the swap has happened.

        These four names were refused by the deny-list and are *not* level-A
        rules: ``.git`` is D6's workspace concern, arriving from the seeded file,
        and the credential names belong to the connector floor, which guards its
        own surface and is not what level A is for. So the pane's answer comes
        from a rule a user can read and change — and pol-08 is what layers the
        connector floor back on top for the agents.
        """
        verdict = policy_modes.mode_for(rel)
        assert verdict.level != shared.LEVEL_BUILTIN, rel
        assert verdict.level == shared.LEVEL_WORKSPACE or verdict.mode == shared.READ_WRITE

    def test_the_apps_own_env_is_no_longer_refused_by_a_code_list(self, app_workspace):
        """The inversion this task exists to make visible.

        Before pol-06, the app's ``.env`` was closed by a hardcoded ``.env``
        prefix check that closed every other project's ``.env`` with it. Now the
        refusal is level A's rule — conditional on this workspace *being* the app,
        quoted by name when it fires, and loosened to a normal file the moment
        the location points elsewhere.

        ``app_workspace`` states that condition instead of inheriting it: the
        suite's autouse guard gives every test a throw-away working location (so
        no test can read the developer's ``.nojae/nojae-settings.json``), and this
        assertion is about the workspace *being* the install — the fact the file's
        own fixture exists to declare.
        """
        with TestClient(create_app()) as client:
            read = client.get("/api/fs/file", params={"path": ".env"})
        assert read.status_code == 404
        assert "connector path policy" not in read.text
        verdict = policy_modes.mode_for(".env")
        assert verdict.level == shared.LEVEL_BUILTIN
        assert policy_modes.refusal_message(verdict, for_write=True)
