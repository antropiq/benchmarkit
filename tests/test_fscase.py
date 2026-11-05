"""Tests for win-05: the per-workspace case-folding probe.

Two separate promises, and they need different handling:

* **the probe itself** answers a question about a real filesystem, so the tests
  run against real ``tmp_path`` folders — and the honest expectation is
  *whatever this runner's volume answers*: ext4 answers ``False``, NTFS answers
  ``True``, and no test may assume either (gb-01b). What the tests pin is the
  *agreement* between the probe and a direct observation of the volume, plus the
  probe's decision logic (identity comparison), reached by patching ``os.stat``.
  Both outcomes stay exercised on one runner: the folded branch is reached by
  writing the cached answer directly or by patching the probe (roadmap §9 rule
  2), never by pretending to be another OS.
* **the cache** is what the app reads, so its key, its bound and its
  invalidation are all asserted as behaviour — a stale answer across a working
  location switch is the same class of bug as a stale rule set, and
  ``tests/test_editorfs_policy_file.py`` pins that one for the policy file.
  These tests never assert *which* answer the volume gives; they assert that an
  answer is remembered per root, re-probed after invalidation, and bounded.

Nothing here pretends to be another platform: no ``sys.platform`` is patched and
no filesystem shape is faked beyond the cached answer.

Since gb-01 the probe *implementation* lives in ``mcp_connectors.fscase`` (the
console connector asks the same question for the root it walks, and a connector
process cannot import ``src``); ``editorfs.fscase`` keeps the app's import path.
Tests that poke the probe's internals (``_probe``, its ``os``) patch the shared
module, because that is where ``case_folded`` looks them up.
"""

import json
import os
from pathlib import Path
from typing import Dict, Iterator, List

import pytest

from mcp_connectors import fscase as probe_shared
from src.core import editorfs
from src.core.editorfs import fscase, policy_modes


@pytest.fixture(autouse=True)
def clean_case_cache() -> Iterator[None]:
    """Each test starts and ends with no remembered answer."""
    fscase.invalidate_case_cache()
    yield
    fscase.invalidate_case_cache()


@pytest.fixture
def location(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """``tmp_path`` as the active working location, with a file in it."""
    monkeypatch.setenv("BENCHMARKIT_PATH_LOCATION", str(tmp_path))
    return tmp_path


class TestTheProbeAnswers:
    def test_the_probe_agrees_with_the_volume_it_walks(self, tmp_path: Path):
        """The probe's answer *is* the volume's answer — on either kind of volume.

        ext4 says ``Fold`` and ``fold`` are two names, NTFS says one; the test
        pins the agreement, not a runner's shape (gb-01b): plant a cased name,
        observe directly whether the folded spelling names the same file, and
        the probe must answer the same thing.
        """
        planted = tmp_path / "ProbeName"
        planted.write_text("x", encoding="utf-8")
        try:
            cased = os.stat(planted)
            try:
                other = os.stat(tmp_path / "probename")
                observed = (cased.st_dev, cased.st_ino) == (other.st_dev, other.st_ino)
            except OSError:
                observed = False
            assert fscase.case_folded(str(tmp_path)) is observed
        finally:
            planted.unlink(missing_ok=True)

    def test_it_leaves_nothing_behind_in_the_workspace(self, tmp_path: Path):
        """Two names are created to ask the question and both are removed."""
        (tmp_path / "keep.txt").write_text("x\n", encoding="utf-8")
        fscase.case_folded(str(tmp_path))
        assert sorted(os.listdir(tmp_path)) == ["keep.txt"]

    def test_a_root_that_cannot_be_written_to_answers_false(self, tmp_path: Path):
        """*Fold nothing you cannot prove* — and a wrong ``False`` is over-strict.

        A non-existent root is the platform-free shape of the same answer; the
        mode-locked folder below is the read-only-mount shape, which needs POSIX
        bits to plant.
        """
        assert fscase.case_folded(str(tmp_path / "not-there")) is False

    @pytest.mark.posix_behaviour(
        "the unwritable root is planted with chmod(0o500): without POSIX mode "
        "bits there is no read-only mount to ask, and the missing-root case "
        "above already pins the same answer"
    )
    def test_an_unwritable_root_answers_false_and_raises_nothing(
        self, tmp_path: Path
    ):
        if os.geteuid() == 0:
            pytest.skip("root ignores the mode bits")
        locked = tmp_path / "locked"
        locked.mkdir(mode=0o500)
        try:
            assert fscase.case_folded(str(locked)) is False
        finally:
            locked.chmod(0o755)

    def test_one_name_under_both_spellings_is_read_as_folded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """The probe's *decision*, not this runner's volume.

        A case-insensitive mount cannot be manufactured here, so ``os.stat`` is
        patched to answer the folded spelling with the cased file's own identity —
        exactly what NTFS does. What is under test is the comparison: it must ask
        for the **folded** name (an FS that preserves the case it was given still
        names one file) and answer on ``st_dev``/``st_ino``, not on metadata that
        merely agrees.
        """
        real_stat = os.stat

        def folded_stat(path: object) -> os.stat_result:
            name = os.path.basename(str(path))
            if name.endswith("aBc"):
                return real_stat(str(path)[: -len("aBc")] + "Abc")
            return real_stat(path)

        monkeypatch.setattr(probe_shared.os, "stat", folded_stat)
        assert fscase.case_folded(str(tmp_path)) is True
        assert sorted(os.listdir(tmp_path)) == []

    def test_a_folded_name_that_is_a_different_file_is_not_a_fold(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """The identity half of the comparison, pinned the other way.

        A folded spelling whose ``stat`` answers some *other* inode is a second
        file, and answering ``True`` there would merge two real folders. The
        stand-in answers the lookup with a file the test planted, so the two
        sides differ only in identity — which is precisely the comparison.
        """
        other = tmp_path / "a-completely-different-file"
        other.write_text("x\n", encoding="utf-8")
        real_stat = os.stat

        def other_inode(path: object) -> os.stat_result:
            if str(path).endswith("aBc"):
                return real_stat(other)
            return real_stat(path)

        monkeypatch.setattr(probe_shared.os, "stat", other_inode)
        assert fscase.case_folded(str(tmp_path)) is False
        assert sorted(os.listdir(tmp_path)) == [other.name]

    def test_the_folded_answer_is_what_the_matcher_needs(self, tmp_path: Path):
        """The other branch, reached through the cache rather than a fake mount.

        This is the value the matcher is handed on a case-insensitive volume; the
        point of asserting it here is that the *consumer* half of win-05 has a
        reachable ``True`` on the Linux runner (roadmap §9 rule 2).
        """
        fscase._CACHE[str(tmp_path)] = True
        assert fscase.case_folded(str(tmp_path)) is True

    def test_the_app_name_and_the_connector_name_are_one_probe(self):
        """gb-01: the console connector asks the *same* question of its root.

        ``editorfs.fscase`` is the app-side name for ``mcp_connectors.fscase``;
        one function and one cache, because two caches for one volume is how the
        pane and the agent start disagreeing about a spelling.
        """
        from mcp_connectors import fscase as connector_fscase

        assert fscase.case_folded is connector_fscase.case_folded
        assert fscase._CACHE is connector_fscase._CACHE


class TestTheCache:
    def test_one_probe_per_root_however_many_verdicts_ask(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """The disk is touched once; every later verdict reads the remembered answer."""
        probes: List[str] = []
        real_probe = probe_shared._probe

        def counting(root: str) -> bool:
            probes.append(root)
            return real_probe(root)

        monkeypatch.setattr(probe_shared, "_probe", counting)
        for _ in range(5):
            fscase.case_folded(str(tmp_path))
        assert probes == [str(tmp_path)]

    def test_two_roots_are_answered_separately(self, tmp_path: Path,
                                               monkeypatch: pytest.MonkeyPatch):
        """The reason the key is the root: one workspace's volume is not another's.

        The answers are forced through the probe seam rather than read off this
        runner's volume (gb-01b): what is pinned is that one root's remembered
        answer never answers another root, whatever the two volumes really are.
        """
        first = tmp_path / "first"
        second = tmp_path / "second"
        first.mkdir()
        second.mkdir()
        answers = {str(first): True, str(second): False}
        monkeypatch.setattr(probe_shared, "_probe", lambda root: answers[root])
        assert fscase.case_folded(str(first)) is True
        assert fscase.case_folded(str(second)) is False

    def test_invalidation_forces_a_fresh_probe(self, tmp_path: Path,
                                               monkeypatch: pytest.MonkeyPatch):
        probes: List[str] = []
        real_probe = probe_shared._probe

        def counting(root: str) -> bool:
            probes.append(root)
            return real_probe(root)

        monkeypatch.setattr(probe_shared, "_probe", counting)
        fscase.case_folded(str(tmp_path))
        fscase.invalidate_case_cache()
        fscase.case_folded(str(tmp_path))
        assert len(probes) == 2

    def test_the_cache_is_bounded(self, tmp_path: Path):
        """Past the bound the cache is dropped rather than growing forever.

        Same shape as ``policy_file``'s: the bound is a guard against a long
        uptime, not a working set, so exceeding it costs a re-probe and never an
        answer.
        """
        roots = [f"/root-{index}" for index in range(fscase._CACHE_MAX_ROOTS + 1)]
        for root in roots:
            fscase.case_folded(root)
        assert len(fscase._CACHE) <= fscase._CACHE_MAX_ROOTS


class TestTheSwitchForgetsIt:
    """AGENTS.md: *"a switch invalidates them all at once"* (win-05 §1)."""

    def test_the_seeder_invalidates_it_alongside_the_policy_cache(
        self, tmp_path: Path
    ):
        from src.core.editorfs import policy_seed

        target = tmp_path / "new-project"
        target.mkdir()
        fscase._CACHE[str(target)] = True
        assert policy_seed.ensure_policy_file(str(target))["state"] == "created"
        assert fscase._CACHE == {}

    def test_the_settings_route_invalidates_it(self, location: Path,
                                               monkeypatch: pytest.MonkeyPatch):
        """``POST /api/path-location`` — the moment the tree re-roots.

        The recording hubs are the pattern ``tests/test_editorfs_policy_seed.py``
        uses, so the switch's restarts stay hers and her declaration files her
        own.
        """
        from fastapi.testclient import TestClient

        from src.core import lsphub, mcphub
        from src.api.server import create_app

        class _RecordingHub:
            installed = True

            def __init__(self) -> None:
                self.restarts: List[object] = []

            async def restart(self, *args: object) -> None:
                self.restarts.append(args)

            def status(self) -> List[Dict[str, object]]:
                return []

        monkeypatch.setattr(mcphub.constants, "MCP_SETTINGS_PATH",
                            location / "mcp-settings.json")
        servers_dir = location / "lsp_connectors"
        servers_dir.mkdir()
        (servers_dir / "restricted-paths.txt").write_text("/etc\n", encoding="utf-8")
        monkeypatch.setattr(lsphub.constants, "LSP_CONNECTORS_DIR", servers_dir)
        monkeypatch.setattr(
            lsphub.constants, "LSP_SETTINGS_PATH", servers_dir / "lsp-settings.json"
        )
        monkeypatch.setattr(
            lsphub.constants, "LSP_RESTRICTED_PATHS",
            servers_dir / "restricted-paths.txt",
        )

        target = Path(os.environ["BENCHMARKIT_PATH_LOCATION"]) / "switched-to"
        target.mkdir()
        app = create_app()
        with TestClient(app) as test_client:
            test_client.app.state.mcp = _RecordingHub()
            test_client.app.state.lsp = _RecordingHub()
            # A remembered answer for a root the process has already probed,
            # plus a planted answer for the one it is about to open: the switch
            # has to forget *both*, so the new workspace is judged by a probe of
            # its own volume.
            fscase._CACHE[str(target)] = True
            response = test_client.post(
                "/api/path-location", json={"path": str(target)}
            )
            assert response.status_code == 200
            assert fscase._CACHE == {}
            # The planted answer is gone and the new root is *re-probed*, not
            # answered from memory (gb-01b: what is pinned is the fresh probe,
            # not which answer this runner's volume gives).
            probe_calls: List[str] = []
            real_probe = probe_shared._probe

            def counting(root: str) -> bool:
                probe_calls.append(root)
                return real_probe(root)

            monkeypatch.setattr(probe_shared, "_probe", counting)
            fscase.case_folded(os.environ["BENCHMARKIT_PATH_LOCATION"])
            assert probe_calls == [os.environ["BENCHMARKIT_PATH_LOCATION"]]

            # A second switch to a root that is *already seeded*: the seeder
            # writes nothing there and so never reaches its own invalidation,
            # which is why the route says it explicitly (routes.py). Planting a
            # wrong answer first is what makes that visible.
            fscase._CACHE[str(target)] = True
            again = test_client.post("/api/path-location", json={"path": str(target)})
            assert again.status_code == 200
            assert again.json()["policy"]["state"] == "existing"
            assert fscase._CACHE == {}


class TestTheVerdictUsesTheProbe:
    """The fold reaches the matcher only through ``mode_for`` (pol-06's one door)."""

    def test_mode_for_hands_the_verdict_the_roots_answer(self, location: Path):
        """Both answers reach the matcher — forced through the cache seam.

        Which answer a real volume gives is the runner's business (gb-01b); what
        ``mode_for`` promises is that *the remembered answer* decides the
        spelling question, in both directions, on one runner (roadmap §9 rule 2).
        """
        (location / "dist").mkdir()
        (location / "dist" / "app.js").write_text("x\n", encoding="utf-8")
        _write_policy(location, [{"path": "dist/**", "mode": "hidden"}])

        fscase._CACHE[str(editorfs.workspace_root())] = False
        assert policy_modes.mode_for("Dist/app.js").mode == "rw"
        fscase._CACHE[str(editorfs.workspace_root())] = True
        assert policy_modes.mode_for("Dist/app.js").mode == "hidden"

    def test_a_folded_match_still_reports_the_pattern_that_was_written(
        self, location: Path
    ):
        """A-D3: folding is comparison only, never the displayed form.

        The rule the user typed is ``dist/**``; the path that reached it is
        spelled ``Dist/app.js``. Neither spelling may be rewritten on the way
        out, or the panel and the 422 describe a rule that is not in the file.
        """
        (location / "dist").mkdir()
        _write_policy(location, [{"path": "dist/**", "mode": "hidden",
                                  "reason": "build output"}])
        fscase._CACHE[str(editorfs.workspace_root())] = True
        verdict = policy_modes.mode_for("Dist/app.js")
        assert verdict.mode == "hidden"
        assert verdict.pattern == "dist/**"
        assert "dist/**" in policy_modes.refusal_message(verdict, for_write=False)
        assert "Dist" not in policy_modes.refusal_message(verdict, for_write=False)

    def test_the_exact_spelling_matches_whatever_the_volume_answers(
        self, location: Path
    ):
        """The rule's own spelling always matches; the *other* one follows the probe.

        The Linux default was a probe answer, not a promise (gb-01b): on a
        folding volume the case-differing spelling is the same folder and the
        rule reaches it folded; on an exact one it is a different name and the
        rule says nothing about it. The exact spelling matches either way.
        """
        (location / "dist").mkdir()
        _write_policy(location, [{"path": "dist/**", "mode": "hidden"}])
        assert policy_modes.mode_for("dist/app.js").mode == "hidden"
        folded = fscase.case_folded(editorfs.workspace_root())
        expected = "hidden" if folded else "rw"
        assert policy_modes.mode_for("Dist/app.js").mode == expected


def _write_policy(root: Path, rules: List[Dict[str, str]]) -> None:
    from src.core.editorfs import policy_file

    folder = root / editorfs.constants.POLICY_DIRNAME
    folder.mkdir(exist_ok=True)
    (folder / editorfs.constants.POLICY_FILENAME).write_text(
        json.dumps(
            {"version": editorfs.constants.POLICY_VERSION, "policies": rules},
            indent=2,
        ),
        encoding="utf-8",
    )
    policy_file.invalidate_policy_cache()
