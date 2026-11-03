"""Unit tests for the shared connector deny-list (``mcp_connectors/path_policy``).

Two of win-07's four behaviour tests live at this layer, because both are
questions about *text*: does a prefix list speak this path's form, and does a
component rule still hit when the volume folds the spelling. Neither needs a
connector, a workspace or a platform -- the Windows shapes are fed as input and
the fold is passed as an argument, so both outcomes are assertable on the Linux
runner (roadmap §9 rule 2).
"""

from pathlib import Path
from typing import Tuple

import pytest

from mcp_connectors import path_policy


class TestComponentHitFold:
    """§6 behaviour test 7, second half: ``.ssh`` in a ``.SSH`` folder."""

    WINDOWS_PARTS: Tuple[str, ...] = ("C:", "Users", "VGR", ".SSH", "id_rsa")

    def test_it_hits_a_differently_spelled_component_when_folding(self):
        assert (
            path_policy.component_hit(self.WINDOWS_PARTS, (".ssh",), fold=True)
            == ".ssh"
        )

    def test_it_misses_a_differently_spelled_component_when_not_folding(self):
        # The Linux answer, unchanged: on an exact filesystem ``.SSH`` is some
        # other folder and ``.ssh`` protects nothing in it.
        assert path_policy.component_hit(self.WINDOWS_PARTS, (".ssh",)) is None
        assert (
            path_policy.component_hit(self.WINDOWS_PARTS, (".ssh",), fold=False)
            is None
        )

    def test_a_multi_component_entry_folds_every_part(self):
        parts = ("D:", "Users", "VGR", ".CONFIG", "GH", "hosts.yml")
        assert (
            path_policy.component_hit(parts, (".config/gh",), fold=True)
            == ".config/gh"
        )
        assert path_policy.component_hit(parts, (".config/gh",)) is None

    def test_an_exact_spelling_hits_either_way(self):
        # The pair above must not be read as "folding breaks matching": the
        # folded rule is a superset, and the spelling that already matched is
        # still refused on a folded volume.
        parts = ("home", "vgr", ".ssh", "id_ed25519")
        assert path_policy.component_hit(parts, (".ssh",)) == ".ssh"
        assert (
            path_policy.component_hit(parts, (".ssh",), fold=True) == ".ssh"
        )

    def test_deny_reason_stays_exact_for_its_own_callers(self):
        """The connectors keep ``fold=False`` until target B passes real values.

        ``deny_reason`` is what ``fedit``/``console`` actually call, so this
        pins that win-07 did not quietly widen the connectors' own verdict: on
        this runner a differently spelled secret store is still not refused by
        it, which is what the ``[fold]`` parameter is reserved for.
        """
        assert path_policy.deny_reason(
            Path("/home/vgr/.SSH/known"), for_write=False
        ) is None
        assert path_policy.deny_reason(
            Path("/home/vgr/.ssh/known"), for_write=False
        ) is not None


class TestListsAreMeaningful:
    """A-D4's zero-match check: does any prefix speak this path's form?

    The zero-match tests name the **POSIX half** of the lists explicitly: that
    is the pair A-D4 reported (POSIX entries against a drive path), and it
    stays true whatever the runner is. B1's answer — the Windows half beside it
    — is asserted in ``TestWindowsValuesArriveFromTheEnvironment``, where the
    module's own default is what gets asked.
    """

    POSIX_ONLY: Tuple[str, ...] = ("/etc", "/usr", "/bin", "/tmp", "/var/tmp")

    def test_posix_prefixes_speak_nothing_about_a_windows_path(self):
        assert not path_policy.lists_are_meaningful(
            r"C:\Users\vgr\.ssh\id_ed25519", prefixes=self.POSIX_ONLY
        )

    def test_a_denominator_the_lists_do_not_name_is_still_silent(self):
        # The pair that matters most is not a path *inside* a denied prefix but
        # a path under a root the list has no word for: nothing about
        # ``C:/Users/vgr`` is ``/etc``, and no refusal is raised to say so.
        assert not path_policy.lists_are_meaningful(
            "C:/Users/vgr/dev/app.py", prefixes=self.POSIX_ONLY
        )
        # And the same list does speak the root it was written for.
        assert path_policy.lists_are_meaningful("/etc/ssl/private/x.pem")

    def test_posix_prefixes_speak_a_posix_path(self):
        assert path_policy.lists_are_meaningful("/home/vgr/.ssh/id_ed25519")

    def test_the_form_is_compared_not_the_path(self):
        # Any drive answers False against the POSIX-only list -- the check is
        # about the two sides being spellable at all, not about this path being
        # safe. ...and a UNC root is a third form, spoken by no list either.
        assert not path_policy.lists_are_meaningful(
            "D:/src/app.py", prefixes=self.POSIX_ONLY
        )
        assert not path_policy.lists_are_meaningful("//fileserver/share/x")

    def test_an_explicit_list_is_what_gets_asked(self):
        """The LSP hub hands this its own deny-list, not the connector prefixes."""
        windows = ["C:/Users/vgr/.ssh", "D:/secret"]
        assert path_policy.lists_are_meaningful("C:/WS/app.py", prefixes=windows)
        assert not path_policy.lists_are_meaningful("/home/vgr/ws", prefixes=windows)
        posix = ["/etc", "/home/vgr/.ssh"]
        assert path_policy.lists_are_meaningful("/home/vgr/ws", prefixes=posix)
        assert not path_policy.lists_are_meaningful("C:/WS/app.py", prefixes=posix)

    def test_a_windows_list_speaks_a_windows_path_on_this_runner(self):
        """Nothing here needs to *be* Windows: a matching list is a matching list.

        This is why the check can be written now and target B only has to supply
        values -- the answer comes from the shapes, never from ``sys.platform``.
        """
        assert path_policy.lists_are_meaningful(
            r"C:\Users\vgr\dev\app.py", prefixes=(r"C:\Users\vgr\.ssh",)
        )

    def test_an_unexpanded_reference_can_still_look_like_a_prefix(self):
        r"""Why the *shape* check cannot be the whole answer (B-D1).

        The classifier is text and a brace is an ordinary character in a name, so
        a reference that resolves to nothing survives with a rooted shape whenever
        it is not the leading component: ``/home/${NOPE}/.ssh`` is classified
        ``absolute`` and speaks a POSIX path. A hub that expands its list at read
        time therefore has to drop unresolved lines as well as check the shape, or
        a guardrail that protects nothing gets counted as a bound that was
        honoured (``lsphub.discovery.restricted_lsp_roots`` does both).
        """
        assert path_policy.root_form("/home/${NOT_SET_YET}/.ssh") == "absolute"
        assert path_policy.lists_are_meaningful(
            "/home/vgr/ws", prefixes=("/home/${NOT_SET_YET}/.ssh",)
        ) is True
        # A leading reference is the harmless half: relative text, spoken by no
        # rooted list — still not a guardrail, just a visibly wrong one.
        assert path_policy.root_form("${NOT_SET_YET}/.ssh") == "relative"

    def test_an_empty_list_speaks_nothing(self):
        # The caller decides what an empty list *means* (the hub treats an
        # absent deny-list as no bound either, so it is not a refusal).
        assert not path_policy.lists_are_meaningful("/home/vgr/ws", prefixes=[])

    def test_the_module_default_is_its_three_prefix_lists(self):
        defaults = (
            path_policy.ALLOW_PREFIXES
            + path_policy.DENY_PREFIXES
            + path_policy.DENY_PREFIXES_READ
        )
        # The POSIX forms are in every runner's lists; the Windows forms join
        # them only where the OS names them (B-D1), so the forms present are
        # the runner's own — never a second list in a second file.
        assert all(
            path_policy.root_form(entry) in ("absolute", "drive")
            for entry in defaults
        )
        assert any(
            path_policy.root_form(entry) == "absolute" for entry in defaults
        )
        assert path_policy.lists_are_meaningful("/tmp/scratch")

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("/etc/shadow", "absolute"),
            ("/", "absolute"),
            (r"\Users\vgr\x", "absolute"),
            ("src/config.py", "relative"),
            ("C:/Users/vgr", "drive"),
            (r"C:\Users\vgr", "drive"),
            ("c:/x", "drive"),
            ("C:", "drive"),
            ("//srv/share/x", "unc"),
            (r"\\srv\share\x", "unc"),
            # Not a drive: ``ntpath.splitdrive`` is looser than the Windows
            # parser it describes, and a POSIX file may be named ``foo:bar``.
            ("foo:bar/x", "relative"),
            ("/foo:bar/x", "absolute"),
            ("", "relative"),
        ],
    )
    def test_root_form_classifies_by_root_not_by_style(self, text, expected):
        assert path_policy.root_form(text) == expected


class TestPrefixHitComponentWise:
    """B-D1/G-D8: a prefix compare is root-wise and component-wise.

    The Windows forms are fed as text — nothing here needs to *be* Windows
    (roadmap §9 rule 2); what is pinned is that the compare honours roots and
    whole components, which is the class the old text-prefix test got wrong.
    """

    SYSTEM = ("C:/Windows", "C:/Program Files", "C:/ProgramData")

    def test_a_drive_prefix_speaks_a_drive_path(self):
        assert path_policy.prefix_hit("C:/Windows/System32", self.SYSTEM) == "C:/Windows"
        assert path_policy.prefix_hit("C:/Windows", self.SYSTEM) == "C:/Windows"

    def test_windows_old_is_not_under_windows(self):
        """The exact mistake a text-prefix test makes with raw Windows text."""
        assert path_policy.prefix_hit("C:/Windows.old/Foo", self.SYSTEM) is None

    def test_the_posix_lists_say_nothing_about_a_drive_tree(self):
        # The zero-match fact A-D4 reports; here it is the *matching* half of
        # the same sentence: a ``/etc`` prefix is silent about ``C:/…``.
        assert path_policy.prefix_hit("C:/Windows/hosts", path_policy.DENY_PREFIXES) is None
        assert path_policy.prefix_hit("/etc/hosts", path_policy.DENY_PREFIXES) == "/etc"

    def test_backslashes_and_slashes_are_one_path(self):
        # ``Path.as_posix()`` keeps drive letters and folds separators; raw
        # Windows text arrives with backslashes. Both spell one prefix compare.
        assert path_policy.prefix_hit(r"C:\Windows\System32", self.SYSTEM) == "C:/Windows"

    def test_the_fold_answers_both_spellings_on_a_folding_volume(self):
        assert path_policy.prefix_hit("c:/WINDOWS/system32", self.SYSTEM, fold=True) == "C:/Windows"
        assert path_policy.prefix_hit("C:/WINDOWS", self.SYSTEM, fold=True) == "C:/Windows"

    def test_without_the_fold_the_other_spelling_is_some_other_folder(self):
        # The Linux answer, unchanged: folding is the volume's answer, not a
        # default, and a differently spelled tree is a different tree.
        assert path_policy.prefix_hit("c:/WINDOWS/system32", self.SYSTEM) is None

    def test_a_posix_prefix_still_refuses_the_name_it_always_refused(self):
        # The old verdicts survive the rewrite: ``/etc`` covers ``/etc/hosts``
        # but never ``/etcx`` — component-wise matching is the same boundary
        # the text test drew, drawn from components instead.
        assert path_policy.prefix_hit("/etc/hosts", ("/etc",)) == "/etc"
        assert path_policy.prefix_hit("/etcx/hosts", ("/etc",)) is None

    def test_a_unc_prefix_speaks_only_its_own_share(self):
        unc = ("//fileserver/share",)
        assert path_policy.prefix_hit("//fileserver/share/x", unc) == "//fileserver/share"
        assert path_policy.prefix_hit("//fileserver/other/x", unc) is None

    def test_deny_reason_threads_the_fold_to_the_prefix_half(self, monkeypatch):
        """The fold reaches the prefix half, and the entry is quoted as stored.

        The write list in force is the POSIX defaults plus one Windows entry,
        stated here instead of grown from the environment: on a Windows box
        ``SystemRoot`` already names that tree in its own spelling, and the
        first match would then be quoted rather than the entry under test.
        """
        windows_entry = "C:/Windows"
        monkeypatch.setattr(
            path_policy,
            "DENY_PREFIXES",
            path_policy.DENY_POSIX_PREFIXES + (windows_entry,),
        )
        # The system root is write-denied but readable, folded or not.
        for read_fold in (True, False):
            assert (
                path_policy.prefix_deny_reason(
                    Path("C:/WINDOWS/System32/cmd.exe"),
                    for_write=False,
                    fold=read_fold,
                )
                is None
            )
        write_folded = path_policy.prefix_deny_reason(
            Path("c:/windows/x"), for_write=True, fold=True
        )
        # The sentence quotes the prefix **as the list spells it**, never the
        # spelling the path arrived with — the fold compares only (A-D3).
        assert write_folded == f"under the protected prefix {windows_entry!r}"
        # On an exact volume the differently spelled tree is some other folder,
        # and the write is not refused by that prefix.
        assert path_policy.prefix_deny_reason(
            Path("c:/windows/x"), for_write=True
        ) is None


class TestWindowsValuesArriveFromTheEnvironment:
    """B-D1/G-D8: the Windows forms join the lists from the environment alone.

    The candidate helpers are pure over a mapping, so both the present and the
    absent answer are assertable on one runner — and the module's own lists are
    what this machine's environment produced, whatever that is.
    """

    def test_the_candidates_follow_the_named_variables(self):
        env = {
            "SystemRoot": r"C:\Windows",
            "ProgramFiles": r"C:\Program Files",
            "ProgramW6432": r"C:\Program Files",
            "ProgramFiles(x86)": r"C:\Program Files (x86)",
            "ProgramData": r"C:\ProgramData",
            "TEMP": r"C:\Users\vgr\AppData\Local\Temp",
            "TMP": r"C:\Users\vgr\AppData\Local\Temp",
        }
        deny = path_policy.system_deny_prefix_candidates(env)
        assert r"C:\Windows" in deny
        assert r"C:\Program Files (x86)" in deny
        assert deny.count(r"C:\Program Files") == 2  # ProgramFiles states it twice
        assert len(path_policy._dedupe(deny)) == 4
        scratch = path_policy.scratch_prefix_candidates(env)
        assert set(scratch) == {r"C:\Users\vgr\AppData\Local\Temp"}

    def test_no_guardrail_is_stated_twice_in_the_module_lists(self):
        """The dedupe is real: one guardrail, one entry, whatever the box names."""
        for entries in (
            path_policy.ALLOW_PREFIXES,
            path_policy.DENY_PREFIXES,
            path_policy.DENY_PREFIXES_READ,
        ):
            assert len(entries) == len(set(entries))

    def test_the_scratch_allowance_is_the_temp_folder_not_its_parent(self):
        # B-D2's narrowness: the allowance names the temp directory itself,
        # never the profile it lives inside.
        env = {"TEMP": r"C:\Users\vgr\AppData\Local\Temp"}
        candidates = path_policy.scratch_prefix_candidates(env)
        assert candidates and all(
            not path_policy.under_prefix(Path(r"C:\Users\vgr"), c) for c in candidates
        )

    def test_an_empty_or_relative_value_is_skipped(self):
        env = {"SystemRoot": "", "ProgramFiles": "relative/path", "TEMP": "  "}
        assert path_policy.system_deny_prefix_candidates(env) == []
        assert path_policy.scratch_prefix_candidates(env) == []

    def test_the_hive_directory_joins_the_read_list_from_the_system_root(self):
        env = {"SystemRoot": r"C:\Windows"}
        read = path_policy.system_read_deny_prefix_candidates(env)
        # Separator as the OS states it, kept verbatim: ``prefix_hit`` folds
        # backslashes itself, so the entry speaks the drive form either way.
        assert read == [r"C:\Windows\System32\config"]
        assert path_policy.system_read_deny_prefix_candidates({}) == []

    def test_the_module_lists_speak_this_runners_own_working_location_form(self):
        """Whatever the box is, the lists in force speak its form (A-D4's gate)."""
        from src.config import path_location

        resolved = Path(path_location()).resolve().as_posix()
        assert path_policy.lists_are_meaningful(resolved)
