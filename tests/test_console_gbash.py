"""Behaviour tests for the gbash toolset mode (target B1, gb-02/gb-03/gb-04).

The mode is forced through its named seam -- ``TOOLSET``, ``GIT_SEARCH_PATHS``,
the discovery candidates, and the explicit tokenizer flag -- never by
pretending to be another OS (roadmap §9 rule 2). What needs a real Windows
tree (the guardrail values from the environment, System32's ``more.com``, a
drive-rooted root) carries the ``windows_behaviour`` marker and skips with a
sentence on Linux; everything else asserts both directions on one runner.

The Linux gate itself is ``test_console_policy.py``: it exercises the same
file with the default mode and must stay green with zero skips added here.
"""
import os
import shutil
import sys
from pathlib import Path
from typing import List

import pytest

from mcp_connectors import console_mcp as c


@pytest.fixture(autouse=True)
def posix_default(monkeypatch):
    """Every test starts from today's mode and re-pins it after patching the seam."""
    monkeypatch.setattr(c, "TOOLSET", c.POSIX_TOOLSET)
    monkeypatch.setattr(c, "GIT_SEARCH_PATHS", [])
    monkeypatch.setattr(c, "ROOT_DIR", None)


def in_gbash_mode(monkeypatch, search_paths=None):
    """Force the gbash half through the seam (startup is what selects it in life)."""
    monkeypatch.setattr(c, "TOOLSET", c.GBASH_TOOLSET)
    monkeypatch.setattr(c, "GIT_SEARCH_PATHS", search_paths or [])


def refused(command: str) -> str:
    ok, msg = c.validate_command(command)
    assert not ok, f"expected refusal for {command!r}"
    return msg


# --- G-D5: the tokenizer flag, asserted in both directions on one runner ----

def test_backslashes_survive_under_the_gbash_flag_and_mangle_under_the_posix_one():
    command = r"cat C:\Users\vgr\a.txt"
    gbash = c.parse_shell_statements(command, literal_backslash=True)
    assert gbash[0].argv == ["cat", r"C:\Users\vgr\a.txt"]
    posix = c.parse_shell_statements(command, literal_backslash=False)
    assert posix[0].argv == ["cat", "C:Usersvgra.txt"]  # today's Linux answer, verbatim


def test_quoting_still_groups_spaces_while_the_backslashes_survive():
    argv = c.parse_shell_statements(
        r'cat "C:\Users\vgr\a file.txt"', literal_backslash=True
    )[0].argv
    assert argv == ["cat", r"C:\Users\vgr\a file.txt"]


def test_the_root_shape_selects_the_flag_without_any_test_editing(monkeypatch):
    """A drive-shaped root turns the path model on; a POSIX-shaped root keeps it exact.

    Both roots are shapes, never opened: the mode is a probe of the root's
    spelling (roadmap §9 rule 2), so each direction asserts on every runner.
    """
    drive_root = Path("C:/workspace")
    monkeypatch.setattr(c, "ROOT_DIR", drive_root)
    assert c.path_model_backslash() is True
    ok, msg = c.validate_command(r"cat C:\Users\vgr\a.txt")
    argv = c.parse_shell_statements(
        r"cat C:\Users\vgr\a.txt", c.path_model_backslash()
    )[0].argv
    assert argv[1] == r"C:\Users\vgr\a.txt", (ok, msg)

    posix_root = Path("/posix-root-shape")
    monkeypatch.setattr(c, "ROOT_DIR", posix_root)
    assert c.path_model_backslash() is False


# --- G-D6: one path model, mount form translated for the verdict only -------

def test_the_mount_form_is_translated_for_the_verdict_only(monkeypatch):
    monkeypatch.setattr(c, "ROOT_DIR", Path("/posix-root-shape"))
    assert c.verdict_path_text("/c/Windows/x") == "/c/Windows/x"  # POSIX root: unchanged

    monkeypatch.setattr(c, "ROOT_DIR", Path("C:/workspace"))
    assert c.verdict_path_text("/c/Windows/x") == "C:/Windows/x"
    assert c.verdict_path_text("/c") == "C:/"
    assert c.verdict_path_text("/etc/hosts") == "/etc/hosts"  # not a mount form
    assert c.verdict_path_text(r"C:\dir\x") == r"C:\dir\x"     # already a drive path
    # argv reaches the tool unchanged, whatever the verdict translated:
    argv = c.parse_shell_statements("ls /c/Windows", literal_backslash=True)[0].argv
    assert argv == ["ls", "/c/Windows"]


def test_a_drive_relative_spec_is_refused_by_name(monkeypatch):
    monkeypatch.setattr(c, "ROOT_DIR", Path("/posix-root-shape"))
    # Under a POSIX-shaped root today's verdict stands: `C:x` is an ordinary name there.
    ok, msg = c.validate_command("cat C:x")
    assert ok, msg

    monkeypatch.setattr(c, "ROOT_DIR", Path("C:/workspace"))
    msg = refused("cat C:x")
    assert "drive-relative" in msg
    assert refused("cat c:")
    # A qualified drive path is not this shape: it is judged by the lists.
    assert not c.drive_relative_spec(r"C:\dir\x")
    assert not c.drive_relative_spec("C:/dir")
    assert not c.drive_relative_spec("//server/share/x")


# --- G-D3/G-D7: the command sets, and the shells named ----------------------

def test_shells_are_refused_by_name_naming_the_design_on_both_platforms():
    for shell in ("bash", "sh", "cmd", "powershell", "pwsh"):
        msg = refused(f"{shell} -c whoami")
        assert "no shell" in msg or "runs no shell" in msg or "runs none" in msg, (shell, msg)
        assert "interpreter" in msg, (shell, msg)
    # The in-root bypass never applies to them either.
    assert refused("bash ./script.sh")


def test_the_gbash_sets_replace_the_absent_names_and_keep_the_grammar(tmp_path, monkeypatch):
    in_gbash_mode(monkeypatch)
    monkeypatch.setattr(c, "ROOT_DIR", tmp_path.resolve())
    # Names the probe found nowhere leave the list (B-D4's principle)...
    assert refused("rg TODO")
    assert refused("tree -L 2")
    # ...the System32 half joins it, and `fc` is a real tool here, not a builtin.
    assert "fc" in c.GBASH_ALLOWED_COMMANDS
    assert "fc" not in c.GBASH_BUILTIN_COMMANDS
    assert "fc" in c.SHELL_BUILTIN_COMMANDS  # the POSIX list is untouched
    assert "findstr" in c.GBASH_ALLOWED_COMMANDS
    # cmd's own words are refused with the builtin hint, not FileNotFoundError.
    for builtin in ("cls", "copy", "move", "ren", "type", "start"):
        msg = refused(f"{builtin} x")
        assert "builtin" in msg, (builtin, msg)
    # The POSIX mode keeps its own builtin set verbatim: `type` is not a
    # builtin there, so out of the root it meets the allow list, not the hint.
    monkeypatch.setattr(c, "TOOLSET", c.POSIX_TOOLSET)
    msg = refused("type /home/anyone/x")
    assert "builtin" not in msg and "not in the allowed commands list" in msg, msg
    msg = refused("fc /home/anyone/a /home/anyone/b")  # POSIX: fc is a builtin there
    assert "builtin" in msg, msg


def test_the_posix_toolset_verdicts_are_unchanged_by_the_mode_existing(tmp_path, monkeypatch):
    # The default mode is the POSIX one and its list is today's, entry for entry.
    assert c.TOOLSET is c.POSIX_TOOLSET
    assert c.TOOLSET.allowed == frozenset(c.ALLOWED_COMMANDS)
    assert c.TOOLSET.builtins == c.SHELL_BUILTIN_COMMANDS
    monkeypatch.setattr(c, "ROOT_DIR", tmp_path.resolve())
    for command in ("rg TODO", "make test", "tree -L 2"):
        ok, msg = c.validate_command(command)
        assert ok, f"{command!r} should stay allowed in the POSIX mode: {msg}"


# --- G-D4: discovery, resolution order, the resolver folded into the verdict --

def _plant(directory: Path, name: str) -> Path:
    """An executable the resolver can find, spelled for this platform."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (f"{name}.exe" if os.name == "nt" else name)
    path.write_text("#!/bin/sh\necho planted\n", encoding="utf-8")
    if os.name != "nt":
        path.chmod(0o755)
    return path


def test_resolution_searches_the_git_trees_before_the_ambient_path(tmp_path, monkeypatch):
    first_dir = tmp_path / "git-usr-bin"
    second_dir = tmp_path / "git-bin"
    first = _plant(first_dir, "gbash-first")
    _plant(second_dir, "gbash-first")  # the same name in the second tree too
    in_gbash_mode(monkeypatch, search_paths=[str(first_dir), str(second_dir)])
    found = c.resolve_command("gbash-first")
    assert found is not None and Path(found).parent == first.parent
    # A name no tree carries resolves from the ambient PATH -- or not at all.
    assert c.resolve_command("gbash-no-such-tool") is None


def test_an_unresolved_command_is_refused_naming_what_was_searched(tmp_path, monkeypatch):
    in_gbash_mode(monkeypatch, search_paths=[str(tmp_path / "empty-tree")])
    monkeypatch.setattr(c, "ROOT_DIR", tmp_path.resolve())
    msg = refused("gbash-no-such-tool x")
    assert "not found in the searched toolset" in msg
    assert str(tmp_path / "empty-tree") in msg
    assert "FileNotFoundError" not in msg
    # The oracle and the execution agree: execute never reaches a missing binary.
    out = c.execute_command("gbash-no-such-tool x")
    assert "not found in the searched toolset" in out
    # In the POSIX mode an absent command surfaces at exec exactly as it always did.
    monkeypatch.setattr(c, "TOOLSET", c.POSIX_TOOLSET)
    out = c.execute_command("gbash-no-such-tool x")
    assert "Command not found" in out


def _fake_git_tree(directory: Path) -> Path:
    """A directory spelled like a Git install root, toolset marker included."""
    (directory / "usr" / "bin").mkdir(parents=True)
    (directory / "usr" / "bin" / "ls.exe").write_text("x", encoding="utf-8")
    return directory


def test_the_discovery_questions_and_the_refusal_sentence(monkeypatch, tmp_path):
    monkeypatch.setattr(c, "GIT_INSTALL_CANDIDATES", [str(tmp_path / "no-git-here")])
    monkeypatch.delenv(c.GIT_ENV_OVERRIDE, raising=False)
    monkeypatch.setattr(c.shutil, "which", lambda name, *a, **k: None)
    assert c.discover_git_install() is None
    sentence = c.no_toolset_refusal()
    assert "no Git for Windows toolset found" in sentence
    assert c.GIT_ENV_OVERRIDE in sentence
    # A candidate carrying usr\bin\ls.exe is found; one that does not is not.
    fake_root = _fake_git_tree(tmp_path / "fake-git")
    monkeypatch.setattr(c, "GIT_INSTALL_CANDIDATES", [str(fake_root)])
    assert c.discover_git_install() == str(fake_root)
    assert c.git_search_paths(str(fake_root))[0].endswith(os.path.join("usr", "bin"))


def test_a_stray_git_exe_selects_no_mode(monkeypatch, tmp_path):
    """G-D4's verification, asserted on its own: a `git.exe` with no toolset is not a root.

    The walk-up from a planted `cmd/git.exe` climbs far enough to be judged, and
    the answer is None — the mode needs `usr\\bin\\ls.exe`, not a launcher.
    """
    stray = _plant(tmp_path / "somewhere" / "cmd", "git")
    monkeypatch.setattr(c.shutil, "which", lambda name, *a, **k: str(stray))
    monkeypatch.setattr(c, "GIT_INSTALL_CANDIDATES", [str(tmp_path / "no-git-here")])
    monkeypatch.delenv(c.GIT_ENV_OVERRIDE, raising=False)
    assert c.discover_git_install() is None


# --- csl-01: the probe is one module the connector delegates to --------------

def test_the_candidates_and_the_override_are_the_shared_probes_default(
    tmp_path, monkeypatch
):
    """One override name, one candidate list: the pane and the agent must not
    disagree about where Git is, because they no longer each ask."""
    from mcp_connectors import gitbash as probe

    assert c.GIT_ENV_OVERRIDE == probe.GIT_ENV_OVERRIDE
    assert c.GIT_INSTALL_CANDIDATES == probe.GIT_INSTALL_CANDIDATES
    # The connector-local env override still answers the connector's question.
    fake_root = _fake_git_tree(tmp_path / "override-git")
    monkeypatch.setattr(c.shutil, "which", lambda name, *a, **k: None)
    monkeypatch.setattr(c, "GIT_INSTALL_CANDIDATES", [str(tmp_path / "no-git-here")])
    monkeypatch.setenv(c.GIT_ENV_OVERRIDE, str(fake_root))
    assert c.discover_git_install() == str(fake_root)
    # And it stays a process-environment name, never a managed .env key: the
    # template's key set and the committed example do not carry it.
    from src.config import SETTING_KEYS

    assert c.GIT_ENV_OVERRIDE not in SETTING_KEYS
    example = Path(probe.__file__).parent.parent / ".env.example"
    assert c.GIT_ENV_OVERRIDE not in example.read_text(encoding="utf-8")


def test_the_discovery_and_search_answers_come_from_the_shared_probe(
    tmp_path, monkeypatch
):
    """csl-01's delegation: the connector answers by asking the probe.

    A fake tree discovered through *one* call is seen by both callers with the
    same search order -- the same "one probe" assertion the fscase move has --
    and the patched candidates reach the probe, so the hermetic seam survives
    the move.
    """
    from mcp_connectors import gitbash as probe

    fake_root = _fake_git_tree(tmp_path / "one-probe-git")
    (fake_root / "ucrt64" / "bin").mkdir(parents=True)
    monkeypatch.setattr(c.shutil, "which", lambda name, *a, **k: None)
    monkeypatch.setattr(c, "GIT_INSTALL_CANDIDATES", [str(fake_root)])
    monkeypatch.delenv(c.GIT_ENV_OVERRIDE, raising=False)

    calls: List[str] = []
    real_probe = probe.has_git_toolset

    def counting(candidate: str):
        calls.append(candidate)
        return real_probe(candidate)

    monkeypatch.setattr(probe, "has_git_toolset", counting)
    connector_answer = c.discover_git_install()
    assert calls == [str(fake_root)], calls  # the connector asked the probe, once
    assert connector_answer == probe.discover_git_install(
        [str(fake_root)], c.GIT_ENV_OVERRIDE, {}
    ) == str(fake_root)
    assert c.git_search_paths(str(fake_root)) == probe.git_search_paths(str(fake_root))
    # The shell answer belongs to the probe alone: the connector runs no shell,
    # and it is the terminal (csl-05) that asks it.
    assert probe.git_shell_program(str(fake_root)) is None
    assert not hasattr(c, "git_shell_program")


def test_the_search_path_names_the_tool_tree_that_is_actually_there(
    tmp_path, monkeypatch
):
    """No task may assume `mingw64` exists: the answer comes from the filesystem.

    Both directions assert on one runner -- a tree carrying `ucrt64` only, and
    one carrying neither -- because the probe is a probe, not a platform name.
    """
    from mcp_connectors import gitbash as probe

    monkeypatch.setattr(probe, "GIT_TOOL_TREES", ["mingw64", "ucrt64"])
    root = _fake_git_tree(tmp_path / "ucrt-only")
    (root / "ucrt64" / "bin").mkdir(parents=True)
    paths = probe.git_search_paths(str(root))
    assert paths == [
        os.path.join(str(root), "usr", "bin"),
        os.path.join(str(root), "bin"),
        os.path.join(str(root), "ucrt64", "bin"),
    ]

    bare = _fake_git_tree(tmp_path / "no-tool-tree")
    assert probe.git_search_paths(str(bare)) == [
        os.path.join(str(bare), "usr", "bin"),
        os.path.join(str(bare), "bin"),
    ]
    # An absent directory is not added, so it never appears in the refusal's
    # "searched here" sentence either.
    assert c.git_search_paths(str(bare))[-1] == os.path.join(str(bare), "bin")


def test_the_shell_program_is_git_bashes_and_is_verified(tmp_path):
    """The terminal's question (csl-05): which program is the interactive shell.

    Answering it here does not make the console connector able to run a shell:
    `bash` is refused by name whatever the probe answers (G-D3, asserted by
    test_shells_are_refused_by_name_naming_the_design_on_both_platforms).
    """
    from mcp_connectors import gitbash as probe

    root = _fake_git_tree(tmp_path / "with-bash")
    (root / "bin").mkdir(parents=True)
    (root / "bin" / "bash.exe").write_text("#!/bin/sh\n", encoding="utf-8")
    assert probe.git_shell_program(str(root)) == os.path.join(
        str(root), "bin", "bash.exe"
    )

    root_without = _fake_git_tree(tmp_path / "without-bash")
    assert probe.git_shell_program(str(root_without)) is None
    assert probe.git_shell_program(None) is None
    assert probe.git_shell_program("") is None


def test_the_child_env_prepends_gits_directories_and_keeps_the_rest(tmp_path):
    """csl-01's fourth answer: the environment a shell or tool started under Git sees.

    A PATH entry carrying a space (the real one: `C:\\Program Files\\Git`) is
    forwarded verbatim, the base environment's other keys are untouched, and the
    order is Windows-form-first -- so a Windows child is not handed MSYS entries.
    """
    from mcp_connectors import gitbash as probe

    root = _fake_git_tree(tmp_path / "Git with space")  # a space *inside* an entry
    (root / "ucrt64" / "bin").mkdir(parents=True)
    ambient = os.pathsep.join(
        [str(tmp_path / "spaced dir"), str(tmp_path / "ambient"), ""]
    )
    base = {"PATH": ambient, "SYSTEMROOT": r"C:\WINDOWS", "TERM": "xterm-256color"}
    child = probe.git_child_env(base, str(root))

    entries = child["PATH"].split(os.pathsep)
    assert entries[:3] == [
        os.path.join(str(root), "usr", "bin"),
        os.path.join(str(root), "bin"),
        os.path.join(str(root), "ucrt64", "bin"),
    ]
    # The ambient entries keep their order and their spaces; no empty part survives.
    assert entries[3:] == [str(tmp_path / "spaced dir"), str(tmp_path / "ambient")]
    assert str(root) in entries[0]
    # Only PATH moves; every other key of the base is copied as it was.
    assert child["SYSTEMROOT"] == base["SYSTEMROOT"]
    assert child["TERM"] == base["TERM"]
    assert set(child) - {"PATH"} == set(base) - {"PATH"}
    # An environment with no PATH at all still gets Git's trees.
    no_path = probe.git_child_env({"SYSTEMROOT": r"C:\WINDOWS"}, str(root))
    assert no_path["PATH"].split(os.pathsep)[:2] == [
        os.path.join(str(root), "usr", "bin"),
        os.path.join(str(root), "bin"),
    ]
    # The base mapping itself is never mutated.
    assert base["PATH"] == ambient


@pytest.mark.windows_behaviour(
    "the tool tree this names is a fact about the Git install on this machine "
    "(it carries ucrt64 and no mingw64); a Linux tmp_path carries neither"
)
def test_the_real_git_roots_tool_tree_is_the_one_that_exists():
    """Measured, not assumed: the startup search path names directories that exist.

    The Windows half of the verification: the trees the connector searches are
    the ones this install actually carries, and a POSIX tool the terminal tests
    ask for resolves out of them rather than from System32.
    """
    refusal = c.select_toolset()
    if refusal is not None:
        pytest.skip(refusal)  # no Git install on this box: the seam says so out loud
    assert c.TOOLSET is c.GBASH_TOOLSET
    assert c.GIT_SEARCH_PATHS[0].endswith(os.path.join("usr", "bin"))
    for directory in c.GIT_SEARCH_PATHS:
        assert Path(directory).is_dir(), directory
    found = shutil.which("stty", path=os.pathsep.join(c.GIT_SEARCH_PATHS))
    assert found is not None, "stty does not resolve from the searched trees"
    assert Path(found).parent in [Path(directory) for directory in c.GIT_SEARCH_PATHS]
    from mcp_connectors import gitbash as probe

    assert probe.git_shell_program(c.discover_git_install()) is not None


@pytest.mark.windows_behaviour(
    "the refusal is what a Windows box with no Git install does at startup: "
    "select_toolset only refuses there, and main() exits non-zero naming the gap"
)
def test_windows_without_a_toolset_refuses_to_start_naming_the_gap(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(c, "GIT_INSTALL_CANDIDATES", [str(tmp_path / "no-git-here")])
    monkeypatch.delenv(c.GIT_ENV_OVERRIDE, raising=False)
    monkeypatch.setattr(c.shutil, "which", lambda name, *a, **k: None)
    refusal = c.select_toolset()
    assert refusal is not None and "no Git for Windows toolset found" in refusal
    assert c.TOOLSET is c.POSIX_TOOLSET  # the refusal installs no half-mode

    monkeypatch.setattr(
        sys, "argv",
        ["console_mcp", "--root-dir", str(tmp_path), "--transport", "stdio"],
    )
    with pytest.raises(SystemExit) as raised:
        c.main()
    assert raised.value.code == 1
    err = capsys.readouterr().err
    assert "refused to start" in err and "Git for Windows" in err


# --- the Windows half, forced by the real Windows facts ---------------------

@pytest.mark.windows_behaviour(
    "the guardrail values this judges come from the Windows environment "
    "(SystemRoot & friends); on Linux those names do not exist"
)
def test_the_windows_guardrails_judge_c_windows_component_wise_and_folded(
    tmp_path, monkeypatch
):
    planted = _plant(tmp_path, "ls")
    in_gbash_mode(monkeypatch, search_paths=[str(planted.parent)])
    monkeypatch.setattr(c, "ROOT_DIR", tmp_path.resolve())
    # One verdict for every spelling -- case-folded comparison, exact storage.
    verdicts = set()
    for spelling in (r"C:\Windows", r"c:\windows", r"C:\WINDOWS"):
        ok, msg = c.validate_command(f"ls {spelling}")
        verdicts.add((ok, "protected prefix" in msg))
    assert len(verdicts) == 1, verdicts
    # The hive directory (the /etc/shadow analogue) is refused whoever spells it.
    assert refused(r"ls C:\WINDOWS\System32\config")
    # Component-wise: C:\Windows.old is not under C:\Windows -- the write half
    # (SystemRoot is a DENY_PREFIX) is where that distinction bites.
    hits = c.path_policy.prefix_hit
    assert hits(r"C:\Windows\x", c.path_policy.DENY_PREFIXES, fold=True) is not None
    assert hits(r"C:\Windows.old", c.path_policy.DENY_PREFIXES, fold=True) is None
    assert hits(r"C:\Windows.old\x", c.path_policy.DENY_PREFIXES, fold=True) is None


@pytest.mark.windows_behaviour(
    "`more.com` lives in System32 and only resolves through PATHEXT; the "
    "invariant is that the connector honours PATHEXT without a shell"
)
def test_a_com_shim_resolves_via_pathext(monkeypatch):
    in_gbash_mode(monkeypatch)
    found = c.resolve_command("more")
    assert found is not None and found.lower().endswith(".com")


@pytest.mark.windows_behaviour(
    "the resolution order this asserts (Git's GNU binaries before System32's "
    "find/sort/tar) needs a real Git install and a real System32"
)
def test_find_sort_tar_resolve_to_gits_gnu_binaries(monkeypatch):
    refusal = c.select_toolset()
    if refusal is not None:
        pytest.skip(refusal)  # no Git install on this box: the seam says so out loud
    assert c.TOOLSET is c.GBASH_TOOLSET
    git_bin = c.GIT_SEARCH_PATHS[0]
    for name in ("find", "sort", "tar", "grep"):
        found = c.resolve_command(name)
        assert found is not None, name
        assert Path(found).parent == Path(git_bin), (name, found)
    # Ambient-only names resolve too (B-D4's search order ends at PATH).
    assert c.resolve_command("ping") is not None


@pytest.mark.windows_behaviour(
    "%TEMP% is a Windows environment fact; the scratch allowance and its "
    "ordering against the component half is what the test pins"
)
def test_temp_is_scratch_but_a_store_named_inside_it_is_refused(tmp_path, monkeypatch):
    planted = _plant(tmp_path, "cat")
    in_gbash_mode(monkeypatch, search_paths=[str(planted.parent)])
    monkeypatch.setattr(c, "ROOT_DIR", tmp_path.resolve())
    temp = os.environ.get("TEMP")
    assert temp, "no %TEMP% to judge"
    ok, msg = c.validate_command(f"cat {os.path.join(temp, 'scratch.txt')}")
    assert ok, msg
    msg = refused(f"cat {os.path.join(temp, '.ssh', 'id_rsa')}")
    assert "off-limits" in msg  # scratch does not outrank the component rule


# --- pol-08 through the Windows path model (same sentences as the pane) ----

def test_hidden_is_judged_through_the_gbash_mode_with_its_own_sentence(tmp_path, monkeypatch):
    """Runner-neutral in verdict, Windows-shaped in spelling."""
    import json

    from mcp_connectors import policy_file as connector_policy
    from mcp_connectors import policy_shared as shared

    monkeypatch.setattr(c, "ROOT_DIR", tmp_path.resolve())
    monkeypatch.setattr(c, "TOOLSET", c.GBASH_TOOLSET)
    monkeypatch.setattr(c, "GIT_SEARCH_PATHS", [str(tmp_path)])

    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "notes.md").write_text("one\n", encoding="utf-8")
    folder = tmp_path / shared.POLICY_DIRNAME
    folder.mkdir()
    (folder / shared.POLICY_FILENAME).write_text(
        json.dumps(
            {"version": shared.POLICY_VERSION,
             "policies": [{"path": "vault", "mode": "hidden"}]},
            indent=2,
        ),
        encoding="utf-8",
    )
    connector_policy.invalidate_workspace_policy_cache()

    rel = f"vault{os.sep}notes.md"
    msg = refused(f"cat {rel}")
    assert "off-limits" in msg and '"vault"' in msg, msg
    out = c.execute_command("ls", cwd="vault")
    assert "Error" in out and '"vault"' in out, out
