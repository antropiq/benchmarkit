"""Path-policy tests for the fedit connector.

Issue #1 was an asymmetry: the deny-lists were applied on the write path only,
so ``read_file`` returned ``~/.git-credentials`` and ``preview_edit`` happily
diffed ``~/.ssh/config``. These tests pin both halves of the fix -- reads are
now held to a policy, and the ordinary editing workflow is untouched.
"""
from pathlib import Path

from mcp_connectors import fedit_mcp as f

DENIED_READ = "may not be read by this connector"
DENIED_WRITE = "may not be written by this connector"
EDIT = "<<<<<<< SEARCH\nx\n=======\ny\n>>>>>>> REPLACE"


def test_reads_of_credential_stores_are_refused():
    for target in (
        Path.home() / ".git-credentials",
        Path.home() / ".ssh" / "config",
        Path.home() / ".ssh" / "id_ed25519",
        "/etc/shadow",
        "/etc/ssh/sshd_config",
    ):
        out = f.read_file(str(target))
        assert DENIED_READ in out, target


def test_preview_is_not_more_permissive_than_apply(tmp_path):
    """A dry run discloses the file just as plainly as writing it would."""
    for target in (Path.home() / ".ssh" / "config", "/etc/shadow"):
        out = f.preview_edit(str(target), EDIT)
        assert DENIED_WRITE in out, target
        assert "@@" not in out, f"preview leaked a diff for {target}"


def test_make_diff_and_list_edits_check_their_arguments():
    assert DENIED_READ in f.make_diff(str(Path.home() / ".git-credentials"), "/dev/null")
    assert DENIED_READ in f.list_edits(str(Path.home() / ".ssh" / "config"))


def test_denied_paths_never_disclose_existence():
    # A "not found" answer for a path inside ~/.ssh would itself disclose what
    # the store holds, so the policy is judged before the file is looked for.
    out = f.read_file(str(Path.home() / ".ssh" / "definitely-not-a-real-key"))
    assert DENIED_READ in out
    assert "not found" not in out


def test_traversal_and_symlinks_cannot_smuggle_a_secret():
    assert ".." in f.read_file(f"{Path.home()}/.ssh/../../etc/shadow")
    assert ".." in f.read_file("/tmp/../etc/shadow")


def test_a_symlink_under_an_allowed_name_cannot_reach_a_secret(tmp_path):
    # A path whose own name looks harmless must still be judged as the file it
    # really points at. Uses a directory named like a credential store so the
    # test does not depend on what happens to live in the developer's home.
    store = tmp_path / "credentials"
    store.mkdir()
    secret = store / "token.txt"
    secret.write_text("super-secret-token", encoding="utf-8")
    link = tmp_path / "notes.txt"
    link.symlink_to(secret)
    assert DENIED_READ in f.read_file(str(link))
    assert DENIED_WRITE in f.apply_edit(str(link), EDIT)
    assert "super-secret-token" not in f.read_file(str(link))
    # The same file reached without the symlink is judged on its real path.
    assert DENIED_READ in f.read_file(str(secret))


def test_secret_shaped_names_are_refused_wherever_they_live():
    for name in (".env", "prod.env", "server.pem", "vault.key", "backup.kdbx"):
        out = f.read_file(f"/definitely/not/real/{name}")
        assert DENIED_READ in out, name


def test_writes_still_refuse_system_prefixes_and_own_sources():
    for target in ("/etc/hosts", "/usr/bin/ls",
                   Path.home() / ".aws" / "credentials"):
        out = f.apply_edit(str(target), EDIT)
        assert DENIED_WRITE in out, target


def test_harmless_system_files_and_sources_stay_readable():
    """The fix must not turn the connector useless: reads of ordinary files
    (including its own source, so an agent can see what constrains it)."""
    for target in ("/etc/hosts", "/etc/passwd", f.__file__):
        out = f.read_file(target)
        assert out.startswith("File:"), target
    # A library inside the interpreter installation is readable but not writable.
    site = Path(f.__file__).parent
    assert DENIED_WRITE in f.apply_edit(str(site / "fedit_mcp.py"), EDIT)


def test_editing_round_trip_still_works(tmp_path):
    target = tmp_path / "app.py"
    target.write_text("def greet(name):\n    return f'hi {name}'\n", encoding="utf-8")
    out = f.apply_edit(
        str(target),
        "<<<<<<< SEARCH\n    return f'hi {name}'\n=======\n"
        "    return f'hello {name}'\n>>>>>>> REPLACE",
    )
    assert "hello" in target.read_text(encoding="utf-8")
    assert "Backup:" in out
    assert f.revert_edit(str(target)).startswith("Reverted")
    assert target.read_text(encoding="utf-8") == (
        "def greet(name):\n    return f'hi {name}'\n")


def test_policy_reports_both_lists():
    report = f.edit_policy()
    assert "denied to READS" in report
    assert "denied to WRITES" in report
