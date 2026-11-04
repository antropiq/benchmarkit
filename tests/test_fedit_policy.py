"""Path-policy tests for the fedit connector.

Issue #1 was an asymmetry: the deny-lists were applied on the write path only,
so ``read_file`` returned ``~/.git-credentials`` and ``preview_edit`` happily
diffed ``~/.ssh/config``. These tests pin both halves of the fix -- reads are
now held to a policy, and the ordinary editing workflow is untouched.

The last section belongs to pol-08: inside the root the connector obeys the
workspace's own ``.nojae/read-write-policy.json``, the same file and the same
verdict the Editor tab applies, so a folder its owner marked ``ro`` or
``hidden`` is off-limits to an agent too. Those tests pin the two floors'
*domains* apart as well -- the workspace policy speaks inside the root, the
connector deny-list outside it, and neither is loosened by the other.
"""
import json
import os
import stat
import time
from pathlib import Path

import pytest

from mcp_connectors import fedit_mcp as f
from mcp_connectors import policy_file as connector_policy
from mcp_connectors import policy_shared as shared

DENIED_READ = "may not be read by this connector"
DENIED_WRITE = "may not be written by this connector"
EDIT = "<<<<<<< SEARCH\nx\n=======\ny\n>>>>>>> REPLACE"
# Same shape as EDIT, but for the fixtures below whose file says "one".
EDIT_ONE = "<<<<<<< SEARCH\none\n=======\nx\n>>>>>>> REPLACE"


@pytest.fixture(autouse=True)
def _backups_go_to_tmp(tmp_path, monkeypatch):
    """Keep every test's backups in its own directory.

    ``BACKUP_DIR`` is a module-level default pointing at ``logs/fedit_backups``,
    so without this the suite writes into the developer's workspace -- and the
    F-9 and F-13 tests, which reason about that directory, need it to be empty.
    """
    monkeypatch.setattr(f, "BACKUP_DIR", tmp_path / "fedit-backups")


@pytest.mark.posix_behaviour(
    "the /etc/shadow and /etc/ssh fixtures are POSIX paths: on another "
    "platform the deny-prefixes have nothing to match (the ~/.ssh and "
    ".git-credentials halves are name-based and run everywhere)"
)
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
    """F-8: traversal is refused for where it *lands*, never for the token.

    The old rule rejected any path containing '..' outside the root, so two
    spellings of one in-root file behaved differently. Now the path is
    normalised first and every form of it is held to the deny rules -- so a
    '..' that arrives at a protected file is still refused, by name.
    """
    assert DENIED_READ in f.read_file(f"{Path.home()}/.ssh/../../etc/shadow")
    assert DENIED_READ in f.read_file("/tmp/../etc/shadow")
    assert DENIED_READ in f.read_file("/etc/hosts/../shadow")


@pytest.mark.requires_symlink(
    "a path whose own name looks harmless is judged as the file it really "
    "points at, so a link cannot reach a credential store"
)
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


@pytest.mark.posix_behaviour(
    "the two entries that make this a refusal are the `/etc` and `/usr` deny "
    "prefixes, which speak no Windows path; the ~/.aws half is name-based and "
    "would still be checked"
)
def test_writes_still_refuse_system_prefixes_and_own_sources():
    for target in ("/etc/hosts", "/usr/bin/ls",
                   Path.home() / ".aws" / "credentials"):
        out = f.apply_edit(str(target), EDIT)
        assert DENIED_WRITE in out, target


@pytest.mark.posix_behaviour(
    "the invariant needs /etc/hosts and /etc/passwd to exist and be readable; "
    "the source-under-a-library check is the half that would still be pinned"
)
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


# --- round-trip properties (fedit-issues F-1) -------------------------------


@pytest.mark.posix_behaviour(
    "the invariant is the st_mode round-trip: chmod and the 0600/0755/0444 "
    "bits it checks are POSIX facts, so on another platform the atomic "
    "replace's mode handling is not what this test can observe"
)
def test_an_edit_preserves_the_file_mode(tmp_path):
    """F-1: the atomic write used to leave every edited file at 0600.

    ``mkstemp`` creates 0600 and ``os.replace`` makes that the file's mode, so a
    755 script lost its ``+x`` and a 444 file silently became writable -- a
    change invisible in the returned diff.
    """
    for mode, expected in ((0o755, 0o755), (0o444, 0o444), (0o644, 0o644)):
        target = tmp_path / f"mode-{oct(mode).replace('0o', '')}.txt"
        target.write_text("one\ntwo\n", encoding="utf-8")
        os.chmod(target, mode)
        out = f.apply_edit(
            str(target),
            "<<<<<<< SEARCH\none\n=======\ntwo\n>>>>>>> REPLACE")
        assert out.startswith("Applied"), out
        assert stat.S_IMODE(target.stat().st_mode) == expected, mode
    # A newly created file keeps mkstemp's own (private) mode.
    created = tmp_path / "created.txt"
    assert f.apply_edit(str(created), "<<<<<<< SEARCH\n=======\nnew\n>>>>>>> REPLACE"
                        ).startswith("Applied")
    assert stat.S_IMODE(created.stat().st_mode) == 0o600


def test_an_edit_preserves_line_terminators(tmp_path):
    """F-2: an edit used to rewrite every CRLF file as LF, and ``splitlines()``
    also treated a form feed (and \\x0b, \\u2028, ...) as a line break.

    Only ``\\r\\n``, ``\\n`` and a lone ``\\r`` may end a line, and the style is
    reused on write, so a one-line change stays a one-line change.
    """
    crlf = tmp_path / "crlf.txt"
    crlf.write_bytes(b"one\r\ntwo\r\nthree\r\n")
    out = f.apply_edit(str(crlf), "<<<<<<< SEARCH\ntwo\n=======\nTWO\n>>>>>>> REPLACE")
    assert out.startswith("Applied"), out
    assert crlf.read_bytes() == b"one\r\nTWO\r\nthree\r\n"
    # read_file says what it is dealing with, so the agent is not surprised.
    assert "endings: CRLF" in f.read_file(str(crlf))

    cr_only = tmp_path / "cr.txt"
    cr_only.write_bytes(b"one\rtwo\r")
    assert f.apply_edit(str(cr_only), "<<<<<<< SEARCH\none\n=======\nONE\n>>>>>>> REPLACE"
                        ).startswith("Applied")
    assert cr_only.read_bytes() == b"ONE\rtwo\r"

    exotic = tmp_path / "formfeed.txt"
    exotic.write_bytes(b"page one\n\x0cpage two\nend\n")
    header = f.read_file(str(exotic)).splitlines()[0]
    assert "lines 1-3 of 3" in header, header     # no invented blank line 2
    assert f.apply_edit(str(exotic), "<<<<<<< SEARCH\npage one\n=======\nPAGE ONE\n>>>>>>> REPLACE"
                        ).startswith("Applied")
    assert exotic.read_bytes() == b"PAGE ONE\n\x0cpage two\nend\n"


def test_mixed_line_terminators_are_reported(tmp_path):
    """A file no model can reproduce line-by-line is at least announced."""
    mixed = tmp_path / "mixed.txt"
    mixed.write_bytes(b"one\r\ntwo\nthree\n")
    out = f.apply_edit(str(mixed), "<<<<<<< SEARCH\ntwo\n=======\nTWO\n>>>>>>> REPLACE")
    assert "mixes line terminators" in out
    # Normalised to the dominant style (two LF beats one CRLF), and said so.
    assert mixed.read_bytes() == b"one\nTWO\nthree\n"


# --- encoding: readable, never silently rewritten (fedit-issues F-3, F-12) --


def test_a_file_with_a_nul_is_refused_as_binary(tmp_path):
    """F-12: showing NUL/undecodable bytes as text invites an unwinnable loop
    -- the tool displays a line, asks for it verbatim, then rejects the copy."""
    target = tmp_path / "nul.dat"
    target.write_bytes(b"a\x00b\n")
    for out in (f.read_file(str(target)), f.apply_edit(str(target), EDIT),
                f.preview_edit(str(target), EDIT)):
        assert "NUL byte at offset 1" in out, out
        assert "not editable text" in out, out
    assert target.read_bytes() == b"a\x00b\n"


def test_non_utf8_text_is_readable_but_not_editable(tmp_path):
    """F-3: Latin-1 and friends are real text -- display them, flagged, but
    refuse to rewrite them (the round-trip could not reproduce those bytes)."""
    target = tmp_path / "latin1.dat"
    target.write_bytes(b"caf\xe9 next\n")
    out = f.read_file(str(target))
    assert out.startswith("File:"), out
    assert "NOT valid UTF-8" in out, out
    edit = f.apply_edit(str(target), "<<<<<<< SEARCH\ncaf\u00e9\n=======\nCAFE\n>>>>>>> REPLACE")
    assert "byte 0xe9 at offset 3" in edit, edit
    assert target.read_bytes() == b"caf\xe9 next\n"
    assert "byte 0xe9 at offset 3" in f.make_diff(str(target), str(target)) or "identical" in f.make_diff(str(target), str(target))


def test_make_diff_refuses_a_non_text_side(tmp_path):
    keep = tmp_path / "a.txt"
    keep.write_bytes(b"one\n")
    bad = tmp_path / "b.dat"
    bad.write_bytes(b"\x00\x01\x02\n")
    out = f.make_diff(str(keep), str(bad))
    assert "NUL byte at offset 0" in out
    assert out.startswith("Error:")


def test_a_utf8_bom_survives_a_search_copied_from_read_file(tmp_path):
    """The F-12 addendum: ``read_file`` shows line 1 without the BOM, so a
    verbatim copy of it used to miss the stored '\\ufeffline1'."""
    target = tmp_path / "bom.txt"
    target.write_bytes("\ufeffline1\nline2\n".encode("utf-8"))
    assert "BOM: UTF-8" in f.read_file(str(target))
    out = f.apply_edit(str(target),
                       "<<<<<<< SEARCH\nline1\n=======\nLINE ONE\n>>>>>>> REPLACE")
    assert out.startswith("Applied"), out
    assert target.read_bytes() == "\ufeffLINE ONE\nline2\n".encode("utf-8")


def test_a_payload_reading_as_both_formats_is_asked_about(tmp_path):
    """'auto' must not guess between the two engines when the payload could be
    either -- but a real diff that merely *quotes* a marker stays a diff."""
    target = tmp_path / "doc.md"
    target.write_text("See:\n\n    <<<<<<< SEARCH\n    x\n    =======\n"
                      "    y\n    >>>>>>> REPLACE\n", encoding="utf-8")
    blocks = "<<<<<<< SEARCH\n@@ -1,2 +1,2 @@\n=======\nkept\n>>>>>>> REPLACE"
    out = f.apply_edit(str(target), blocks)
    assert "Ambiguous edit" in out, out
    diff_payload = ("--- a/doc.md\n+++ b/doc.md\n@@ -1,3 +1,3 @@\n See:\n \n"
                    "-    <<<<<<< SEARCH\n+    SEARCH/REPLACE blocks\n")
    # The same payload with its --- / +++ header is unambiguously a diff, even
    # though it quotes a SEARCH marker -- that is how these docstrings get edited.
    applied = f.apply_edit(str(target),
                           diff_payload)
    assert applied.startswith("Applied"), applied


def test_revert_restores_exact_bytes(tmp_path):
    """F-3: a revert is recovery -- it must not depend on decoding anything.

    The old implementation read the backup as UTF-8 text (so a non-UTF-8
    backup was unrecoverable) and wrote it back through universal-newline
    translation (so even a CRLF file came back as LF).
    """
    target = tmp_path / "crlf.txt"
    original = b"one\r\ntwo\r\n"
    target.write_bytes(original)
    assert f.apply_edit(str(target), "<<<<<<< SEARCH\ntwo\n=======\nTWO\n>>>>>>> REPLACE"
                        ).startswith("Applied")
    assert target.read_bytes() == b"one\r\nTWO\r\n"

    # Clobber it the way an external tool would, with bytes the connector could
    # never have produced -- then revert.
    target.write_bytes(b"\xff\xfe scrambled")
    out = f.revert_edit(str(target))
    assert out.startswith("Reverted"), out
    assert target.read_bytes() == original


# --- friction fixes (fedit-issues F-6, F-10, F-7, F-11) ---------------------


def test_an_empty_file_reads_as_empty(tmp_path):
    """F-6: the old answer was an error, which reads as an unreadable path."""
    target = tmp_path / "empty.txt"
    target.write_bytes(b"")
    out = f.read_file(str(target))
    assert out.startswith("File:"), out
    assert "lines 0-0 of 0" in out
    assert "Error" not in out
    # And the create-by-empty-SEARCH path is reachable after confirming it.
    assert f.apply_edit(str(target), "<<<<<<< SEARCH\n=======\nhello\n>>>>>>> REPLACE"
                        ).startswith("Applied")
    assert target.read_text(encoding="utf-8") == "hello\n"


def test_a_missing_parent_directory_says_so(tmp_path):
    """F-10: 'Directory ... is not writable' sends the agent off to chmod when
    what it needs is mkdir."""
    out = f.apply_edit(str(tmp_path / "nosuchdir" / "deep.txt"),
                       "<<<<<<< SEARCH\n=======\nhello\n>>>>>>> REPLACE")
    assert "does not exist" in out, out
    assert "mkdir" in out, out


def test_fuzz_is_clamped_to_the_documented_range(tmp_path):
    """F-7: an unclamped fuzz lets a hunk trim its whole context away, which
    turns 'reposition a little' into 'match anywhere' -- how an edit lands in
    the wrong function of a large file."""
    target = tmp_path / "two.txt"
    target.write_text("alpha\nomega\n", encoding="utf-8")
    stray = ("--- a/two.txt\n+++ b/two.txt\n@@ -40,3 +40,4 @@\n absent one\n"
             " absent two\n+inserted\n absent three\n")
    refused = f.apply_edit(str(target), stray, format="diff", fuzz=50)
    assert "does not match" in refused, refused
    assert target.read_text(encoding="utf-8") == "alpha\nomega\n"
    assert "above the" in refused and "fuzz" in refused, refused

    # A legal fuzz still repositions, saying which context it ignored.
    landing = ("--- a/two.txt\n+++ b/two.txt\n@@ -40,3 +40,3 @@\n absent one\n"
               "-alpha\n+ALPHA\n trailing\n")
    accepted = f.apply_edit(str(target), landing, format="diff", fuzz=1)
    assert "context line(s) ignored" in accepted, accepted


def test_a_file_over_the_cap_offers_a_route_forward(tmp_path):
    """F-11: the cap is right, the dead end is not."""
    big = tmp_path / "huge.bin.txt"
    big.write_text("x" * (f.MAX_FILE_BYTES + 10), encoding="utf-8")
    out = f.read_file(str(big))
    assert "over the" in out
    assert "sed -n" in out, out


# --- ambiguous payloads and path identity (fedit-issues F-4, F-8, F-9, F-13) --


AMBIGUOUS = ("<<<<<<< SEARCH\nSome doc text\n=======\nmore text\n=======\n"
             "Rewritten\n>>>>>>> REPLACE")
MARKER_FILE = "Some doc text\n=======\nmore text\n"


def test_a_payload_whose_body_holds_a_marker_line_is_refused(tmp_path):
    """F-4: the dangerous one. A '=======' line inside the SEARCH text used to
    be re-tokenised as the block's own divider, so the payload parsed as
    several blocks and the edit landed somewhere nobody asked for -- with no
    error and no warning. Refusing is the honest answer; a diff handles it.
    """
    target = tmp_path / "marker.txt"
    target.write_text(MARKER_FILE, encoding="utf-8")
    for out in (f.apply_edit(str(target), AMBIGUOUS),
                f.preview_edit(str(target), AMBIGUOUS)):
        assert "Ambiguous SEARCH/REPLACE payload" in out, out
        assert 'format="diff"' in out, out
        assert target.read_text(encoding="utf-8") == MARKER_FILE, out
    # The route the error names really does work on the same content.
    applied = f.apply_edit(
        str(target),
        "--- a/marker.txt\n+++ b/marker.txt\n@@ -1,2 +1,2 @@\n-Some doc text\n"
        "+A doc text\n =======\n", format="diff")
    assert applied.startswith("Applied"), applied
    assert target.read_text(encoding="utf-8") == "A doc text\n=======\nmore text\n"


def test_a_conflicted_file_cannot_be_edited_with_search_replace(tmp_path):
    """The same trap in its common form: git conflict markers."""
    conflicted = "<<<<<<< HEAD\nours\n=======\ntheirs\n>>>>>>> branch\n"
    target = tmp_path / "conflict.txt"
    target.write_text(conflicted, encoding="utf-8")
    out = f.apply_edit(
        str(target),
        "<<<<<<< SEARCH\n<<<<<<< HEAD\nours\n=======\nOURS\n=======\nX\n>>>>>>> REPLACE")
    assert "Ambiguous SEARCH/REPLACE payload" in out, out
    assert target.read_text(encoding="utf-8") == conflicted


def test_an_edit_of_a_marker_bearing_file_warns_about_the_format(tmp_path):
    """F-5: the safe path has to be named, or the next block walks into F-4."""
    target = tmp_path / "marker.txt"
    target.write_text(MARKER_FILE, encoding="utf-8")
    out = f.apply_edit(str(target),
                       "<<<<<<< SEARCH\nmore text\n=======\nMORE\n>>>>>>> REPLACE")
    assert out.startswith("Applied"), out
    assert "read as SEARCH/REPLACE markers" in out, out
    assert 'format="diff"' in out, out


def test_a_traversing_path_is_judged_where_it_lands(tmp_path):
    """F-8: '..' inside the root is ordinary, '..' outside it is judged by its
    destination -- the token itself is no longer the test."""
    inner = tmp_path / "pkg" / "sub"
    inner.mkdir(parents=True)
    (inner / "notes.txt").write_text("one\n", encoding="utf-8")
    outside = tmp_path / "sibling.txt"
    outside.write_text("x\n", encoding="utf-8")
    traversing = str(inner / "../.." / "sibling.txt")
    assert f.read_file(traversing).startswith("File:")
    assert f.apply_edit(traversing, EDIT).startswith("Applied")
    # A traversal that ends inside a protected name is still refused.
    assert DENIED_WRITE in f.apply_edit("/etc/hosts/../shadow", EDIT)
    assert DENIED_READ in f.read_file(f"{Path.home()}/.config/../.config/gh/config.yml")


@pytest.mark.requires_symlink(
    "two spellings of one file share a policy verdict and a backup chain, which "
    "needs a second name for the same bytes"
)
def test_two_spellings_of_one_file_share_a_policy_and_a_backup_chain(tmp_path):
    """F-9: the workspace here is reachable through two prefixes (a mount
    point). One file must not be editable under one name and protected under
    the other, and the backup chain must be found under either.
    """
    real = tmp_path / "real"
    real.mkdir()
    (tmp_path / "alias").symlink_to(real)
    (real / "plain.txt").write_text("one\n", encoding="utf-8")

    out = f.apply_edit(str(tmp_path / "alias" / "plain.txt"), EDIT_ONE)
    assert out.startswith("Applied"), out
    # Seen through the other spelling, the edit is still visible and revertible.
    listing = f.list_edits(str(real / "plain.txt"))
    assert "1 backup(s)" in listing, listing
    assert f.revert_edit(str(real / "plain.txt")).startswith("Reverted")
    assert (real / "plain.txt").read_text(encoding="utf-8") == "one\n"
    # And a secret is a secret under both spellings.
    (real / ".env").write_text("SECRET=1\n", encoding="utf-8")
    assert DENIED_READ in f.read_file(str(tmp_path / "alias" / ".env"))
    assert DENIED_READ in f.read_file(str(real / ".env"))


def test_old_backups_are_swept(tmp_path):
    """F-13: the per-file keep never collected the backups of a file that was
    renamed or deleted, so a long-lived workspace accumulated them for ever."""
    ancient = time.time() - (f.BACKUP_PRUNE_DAYS + 5) * 86400

    # A backup is stamped with the time of the edit, not the mtime copy2 would
    # carry over -- so this one is still inside the window and survives.
    fresh = tmp_path / "fresh.txt"
    fresh.write_bytes(b"x\n" + b"old but untouched\n" * 400)
    os.utime(fresh, (ancient, ancient))
    assert f.apply_edit(str(fresh), EDIT).startswith("Applied")

    ghost = f.BACKUP_DIR / "000000000000_ghost_20200101-000000.000000.orig"
    ghost.write_bytes(b"gone\n")
    os.utime(ghost, (ancient, ancient))

    assert f.prune_old_backups() == 1
    assert not ghost.exists()
    assert f.revert_edit(str(fresh)).startswith("Reverted")
    # The window is what the tool says it is, not a mystery.
    assert "swept after" in f.edit_policy()


@pytest.mark.requires_symlink(
    "the root bypass must not be reachable *through* a link planted in a "
    "system directory: the address is judged, not only where the target is"
)
def test_the_root_bypass_needs_a_sane_address(tmp_path, monkeypatch):
    """F-9 made the root bypass decide on the resolved target -- it must not
    become a way of writing *through* a link planted in a system directory.

    ``/etc/x -> <root>/f`` physically reaches the workspace, so the target test
    passes; the address test is what refuses it, keeping the pre-existing rule
    that a symlink cannot carry a path into the bypass.
    """
    root = tmp_path / "root"
    root.mkdir()
    (root / "f.txt").write_text("one\n", encoding="utf-8")
    system = tmp_path / "etc"
    system.mkdir()
    (system / "x").symlink_to(root / "f.txt")
    monkeypatch.setattr(f, "ROOT_DIR", root)
    monkeypatch.setattr(f.path_policy, "DENY_PREFIXES", (str(system),))

    assert DENIED_WRITE in f.apply_edit(str(system / "x"), EDIT_ONE)
    # The same file addressed directly is still the operator's to edit.
    assert f.apply_edit(str(root / "f.txt"), EDIT_ONE).startswith("Applied")
    # Reading through the link discloses nothing the operator does not own.
    assert f.read_file(str(system / "x")).startswith("File:")


# --- pol-08: the workspace's own policy constrains the agent inside the root ---


@pytest.fixture
def ws(tmp_path, monkeypatch):
    """A throw-away workspace rooted like --root-dir, with no policy file yet.

    The connector resolves paths against ``ROOT_DIR`` and asks the workspace
    policy through it, so the root *is* the workspace here exactly as
    ``externals.json`` configures it (``--root-dir ${BENCHMARKIT_PATH_LOCATION}``).
    The cache in the connector process is keyed on that root, so it is cleared on
    both sides of every test -- otherwise one test's parsed rules answer the next
    test's paths (the same reason pol-04's app-side cache carries the root).
    """
    root = tmp_path.resolve()
    monkeypatch.setattr(f, "ROOT_DIR", root)
    connector_policy.invalidate_workspace_policy_cache()
    yield root
    connector_policy.invalidate_workspace_policy_cache()


def write_policy(root: Path, rules, *, raw: str = None) -> Path:
    """Drop a ``.nojae/read-write-policy.json`` in ``root``, verbatim or built."""
    folder = root / shared.POLICY_DIRNAME
    folder.mkdir(exist_ok=True)
    path = folder / shared.POLICY_FILENAME
    text = raw if raw is not None else json.dumps(
        {"version": shared.POLICY_VERSION, "policies": list(rules)}, indent=2
    )
    path.write_text(text, encoding="utf-8")
    return path


def test_a_read_only_folder_is_not_editable_by_an_agent(ws: Path):
    """The headline of pol-08: ``ro`` means the same thing to both surfaces."""
    (ws / "secret").mkdir()
    target = ws / "secret" / "notes.md"
    target.write_text("one\n", encoding="utf-8")
    write_policy(ws, [{"path": "secret", "mode": "ro"}])

    out = f.apply_edit(str(target), EDIT_ONE)
    assert DENIED_WRITE in out, out
    # The refusal names the rule the user wrote, in the pane's own words.
    assert '"secret"' in out and "(read-only)" in out
    assert "the workspace policy" in out
    assert "nothing under it may change" in out

    # A dry run is judged by the rules of the write it previews.
    assert DENIED_WRITE in f.preview_edit(str(target), EDIT_ONE)
    assert "@@" not in f.preview_edit(str(target), EDIT_ONE)


def test_a_read_only_folder_stays_readable(ws: Path):
    """``ro`` is not ``hidden``: reading is the reason the mode exists."""
    (ws / "vendor").mkdir()
    target = ws / "vendor" / "README.md"
    target.write_text("one\n", encoding="utf-8")
    write_policy(ws, [{"path": "**/vendor", "mode": "ro"}])

    assert f.read_file(str(target)).startswith("File:")
    out = f.apply_edit(str(target), EDIT_ONE)
    assert DENIED_WRITE in out and "(read-only)" in out, out


def test_a_hidden_path_is_not_readable_either(ws: Path):
    """``hidden`` refuses both directions, and says which pattern did it.

    This is the half that makes the seeded ``**/__pycache__`` rule mean
    "unsearchable by an agent" too, so the pattern has to be in the message for
    the model to ask about something else instead of retrying.
    """
    (ws / "a" / "__pycache__").mkdir(parents=True)
    target = ws / "a" / "__pycache__" / "job.pyc"
    target.write_text("one\n", encoding="utf-8")
    write_policy(ws, [{"path": "**/__pycache__", "mode": "hidden"}])

    out = f.read_file(str(target))
    assert DENIED_READ in out, out
    assert '"**/__pycache__"' in out and "(hidden)" in out
    # A mutation still names the rule rather than pretending absence: "not there"
    # is the green light for a create.
    out = f.apply_edit(str(target), EDIT_ONE)
    assert DENIED_WRITE in out and '"**/__pycache__"' in out


def test_a_carve_out_below_a_folder_rule_edits(ws: Path):
    """Last match wins -- the idiom the policy file is designed around.

    ``**/node_modules`` read-only with one README carved back to ``rw``: the pane
    and the agent must agree on the exception as well as on the rule, so the
    connector's verdict runs the same merge (``policy_shared.verdict_for``).
    """
    module = ws / "web" / "node_modules" / "react"
    module.mkdir(parents=True)
    readme = module / "README.md"
    readme.write_text("one\n", encoding="utf-8")
    source = module / "index.js"
    source.write_text("one\n", encoding="utf-8")
    write_policy(ws, [
        {"path": "**/node_modules", "mode": "ro"},
        {"path": "**/node_modules/**/README.md", "mode": "rw"},
    ])

    assert f.apply_edit(str(readme), EDIT_ONE).startswith("Applied")
    out = f.apply_edit(str(source), EDIT_ONE)
    assert DENIED_WRITE in out and '"**/node_modules"' in out, out


def test_the_reserved_policy_folder_is_addressable_by_neither_surface(ws: Path):
    """An agent rewriting the rules that constrain it is the accident D5 prevents.

    Refused for reading *and* writing, and refused with a deliberately corrupt
    file in place -- the reservation is a component check, not a rule, so a JSON
    typo cannot open the folder that holds the JSON.
    """
    policy = write_policy(ws, [{"path": "src", "mode": "ro"}])
    for read in (True, False):
        call = f.read_file if read else (
            lambda p: f.apply_edit(p, EDIT_ONE))
        out = call(str(policy))
        assert (DENIED_READ if read else DENIED_WRITE) in out, (read, out)
        assert shared.POLICY_DIRNAME in out

    # And the same answer when the file cannot be parsed at all.
    write_policy(ws, [], raw='{"version": 1, "policies": [')
    connector_policy.invalidate_workspace_policy_cache()
    assert DENIED_READ in f.read_file(str(policy))
    assert DENIED_WRITE in f.apply_edit(str(policy), EDIT_ONE)


def test_a_corrupt_policy_file_refuses_writes_and_leaves_reads_open(ws: Path):
    """pol-00 D4, connector edition: the same asymmetry the pane uses.

    A file that did not load contributes no rules, so the user can still look at
    it (reads stay open) -- but every write is refused with the parse message,
    because the policy that was meant to protect a folder demonstrably did not
    run, and letting an agent edit inside it is the one answer that is not
    acceptable.
    """
    (ws / "src").mkdir()
    target = ws / "src" / "job.py"
    target.write_text("one\n", encoding="utf-8")
    write_policy(ws, [], raw="{ not json")
    connector_policy.invalidate_workspace_policy_cache()

    out = f.apply_edit(str(target), EDIT_ONE)
    assert DENIED_WRITE in out, out
    assert "did not load" in out
    assert str(ws / shared.POLICY_DIRNAME / shared.POLICY_FILENAME) in out
    assert "not valid JSON" in out
    assert target.read_text(encoding="utf-8") == "one\n"
    # The read half: the file has to stay reachable so it can be fixed.
    assert f.read_file(str(target)).startswith("File:")


def test_the_workspace_policy_says_nothing_outside_the_root(tmp_path, ws: Path):
    """The two floors' domains, pinned apart.

    A sibling tree of the root is not governed by the root's rules -- that file
    is written relative to *this* workspace -- and the connector deny-list answers
    there exactly as it always did. This is the test that keeps pol-08 from
    quietly becoming "the policy replaces the deny-list".
    """
    outside = ws.parent / "sibling-project"
    (outside / "secret").mkdir(parents=True)
    target = outside / "secret" / "notes.md"
    target.write_text("one\n", encoding="utf-8")
    write_policy(ws, [{"path": "secret", "mode": "hidden"}])
    connector_policy.invalidate_workspace_policy_cache()

    # The name is innocent, so neither floor speaks: an ordinary file.
    assert f.read_file(str(target)).startswith("File:")
    assert f.apply_edit(str(target), EDIT_ONE).startswith("Applied")
    # A name the deny-list knows is refused outside the root as before, and that
    # answer does not mention the workspace policy at all.
    secret = outside / ".env"
    secret.write_text("A=1\n", encoding="utf-8")
    out = f.read_file(str(secret))
    assert DENIED_READ in out
    assert "secrets/config file" in out and "workspace policy" not in out


def test_the_same_named_folder_inside_the_root_is_governed(ws: Path):
    """The mirror of the test above, so the split is not an accident of naming.

    One pattern, two paths that differ only by which side of ``--root-dir`` they
    sit on: in-root is refused by the policy, out-of-root is judged by the lists.
    """
    inner = ws / "sub" / "secret"
    inner.mkdir(parents=True)
    (inner / "a.md").write_text("one\n", encoding="utf-8")
    write_policy(ws, [{"path": "**/secret", "mode": "ro"}])
    assert DENIED_WRITE in f.apply_edit(str(inner / "a.md"), EDIT_ONE)


@pytest.mark.requires_symlink(
    "the workspace policy is decided on where a file really is, so a link "
    "cannot carry a path around a hidden folder"
)
def test_a_symlink_cannot_carry_a_path_around_the_workspace_policy(ws: Path):
    """The bypass is decided on where a file *really* is, and so is the policy.

    A link inside the root pointing at a hidden folder below it still lands on
    the hidden path, so the verdict is the same one the real spelling gets.
    """
    hidden = ws / "vault"
    hidden.mkdir()
    target = hidden / "notes.md"
    target.write_text("one\n", encoding="utf-8")
    link = ws / "notes.md"
    link.symlink_to(target)
    write_policy(ws, [{"path": "vault", "mode": "hidden"}])

    assert DENIED_READ in f.read_file(str(link))
    assert DENIED_WRITE in f.apply_edit(str(link), EDIT_ONE)


def test_an_ordinary_workspace_env_is_still_editable_here(ws: Path):
    """The documented asymmetry, pinned rather than hidden (task §1 and §8).

    Inside the root, ``.env`` is an ordinary file to a connector: the Editor
    pane's rule for it is **level A** (benchmarkit regenerates its own ``.env``
    from the settings template), which is a fact about the pane, not about an
    agent's edit. If level A is ever applied to the connectors, this is the test
    that changes -- deliberately, and with the docstrings in both connectors
    saying so.
    """
    env = ws / ".env"
    env.write_text("MODEL=one\n", encoding="utf-8")
    assert f.apply_edit(str(env), "<<<<<<< SEARCH\nMODEL=one\n=======\nMODEL=x\n>>>>>>> REPLACE"
                        ).startswith("Applied")
    assert "MODEL=x" in env.read_text(encoding="utf-8")
    # And the workspace may hide it if its owner wants that.
    write_policy(ws, [{"path": "**/.env", "mode": "hidden"}])
    connector_policy.invalidate_workspace_policy_cache()
    assert DENIED_READ in f.read_file(str(env))


def test_edit_policy_reports_the_workspace_rules_it_enforces(ws: Path):
    """A tool that reports a policy it does not enforce is worse than silence."""
    write_policy(ws, [
        {"path": ".git", "mode": "hidden"},
        {"path": "**/__pycache__", "mode": "hidden"},
        {"path": "**/node_modules", "mode": "ro"},
        {"path": "**/dist", "mode": "ro"},
    ])
    connector_policy.invalidate_workspace_policy_cache()
    report = f.edit_policy()

    assert "no policy restrictions" not in report
    assert "the workspace policy of that folder applies" in report
    assert str(ws / shared.POLICY_DIRNAME / shared.POLICY_FILENAME) in report
    assert "(4 rules)" in report
    assert "hidden: .git  **/__pycache__" in report
    assert "ro:     **/node_modules  **/dist" in report
    assert f"{shared.POLICY_DIRNAME}: reserved by benchmarkit" in report


def test_edit_policy_labels_a_truncated_rule_list(ws: Path):
    """The list is capped so the caps stay readable -- and the cap is admitted."""
    rules = [{"path": f"dir{i}", "mode": "ro"} for i in range(9)]
    for rule in rules:
        (ws / rule["path"]).mkdir()
    write_policy(ws, rules)
    connector_policy.invalidate_workspace_policy_cache()

    report = f.edit_policy()
    assert "(9 rules)" in report
    assert "+4 more" in report
    assert "dir5" not in report


def test_edit_policy_names_a_policy_file_it_could_not_load(ws: Path):
    write_policy(ws, [], raw="nope")
    connector_policy.invalidate_workspace_policy_cache()
    report = f.edit_policy()
    assert "(did not load)" in report
    assert "writes inside the root are refused" in report


def test_the_verdict_and_the_pane_agree_word_for_word(ws: Path):
    """The acceptance condition, stated as a test.

    The Editor pane refuses a save with ``refusal_message`` over the same
    ``Rule`` list; the connector refuses with the same function over the same
    file. Neither paraphrases, so an agent's error and the banner the user sees
    describe one rule the same way.
    """
    (ws / "vendor").mkdir()
    write_policy(ws, [{"path": "**/vendor", "mode": "ro", "reason": "vendored"}])
    connector_policy.invalidate_workspace_policy_cache()

    rules, _ = connector_policy.load_workspace_policy(str(ws))
    expected = shared.refusal_message(
        shared.Verdict(shared.READ_ONLY, shared.LEVEL_WORKSPACE, "**/vendor",
                       "vendored"),
        for_write=True,
    )
    out = f.apply_edit(str(ws / "vendor" / "x.md"), EDIT)
    assert expected in out, out
    assert "vendored" in out
    # The pane's own sentence, from the app side, is the same string.
    from src.core.editorfs import policy_modes

    assert policy_modes.refusal_message(
        shared.Verdict(shared.READ_ONLY, shared.LEVEL_WORKSPACE, "**/vendor",
                       "vendored"),
        for_write=True,
    ) == expected


def test_the_connector_cache_is_keyed_on_the_root_not_just_the_file(ws: Path):
    """pol-04's rule about the cache key, restated for the connector process.

    Two workspaces with byte-identical policy files (same ``mtime_ns``, same
    size) are one stamp and two roots. A cache keyed on the stamp alone would
    answer the second workspace with the first one's rules -- and name the wrong
    file while doing it.
    """
    broken = '{"version": 1, "policies": ['
    other = ws.parent / "pol08-second-root"
    for root in (ws, other):
        folder = root / shared.POLICY_DIRNAME
        folder.mkdir(parents=True, exist_ok=True)
        (folder / shared.POLICY_FILENAME).write_text(broken, encoding="utf-8")
    def policy_of(root: Path) -> Path:
        return root / shared.POLICY_DIRNAME / shared.POLICY_FILENAME

    first = policy_of(ws).stat()
    os.utime(policy_of(other), ns=(first.st_atime_ns, first.st_mtime_ns))
    connector_policy.invalidate_workspace_policy_cache()

    # Read the first root (caching its stamp), then the second -- same stamp,
    # different file. Each answer has to name the file of the root asked about.
    for root in (ws, other):
        reason = connector_policy.workspace_policy_reason(
            "src/a.py", root=str(root), for_write=True
        )
        assert reason is not None
        assert str(policy_of(root)) in reason, (root, reason)
        assert str(policy_of(ws if root == other else other)) not in reason, root

