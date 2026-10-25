#!/usr/bin/env python3
"""
Console MCP Server - Allows LLMs to execute console commands via MCP protocol.

This server exposes tools that enable AI models to run shell commands on the local system.
It implements the Model Context Protocol (MCP) and serves at http://localhost:9999/mcp

Reading and editing files is the job of the fedit-mcp connector (line-numbered
reads, match-verified edits, backups and revert). Enable both together
(MCP_ENABLED=console_mcp,fedit_mcp): this server for shell commands, fedit-mcp
for any file content work.

Root scoping: the server MUST be started with --root-dir. The console starts
in that directory and the security policy is path-scoped: commands whose path
arguments all stay inside the root run WITHOUT any policy restrictions; the
block/allow lists only apply to commands that reference paths outside the
root. Privilege/power/disk/service commands
(sudo, reboot, mkfs, systemctl, ...) are never path work and stay blocked
everywhere.

There is no shell: the command is parsed into arguments and exec'd directly, so
what was validated is exactly what runs (no pipes, chains or substitution).

Credential stores (``~/.ssh``, ``~/.aws``, ``.git``, ``.env``, private keys and
friends) are refused as command arguments whatever the command is, using the
same lists as the fedit connector -- see ``mcp_connectors/path_policy.py``. A
secret read by ``cat`` does as much damage as one read by ``read_file``, so the
two connectors must not drift apart.

Usage:
    python -m mcp_connectors.console_mcp --root-dir /path/to/root              # default port 9999
    python -m mcp_connectors.console_mcp --root-dir /path --port 8080          # custom port
    python -m mcp_connectors.console_mcp --root-dir /path --transport stdio    # stdio (for MCP hubs)
"""

import argparse
import os
import re
import subprocess
import shlex
import sys
from pathlib import Path
from typing import Optional
from mcp.server.mcpserver import MCPServer

try:  # imported as part of the package (tests, `python -m mcp_connectors...`)
    from . import path_policy
except ImportError:  # run as a plain script path, as externals.json does
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import path_policy  # type: ignore[no-redef]

# Root directory the console is scoped to; set by main() from --root-dir.
# When None (e.g. module imported without main()), the full policy applies to
# every command (legacy behaviour).
ROOT_DIR: Optional[Path] = None

# Create MCP server instance
# streamable_http_path="/mcp" is the default, which matches your requirement
server = MCPServer(
    name="console-mcp",
    instructions="A server that allows LLMs to execute console commands, scoped to a root "
                 "directory chosen at startup (--root-dir). Commands that only touch paths "
                 "inside that root run without policy restrictions; commands referencing "
                 "paths outside the root must pass the allow/block policy. "
                 "There is NO shell: each call runs one command, split into arguments and "
                 "executed directly, so pipes, ';'/'&&' chains, redirections, backgrounding "
                 "and $( ) / $VAR expansion are refused rather than re-interpreted -- issue "
                 "one command per call and read its output. "
                 "For reading or editing file contents, use the fedit-mcp connector instead "
                 "(read_file / preview_edit / apply_edit / revert_edit).",
)

# ---------------------------------------------------------------------------
# Security policy.
#
# The policy is PATH-SCOPED, deliberately: the root is the operator's own
# workspace, so a command whose path arguments all resolve inside it runs
# without a list check. That makes ALLOWED_COMMANDS NOT a whitelist of what
# may run -- it is the list of commands additionally tolerated when the
# command reaches *outside* the root. BLOCKED_COMMANDS / ALWAYS_BLOCKED_COMMANDS
# are the part with real consequence, and they apply to every statement.
# Keep the two lists disjoint -- entries in both are effectively blocked and
# only create confusion.
# ---------------------------------------------------------------------------

# Commands that are always blocked (checked first, security)
BLOCKED_COMMANDS = {
    # destructive file operations
    "rm", "rmdir", "shred", "truncate",
    # permissions / ownership
    "chmod", "chown",
    # disk / system destruction
    "mkfs", "dd", "fdisk", "parted", "mount", "umount",
    # power state
    "shutdown", "reboot", "poweroff", "halt",
    # privilege escalation
    "sudo", "su", "doas", "runuser",
    # package / service control
    "pacman", "systemctl", "crontab",
}

# Privilege / power / disk / service commands are NOT path-scoped file work:
# they stay blocked even when a command would otherwise run inside the root.
ALWAYS_BLOCKED_COMMANDS = {
    "mkfs", "dd", "fdisk", "parted", "mount", "umount",
    "shutdown", "reboot", "poweroff", "halt",
    "sudo", "su", "doas", "runuser",
    "pacman", "systemctl", "crontab",
}

# Shell builtins are not programs to exec, and there is no shell here: a token
# like `cd` would pass the path bypass, then die as "Command not found" -- a
# refusal that blames a tokenisation accident for what is really a missing
# shell. Naming them makes the real reason the answer.
SHELL_BUILTIN_COMMANDS: frozenset[str] = frozenset({
    "cd", "export", "unset", "source", ".", "alias", "unalias", "set",
    "shift", "trap", "wait", "jobs", "bg", "fg", "history", "fc", "exec",
})

BUILTIN_HINT: str = (
    "is a shell builtin and this connector runs no shell: choose the working "
    "directory with the cwd argument, and read configuration through the "
    "fedit connector instead of `export`/`source`"
)

# Define allowed commands (security whitelist)
ALLOWED_COMMANDS = {
    # --- basic file & text (read-only) ---
    "ls", "dir", "cat", "echo", "pwd", "whoami", "date", "time",
    "head", "tail", "wc", "sort", "uniq", "cut",
    "grep", "sed", "awk", "tr", "tac", "nl", "column", "comm", "paste",
    "less", "more", "man", "which", "file", "stat",

    # --- paths & paths metadata ---
    "find", "basename", "dirname", "realpath", "readlink",
    "diff", "cmp", "seq", "printf", "bc", "numfmt",

    # --- benign file creation (destructive ops stay blocked) ---
    "mkdir", "touch", "cp", "mv", "ln", "mktemp",

    # --- checksums & archives ---
    "md5sum", "sha1sum", "sha256sum",
    "tar", "gzip", "unzip", "zip",

    # --- system visibility (read-only) ---
    "ps", "top", "vmstat", "df", "du", "free", "uptime", "hostname",
    "uname", "lscpu", "lsb_release", "nproc", "id", "groups", "who", "w",

    # --- network ---
    "curl", "wget", "ping", "nslookup", "dig", "ss", "ip", "tracepath",

    # --- environment ---
    "env", "printenv", "sleep",

    # --- dev tooling ---
    "python3", "python", "node", "npm", "pip", "git",
    "jq", "sqlite3", "openssl",

    # --- dev tooling an agent reaches for by name (C-2) ---
    # These have to be listed because a command naming *no* path never takes
    # the in-root bypass, so a bare `pytest -q` used to be refused for the
    # tokenisation accident rather than for any danger.
    "pytest", "rg", "ag", "fd", "make", "npx", "lsof", "tree",

    # --- editors (kept from your original list) ---
    "vim", "nano",
}


# ---------------------------------------------------------------------------
# Shell structure (C-1, C-3, C-4, C-5)
#
# What the shell treats as an *operator* is only visible after parsing. The old
# check was a substring test on the raw command string, so it refused
# `grep "<div" file` and `curl "https://host/?a=1&b=2"` -- a quoted `|`/`&` is
# not an operator at all -- while it never saw the separators that *are*
# operators: a newline started a second statement that no check ever looked at.
#
# So the command is parsed once into an explicit structure: statements split on
# `;`, `&&`, `||`, `|` and newlines, each statement an argv plus any redirections
# and any shell substitution. Every policy decision below reads that structure,
# and the command that runs is exactly the argv that was validated -- the shell
# is not asked to interpret it a second time.
# ---------------------------------------------------------------------------

# Separators that end a statement. A newline is a statement separator for
# /bin/sh exactly like `;` is, so it is listed here rather than left as
# "whitespace" that shlex swallows (C-3). `&` is deliberately NOT one of them:
# backgrounding has no meaning without a shell, so it is recorded as a refused
# shell feature instead of splitting anything.
SHELL_OPERATOR_TOKENS: frozenset[str] = frozenset({";", "&&", "||", "|", "\n"})

# Characters that make up a separator run. shlex groups a stretch of them into
# a single token (`ls\n\n&& pwd` yields '\n\n&&'), so a run is taken apart
# before it is dispatched.
_SEPARATOR_CHARS: frozenset[str] = frozenset(";&|\n")
_TWO_CHAR_OPERATORS: tuple[str, ...] = ("&&", "||")


def operator_run(token: str) -> Optional[list[str]]:
    """Split a separator token into its operators, or None if it is not one.

    ``'\n\n&&'`` becomes ``['\n', '\n', '&&']``. A token holding anything else
    (a redirection like ``>&``, or a plain argument) is not a run.
    """
    if not token or any(char not in _SEPARATOR_CHARS for char in token):
        return None
    parts: list[str] = []
    position = 0
    while position < len(token):
        pair = token[position: position + 2]
        if pair in _TWO_CHAR_OPERATORS:
            parts.append(pair)
            position += 2
        else:
            parts.append(token[position])
            position += 1
    return parts

# The metacharacters the parser keeps as operators; the newline is added so it
# is tokenised instead of skipped. Quotes stay attached to their token (non-POSIX
# mode), which is what lets a caller tell `&&` from `'&&'`.
_PUNCTUATION_CHARS: str = ";|&<>()\n"

# Shell features this connector refuses outright wherever they appear, because
# they make the string mean something the checker never saw: `$(` … `)`, `` ` ``,
# `${`, `(( ))`, brace expansion and subshells are shell *syntax*, and the shell
# is no longer in the loop, so nothing here can honour them.
UNSUPPORTED_SHELL_CHARS: frozenset[str] = frozenset("$`{}()")


class ShellSyntaxError(ValueError):
    """The command string cannot be parsed into shell statements."""


class ShellStatement:
    """One statement of a command: argv plus the redirections that decorate it.

    ``argv`` holds the already-unquoted words (the program and its arguments),
    so it can be executed as-is -- nothing is re-parsed by a shell later.
    ``redirections`` are the redirect operators that were seen (only the
    ``2>&1`` merge is accepted, and it lands in ``merges`` instead), and
    ``shell_features`` lists the metacharacters found outside quoting.
    """

    # ``separator`` is the operator that introduced this statement, empty for
    # the first one -- enough to name the culprit in a refusal.
    def __init__(self, separator: str = "") -> None:
        self.separator: str = separator
        self.argv: list[str] = []
        self.redirections: list[str] = []
        self.merges: list[str] = []
        self.shell_features: list[str] = []


def unquoted_part(token: str) -> str:
    """The characters of ``token`` that are *not* inside a quoted span.

    This is what tells an operator from a letter that merely looks like one:
    the `&` of ``"https://host/?a=1&b=2"`` is inside quotes, the `&` of ``&&``
    is not. Quoting is a property of the token, so it is read off the token.
    """
    chars: list[str] = []
    quote = ""
    for char in token:
        if quote:
            if char == quote:
                quote = ""
            continue
        if char in "\"'":
            quote = char
            continue
        chars.append(char)
    return "".join(chars)


def shell_tokens(command: str) -> list[str]:
    """Tokenise ``command`` with operators kept as separate tokens.

    Raises:
        ShellSyntaxError: the string cannot be tokenised (unbalanced quotes).
    """
    lexer = shlex.shlex(command, punctuation_chars=_PUNCTUATION_CHARS)
    lexer.whitespace_split = True
    # shlex counts a newline as whitespace by default and swallows it, which is
    # exactly how a second statement used to disappear from the verdict. Here it
    # is whitespace only inside quoting; outside it, it is the operator token.
    lexer.whitespace = " \t\r"
    # A `#` in an argument (`grep a#b f`, `make -j#`) is an argument here, not a
    # comment: the command is never handed to a shell, so no comment syntax.
    lexer.commenters = ""
    try:
        return list(lexer)
    except ValueError as exc:
        raise ShellSyntaxError(str(exc)) from exc


def unquote(token: str) -> str:
    """Strip the quoting a token carries, so ``argv`` holds the word to exec.

    A single open quote is closed as a double quote rather than treated as
    unbalanced, so an odd number of quotes degrades the way a shell would
    degrade rather than losing the whole argument.
    """
    if len(token) > 1 and token.startswith("'") and not token.endswith("'"):
        token = '"' + token[1:]
    try:
        words = shlex.split(token)
    except ValueError:
        return token.strip("\"'")
    return words[0] if words else ""


# The stderr-merge spellings (``2>&1``, ``1>&2``, ``>&2``). Merging is not an
# extra capability -- the tool already returns stderr next to stdout -- so the
# most ordinary redirect there is is accepted instead of being refused for a
# character that does no harm (C-1). ``shlex`` keeps the fd digits apart, so a
# merge is recognised as a *sequence* (``['2', '>&', '1']``).
STDERR_MERGE_TOKENS: frozenset[str] = frozenset({">&", "&>"})
STDERR_MERGE_FORM: str = "2>&1"


def _is_fd_token(token: str) -> bool:
    """True for the file-descriptor number a redirect target is written as."""
    return token.isdigit()


def is_supported_stderr_merge(tokens: list[str], idx: int) -> bool:
    """True when ``tokens[idx]`` merges one stream into another (``2>&1``).

    Only the fd-to-fd form qualifies: the operator must be preceded by the
    source descriptor and followed by the destination one, which is also what
    makes a lone ``&>``/``>&`` an incomplete (and therefore refused) redirect.
    """
    if tokens[idx] not in STDERR_MERGE_TOKENS:
        return False
    if idx + 1 >= len(tokens) or not _is_fd_token(tokens[idx + 1]):
        return False
    return idx > 0 and _is_fd_token(tokens[idx - 1])


def is_redirection_token(token: str) -> bool:
    """True for a token that is only ever a redirection operator."""
    stripped = token.strip("0123456789")
    return bool(stripped) and all(char in "<>|&" for char in stripped)


def parse_shell_statements(command: str) -> list[ShellStatement]:
    """Split ``command`` into statements: argv, redirections, shell features.

    Quoting decides what counts: `grep "<div" file` is one argv with no
    redirection and `curl "…?a=1&b=2"` has no operator, while `ls web; id` and
    `ls web` + newline + `id` are both **two** statements -- every one of which
    then goes through the policy on its own (C-3, C-5).

    Raises:
        ShellSyntaxError: the command cannot be tokenised, or holds no
            statement at all.
    """
    tokens = shell_tokens(command)
    statements: list[ShellStatement] = []
    current = ShellStatement()

    def close_statement(separator: str = "") -> None:
        nonlocal current
        if current.argv or current.redirections or current.shell_features or current.merges:
            statements.append(current)
        current = ShellStatement(separator)

    index = 0
    while index < len(tokens):
        token = tokens[index]
        run = operator_run(token)
        if run is not None:
            # `ls web\n\n&& pwd` tokenises the separator stretch as ONE token
            # ('\n\n&&'), so a run is taken apart before it is dispatched --
            # otherwise a run would be handed over as an argument name.
            for part in run:
                if part == "&":
                    # Backgrounding has nothing to mean for this tool: the
                    # process table is not the agent's to pollute.
                    current.shell_features.append("&")
                    continue
                if part in SHELL_OPERATOR_TOKENS:
                    close_statement(part)
                    continue
                current.redirections.append(part)
            index += 1
            continue
        if is_redirection_token(token):
            if is_supported_stderr_merge(tokens, index):
                if current.argv and current.argv[-1].isdigit():
                    current.argv.pop()
                current.merges.append(STDERR_MERGE_FORM)
                index += 2
                continue
            current.redirections.append(token)
            index += 1
            continue
        current.shell_features.extend(
            char for char in unquoted_part(token) if char in UNSUPPORTED_SHELL_CHARS
        )
        current.argv.append(unquote(token))
        index += 1
    close_statement()
    if not statements:
        raise ShellSyntaxError("no statement to run")
    return statements


def resolve_as_path(token: str) -> Path:
    """Resolve a command token as a path; relative paths anchor at ROOT_DIR."""
    expanded = os.path.expanduser(token)
    path = Path(expanded)
    if not path.is_absolute():
        path = (ROOT_DIR or Path.cwd()) / path
    try:
        return path.resolve()
    except OSError:
        return Path(os.path.normpath(str(path)))


def absolute_as_written(token: str) -> Path:
    """The absolute path a token names, as written (nothing collapsed).

    ``os.path.abspath`` would fold ``~/.ssh/../../etc`` into an ordinary
    ``/home`` path and hide the store it walks out of, so the components are
    kept as given; the symlink-resolved target is checked separately.
    """
    path = Path(os.path.expanduser(token))
    if not path.is_absolute():
        path = (ROOT_DIR or Path.cwd()) / path
    return path


def is_within_root(token: str) -> bool:
    """True when the token resolves to a path inside (or equal to) ROOT_DIR."""
    if ROOT_DIR is None:
        return False
    resolved = resolve_as_path(token)
    return resolved == ROOT_DIR or ROOT_DIR in resolved.parents


def path_argument_tokens(parts: list[str]) -> list[str]:
    """Tokens of a parsed command that could name a filesystem path."""
    tokens: list[str] = []
    for token in parts[1:]:
        if token.startswith("-"):
            _, sep, value = token.partition("=")
            if sep and value.startswith(("/", "~", "./", "../")):
                tokens.append(value)
        elif looks_like_path(token):
            tokens.append(token)
    return tokens


def looks_like_path(token: str) -> bool:
    """False for tokens that cannot name a path in context (C-2).

    A bare number (``kill -9 1``, ``chmod 644``) used to count as an in-root
    "path", firing the root bypass for commands that touch no file at all --
    ``path_argument_tokens`` should mean what its name says. ``.`` and ``..``
    stay paths: they resolve outside the root, which is exactly where the
    policy has to apply. (Shell operators never reach here at all; they are
    split off by ``parse_shell_statements``.)
    """
    if not token:
        return False
    if token.isdigit():
        return False
    # Belt and braces: ``parse_shell_statements`` already splits operators off,
    # so an operator can only reach here if a caller hands raw tokens in.
    if operator_run(token) is not None or is_redirection_token(token):
        return False
    return True


def _trusted_prefix_hit(path: Path) -> Optional[str]:
    """The scratch prefix (``/tmp`` …) ``path`` stays inside, or None."""
    return path_policy.prefix_hit(path.as_posix(), path_policy.ALLOW_PREFIXES)


def protected_path_error(token: str) -> Optional[str]:
    """Refuse a path token that names a credential store (C-6).

    fedit refused ``~/.ssh``, ``~/.git-credentials`` and ``.env`` through any
    path, while this connector ran ``cat .env`` -- so the careful policy of one
    connector was bypassable through the sibling one enabled in the same
    session. Both now read the same lists from ``path_policy``, and this check
    runs whatever the root bypass decided, because being allow-listed was never
    a limit on *which* file a command may touch.

    Two carve-outs, matching fedit rather than inventing a third policy:

    * a path that really is inside the root: the operator chose that tree, and
      the root bypass is documented for both connectors;
    * a path inside a scratch prefix (``/tmp``, ``/var/tmp``).

    A token is judged both as written and as its symlink-resolved target, so
    ``~/.ssh/../../etc`` cannot read as an ordinary home path.
    """
    as_written = absolute_as_written(token)
    real = resolve_as_path(token)

    inside_root = is_within_root(token)
    if inside_root and (real == as_written or _within_root_or_none(real)):
        return None
    if _trusted_prefix_hit(as_written) and _trusted_prefix_hit(real):
        return None

    for label, candidate in (("the path", as_written), ("its real target", real)):
        reason = path_policy.deny_reason(candidate, for_write=False)
        if reason:
            detail = "" if candidate == as_written else f" ({label} {real})"
            return (
                f"{as_written}{detail} is off-limits: it is {reason}. "
                "Credential stores and secret files are refused by every "
                "connector in this project, whatever the command is."
            )
    return None


def _within_root_or_none(path: Path) -> bool:
    """True when ``path`` is inside the root; False when the server is unrooted."""
    if ROOT_DIR is None:
        return False
    return path == ROOT_DIR or ROOT_DIR in path.parents


# ``errors="backslashreplace"`` turns an undecodable byte into a visible `\xNN`
# escape; counting those is how the output says out loud that it was lossy.
ESCAPED_BYTE_RE = re.compile(r"\\x[0-9a-f]{2}")

# How long a killed command is given to flush what it already wrote to the pipe
# buffers after a timeout (C-8). It is already dead; this only drains the pipe.
TIMEOUT_DRAIN_SECONDS: float = 2.0


def decode_stream(raw: Optional[bytes]) -> str:
    """Decode captured output without ever raising (C-7).

    ``errors="backslashreplace"`` keeps the bytes that decoded and shows the
    rest as \\xNN escapes, which is what ``decode_note`` then counts.

    A stream the child never wrote to can reach here as ``None`` (a killed
    process, a pipe closed early), so it is treated as the empty stream.
    """
    return (raw or b"").decode("utf-8", errors="backslashreplace")


# Output cap per stream (C-9): one careless `cat` of a bundle used to land in
# the conversation whole, ending it. Kept as head + tail, because the head is
# where a listing starts making sense and the tail is where an error lives.
MAX_OUTPUT_LINES: int = int(os.environ.get("BENCHMARKIT_MAX_OUTPUT_LINES", "200"))
# A line count alone does not help a single multi-megabyte line (a minified
# bundle has exactly one), so there is a character ceiling as well.
MAX_OUTPUT_CHARS: int = int(os.environ.get("BENCHMARKIT_MAX_OUTPUT_CHARS", "20000"))


def cap_output(text: str, limit: int = MAX_OUTPUT_LINES, char_limit: int = MAX_OUTPUT_CHARS) -> str:
    """Return at most ``limit`` lines / ``char_limit`` chars, head + tail.

    stdout and stderr are capped separately so a chatty build log cannot
    squeeze a stack trace out of the answer.
    """
    lines = text.splitlines(keepends=True)
    if len(lines) > limit:
        head = max(1, limit // 2)
        tail = max(1, limit - head)
        omitted = len(lines) - head - tail
        marker = f"\n[... {omitted} line(s) omitted of {len(lines)} total ...]\n"
        text = "".join(lines[:head]) + marker + "".join(lines[-tail:])
    if len(text) > char_limit:
        head = char_limit // 2
        tail = char_limit - head
        omitted = len(text) - char_limit
        marker = f"\n[... {omitted} character(s) omitted of {len(text)} total ...]\n"
        text = text[:head] + marker + text[-tail:]
    if not text.endswith("\n"):
        text += "\n"
    return text


def decode_note(text: str) -> str:
    """Add one flag line when the output held bytes that could not be decoded.

    The run itself never fails on decoding (C-7): losing the whole result --
    exit code included -- because one byte of a PNG went through a strict
    decoder was the failure being fixed. What the caller gets instead is the
    text with the undecodable bytes escaped, plus a note naming them, so the
    answer is honest about being lossy and points at a route that works. The
    count covers the escapes that are actually shown, after the output cap.
    """
    escapes = ESCAPED_BYTE_RE.findall(text)
    if not escapes:
        return text
    sample = ", ".join(escapes[:3])
    return (
        text
        + f"\n[{len(escapes)} undecodable byte(s) shown escaped ({sample}); "
        "for a real binary read run the encoder itself: base64 <file>]"
    )


def statement_position(index: int, total: int) -> str:
    """Human hint naming which statement a refusal came from (C-5)."""
    if total <= 1:
        return ""
    return f" (in statement {index} of {total})"


def describe_statement(statement: ShellStatement) -> str:
    """Short, quoting-safe rendering of a statement for an error message."""
    return shlex.join(statement.argv)


def statement_policy_error(statement: ShellStatement) -> Optional[str]:
    """Run the list/path policy on ONE statement, or None when it passes.

    The verdict is per statement, not per command string: the old code looked
    at ``parts[0]`` of the whole string, so a `..` anywhere retro-actively
    changed the verdict of every other part, and no command past the first
    separator was ever examined (C-5).
    """
    base_command = statement.argv[0]

    # Privilege / power / disk / service commands are blocked everywhere.
    if base_command in ALWAYS_BLOCKED_COMMANDS:
        return f"Command '{base_command}' is blocked for security reasons"

    # Builtins would otherwise reach exec() and come back as "Command not
    # found", which reads like a policy accident rather than what it is.
    if base_command in SHELL_BUILTIN_COMMANDS:
        return f"Command '{base_command}' {BUILTIN_HINT}"

    path_tokens = path_argument_tokens(statement.argv)

    # Credential stores are off-limits whatever the rest of the policy decided
    # (C-6): the lists below answer "may this command run", never "is this
    # target allowed", so an allow-listed `cat` used to reach ~/.ssh.
    for token in path_tokens:
        protected = protected_path_error(token)
        if protected:
            return protected

    # A statement whose path arguments all stay inside the root is ordinary
    # workspace work and is not list-checked.
    if ROOT_DIR is not None and path_tokens and all(is_within_root(t) for t in path_tokens):
        return None

    if base_command in BLOCKED_COMMANDS:
        return f"Command '{base_command}' is blocked for security reasons"
    if base_command not in ALLOWED_COMMANDS:
        return f"Command '{base_command}' is not in the allowed commands list"
    return None


def shell_feature_error(statements: list[ShellStatement]) -> Optional[str]:
    """Refuse what this connector cannot execute, naming the offending part.

    The command is executed as the parsed argv, without a shell (C-3/C-4), so
    a shell separator, a redirection or a substitution cannot be honoured: a
    newline would be one argument, `$( )` would be literal text. Refusing them
    keeps ``check_command_allowed`` an honest oracle -- whatever it allows is
    exactly what ``execute_command`` runs.
    """
    total = len(statements)
    for index, statement in enumerate(statements, start=1):
        if statement.shell_features:
            feature = statement.shell_features[0]
            return (
                f"Command contains the shell feature {feature!r}, which is not "
                f"executed here (no shell is used, so $VAR is not expanded): "
                f"{describe_statement(statement)}"
                + statement_position(index, total)
            )
        if statement.redirections:
            redirect = statement.redirections[0]
            return (
                f"Command redirects with {redirect!r}; output files are not "
                "written here -- read the output instead, or use 2>&1 to merge "
                "stderr into stdout" + statement_position(index, total)
            )
    if total > 1:
        second = statements[1]
        return (
            f"Command chains statements with {second.separator!r}; this connector "
            f"runs one command per call, so run them as two separate calls "
            f"(second statement: {describe_statement(second)})"
        )
    return None


def validate_command(command: str) -> tuple[bool, str]:
    """
    Validate if a command is safe to execute.

    The command string is parsed once into statements (``;``, ``&&``, ``||``,
    ``|``, ``&`` and a newline separate them), and the policy runs on every
    statement: its base command against the block lists, and its path arguments
    against the root. Policy is path-scoped -- a statement whose path arguments
    all stay inside ROOT_DIR is not list-checked (the root is the operator's
    workspace), anything reaching outside the root must pass the block list and
    the allow list, which is NOT a whitelist of what may run but what is
    additionally tolerated outside the root.

    Anything that needs the shell to have a meaning -- separators,
    redirections, substitution -- is refused, because what runs is exactly the
    argv that was validated.

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not command or not command.strip():
        return False, "Command cannot be empty"

    try:
        statements = parse_shell_statements(command)
    except ShellSyntaxError as exc:
        return False, f"Invalid command format: {exc}"

    total = len(statements)

    # A statement with no command word (`2>&1` on its own) has nothing to run;
    # refusing it beats indexing an empty argv.
    for index, statement in enumerate(statements, start=1):
        if not statement.argv:
            return False, (
                "Command has no program to run" + statement_position(index, total)
            )

    # Blocked commands first, on every statement (C-2): the security part of
    # the policy is reported even when the command will be refused anyway.
    for index, statement in enumerate(statements, start=1):
        if not statement.argv:
            continue
        base_command = statement.argv[0]
        if base_command in ALWAYS_BLOCKED_COMMANDS:
            return False, (
                f"Command '{base_command}' is blocked for security reasons"
                + statement_position(index, total)
            )

    # Then the structure the connector cannot honour.
    structure_error = shell_feature_error(statements)
    if structure_error:
        return False, structure_error

    for index, statement in enumerate(statements, start=1):
        error = statement_policy_error(statement)
        if error:
            return False, error + statement_position(index, total)

    return True, ""


@server.tool()
def execute_command(
    command: str,
    timeout: int = 30,
    cwd: Optional[str] = None
) -> str:
    """
    Execute a console command and return the output.

    Args:
        command: One command to execute (e.g., "ls -la", "cat /etc/hostname").
             There is no shell: the command is split into arguments and exec'd
             directly, so pipes, redirections, `;`/`&&` chains, backgrounding
             and `$( )`/`$VAR` expansion do NOT work -- write one plain command
             per call and read its output. Shell builtins (`cd`, `export`,
             aliases) do not exist here either: pick the directory with the
             `cwd` argument. stderr is reported separately, so a `2>&1` is
             accepted but needs no writing. Quoting is honoured.
        timeout: Maximum execution time in seconds (default: 30)
        cwd: Working directory for command execution (optional, must stay
             inside the server's --root-dir; defaults to the root)

    Returns:
        Command output as a string, or error message if execution fails.
        Each stream is capped (first and last lines, with an explicit
        "[... N line(s) omitted ...]" marker), so ask for the slice you want
        instead: `head -50 file`, `tail -30 file` or a tool with its own
        limit (`wc -l`, `grep -m 20`).
        Output that is not valid UTF-8 comes back with the offending bytes shown
        as \\xNN escapes plus a note counting them -- one stray byte no longer
        costs the whole result. To read a binary artifact properly, encode it:
        execute_command("base64 file.png").

    Example:
        execute_command("ls -la sub/dir", timeout=10)
    """
    if ROOT_DIR is None:
        return "Error: server is not rooted; start it with --root-dir"

    work_dir = str(ROOT_DIR)
    if cwd is not None:
        candidate = resolve_as_path(cwd)
        if candidate != ROOT_DIR and ROOT_DIR not in candidate.parents:
            return f"Error: cwd '{cwd}' is outside the server root directory"
        work_dir = str(candidate)

    # Validate the command
    is_valid, error_msg = validate_command(command)
    if not is_valid:
        return f"Error: {error_msg}"

    # What runs is the argv that was just validated, not the string handed to a
    # shell: `shell=True` re-interpreted the same text with a richer language
    # than the checker understood, so a newline started a second statement no
    # policy ever looked at and `$( )` ran inside the in-root bypass (C-3, C-4).
    # Validation above already refused every multi-statement command, so there
    # is exactly one statement here.
    try:
        statements = parse_shell_statements(command)
    except ShellSyntaxError as exc:
        return f"Error: Invalid command format: {exc}"
    argv = statements[0].argv


    # The pipes are read in binary and decoded by hand (C-7): the default
    # strict UTF-8 decode raised on a single stray byte and the bare
    # `except Exception` below threw the whole result away, exit code included.
    # Decoding never fails a run now; undecodable bytes come back escaped.
    #
    # Popen is used directly rather than subprocess.run so a timeout can still
    # hand back what the command had already printed (C-8): a build that hangs
    # after a wall of progress used to return nothing at all, which invited a
    # rerun of the same expensive command.
    try:
        process = subprocess.Popen(
            argv,
            shell=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=work_dir,
        )
    except FileNotFoundError:
        return "Error: Command not found"
    except PermissionError:
        return "Error: Permission denied"

    timed_out = False
    try:
        raw_out, raw_err = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        timed_out = True
        try:
            raw_out, raw_err = process.communicate(timeout=TIMEOUT_DRAIN_SECONDS)
        except (subprocess.TimeoutExpired, OSError):
            raw_out, raw_err = b"", b""
    except Exception as exc:  # pragma: no cover - unexpected transport failure
        process.kill()
        return f"Error executing command: {str(exc)}"

    stdout_text = decode_stream(raw_out)
    stderr_text = decode_stream(raw_err)

    # Build response
    output_parts = []

    # Capped per stream (C-9), so a chatty stdout cannot squeeze a stderr stack
    # trace out of the answer.
    if stdout_text:
        output_parts.append(f"STDOUT:\n{decode_note(cap_output(stdout_text))}")

    if stderr_text:
        output_parts.append(f"STDERR:\n{decode_note(cap_output(stderr_text))}")

    if timed_out:
        output_parts.append(
            f"\nCommand timed out after {timeout} seconds and was killed."
            + (" The output above is what it managed to print first."
               if (stdout_text or stderr_text) else "")
        )
    elif process.returncode != 0:
        output_parts.append(f"\nExit Code: {process.returncode}")

    if not output_parts:
        output_parts.append("Command executed successfully (no output)")

    return "\n".join(output_parts)


@server.tool()
def list_allowed_commands() -> str:
    """
    List all commands that are allowed to be executed by this server.

    Returns:
        Formatted list of allowed commands
    """
    sorted_commands = sorted(ALLOWED_COMMANDS)
    # Say what the list actually decides (C-2): inside the root no list is
    # consulted at all, so an agent hunting here for the "allowed spelling" of
    # a refused in-root command is wasting its turns.
    return (
        f"Commands accepted for work OUTSIDE the root ({len(sorted_commands)} total).\n"
        f"  This list is only consulted when the command references a path outside\n"
        f"  the server root ({ROOT_DIR}); inside the root, commands run without a\n"
        f"  list check, so a refusal there has a different reason (a blocked\n"
        f"  command, or a shell operator). Privilege/power commands stay blocked\n"
        f"  everywhere.\n\n"
        + "\n".join(f"  - {cmd}" for cmd in sorted_commands)
    )


@server.tool()
def check_command_allowed(command: str) -> str:
    """
    Check if a specific command is allowed to be executed.

    Args:
        command: The command to check

    Returns:
        Whether the command is allowed and any restrictions. The verdict is the
        one execute_command applies (same parse, same policy), so it can be used
        as an oracle without running anything. Note this tool takes no timeout:
        when the command does run, execute_command's default of 30 seconds
        applies, and a timeout returns whatever the command had already printed.
    """
    is_valid, error_msg = validate_command(command)
    if is_valid:
        return f"Command '{command}' is ALLOWED for execution"
    else:
        return f"Command '{command}' is NOT ALLOWED: {error_msg}"


def main():
    parser = argparse.ArgumentParser(
        description="Console MCP Server - Execute console commands via MCP protocol"
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=9999,
        help="Port to listen on (default: 9999)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--root-dir",
        type=str,
        required=True,
        help="Root directory the console starts in and is scoped to (required). "
             "Commands touching only paths inside it bypass the security policy."
    )
    parser.add_argument(
        "--transport", "-t",
        type=str,
        choices=("stdio", "streamable-http"),
        default="streamable-http",
        help="Transport to serve on (default: streamable-http)"
    )
    parser.add_argument(
        "--debug", "-d",
        action="store_true",
        help="Enable debug mode"
    )

    args = parser.parse_args()

    global ROOT_DIR
    root = Path(os.path.expanduser(args.root_dir))
    if not root.is_dir():
        parser.error(f"--root-dir '{args.root_dir}' is not an existing directory")
    ROOT_DIR = root.resolve()

    if args.debug:
        server.settings.log_level = "DEBUG"

    # stdio mode carries JSON-RPC on stdout: keep the banner off it.
    banner = sys.stderr if args.transport == "stdio" else sys.stdout
    print(f"🚀 Console MCP Server starting...", file=banner)
    if args.transport == "stdio":
        print(f"   Transport: stdio", file=banner)
    else:
        print(f"   Endpoint: http://{args.host}:{args.port}/mcp", file=banner)
    print(f"   Root directory: {ROOT_DIR}", file=banner)
    print(f"   Allowed commands: {len(ALLOWED_COMMANDS)}", file=banner)
    print(f"   Blocked commands: {len(BLOCKED_COMMANDS)}", file=banner)
    print(f"\n   Press Ctrl+C to stop\n", file=banner)

    # Run the server with the selected transport
    try:
        if args.transport == "stdio":
            server.run(transport="stdio")
        else:
            server.run(
                transport="streamable-http",
                host=args.host,
                port=args.port,
            )
    except (KeyboardInterrupt, SystemExit):
        print("\nConsole MCP Server stopped.", file=banner)


if __name__ == "__main__":
    main()
