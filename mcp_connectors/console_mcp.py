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
Shells (bash, cmd, powershell, ...) are refused by name everywhere -- on
Windows the tools come from Git for Windows' POSIX toolset, never an
interpreter. The command sets and the program search path are one startup-
selected block (gb-03): found Git toolset -> the gbash sets resolved through
Git's trees, absent on Windows -> the connector refuses to start naming the
gap, POSIX -> today's behaviour unchanged.

Credential stores (``~/.ssh``, ``~/.aws``, ``.git``, ``.env``, private keys and
friends) are refused as command arguments whatever the command is, using the
same lists as the fedit connector -- see ``mcp_connectors/path_policy.py``. A
secret read by ``cat`` does as much damage as one read by ``read_file``, so the
two connectors must not drift apart.

Inside the root, one policy does apply (pol-08): the workspace's own
``<root>/.nojae/read-write-policy.json``. A path its owner marked ``hidden`` is
refused as an argument (and as the ``cwd``); one marked ``ro`` is refused for the
commands that would create or rewrite it (see ``WRITE_ALL_PATHS_COMMANDS`` /
``WRITE_DEST_LAST_COMMANDS``), which is the whole of this connector's write side
-- there is no shell, so the token scan *is* the check, and a tool that writes
through its own flags (``tar -czf``) is out of its reach. The rule is the one the
Editor pane applies to the same path, read from the same file with the same
matcher, and a refusal names the pattern so the model can ask differently instead
of retrying the same command.

That narrows the agent; it adds no power anywhere. The two rule sources stay
separate lists and must not be merged: ``path_policy`` answers "is this a
credential store" -- a question with the same answer in every tree, and the only
rule that speaks for paths *outside* the root -- while the workspace policy
answers "which parts of this project may change", a question the operator wrote
down. The resulting asymmetry is documented rather than fixed: inside the root a
``.env`` is reachable through ``cat`` here (the root bypass, and the pane's own
``.env`` rule is level A -- benchmarkit regenerates that file, which is not a
fact about an agent's edit) unless the workspace policy says otherwise.

Usage:
    python -m mcp_connectors.console_mcp --root-dir /path/to/root              # default port 9999
    python -m mcp_connectors.console_mcp --root-dir /path --port 8080          # custom port
    python -m mcp_connectors.console_mcp --root-dir /path --transport stdio    # stdio (for MCP hubs)
"""

import argparse
import ntpath
import os
import re
import shutil
import subprocess
import shlex
import sys
from pathlib import Path
from typing import Optional, Tuple
from mcp.server.mcpserver import MCPServer

try:  # imported as part of the package (tests, `python -m mcp_connectors...`)
    from . import fscase, gitbash, path_norm, path_policy, policy_file, policy_shared
except ImportError:  # run as a plain script path, as externals.json does
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import fscase  # type: ignore[no-redef]
    import gitbash  # type: ignore[no-redef]
    import path_norm  # type: ignore[no-redef]
    import path_policy  # type: ignore[no-redef]
    import policy_file  # type: ignore[no-redef]
    import policy_shared  # type: ignore[no-redef]

# Root directory the console is scoped to; set by main() from --root-dir.
# When None (e.g. module imported without main()), the full policy applies to
# every command (legacy behaviour).
ROOT_DIR: Optional[Path] = None

# The case answer for the volume this connector walks (gb-01/gb-02, win-05's
# probe from ``mcp_connectors.fscase``). The deny-lists take ``fold`` as an
# explicit argument everywhere (roadmap §9 rule 2), and this module reads it
# only through that argument — so the Linux default (``False``, an exact
# filesystem) is what the existing policy tests exercise. Under a drive-rooted
# root the answer comes from the probe of that root (:func:`path_model_fold`);
# this attribute is the fallback for an unrooted or POSIX-rooted server and the
# seam tests force either direction through (gb-02's design decision).
CASE_FOLDED: bool = False


# --- the Windows path model (gb-02, G-D5/G-D6) -------------------------------
#
# The path-model half — the backslash rule, the mount-form translation, the
# drive-relative refusal, the fold — is selected by the *shape of the root*,
# not by an OS name: ``root_form(ROOT_DIR) == "drive"`` is a probe of the root
# (roadmap §9 rule 2), so a Windows test root gets the Windows path model with
# no test edits, a Linux root keeps today's exact verdicts, and tests force
# either direction by patching ``ROOT_DIR`` or passing the explicit flag. The
# *toolset* half (Git discovery, resolver, command sets) is a separate,
# startup-selected mode (gb-03): this section changes no command verdict.

def root_is_drive_form() -> bool:
    """True when ``ROOT_DIR`` is a drive-spec path — the path-model mode's shape probe."""
    return ROOT_DIR is not None and path_policy.root_form(str(ROOT_DIR)) == "drive"


def path_model_backslash(literal_backslash: Optional[bool] = None) -> bool:
    """Whether argv keeps backslashes verbatim: derived from the root's shape unless forced."""
    if literal_backslash is not None:
        return literal_backslash
    return root_is_drive_form()


def path_model_fold() -> bool:
    """The case answer for the paths this verdict walks (win-05's probe, per root)."""
    if not root_is_drive_form():
        return CASE_FOLDED
    return fscase.case_folded(str(ROOT_DIR))


def verdict_path_text(token: str) -> str:
    r"""The token a *policy question* asks about, with the mount form translated (G-D6).

    Git's tools accept the MSYS mount form (``/c/dev/x`` is ``C:\dev\x``), and
    argv must reach them unchanged — so the translation happens here, for the
    verdict only, and only under a drive-rooted root: under a POSIX root
    ``/c/x`` stays an ordinary absolute path and today's verdicts stay exact.
    """
    if root_is_drive_form():
        mount = path_policy.mount_form_to_drive(token)
        if mount is not None:
            return mount
    return token

# Create MCP server instance
# streamable_http_path="/mcp" is the default, which matches your requirement
server = MCPServer(
    name="console-mcp",
    instructions="A server that allows LLMs to execute console commands, scoped to a root "
                 "directory chosen at startup (--root-dir). Commands that only touch paths "
                 "inside that root skip the allow/block lists, but they still obey that "
                 "folder's own workspace policy (.nojae/read-write-policy.json): a path "
                 "marked hidden is not readable, one marked ro cannot be written to, and "
                 "both refusals name the rule. Commands referencing paths outside the root "
                 "must pass the allow/block policy. "
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
# Target B1 (G-D3/G-D7) adds the shells and the Windows power/admin names to
# the same list: a shell is never path work (Git Bash supplies the *tools*, not
# an interpreter), and System32's bash.exe -- the WSL launcher -- is why
# refusing them by name is not theoretical.
ALWAYS_BLOCKED_COMMANDS = {
    "mkfs", "dd", "fdisk", "parted", "mount", "umount",
    "shutdown", "reboot", "poweroff", "halt",
    "sudo", "su", "doas", "runuser",
    "pacman", "systemctl", "crontab",
    # shells and script hosts, blocked on both platforms (G-D3)
    "bash", "sh", "cmd", "powershell", "pwsh", "cscript", "wscript",
    # Windows power / disk / service / registry control (G-D7)
    "del", "erase", "rd", "format", "takeown", "icacls", "cacls",
    "net", "sc", "reg", "schtasks", "runas", "cipher", "diskpart",
    "mountvol", "subst", "fsutil", "bcdedit", "vssadmin", "wmic",
    "msiexec", "regsvr32",
}

# Refused with the design named, not a generic "blocked for security reasons"
# (G-D3's sentence): the model should learn *why* a shell never works here.
SHELL_COMMANDS: frozenset[str] = frozenset({
    "bash", "sh", "cmd", "powershell", "pwsh", "cscript", "wscript",
})

NO_SHELL_SENTENCE: str = (
    "is a shell, and this connector runs none: Git for Windows supplies the "
    "tools, not an interpreter -- issue one plain command per call"
)

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
# The gbash toolset mode (gb-03, G-D3–G-D7) -- the one startup-selected block.
#
# The mode is a startup capability probe, not a platform branch inside the
# policy: main() asks whether Git for Windows' toolset is there; found -> the
# gbash semantics, absent on Windows -> the connector refuses to start naming
# the gap, POSIX -> today's behaviour, unchanged. Only the command sets and
# the resolver are mode-selected here (the backslash rule and the mount-form
# translation follow the root's shape -- gb-02); the grammar, the per-statement
# policy and the output honesty above stay one copy both modes share, so a
# policy rule changes once and both modes inherit it (G-D2's invariant).
#
# The Windows half is one named seam: TOOLSET, GIT_SEARCH_PATHS and the
# discovery candidates are module attributes, and tests force either direction
# by patching them (roadmap §9 rule 2) -- never by pretending to be Windows.
# ---------------------------------------------------------------------------

# The names the probe found in no tree on this machine (target B1 §1): they
# leave the gbash allow list because nothing resolves them (B-D4's principle --
# the list is what the connector can actually run, not a guess).
GBASH_ABSENT_COMMANDS: frozenset[str] = frozenset({
    "man", "bc", "zip", "top", "vmstat", "free", "uptime", "lscpu",
    "lsb_release", "w", "tracepath", "ss", "ip", "dig", "jq", "sqlite3",
    "rg", "ag", "fd", "make", "lsof", "tree", "wget",
})

# The System32 / ambient-PATH half (G-D7): real Windows tools the POSIX list
# never named. ``certutil -hashfile`` fills the md5sum/sha256sum role if Git's
# coreutils are ever absent; ``fc`` is a real file-compare tool here, so it
# leaves the builtin list on this connector.
GBASH_AMBIENT_COMMANDS: frozenset[str] = frozenset({
    "python", "python3", "node", "npm", "pip", "git", "curl", "ping",
    "nslookup", "ipconfig", "where", "certutil", "systeminfo", "tasklist",
    "findstr", "tracert", "netstat", "more", "fc",
})

GBASH_ALLOWED_COMMANDS: frozenset[str] = (
    (frozenset(ALLOWED_COMMANDS) - GBASH_ABSENT_COMMANDS) | GBASH_AMBIENT_COMMANDS
)

# cmd's own words: no `.exe` stands behind them under shell=False (`type`
# among them), so they are refused with the builtin hint rather than coming
# back as "Command not found". `fc` leaves the set -- fc.exe is a real tool.
GBASH_BUILTIN_COMMANDS: frozenset[str] = (
    (SHELL_BUILTIN_COMMANDS - {"fc"})
    | frozenset({
        "cls", "copy", "move", "ren", "md", "ver", "vol", "start",
        "call", "if", "for", "goto", "type",
    })
)

# Git install roots searched when `where git` finds nothing (G-D4), and the
# connector-local override -- deliberately not a managed .env key, the same
# reasoning as BENCHMARKIT_PATH_LOCATION: a settings save must not erase it.
# Module attributes so tests patch them (the hermetic-tests convention): these
# are the shared probe's own defaults (csl-01 -- one probe for the connector and
# for the terminal alike), rebound to this module's names because its tests and
# its refusal sentence speak them.
GIT_INSTALL_CANDIDATES: list[str] = list(gitbash.GIT_INSTALL_CANDIDATES)
GIT_ENV_OVERRIDE: str = gitbash.GIT_ENV_OVERRIDE

# The command search path once a Git root is known (G-D4): Git's trees first,
# so `find`/`sort`/`tar` mean GNU, then the ambient PATH for python/npm/curl.
GIT_SEARCH_PATHS: list[str] = []


class Toolset:
    """One mode's selected half: the command sets and whether a resolver runs.

    The named block G-D1 asks for. The grammar and the path policy are not
    here -- they are the shared core above, and both modes read them.
    """

    def __init__(
        self, name: str, allowed: frozenset[str], builtins: frozenset[str],
        resolves: bool,
    ) -> None:
        self.name = name
        self.allowed = allowed
        self.builtins = builtins
        self.resolves = resolves


POSIX_TOOLSET = Toolset("posix", frozenset(ALLOWED_COMMANDS), SHELL_BUILTIN_COMMANDS, False)
GBASH_TOOLSET = Toolset("gbash", GBASH_ALLOWED_COMMANDS, GBASH_BUILTIN_COMMANDS, True)

# Startup-selected; the POSIX default keeps every existing test exercising
# today's behaviour verbatim (the Linux gate stays literal).
TOOLSET: Toolset = POSIX_TOOLSET


def discover_git_install() -> Optional[str]:
    """The Git for Windows install root, or None (G-D4's four questions).

    1. `shutil.which("git")` walked up from `...\\cmd\\git.exe` to the install
       dir; 2. the fixed candidates; 3. the connector-local env override;
       4. None -- which main() turns into the refusal. A found root is always
       verified by `usr\\bin\\ls.exe`, so a stray `git.exe` elsewhere on PATH
       cannot select the mode.

    The asking is csl-01's shared probe (`mcp_connectors.gitbash` -- the same
    one the terminal will use to pick its shell); this function hands it this
    module's patchable candidates and override name, so a test that patches
    them still answers here and the verdict order is unchanged.
    """
    return gitbash.discover_git_install(
        GIT_INSTALL_CANDIDATES, GIT_ENV_OVERRIDE, os.environ
    )


def git_search_paths(git_root: str) -> list[str]:
    """The command search path for a Git root, in resolution order (G-D4).

    Git's trees first, then whichever C-runtime tool tree the install actually
    carries (`ucrt64` on this machine's install, `mingw64` on others): the probe
    answers from what is there rather than naming a directory that may be
    missing. The ambient PATH is appended by :func:`resolve_command`.
    """
    return gitbash.git_search_paths(git_root)


def resolve_command(name: str) -> Optional[str]:
    r"""The program ``name`` resolves to, or None -- never a shell lookup.

    ``shutil.which`` honours ``PATHEXT`` (B-D6: ``more.com`` and ``.cmd``
    shims resolve) and searches ``GIT_SEARCH_PATHS`` before the ambient PATH,
    so a name Git's trees carry never falls through to ``System32``. A name
    that already spells a path (``C:\tools\x.exe``, ``./x``) is asked directly.
    The answer is what the verdict promises *and* what execute_command runs --
    the ``FileNotFoundError`` folded into the oracle (G-D4).
    """
    if os.sep in name or (os.altsep and os.altsep in name) or "/" in name:
        candidate = Path(name)
        if candidate.is_file():
            return str(candidate)
        with_ext = shutil.which(name)
        return with_ext
    for directory in GIT_SEARCH_PATHS:
        found = shutil.which(name, path=directory)
        if found:
            return found
    return shutil.which(name)


def no_toolset_refusal() -> str:
    """The sentence naming the gap when discovery finds nothing (win-07's shape)."""
    searched = ", ".join(GIT_INSTALL_CANDIDATES)
    return (
        f"no Git for Windows toolset found: `git` was not on PATH, none of "
        f"the install candidates ({searched}) carries usr\\bin\\ls.exe, and "
        f"${GIT_ENV_OVERRIDE} names none. This connector runs its commands "
        "over Git's POSIX toolset and starts no shell; install Git for "
        f"Windows, or point {GIT_ENV_OVERRIDE} at its install root."
    )


def select_toolset() -> Optional[str]:
    """Install the startup mode; return the refusal sentence when Windows has no toolset.

    A POSIX system keeps today's behaviour unchanged -- it is not probed,
    because a Linux box's own ``git`` must not select the gbash sets. On
    Windows the probe decides: found -> the gbash mode (and the fold answer
    for the root, G-D8); absent -> the connector refuses to start naming the
    gap, which is what keeps "each connector's reach is bounded by a deny-list"
    true until the guardrails say so (gb-01 landed that half).
    """
    global TOOLSET, GIT_SEARCH_PATHS
    if os.name != "nt":
        TOOLSET = POSIX_TOOLSET
        return None
    git_root = discover_git_install()
    if git_root is None:
        return no_toolset_refusal()
    GIT_SEARCH_PATHS = git_search_paths(git_root)
    TOOLSET = GBASH_TOOLSET
    return None


def toolset_split() -> tuple[list[str], list[str]]:
    """(resolved, unresolved) of the active allow list -- the startup self-check.

    The list is "selected by which platform's binaries the connector can
    actually resolve" (B-D4's principle); this is the honest log of that claim,
    printed once at startup rather than a hand-verified guess.
    """
    resolved: list[str] = []
    unresolved: list[str] = []
    for command in sorted(TOOLSET.allowed):
        (resolved if resolve_command(command) else unresolved).append(command)
    return resolved, unresolved



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


def unquote(token: str, literal_backslash: bool = False) -> str:
    """Strip the quoting a token carries, so ``argv`` holds the word to exec.

    A single open quote is closed as a double quote rather than treated as
    unbalanced, so an odd number of quotes degrades the way a shell would
    degrade rather than losing the whole argument.

    ``literal_backslash=True`` (G-D5, gb-02) keeps every ``\\`` in the word:
    ``shlex.split`` eats backslashes -- ``cat C:\\Users\\vgr\\a.txt`` used to
    reach argv as ``C:Usersvgra.txt``, which read as an in-root name and
    bypassed the out-of-root policy. The quote-stripping here is verbatim, so
    quoting still groups spaces (`cat "C:\\Users\\vgr\\a file.txt"`) while the
    backslashes survive for the tool that accepts them. The default ``False``
    keeps the Linux verdicts exact (roadmap §9 rule 2).
    """
    if literal_backslash:
        return unquote_literal(token)
    if len(token) > 1 and token.startswith("'") and not token.endswith("'"):
        token = '"' + token[1:]
    try:
        words = shlex.split(token)
    except ValueError:
        return token.strip("\"'")
    return words[0] if words else ""


def unquote_literal(token: str) -> str:
    """Strip quoting verbatim: quotes go, every other character -- a ``\\``
    included -- stays exactly where the caller wrote it."""
    chars: list[str] = []
    quote = ""
    for char in token:
        if quote:
            if char == quote:
                quote = ""
            else:
                chars.append(char)
            continue
        if char in "\"'":
            quote = char
            continue
        chars.append(char)
    return "".join(chars)


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


def parse_shell_statements(
    command: str, literal_backslash: bool = False
) -> list[ShellStatement]:
    """Split ``command`` into statements: argv, redirections, shell features.

    Quoting decides what counts: `grep "<div" file` is one argv with no
    redirection and `curl "…?a=1&b=2"` has no operator, while `ls web; id` and
    `ls web` + newline + `id` are both **two** statements -- every one of which
    then goes through the policy on its own (C-3, C-5).

    ``literal_backslash`` reaches :func:`unquote` (G-D5): with it, a ``\\`` is a
    character of argv, not an escape shlex eats. The grammar itself -- the
    operators, the redirections, the refusal sentences -- is identical either
    way: the same parse is what :func:`validate_command` judges and what
    :func:`execute_command` runs, in both directions.

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
        current.argv.append(unquote(token, literal_backslash))
        index += 1
    close_statement()
    if not statements:
        raise ShellSyntaxError("no statement to run")
    return statements


def resolve_as_path(token: str, base: Optional[Path] = None) -> Path:
    """Resolve a command token as a path; relative paths anchor at ``base``.

    ``base`` defaults to the server root, which is where the connector starts. It
    is a parameter because of the tool's ``cwd`` argument: a command run with
    ``cwd="vendor"`` resolves its relative arguments in ``vendor``, so a policy
    check that anchored them at the root would judge ``x.txt`` as a file that is
    not being written at all (pol-08's write gate would then wave a create inside
    a read-only folder past).

    The token first passes :func:`verdict_path_text`: a mount-form argument
    (``/c/dev/x``) is judged as the drive path it names (G-D6), while argv
    reaches the tool unchanged.
    """
    expanded = os.path.expanduser(verdict_path_text(token))
    path = Path(expanded)
    if not path.is_absolute():
        path = (base or ROOT_DIR or Path.cwd()) / path
    try:
        return path.resolve()
    except OSError:
        return Path(os.path.normpath(str(path)))


def absolute_as_written(token: str, base: Optional[Path] = None) -> Path:
    """The absolute path a token names, as written (nothing collapsed).

    ``os.path.abspath`` would fold ``~/.ssh/../../etc`` into an ordinary
    ``/home`` path and hide the store it walks out of, so the components are
    kept as given; the symlink-resolved target is checked separately.

    ``base`` is the directory the command will actually run in (the tool's
    ``cwd`` argument), for the same reason as :func:`resolve_as_path`'s. The
    mount-form translation applies here too -- the as-written half of a verdict
    asks the same Windows questions as the resolved half.
    """
    path = Path(os.path.expanduser(verdict_path_text(token)))
    if not path.is_absolute():
        path = (base or ROOT_DIR or Path.cwd()) / path
    return path


def is_within_root(token: str, base: Optional[Path] = None) -> bool:
    """True when the token resolves to a path inside (or equal to) ROOT_DIR."""
    if ROOT_DIR is None:
        return False
    resolved = resolve_as_path(token, base)
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
    return path_policy.prefix_hit(
        path.as_posix(), path_policy.ALLOW_PREFIXES, fold=path_model_fold()
    )


def _workspace_rel(real: Path) -> Optional[str]:
    """``real`` as a workspace-relative POSIX path, or None when outside the root.

    The connector's root *is* the workspace (``externals.json`` passes
    ``--root-dir ${BENCHMARKIT_PATH_LOCATION}``), and the workspace policy's
    patterns are written relative to that folder, so this is the form they are
    matched against. ``None`` -- a path outside the root, or an unrooted server
    -- means the workspace policy has nothing to say here, which is the domain
    split this task pins: outside the root the connector deny-list answers, and
    it answers by *name*, so ``~/.ssh`` is refused in every workspace.
    """
    if ROOT_DIR is None:
        return None
    try:
        rel = real.relative_to(ROOT_DIR)
    except ValueError:
        return None
    text = rel.as_posix()
    return "" if text == "." else text


def workspace_policy_error(real: Path, *, for_write: bool) -> Optional[str]:
    """The workspace policy's refusal for an in-root path token, or None (pol-08).

    The same file the Editor pane reads, matched by the same matcher, and answered
    with the same sentence -- a pane and an agent disagreeing about one folder is
    the state this prevents. ``hidden`` refuses reading and writing; ``ro`` refuses
    only what would change the path, because *browsing* an ``ro`` folder is the
    reason the mode exists (``**/node_modules`` is listed read-only so its README
    stays readable).

    Level A (benchmarkit's own rules about the files it regenerates) is not
    consulted here, deliberately: it exists because the pane fights the settings
    dialog over a generated ``.env``, which is not a property of an agent's edit.
    The reserved ``.nojae`` folder *is* enforced, and it is checked on the
    component list before the policy file is opened at all -- an agent rewriting
    the rules that constrain it is the same accident ``DENY_COMPONENTS`` includes
    ``mcp_connectors`` to prevent, and that refusal has to survive a corrupt
    file. (A *write* is refused whether or not the name exists; a *read* of a name
    that does not is skipped by the rule below, which is what a hidden path is
    supposed to look like anyway.)

    A policy file that does not parse refuses **writes** naming the cause and
    leaves reads alone (pol-00 D4), matching what the app does with the same file.
    """
    rel = _workspace_rel(real)
    if rel is None:
        return None
    # ``path_argument_tokens`` cannot tell a path from an argument word -- ``git
    # status`` yields the token ``status``, resolving to <root>/status, and a
    # ``hidden`` rule would then refuse a command that touches no file. So the
    # read half is asked only of a name that is actually on disk: a token naming
    # nothing discloses nothing. A **write** is never skipped, because ``touch x``
    # inside a read-only folder creates the name -- its verdict comes from the
    # rule, not from whether the path exists yet.
    if not for_write and not os.path.lexists(real):
        return None
    root = str(ROOT_DIR) if ROOT_DIR is not None else ""
    reason = policy_file.workspace_policy_reason(
        rel, root=root or None, for_write=for_write
    )
    if reason is None:
        return None
    return (
        f"{real} is off-limits in this workspace: {reason}. The folder's own "
        "policy decides what an agent may read or change there, with the same "
        f"rules the Editor tab follows; it is edited outside benchmarkit, in "
        f"{policy_file.workspace_policy_path(root)}."
    )


# Which path tokens a command would *change*, as opposed to only look at. This is
# what makes an ``ro`` folder mean something to an agent: ``cat`` and ``ls`` stay
# allowed inside it, and the commands that create, rewrite or remove a name do not.
#
# The grammar is deliberately narrow and its limit is named rather than hidden.
# ``cp``/``ln`` name their destination last, so only that token is a write --
# copying *out* of a read-only folder changes nothing in it. ``mv`` counts every
# path token as a write: a move rewrites the **source** folder's directory
# entries too, which is why the Editor's rename gate refuses a move out of an
# ``ro`` folder (pol-00 D3). The destructive names are blocked by name only
# outside the root, so inside it they are real writes and are judged as such.
#
# A tool that reaches its output through its *own* flags (``tar -czf out``,
# ``zip``, ``git commit``, ``npm install``) is not parsed here at all: ``ro`` is
# the operator's rule about the paths they and their agent name, not a sandbox
# claim, and a scan that pretended otherwise would be a promise this connector
# cannot keep.
WRITE_ALL_PATHS_COMMANDS: frozenset[str] = frozenset({
    "mkdir", "touch", "gzip", "mv", "rm", "rmdir", "truncate", "shred",
})
WRITE_DEST_LAST_COMMANDS: frozenset[str] = frozenset({"cp", "ln"})


def write_target_tokens(base_command: str, path_tokens: list[str]) -> frozenset[str]:
    """The path tokens a command would create or rewrite, out of its path tokens.

    Empty for a command that only looks (``cat``, ``ls``, ``grep``, ``head``),
    which is what keeps a read-only folder browsable -- reading it is the reason
    the mode is called ``ro`` rather than ``hidden``.
    """
    if base_command in WRITE_ALL_PATHS_COMMANDS:
        return frozenset(path_tokens)
    if base_command in WRITE_DEST_LAST_COMMANDS and path_tokens:
        return frozenset({path_tokens[-1]})
    return frozenset()


def protected_path_error(
    token: str, base: Optional[Path] = None
) -> Optional[str]:
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
    * a path inside a scratch prefix (``/tmp``, ``/var/tmp``, ``%TEMP%``) --
      but *after* the component/name half is asked (B-D2's ordering, gb-01):
      a secret named inside the scratch tree (``%TEMP%\\.ssh\\id_rsa``,
      ``/tmp/known_hosts``) is refused, because the allowance covers
      throwaway work, not a store parked in a temp folder. It can still
      rescue a path the location half alone would refuse.

    A token is judged both as written and as its symlink-resolved target, so
    ``~/.ssh/../../etc`` cannot read as an ordinary home path -- and a name
    planted in temp whose target *is* a store is refused by the same pair.

    ``base`` is the directory the command will run in -- the tool's ``cwd``
    argument -- because that is where a relative token lands: judging ``x``
    against the root when the command runs in ``<root>/vendor`` would ask about
    a file the command never touches.
    """
    as_written = absolute_as_written(token, base)
    real = resolve_as_path(token, base)

    inside_root = is_within_root(token, base)
    if inside_root and (real == as_written or _within_root_or_none(real)):
        return None

    def refusal_from(ask) -> Optional[str]:
        for label, candidate in (("the path", as_written), ("its real target", real)):
            reason = ask(candidate)
            if reason:
                detail = "" if candidate == as_written else f" ({label} {real})"
                return (
                    f"{as_written}{detail} is off-limits: it is {reason}. "
                    "Credential stores and secret files are refused by every "
                    "connector in this project, whatever the command is."
                )
        return None

    # Identity first: what the path *is* outranks where it sits (B-D2).
    refusal = refusal_from(
        lambda candidate: path_policy.name_component_deny_reason(
            candidate, for_write=False, fold=path_model_fold()
        )
    )
    if refusal:
        return refusal

    if _trusted_prefix_hit(as_written) and _trusted_prefix_hit(real):
        return None

    # Location second: the protected prefixes, with the scratch allowance
    # already weighed against them.
    return refusal_from(
        lambda candidate: path_policy.prefix_deny_reason(
            candidate, for_write=False, fold=path_model_fold()
        )
    )


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


def drive_relative_spec(token: str) -> bool:
    r"""True for a drive-relative token (`C:x`, `C:`) — the hole win-02 names.

    ``path_norm.is_qualified`` calls a one-letter drive spec a hole because
    ``ntpath.join(root, "C:x")`` lands on *another drive's* current directory:
    the containment lock would stop it only by luck, and the in-root bypass
    could wave it through as a relative name. The connector refuses the shape
    by name instead (G-D6). Only the drive spec with nothing but a name after
    the colon qualifies: a qualified drive path (``C:\dir\x``, ``C:/dir``), a
    UNC share and a POSIX-shaped token are other roots with their own
    verdicts, and a name like ``foo:bar`` is no drive spec at all. The test is
    text -- ``Path.is_absolute`` would answer differently per platform.
    """
    expanded = os.path.expanduser(verdict_path_text(token))
    if path_policy.root_form(expanded) != "drive" or not path_norm.is_qualified(expanded):
        return False
    _drive, rest = ntpath.splitdrive(expanded)
    return rest[:1] not in ("/", "\\")


def _resolver_refusal(base_command: str) -> Optional[str]:
    """The unresolved-command refusal (G-D4), or None when it resolves.

    Folded into the verdict at the two points a statement can pass the name
    checks -- the in-root bypass and the end of the list -- so the oracle never
    promises a command ``execute_command`` cannot run, and a name the lists
    already refuse keeps *its* sentence. Only the gbash toolset resolves; the
    POSIX mode keeps today's behaviour, where an absent command surfaces at
    exec as it always did.
    """
    if not TOOLSET.resolves:
        return None
    if resolve_command(base_command) is not None:
        return None
    searched = ", ".join(GIT_SEARCH_PATHS + ["PATH"])
    return (
        f"Command '{base_command}' was not found in the searched toolset "
        f"({searched}); it names no program this connector can run"
    )


def statement_policy_error(
    statement: ShellStatement, work_dir: Optional[Path] = None
) -> Optional[str]:
    """Run the list/path policy on ONE statement, or None when it passes.

    The verdict is per statement, not per command string: the old code looked
    at ``parts[0]`` of the whole string, so a `..` anywhere retro-actively
    changed the verdict of every other part, and no command past the first
    separator was ever examined (C-5).

    ``work_dir`` is the directory the statement will be executed in (the tool's
    ``cwd`` argument, defaulting to the root). Every path token is anchored
    there, because that is where the child resolves it: judging ``notes.md``
    against the root while the command runs in ``<root>/vendor`` asks about a
    file nobody touches and lets a create land in a read-only folder.
    """
    base_command = statement.argv[0]
    base = work_dir if work_dir is not None else ROOT_DIR

    # Privilege / power / disk / service commands are blocked everywhere.
    # The shells answer with the design, not a generic block sentence (G-D3):
    # `cmd /c` and `bash -c` are the same refusal, on both platforms.
    if base_command in ALWAYS_BLOCKED_COMMANDS:
        if base_command in SHELL_COMMANDS:
            return f"Command '{base_command}' {NO_SHELL_SENTENCE}"
        return f"Command '{base_command}' is blocked for security reasons"

    # Builtins would otherwise reach exec() and come back as "Command not
    # found", which reads like a policy accident rather than what it is.
    if base_command in TOOLSET.builtins:
        return f"Command '{base_command}' {BUILTIN_HINT}"

    path_tokens = path_argument_tokens(statement.argv)

    # A drive-relative token (`C:x`) names a location on *another drive's*
    # current directory -- the hole win-02's `path_norm.is_qualified` names,
    # and one the in-root bypass would wave through as a relative name. Under a
    # drive-rooted root it is refused by shape, before any other question
    # (G-D6); under a POSIX root today's verdict stands (an ordinary filename).
    if root_is_drive_form():
        for token in path_tokens:
            if drive_relative_spec(token):
                return (
                    f"Path {token!r} is a drive-relative spec: it names a "
                    "location relative to a drive's current directory, which "
                    "this connector cannot anchor. Spell it absolute "
                    r"(C:\dir\name) or leave it workspace-relative."
                )

    # Credential stores are off-limits whatever the rest of the policy decided
    # (C-6): the lists below answer "may this command run", never "is this
    # target allowed", so an allow-listed `cat` used to reach ~/.ssh.
    for token in path_tokens:
        protected = protected_path_error(token, base)
        if protected:
            return protected

    # The workspace's own policy, asked of every path token and *before* the
    # in-root bypass below (pol-08): a folder its owner marked ``ro`` or
    # ``hidden`` constrains an agent exactly as it constrains the Editor pane.
    # It answers None for a path outside the root, where the deny-list above is
    # what spoke -- so this neither extends the policy beyond the workspace nor
    # loosens the floor. Whether a token is a *write* comes from the command
    # word, because ``ro`` must leave a folder browsable -- that is the whole
    # difference between it and ``hidden``.
    #
    # This is a sibling of :func:`protected_path_error` rather than a line inside
    # it for that one reason: the write/read mode is a property of the *command*,
    # and ``protected_path_error`` is called with a bare token. Same coverage --
    # both run over every path token of every statement, and neither is subject to
    # the root bypass.
    write_tokens = write_target_tokens(base_command, path_tokens)
    for token in path_tokens:
        refusal = workspace_policy_error(
            resolve_as_path(token, base), for_write=token in write_tokens
        )
        if refusal:
            return refusal

    # A statement whose path arguments all stay inside the root is ordinary
    # workspace work and is not list-checked. Anchored at ``base`` for the same
    # reason as every other question above: with ``cwd`` set, a relative token
    # resolves under that directory, and "does this reach outside the root" has
    # to be asked about the file the child will touch.
    if ROOT_DIR is not None and path_tokens and all(
        is_within_root(t, base) for t in path_tokens
    ):
        return _resolver_refusal(base_command)

    if base_command in BLOCKED_COMMANDS:
        return f"Command '{base_command}' is blocked for security reasons"
    if base_command not in TOOLSET.allowed:
        return f"Command '{base_command}' is not in the allowed commands list"
    return _resolver_refusal(base_command)


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


def resolve_work_dir(
    cwd: Optional[str],
) -> Tuple[Optional[Path], Optional[str]]:
    """``(working directory, error)`` for a tool call's ``cwd`` argument.

    One copy of the rules, because :func:`check_command_allowed` promises to be
    the verdict :func:`execute_command` applies: the working directory decides
    where every relative path token lands, so an oracle that resolved it
    differently would be lying about a command it never runs.

    Three questions, in order of who owns them: the server must be rooted (the
    root is what every scope here is relative to); the directory must stay inside
    it (the tool's own contract, predating pol-08); and it must be a directory the
    workspace policy will let this connector use -- a command started *inside* a
    folder lists its contents without ever naming a path, which is the observation
    ``hidden`` exists to refuse, and the only way the reserved policy folder would
    be reachable from here.
    """
    if ROOT_DIR is None:
        return (None, "server is not rooted; start it with --root-dir")
    if cwd is None:
        return (ROOT_DIR, None)
    candidate = resolve_as_path(cwd)
    if candidate != ROOT_DIR and ROOT_DIR not in candidate.parents:
        return (None, f"cwd '{cwd}' is outside the server root directory")
    refusal = workspace_policy_error(candidate, for_write=False)
    if refusal is not None:
        return (None, refusal)
    return (candidate, None)


def validate_command(
    command: str,
    work_dir: Optional[Path] = None,
    literal_backslash: Optional[bool] = None,
) -> tuple[bool, str]:
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

    ``work_dir`` is the directory the command will be executed in -- the same
    value ``execute_command`` hands to ``Popen(cwd=...)`` -- so a relative path
    token is judged where the child resolves it, not where the server started.
    Passing the root here while running the child elsewhere would check one file
    and execute another.

    Anything that needs the shell to have a meaning -- separators,
    redirections, substitution -- is refused, because what runs is exactly the
    argv that was validated.

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not command or not command.strip():
        return False, "Command cannot be empty"

    literal = path_model_backslash(literal_backslash)
    try:
        statements = parse_shell_statements(command, literal)
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
            if base_command in SHELL_COMMANDS:
                refusal = f"Command '{base_command}' {NO_SHELL_SENTENCE}"
            else:
                refusal = f"Command '{base_command}' is blocked for security reasons"
            return False, refusal + statement_position(index, total)

    # Then the structure the connector cannot honour.
    structure_error = shell_feature_error(statements)
    if structure_error:
        return False, structure_error

    for index, statement in enumerate(statements, start=1):
        error = statement_policy_error(statement, work_dir)
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
             inside the server's --root-dir and inside a folder the workspace
             policy does not hide; defaults to the root). Relative path
             arguments resolve against it, so the directory is checked too.

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
    work_dir_path, work_dir_error = resolve_work_dir(cwd)
    if work_dir_error is not None:
        return f"Error: {work_dir_error}"
    assert work_dir_path is not None  # resolve_work_dir returns one or the other
    work_dir = str(work_dir_path)

    # Validate the command **with the working directory it will run in**: the
    # child resolves a relative argument against its own cwd, so a check that
    # anchored those arguments at the server root would judge one file and
    # execute another -- `execute_command("touch x", cwd="vendor")` writes
    # <root>/vendor/x, and an ``ro`` rule on ``vendor`` has to see that.
    is_valid, error_msg = validate_command(command, work_dir_path)
    if not is_valid:
        return f"Error: {error_msg}"

    # What runs is the argv that was just validated, not the string handed to a
    # shell: `shell=True` re-interpreted the same text with a richer language
    # than the checker understood, so a newline started a second statement no
    # policy ever looked at and `$( )` ran inside the in-root bypass (C-3, C-4).
    # Validation above already refused every multi-statement command, so there
    # is exactly one statement here.
    try:
        statements = parse_shell_statements(command, path_model_backslash())
    except ShellSyntaxError as exc:
        return f"Error: Invalid command format: {exc}"
    argv = statements[0].argv

    # The gbash toolset execs the program the resolver found (G-D4): the search
    # order that decided the verdict is the search order that finds the binary,
    # so `find`/`sort`/`tar` are Git's GNU tools here too, never System32's
    # look-alikes. The POSIX mode hands argv over unchanged, exactly as today.
    if TOOLSET.resolves:
        program = resolve_command(argv[0])
        if program is None:
            return (
                f"Error: Command '{argv[0]}' was not found in the searched "
                f"toolset ({', '.join(GIT_SEARCH_PATHS + ['PATH'])})"
            )
        argv = [program] + argv[1:]


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
    sorted_commands = sorted(TOOLSET.allowed)
    # Say what the list actually decides (C-2): inside the root no list is
    # consulted at all, so an agent hunting here for the "allowed spelling" of
    # a refused in-root command is wasting its turns.
    toolset_note = (
        f"  Toolset mode: {TOOLSET.name} -- the command sets are selected at "
        "startup by what the connector can actually resolve\n"
        "  (Git's POSIX toolset on Windows; the POSIX list unchanged "
        "elsewhere). Shells are refused everywhere.\n"
        if TOOLSET.resolves else
        "  Toolset mode: posix -- the POSIX command list, unchanged.\n"
    )
    policy_note = (
        f"  Inside the root the rules are the workspace's own policy file\n"
        f"  ({policy_file.workspace_policy_path(str(ROOT_DIR)) if ROOT_DIR else '(no root)'}):\n"
        f"  a folder marked hidden is not readable, one marked ro cannot be written\n"
        f"  into, and {policy_shared.POLICY_DIRNAME} itself is reserved. Those "
        f"refusals name the rule.\n"
    )
    return (
        f"Commands accepted for work OUTSIDE the root ({len(sorted_commands)} total).\n"
        f"  This list is only consulted when the command references a path outside\n"
        f"  the server root ({ROOT_DIR}); inside the root, commands run without a\n"
        f"  list check, so a refusal there has a different reason (a blocked\n"
        f"  command, a shell operator, or the workspace policy below).\n"
        + policy_note +
        f"  Privilege/power commands stay blocked everywhere.\n\n"
        + "\n".join(f"  - {cmd}" for cmd in sorted_commands)
    )


@server.tool()
def check_command_allowed(command: str, cwd: Optional[str] = None) -> str:
    """
    Check if a specific command is allowed to be executed.

    Args:
        command: The command to check
        cwd: The working directory the command would be run in, exactly as it
             would be passed to execute_command. Pass it when the real call
             would: relative path arguments are resolved against it, so a
             command whose tokens land in a read-only or hidden folder is
             judged there rather than at the root.

    Returns:
        Whether the command is allowed and any restrictions. The verdict is the
        one execute_command applies (same parse, same policy, same working
        directory), so it can be used as an oracle without running anything.
        Note this tool takes no timeout: when the command does run,
        execute_command's default of 30 seconds applies, and a timeout returns
        whatever the command had already printed.
    """
    work_dir, work_dir_error = resolve_work_dir(cwd)
    if work_dir_error is not None:
        return f"Command '{command}' is NOT ALLOWED: {work_dir_error}"
    is_valid, error_msg = validate_command(command, work_dir)
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
             "Commands touching only paths inside it bypass the block/allow "
             "lists, but still obey that folder's own workspace policy "
             "(.nojae/read-write-policy.json)."
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

    # stdio mode carries JSON-RPC on stdout: keep the banner off it.
    banner = sys.stderr if args.transport == "stdio" else sys.stdout

    # The startup capability probe (G-D1/G-D4): found -> the gbash semantics;
    # absent on Windows -> refuse to start naming the gap (win-07's honesty
    # shape: stderr, non-zero exit, the hub reports the failed connection);
    # POSIX -> today's behaviour, unchanged.
    refusal = select_toolset()
    if refusal is not None:
        print(f"Console MCP Server refused to start: {refusal}", file=sys.stderr)
        sys.exit(1)
    if TOOLSET.name == "gbash":
        global CASE_FOLDED
        CASE_FOLDED = fscase.case_folded(str(ROOT_DIR))

    if args.debug:
        server.settings.log_level = "DEBUG"

    print(f"🚀 Console MCP Server starting...", file=banner)
    if args.transport == "stdio":
        print(f"   Transport: stdio", file=banner)
    else:
        print(f"   Endpoint: http://{args.host}:{args.port}/mcp", file=banner)
    print(f"   Root directory: {ROOT_DIR}", file=banner)
    print(f"   Toolset: {TOOLSET.name}", file=banner)
    if TOOLSET.resolves and GIT_SEARCH_PATHS:
        print(f"   Git toolset: {GIT_SEARCH_PATHS[0]}", file=banner)
    print(f"   Allowed commands: {len(TOOLSET.allowed)}", file=banner)
    print(f"   Blocked commands: {len(BLOCKED_COMMANDS)}", file=banner)
    if TOOLSET.resolves:
        # The honest log of what the list claims (B-D4's principle): which
        # names actually resolve, and from where.
        resolved, unresolved = toolset_split()
        print(
            f"   Resolved: {len(resolved)} allowed commands resolve; "
            f"unresolved: {', '.join(unresolved) if unresolved else 'none'}",
            file=banner,
        )
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
