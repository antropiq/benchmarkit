#!/usr/bin/env python3
"""
File Edit MCP Server - Edits files through unified diffs or SEARCH/REPLACE blocks.

This server exposes tools that let an AI model modify an existing file by sending
only the changed regions (a unified diff, or git-style SEARCH/REPLACE blocks)
instead of rewriting the whole file, or of reaching for low-level shell tools such
as ``sed``. Edits are validated against the file's actual content before anything
is written: a mismatch fails loudly and leaves the file untouched, and every write
keeps a numbered backup that ``revert_edit`` can restore.

Edits are applied by a pure-Python unified-diff engine (stdlib only) -- this is
*not* a wrapper around GNU ``patch``, so the accepted syntax is documented in
``apply_edit``'s docstring rather than delegated to ``patch(1)``.

Every tool -- reads included -- goes through one path policy before touching the
filesystem. Credential stores (``~/.ssh``, ``~/.aws``, ``.git``, ``.env`` and
friends) are refused for reading as well as writing, since a secret pulled into
the context window does as much damage as one that gets overwritten. The lists
and their environment overrides are reported by ``edit_policy()``.

Root scoping: the server MUST be started with --root-dir. The root and every
directory and file inside it is the operator's workspace: a path that stays
inside the root (literally and after symlink resolution) bypasses the policy
entirely. The protected prefixes, deny-lists and secret-file checks only apply
to paths outside the root. Relative paths are anchored at the root.

Usage:
    python -m mcp_connectors.fedit_mcp --root-dir /path/to/root              # default port 8946
    python -m mcp_connectors.fedit_mcp --root-dir /path --port 8080          # custom port
    python -m mcp_connectors.fedit_mcp --root-dir /path --transport stdio    # stdio (for MCP hubs)
"""

import argparse
import difflib
import fnmatch
import hashlib
import os
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

from mcp.server.mcpserver import MCPServer

# Root directory the connector is scoped to; set by main() from --root-dir.
# When None (e.g. module imported without main()), the full policy applies to
# every path (legacy behaviour).
ROOT_DIR: Optional[Path] = None

# Create MCP server instance
# streamable_http_path="/mcp" is the default, which matches your requirement
server = MCPServer(
    name="fedit-mcp",
    instructions="A server that edits existing files in place, given only the parts that "
                 "change: either a unified diff (```--- a/file / +++ b/file / @@ -l,s +l,s @@```) "
                 "or SEARCH/REPLACE blocks. Always call read_file first and copy its lines "
                 "verbatim into the edit -- the edit is matched against the real file content "
                 "and is rejected (without touching the file) if the context does not match. "
                 "Use preview_edit to check an edit before applying it. The server is scoped "
                 "to a root directory chosen at startup (--root-dir): files inside that root "
                 "are editable without policy restrictions; paths outside it are judged "
                 "against the protected-path lists reported by edit_policy().",
)

# ---------------------------------------------------------------------------
# Configuration: defaults, each overridable through the environment so the
# harness can tighten the policy without editing this module.
# ---------------------------------------------------------------------------

DEFAULT_PORT = 8946

# Refuse to touch files larger than this (bytes). An agent editing a huge file
# is almost certainly a mistake, and the diff would blow the context window.
MAX_FILE_BYTES = int(os.environ.get("FEDIT_MAX_BYTES", 2_000_000))

# Hard cap on the size of a single edit payload.
MAX_EDIT_BYTES = int(os.environ.get("FEDIT_MAX_EDIT_BYTES", 400_000))

# How many lines of a hunk's context may be ignored when repositioning it
# (the equivalent of ``patch --fuzz``). 0 disables repositioning entirely.
MAX_FUZZ = int(os.environ.get("FEDIT_MAX_FUZZ", 2))

# Number of backups kept per file; older ones are pruned.
BACKUP_KEEP = int(os.environ.get("FEDIT_BACKUP_KEEP", 10))

# Where pre-edit copies are stored. Kept out of the edited file's directory on
# purpose: backups must not pollute a working tree or show up in ``git status``.
BACKUP_DIR = Path(os.environ.get("FEDIT_BACKUP_DIR", "logs/fedit_backups"))

# System prefixes the connector never writes to.
DENY_PREFIXES: tuple[str, ...] = tuple(
    p for p in os.environ.get(
        "FEDIT_DENY_PREFIXES",
        "/etc,/usr,/bin,/sbin,/lib,/lib64,/boot,/sys,/proc,/dev,/run",
    ).split(",") if p
)

# Path components that must never appear in a path this connector writes: the
# version-control internals, credentials, the interpreter installation, and this
# connector's own sources (an agent must not be able to rewrite the policy that
# constrains it). An entry may span several components.
DENY_COMPONENTS = {".git", ".ssh", ".pyenv", "mcp_connectors"}

# Components that name a *secret store*, and are therefore off-limits to reads
# as well as writes: a private key or an access token pulled into the context
# window does as much damage as overwriting the file that held it. Everything
# below is env-overridable so a deployment can tighten or relax it without a
# code edit.
DENY_COMPONENTS_SECRET: frozenset[str] = frozenset(
    p.strip("/") for p in os.environ.get(
        "FEDIT_DENY_COMPONENTS_SECRET",
        ".git,.ssh,.gnupg,.aws,.azure,.gcloud,.kube,.docker,"
        ".password-store,.local/share/keyrings,.config/gh,credentials",
    ).split(",") if p.strip("/")
)

# Filename prefixes that hold secrets the app itself reads/writes.
DENY_FILE_PREFIXES = (".env",)

# Filenames that are secrets outright, and extensions that mark a private key or
# a certificate. Matched on the last path component, case-insensitively.
DENY_FILENAMES = {".git-credentials", "shadow", "known_hosts", "master.key",
                  "id_rsa", "id_dsa", "id_ecdsa", "id_ed25519", "id_sk_ecdsa",
                  "id_sk_ed25519"}
DENY_FILE_SUFFIXES = (".env", ".pem", ".key", ".p12", ".pfx", ".kdbx")

# System prefixes that are readable but never writable -- ordinary files here are
# useful debugging material, unlike the credential stores below.
DENY_PREFIXES_READ: tuple[str, ...] = tuple(
    p for p in os.environ.get(
        "FEDIT_DENY_PREFIXES_READ",
        "/etc/ssh,/etc/ssl/private,/etc/sudoers,/etc/shadow,/etc/gshadow,"
        "/proc,/sys,/dev,/boot,/run",
    ).split(",") if p
)

# Whole-path globs for stores whose directory name alone is too common to deny
# outright (``~/.config`` legitimately holds plenty of harmless configuration).
DENY_PATH_GLOBS: tuple[str, ...] = tuple(
    g for g in os.environ.get(
        "FEDIT_DENY_PATH_GLOBS",
        "*/.config/google-chrome*/*,*/.config/chromium*/*,*/.config/opera*/*",
    ).split(",") if g
)

# Effective component lists per operation: reads are blocked on secret stores
# only (an agent debugging a dependency legitimately needs to read a library
# source under .pyenv), writes are blocked on both.
DENY_COMPONENTS_READ: frozenset[str] = DENY_COMPONENTS_SECRET
DENY_COMPONENTS_WRITE: frozenset[str] = frozenset(DENY_COMPONENTS) | DENY_COMPONENTS_SECRET

HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", re.MULTILINE)
# Unanchored, for sniffing a hunk header anywhere inside a payload.
HUNK_ANYWHERE_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@", re.MULTILINE)


# ---------------------------------------------------------------------------
# Errors -- every failure carries the detail an agent needs to self-correct.
# ---------------------------------------------------------------------------


class FeditError(ValueError):
    """An edit could not be applied. The message is written for the model."""


# ---------------------------------------------------------------------------
# Text helpers: a file is modelled as lines + a trailing-newline flag, so that
# "no newline at end of file" survives an edit round-trip.
# ---------------------------------------------------------------------------


def split_text(text: str) -> tuple[list[str], bool]:
    """Split into lines without their terminators, and record the EOF newline."""
    return text.splitlines(), text.endswith("\n")


def join_text(lines: Iterable[str], final_newline: bool) -> str:
    lines = list(lines)
    if not lines:
        return ""
    return "\n".join(lines) + ("\n" if final_newline else "")


# ---------------------------------------------------------------------------
# Unified-diff engine
# ---------------------------------------------------------------------------


@dataclass
class Hunk:
    """One ``@@`` block of a unified diff, plus the header line it came from."""

    old_start: int          # 1-based line number in the old file
    old_count: int
    new_start: int
    new_count: int
    lines: list[tuple[str, str]]   # (' ', text) | ('-', text) | ('+', text)
    header_line: int
    warnings: list[str]
    new_no_eof: bool = False   # hunk ended with "\\ No newline at end of file"
    old_no_eof: bool = False   # same marker attached to a '-' line


def parse_unified_diff(diff_text: str) -> list[Hunk]:
    """Parse a unified diff into hunks. Raises FeditError on malformed input.

    Tolerated on purpose (they are the usual LLM slips, and the body of a hunk
    is what the matcher actually trusts): a hunk whose ``-l,s``/``+l,s`` counts
    disagree with its body (counts are recomputed, a warning is recorded), and
    ``--- / +++`` headers naming a different path than the tool argument (the
    ``path`` argument stays authoritative, a warning is recorded).
    """
    if not diff_text or not diff_text.strip():
        raise FeditError("The diff is empty.")

    raw = diff_text.split("\n")
    if raw and raw[-1] == "":
        raw.pop()

    hunks: list[Hunk] = []
    index = 0
    while index < len(raw):
        line = raw[index]
        match = HUNK_RE.match(line)
        if not match:
            if line.startswith(("--- ", "+++ ", "diff ", "index ", "\\", "new file",
                                "deleted file", "similarity", "rename ", "@@")):
                index += 1
                continue
            if not line.strip():
                index += 1
                continue
            raise FeditError(
                f"Unrecognised diff line {index + 1}: {line!r}. Expected a hunk "
                f"header like '@@ -12,7 +12,9 @@'."
            )

        old_start = int(match.group(1))
        old_count = 1 if match.group(2) is None else int(match.group(2))
        new_start = int(match.group(3))
        new_count = 1 if match.group(4) is None else int(match.group(4))
        header_line = index + 1

        body: list[tuple[str, str]] = []
        new_no_eof = False
        old_no_eof = False
        index += 1
        while index < len(raw):
            line = raw[index]
            if HUNK_RE.match(line) or line.startswith(("--- ", "+++ ", "diff ", "index ")):
                break
            if line.startswith("\\"):
                # "\ No newline at end of file": attached to the line above,
                # so it says whether that side of the hunk ends without a
                # newline. Only meaningful on the last hunk of the file.
                marker_for = body[-1][0] if body else " "
                if marker_for == "+":
                    new_no_eof = True
                elif marker_for == "-":
                    old_no_eof = True
                index += 1
                continue
            marker = line[:1]
            if marker in (" ", "-", "+"):
                body.append((marker, line[1:]))
            elif not line:
                body.append((" ", ""))  # a stripped trailing-whitespace context line
            else:
                raise FeditError(
                    f"Invalid hunk body line {index + 1}: {line!r}. Lines must start "
                    f"with a space (context), '-' (remove) or '+' (add)."
                )
            index += 1

        got_old = sum(1 for kind, _ in body if kind in " -")
        got_new = sum(1 for kind, _ in body if kind in " +")
        warnings: list[str] = []
        if (got_old, got_new) != (old_count, new_count):
            warnings.append(
                f"hunk at line {header_line}: header counts ({old_count},{new_count}) "
                f"disagree with the body ({got_old},{got_new}); the body was used"
            )
        hunks.append(Hunk(old_start, got_old, new_start, got_new, body, header_line,
                          warnings, new_no_eof, old_no_eof))

    if not hunks:
        raise FeditError(
            "No '@@ -l,s +l,s @@' hunk header found. A unified diff needs at least one, "
            "with 3 lines of surrounding context copied verbatim from read_file."
        )
    return hunks


def _old_window(hunk: Hunk, trim_leading: int, trim_trailing: int) -> list[str]:
    """The lines a hunk expects to find in the old file, edges trimmed by fuzz."""
    window = [text for kind, text in hunk.lines if kind in " -"]
    return window[trim_leading: len(window) - trim_trailing if trim_trailing else None]


def _locate(lines: list[str], window: list[str], target: int) -> Optional[int]:
    """Find the index where ``window`` matches, nearest to ``target`` first."""
    if not window:
        return max(0, min(target, len(lines)))
    if len(window) > len(lines):
        return None
    candidates = [i for i in range(len(lines) - len(window) + 1)
                  if lines[i:i + len(window)] == window]
    if not candidates:
        return None
    return min(candidates, key=lambda i: abs(i - target))


def apply_diff_to_text(before: str, diff_text: str,
                       fuzz: Optional[int] = None) -> tuple[str, list[str]]:
    """Apply a unified diff to ``before``, returning (new_text, warnings).

    Each hunk must match exactly at its declared offset; failing that, and if
    ``fuzz`` allows, the hunk is repositioned by matching its context elsewhere
    in the file (``patch``-style fuzzing, edges trimmed first). The operation is
    atomic: one unmatched hunk means nothing is changed anywhere.
    """
    hunks = parse_unified_diff(diff_text)
    max_fuzz = MAX_FUZZ if fuzz is None else max(0, int(fuzz))
    lines, final_newline = split_text(before)
    warnings = [w for hunk in hunks for w in hunk.warnings]

    offset = 0  # accumulated line shift caused by already-applied hunks
    for hunk in hunks:
        applied = False
        for trim in range(0, max_fuzz + 1):
            window = _old_window(hunk, trim, trim)
            target = (hunk.old_start - 1) + offset
            found = _locate(lines, window, target)
            if found is None:
                continue
            new_lines = [text for kind, text in hunk.lines if kind in " +"]
            lines[found:found + len(window)] = new_lines
            # Accumulate both effects of this hunk for the hunks that follow:
            # where it actually landed versus where it was declared (GNU patch
            # tracks the same), and how much it changed the file length.
            offset += (found - target) + (len(new_lines) - len(window))
            if found != target or trim:
                detail = (f"hunk at diff line {hunk.header_line} was declared for "
                          f"line {hunk.old_start} but applied at line {found + 1}")
                if trim:
                    detail += f" with {trim} context line(s) ignored"
                warnings.append(detail)
            applied = True
            break
        if not applied:
            raise FeditError(_hunk_failure(hunk, lines))

    # "\\ No newline at end of file" on a '+' line of the last hunk asks for a
    # file that ends without a newline; on a '-' line it says the old file had
    # none, which the preserved flag already reflects.
    last = hunks[-1]
    if last.new_no_eof:
        final_newline = False
    elif last.old_no_eof and before and not before.endswith("\n"):
        final_newline = True if lines else False

    # A brand-new file gets a final newline; an existing one keeps its own state.
    if not before and lines:
        final_newline = True
    return join_text(lines, final_newline), warnings


def _hunk_failure(hunk: Hunk, lines: list[str]) -> str:
    """A correction-oriented error for a hunk that did not match."""
    window = [text for kind, text in hunk.lines if kind in " -"]
    start = max(0, hunk.old_start - 1)
    actual = lines[start:start + max(len(window), 1)]
    expected = "\n".join(f"    | {line}" for line in window) or "    (empty)"
    got = "\n".join(f"    | {line}" for line in actual) or "    (empty)"
    total = len(lines)
    return (
        f"Hunk at diff line {hunk.header_line} (declared old line {hunk.old_start}, "
        f"{hunk.old_count} line(s)) does not match anywhere in the file "
        f"({total} line(s)), even with fuzzing.\n"
        f"  Expected context:\n{expected}\n"
        f"  Found there:\n{got}\n"
        f"  Fix: re-read the file and copy those lines verbatim into the diff, or "
        f"change the hunk's @@ line number. No changes were written."
    )


# ---------------------------------------------------------------------------
# SEARCH/REPLACE engine
# ---------------------------------------------------------------------------

SEARCH_MARKER = "<<<<<<< SEARCH"
DIVIDER_MARKER = "======="
REPLACE_MARKER = ">>>>>>> REPLACE"


@dataclass
class Block:
    search: list[str]
    replace: list[str]
    block_index: int


def parse_search_replace(edit_text: str) -> list[Block]:
    """Parse one or more SEARCH/REPLACE blocks.

    Format (marker lines may carry trailing labels, which are ignored)::

        <<<<<<< SEARCH
        lines to find, copied verbatim
        =======
        lines to put in their place
        >>>>>>> REPLACE
    """
    if not edit_text or not edit_text.strip():
        raise FeditError("The edit is empty.")
    if SEARCH_MARKER not in edit_text:
        raise FeditError(
            "No '<<<<<<< SEARCH' marker found. Expected a block shaped like:\n"
            "    <<<<<<< SEARCH\n    <lines to find>\n    =======\n"
            "    <replacement>\n    >>>>>>> REPLACE"
        )

    raw = edit_text.split("\n")
    if raw and raw[-1] == "":
        raw.pop()

    def is_marker(line: str, marker: str) -> bool:
        return line.strip() == marker or line.startswith(marker)

    blocks: list[Block] = []
    index = 0
    while index < len(raw):
        if not is_marker(raw[index], SEARCH_MARKER):
            index += 1
            continue
        index += 1
        search: list[str] = []
        while index < len(raw) and not is_marker(raw[index], DIVIDER_MARKER):
            search.append(raw[index])
            index += 1
        if index >= len(raw):
            raise FeditError(
                f"SEARCH block #{len(blocks) + 1} has no '{DIVIDER_MARKER}' divider."
            )
        index += 1
        replace: list[str] = []
        while index < len(raw) and not is_marker(raw[index], REPLACE_MARKER):
            replace.append(raw[index])
            index += 1
        if index >= len(raw):
            raise FeditError(
                f"SEARCH block #{len(blocks) + 1} has no '{REPLACE_MARKER}' terminator."
            )
        index += 1
        blocks.append(Block(search, replace, len(blocks) + 1))

    if not blocks:
        raise FeditError("No complete SEARCH/REPLACE block found.")
    return blocks


def apply_search_replace_to_text(before: str, edit_text: str) -> tuple[str, list[str]]:
    """Apply SEARCH/REPLACE blocks sequentially. Raises FeditError on any problem.

    Each SEARCH must match exactly one place in the text as it stands at that
    step: zero matches is a bad copy, several matches is an under-specified
    block. An empty SEARCH only makes sense for an empty (new) file.
    """
    blocks = parse_search_replace(edit_text)
    lines, final_newline = split_text(before)
    warnings: list[str] = []

    for block in blocks:
        if not block.search:
            if lines:
                raise FeditError(
                    f"Block #{block.block_index}: an empty SEARCH can only create a new "
                    f"file, but the target already has {len(lines)} line(s)."
                )
            lines = list(block.replace)
            continue
        occurrences = [i for i in range(len(lines) - len(block.search) + 1)
                       if lines[i:i + len(block.search)] == block.search]
        if not occurrences:
            raise FeditError(_block_failure(block, lines))
        if len(occurrences) > 1:
            where = ", ".join(str(i + 1) for i in occurrences[:8])
            raise FeditError(
                f"Block #{block.block_index}: SEARCH matches {len(occurrences)} times "
                f"(at lines {where}). Add more surrounding lines so it matches exactly once."
            )
        at = occurrences[0]
        lines[at:at + len(block.search)] = list(block.replace)

    # A brand-new file gets a final newline; an existing one keeps its own state.
    if not before and lines:
        final_newline = True
    return join_text(lines, final_newline), warnings


def _block_failure(block: Block, lines: list[str]) -> str:
    """Point the model at the closest region to its SEARCH text."""
    expected = "\n".join(f"    | {line}" for line in block.search) or "    (empty)"
    window = block.search[:6]
    best, best_ratio = 0, -1.0
    span = max(len(window), 1)
    for i in range(max(1, len(lines) - span + 1)):
        ratio = difflib.SequenceMatcher(None, window, lines[i:i + span]).ratio()
        if ratio > best_ratio:
            best, best_ratio = i, ratio
    near = lines[best:best + max(span, len(block.search))]
    closest = "\n".join(f"    | {line}" for line in near) or "    (empty)"
    return (
        f"Block #{block.block_index}: SEARCH text not found in the file "
        f"({len(lines)} line(s), closest region at line {best + 1}, "
        f"{best_ratio:.0%} similar).\n"
        f"  Searched for:\n{expected}\n"
        f"  Closest content there:\n{closest}\n"
        f"  Fix: re-read the file and copy those lines verbatim (whitespace and "
        f"indentation included). No changes were written."
    )


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------

_FORMAT_ALIASES = {
    "auto": "auto",
    "diff": "diff", "unified": "diff", "patch": "diff", "unified_diff": "diff",
    "search_replace": "sr", "search": "sr", "sr": "sr", "searchreplace": "sr",
}


def resolve_format(fmt: str, edit_text: str) -> str:
    """Map a requested format to 'diff' or 'sr', sniffing when 'auto'."""
    key = _FORMAT_ALIASES.get((fmt or "auto").strip().lower().replace("-", "_"))
    if key is None:
        raise FeditError(
            f"Unknown format {fmt!r}. Use 'search_replace', 'diff', or 'auto'."
        )
    if key != "auto":
        return key
    if SEARCH_MARKER in edit_text:
        return "sr"
    if HUNK_ANYWHERE_RE.search(edit_text):
        return "diff"
    raise FeditError(
        "Could not detect the edit format. Send SEARCH/REPLACE blocks, a unified "
        "diff with an '@@ -l,s +l,s @@' header, or pass format explicitly."
    )


# ---------------------------------------------------------------------------
# Path policy
# ---------------------------------------------------------------------------


class PathDenied(FeditError):
    """The target path is outside what this connector may read or write."""


def _within_root(path: Path) -> bool:
    """True when ``path`` is inside (or equal to) ROOT_DIR; False when unrooted."""
    if ROOT_DIR is None:
        return False
    return path == ROOT_DIR or ROOT_DIR in path.parents


def _component_hit(parts: tuple[str, ...], entries: Iterable[str]) -> Optional[str]:
    """Return the first entry matching ``parts``, or None.

    An entry is either a single component (matched by name) or a sequence such as
    ``.config/gh``, matched against consecutive components of the path -- which is
    how a store like ``~/.config/gh`` is denied without denying all of
    ``~/.config``.
    """
    for entry in entries:
        needle = tuple(entry.split("/"))
        if len(needle) == 1:
            if needle[0] in parts:
                return entry
        else:
            for offset in range(len(parts) - len(needle) + 1):
                if parts[offset:offset + len(needle)] == needle:
                    return entry
    return None


def _deny_reason(resolved: Path, *, for_write: bool) -> Optional[str]:
    """Explain why ``resolved`` is off-limits, or None if it is allowed.

    Reads and writes share the shape of the test but not its list: writing is
    refused under every protected prefix, reading only under the ones that hold
    credentials (``/etc/hosts`` is fair debugging material, ``~/.ssh`` is not).
    """
    text = resolved.as_posix()
    prefixes = DENY_PREFIXES if for_write else DENY_PREFIXES_READ
    for prefix in prefixes:
        if text == prefix or text.startswith(prefix.rstrip("/") + "/"):
            return f"under the protected prefix {prefix!r}"

    components = DENY_COMPONENTS_WRITE if for_write else DENY_COMPONENTS_READ
    hit = _component_hit(resolved.parts, components)
    if hit:
        verb = "write to" if for_write else "read from"
        return f"inside {hit!r}, which this connector never {verb}"

    name = resolved.name.lower()
    if name.startswith(DENY_FILE_PREFIXES):
        return "a secrets/config file (its name starts with '.env')"
    if name in DENY_FILENAMES:
        return f"a credential file ({resolved.name})"
    if name.endswith(DENY_FILE_SUFFIXES):
        return f"a private key or certificate ({resolved.name})"
    for pattern in DENY_PATH_GLOBS:
        if fnmatch.fnmatchcase(text, pattern):
            return f"a browser profile/credential store (it matches {pattern!r})"
    return None


def check_target_path(path: str, *, for_write: bool) -> Path:
    """Resolve one path and apply the policy for the requested operation.

    This is the single entry point every tool goes through, so a read can never
    slip past a check that only the write path performed -- that asymmetry was
    issue #1: the deny-lists guarded ``apply_edit`` while ``read_file`` happily
    returned ``~/.git-credentials``.

    Root scoping: when the server runs with --root-dir, a path whose literal
    and real (symlink-resolved) forms both stay inside that root bypasses the
    policy entirely -- the root is the operator's workspace. Everything below
    only constrains paths outside the root. Relative paths anchor at the root.

    Outside-root paths are made absolute rather than fully resolved, because
    ``..`` is rejected on the *literal* path; ``~/.ssh/../../etc`` would
    otherwise smuggle traversal back in after ``expanduser()``. The real
    (symlink-resolved) target is then checked as well, so a link cannot point
    the connector at a store the path itself does not mention.
    """
    if not path or not path.strip():
        raise FeditError("path is empty")
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = (ROOT_DIR or Path.cwd()) / candidate

    absolute = Path(os.path.abspath(str(candidate)))
    real = absolute.resolve()

    # Inside the root the policy is bypassed; the operator chose that tree.
    # Both the literal path and its symlink-resolved target must be inside, so
    # a symlink cannot smuggle an outside file into the bypass.
    if _within_root(absolute) and _within_root(real):
        return absolute

    action = "written" if for_write else "read"
    if ".." in candidate.parts:
        raise FeditError(f"path {path!r} contains '..'")

    # A path is judged both as given and as the file it really is.
    for label, target in (("the path", absolute), ("its real target", real)):
        reason = _deny_reason(target, for_write=for_write)
        if reason:
            detail = "" if target == absolute else f" ({label} {real})"
            raise PathDenied(
                f"{absolute}{detail} may not be {action} by this connector: "
                f"it is {reason}. Call edit_policy() for the full list."
            )
    return absolute


def check_read_path(path: str) -> Path:
    """Resolve a path for reading, applying the read policy.

    Existence is deliberately checked *after* the policy: answering "not found"
    for a path inside ``~/.ssh`` would itself disclose what a secret store holds.
    """
    resolved = check_target_path(path, for_write=False)
    if not resolved.exists():
        raise FeditError(f"File not found: {resolved}")
    if not resolved.is_file():
        raise FeditError(f"Not a regular file: {resolved}")
    return resolved


def check_write_path(path: str) -> Path:
    """Resolve a path for writing, applying the write policy.

    The parent directory must exist and be writable: creating whole new directory
    trees is left to the console connector (``mkdir``), where it is visible.
    """
    resolved = check_target_path(path, for_write=True)
    if resolved.is_dir():
        raise FeditError(f"{resolved} is a directory, not a file")
    if not os.access(resolved.parent, os.W_OK):
        raise PathDenied(f"Directory {resolved.parent} is not writable")
    return resolved


def check_preview_path(path: str) -> Path:
    """Policy-check a path an edit would target, whether or not it exists yet.

    A dry run is judged by the rules of the write it previews: showing the diff
    against ``~/.ssh/config`` discloses that file just as plainly as editing it
    would, so ``preview_edit`` must not get a more permissive check than
    ``apply_edit``.
    """
    resolved = check_target_path(path, for_write=True)
    if resolved.is_dir():
        raise FeditError(f"{resolved} is a directory, not a file")
    return resolved


def check_listing_path(path: str) -> Path:
    """Policy-check the path whose backups are being listed.

    Backups hold the file's full content, so listing them is a read and is held
    to the read policy.
    """
    return check_target_path(path, for_write=False)


# ---------------------------------------------------------------------------
# Backups -- a real file is never rewritten without a restorable copy.
# ---------------------------------------------------------------------------


def backup_path_for(resolved: Path) -> Path:
    digest = hashlib.sha1(str(resolved).encode("utf-8")).hexdigest()[:12]
    safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", resolved.name)[:60]
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S.%f")
    return BACKUP_DIR / f"{digest}_{safe_name}_{stamp}.orig"


def list_backups(resolved: Path) -> list[Path]:
    digest = hashlib.sha1(str(resolved).encode("utf-8")).hexdigest()[:12]
    if not BACKUP_DIR.is_dir():
        return []
    return sorted(BACKUP_DIR.glob(f"{digest}_*.orig"))


def make_backup(resolved: Path) -> Path:
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    destination = backup_path_for(resolved)
    shutil.copy2(resolved, destination)
    existing = list_backups(resolved)
    for stale in existing[:-BACKUP_KEEP] if len(existing) > BACKUP_KEEP else []:
        try:
            stale.unlink()
        except OSError:
            pass
    return destination


def write_atomic(resolved: Path, content: str) -> None:
    """Write via a temp file in the same directory, then rename over the target."""
    handle, tmp_name = tempfile.mkstemp(
        dir=str(resolved.parent), prefix=f".{resolved.name}.", suffix=".fedit-tmp"
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="") as handle_file:
            handle_file.write(content)
        os.replace(tmp_name, resolved)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@server.tool()
def read_file(path: str, start_line: int = 1, max_lines: int = 400) -> str:
    """Read a file with line numbers, so an edit can quote its context exactly.

    Always call this before preview_edit/apply_edit, and copy the text after the
    tab verbatim (indentation included) into your SEARCH or diff block.

    Args:
        path: File to read
        start_line: 1-based line to start from (default: 1)
        max_lines: How many lines to return (default: 400)

    Returns:
        A header line, then one ``<lineno>\\t<text>`` line per line

    Example:
        read_file("/srv/app/main.py", start_line=40, max_lines=60)
    """
    try:
        resolved = check_read_path(path)
        size = resolved.stat().st_size
        if size > MAX_FILE_BYTES:
            return f"Error: {resolved} is {size} bytes, over the {MAX_FILE_BYTES} limit"
        text = resolved.read_text(encoding="utf-8")
    except (FeditError, UnicodeDecodeError, OSError) as exc:
        return f"Error: {exc}"

    lines, final_newline = split_text(text)
    start = max(1, int(start_line))
    if start > len(lines):
        return f"Error: start_line {start} is past the end of a {len(lines)}-line file"
    chunk = lines[start - 1:start - 1 + max(1, int(max_lines))]
    numbered = "\n".join(f"{start + i:>6}\t{line}" for i, line in enumerate(chunk))
    end = start + len(chunk) - 1
    eof = "newline" if final_newline else "NO final newline"
    header = (f"File: {resolved}  lines {start}-{end} of {len(lines)}  "
              f"{size} bytes  EOF: {eof}")
    if end < len(lines):
        header += f"  (continues at line {end + 1})"
    return f"{header}\n{numbered}"


@server.tool()
def preview_edit(path: str, edit: str, format: str = "auto", fuzz: Optional[int] = None) -> str:
    """Dry-run an edit: validates and shows the resulting diff, writes nothing.

    Args:
        path: File that would be edited
        edit: A unified diff, or SEARCH/REPLACE blocks (see apply_edit)
        format: 'auto' (detect), 'search_replace' or 'diff'
        fuzz: Max context lines a hunk may ignore when repositioning (0..3)

    Returns:
        The diff between the current and the proposed content, or the error

    Example:
        preview_edit("/srv/app/main.py", edit)
    """
    try:
        resolved = check_preview_path(path)
        before = resolved.read_text(encoding="utf-8") if resolved.is_file() else ""
        after, warnings = run_edit(resolved, before, edit, format, fuzz)
    except (FeditError, UnicodeDecodeError, OSError) as exc:
        return f"Error: {exc}"

    if before == after:
        return f"VALID but a no-op: {resolved} would be unchanged."
    diff = "\n".join(difflib.unified_diff(
        before.splitlines(), after.splitlines(),
        fromfile=f"a/{resolved.name}", tofile=f"b/{resolved.name}", lineterm="", n=2))
    report = [f"PREVIEW (not written) for {resolved}"]
    if warnings:
        report.append("Warnings:\n" + "\n".join(f"  - {w}" for w in warnings))
    report.append(diff)
    return "\n\n".join(report)


def run_edit(resolved: Path, before: str, edit: str,
             fmt: str, fuzz: Optional[int]) -> tuple[str, list[str]]:
    """Dispatch one edit payload through the right engine."""
    if len(edit.encode("utf-8")) > MAX_EDIT_BYTES:
        raise FeditError(f"edit payload is over {MAX_EDIT_BYTES} bytes")
    kind = resolve_format(fmt, edit)
    if kind == "sr":
        return apply_search_replace_to_text(before, edit)
    return apply_diff_to_text(before, edit, fuzz=fuzz)


@server.tool()
def apply_edit(path: str, edit: str, format: str = "auto", fuzz: Optional[int] = None) -> str:
    """Apply an edit to a file in place. A backup is taken first (see revert_edit).

    Two accepted formats, whichever you pass with ``format`` (default 'auto'):

    SEARCH/REPLACE -- quote the file verbatim, once each::

        <<<<<<< SEARCH
        def greet(name):
            return f"hi {name}"
        =======
        def greet(name: str) -> str:
            return f"hello, {name}"
        >>>>>>> REPLACE

    Unified diff -- 3 lines of context around each change, no line numbers needed
    (hunks are repositioned by their context, GNU-patch style)::

        --- a/path/to/file
        +++ b/path/to/file
        @@ -12,7 +12,7 @@
         unchanged context
        -old line
        +new line
         unchanged context

    Rules: the edit is matched against the file as it is now; whitespace counts;
    several SEARCH/REPLACE blocks are applied in order; if any part fails, nothing
    is written at all; the file's final-newline state is preserved.

    Args:
        path: File to edit (parent directory must exist)
        edit: The diff or SEARCH/REPLACE payload
        format: 'auto' (detect), 'search_replace' or 'diff'
        fuzz: Max context lines a hunk may ignore when repositioning (0..3)

    Returns:
        The diff that was applied, plus the backup path for reverting

    Example:
        apply_edit("/srv/app/main.py", edit, format="search_replace")
    """
    try:
        resolved = check_write_path(path)
        creating = not resolved.is_file()
        if creating:
            before = ""
        else:
            if resolved.stat().st_size > MAX_FILE_BYTES:
                return f"Error: {resolved} is over the {MAX_FILE_BYTES} byte limit"
            before = resolved.read_text(encoding="utf-8")
        after, warnings = run_edit(resolved, before, edit, format, fuzz)
    except (PathDenied, FeditError, UnicodeDecodeError, OSError) as exc:
        return f"Error: {exc}"

    if before == after:
        return f"No change: {resolved} already matches the edit. Nothing was written."

    try:
        backup_note = ""
        if not creating:
            backup = make_backup(resolved)
            backup_note = f"\nBackup: {backup}  (restore with revert_edit)"
        write_atomic(resolved, after)
    except OSError as exc:
        return f"Error: could not write {resolved}: {exc}"

    diff = "\n".join(difflib.unified_diff(
        before.splitlines(), after.splitlines(),
        fromfile=f"a/{resolved.name}", tofile=f"b/{resolved.name}", lineterm="", n=1))
    lines_before = len(before.splitlines())
    lines_after = len(after.splitlines())
    report = [f"Applied to {resolved}: {lines_before} -> {lines_after} line(s)"]
    if warnings:
        report.append("Warnings:\n" + "\n".join(f"  - {w}" for w in warnings))
    report.append(diff)
    if backup_note:
        report.append(backup_note.strip())
    return "\n\n".join(report)


@server.tool()
def list_edits(path: str) -> str:
    """List the backups kept for a file, oldest first (newest is revert_edit's).

    Args:
        path: The edited file

    Returns:
        One line per backup, numbered from 1 (oldest)

    Example:
        list_edits("/srv/app/main.py")
    """
    try:
        resolved = check_listing_path(path)
    except FeditError as exc:
        return f"Error: {exc}"
    backups = list_backups(resolved)
    if not backups:
        return f"No backup for {resolved} (it has not been edited by this server)"
    rows = [f"{i}. {item.name}  {item.stat().st_size} bytes"
            for i, item in enumerate(backups, start=1)]
    return f"{len(backups)} backup(s) for {resolved} (newest first is #{len(backups)}):\n" + \
           "\n".join(rows)


@server.tool()
def revert_edit(path: str) -> str:
    """Restore a file to its state before the most recent apply_edit.

    The current content is itself backed up first, so a revert can be reverted.

    Args:
        path: The file to restore

    Returns:
        What was restored, or the available backups when there is none

    Example:
        revert_edit("/srv/app/main.py")
    """
    try:
        resolved = check_write_path(path)
    except FeditError as exc:
        return f"Error: {exc}"
    backups = list_backups(resolved)
    if not backups:
        return f"Error: no backup for {resolved} — nothing to revert"

    newest = backups[-1]
    try:
        restored = newest.read_text(encoding="utf-8")
        current_note = ""
        if resolved.is_file():
            current_note = f"\nPre-revert copy: {make_backup(resolved)}"
            write_atomic(resolved, restored)
        else:
            write_atomic(resolved, restored)
        newest.unlink()
    except (UnicodeDecodeError, OSError) as exc:
        return f"Error: could not revert {resolved}: {exc}"
    return (f"Reverted {resolved} to its content from {newest.stem.rsplit('_', 1)[-1]}"
            f" ({len(restored.splitlines())} lines).{current_note}")


@server.tool()
def make_diff(path_a: str, path_b: str, context: int = 3) -> str:
    """Diff two files (unified), for comparing versions before choosing an edit.

    Args:
        path_a: Old file
        path_b: New file
        context: Context lines per hunk (default: 3)

    Returns:
        The unified diff, or a note when the files are identical

    Example:
        make_diff("/srv/app/main.py", "/tmp/main.py.rejected")
    """
    try:
        a = check_read_path(path_a)
        b = check_read_path(path_b)
        text_a = a.read_text(encoding="utf-8")
        text_b = b.read_text(encoding="utf-8")
    except (FeditError, UnicodeDecodeError, OSError) as exc:
        return f"Error: {exc}"
    if text_a == text_b:
        return f"{a} and {b} are identical."
    return "\n".join(difflib.unified_diff(
        text_a.splitlines(), text_b.splitlines(),
        fromfile=str(a), tofile=str(b), lineterm="", n=max(0, int(context))))


@server.tool()
def edit_policy() -> str:
    """Report the connector's own limits and protected paths (no guessing needed).

    Returns:
        Backup directory, size caps, fuzz level and the deny-lists, separated
        into what reads and writes are each refused

    Example:
        edit_policy()
    """
    return (
        f"fedit-mcp policy\n"
        f"  root dir        : {ROOT_DIR if ROOT_DIR is not None else '(unset -- full policy on every path)'}\n"
        f"  inside the root : no policy restrictions (paths fully bypass the lists below)\n"
        f"  backup dir      : {BACKUP_DIR.resolve()}\n"
        f"  backups kept    : {BACKUP_KEEP}\n"
        f"  max file bytes  : {MAX_FILE_BYTES}\n"
        f"  max edit bytes  : {MAX_EDIT_BYTES}\n"
        f"  default fuzz    : {MAX_FUZZ}\n"
        f"  denied to READS  (prefixes) : {', '.join(DENY_PREFIXES_READ) or '(none)'}\n"
        f"  denied to READS  (parts)    : {', '.join(sorted(DENY_COMPONENTS_READ))}\n"
        f"  denied to WRITES (prefixes) : {', '.join(DENY_PREFIXES)}\n"
        f"  denied to WRITES (parts)    : {', '.join(sorted(DENY_COMPONENTS_WRITE))}\n"
        f"  denied names    : starts with {', '.join(DENY_FILE_PREFIXES)}, or one of "
        f"{', '.join(sorted(DENY_FILENAMES))}\n"
        f"  denied suffixes : {', '.join(DENY_FILE_SUFFIXES)}\n"
        f"  denied globs    : {', '.join(DENY_PATH_GLOBS) or '(none)'}\n"
        f"  (all lists overridable through FEDIT_DENY_* environment variables)"
    )


def main():
    parser = argparse.ArgumentParser(
        description="File Edit MCP Server - diff and SEARCH/REPLACE edits via MCP"
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=DEFAULT_PORT,
        help=f"Port to listen on (default: {DEFAULT_PORT})"
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
        help="Root directory the connector starts in and is scoped to (required). "
             "Files and directories inside it bypass the path policy; paths "
             "outside it are checked against the protected-path lists."
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
    print(f"🚀 File Edit MCP Server starting...", file=banner)
    if args.transport == "stdio":
        print(f"   Transport: stdio", file=banner)
    else:
        print(f"   Streamable HTTP   : http://{args.host}:{args.port}/mcp", file=banner)
    print(f"   Root directory: {ROOT_DIR}", file=banner)
    print(f"   Backup dir    : {BACKUP_DIR}", file=banner)
    print(f"   Max file size : {MAX_FILE_BYTES} bytes", file=banner)

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
        print("\nFile Edit MCP Server stopped.", file=banner)


if __name__ == "__main__":
    sys.exit(main())
