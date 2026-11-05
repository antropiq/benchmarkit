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

Two regions are trusted outright, and the checks below only apply outside them:

* the root directory (``--root-dir``) and everything under it -- that tree is
  the operator's own workspace, so the *deny-lists* below do not apply inside
  it, including to the connector's own sources;
* scratch directories (``/tmp``, ``/var/tmp``; see ``ALLOW_PREFIXES``) --
  throwaway work, never the operator's secrets.

Inside the root, one policy does apply: the workspace's own
``<root>/.nojae/read-write-policy.json`` (pol-08). A folder its owner marked
``ro`` is uneditable here exactly as it is in the Editor pane, and one marked
``hidden`` is not readable either -- the same file, the same matcher and the
same refusal sentence, read in this process. Narrowing the agent to what the
human can see is the point; nothing is added anywhere.

The two rule sources stay **separate lists** on purpose, and a reader tempted
to merge them will break both. The deny-list answers "is this a
credential store" and that question means the same thing in every tree, which
is why it must keep guarding the paths *outside* the root where it is the only
rule that speaks. The workspace policy answers "which parts of *this* project
may change" -- a question the operator wrote down, and one the deny-list cannot
ask. Note the resulting asymmetry rather than "fixing" it: an ordinary
workspace ``.env`` is editable through this connector while the Editor pane
cannot read it, because the pane's rule is level A (benchmarkit regenerates its
own ``.env`` from the settings template), which is a fact about the pane, not
about an agent's edit. ``edit_policy()`` reports both halves.

Both bypasses are decided on where a path *really* is (``realpath``), not on the
spelling that reached it: one physical file therefore gets one verdict however it
was addressed -- the same tree reached through a mount point and its alias
behaves identically. A symlink cannot smuggle a protected file into a trusted
prefix either, because the link's *target* is what is tested. Relative paths are
anchored at the root.

Traversal is judged the same way: the path is normalised and every form of it is
held to the deny rules, so ``..`` is refused for arriving somewhere protected
rather than for being that token -- which keeps ``logs/x/../src/y.py`` (an
ordinary in-root path) working.

Usage:
    python -m mcp_connectors.fedit_mcp --root-dir /path/to/root              # default port 8946
    python -m mcp_connectors.fedit_mcp --root-dir /path --port 8080          # custom port
    python -m mcp_connectors.fedit_mcp --root-dir /path --transport stdio    # stdio (for MCP hubs)
"""

import argparse
import difflib
import hashlib
import os
import re
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

from mcp.server.mcpserver import MCPServer

try:  # imported as part of the package (tests, `python -m mcp_connectors...`)
    from . import path_policy, policy_file, policy_shared
except ImportError:  # run as a plain script path, as externals.json does
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import path_policy  # type: ignore[no-redef]
    import policy_file  # type: ignore[no-redef]
    import policy_shared  # type: ignore[no-redef]

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
                 "obey that folder's own workspace policy (a read-only or hidden folder is "
                 "refused, naming the rule); paths outside it are judged against the "
                 "protected-path lists. Both sets of rules are reported by edit_policy().",
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

# Absolute ceiling on fuzz, for both the default and a per-call request. A big
# fuzz stops being "the file shifted a little" and becomes "match this
# anywhere" -- with enough slack a hunk's whole context is trimmed away and the
# edit lands in whatever function happens to share its '-' lines.
FUZZ_CEILING = 3

# Number of backups kept per file; older ones are pruned.
BACKUP_KEEP = int(os.environ.get("FEDIT_BACKUP_KEEP", 10))

# Backups older than this are swept regardless of their file, because a file
# that was renamed or deleted leaves backups nobody will ever ask for (the
# per-file keep above cannot see them). 0 disables the sweep.
BACKUP_PRUNE_DAYS = int(os.environ.get("FEDIT_BACKUP_PRUNE_DAYS", 30))

# Where pre-edit copies are stored. Kept out of the edited file's directory on
# purpose: backups must not pollute a working tree or show up in ``git status``.
BACKUP_DIR = Path(os.environ.get("FEDIT_BACKUP_DIR", "logs/fedit_backups"))

# Prefixes that are trusted outright, like the root directory: scratch space is
# where an agent puts throwaway work, so a path that stays inside one of these
# (literally and after symlink resolution) skips the policy below too.
# The protected-path lists live in ONE module shared with console_mcp. Keeping a
# single copy is the point: while fedit refused ``~/.ssh`` and console ran
# ``cat .env``, the careful policy here was bypassable through the sibling
# connector enabled in the same session. These names are re-exported so
# ``edit_policy()`` and the tests keep seeing them on this module.
ALLOW_PREFIXES: tuple[str, ...] = path_policy.ALLOW_PREFIXES
DENY_PREFIXES: tuple[str, ...] = path_policy.DENY_PREFIXES
DENY_PREFIXES_READ: tuple[str, ...] = path_policy.DENY_PREFIXES_READ
DENY_COMPONENTS: frozenset[str] = path_policy.DENY_COMPONENTS
DENY_COMPONENTS_SECRET: frozenset[str] = path_policy.DENY_COMPONENTS_SECRET
DENY_FILE_PREFIXES: tuple[str, ...] = path_policy.DENY_FILE_PREFIXES
DENY_FILENAMES: frozenset[str] = path_policy.DENY_FILENAMES
DENY_FILE_SUFFIXES: tuple[str, ...] = path_policy.DENY_FILE_SUFFIXES
DENY_PATH_GLOBS: tuple[str, ...] = path_policy.DENY_PATH_GLOBS
DENY_COMPONENTS_READ: frozenset[str] = path_policy.DENY_COMPONENTS_READ
DENY_COMPONENTS_WRITE: frozenset[str] = path_policy.DENY_COMPONENTS_WRITE

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
# "no newline at end of file" survives an edit round-trip. The terminator
# *style* is part of that model too: splitting on ``str.splitlines()`` and
# re-joining with "\n" would rewrite every CRLF file as LF (and turn a form
# feed into a line break), which shows up as whole-file churn in ``git diff``.
# ---------------------------------------------------------------------------


# Only these three end a line. ``str.splitlines()`` also splits on \\x0b \\x0c
# \\x1c-\\x1e \\x85 \\u2028 \\u2029, which would invent lines a text editor does
# not have -- and could not be put back, so those characters stay line content.
_LINE_END_RE = re.compile(r"\r\n|\n|\r")

# Display names for the three terminators, used by the header and the warnings.
TERMINATOR_NAMES = {"\r\n": "CRLF", "\n": "LF", "\r": "CR"}


def detect_terminator(text: str) -> str:
    """The terminator a file is written with: "\\r\\n", "\\n" or "\\r".

    The style used by most lines wins, so a CRLF file stays CRLF and a
    classic-Mac file (lone "\\r" only) stays "\\r". A file that mixes styles is
    normalised to its dominant one -- unavoidable once lines are matched by
    their text; ``_terminator_warning`` says so out loud.
    """
    found = _LINE_END_RE.findall(text)
    if not found:
        return "\n"
    counts = {style: found.count(style) for style in ("\r\n", "\n", "\r")}
    # max() breaks ties towards the first entry, i.e. the most specific style.
    return max(("\r\n", "\n", "\r"), key=lambda style: counts[style])


# How much of a file is looked at when the answer only needs to be indicative.
SNIFF_BYTES = 8192

# A UTF-8 BOM is encoding metadata, not text: it is stripped before lines are
# matched and re-attached on write, so the first line of a BOM'd file can be
# copied out of read_file and back into a SEARCH block verbatim.
UTF8_BOM = "\ufeff"


def strip_bom(text: str) -> tuple[str, bool]:
    """The text without its leading UTF-8 BOM, and whether it had one."""
    if text.startswith(UTF8_BOM):
        return text[1:], True
    return text, False


def split_text(text: str) -> tuple[list[str], bool, str]:
    """Split into lines without their terminators, plus the EOF newline flag
    and the terminator style used when writing the lines back."""
    lines = _LINE_END_RE.split(text)
    if lines and lines[-1] == "":
        lines.pop()      # the terminator after the last line, not an empty line
    return lines, text.endswith(("\n", "\r")), detect_terminator(text)


def join_text(lines: Iterable[str], final_newline: bool,
              terminator: str = "\n", bom: bool = False) -> str:
    lines = list(lines)
    if not lines:
        return UTF8_BOM if bom else ""
    body = terminator.join(lines) + (terminator if final_newline else "")
    return (UTF8_BOM + body) if bom else body


def binary_reason(payload: bytes) -> Optional[str]:
    """Why ``payload`` is a binary file, or None when it is not.

    A NUL byte is what ``file(1)`` calls binary. It is the one property that
    makes a file useless to *every* text tool, so it is refused everywhere --
    showing such content as lines only invites the agent to copy a line that
    cannot be copied back. The whole payload is checked, not just a leading
    window: a NUL at offset 9000 breaks an edit just as surely.
    """
    nul = payload.find(b"\x00")
    if nul == -1:
        return None
    return f"it contains a NUL byte at offset {nul}"


def undecodable_reason(payload: bytes) -> Optional[str]:
    """Why ``payload`` is not valid UTF-8, or None when it is.

    Text in another encoding (Latin-1, UTF-16, Shift-JIS) is real text: it can
    be displayed with the bad bytes replaced, it just cannot be rewritten,
    because the edit engine's output would no longer round-trip those bytes.
    Naming the byte and its offset is what makes the answer actionable instead
    of a codec exception the model cannot act on.
    """
    try:
        payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        return (f"byte 0x{payload[exc.start]:02x} at offset {exc.start} is not a "
                f"valid UTF-8 sequence")
    return None


def not_editable_reason(payload: bytes) -> Optional[str]:
    """Why ``payload`` cannot be edited by this connector, or None if it can."""
    return binary_reason(payload) or undecodable_reason(payload)


def not_editable_message(resolved: Path, reason: str) -> str:
    """The refusal an agent can act on: this file is not editable text."""
    return (
        f"{resolved} is not editable text ({reason}); fedit edits UTF-8 text "
        f"files only. Inspect it through console, e.g. ``file {resolved}`` or "
        f"``head -c {SNIFF_BYTES} {resolved}``."
    )


def read_payload_for_edit(resolved: Path) -> str:
    """Read a file an edit is about to rewrite, refusing anything not text.

    A NUL byte or an undecodable byte makes an edit unrecoverable: the reported
    diff cannot be round-tripped, so the request is refused with the reason
    rather than answered with a codec exception.
    """
    payload = resolved.read_bytes()
    reason = not_editable_reason(payload)
    if reason:
        raise FeditError(not_editable_message(resolved, reason))
    return payload.decode("utf-8")


def _terminator_warning(text: str) -> Optional[str]:
    """Warn when a file mixes terminator styles, which the write normalises.

    Matching is line-based, so a mixed file cannot be reproduced byte for byte;
    the alternative (silently rewriting half its line endings) is worse, so the
    normalisation is reported in the tool's warning list.
    """
    styles = set(_LINE_END_RE.findall(text))
    if len(styles) <= 1:
        return None
    names = ", ".join(sorted(TERMINATOR_NAMES[s] for s in styles))
    chosen = TERMINATOR_NAMES[detect_terminator(text)]
    return (f"this file mixes line terminators ({names}); every line ending is "
            f"rewritten as {chosen}")


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


def apply_diff_to_text(before: str, diff_text: str, fuzz: Optional[int] = None,
                       creating: bool = False) -> tuple[str, list[str]]:
    """Apply a unified diff to ``before``, returning (new_text, warnings).

    Each hunk must match exactly at its declared offset; failing that, and if
    ``fuzz`` allows, the hunk is repositioned by matching its context elsewhere
    in the file (``patch``-style fuzzing, edges trimmed first). The operation is
    atomic: one unmatched hunk means nothing is changed anywhere.

    Fuzz is clamped to ``FUZZ_CEILING``, and never trims a hunk's context away
    entirely: an empty window matches every position in the file, so a hunk
    trimmed that far is an edit dropped at an arbitrary line rather than a
    repositioned one.

    ``creating`` says the target has no bytes because the file does not exist
    (as opposed to existing and being empty), so a failed hunk can say that
    instead of sending the model off to re-read a file that is not there.
    """
    hunks = parse_unified_diff(diff_text)
    requested = MAX_FUZZ if fuzz is None else max(0, int(fuzz))
    max_fuzz = min(FUZZ_CEILING, requested)
    max_fuzz_note = (f"fuzz={requested} is above the {FUZZ_CEILING} ceiling and "
                     f"was reduced to {max_fuzz}" if requested > max_fuzz else "")
    body, bom = strip_bom(before)
    lines, final_newline, terminator = split_text(body)
    warnings = [w for hunk in hunks for w in hunk.warnings]
    if max_fuzz_note:
        warnings.append(max_fuzz_note)
    mixed = _terminator_warning(body)
    if mixed:
        warnings.append(mixed)

    offset = 0  # accumulated line shift caused by already-applied hunks
    for hunk in hunks:
        applied = False
        full_window = _old_window(hunk, 0, 0)
        # Trimming both edges past the middle would leave nothing to match on.
        deepest = max(0, (len(full_window) - 1) // 2)
        for trim in range(0, min(max_fuzz, deepest) + 1):
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
            # A hunk that declares no old content has nothing to match on, so
            # _locate only clamps its declared line into the file; that clamp is
            # not a hunk that moved, and reporting it would accuse a creation of
            # having been applied at the wrong place.
            if (found != target and window) or trim:
                detail = (f"hunk at diff line {hunk.header_line} was declared for "
                          f"line {hunk.old_start} but applied at line {found + 1}")
                if trim:
                    detail += f" with {trim} context line(s) ignored"
                warnings.append(detail)
            applied = True
            break
        if not applied:
            raise FeditError(_hunk_failure(hunk, lines, clamped_note=max_fuzz_note,
                                           creating=creating))

    # "\\ No newline at end of file" on a '+' line of the last hunk asks for a
    # file that ends without a newline; on a '-' line it says the old file had
    # none, which the preserved flag already reflects.
    last = hunks[-1]
    if last.new_no_eof:
        final_newline = False
    elif last.old_no_eof and body and not body.endswith(("\n", "\r")):
        final_newline = True if lines else False

    # A brand-new file gets a final newline; an existing one keeps its own state.
    if not before and lines:
        final_newline = True
    return join_text(lines, final_newline, terminator, bom), warnings


def _hunk_failure(hunk: Hunk, lines: list[str],
                  clamped_note: str = "", creating: bool = False) -> str:
    """A correction-oriented error for a hunk that did not match.

    ``clamped_note`` is non-empty when the caller had to reduce the requested
    fuzz: "does not match anywhere, even with fuzzing" would be an invitation to
    raise ``fuzz`` again, when the real reason it did not match is that the fuzz
    asked for is above the ceiling and was not honoured.

    ``creating`` names the one case where "re-read the file" is not a route
    forward: the target does not exist, so there are no lines to copy verbatim,
    and what the model needs is the creation shape -- a hunk that only adds.
    """
    window = [text for kind, text in hunk.lines if kind in " -"]
    start = max(0, hunk.old_start - 1)
    actual = lines[start:start + max(len(window), 1)]
    expected = "\n".join(f"    | {line}" for line in window) or "    (empty)"
    got = "\n".join(f"    | {line}" for line in actual) or "    (empty)"
    total = len(lines)
    fuzz_note = (f"  Note: {clamped_note}; raising it further cannot help -- "
                 f"fix the hunk's context instead.\n" if clamped_note else "")
    creation_note = (
        f"  This file does not exist yet, so there is nothing to match: a hunk "
        f"cannot change or quote lines of a file that is absent. To create it, "
        f"send a hunk that only adds lines (@@ -0,0 +1,N @@) with every line "
        f"prefixed by '+', or a SEARCH/REPLACE block whose SEARCH is empty.\n"
        if creating else "")
    return (
        f"Hunk at diff line {hunk.header_line} (declared old line {hunk.old_start}, "
        f"{hunk.old_count} line(s)) does not match anywhere in the file "
        f"({total} line(s)).\n"
        f"  Expected context:\n{expected}\n"
        f"  Found there:\n{got}\n"
        f"{fuzz_note}"
        f"{creation_note}"
        f"  Fix: re-read the file and copy those lines verbatim into the diff, or "
        f"change the hunk's @@ line number. No changes were written."
    )


# ---------------------------------------------------------------------------
# SEARCH/REPLACE engine
# ---------------------------------------------------------------------------

SEARCH_MARKER = "<<<<<<< SEARCH"
DIVIDER_MARKER = "======="
REPLACE_MARKER = ">>>>>>> REPLACE"
BLOCK_MARKERS: tuple[str, ...] = (SEARCH_MARKER, DIVIDER_MARKER, REPLACE_MARKER)


def is_marker_line(line: str, marker: str) -> bool:
    """True when ``line`` reads as ``marker`` (a trailing label is tolerated)."""
    return line.strip() == marker or line.startswith(marker)


def marker_lines(lines: list[str]) -> list[tuple[int, str]]:
    """``(index, marker)`` for every line of ``lines`` that tokenises as a marker.

    A body line that reads as a marker -- a git conflict ``=======``, or
    documentation about this very format -- silently re-splits the payload, and
    the edit then lands somewhere its author never pointed at. Counting the
    markers first is what turns that silent corruption into a refusal.
    """
    found: list[tuple[int, str]] = []
    for index, line in enumerate(lines):
        for marker in BLOCK_MARKERS:
            if is_marker_line(line, marker):
                found.append((index, marker))
                break
    return found


def marker_in_file_warning(before: str) -> Optional[str]:
    """Warn when the target file holds marker-looking lines SEARCH cannot quote.

    The edit itself applied, so this is a warning rather than a refusal -- but
    the next block that tries to copy one of those lines verbatim cannot
    express it, and the agent should know that before it writes one.
    """
    hits = [line.strip()[:20] for line in split_text(before)[0]
            if any(is_marker_line(line, marker) for marker in BLOCK_MARKERS)]
    if not hits:
        return None
    return (f"this file contains {len(hits)} line(s) that read as SEARCH/REPLACE "
            f"markers ({', '.join(dict.fromkeys(hits))}); such a line cannot be "
            f"quoted inside a SEARCH or REPLACE body -- keep it out of those "
            f"bodies, or use format=\"diff\", which tokenises no markers")


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

    # Tokenise the whole payload, then check its shape before trusting any of
    # it. A well-formed payload is a whole number of SEARCH / DIVIDER / REPLACE
    # markers in that order; anything else means a body line was tokenised as a
    # marker -- a git conflict's '=======', or a file documenting this very
    # format. Applying such a payload used to be silently accepted, and the
    # edit landed where nobody asked for: the divider inside the SEARCH text
    # became the block's own divider and the rest of the body became a REPLACE.
    # Refusing is the honest answer, and it names the format that does handle
    # that content (see ``apply_edit``: a diff tokenises no markers).
    markers = marker_lines(raw)
    shape_error = payload_shape_error(markers)
    if shape_error:
        raise FeditError(shape_error)

    blocks: list[Block] = []
    for position in range(0, len(markers), 3):
        search_at = markers[position][0]
        divider_at = markers[position + 1][0]
        replace_at = markers[position + 2][0]
        blocks.append(Block(raw[search_at + 1:divider_at],
                            raw[divider_at + 1:replace_at],
                            position // 3 + 1))
    return blocks


def payload_shape_error(markers: list[tuple[int, str]]) -> Optional[str]:
    """Why a tokenised payload does not describe complete blocks, or None.

    ``markers`` is what ``marker_lines`` found. A valid payload reads as
    SEARCH, DIVIDER, REPLACE, SEARCH, DIVIDER, REPLACE ... Any deviation means
    one marker-looking line belongs to the file's own text rather than to the
    block structure -- so no block boundary in the payload is knowable, and an
    edit built from them is a guess.
    """
    cycle = (SEARCH_MARKER, DIVIDER_MARKER, REPLACE_MARKER)
    if not markers:
        return (f"No complete SEARCH/REPLACE block found: there is a "
                f"'{SEARCH_MARKER}' line but no matching '{DIVIDER_MARKER}' and "
                f"'{REPLACE_MARKER}' pair. No changes were written.")
    for position, (line_no, marker) in enumerate(markers):
        expected = cycle[position % len(cycle)]
        if marker != expected:
            return _shape_message(
                f"'{marker}' at payload line {line_no + 1} is where a "
                f"'{expected}' belongs")
    if len(markers) % len(cycle):
        missing = cycle[len(markers) % len(cycle)]
        return _shape_message(
            f"the payload ends after '{markers[-1][1]}' (payload line "
            f"{markers[-1][0] + 1}), so a '{missing}' is missing")
    return None


def _shape_message(detail: str) -> str:
    """The shared tail of every shape error: what the agent should do instead."""
    return (
        f"Ambiguous SEARCH/REPLACE payload: {detail}. A line copied out of the "
        f"file can itself look like a marker -- a git conflict '=======', "
        f"'<<<<<<<' or '>>>>>>>' line, or documentation about this format -- and "
        f"then the block boundaries are not knowable, so nothing is applied. "
        f"For a file containing such lines use format=\"diff\": a unified diff "
        f"never tokenises markers. No changes were written."
    )


def apply_search_replace_to_text(before: str, edit_text: str,
                                 creating: bool = False) -> tuple[str, list[str]]:
    """Apply SEARCH/REPLACE blocks sequentially. Raises FeditError on any problem.

    Each SEARCH must match exactly one place in the text as it stands at that
    step: zero matches is a bad copy, several matches is an under-specified
    block. An empty SEARCH only makes sense for an empty (new) file.

    ``creating`` says the target has no bytes because the file does not exist,
    so a SEARCH that matches nothing can name the creation shape instead of
    telling the model to re-read a file that is not there.
    """
    blocks = parse_search_replace(edit_text)
    body, bom = strip_bom(before)
    lines, final_newline, terminator = split_text(body)
    warnings: list[str] = []
    mixed = _terminator_warning(body)
    if mixed:
        warnings.append(mixed)
    file_markers = marker_in_file_warning(body)
    if file_markers:
        warnings.append(file_markers)

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
            raise FeditError(_block_failure(block, lines, creating=creating))
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
    return join_text(lines, final_newline, terminator, bom), warnings


def _block_failure(block: Block, lines: list[str], creating: bool = False) -> str:
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
    # "Re-read the file" is the wrong instruction when there is no file: it
    # sends the model to read_file, which answers File not found, and the same
    # edit comes back round again. Name the shape that does create one.
    creation_note = (
        f"  This file does not exist yet, so there is nothing to search for. To "
        f"create it, send one block with an empty SEARCH and the whole new "
        f"content in REPLACE:\n"
        f"    <<<<<<< SEARCH\n"
        f"    =======\n"
        f"    (every line of the new file)\n"
        f"    >>>>>>> REPLACE\n"
        if creating else "")
    return (
        f"Block #{block.block_index}: SEARCH text not found in the file "
        f"({len(lines)} line(s), closest region at line {best + 1}, "
        f"{best_ratio:.0%} similar).\n"
        f"  Searched for:\n{expected}\n"
        f"  Closest content there:\n{closest}\n"
        f"{creation_note}"
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


def _opens_as_diff(edit_text: str) -> bool:
    """True when the payload opens with a ``--- a/x`` / ``+++ b/x`` header pair.

    That pair is written only by diff producers, so it outranks the marker
    sniff: a diff that *quotes* '<<<<<<< SEARCH' as context -- which is what
    editing this connector's own docstrings requires -- stays a diff.
    """
    lines = [line for line in edit_text.split("\n") if line.strip()][:2]
    return (len(lines) == 2 and lines[0].startswith("--- ")
            and lines[1].startswith("+++ "))


def resolve_format(fmt: str, edit_text: str) -> str:
    """Map a requested format to 'diff' or 'sr', sniffing when 'auto'.

    A payload that could be read as *both* is refused rather than guessed at:
    handing it to the wrong engine produces a confusing parse error (or, in the
    SEARCH/REPLACE case, an edit at the wrong place).
    """
    key = _FORMAT_ALIASES.get((fmt or "auto").strip().lower().replace("-", "_"))
    if key is None:
        raise FeditError(
            f"Unknown format {fmt!r}. Use 'search_replace', 'diff', or 'auto'."
        )
    if key != "auto":
        return key
    if _opens_as_diff(edit_text):
        return "diff"
    looks_like_blocks = SEARCH_MARKER in edit_text
    looks_like_diff = HUNK_ANYWHERE_RE.search(edit_text) is not None
    if looks_like_blocks and looks_like_diff:
        # A diff whose context quotes '<<<<<<< SEARCH' (documentation about this
        # very connector) and a SEARCH/REPLACE body holding a '@@ ... @@' line
        # both look like either format. Guessing here would send the payload to
        # the engine that cannot parse it, so ask.
        raise FeditError(
            "Ambiguous edit: the payload has both a '<<<<<<< SEARCH' marker and "
            "an '@@ -l,s +l,s @@' hunk header, so it could be either format. "
            "Pass format=\"diff\" or format=\"search_replace\" explicitly."
        )
    if looks_like_blocks:
        return "sr"
    if looks_like_diff:
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


def _system_prefix_hit(path: Path) -> Optional[str]:
    """The system prefix ``path`` is addressed under, or None.

    Used only to keep the *address* honest when the root bypass is decided on the
    resolved target: reaching the workspace through a link planted in ``/etc``
    still leaves an entry in a system tree, and that is not what --root-dir
    agreed to. Name-based rules are deliberately not part of this -- they judge
    what a file is, and ``realpath`` already settles that.
    """
    return path_policy.prefix_hit(path.as_posix(), path_policy.DENY_PREFIXES)


def _workspace_rel(real: Path) -> Optional[str]:
    """``real`` as a workspace-relative POSIX path, or None when outside the root.

    The connector's root *is* the workspace (``externals.json`` passes
    ``--root-dir ${BENCHMARKIT_PATH_LOCATION}``), so the workspace policy's rules
    -- which are written relative to that folder -- are matched against this.
    ``None`` (outside the root, or an unrooted server) means "the workspace policy
    has nothing to say here", which is the documented domain split: outside the
    root the connector deny-list is what speaks.

    Judged on the **resolved** path, exactly like the root bypass is: one physical
    file gets one verdict however it was addressed (mount point vs its alias), and
    a link planted in the workspace cannot point at a ``ro`` folder elsewhere and
    have it read as one here.
    """
    if ROOT_DIR is None:
        return None
    try:
        rel = real.relative_to(ROOT_DIR)
    except ValueError:
        return None
    text = rel.as_posix()
    return "" if text == "." else text


def _workspace_policy_reason(resolved: Path, *, for_write: bool) -> Optional[str]:
    """The workspace's own policy refusal for an in-root path, or None.

    pol-08: a folder the user marked ``ro`` in
    ``<root>/.nojae/read-write-policy.json`` is uneditable by an agent too, and a
    folder marked ``hidden`` is not readable either -- the same answer the Editor
    pane gives for the same path, from the same file and the same matcher
    (``policy_file.workspace_policy_reason``).

    What this does NOT apply is level A, benchmarkit's own rules about the files
    it regenerates: those exist because the pane fights the settings dialog over
    a generated file, which is not a property of an agent's edit. And the
    connector's own deny-list stays where it was -- the name-based floor that
    keeps a private key out of a context window -- because that floor guards a
    different trust surface, outside the root, where it is the only rule that
    means anything.
    """
    rel = _workspace_rel(resolved)
    if rel is None:
        return None
    return policy_file.workspace_policy_reason(
        rel, root=str(ROOT_DIR) if ROOT_DIR is not None else None,
        for_write=for_write,
    )


def _deny_reason(resolved: Path, *, for_write: bool) -> Optional[str]:
    """Explain why ``resolved`` is off-limits, or None if it is allowed.

    Delegates to the shared module so console and fedit answer with the same
    lists and the same wording.
    """
    return path_policy.deny_reason(resolved, for_write=for_write)


def check_target_path(path: str, *, for_write: bool) -> Path:
    """Resolve one path and apply the policy for the requested operation.

    This is the single entry point every tool goes through, so a read can never
    slip past a check that only the write path performed -- that asymmetry was
    issue #1: the deny-lists guarded ``apply_edit`` while ``read_file`` happily
    returned ``~/.git-credentials``.

    Root scoping: when the server runs with --root-dir, a path whose literal
    and real (symlink-resolved) forms both stay inside that root bypasses the
    *deny-lists* -- the root is the operator's workspace -- but not that
    workspace's own policy (pol-08): ``<root>/.nojae/read-write-policy.json`` is
    consulted for an in-root path, so a folder its owner marked ``ro`` or
    ``hidden`` is refused here exactly as it is in the Editor tab. Everything
    below only constrains paths outside the root. Relative paths anchor at the
    root.

    Traversal is judged where it *lands*, not on the token: the path is
    normalised (``abspath``) and fully resolved (``realpath``), and both forms
    are held to the deny rules -- so ``/tmp/../etc/shadow`` is refused because
    it *is* ``/etc/shadow``, while ``logs/probe/../../src/job.py`` is allowed
    because normalising it stays inside the root. Checking the resolved target
    as well as the normalised one is what stops a symlink from pointing the
    connector at a store the path itself does not mention.
    """
    if not path or not path.strip():
        raise FeditError("path is empty")
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = (ROOT_DIR or Path.cwd()) / candidate

    absolute = Path(os.path.abspath(str(candidate)))
    real = absolute.resolve()

    # The workspace's own policy, asked of the resolved target and *before* the
    # root bypass is reached (pol-08): that file is the user's answer to "which
    # parts of my project may change", and it constrains an agent exactly as it
    # constrains the Editor pane. It is asked about ``real`` for the same reason
    # the bypass is decided on it: one physical file, one verdict -- so a link
    # planted in a system prefix cannot reach a hidden workspace folder through
    # an address that never mentions it. It answers None for anything outside the
    # root, which is where the deny-lists below speak instead.
    reason = _workspace_policy_reason(real, for_write=for_write)
    if reason:
        action = "written" if for_write else "read"
        raise PathDenied(
            f"{absolute} may not be {action} by this connector: {reason}. "
            "Call edit_policy() to see the workspace policy."
        )

    # Inside the root the policy is bypassed; the operator chose that tree.
    # That decision is made on ``real`` -- where the file actually is -- and not
    # on the spelling that reached it. Two names for one tree (the mount point
    # ``/mnt/bigfoot/...`` and ``/home/vgr/...`` here) therefore get one verdict:
    # a workspace file is not suddenly "a secrets/config file" because it was
    # addressed through the other prefix. And a symlink cannot smuggle a store
    # *into* the bypass, because the link's target is what is tested --
    # ``logs/link -> /etc/shadow`` has a ``real`` outside the root and is judged
    # by the full policy below.
    # One exception: the *address* still has to be a sane one. A path that
    # physically reaches the workspace by way of a link planted inside a system
    # prefix (``/etc/x -> <root>/f``) is refused, because writing through it
    # means the connector created/used an entry in a tree the operator never
    # agreed to hand over. Name- and component-based rules are not applied this
    # way -- they judge the file's content, which ``real`` already decides.
    # Asymmetric by design, exactly as it was before: reading through such a
    # link discloses nothing the operator does not already own, so it stays
    # allowed (only the write is blocked), matching the rule that a link to a
    # benign outside file is readable but never writable.
    if _within_root(real) and not _system_prefix_hit(absolute):
        # The deny-lists are bypassed here -- the operator chose that tree. That
        # bypass is what keeps an ordinary workspace ``.env`` editable through
        # this connector while the Editor pane cannot read it: the pane's rule
        # for it is **level A** (benchmarkit regenerates its own ``.env`` from
        # the settings template), a fact about the pane rather than about an
        # agent's edit, and connectors do not consult level A.
        return absolute

    action = "written" if for_write else "read"

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
    if not resolved.parent.is_dir():
        # Still a refusal to create trees, but named for what it is: "not
        # writable" reads as a permissions problem and sends the agent off to
        # chmod (or to give up) instead of running mkdir.
        raise FeditError(
            f"Directory {resolved.parent} does not exist. Create it first with "
            f"console's mkdir; fedit only writes files into existing directories."
        )
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


def backup_key(resolved: Path) -> str:
    """The digest that identifies *the file* a backup belongs to.

    Keyed on the symlink-resolved path, not the spelling used in the tool call:
    the same tree here is reachable as ``/mnt/bigfoot/dev/...`` and
    ``/home/vgr/dev/...``. Hashing the literal string gave those two spellings
    separate chains, so an edit applied under one name left ``list_edits`` and
    ``revert_edit`` answering "this file has not been edited by this server"
    under the other -- which reads as "there is nothing to recover" and silently
    destroys the recovery path.
    """
    return hashlib.sha1(str(resolved.resolve()).encode("utf-8")).hexdigest()[:12]


def backup_path_for(resolved: Path) -> Path:
    digest = backup_key(resolved)
    safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", resolved.name)[:60]
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S.%f")
    return BACKUP_DIR / f"{digest}_{safe_name}_{stamp}.orig"


def list_backups(resolved: Path) -> list[Path]:
    digest = backup_key(resolved)
    if not BACKUP_DIR.is_dir():
        return []
    return sorted(BACKUP_DIR.glob(f"{digest}_*.orig"))


def make_backup(resolved: Path) -> Path:
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    destination = backup_path_for(resolved)
    shutil.copy2(resolved, destination)
    # ``copy2`` carries the source's mtime over, which would make a backup of a
    # long-untouched file be *born* past the age limit and swept seconds later.
    # The sweep's clock is "how long has this edit been recoverable", so stamp
    # the backup with the time of the edit. (The name already carries it; this
    # just makes the filesystem agree.)
    try:
        os.utime(destination, None)
    except OSError:
        pass
    existing = list_backups(resolved)
    for stale in existing[:-BACKUP_KEEP] if len(existing) > BACKUP_KEEP else []:
        try:
            stale.unlink()
        except OSError:
            pass
    prune_old_backups()
    return destination


def prune_old_backups(*, older_than_days: int = BACKUP_PRUNE_DAYS,
                      now: Optional[float] = None) -> int:
    """Delete every backup older than ``older_than_days``. Returns how many went.

    ``make_backup`` keeps ``BACKUP_KEEP`` copies *per file*, which is right but
    incomplete: once a file is renamed or deleted, nothing ever collects the
    backups that referred to it, so a long-lived workspace accumulates them for
    ever (537 ``.orig`` files before this existed). Age is the only signal that
    still works for a path that no longer exists, so the sweep is by age.

    The consequence is stated rather than hidden: ``revert_edit`` reaches back
    ``BACKUP_PRUNE_DAYS``, not for ever. That window is what ``edit_policy()``
    reports, and it is far longer than any in-session undo -- older history
    belongs in git, not in this directory.

    Cheap by design (one directory scan) and called from the write path, so it
    never raises: a backup that cannot be deleted is not worth failing an edit.
    """
    if older_than_days <= 0 or not BACKUP_DIR.is_dir():
        return 0
    cutoff = (now if now is not None else time.time()) - older_than_days * 86400
    removed = 0
    try:
        candidates = list(BACKUP_DIR.glob("*.orig"))
    except OSError:
        return 0
    for item in candidates:
        try:
            if item.stat().st_mtime < cutoff:
                item.unlink()
                removed += 1
        except OSError:
            continue
    return removed


def write_atomic_bytes(resolved: Path, payload: bytes) -> None:
    """Write bytes via a temp file in the same directory, then rename over target.

    The target's permission bits are copied onto the temp before the rename:
    ``mkstemp`` creates 0600, and ``os.replace`` makes *that* the file's new
    mode, so without the ``chmod`` every edited file came out 0600 -- a 755
    script lost its ``+x``, a 444 file silently became writable, and
    ``git diff`` showed an unexplained mode change.

    Bytes, not text: a revert has to restore exactly what the backup holds,
    whether or not those bytes decode as UTF-8.
    """
    mode: Optional[int] = None
    try:
        mode = resolved.stat().st_mode & 0o7777
    except OSError:
        pass  # creating a new file: keep mkstemp's own mode
    handle, tmp_name = tempfile.mkstemp(
        dir=str(resolved.parent), prefix=f".{resolved.name}.", suffix=".fedit-tmp"
    )
    try:
        with os.fdopen(handle, "wb") as handle_file:
            handle_file.write(payload)
        if mode is not None:
            os.chmod(tmp_name, mode)
        os.replace(tmp_name, resolved)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def write_atomic(resolved: Path, content: str) -> None:
    """Text flavour of ``write_atomic_bytes`` (UTF-8, no newline translation)."""
    write_atomic_bytes(resolved, content.encode("utf-8"))


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
            # The cap is right (an agent "editing" a vendored bundle is a
            # mistake), but a bare refusal is a dead end: name the route that
            # still works, since console has no size limit at all.
            return (f"Error: {resolved} is {size} bytes, over the "
                    f"{MAX_FILE_BYTES} limit. Read or search a slice with "
                    f"console (no size cap): ``sed -n '{max(1, int(start_line))},"
                    f"{max(1, int(start_line)) + max(1, int(max_lines)) - 1}p' {resolved}`` "
                    f"or ``grep -n PATTERN {resolved}``.")
        payload = resolved.read_bytes()
    except (FeditError, OSError) as exc:
        return f"Error: {exc}"

    reason = binary_reason(payload)
    if reason:
        return f"Error: {not_editable_message(resolved, reason)}"
    # Text in some other encoding is still worth showing: decode with the bad
    # bytes replaced and say so, so the read is never mistaken for an
    # authoritative copy (it is not something a SEARCH block can quote back).
    lossy = undecodable_reason(payload) is not None
    text = payload.decode("utf-8", "replace") if lossy else payload.decode("utf-8")

    body, bom = strip_bom(text)
    lines, final_newline, terminator = split_text(body)
    start = max(1, int(start_line))
    if not lines:
        # An empty file is a readable file, not an unreadable one: answering
        # "start_line 1 is past the end" sends the agent hunting for a
        # permission problem instead of telling it the file is empty.
        return f"File: {resolved}  lines 0-0 of 0  {size} bytes  EOF: NO final newline"
    if start > len(lines):
        return f"Error: start_line {start} is past the end of a {len(lines)}-line file"
    chunk = lines[start - 1:start - 1 + max(1, int(max_lines))]
    numbered = "\n".join(f"{start + i:>6}\t{line}" for i, line in enumerate(chunk))
    end = start + len(chunk) - 1
    eof = "newline" if final_newline else "NO final newline"
    header = (f"File: {resolved}  lines {start}-{end} of {len(lines)}  "
              f"{size} bytes  EOF: {eof}")
    if lossy:
        header += ("  encoding: NOT valid UTF-8 (undecodable bytes shown as "
                   "\ufffd, so this text cannot be copied back into an edit)")
    if bom:
        header += "  BOM: UTF-8 (not shown, and not part of line 1)"
    if terminator != "\n":
        header += f"  endings: {TERMINATOR_NAMES[terminator]}"
    if end < len(lines):
        header += f"  (continues at line {end + 1})"
    return f"{header}\n{numbered}"


@server.tool()
def preview_edit(path: str, edit: str, format: str = "auto", fuzz: Optional[int] = None) -> str:
    """Dry-run an edit: validates and shows the resulting diff, writes nothing.

    Same formats and the same rule as ``apply_edit``: a file holding '=======',
    '<<<<<<<' or '>>>>>>>' lines must be edited with ``format="diff"``, because
    SEARCH/REPLACE cannot quote such a line and refuses instead of guessing
    where the block ends.

    It previews a creation too: a path with no file at it is diffed against
    emptiness, the same shapes ``apply_edit`` accepts to create one.

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
        before = ""
        creating = not resolved.is_file()
        if not creating:
            size = resolved.stat().st_size
            if size > MAX_FILE_BYTES:
                return (f"Error: {resolved} is {size} bytes, over the "
                        f"{MAX_FILE_BYTES} limit -- the preview of an edit to "
                        f"it would be as unreviewable as the edit.")
            before = read_payload_for_edit(resolved)
        after, warnings = run_edit(resolved, before, edit, format, fuzz,
                                   creating=creating)
    except (FeditError, UnicodeDecodeError, OSError) as exc:
        return f"Error: {exc}"

    if before == after:
        return f"VALID but a no-op: {resolved} would be unchanged."
    diff = "\n".join(difflib.unified_diff(
        split_text(before)[0], split_text(after)[0],
        fromfile=f"a/{resolved.name}", tofile=f"b/{resolved.name}", lineterm="", n=2))
    report = [f"PREVIEW (not written) for {resolved}"]
    if warnings:
        report.append("Warnings:\n" + "\n".join(f"  - {w}" for w in warnings))
    report.append(diff)
    return "\n\n".join(report)


def run_edit(resolved: Path, before: str, edit: str,
             fmt: str, fuzz: Optional[int],
             creating: bool = False) -> tuple[str, list[str]]:
    """Dispatch one edit payload through the right engine."""
    if len(edit.encode("utf-8")) > MAX_EDIT_BYTES:
        raise FeditError(f"edit payload is over {MAX_EDIT_BYTES} bytes")
    kind = resolve_format(fmt, edit)
    if kind == "sr":
        return apply_search_replace_to_text(before, edit, creating=creating)
    return apply_diff_to_text(before, edit, fuzz=fuzz, creating=creating)


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

    A file that does not exist yet is **created** when the edit adds rather than
    replaces -- quote nothing to replace. Send an empty SEARCH with the whole new
    content in REPLACE::

        <<<<<<< SEARCH
        =======
        first line of the new file
        second line
        >>>>>>> REPLACE

    or the diff's pure-addition shape, ``@@ -0,0 +1,2 @@`` with every line
    prefixed by '+'. A SEARCH holding any text at all must still match bytes the
    file has, so against a missing file it is refused, never guessed.

    If the file contains '=======', '<<<<<<<' or '>>>>>>>' lines (a git conflict,
    or documentation about this very format), pass ``format="diff"``: such a line
    cannot be quoted inside a SEARCH or REPLACE body, and SEARCH/REPLACE refuses
    rather than guessing. A diff tokenises no markers, so it handles them.

    Args:
        path: File to edit; a missing file is created, but its parent directory
            must already exist (make it with console's ``mkdir``)
        edit: The diff or SEARCH/REPLACE payload
        format: 'auto' (detect), 'search_replace' or 'diff'
        fuzz: Max context lines a hunk may ignore when repositioning (0..3)

    Returns:
        The diff that was applied, plus the backup path for reverting (a created
        file has no backup, there was nothing to preserve)

    Example:
        apply_edit("/srv/app/main.py", edit, format="search_replace")
    """
    try:
        resolved = check_write_path(path)
        creating = not resolved.is_file()
        if creating:
            before = ""
        else:
            size = resolved.stat().st_size
            if size > MAX_FILE_BYTES:
                return (f"Error: {resolved} is {size} bytes, over the "
                        f"{MAX_FILE_BYTES} limit. An edit rewrites the whole "
                        f"file, so the cap applies to edits too; change it "
                        f"through console (``sed``/``awk``) instead.")
            before = read_payload_for_edit(resolved)
        after, warnings = run_edit(resolved, before, edit, format, fuzz,
                                   creating=creating)
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
        split_text(before)[0], split_text(after)[0],
        fromfile=f"a/{resolved.name}", tofile=f"b/{resolved.name}", lineterm="", n=1))
    lines_before = len(split_text(before)[0])
    lines_after = len(split_text(after)[0])
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
        return (f"Error: no backup for {resolved} — nothing to revert (backups "
                f"are kept per file up to {BACKUP_KEEP}, and any copy older "
                f"than {BACKUP_PRUNE_DAYS} days is swept; see edit_policy()).")

    # A backup is a byte copy (``shutil.copy2``), so it is restored as bytes:
    # decoding it first would make recovery impossible exactly when the file was
    # not valid UTF-8 -- which is when a working revert matters most.
    newest = backups[-1]
    try:
        restored = newest.read_bytes()
        current_note = ""
        if resolved.is_file():
            current_note = f"\nPre-revert copy: {make_backup(resolved)}"
        write_atomic_bytes(resolved, restored)
        newest.unlink()
    except OSError as exc:
        return f"Error: could not revert {resolved}: {exc}"
    line_count = len(split_text(restored.decode("utf-8", "replace"))[0])
    return (f"Reverted {resolved} to its content from {newest.stem.rsplit('_', 1)[-1]}"
            f" ({len(restored)} bytes, {line_count} line(s)).{current_note}")


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
        for item in (a, b):
            size = item.stat().st_size
            if size > MAX_FILE_BYTES:
                return (f"Error: {item} is {size} bytes, over the {MAX_FILE_BYTES} "
                        f"limit -- compare them with console instead (it has no "
                        f"size cap): ``diff {a} {b}``, or inspect one slice with "
                        f"``sed -n '10,40p' {item}``.")
        payload_a = a.read_bytes()
        payload_b = b.read_bytes()
    except (FeditError, OSError) as exc:
        return f"Error: {exc}"
    # Byte identity is decided before the text check, so two identical binaries
    # still get the honest answer.
    if payload_a == payload_b:
        return f"{a} and {b} are identical."
    # A diff line is meant to be copied into an edit, so both sides have to be
    # real UTF-8 text here -- unlike read_file, which may display a lossy view.
    for item, payload in ((a, payload_a), (b, payload_b)):
        reason = not_editable_reason(payload)
        if reason:
            return f"Error: {not_editable_message(item, reason)}"
    text_a = strip_bom(payload_a.decode("utf-8"))[0]
    text_b = strip_bom(payload_b.decode("utf-8"))[0]
    return "\n".join(difflib.unified_diff(
        split_text(text_a)[0], split_text(text_b)[0],
        fromfile=str(a), tofile=str(b), lineterm="", n=max(0, int(context))))


# How many patterns one mode may list in `edit_policy()`. The model acts on this
# text, so its shape matters more than its completeness: a full dump of a
# 40-rule file pushes the caps and the deny-lists out of the answer, and a
# truncated list that *says* it is truncated still tells the agent where to ask.
POLICY_PATTERNS_PER_MODE = 5


def _workspace_policy_lines() -> str:
    """The workspace policy block of `edit_policy()` -- what this connector enforces *in the root*.

    A tool that reports a policy it does not enforce is worse than one that
    reports nothing, so this block is the rules the checks in
    :func:`_workspace_policy_reason` actually applied, read through the same
    loader, in the same words the Editor's own policy panel uses (pol-07): the
    pattern verbatim grouped by mode.
    """
    if ROOT_DIR is None:
        return (
            "  policy file     : (no root dir, so no workspace policy applies)\n"
        )
    path = policy_file.workspace_policy_path(str(ROOT_DIR))
    rules, findings = policy_file.load_workspace_policy(str(ROOT_DIR))
    errors = [f for f in findings if f.kind == policy_shared.KIND_ERROR]

    if errors:
        # pol-00 D4, connector edition: an unparseable file contributes no rules,
        # so every write inside the root is refused until it loads -- and the
        # agent has to be told that, or it reads "no rules" and keeps trying.
        return (
            f"  policy file     : {path} (did not load)\n"
            f"                    {errors[0].message}\n"
            f"                    writes inside the root are refused until it does\n"
        )
    if not rules:
        return (
            f"  policy file     : {path} (no rules)\n"
            f"                    nothing inside the root is restricted beyond the\n"
            f"                    reserved {policy_shared.POLICY_DIRNAME} folder\n"
        )

    lines = [f"  policy file     : {path} ({len(rules)} rules)"]
    for mode in (policy_shared.HIDDEN, policy_shared.READ_ONLY):
        patterns = [rule.pattern for rule in rules if rule.mode == mode]
        if not patterns:
            continue
        shown = patterns[:POLICY_PATTERNS_PER_MODE]
        extra = len(patterns) - len(shown)
        tail = f" +{extra} more" if extra else ""
        lines.append(f"                    {mode + ':':<8}{'  '.join(shown)}{tail}")
    lines.append(
        f"                    {policy_shared.POLICY_DIRNAME}: reserved by benchmarkit "
        f"(hidden, and no rule can unlock it)"
    )
    return "\n".join(lines) + "\n"


@server.tool()
def edit_policy() -> str:
    """Report the connector's own limits and protected paths (no guessing needed).

    Returns:
        Backup directory, size caps, fuzz level, the workspace policy this
        connector enforces inside the root (its rules by mode), and the
        deny-lists it applies outside the root

    Example:
        edit_policy()
    """
    return (
        f"fedit-mcp policy\n"
        f"  root dir        : {ROOT_DIR if ROOT_DIR is not None else '(unset -- full policy on every path)'}\n"
        f"  inside the root : the workspace policy of that folder applies (see below);\n"
        f"                    the connector deny-list below does not\n"
        f"  backup dir      : {BACKUP_DIR.resolve()}\n"
        f"  backups kept    : {BACKUP_KEEP}\n"
        f"  max file bytes  : {MAX_FILE_BYTES}\n"
        f"  max edit bytes  : {MAX_EDIT_BYTES}\n"
        + _workspace_policy_lines() +
        f"  fuzz            : default {MAX_FUZZ}, ceiling {FUZZ_CEILING} (a higher "
        f"request is reduced)\n"
        f"  backup retention: {BACKUP_KEEP} per file, swept after {BACKUP_PRUNE_DAYS} days\n"
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
             "Files and directories inside it bypass the protected-path lists "
             "but obey that folder's own workspace policy "
             "(.nojae/read-write-policy.json); paths outside it are checked "
             "against the protected-path lists."
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
