#!/usr/bin/env python3
"""
Shared path policy for the file-touching MCP connectors.

Both ``fedit_mcp`` (reads, edits) and ``console_mcp`` (shell commands) can reach
any file the user can read, so they share ONE set of protected paths. Keeping a
single module is the point: when ``fedit`` carefully refused ``~/.ssh`` while
``console`` ran ``cat .env``, the careful policy was worth nothing -- the agent
simply used the sibling connector that was enabled in the same session.

**Do not merge this with the workspace policy** (``policy_file.py`` and the
``.nojae/read-write-policy.json`` it reads, which both connectors honour *inside*
their root since pol-08). They answer different questions and their domains do
not overlap, and a contributor who reads "two rule lists, one connector" will be
tempted to unify them:

* this module asks *"is this a credential store?"* -- the same answer in every
  tree, which is why it is hardcoded and env-overridable rather than authored
  per workspace, and why it is the only rule that speaks for paths **outside**
  ``--root-dir``;
* the workspace policy asks *"which parts of this project may change?"* -- a
  question the operator wrote down, and one no list here can ask.

Inside the root these deny-lists are bypassed, because the operator chose that
tree; pol-08 narrowed the agent there by adding the workspace's rules, not by
extending these. So one asymmetry is expected and must not be "fixed" by making
``editorfs`` import this module again, or by making these lists read the
workspace file: an ordinary workspace ``.env`` is editable through ``fedit``
while the Editor pane cannot read it, because the pane's rule is about who owns
and regenerates that file, not about its name.

The lists are env-overridable so a deployment can tighten or relax them without
a code edit. The shared names (``BENCHMARKIT_DENY_*``) win over the historical
fedit names (``FEDIT_DENY_*``), which keep working.
"""

import fnmatch
import os
from pathlib import Path
from typing import Iterable, Optional


def env_list(shared_key: str, legacy_key: str, default: str) -> list[str]:
    """One comma-separated list from the environment, shared name first."""
    raw = os.environ.get(shared_key)
    if raw is None:
        raw = os.environ.get(legacy_key, default)
    return [item.strip() for item in raw.split(",") if item.strip()]


# Scratch space is where an agent puts throwaway work, so a path that stays
# inside one of these skips the policy below.
ALLOW_PREFIXES: tuple[str, ...] = tuple(
    env_list("BENCHMARKIT_ALLOW_PREFIXES", "FEDIT_ALLOW_PREFIXES", "/tmp,/var/tmp")
)

# System prefixes that must never be written to.
DENY_PREFIXES: tuple[str, ...] = tuple(
    env_list(
        "BENCHMARKIT_DENY_PREFIXES", "FEDIT_DENY_PREFIXES",
        "/etc,/usr,/bin,/sbin,/lib,/lib64,/boot,/sys,/proc,/dev,/run",
    )
)

# System prefixes that are readable but never writable -- ordinary files here
# are useful debugging material, unlike the credential stores below.
DENY_PREFIXES_READ: tuple[str, ...] = tuple(
    env_list(
        "BENCHMARKIT_DENY_PREFIXES_READ", "FEDIT_DENY_PREFIXES_READ",
        "/etc/ssh,/etc/ssl/private,/etc/sudoers,/etc/shadow,/etc/gshadow,"
        "/proc,/sys,/dev,/boot,/run",
    )
)

# Path components that must never appear in a path a connector writes: the
# version-control internals, credentials, the interpreter installation, and the
# connectors' own sources (an agent must not rewrite the policy that constrains
# it). An entry may span several components.
DENY_COMPONENTS: frozenset[str] = frozenset({".git", ".ssh", ".pyenv", "mcp_connectors"})

# Components that name a *secret store*, and are therefore off-limits to reads
# as well as writes: a private key or an access token pulled into the context
# window does as much damage as overwriting the file that held it.
DENY_COMPONENTS_SECRET: frozenset[str] = frozenset(
    part.strip("/")
    for part in env_list(
        "BENCHMARKIT_DENY_COMPONENTS_SECRET", "FEDIT_DENY_COMPONENTS_SECRET",
        ".git,.ssh,.gnupg,.aws,.azure,.gcloud,.kube,.docker,"
        ".password-store,.local/share/keyrings,.config/gh,credentials",
    )
)

# Filename prefixes that hold secrets the app itself reads/writes.
DENY_FILE_PREFIXES: tuple[str, ...] = (".env",)

# Filenames that are secrets outright, and extensions that mark a private key
# or a certificate. Matched on the last path component, case-insensitively.
DENY_FILENAMES: frozenset[str] = frozenset({
    ".git-credentials", "shadow", "known_hosts", "master.key",
    "id_rsa", "id_dsa", "id_ecdsa", "id_ed25519", "id_sk_ecdsa", "id_sk_ed25519",
})
DENY_FILE_SUFFIXES: tuple[str, ...] = (".env", ".pem", ".key", ".p12", ".pfx", ".kdbx")

# Whole-path globs for stores whose directory name alone is too common to deny
# outright (``~/.config`` legitimately holds plenty of harmless configuration).
DENY_PATH_GLOBS: tuple[str, ...] = tuple(
    env_list(
        "BENCHMARKIT_DENY_PATH_GLOBS", "FEDIT_DENY_PATH_GLOBS",
        "*/.config/google-chrome*/*,*/.config/chromium*/*,*/.config/opera*/*",
    )
)

# Effective component lists per operation: reads are blocked on secret stores
# only (an agent debugging a dependency legitimately needs to read a library
# source under .pyenv), writes are blocked on both.
DENY_COMPONENTS_READ: frozenset[str] = DENY_COMPONENTS_SECRET
DENY_COMPONENTS_WRITE: frozenset[str] = frozenset(DENY_COMPONENTS) | DENY_COMPONENTS_SECRET


def component_hit(parts: tuple[str, ...], entries: Iterable[str]) -> Optional[str]:
    """Return the first entry matching ``parts``, or None.

    An entry is either a single component (matched by name) or a sequence such
    as ``.config/gh``, matched against consecutive components of the path --
    which is how a store like ``~/.config/gh`` is denied without denying all of
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


def prefix_hit(text: str, prefixes: Iterable[str]) -> Optional[str]:
    """The protected prefix ``text`` sits under, or None."""
    for prefix in prefixes:
        if text == prefix or text.startswith(prefix.rstrip("/") + "/"):
            return prefix
    return None


def under_prefix(path: Path, prefix: str) -> bool:
    """True when ``path`` is inside (or equal to) ``prefix``."""
    return prefix_hit(path.as_posix(), (prefix,)) is not None


def glob_hit(text: str, patterns: Iterable[str]) -> Optional[str]:
    """The whole-path glob ``text`` matches, or None."""
    for pattern in patterns:
        if fnmatch.fnmatchcase(text, pattern):
            return pattern
    return None


def deny_reason(resolved: Path, *, for_write: bool) -> Optional[str]:
    """Explain why ``resolved`` is off-limits, or None when it is allowed.

    Reads and writes share the shape of the test but not its list: writing is
    refused under every protected prefix, reading only under the ones that hold
    credentials (``/etc/hosts`` is fair debugging material, ``~/.ssh`` is not).
    """
    text = resolved.as_posix()
    prefixes = DENY_PREFIXES if for_write else DENY_PREFIXES_READ
    prefix = prefix_hit(text, prefixes)
    if prefix:
        return f"under the protected prefix {prefix!r}"

    components = DENY_COMPONENTS_WRITE if for_write else DENY_COMPONENTS_READ
    hit = component_hit(resolved.parts, components)
    if hit:
        verb = "write to" if for_write else "read from"
        return f"inside {hit!r}, which connectors never {verb}"

    name = resolved.name.lower()
    if name.startswith(DENY_FILE_PREFIXES):
        return "a secrets/config file (its name starts with '.env')"
    if name in DENY_FILENAMES:
        return f"a credential file ({resolved.name})"
    if name.endswith(DENY_FILE_SUFFIXES):
        return f"a private key or certificate ({resolved.name})"
    pattern = glob_hit(text, DENY_PATH_GLOBS)
    if pattern:
        return f"a browser profile/credential store (it matches {pattern!r})"
    return None
