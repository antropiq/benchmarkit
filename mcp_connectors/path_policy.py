#!/usr/bin/env python3
r"""
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

**Windows forms (target B1 / G-D8, B-D1).** One list carries both path forms:
the committed defaults stay POSIX, and the Windows system trees arrive *at read
time* from the environment the OS itself states them in (``SystemRoot``,
``ProgramFiles``, ``ProgramW6432``, ``ProgramFiles(x86)``, ``ProgramData``;
``TEMP``/``TMP`` for the scratch allowance). A runner without those names gains
nothing, so Linux verdicts are untouched and no machine path is committed.
Prefix matching is **component-wise** (``prefix_hit`` compares roots first, then
whole components): a text-prefix test would call ``C:\Windows.old`` a child of
``C:\Windows``, and the component compare makes that class impossible. The
``fold`` argument is the A-D3 case answer for the volume actually walked,
supplied by the caller from ``mcp_connectors.fscase`` — never read from a
global or from the platform's name.
"""

import fnmatch
import ntpath
import os
from pathlib import Path
from typing import Iterable, Mapping, Optional


def env_list(shared_key: str, legacy_key: str, default: str) -> list[str]:
    """One comma-separated list from the environment, shared name first."""
    raw = os.environ.get(shared_key)
    if raw is None:
        raw = os.environ.get(legacy_key, default)
    return [item.strip() for item in raw.split(",") if item.strip()]


def _env_absolute(
    name: str, env: Optional[Mapping[str, str]] = None
) -> Optional[str]:
    """One absolute directory the OS states in ``name``, or None.

    The Windows half of the guardrails (B-D1/G-D8): the values arrive at read
    time from the environment, so a committed file carries no machine paths and
    a runner without the name (Linux) contributes nothing. An empty or relative
    value is not a prefix and is skipped — a guardrail that cannot be honoured
    is absent, not guessed.
    """
    source = os.environ if env is None else env
    value = source.get(name, "").strip().strip('"')
    if not value:
        return None
    # Absoluteness is a *text* question, asked with ``ntpath`` (roadmap §9
    # rule 2): ``os.path.isabs`` on Linux cannot see ``C:\Windows``, so the
    # Windows guardrails would silently never join the lists on the runner that
    # writes their tests. POSIX values answer the same either way, and
    # ``prefix_hit`` folds separators on both sides of the compare.
    if not ntpath.isabs(value):
        return None
    return value


def _dedupe(entries: list[str]) -> tuple[str, ...]:
    """Order-preserving dedupe: one guardrail stated twice is noise in a list."""
    seen: dict[str, None] = {}
    for entry in entries:
        seen.setdefault(entry, None)
    return tuple(seen)


# The system trees a Windows deployment must never write into, each named by
# the environment variable the OS states it in (B-D1). ``ProgramFiles(x86)`` is
# a real variable name, parentheses included. They join the POSIX defaults in
# the same list — one list, both forms, per B-D1's rejected alternatives.
_SYSTEM_DENY_ENV: tuple[str, ...] = (
    "SystemRoot", "ProgramFiles", "ProgramW6432", "ProgramFiles(x86)", "ProgramData",
)

# The scratch allowance's Windows half (B-D2): the temp directory the OS names,
# never its parent — the allowance stays narrower than the profile it lives in.
_SCRATCH_ENV: tuple[str, ...] = ("TEMP", "TMP")


def scratch_prefix_candidates(env: Optional[Mapping[str, str]] = None) -> list[str]:
    """The scratch directories ``env`` names (B-D2's Windows half).

    ``TEMP`` and ``TMP`` are usually one folder; the module-level dedupe keeps
    the guardrail stated once.
    """
    return [v for v in (_env_absolute(name, env) for name in _SCRATCH_ENV) if v]


def system_deny_prefix_candidates(env: Optional[Mapping[str, str]] = None) -> list[str]:
    """The Windows system trees ``env`` states (B-D1) — never written to.

    ``ProgramW6432`` names the same folder ``ProgramFiles`` does on a 64-bit
    system; the module-level dedupe keeps such a guardrail stated once.
    """
    return [v for v in (_env_absolute(name, env) for name in _SYSTEM_DENY_ENV) if v]


def system_read_deny_prefix_candidates(
    env: Optional[Mapping[str, str]] = None,
) -> list[str]:
    """The Windows analogue of ``/etc/shadow``: the hive directory the system
    root keeps — readable material is fine, this directory is not.

    Joined with the separator the value itself uses, so the entry keeps the
    spelling the OS stated; :func:`prefix_hit` folds separators on both sides.
    """
    root = _env_absolute("SystemRoot", env)
    return [ntpath.join(root, "System32", "config")] if root else []


# Scratch space is where an agent puts throwaway work, so a path that stays
# inside one of these skips the policy below — but never ahead of the
# component/name half of it (B-D2's ordering: the console asks those first).
ALLOW_PREFIXES: tuple[str, ...] = _dedupe(
    env_list("BENCHMARKIT_ALLOW_PREFIXES", "FEDIT_ALLOW_PREFIXES", "/tmp,/var/tmp")
    + scratch_prefix_candidates()
)

# System prefixes that must never be written to.
# The POSIX defaults are named apart from the environment-grown Windows half
# (B-D1/G-D8) so a caller can ask for one without the other — a test that
# patches its own Windows entry must not have this runner's ``SystemRoot``
# answer first, and production honours both.
_DENY_PREFIXES_DEFAULT = (
    "/etc,/usr,/bin,/sbin,/lib,/lib64,/boot,/sys,/proc,/dev,/run"
)

DENY_POSIX_PREFIXES: tuple[str, ...] = _dedupe(
    env_list("BENCHMARKIT_DENY_PREFIXES", "FEDIT_DENY_PREFIXES", _DENY_PREFIXES_DEFAULT)
)

DENY_PREFIXES: tuple[str, ...] = _dedupe(
    list(DENY_POSIX_PREFIXES) + system_deny_prefix_candidates()
)

# System prefixes that are readable but never writable -- ordinary files here
# are useful debugging material, unlike the credential stores below. The
# Windows entry is the analogue of ``/etc/shadow``: the hive directory under
# the system root.
DENY_PREFIXES_READ: tuple[str, ...] = _dedupe(
    env_list(
        "BENCHMARKIT_DENY_PREFIXES_READ", "FEDIT_DENY_PREFIXES_READ",
        "/etc/ssh,/etc/ssl/private,/etc/sudoers,/etc/shadow,/etc/gshadow,"
        "/proc,/sys,/dev,/boot,/run",
    )
    + system_read_deny_prefix_candidates()
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


# A drive is one ASCII letter plus ``:`` -- the same constraint win-02's
# ``path_norm`` puts on ``ntpath.splitdrive``, which is more permissive than the
# Windows parser it describes and would hand back ``foo:`` as a drive.
_ASCII_LETTERS: frozenset[str] = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
)


def component_hit(
    parts: tuple[str, ...], entries: Iterable[str], *, fold: bool = False
) -> Optional[str]:
    """Return the first entry matching ``parts``, or None.

    An entry is either a single component (matched by name) or a sequence such
    as ``.config/gh``, matched against consecutive components of the path --
    which is how a store like ``~/.config/gh`` is denied without denying all of
    ``~/.config``.

    ``fold=True`` compares case-insensitively, which is what a case-folding
    filesystem needs for ``.ssh`` to hit ``C:\\Users\\VGR\\.SSH\\id_ed25519`` --
    the two spellings are one folder there (win-07 / A-D3). It is an explicit
    argument, never read from a global or from the platform's name, so both
    outcomes stay assertable on one runner (roadmap §9 rule 2), and the caller
    supplies the probe answer for the volume actually walked. The default is
    ``False`` because that is what an exact filesystem is: folding
    unconditionally would make ``Dist`` and ``dist`` one folder.
    """
    if fold:
        parts = tuple(part.casefold() for part in parts)
    for entry in entries:
        needle = tuple(entry.split("/"))
        if fold:
            needle = tuple(part.casefold() for part in needle)
        if len(needle) == 1:
            if needle[0] in parts:
                return entry
        else:
            for offset in range(len(parts) - len(needle) + 1):
                if parts[offset:offset + len(needle)] == needle:
                    return entry
    return None


def _rooted_parts(text: str) -> tuple[str, tuple[str, ...]]:
    r"""``text`` as ``(root, components)`` — the shape one prefix compare uses.

    The root is ``"/"`` for a POSIX absolute path, ``"C:"`` for a drive-spec
    path, ``"//server"`` for a UNC share, ``""`` for relative text; the
    components are the whole names under it. Backslashes are separators, which
    is what a Windows ``Path.as_posix()`` string looks like. Roots first,
    components whole: that is what makes ``C:/Windows.old`` not a child of
    ``C:/Windows`` and a ``/etc``-shaped prefix silent about a ``C:`` tree
    (B-D1), and it is the same root taxonomy :func:`root_form` classifies.
    """
    normalised = text.replace("\\", "/")
    drive, rest = ntpath.splitdrive(normalised)
    if drive:
        # A one-letter drive (``C:``), or a whole UNC share: ``splitdrive``
        # hands back ``//server/share`` as the drive — the share *is* the root,
        # the same reading :func:`root_form` classifies as ``"unc"``.
        return drive, tuple(part for part in rest.split("/") if part)
    if normalised.startswith("/"):
        return "/", tuple(part for part in normalised.split("/") if part)
    return "", tuple(part for part in normalised.split("/") if part)


def prefix_hit(
    text: str, prefixes: Iterable[str], *, fold: bool = False
) -> Optional[str]:
    r"""The protected prefix ``text`` sits under, or None.

    **Component-wise** (B-D1/G-D8): the roots must agree first — a POSIX
    absolute path is not under a drive prefix, and a drive path is not under a
    ``/etc`` — and only then are the prefix's components compared against the
    leading components of the path. The old text-prefix test
    (``text.startswith(prefix + "/")``) was correct for the ``/``-normalised
    form it was fed but handed raw Windows text it called ``C:\Windows.old`` a
    child of ``C:\Windows``; comparing whole components removes that whole
    class.

    ``fold`` is explicit, for the same reason ``component_hit`` takes it: the
    case answer belongs to the volume walked, and the caller supplies it from
    the probe (win-05 / G-D8). ``c:\windows\system32`` and
    ``C:\WINDOWS\System32`` are one prefix on a folding volume and two
    spellings on an exact one; both outcomes stay assertable on one runner.
    """
    text_root, text_parts = _rooted_parts(text)
    if fold:
        text_root = text_root.casefold()
        text_parts = tuple(part.casefold() for part in text_parts)
    for prefix in prefixes:
        proot, pparts = _rooted_parts(prefix)
        if fold:
            proot = proot.casefold()
            pparts = tuple(part.casefold() for part in pparts)
        if proot != text_root or len(pparts) > len(text_parts):
            continue
        if text_parts[: len(pparts)] == pparts:
            return prefix
    return None


def under_prefix(path: Path, prefix: str, *, fold: bool = False) -> bool:
    """True when ``path`` is inside (or equal to) ``prefix``."""
    return prefix_hit(path.as_posix(), (prefix,), fold=fold) is not None


def glob_hit(
    text: str, patterns: Iterable[str], *, fold: bool = False
) -> Optional[str]:
    """The whole-path glob ``text`` matches, or None."""
    if fold:
        # The patterns are written lowercase; folding the compared text is the
        # A-D3 rule again — comparison only, never the stored form.
        text = text.casefold()
    for pattern in patterns:
        if fnmatch.fnmatchcase(text, pattern.casefold() if fold else pattern):
            return pattern
    return None


def prefix_deny_reason(
    resolved: Path, *, for_write: bool, fold: bool = False
) -> Optional[str]:
    """The *location* half of :func:`deny_reason` — protected prefixes only.

    Split out so a caller can ask the name/component half first: the scratch
    allowance must not outrank what a path *is* (B-D2's ordering), only where it
    sits.
    """
    prefixes = DENY_PREFIXES if for_write else DENY_PREFIXES_READ
    prefix = prefix_hit(resolved.as_posix(), prefixes, fold=fold)
    if prefix:
        return f"under the protected prefix {prefix!r}"
    return None


def name_component_deny_reason(
    resolved: Path, *, for_write: bool, fold: bool = False
) -> Optional[str]:
    """The *identity* half of :func:`deny_reason` — components, names, globs."""
    components = DENY_COMPONENTS_WRITE if for_write else DENY_COMPONENTS_READ
    hit = component_hit(resolved.parts, components, fold=fold)
    if hit:
        verb = "write to" if for_write else "read from"
        return f"inside {hit!r}, which connectors never {verb}"

    name = resolved.name.casefold() if fold else resolved.name.lower()
    if name.startswith(DENY_FILE_PREFIXES):
        return "a secrets/config file (its name starts with '.env')"
    if name in DENY_FILENAMES:
        return f"a credential file ({resolved.name})"
    if name.endswith(DENY_FILE_SUFFIXES):
        return f"a private key or certificate ({resolved.name})"
    pattern = glob_hit(resolved.as_posix(), DENY_PATH_GLOBS, fold=fold)
    if pattern:
        return f"a browser profile/credential store (it matches {pattern!r})"
    return None


def deny_reason(
    resolved: Path, *, for_write: bool, fold: bool = False
) -> Optional[str]:
    """Explain why ``resolved`` is off-limits, or None when it is allowed.

    Reads and writes share the shape of the test but not its list: writing is
    refused under every protected prefix, reading only under the ones that hold
    credentials (``/etc/hosts`` is fair debugging material, ``~/.ssh`` is not).

    The order is the one the scratch allowance is measured against (B-D2):
    location first, then identity — a caller that wants the allowance to skip
    only the location half asks the two helpers directly.
    """
    reason = prefix_deny_reason(resolved, for_write=for_write, fold=fold)
    if reason:
        return reason
    return name_component_deny_reason(resolved, for_write=for_write, fold=fold)


def mount_form_to_drive(text: str) -> Optional[str]:
    r"""The MSYS mount form ``/c/dev/x`` as ``C:/dev/x``, or None when it is not one.

    Git's Windows toolset accepts this spelling of a drive path (target B1's
    probe: ``ls.exe`` takes ``/d/dev/benchmarkit/docs`` and ``D:\dev\...``
    alike). The translation exists so a *policy verdict* can ask the Windows
    ``Path`` questions about such a token while argv reaches the tool unchanged
    (G-D6). Pure text (roadmap §9 rule 2): the caller decides when it applies —
    only for a verdict under a drive-rooted root, never for a POSIX one, where
    ``/c/x`` is an ordinary absolute path.
    """
    if root_form(text) != "absolute":
        return None
    parts = text.replace("\\", "/").split("/")
    # parts[0] is "" for the leading separator; parts[1] is the mount letter.
    if len(parts) > 1 and len(parts[1]) == 1 and parts[1] in _ASCII_LETTERS:
        rest = "/".join(parts[2:])
        return f"{parts[1].upper()}:/{rest}" if rest else f"{parts[1].upper()}:/"
    return None


def root_form(text: str) -> str:
    """How ``text`` is rooted: ``"drive"``, ``"unc"``, ``"absolute"`` or ``"relative"``.

    Pure text, and correct on every runner, because :func:`ntpath.splitdrive` is
    a *text* function importable on Linux (roadmap §9 rule 2 -- the Windows
    shapes are decided by their shape, not by ``sys.platform``). Backslashes are
    read as separators, which is what a Windows ``Path.as_posix()`` string looks
    like: separators folded, drive kept.

    The four answers are not styles, they are *roots*: ``/etc`` and ``/usr``
    speak to each other, ``C:/Users`` and ``D:/src`` speak to each other, and
    nothing speaks to a drive-spec path except a drive-spec prefix. That is the
    whole basis of :func:`lists_are_meaningful` below, and it is why the
    classification stops at the root rather than comparing component lists --
    a case fold is win-05's business, asked separately.
    """
    normalised = text.replace("\\", "/")
    drive, _rest = ntpath.splitdrive(normalised)
    if drive.startswith("//"):
        return "unc"
    if len(drive) == 2 and drive[1] == ":" and drive[0] in _ASCII_LETTERS:
        return "drive"
    if normalised.startswith("/"):
        return "absolute"
    return "relative"


def lists_are_meaningful(text: str, prefixes: Optional[Iterable[str]] = None) -> bool:
    """Does any of these prefixes speak ``text``'s path form at all? (win-07 / A-D4.)

    A prefix is honoured by text comparison (:func:`prefix_hit`), so a list of
    ``/etc``-shaped entries covers *nothing* once the paths are ``C:/…`` -- and
    the miss is silent: the verdict falls back to allowing, which is the
    shape this whole lot exists to remove. The answer here lets both hubs say
    "these guardrails do not bound anything on this filesystem" instead of
    quoting a bound they do not have.

    Deliberately **not** "is this path denied?": it asks whether the two sides
    are spellable at all, so a ``True`` answer means the list is worth honouring,
    not that the path is safe. The comparison is exact-text on the rooted part
    (``root_form``) -- case folding is a property of the volume, not of the
    spelling, and arrives through B.

    ``prefixes`` defaults to the three prefix lists this module itself builds
    (``ALLOW_PREFIXES`` / ``DENY_PREFIXES`` / ``DENY_PREFIXES_READ``); the LSP
    hub passes its own lines, so the check is asked of the list that is actually
    in force there. The Windows *values* arrive through the environment at read
    time (B-D1/G-D8) — A added the check, target B wrote the lists. An empty
    list answers ``False`` (there is nothing to speak); a caller that reads that
    as a refusal is expected to decide so itself, from a non-empty list.
    """
    form = root_form(text)
    if prefixes is None:
        prefixes = ALLOW_PREFIXES + DENY_PREFIXES + DENY_PREFIXES_READ
    return any(root_form(prefix) == form for prefix in prefixes)
