"""The path-aware matcher both policy levels are evaluated with (pol-02).

One question, answered identically for every surface and both processes: *does
this pattern match this workspace-relative path?* Nothing here reads the
filesystem, resolves a root or knows what a mode is -- pol-04 wires the matcher
to rules, pol-06 to paths on disk, and pol-08 lets the two stdio connectors ask
it too. That is why the module is stdlib-only and lives outside ``src`` beside
:mod:`mcp_connectors.policy_shared`, whose docstring records the packaging
constraint (the wheel ships ``src`` only, and a 400-line connector must not
pull in FastAPI to compare two strings).

Why not ``fnmatch``: its ``*`` matches any characters *including* ``/`` (what
``path_policy.glob_hit`` relies on today, line 132 -- harmless for a one-level
deny-list entry), so ``web/*`` would cover ``web/src/api/routes.py`` as well as
``web/index.html``, and a lone ``*`` would match everything. A user-authored
policy is read the other way: ``*`` means *the names in this folder*, so the
over-match silently hides or locks whole trees.
``pathlib.PurePosixPath.match`` is no help either -- a pattern with no ``**`` in
it is right-anchored, so it compares a *suffix*
(``PurePosixPath("a/b/c").match("b/c")`` is ``True``), which is exactly the
surprise a policy author must not get. The semantics below are written out
instead of delegated.

The vocabulary pol-00, pol-04 and pol-06 rely on:

* ``*`` matches any characters **within one name**, ``?`` one character, and
  ``**`` **zero or more whole path components**.
* **Component boundary, always**: a wildcard ends on a ``/`` or at the end of a
  name, never mid-name -- unless the pattern says so (``.env*``, ``node*``). So
  ``web/node`` never matches ``web/node_modules/x``. Case-sensitive on POSIX:
  ``Dist`` and ``dist`` are two directories.
* **A rule naming a directory covers its subtree**: the pattern is tested as a
  whole-component prefix of the path too, so ``web/node_modules`` matches
  ``web/node_modules/react/index.js`` without writing ``web/node_modules/**``.
  This is what makes ``{"path": ".git", "mode": "hidden"}`` hide everything
  inside it.
* No character classes: ``*``, ``?`` and ``**`` are the only special spelling,
  every other character (``[`` included) is literal, so an author gets exactly
  what they typed.

One thing besides matching lives here: :func:`reserved_component_hit`, the
component test for the folder that holds the policy (pol-00 D5). It is not a
pattern, but it is the same kind of pure text answer about a workspace-relative
path, and pol-08 needs the two stdio connectors to reach that one too.

The workspace root itself (``rel == ""``) is matched by **no** pattern: the only
pattern that would mean "this whole workspace" is a bare ``**``, and
:func:`pattern_error` refuses it, so the root is never caught by accident.
"""

from typing import List, Optional

# The two characters with meaning inside a name. Everything else, ``[``
# included, is literal.
_WILDCARD_CHARS: frozenset[str] = frozenset({"*", "?"})

# ``**`` is the only component that crosses a directory boundary.
# Public because pol-04's parser names it too, when it asks what a pattern's
# components mean rather than matching them: the spelling of the one
# cross-directory wildcard belongs to this module, and a second literal ``"**"``
# somewhere else is a second definition of the language.
CROSS_COMPONENT = "**"


def pattern_matches(pattern: str, rel: str) -> bool:
    """True when ``pattern`` matches the normalised workspace-relative ``rel``.

    Both sides are POSIX-joined relative paths: no leading ``"./"``, no trailing
    ``"/"``. The workspace root (``""``) is matched only by a pattern that means
    it, never by accident -- and no valid pattern does (see the module
    docstring), which is what keeps a rule about ``*`` from locking the root
    folder's own listing.

    Never raises on a weird pattern: one that :func:`pattern_error` rejects
    matches nothing. pol-04 turns that verdict into a message, and "a rule that
    cannot be read applies to no path" is the safe half of the deal.
    """
    if pattern_error(pattern) is not None:
        return False

    path = _path_segments(rel)
    if path is None:
        return False
    return _match_with_subtree(pattern.split("/"), path)


def literal_prefix(pattern: str) -> str:
    """The leading run of literal components of ``pattern``.

    ``"web/node_module/**"`` -> ``"web/node_module"``; ``"**/__pycache__"`` ->
    ``""``; ``"a/*/b"`` -> ``"a"``; ``".env*"`` -> ``""`` (a wildcard already in
    the first segment).

    pol-04 calls a rule **unreachable** when that prefix does not exist in the
    workspace, which is how a typo becomes a message instead of a rule that
    quietly never fires. An empty answer is legitimate and means "this rule
    starts with a wildcard, so there is no prefix to check" -- not a failure.
    """
    prefix: List[str] = []
    for component in pattern.split("/"):
        if not component or component in (".", ".."):
            break
        if _WILDCARD_CHARS.intersection(component):
            break
        prefix.append(component)
    return "/".join(prefix)


def pattern_error(pattern: str) -> Optional[str]:
    """Why ``pattern`` can never match a workspace-relative path, or ``None``.

    These are the shapes that cannot be what the author meant -- not the shapes
    the matcher would find awkward (:func:`pattern_matches` is total either
    way). The answer is a sentence, because pol-04 shows it verbatim next to the
    line that caused it.
    """
    if "\x00" in pattern:
        return "it contains a NUL byte"
    if not pattern:
        return "it is empty"
    if pattern.startswith("/"):
        return "it is absolute: a rule is written relative to the workspace"
    if pattern.startswith("~"):
        return "it starts with '~': there is no home directory here, only the workspace"
    if pattern.endswith("/"):
        return "it ends with '/': a rule names a path, it does not ask to descend one"

    components = pattern.split("/")
    for component in components:
        if not component:
            return "it has an empty component ('//')"
        if component == "..":
            return "it has a '..' component: a rule cannot reach outside the workspace"
        if component == ".":
            return "it has a '.' component, which a normalised path never has"

    if components == [CROSS_COMPONENT]:
        return (
            "it is a bare '**', which would apply to the whole workspace "
            "(did you mean '**/*'?)"
        )
    return None


def reserved_component_hit(rel: str, dirname: str) -> Optional[str]:
    """The component of ``rel`` equal to the reserved folder name, else ``None``.

    Checked on the **component list, before any ``stat``**: an absent path still
    has a name, so the answer for ``.nojae/whatever`` must not depend on whether
    the folder exists. It is not a pattern and is not matched by the matcher — a
    rule naming the folder that holds the rules is a contradiction (pol-00 D5),
    so the check has to survive a broken or absent policy file.

    It lives in this module because it is the same kind of pure text test, and
    pol-08 has the two stdio connectors make it too: they cannot import ``src``,
    so the one implementation has to sit where both reach it.
    """
    if not dirname:
        return None
    for component in rel.split("/"):
        if component == dirname:
            return dirname
    return None


# -- the matcher ---------------------------------------------------------------


def _path_segments(rel: str) -> Optional[List[str]]:
    """``rel`` as components, or ``None`` when it is not a normalised path.

    Callers normalise (``editorfs.policy.normalise_rel``), so an un-normalised
    ``"a//b"`` or ``"./a"`` reaching this far means a lock was skipped -- and
    answering "no match" is what keeps that mistake from being reported as a
    rule that hides a folder.
    """
    if not rel:
        return []
    components = rel.split("/")
    for component in components:
        if component in ("", ".", ".."):
            return None
    return components


def _match_with_subtree(pattern: List[str], path: List[str]) -> bool:
    """The direct match, plus the folder rule that covers its subtree.

    Testing the pattern against every whole-component prefix of the path *is*
    the subtree rule: ``web/node_modules`` matches the two-component path
    directly, and therefore everything below it at the cut where the pattern
    ends. The cut starts at 1, so the empty path -- the workspace root -- is
    matched by nothing.
    """
    for cut in range(1, len(path) + 1):
        if _match_segments(pattern, path[:cut]):
            return True
    return False


def _match_segments(pattern: List[str], path: List[str]) -> bool:
    """One pattern against one complete path, component by component."""
    if not pattern:
        return not path

    if pattern[0] == CROSS_COMPONENT:
        # ``**`` absorbs zero or more whole components, so the rest of the
        # pattern is tried against every remaining suffix of the path.
        for skip in range(len(path) + 1):
            if _match_segments(pattern[1:], path[skip:]):
                return True
        return False

    if not path:
        return False
    return _name_matches(pattern[0], path[0]) and _match_segments(pattern[1:], path[1:])


def _name_matches(component: str, name: str) -> bool:
    """One pattern component against one path component -- ``*`` and ``?`` only.

    ``current[n]``: does the pattern component so far match ``name[:n]``? Carried
    as two rows. The ``*`` row is the only interesting one: it takes zero or more
    characters **of this name**, and a name never contains ``/``, which is where
    the component boundary comes from for free. Comparison is exact -- no
    case-folding, the temptation that would make ``Dist`` and ``dist`` one
    folder.
    """
    previous = [False] * (len(name) + 1)
    previous[0] = True

    for index in range(1, len(component) + 1):
        char = component[index - 1]
        current = [False] * (len(name) + 1)
        if char == "*":
            for taken in range(len(name) + 1):
                current[taken] = previous[taken] or (taken > 0 and current[taken - 1])
        else:
            for taken in range(1, len(name) + 1):
                current[taken] = previous[taken - 1] and (
                    char == "?" or char == name[taken - 1]
                )
        previous = current

    return previous[len(name)]
