r"""What a path *text* is -- never whether it is allowed (win-02).

Three questions, answered the same way in every process and on every platform:
how a caller's path text becomes the canonical form, whether that text is
qualified (absolute in some way a workspace-relative path cannot be), and what
its components are. Nothing here knows a mode, a rule list or a filesystem.

Why it lives here rather than under ``src``: the two stdio connectors
(``fedit_mcp``, ``console_mcp``) run as scripts launched by absolute path and
cannot ``import src`` -- the wheel packages ``src`` only. It is the same reason
:mod:`mcp_connectors.policy_glob` (pol-02) and :mod:`mcp_connectors.policy_shared`
(pol-08) sit beside it: stdlib-only, no FastAPI, no ``src`` import. So do not
tidy this into ``editorfs`` -- the connectors would lose the answer and start
disagreeing with the Editor pane about what a path *is*, which is the disagreement
this module exists to prevent.

The three things a later contributor must not lose:

* **The canonical form is ``/``-joined workspace-relative text on every
  platform** (roadmap §9 rule 1). A backslash is accepted *only at the inbound
  lock* (``editorfs.policy.normalise_rel``, win-03) and is **never emitted**: not
  into a payload, the tree, ``results.json`` or the browser. So the Windows user
  typing ``src\config.py`` and the Linux test writing ``src/config.py`` mean one
  path, and one path string still means the same thing in a report from either box.
* **This module answers what a path text is, never whether it is allowed.** The
  verdict belongs to ``policy_modes.mode_for`` (pol-06, its one implementation)
  and the locks to ``editorfs.policy``.
* **It is pure text, so it carries no platform branch**: no ``sys.platform``, no
  ``os.name``, no ``os.sep``, no ``os.path.normpath``, no ``pathlib`` path class.
  Only :func:`ntpath.splitdrive` is consulted, and only because it is a *text*
  function -- importable and correct on Linux too, which is what lets the Windows
  shapes be tested on the Linux runner (roadmap §9 rule 2).
"""

import ntpath

# The two characters that separate components in an *inbound* path text. Only
# the slash ever comes back out.
_SEPARATORS: tuple[str, str] = ("/", "\\")

# A Windows drive letter is an ASCII letter, and no more: ``ntpath.splitdrive``
# is more permissive than the Windows parser it describes -- it answers
# ``("é:", "x")`` for ``é:x``, while ``ntpath.join`` treats that as an ordinary
# relative name. Refusing the whole ``splitdrive`` output would refuse legal
# POSIX filenames, so the drive part has to be constrained by hand.
_DRIVE_LETTERS: frozenset[str] = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
)


def canonical_rel(text: str) -> str:
    r"""``text`` in canonical form: both separators in, ``/`` out.

    Splits on the slash and the backslash, drops ``.`` components and duplicate
    separators, applies ``..`` **lexically**, and re-joins with ``/``. So
    ``src\config.py`` becomes ``src/config.py``, ``a//b`` and ``a\/b`` become
    ``a/b``, ``a\..\b`` becomes ``b``, and a Windows user typing
    ``todos\drafts`` gets the same folder as ``todos/drafts``.

    What it deliberately does *not* do:

    * **Translate away a leading separator.** ``/a`` stays ``/a`` (a leading
      backslash becomes that same leading slash, since only ``/`` is emitted).
      Whether a text is qualified is a separate question -- :func:`is_qualified`
      answers it, and the lock refuses before this is ever asked about such a
      text. Stripping the shape here is how the lock loses the thing to refuse.
    * **Drop a ``..`` that escapes the start.** It survives as a leading
      component: ``a/../b`` is ``b``, but ``../a`` is ``../a`` and ``a/../../b``
      is ``../b``. That is the property ``editorfs.policy.normalise_rel`` relies
      on (``policy.py:201``): a traversal that stays inside the workspace stays
      legal, and only a surviving ``..`` is refused.
    * **Touch whitespace.** The lock strips before calling.

    Idempotent, because a caller may hand it text that is already canonical
    (``batch.py`` re-normalises a lot folder it may have produced itself).
    ``""`` and ``"."`` both give ``""`` -- how the workspace root is spelled
    everywhere else in the app.
    """
    parts = text.replace(_SEPARATORS[1], _SEPARATORS[0]).split(_SEPARATORS[0])

    # A leading separator is kept, in the one spelling this module emits. Asked
    # of the text rather than of ``parts`` -- an empty path has an empty first
    # component too, and ``""`` is the workspace root, not something absolute.
    # A doubled one ("//srv/share", a UNC root) keeps a single leading slash: the
    # duplicate is a separator, and :func:`is_qualified` has already had its say
    # about the shape.
    leading = _SEPARATORS[0] if text[:1] in _SEPARATORS else ""

    resolved: list[str] = []
    for component in parts:
        if component in ("", "."):
            continue
        if component == "..":
            if resolved and resolved[-1] != "..":
                resolved.pop()
            else:
                # Nothing left to cancel it against -- keep it, so the caller's
                # own traversal check still sees the escape.
                resolved.append("..")
        else:
            resolved.append(component)

    return leading + "/".join(resolved)


def is_qualified(text: str) -> bool:
    r"""True when ``text`` cannot be a workspace-relative path on **any** platform.

    Each shape is a hole in ``os.path.isabs``, not a rule invented for Windows
    (all verified on 3.13 with ``ntpath``):

    * **A leading slash or backslash** -- ``/etc/x``, ``\Users\vgr``. On Windows
      ``ntpath.isabs("/etc/x")`` is ``False``, so such a text is not refused as
      absolute: it normalises to a backslash form and escapes the workspace
      through ``os.path.join`` (``join("C:\ws\a", "\etc\passwd")`` ->
      ``C:\etc\passwd``). The containment lock catches it there, but the
      *sentence* is then wrong -- "outside the working location" for a path that
      was absolute. A UNC root (``//srv/share``, ``\\srv\share``) is caught by
      this same rule, because it too begins with a separator; there is
      deliberately no third case for it (``ntpath.splitdrive`` would hand back the
      whole share as the drive, so the order of the two checks below matters).
    * **A one-letter drive spec** -- ``C:x``, ``c:/x``, ``C:`` alone.
      ``ntpath.isabs("C:x")`` is ``False``, and
      ``ntpath.join(root, "C:x")`` returns a path on *another* drive's current
      directory -- an escape the containment lock stops only by luck. This is its
      own check, never folded into the separator one.

    ``~`` stays the lock's business: it is a shell convention that no connector
    expands, and the check for it belongs where it already runs.

    **The single-letter constraint.** ``ntpath.splitdrive`` answers ``("a:", "b")``
    for ``a:b`` but also ``("foo:", "bar")`` for ``foo:bar`` -- a shape no Windows
    drive can be, and a workspace may legitimately hold a file named ``foo:bar``.
    So only an ASCII letter plus ``:`` qualifies: ``C:x`` is refused,
    ``foo:bar/src`` and ``http://host:8080/v1``-shaped text are not, and a colon
    anywhere *after* the first component (``src/a:b``) stays legal.

    **The residue, named rather than left to chance.** A leading one-letter
    component plus a colon is a drive spec on both platforms, so ``a:b`` is
    refused even though it is a legal POSIX filename -- a second widening of
    accepted input beyond the ``a\b`` one. No rule can refuse ``C:x`` without also
    refusing ``a:b``; win-03 puts that in the lock's refusal message, and the test
    here pins it in both directions.
    """
    if text[:1] in _SEPARATORS:
        return True
    if _drive_spec_at_start(text):
        return True
    # The drive rule is also asked of the canonical form: ``..`` cancellation can
    # *create* a leading drive spec (``a\..\C:x`` -> ``C:x``), and that text is
    # what a later ``join`` would receive.
    return _drive_spec_at_start(canonical_rel(text))


def _drive_spec_at_start(text: str) -> bool:
    """Does ``text`` begin with one ASCII letter plus ``:``? (A drive spec.)"""
    drive, _rest = ntpath.splitdrive(text)
    return len(drive) == 2 and drive[1] == ":" and drive[0] in _DRIVE_LETTERS


def path_parts(text: str) -> tuple[str, ...]:
    """``text`` as its components -- the one splitter every consumer shares.

    Exactly ``canonical_rel(text).split("/")``, so the ancestor walk of the strict
    write gate, the parent of a create, the ``.gitignore`` prune in the search, the
    batch lot folder and the matcher's own segment list agree **by construction**
    instead of by a coincidence of separator -- which is what silently degraded on
    Windows (roadmap §2.2 and §2.4).

    The contract is a *pre-normalised* input: callers pass what
    ``normalise_rel`` returned, so no leading separator and no surviving ``..``.
    It stays **total** anyway and never raises -- this module answers questions,
    raising is the lock's job. ``""`` gives ``()``: the workspace root has no
    components.
    """
    canonical = canonical_rel(text)
    if not canonical:
        return ()
    return tuple(canonical.split("/"))
