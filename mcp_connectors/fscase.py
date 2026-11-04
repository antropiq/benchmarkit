"""Does this filesystem treat ``Dist`` and ``dist`` as one name? (win-05, gb-01)

One question, asked per **root** and answered by *trying*, never by asking
which operating system this is (roadmap §9 rule 2): a case-insensitive mount on
a Linux box answers ``True`` and a volume configured case-sensitive answers
``False``, and no name of a platform can see either of those.

It lives in ``mcp_connectors/`` because **both** surfaces ask it. The app's
verdict (:func:`editorfs.policy_modes.mode_for`, pol-06) threads the answer to
the matcher as an explicit ``fold`` argument; the console connector threads the
same answer to ``path_policy`` for the paths its commands name (target B1 /
G-D8). The matcher (:mod:`mcp_connectors.policy_glob`) and the deny-lists
(:mod:`mcp_connectors.path_policy`) stay pure text, so the one place allowed to
touch disk for this answer is here — stdlib-only, importable by the two stdio
connector processes and by ``src`` alike.

Why the answer has to come from the filesystem at all: on NTFS a rule
``{"path": "dist", "mode": "hidden"}`` that compares exactly leaves
``GET /api/fs/file?path=Dist/app.js`` matching **nothing**, and a read of
``.NOJAE\\x`` walks past the reserved-folder check — the silently-permissive
shape this lot exists to remove. Folding *unconditionally* is the mirror
mistake (a workspace holding ``dist`` and ``Dist`` as two folders would have
them merged for policy), so the value travels as an explicit ``fold`` argument
and this module only supplies it.

The probe is the whole file, plus its cache:

* **it writes two names and removes them again.** They are this function's own
  bytes, created and unlinked inside one call, so plain ``os.unlink`` is right
  here and the Editor's trash rule (a delete *moves* bytes) is not: there is
  none of the user's bytes to keep.
* **a root it cannot write to answers ``False``** — *fold nothing you cannot
  prove*. A wrong ``False`` is over-strict, which is the safe direction for a
  security matcher: it hides or locks what it would otherwise have let through,
  and it never opens anything.
* **the cache is keyed on the root**, for the same three reasons
  :mod:`editorfs.policy_file` spells out for its own cache (a hot
  ``BENCHMARKIT_PATH_LOCATION`` swap, two roots holding look-alike files, and a
  suite that patches the location per test), and it is forgotten at the **same
  two moments** that one is: the seeder's write and the settings switch
  (AGENTS.md: *"a switch invalidates them all at once"*).

One ordering fact keeps the probe invisible on the app side: the first question
about a root is always asked by the gate resolving the *requested* path, which
happens before any directory walk of that root starts. So the probe file is
created and removed before the ``scandir`` that lists the root, and the cached
answer is what every verdict of that walk reads — a probe name does not turn up
as a phantom entry in the listing that asked for it.

Nothing here branches on the platform: the probe *is* the answer. A test reaches
the other outcome by forcing the cached value, which is what keeps both branches
exercised on the Linux runner.
"""

import os
import threading
import uuid
from typing import Dict

# How many roots an answer is remembered for — the same bound and the same
# argument as ``policy_file._CACHE_MAX_ROOTS``: a process switches working
# location rarely, so this guards a long uptime rather than a working set. Past
# it the cache is dropped whole, which costs a re-probe and never an answer.
_CACHE_MAX_ROOTS = 32

_CACHE: Dict[str, bool] = {}
_LOCK = threading.Lock()

# The probe file's own prefix. It is deliberately *not* the reserved folder name
# (that one is unreachable by design) and it is not policy-checked: the probe
# writes into the root it is asking about, which the app owns as its working
# location. One call creates and removes it, so it never reaches a listing.
_PROBE_PREFIX = ".benchmarkit-case-probe-"


def case_folded(root: str) -> bool:
    """True when names differing only in case are **one** name under ``root``.

    The root string is used as handed in — the app passes
    :func:`editorfs.policy.workspace_root`, already a ``realpath``, and the
    console connector passes its own resolved ``ROOT_DIR`` — so two spellings
    of one folder cost one extra probe and nothing else. Never raises: a root
    that is missing, unreadable or unwritable answers ``False``.
    """
    with _LOCK:
        cached = _CACHE.get(root)
        if cached is not None:
            return cached

    answer = _probe(root)

    with _LOCK:
        if len(_CACHE) >= _CACHE_MAX_ROOTS:
            _CACHE.clear()
        _CACHE[root] = answer
    return answer


def invalidate_case_cache() -> None:
    """Forget every cached answer. For tests and for a workspace switch."""
    with _LOCK:
        _CACHE.clear()


def _probe(root: str) -> bool:
    """Ask the filesystem once, leaving nothing behind either way.

    Create a cased name, ``stat`` the same name in its folded spelling, and
    compare by **identity** (``st_dev`` + ``st_ino``, which says "one file"
    rather than "two files whose metadata happens to agree"). The folded
    spelling is what is looked for because a case-insensitive filesystem that
    *preserves* the case it was given — NTFS does — then still answers, while a
    case-sensitive one has no such name and the ``stat`` simply fails.
    """
    base = f"{_PROBE_PREFIX}{uuid.uuid4().hex}"
    cased = os.path.join(root, f"{base}Abc")
    folded = os.path.join(root, f"{base}aBc")

    try:
        with open(cased, "wb"):
            pass
        try:
            made = os.stat(cased)
            other = os.stat(folded)
        except OSError:
            return False
        return (made.st_dev, made.st_ino) == (other.st_dev, other.st_ino)
    except OSError:
        # A read-only mount, a root owned by someone else, a root that is not
        # there: fold nothing you cannot prove.
        return False
    finally:
        # Both spellings: on a folded filesystem they name one file, on an exact
        # one the second unlink just fails. Best-effort — a leftover probe file
        # would be a nuisance, not a policy answer.
        for name in (cased, folded):
            try:
                os.unlink(name)
            except OSError:
                pass
