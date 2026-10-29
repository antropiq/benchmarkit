# pol-05 — Seed a default policy the moment a workspace is targeted

A workspace gets its `.nojae/read-write-policy.json` **when it becomes active**,
so the user sees the syntax as a worked example instead of discovering a config
file exists. This task also decides the last open question of pol-00: the
seeded rules *replace* the hardcoded `IGNORED_DIR_NAMES`, so the policy file is
the only visibility mechanism (D7) — deletion of the constant itself is pol-06's.

Depends on: pol-01 (constants), pol-04 (the loader validates what we write — the
seed must round-trip through it, byte for byte).

## 1. The helper

```python
# src/core/editorfs/policy_seed.py
def ensure_policy_file(root: str) -> Dict[str, Any]:
    """Create `<root>/.nojae/<POLICY_FILENAME>` if absent. Never overwrites.

    Returns {"state": "created" | "existing" | "failed", "path": …,
             "error": Optional[str]}.

    The whole function is one try/except: a seed that cannot happen (read-only
    mount, root owned by someone else, no disk) is a NOTE for the UI, never a
    reason to refuse opening a project.
    """
```

Rules that are not negotiable:

* **Exclusive create.** `os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL,
  0o644)` — one syscall for check-and-create, so a user's edited file is never
  clobbered and two concurrent requests have one winner. An `EEXIST` is the
  `"existing"` state, not an error. A hand-written `if os.path.isfile(): …`
  before the write is a TOCTOU and must not appear.
* **The directory is created with `os.mkdir`, one level, not `makedirs`** — the
  root already exists by contract, and a `.nojae` whose parent is missing is a
  symptom worth failing on, not a chain to paper over.
* **Never called from `path_location()`** (`config.py:281`), which is a
  per-request *reader*. A reader that writes is how a stray folder shows up in a
  project the user only meant to browse.

## 2. The two call sites

| when | where | why there |
|---|---|---|
| boot | `config.setup_path_location()` (`config.py:268`), after the env value is settled — in **both** branches, so an operator-set `BENCHMARKIT_PATH_LOCATION` gets seeded too | this is how benchmarkit gets its own `.nojae` on first launch |
| UI switch | `POST /api/path-location`, **after** the two hub restarts (`routes.py:1221-1222`) | before them, a seed failure would look like a hub failure and trip the env-rollback at `routes.py:1229-1232` |

Import direction matters at boot: `src/config.py` may not import
`src.core.editorfs` (config is the bottom of the dependency graph — `editorfs`
imports `path_location` *from* it). So the seeding call at boot goes through a
**local import inside the function**, wrapped in its own try/except, with a
comment saying exactly that. If that feels wrong, the alternative is to call
`ensure_policy_file()` from `api/server.py:49` right after
`setup_path_location()` — one line, no import cycle, and the app is the only
thing that boots. Prefer the server.py form; note the choice in the code.

## 3. The payload and the UI hook

`_path_location_payload()` (`routes.py:1138`) gains one key:

```json
{"key": "BENCHMARKIT_PATH_LOCATION", "path": "…", "default": "…",
 "policy": {"state": "created", "path": "/srv/app/.nojae/read-write-policy.json",
            "error": null}}
```

`GET /api/path-location` returns the same shape, with `"state"` = `"existing"`
when the file was already there — the settings dialog can then say *"this project
has its own policy"* rather than the user finding out by accident.

Frontend: `useBenchmark.js` already posts the switch and re-roots the tree
(`:804-852`). Add, from the response body:

* `state.pathLocationPolicy = {state, path, error}`;
* a `failed` state appends one line to `state.pathLocationError`-adjacent text —
  the switch **succeeded**, so say "working location set; could not create the
  policy file: …" in `SettingsDialog.vue`, and let the Editor banner (pol-07)
  carry it too;
* the existing `reRoot(state.pathLocationCurrent)` call stays as it is: seeding
  happens server-side before the tree re-lists, so the new tree already arrives
  filtered.

## 4. The seeded content

```json
{
  "version": 1,
  "readme": [
    "benchmarkit reads this file when it browses this folder.",
    "Rules are matched in order and the LAST match wins, so a later rule carves",
    "an exception out of an earlier one:",
    "    {\"path\": \"**/node_modules\", \"mode\": \"ro\"},",
    "    {\"path\": \"**/node_modules/**/README.md\", \"mode\": \"rw\"}",
    "path is relative to this folder. * stays inside one name, ** crosses",
    "directories, and a rule naming a folder covers everything inside it.",
    "mode: hidden = not shown and not reachable; ro = browsable and readable but",
    "nothing under it changes; rw = the default, and what an unmatched path gets.",
    ".nojae itself is always hidden and is never listed by a rule.",
    "This file is machine-local: it is gitignored, and benchmarkit does not edit it."
  ],
  "policies": [
    {"path": ".git",             "mode": "hidden"},
    {"path": ".hg",              "mode": "hidden"},
    {"path": ".svn",             "mode": "hidden"},
    {"path": "**/__pycache__",   "mode": "hidden"},
    {"path": "**/.pytest_cache", "mode": "hidden"},
    {"path": "**/.mypy_cache",   "mode": "hidden"},
    {"path": "**/.ruff_cache",   "mode": "hidden"},
    {"path": "**/node_modules",  "mode": "ro",  "reason": "browse the docs, edit nothing"},
    {"path": "**/.venv",         "mode": "ro"},
    {"path": "**/venv",          "mode": "ro"},
    {"path": "**/dist",          "mode": "ro"},
    {"path": "**/build",         "mode": "ro"}
  ]
}
```

The differences from pol-04's example are deliberate: `ro` for dependency and
build trees (their *documentation* is the reason to keep them visible — the
`node_modules` reasoning from pol-00), `hidden` for pure derived noise. And the
`readme` line about last-match-wins carries the same JSON example pol-00 uses, so
the two documents teach one syntax.

Store the seed as a **module-level constant string** in `policy_seed.py`
(`DEFAULT_POLICY_JSON`), formatted with `json.dumps(..., indent=2)` of a Python
literal rather than a hand-maintained blob: it must round-trip through pol-04's
loader, and a test asserts that.

## 5. `.gitignore`

One entry, with the same two-line rationale the file already uses for `.env.test`
and the hub declaration files:

```
# The workspace policy folder: machine-local per checkout, and the app reserves
# it (hidden and unwritable from the Editor), so it is never source.
.nojae/
```

Anchor-free (a leading `/` would only ignore the app's own), so it also covers a
foreign workspace that happens to be a git repo — but the entry only matters for
this repo; the app cannot write anyone else's `.gitignore` and must not try.

## 6. Tests

New `tests/test_editorfs_policy_seed.py`:

1. `ensure_policy_file(tmp)` creates `.nojae/read-write-policy.json`, valid JSON,
   `version` present, `policies` a non-empty list;
2. a second call returns `"existing"` and leaves the bytes untouched (write a
   sentinel into the file first, re-call, compare);
3. `mode_for` answers for the seeded rules immediately after seeding, in that
   workspace — the seed and the loader agree on their own file (this is the test
   that catches a syntax drift between the two);
4. a read-only parent (patch `POLICY_DIRNAME`'s parent, or `chmod` the root to
   `0o500`) → `"failed"` with an error string, **no exception**;
5. `POST /api/path-location` on a scratch dir answers 200 with
   `policy.state == "created"`, and a second call answers `"existing"`;
6. the payload of `GET /api/path-location` carries the `policy` block;
7. `.nojae` does not appear in the tree of that workspace, and `stat`/read on
   `.nojae/read-write-policy.json` is refused — via the reservation, before the
   loader is even asked.

## 7. Acceptance

* Boot benchmarkit with no `.nojae`: after start, the app's own
  `.nojae/read-write-policy.json` exists at the repo root, is gitignored, and
  `git status` stays clean.
* Point the working location at a fresh project: the switch succeeds, the file
  appears, the tree arrives already without `.git` and `__pycache__`.
* Point it at a read-only dir: the switch still succeeds, and the note is visible.
