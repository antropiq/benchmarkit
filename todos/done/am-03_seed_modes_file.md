# am-03 — Seed `.nojae/agent-modes.json` when a workspace becomes active

The modes file arrives the same way the policy file does: written once, at the two
moments a folder becomes the working location, and never again by the app. The
user gets the syntax as a worked example in a folder they can open outside the
app, rather than discovering a config file exists.

Depends on: am-01 (the validator, the cache, `POLICY_DIRNAME`), am-02
(`DEFAULT_AGENTS_DOCUMENT` — the thing being written).

## 1. The helper

```python
# src/core/agent_modes.py  (alongside the loader, NOT a new module)
def ensure_agent_modes_file(root: str) -> Dict[str, Any]:
    """Create `<root>/.nojae/agent-modes.json` if absent. Never overwrites.

    Returns {"state": "created" | "existing" | "failed", "path": str,
             "error": Optional[str]}.

    The whole function is one try/except: a seed that cannot happen (read-only
    mount, a root owned by someone else, no disk) is a NOTE for the UI, never a
    reason to refuse opening a project.
    """
```

Non-negotiables — copy them from `editorfs/policy_seed.py`, they are the same
three, and the reasoning is already written there (link to it in a comment rather
than restating it at length):

* **exclusive create**: `os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)`.
  One syscall for check-and-create, so a hand-edited file is never clobbered and
  two concurrent switches have one winner. `EEXIST` **is** the `"existing"`
  state, not an error. A hand-written `if os.path.isfile(): …` before the write is
  a TOCTOU and must not appear;
* **`os.mkdir` for `.nojae`, one level, not `makedirs`** — the root exists by
  contract, and a `.nojae` whose parent is missing is a symptom worth failing on.
  `EEXIST` on the mkdir is fine (the policy seed usually created it a moment
  earlier);
* **never called from `config.path_location()`** — that is a per-request *reader*,
  and a reader that writes is how a stray folder turns up in a project someone
  only meant to browse.

Write with `json.dumps(DEFAULT_AGENTS_DOCUMENT, indent=2, ensure_ascii=False)`
plus a trailing newline: the seed and the loader share am-02's object, and the test
in §4 asserts the round-trip.

## 2. The two call sites

| when | where | why there |
|---|---|---|
| boot | `api/server.py`, immediately after `_seed_workspace_policy()` (line 111), through a second tiny wrapper `_seed_agent_modes()` shaped like the existing one | the policy seed already proved the import direction: `src/config.py` sits at the bottom of the graph and may not import `core`, so the boot call belongs in the app that boots. Do **not** add a local import inside `setup_path_location()` — the precedent in `pol-05` was rejected in favour of this form, and now there are two seeds to place |
| UI switch | `POST /api/path-location`, **after** the two hub restarts, next to the existing `"policy": policy_seed.ensure_policy_file(expanded)` (line 1263) | before them, a seed failure would look like a hub failure and trip the env-rollback |

The `GET /api/path-location` / settings-read side (`routes.py:1157`, currently
`policy_seed.policy_file_state(...)`) gains the same treatment for modes:
`agent_modes_file_state(root)` — a **reader** that stats and reports, creating
nothing.

## 3. The payload

`_path_location_payload()` gains one key beside `"policy"`:

```json
{"key": "BENCHMARKIT_PATH_LOCATION", "path": "…", "default": "…",
 "policy": {"state": "existing", "path": "…read-write-policy.json", "error": null},
 "agentModes": {"state": "created", "path": "…/.nojae/agent-modes.json",
                "error": null}}
```

Settings dialog: a `failed` state appends one line to the switch's note — the
switch **succeeded**, so the sentence is "working location set; could not create
the agent-modes file: …" and the modes dropdown (am-07) falls back to the built-in
defaults without saying anything alarming.

## 4. Invalidation — the part that is easy to forget

A workspace switch changes the file being read, so the switch must:

1. call `invalidate_agent_modes_cache()` (am-01) — the cache is keyed on the root,
   so a switch would miss anyway, but an explicit invalidation next to the
   existing policy invalidation is what makes that obvious to the next reader;
2. do it **after** the seed, so a first read of the new root sees the freshly
   created file rather than a cached "absent".

Check how `pol-05`/`pol-06` order the policy invalidation on the same path and
match it; if the existing code invalidates before the hub restarts, put the modes
invalidation in the same place with the same ordering comment.

## 5. Tests — `tests/test_agent_modes_seed.py`

Shaped after `tests/test_editorfs_policy_seed.py` (read it first; reuse its
fixtures rather than inventing parallel ones):

* seeds into an empty temp root: file exists, parses with **zero findings**, and
  `resolve_modes()` on that root returns the three defaults + `todoBuilder`;
* **does not overwrite**: write a hand-edited file, call the helper, assert the
  bytes are unchanged and the state is `"existing"`;
* a second concurrent-ish call (just call it twice) is `"existing"`, not an error;
* `.nojae` absent → created; `.nojae` present → not touched;
* an unwritable root (`chmod 0o500`, or patch `os.open` to raise `EACCES`) →
  `{"state": "failed", "error": …}` and **no exception**;
* `agent_modes_file_state()` on a root with no file reports `exists: false` and
  creates nothing — the reader-must-not-write rule as a test;
* `POST /api/path-location` response carries `agentModes` with the right shape
  (extend the existing path-location test rather than adding a new endpoint test);
* the seeded bytes round-trip: `json.loads(seeded_text) == DEFAULT_AGENTS_DOCUMENT`.

## 6. Acceptance

* A brand-new workspace, opened through the settings switch, ends up with exactly
  two files under `.nojae/` and no other change to the tree.
* `.gitignore` already ignores `.nojae/` as a folder — check, and do **not** add a
  line for this one file (a per-file entry would suggest the folder is partly
  tracked).
* Nothing in this task reads the file for a *request* — that is am-06. If the
  seeding helper is also the loader, it is wrong.
