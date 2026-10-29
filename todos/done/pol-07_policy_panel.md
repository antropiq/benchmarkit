# pol-07 — Make the policy visible: `GET /api/fs/policy`, and honest wording

A refusal the user cannot explain is a bug report, and a policy file nobody can
see is a guess about syntax. This task puts both on screen: the two levels' rules
and every finding from pol-04, plus the wording the Editor already carries (which
currently names a policy that no longer exists by then).

Depends on: pol-04 (the findings), pol-06 (the surfaces answer from one policy, so
the panel describes reality rather than one of two lists).

## 1. The route

```python
@router.get("/fs/policy")
def get_fs_policy() -> Dict[str, Any]:
    """What governs this workspace's Editor tree: both levels, verbatim."""
```

Read-only, like the other `/fs` reads; no body. It answers about the *active*
workspace, so a location switch re-fetches it (the tree already re-roots —
`useBenchmark.js:852`).

```json
{
  "workspaceRoot": "/home/vgr/anotherproject",
  "isSelfWorkspace": false,
  "file": {"path": "/home/vgr/anotherproject/.nojae/read-write-policy.json",
           "exists": true, "version": 1, "mtime": 1764300000.0},
  "reserved": [".nojae"],
  "builtin": [{"path": ".env**", "mode": "hidden", "reason": "…"}],
  "rules": [{"path": "**/node_modules", "mode": "ro", "reason": "…"}],
  "findings": [{"kind": "error", "index": 3,
                "message": "line 12 column 5: expected ',' — …"}],
  "blockingWrites": true
}
```

Notes on shape:

* `rules` (level B) and `builtin` (level A) are distinct keys, and neither is
  called `workspace` — the root path already owns that word, and a payload with
  two same-named keys is a bug waiting to be copy-pasted into a client.
* `builtin` is `[]` when `isSelfWorkspace` is false — the panel must show the
  *condition*, not a rule list that would look editable. Pair it with one caption
* The two rule lists are the same `Rule` objects the resolver used, so the panel
  cannot drift from behaviour (one source, the whole point of pol-01).
* `blockingWrites` is `policy_blocking_error() is not None`, so the Editor can say
  "edits are off until the file parses" before the user discovers it by failing to
  save.
* Do not include the file's raw text. The rules are what matter, and returning the
  file would make the route a read of a reserved path (pol-03 reserves `.nojae`
  from the file surface; this route reads it through the loader, which is the one
  reader pol-04 documents).

## 2. The composable

`web/src/composables/useEditorPolicy.js` — new, one singleton like the other
Editor composables (`useGitStatus.js` is the closest sibling: fetch, `state`,
`load()`):

```js
state = {
  loaded: false, loading: false, error: null,
  file: null, isSelfWorkspace: false,
  builtin: [], workspaceRules: [], findings: [], blockingWrites: false,
}
load({ force } = {})   // cached like useGitStatus; re-root forces a reload
```

Rules it owns:

* load on the Editor tab's first mount and on every `reRoot` (hook it into the
  same place `useWorkspaceTree().reRoot(...)` is called, `useBenchmark.js:852`);
* reload after a *failed* write, so a policy the user edited in an external editor
  is reflected without a page reload;
* no polling. The policy changes rarely and the user knows when they changed it.

## 3. The footer, and where findings go

`WorkspaceNavigator.vue:458-460` currently says *"Some files are hidden by the
connector policy."* By pol-06 that sentence names a policy that has nothing to do
with the tree. Replace it with three states, all in the same caption slot:

| condition | text |
|---|---|
| clean, rules present | `N policy rules — click to review.` (a button, not a `<p>`) |
| errors | `Policy error — edits are off.` in the existing notice/error colour, with the first finding's text as its `title` |
| no file, no findings | `No policy file for this folder.` plus the path it would be created at |

Clicking opens a small panel (a `Popover`/dialog in the `EditorTreeDialogs.vue`
shape — reuse that file rather than adding a third dialog component): both levels'
rules as rows, the reserved `.nojae` line, and the findings list with the absolute
file path in a `<code>` the user can select and copy. It is a *reader*, never an
editor (D5): no inputs, no save button, and one line saying
*"benchmarkit does not edit this file — change it outside the app."*

Styles go in `styles/theme.css` (frontend invariant), not inline.

## 4. One line in the settings dialog

`SettingsDialog.vue` owns the working-location input. Next to it, after a
successful switch, the `policy` block from pol-05's payload renders as one caption:

* `created` → `Default policy created in the new folder.`
* `existing` → `This folder keeps its own policy.`
* `failed` → `Could not create a policy file here: <error>` — in the warning
  colour, **not** `pathLocationError`, because the switch itself worked.

## 5. Tests

**pytest** (extend `tests/test_editorfs.py`, or a small
`tests/test_editorfs_policy_route.py` — keep the read-route suite about files):

1. the payload carries `version`, both rule lists and `reserved: [".nojae"]`;
2. self workspace → `isSelfWorkspace: true` and a non-empty `builtin`; foreign →
   `false` and `builtin: []`;
3. a malformed file → `findings[0].kind == "error"`, `blockingWrites: true`, and
   the route itself answers **200** (the panel must be able to explain a broken
   file — a 4xx here would hide the diagnosis);
4. no file → `file.exists: false`, `findings: []`, `blockingWrites: false`;
5. the route is a `GET` and appears in no write list — the existing
   `test_every_read_route_is_get_and_the_mutations_are_the_named_ones`
   (`tests/test_editorfs.py:440`) must gain it deliberately.

**Vitest** — `web/tests/useEditorPolicy.test.js`, with `fetch` stubbed (the style
of the existing `useEditorFilesDiff.test.js`):

1. `load()` once populates `workspaceRules` and caches; a second call issues no
   request, `{force: true}` does;
2. an unreachable API sets `error` and leaves the previous rules in place (the
   panel must not blank itself on a network hiccup);
3. `blockingWrites: true` is exposed as-is, not folded into `error`;
4. a location switch (`reRoot`) reloads with `{force: true}`.

**Playwright** (`web/e2e`, new spec or an added block in the editor one): switch
to a scratch workspace, open the panel, assert a rule from its seeded file is
listed by name — and assert `.nojae` is *not* in the tree. No LLM call needed.

## 6. Acceptance

* The footer names the real mechanism and counts the real rules.
* A JSON typo turns the footer into a visible error, puts "edits are off" in the
  panel, and the write route's 422 carries the same sentence.
* Nothing in the UI can change the policy file: the panel has no input, the tree
  has no row for `.nojae`, and no `/api/fs` write route will accept that path.
* `cd web && npm test` and `pytest` green; the lazy Editor chunk stays lazy
  (`useEditorPolicy.js` is imported from an already-lazy component, not from
  `App.vue`).
