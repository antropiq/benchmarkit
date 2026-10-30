# search-09 — The Playwright sweep, and the one clause in AGENTS.md

Prove the feature in a real browser — including the two things jsdom cannot show: a
caret that lands on a line, and a modal that survives a re-open — and write the single
invariant worth six lines into `AGENTS.md`.

Depends on: **all of search-01 … search-08.** This is the last task and the only one
that runs the whole feature.
Read `docs/tree-search-spec.md` §6's e2e section and §7 in full; §7 is the wording the
`AGENTS.md` clause must land with.

## Files

| file | change |
|---|---|
| `web/e2e/06-tree-search.spec.js` | **new** — the five flows of §6 |
| `AGENTS.md` | one clause in the Editor's file-surface row |

Nothing else. If a flow fails, the bug is in an earlier task — **fix the earlier task's
code, not the assertion.** Weakening an assertion to make a spec pass is the one way
this lot can ship a broken feature, because this file is the only evidence the reveal
works at all.

## The spec

Alongside `02-editor-write.spec.js`, in its style, and reusing its helpers as that file
does locally (its `openEditor`, `openFolder(s)`, `useMenuItem`, `create`, `trash` shapes
are worth copying rather than importing — e2e specs in this repo are self-contained on
purpose, and search-07 already touched that file, so a diff there is not yours).

The harness facts, already settled — do not re-decide them:

- one server, its own port (`E2E_PORT ?? 8123`), the endpoint from **`.env.test`**,
  never the developer's `.env`; `globalSetup` builds `web/dist` itself, so a spec never
  tests a stale frontend;
- `BENCHMARKIT_DISABLE_MCP=1` / `_LSP=1` are set for the whole run, so no connector and
  no language server is involved — **this feature needs no LLM call at all**, which
  makes it the cheapest e2e proof in the lot;
- `fullyParallel: false`, `workers: 1`: one global server, so specs must not fight over
  it. A new spec file just joins the queue.

**Seed the files through the API, not through the pane.** `POST /api/fs/create` then
`PUT /api/fs/file` (the way `02-editor-write.spec.js:186-189` seeds known text) writes
byte-exact content with a known line number, which is what a caret assertion needs. Type
into Monaco only where the *point* of the case is typing. Trash the scratch folder in
`beforeEach`/`afterEach` exactly as that spec does — a leftover from an interrupted run
turns this run's create into a 409. Use a **top-level** scratch folder (a `logs/…` one is
hidden by benchmarkit's own level-A rules and would make a search find nothing for the
wrong reason).

### The five flows of §6, one test each

1. **folder → hit → caret.** Create `search-e2e/notes.md` with a known needle on a known
   line (say line 4), plus a decoy file elsewhere that does *not* contain it. Right-click
   the folder row → `Search…` → assert the scope label names that folder → type the word
   → the row appears → click it → the tab strip holds `notes.md` **and the caret is on
   line 4**. Assert the line, not a screenshot (§6's words): read Monaco's own position.
   The way that survives without poking at internals is the caret's rendered position in
   the DOM, e.g. `page.locator('.monaco-editor .cursors-layer > .cursor')` is present and
   — better — assert the *active line*'s rendered band: Monaco gives the current line a
   `.view-line` inside a `.monaco-editor` with `renderLineHighlight: 'line'`, so
   `.view-overlays .active-line` exists, and its y-offset is comparable against the row
   of the target line. Pick the assertion you can make deterministic and **say in a
   comment how you read the line number**, because the next reader cannot tell a clever
   assertion from a fragile one. A timing-sensitive assertion is worse than none: use
   `expect(…)` with its own wait, never a `waitForTimeout`.
2. **file scope.** Right-click one file row, search a word it contains, and assert every
   result row's path column is that same file — the scope narrowed to one path.
3. **root scope.** Right-click the root row: the label is the workspace's own name and
   **not** an empty string (`rootLabel()` is the basename of the working location, which
   the spec can read from `GET /api/path-location` the way `02-editor-write.spec.js:52`
   does — do not hardcode `benchmarkit`).
4. **regex on, `(` typed.** The inline error inside the dialog carrying `re`'s message,
   and the table still holds the previous answer (this is D5 and D14 meeting: an invalid
   pattern costs no scan and destroys no result). Assert the previous row is still on
   screen *after* the error, and that no notice-stack entry appeared.
5. **re-open keeps the answer.** Close the dialog (Escape), re-open it from the same row
   without searching: the same table is on screen and the field still holds the query —
   and assert on the network too, that **no second `/api/fs/search` request** went out
   (a `page.on('request')` counter is the honest form of §6's "without searching").

Plus three that the earlier tasks could not see from jsdom, cheap to add here:

6. **a hit inside a collapsed branch opens.** Seed the file two folders deep, never
   expand either, search from the root, click the hit, and the tab opens. This is the
   `entryFor`-returns-`null` trap search-05 had to handle — the case that silently
   opened nothing.
7. **the reveal beats a buffer swap.** Open a second file, then click a hit for a
   *third* file, then click back to the hit's file. The caret must still land on the
   match's line — the `setValue`-clears-the-cursor case (D11) seen from a browser, where
   the `MonacoEditor` instance was already mounted.
8. **a hidden path is absent from the payload, and its bytes are never quoted.** Do this
   one **through the HTTP layer only, against a name level A already hides** — the run's
   working location is this checkout, so `logs/**` is hidden **and already has files in
   it** (the app's own backups) — seed nothing, and do not write into `logs/` either.
   There is no needle that only a `logs/` file could hold, so assert the **shape**:
   `request.get('/api/fs/search?path=&q=e')` — a needle `logs/`'s content certainly
   contains — and assert **no row's `path` starts with `logs/`** and no
   `snippet` carries a backup path, while `matchCount` and `filesScanned` are both
   non-zero — proving the scan ran and still said nothing about the hidden tree. The
   payload must have **no `skipped.policy` key** (D2, from the live route). Also
   `?path=logs` as a scope ⇒ **404**, indistinguishable from a folder that is not
   there.

   Do **not** write to `.nojae/read-write-policy.json` from a spec: that is the
   developer's real policy file for this checkout, and a test that mutates it changes
   what their pane hides afterwards. Level-B rules belong to the backend suite, which
   owns a `tmp_path` (search-01's rows). Say that in one comment above this test so the
   next reader does not "extend" the spec into someone's config.

Also assert the **payload key set** once against the live route — the eight keys of the
map's Contract, spelled out — since a browser run is the only place the real route, the
real policy and the real frontend all agree.

## The `AGENTS.md` clause (§7, verbatim)

Add exactly this to the `core/editorfs/` row of the module table, appended to the
existing sentence and nothing longer:

> The Editor's search answers about the same set of paths the tree lists: a `hidden`
> folder is pruned and its bytes never quoted, `ro` is searched, and no count of
> policy-skipped files is reported — an observation of a hidden path is still
> indistinguishable from absence.

Rules for this edit, from `AGENTS.md`'s own *Editing AGENTS.md* section — read it before
writing:

- **one clause, no more.** No route list, no cap values, no column layout ("the route
  list, the caps and the column layout belong in the code");
- no counts: do not write "nine files", "500 matches" or "four caps";
- describe the app as it is now — no "search was added", no reference to the spec file's
  status line;
- stay under the file's ~300-line ceiling: `wc -l AGENTS.md` is 286 today, so this edit
  must fit in ~10 lines *including* the re-wrap of the sentence it joins. If the row has
  to grow, shrink another sentence in the same cell rather than letting the table cell
  sprawl;
- it feeds every WorkBench system prompt, so it is paid for on each message — that is why
  the invariant about *content* is the one kept and the mechanics are not.

## Acceptance

- `cd web && npm run test:e2e` green — the new `06-tree-search.spec.js` **and**
  `02-editor-write.spec.js` (whose menu lists search-07 updated) and the four other
  existing specs, unchanged. This task's diff must not be the reason any existing spec
  changed.
- `cd web && npm test` and `pytest` both still green (this task touches no code file).
- `wc -l AGENTS.md` ≤ 300, and `git diff AGENTS.md` shows **one** hunk in the
  `core/editorfs/` table row — no new section, no new bullet list, no other line of the
  file touched.
- `grep -c "search" AGENTS.md` is small (the clause plus at most the word in the row it
  joins): a feature this size earns one sentence, and `AGENTS.md` is the one document
  whose length is a cost.
- Every assertion in the new spec is on app state or on a rendered DOM fact, never on a
  sleep. Grep the file for `waitForTimeout` — there must be none.
- Say in the final note **which caret assertion you chose and why it is deterministic**,
  since it is the one assertion a future reader will be tempted to weaken.
- Move this file to `todos/done/` when the above is true — that also ends the lot:
  `search-00_big_picture.md` stays in `todos/` (it is the map, not a task, and the
  listing ignores `NN == 0` as a runnable task).

## Done — what to report

A short final summary for the human: the two prefixes' worth of tasks that ran (01-03
backend, 04-08 frontend, 09 the sweep), the one decision search-02 settled that the spec
left open (the exact `footerNote` wording), and anything the e2e run could not cover on
this machine (say so plainly rather than implying a green run that did not happen).
