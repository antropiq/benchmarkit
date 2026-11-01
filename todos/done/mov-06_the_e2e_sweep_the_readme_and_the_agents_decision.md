# mov-06 — the Playwright sweep, the README paragraph, the AGENTS.md decision

Prove the batch in a real browser — the things jsdom cannot show: a `Ctrl`+click that
marks two rows, a modal that survives a refusal, a prefill whose exact text is asserted,
and two files that land on disk. Then write the one paragraph a human needs about what
moving a file inside `todos/` does to a running lot, and make the AGENTS.md call.

This is the last task and the only one that runs the whole feature.

Depends on: **all of mov-01 … mov-05.**
Read `docs/tree-move-batch-spec.md` §7's e2e list and §5.3's closing paragraph (the
`todos/` consequence), plus §8 in full.

## Files

| file | change |
|---|---|
| `web/e2e/02-editor-write.spec.js` | the new flows, after the existing rename tests and before the markdown ones, in that file's own style |
| `README.md` | one paragraph about moving files into and out of `todos/` |
| `AGENTS.md` | **probably nothing** — see the decision below. Only the write-surface bullet, if its wording turns out to be wrong |

Nothing else. If a flow fails, the bug is in an earlier task: **fix that task's code, not
the assertion.** Weakening an assertion to make a spec pass is the one way this lot ships
a broken feature, because these tests are the only evidence the gesture works at all.

## The e2e flows (§7, one test each)

Seed through the API (`POST /api/fs/create`, then `PUT /api/fs/file`) — byte-exact content
with known names — and trash the scratch folder in `beforeEach`/`afterEach` exactly as the
file already does, so a leftover from an interrupted run cannot turn this run's create into
a 409 or its move into a collision. Use a **top-level** scratch folder (a `logs/…` one is
hidden by benchmarkit's own rules and would make the tree list nothing for the wrong
reason). The scratch tree is `<scratch>/nested/a.any`, `<scratch>/nested/b.any`,
`<scratch>/nested/c.any` and `<scratch>/other/` — **never the real `todos/`**: the `todos/`
paths in the spec illustrate the gesture only, and a test that moves a lot file would edit
this repository's own lot.

1. **the batch.** `Ctrl`+click two of the three `nested/` rows → right-click one →
   `Move to…` → type `<scratch>/other` → Move. Assert both rows are in the destination
   folder in the tree **and** both exist on disk through `/api/fs/stat` (the server is the
   witness, never what the DOM drew), with no error notice. Then: open one of them in a
   buffer, type into it (so it is dirty), move the pair, and assert the buffer is **still
   dirty and its next save writes to the new path** — the rename flow's proof, at batch
   scale.
2. **the N=1 case.** A plain right-click on one file with no marks → `Move to…` → assert
   the field is prefilled with the file's **own folder, spelled exactly as the tree spells
   it**: `<scratch>/nested` — not the absolute working location (read it from
   `GET /api/path-location` and assert the value is *not* that), not the bare `nested`.
   Edit it to `<scratch>/other`, move, assert one row moved and its tab re-pointed.
3. **a refusal.** Destination typed as `<scratch>/nope` (not there) → the dialog is **still
   open**, the field still holds what was typed, the server's not-there sentence naming that
   folder is visible under the field, and every source is still at home on disk.
4. **the client's own refusals, before any request.** A destination with a **leading slash**
   is refused in the field with **no `/api/fs/move-batch` request going out** (a
   `page.on('request')` counter is the honest form of "before any request"). A destination
   typed as a **bare segment** (e.g. `nested`) does reach the server and comes back with the
   same not-there sentence — proving §2.3's rule that a bare name resolves from the
   workspace root, not from the sources' parent.
5. **the narrowing.** `Ctrl`+click one `nested/` file and one `other/` file (marks across
   parents) → right-click one → the dialog's list names exactly **one** path and the title
   says one file. Move it, and assert the other mark did **not** move.

Also assert the marked rows' look changed (the `Ctrl`+click case reads the row's class or
`aria-selected`) — that is the only place a browser proves the marks are visible at all.
Use `expect(…)` with its own wait; never `waitForTimeout`.

Harness facts, already settled — do not re-decide them: one server on its own port
(`E2E_PORT ?? 8123`), the endpoint from `.env.test` and never the developer's `.env`,
`globalSetup` builds `web/dist` itself, `BENCHMARKIT_DISABLE_MCP=1`/`_LSP=1` for the whole
run, `fullyParallel: false`, `workers: 1`. **No LLM call is needed by any of this**, which
makes it the cheapest e2e proof in the repo.

## The README paragraph (§5.3's consequence)

Add it where the README already describes the Editor's write loop (the Features bullet about
the Editor tab / the deleted-files-not-lost bullet, or the batch-mode area if you find one
closer — read the file and choose, do not start a new top-level section). Two or three
sentences, in the README's voice:

- `Move to…` moves **whole files into an existing folder**, one request, all-or-nothing;
- `todos/` is the batch lot's folder, so moving a task file **into** `todos/` re-arms it as
  a pending task on the loop's next listing, and moving one **out** retires it — that is the
  mover working correctly (the loop reads the filesystem, not a flag), stated so the gesture
  is not mistaken for a no-op;
- and no guard was added for it, deliberately: the app cannot know a `.md` is a task.

## The AGENTS.md decision (§8)

Read the bullet that says a `/api/fs` **write** route is reachable from the SPA only, that
every write route is refused while a chat reply streams, and that names
`tests/test_write_surface_isolation.py` as the test that asserts it.

**The expected answer is: change nothing.** That bullet is written as an invariant about the
class of route, not as a list, and mov-01 registered the new route in the very test the
bullet already points at. Adding "including the batch move" is a count-shaped edit the file's
own rules forbid, and it would go stale the next time a route is added.

Only edit the bullet if reading it now would **mislead** — e.g. if it happens to name the
write routes individually, in which case shrink it to the class statement rather than adding
a name. State your decision and its one-line reason in the task report either way, so the
next reader knows the question was asked. Keep `AGENTS.md` under ~300 lines and one idea per
bullet if you do touch it.

## Rules this task must not break

- *A `/api/fs` write route is reachable from the SPA only*: this task adds tests, not
  callers. No connector, hub or chat module may gain a reference to the new route.
- The e2e specs in this repo are **self-contained on purpose**: reuse `02-editor-write.spec.js`'s
  local helpers (`openEditor`, `openFolders`, `useMenuItem`, `create`, `trash`,
  `diskExists`) rather than importing from another spec file.
- e2e asserts on the **server's wording** for refusals (that file's standing rule: a rule a
  human cannot read is a rule that gets worked around) and on disk state for success.
- The scratch is always trashed afterwards; no test may leave bytes in the real tree, and
  none may touch `todos/`, `resources/` or `.env`.
- The README and AGENTS.md edits are prose about invariants, not feature announcements: no
  route lists, no counts, no "new in this version".

## Done when

```bash
cd web && npm run test:e2e              # the whole e2e suite, including the arrays mov-05 fixed
pytest -q                               # backend green, guards untouched
cd web && npm test                      # frontend unit suite green
cd web && npm run build                 # web/dist exists and is current
```

All five flows pass, the README paragraph is in, and the report states the AGENTS.md
decision with its reason. When this task finishes, move its own file to `done/` like the
others.

## Out of scope here

new source-code behaviour of any kind; a performance test of a 200-file batch (the cap is
already unit-tested); visual regression of the marked row tone; a docs page for the feature;
moving the existing rename e2e tests; anything about drag-and-drop.
