# Prompt helpers: `.nojae/prompt-helpers.json`, one per workspace

**Status**: spec written, implementation not started.

**Surface**: one file moves and is renamed; one new core module takes it over from
`src/config.py`; two existing routes gain a third `.nojae` block; the composable
that reads it reloads on a workspace switch. No new tab, no new layout, no change
to the bullet row's shape.

## 1. The need

The coloured bullets above the WorkBench textarea hold frequently-reused prompts
("check the app by running the unit tests", a commit-message recipe). They come
from `resources/memory_registration.json`, which has two problems that are really
one problem: the file belongs to the **app**, while what it stores belongs to the
**workspace**.

* Point the working location at another project and the bullets are unchanged —
  the helper prompts of *benchmarkit* are offered while the model works on
  something else. A helper is a habit of one tree ("run the pyright tool, avoid
  the console"), so it has to travel with that tree.
* It sits in `resources/`, the folder whose every `*.json` is a **question set**,
  which is why `config.list_question_sets()` has a name-based skip
  (`src/config.py:397-399`) — one folder holding two unrelated kinds of file, told
  apart by an `if`.

The name is wrong too: nothing in the feature is "memory" in the model's sense —
they are pre-written prompts. `prompt-helpers.json`.

## 2. What exists today (verified against the code)

| piece | where | what it does |
|---|---|---|
| constants | `src/config.py:21-28` | `MEMORY_REGISTRATION_FILENAME`, `MEMORY_SLOT_COUNT = 9`, `MEMORY_DEFAULT_COLOR`, `MEMORY_CONTENT_MAX_CHARS = 32000`, `MEMORY_COLOR_RE`. |
| path | `config.memory_path()` (`:425`) | `RESOURCES_DIR / MEMORY_REGISTRATION_FILENAME` — app-level, and the only reason the question-set skip exists. |
| read | `config.load_memory_registrations()` (`:464`) | Falls back to defaults for a missing file / bad JSON / missing key, coerces each slot (`_normalize_memory_slot`, `:438`), then **pads or truncates to 9** and rewrites the file whenever it did not already hold exactly those 9 slots. |
| write | `config.save_memory_registration(index, color, content)` (`:491`) | Reads, replaces one slot, writes all 9. |
| routes | `GET /api/memory` (`routes.py:905`), `POST /api/memory` (`:910`), `MemorySlotIn` (`:128`) | Payload key is `registration`; `index` is validated `ge=0, le=MEMORY_SLOT_COUNT - 1`. Not gated while a run streams (deliberate, `:912-915`). |
| level-A rule | `editorfs/builtin_policy.py:86-91` | `resources/memory_registration.json` is `ro` — "fixed and self-healing, not a file to hand-edit". |
| frontend | `composables/useMemory.js`, `components/MemoryBullets.vue`, `components/RegistrationMemoryDlg.vue` | `useMemory.js:3-6` mirrors the backend constants by hand; the slot array is initialised to 9 empty slots so the row renders before the first load. |
| when it loads | `WorkBenchView.vue:28-39` | Only when the chat tab is selected. `setPathLocation()` (`useBenchmark.js:843`) reloads MCP, LSP and the agent modes — **not** the bullets. |
| docs/ignore | `AGENTS.md:135`, `.gitignore:29` | Both name the old path. |

The live file on this machine holds five filled slots (a pyright-and-commit
recipe, a TypeScript one, a commit-only one, a French scratch note, a
guardCrash feature note) — so the content is real and must survive.

## 3. Decisions

### D1 — the app stays the file's only writer

`.nojae/` currently holds two files that follow one rule (AGENTS.md:139): seeded
by exclusive create, then *edited only outside the app*. This file joins the
folder but **not** that rule: it is a workspace-scoped **store**, not a
workspace-declared **config**. `POST /api/memory` keeps writing it, because the
bullet picker *is* the feature — colour and text in a dialog (`RegistrationMemoryDlg.vue`).

Rejected: making it hand-edited like the other two. The only alternative UX is a
JSON editor in the chat composer, which is a regression for a nine-slot picker.

Consequence for the docs: the reserved-folder sentence must stop applying to the
whole folder. `.nojae` holds *workspace state the app owns* as well as *workspace
config the user owns*; what stays reserved is the folder, not the editing rights.

### D2 — name and location come from one place

`<working location>/.nojae/prompt-helpers.json`. The folder name is **not**
re-declared: it is `mcp_connectors/policy_shared.POLICY_DIRNAME` (".nojae"),
imported the way `core/agent_modes.py:48-63` imports it (with the `sys.path`
fallback, because that module lives outside `src` for the connectors' sake). The
file name is a module attribute — `PROMPT_HELPERS_FILENAME` — so a test can move
the lookup, which is the repo's stated convention (`editorfs.constants`,
`mcphub.constants`, `agent_modes.AGENT_MODES_FILENAME`).

Rejected: keeping it in `resources/` and just making the path workspace-relative.
That keeps the question-set skip and leaves a workspace-config file in a folder
the app ships as source.

### D3 — the logic leaves `src/config.py` and becomes `src/core/prompt_helpers.py`

`config.py` sits at the bottom of the dependency graph — `editorfs.policy` and
`agent_modes` import `path_location` *from* it (`server.py:127-133` says this out
loud). A module that reads `.nojae` and seeds it cannot live there without an
import cycle. So the whole block moves: the five constants, `prompt_helpers_path()`,
`load_prompt_helpers()`, `save_prompt_helper()`, the seeder and the reader-only
state probe. `config.py` keeps **nothing** of it, and `list_question_sets()`'
skip at `:397-399` is deleted — after the move there is no non-question-set file
in `resources/` to skip.

The new module mirrors `core/agent_modes.py` (loader + seeder in one file,
because the seeder must write exactly the bytes the loader reads — the reason
stated at `agent_modes.py:809-814`), not `editorfs/policy_seed.py` (which is a
separate module because the policy has a whole package of readers).

Rejected: leaving the functions in `config.py` and doing the `.nojae` join with a
local import — that is the cycle the app already refused once, in prose.

Renamed with the module: `load_memory_registrations` → `load_prompt_helpers`,
`save_memory_registration` → `save_prompt_helper`, `default_memory_registrations`
→ `default_prompt_helpers`, `MEMORY_*` → `PROMPT_HELPER_*`. Not renamed: the
routes' payload key stays a list of **slots**, and the UI keeps saying "bullet".

### D4 — exactly 9 slots, and a wrong count is an error, never repaired

The bullet row is nine buttons (`MemoryBullets.vue:64` iterates
`MEMORY_SLOT_COUNT`; `theme.css:1669` is written for "nine numbered slots"), so
the count is a layout contract. Today a wrong count is silently fixed
(`config.py:480-487`) and the two tests `test_short_array_is_padded` /
`test_long_array_is_truncated_and_bad_color_coerced` assert that fixing.

New contract: a document whose `helpers` list does not hold exactly nine entries
is a **raised `PromptHelpersFileError`** naming the path and the count found. It
is never padded, never truncated, and never rewritten — the app is the only
writer, so a wrong count means a hand edit or corruption, and overwriting the
user's bytes is the worse answer.

The route turns that into **422** with the same sentence; it does not become a
500. The UI keeps working because the client's own state is already nine slots
before any fetch (`useMemory.js:15`) and `slot(index)` has a per-index fallback
(`MemoryBullets.vue:16`), so a failed load costs a red note, not a broken row.

Rejected: hard-failing the composer (an error in a small JSON file must not make
the chat tab unusable — the same reasoning as `agent_modes.py:16-22`, am-00 D11),
and rejected: keeping the silent repair (it makes the on-disk file's shape a
secret of the reader, which is what lets a wrong count go unnoticed).

Malformed *fields* keep today's coercion (`_normalize_memory_slot`): a bad colour
falls back to gray, a non-string content becomes empty. That is a cosmetic
problem, not a layout one, and the fix is invisible to the user either way.

The nine-invariant is enforced on the write side too: `save_prompt_helper` can
only have read a valid list (else it raised), and it replaces one element, so it
cannot emit another length. The seeder emits exactly nine. Nothing else writes.

### D5 — seeding follows the pair, at the same two moments

`ensure_prompt_helpers_file(root)` — exclusive create (`os.open(..., O_CREAT |
O_EXCL)`), one-level `os.mkdir` for `.nojae`, **never called from
`config.path_location()`**, failure answered as a dict and never raised. Called
from `api/server.py` next to `_seed_workspace_policy()` / `_seed_agent_modes()`
(`:138, :143`) and from `POST /api/path-location` next to the other two
(`routes.py:1346-1347`). `prompt_helpers_file_state(root)` is the read-only
mirror used by `GET /api/path-location`, so the settings dialog reports the file
without ever leaving it behind.

The seeded document is a Python literal rendered with `json.dumps`, not a `.json`
on disk — the `policy_seed.py:9-15` argument: the seeder's bytes and the loader's
expectations must not agree by luck, and one test asserts the literal round-trips
through the loader with no findings.

**No cache.** `agent_modes` caches per root because it is consulted on every chat
turn; this file is read on a tab switch and on a switch. Nine small strings do not
earn a cache and its invalidation rules.

**The read side never writes — which is a behaviour change.** Today
`load_memory_registrations()` rewrites the file whenever its bytes are not
exactly the nine normalized slots (`config.py:482-487`), so the *first GET of a
session* is what materialises it. That is the pattern the seeding rules exist to
forbid — `policy_seed.py:22-25` and `agent_modes.py:805-807` both say "a reader
that writes is how a stray folder turns up in a project someone only meant to
browse", and this app now points at projects someone only means to browse.

So `load_prompt_helpers()` answers:

| on disk | answer |
|---|---|
| file absent | the nine default slots, **nothing written** — the composer shows nine empty bullets, and the settings dialog's `promptHelpers` block says `absent` |
| unreadable / bad JSON / bad `version` / wrong count | raises `PromptHelpersFileError`, which the route turns into 422 (D4) |
| nine slots, one bad field | the normalized list, and **the file is rewritten** |

The last row is the one remaining write on a read, and it is deliberate: it is
what today's `_normalize_memory_slot` + rewrite pair does for colours, and
normalising a colour in memory only would make the dialog show a grey bullet
whose stored value is `bogus`, so the next save would re-commit the bad field.
Field coercion is idempotent and the app is the file's only writer, so this
write cannot contradict anyone.

### D6 — migration: the old file is read once, into the new one

`resources/memory_registration.json` is gitignored, so it exists only on the
machines that already have it — and on this one it holds five real prompts. The
move is therefore a **lift**, at the seeding moment only:

`ensure_prompt_helpers_file(root)` becomes: exclusive-create the new path; if it
already exists, do nothing as today; if the create happened **and** the legacy
path exists and parses, then rewrite the new file with the legacy slots
normalised to nine and report `"state": "migrated"`.

Order matters, and it is the whole design:

* the exclusive create still decides *whether* this workspace is being seeded, so
  a workspace whose file exists (edited, or seeded last boot) is never touched by
  the migration — no clobbering, no surprise second copy;
* the read side never looks at the legacy path (no "fall back to the old file"):
  one lookup, one truth. The lift is one-shot and happens at a write moment, so
  the rule "a reader never writes" survives.
* the legacy file is **left in place**. It is not deleted and not moved to the
  trash: the Editor's delete surface owns `.nojae/trash`, this is a boot-time
  seed, and deleting a user's file as a side effect of a feature rename is not
  this function's business. Nothing reads that path again after this change, and
  the user can remove it by hand. (Rejected: `os.replace` — one fewer file on
  disk, at the cost of destroying the only copy of the prompts if the write of
  the new one fails.)

`lib/path-location-policy.js` grows a third describer, `describePromptHelpers`,
over the states this file answers: the four the other two share (`created` /
`existing` / `absent` / `failed`) plus two of its own, `"migrated"` and
`"migration-skipped"`. Same pattern as its siblings: a complete sentence with its
own preamble (the notes fail independently), `warn` tone only for `failed` and
`migration-skipped`.

**A legacy file whose count is not nine is not migrated.** D4's rule is about the
file the app owns from now on, and the lift gets the stricter escape hatch: the
legacy list is copied only when it holds exactly nine entries, otherwise the new
file gets the nine defaults and the answer is a sixth state,
`"migration-skipped"`, naming the count found and the path. Silent repair during
migration is the one place where overwriting somebody's bytes would stay invisible
forever — the old file is never read again, so nothing would ever report it.

### D7 — the level-A rule is deleted, not moved

`builtin_policy.py:86-91` currently marks the old path `ro` so the Editor cannot
hand-edit a file the app rewrites. The path moves into `.nojae`, which
`builtin_rules()` already answers as **`hidden`, unconditionally, in every
workspace** (`:148-170`, D5's reservation) — so the rule is *deleted*, not moved.
The file is less reachable than before, not more, and there is no level-B rule
that could expose it.

This is a visible behaviour change worth stating: the old file was *browsable* in
the tree and refused writes; the new one is invisible, like its two neighbours.

Rejected: keeping a level-A rule for the new path — a pattern naming `.nojae` is
rejected by the policy loader (D5), so such a rule would be un-writeable anyway.

### D8 — the UI reloads on a workspace switch

`useBenchmark.js:843 setPathLocation()` already reloads MCP (`:894`), LSP
(`:903`), the agent modes with `reseed: true` (`:912`) and re-roots the tree
(`:924`). It gains one call, `loadPromptHelpers()` from `usePromptHelpers`, at the
same point — otherwise the switch leaves the previous workspace's bullets on
screen until the user leaves the chat tab and returns (`WorkBenchView.vue:28-39`).

`useMemory.js` → `usePromptHelpers.js` (its state key `registration` → `slots`,
and the mirrored constants follow the new backend names). The name change is
cosmetic and free now; doing it later means touching three components again.

The payload key on the wire changes with it: `{"registration": [...]}` →
`{"slots": [...]}` on both `/api/memory` and `/api/prompt-helpers`. One reader
each side, no third party — the app and its SPA move together.

Rejected: keeping the route names `/api/memory` while the file is renamed — the
rename is the point of the feature, and both sides change at once so the cost is
nil. This spec renames them. `tests/test_write_surface_isolation.py`'s markers
(`WRITE_MARKERS`, its line 50) cover `/api/fs` writes only, so an ordinary
`/api/*` POST rename neither breaks that invariant test nor needs it extended:
`POST /api/prompt-helpers` stays browser-only for the same reason it is today.

## 4. Shape and files

### The document

```json
{
  "version": 1,
  "helpers": [
    { "color": "#2ec27e", "content": "# Session goal\n\n* ..." },
    { "color": "#808080", "content": "" }
  ]
}
```

`version: 1` matches the folder's other two files (`POLICY_VERSION`,
`AGENT_MODES_VERSION`): a version mismatch is an error, not a guess. The key is
`helpers`, not `registration`, because the name of the file is the name of the
feature. Exactly nine entries — D4.

The default set: **nine empty gray slots**, i.e. what
`config.default_memory_registrations()` returns today (`:430-435`), unchanged.

Rejected: shipping real example prompts in the seed (e.g. "check the app by
running the unit tests"). Tempting, and it makes a fresh workspace look alive —
but the helpers of a *foreign* project are precisely the thing that must not be
invented by the app, and nine slots pre-filled with benchmarkit's own habits is
the app-level scoping this change exists to remove. A user's own defaults, once,
is what D6's migration brings.

### New / changed files

| file | change |
|---|---|
| `src/core/prompt_helpers.py` | **new.** Constants (`PROMPT_HELPERS_FILENAME`, `PROMPT_HELPERS_VERSION = 1`, `SLOT_COUNT = 9`, `DEFAULT_COLOR`, `CONTENT_MAX_CHARS`, `COLOR_RE`), `prompt_helpers_path()`, `prompt_helpers_file_path(root)`, `load_prompt_helpers()` (raises `PromptHelpersFileError` on a wrong count), `save_prompt_helper(index, color, content)`, `default_prompt_helpers()`, `_normalize_slot()`, `ensure_prompt_helpers_file(root)` + the D6 lift from the legacy path, `prompt_helpers_file_state(root)`, `LEGACY_PROMPT_HELPERS_PATH` (module attribute, so a test moves it — the `mcphub.constants` convention), and a `_self_check()` on the numeric constants the way `agent_modes._self_check()` guards its caps (`:942-959`). |
| `src/config.py` | **deletes** lines 21-28 and 425-496, and the skip at 397-399. |
| `src/api/routes.py` | `MemorySlotIn` → `PromptHelperSlotIn` (`index` bounded by `prompt_helpers.SLOT_COUNT`); `GET /api/memory` → `GET /api/prompt-helpers`, `POST` likewise; both payloads answer `{"slots": [...]}`; both map `PromptHelpersFileError` → 422. `_path_location_payload()` gains a `"promptHelpers"` block on GET (`:1237-1240`) and the POST returns the seed result (`:1360-1368`). |
| `src/api/server.py` | `_seed_prompt_helpers()` beside the two existing seeders, called at `:143`'s side — after the policy seed, which is the one that creates `.nojae` (the argument already written at `:144-150`). |
| `src/core/editorfs/builtin_policy.py` | deletes the rule at `:86-91`. |
| `web/src/composables/useMemory.js` | renamed `usePromptHelpers.js`; constants follow the backend; state key `slots`; `loadMemory` → `loadPromptHelpers`. |
| `web/src/components/MemoryBullets.vue`, `RegistrationMemoryDlg.vue`, `WorkBenchView.vue`, `WorkBenchInput.vue` | import rename only; no markup, no CSS. |
| `web/src/lib/path-location-policy.js` | third describer, `describePromptHelpers`, same four/five states and tone rule. |
| `web/src/composables/useBenchmark.js` | `setPathLocation()` reads `body.promptHelpers` and calls `loadPromptHelpers()`; two note fields beside `pathLocationPolicyNote` / `pathLocationAgentModesNote`. |
| `SettingsDialog.vue` | renders the third note, as it renders the other two. |
| `.gitignore:29` | `resources/memory_registration.json` → nothing (`.nojae/` is already ignored at line 27). |
| `AGENTS.md` | see §6. |
| `README.md` | the bullet feature is **not documented there today** (its two `memory` hits are about the in-memory event bus), so the only required edit is the "Whichever folder you point at gets its own read-write policy in `.nojae/…`" paragraph (`:178-185`): that folder now also holds the workspace's prompt helpers, which the app writes. A short subsection beside "Agent modes" (`:186`) describing the nine bullets is worth adding while this is being touched, but is not required by the move. |

## 5. Out of scope

* **Per-mode helpers.** Nothing links a helper to an agent mode; the nine bullets
  are mode-independent and stay that way.
* **A tenth helper, or fewer.** D4 fixes nine for the layout. A growable list
  means the row wraps, and the wrapping is a design of its own.
* **Editing `.nojae` from the Editor.** The folder stays hidden (D7).
* **Deleting the legacy file.** Left to the user (D6).
* **Sharing helpers across workspaces**, importing/exporting, or a global
  fallback set. Explicitly the opposite of the request.
* **Anything in `resources/`.** Unchanged apart from losing one file.

## 6. AGENTS.md, exactly

Three edits, and the folder's rule needs rewording because D1 breaks it:

* `:135` (the `resources/memory_registration.json` row) — **delete**.
* `:139` (the `.nojae/` row) — the two files become three, and the one rule
  becomes two:
  > Its files follow one shared seeding rule and split on who edits them after:
  > `read-write-policy.json` and `agent-modes.json` are seeded by exclusive
  > create when a folder becomes the working location, then edited only outside
  > the app; `prompt-helpers.json` is seeded the same way and is the app's own
  > store, written by the WorkBench bullet picker. `trash/` holds the dated
  > batches the Editor's deletes move.
* `:68`'s module table gains one row for `core/prompt_helpers.py`.

The count "two files" is already the kind of detail AGENTS.md says to avoid, so
the rewrite states the *rule split* rather than a number, and the row that named
the file is the one that disappears.

## 7. How it is verified

`tests/test_memory.py` → `tests/test_prompt_helpers.py`, with the fixture
switched from `config.RESOURCES_DIR` to `monkeypatch.setenv("BENCHMARKIT_PATH_LOCATION", …)`
the way `tests/test_agent_modes.py:128-131` does it (the location is process
state, so a test that sets it must also set it per test, not per module).

| test | asserts |
|---|---|
| a reader never writes | `GET /api/prompt-helpers` on an empty root answers the 9 default slots and leaves **no `.nojae` folder behind**; the file appears only after `ensure_prompt_helpers_file` (boot/switch) or a `POST` |
| bad `version` / bad JSON | 422 naming the path, file untouched |
| **the count is an error** | replaces `test_short_array_is_padded` and `test_long_array_is_truncated_and_bad_color_coerced`: 8 and 10 entries answer **422**, the file on disk is byte-identical to what the test wrote, and the client keeps nine slots |
| a bad colour is still coerced | `"bogus"` → `DEFAULT_COLOR`, and the file is rewritten (field coercion stays self-healing, D4) |
| the literal round-trips | `default_prompt_helpers_text()` parses through `load_prompt_helpers` with no error — the `policy_seed`/`agent_modes` pairing test |
| exclusive create | a pre-existing hand-written file is untouched by `ensure_prompt_helpers_file`, and a folder standing at the path answers `failed`, not `created` |
| **the lift** | legacy file with 5 filled slots + absent new path → `state: "migrated"`, the 5 slots are in the new file in order, the legacy file still exists, and a second call answers `existing` and changes nothing |
| a legacy file that will not lift | legacy list of 7 entries → the new file holds the nine defaults, `state: "migration-skipped"` naming the count, and the legacy bytes are untouched |
| no lift over an existing file | both paths present → new file wins byte-for-byte |
| per-workspace | switch the location, `GET` answers the new root's slots; the old root's file is not read |
| switch payload | `POST /api/path-location` reports `promptHelpers.state`, and a failing seed is a note, never a 4xx |
| level A | `tests/test_editorfs_builtin_policy.py:130-160, 243, 412` lose the `resources/…` rows and gain that `.nojae/prompt-helpers.json` is `hidden` in a **foreign** workspace too (the reservation is unconditional) |
| question sets | `test_memory.py:157-165` becomes a test that `resources/` holds only question sets — the skip is gone, so the guard moves to "nothing else lives there" |

Frontend: no Playwright spec asserts the bullet *count* or the payload key (only
`04-agent-modes.spec.js:180` measures `.memory-bullets`' box, which the layout
does not change), so this needs one new e2e — switch the working location and
assert the bullets change — and one Vitest case over `describePromptHelpers`
alongside the two existing describers.

Manual: with the five real slots on this machine, boot, open WorkBench, confirm
the five prompts are still there and the tree shows no `.nojae` entry.
