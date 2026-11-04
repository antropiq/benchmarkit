# win-09 — Docs: the README's Windows section, and the one AGENTS.md clause

Two documentation edits, and they are the last task because they describe
behaviour that only exists once win-01 … win-08 have landed. Writing them earlier
would put a sentence into AGENTS.md — which is injected into **every** WorkBench
system prompt — that the code does not yet back up.

Depends on: win-01 through win-08.
Spec: `docs/windows-support/windows-support-target-A-no-crash.md` **A-D8** (the
README section) and **§7's last paragraph** (the AGENTS.md clause). Also re-read
the *Editing AGENTS.md* section at the end of `AGENTS.md` before touching it — it
is a set of rules about this file's shape, not only its content.

## Files

| file | change |
|---|---|
| `README.md` | a new **Windows** section (A-D8), linking `docs/windows-support/` |
| `AGENTS.md` | one added sentence to the *"Path rules come from two levels, one verdict"* bullet, plus the one refusal invariant |

Nothing else. No `docs/windows-support/*` edit (those are the plans, already
written), no code, no test.

## 1. The README's Windows section (A-D8)

Place it inside **"Running the app"** (after "1. Install Python dependencies" is
the natural home — the install commands there are bash/pyenv-flavoured), as a
`### Windows` subsection. Four required contents, in A-D8's own order:

1. **How to install**: `python -m venv .venv`, activate with
   `.venv\Scripts\activate`, `pip install -e .`, then `cd web` +
   `npm install && npm run build`. Node is still required for the SPA, and the
   frontend's TypeScript detail (AGENTS.md's *"the frontend's
   `node_modules/typescript`, passed to Vue via `--tsdk`"*) is a thing a Windows
   user hits while setting the LSP side up — mention it under "what refuses", not
   as a working instruction.
2. **What works**: the Benchmark tab, WorkBench chat, the verifier, and the
   Editor's file surface — tree, open, search, git status, diff, create, rename,
   **folder delete** (broken today, fixed by win-06 — do not promise it before
   that task lands), and saving a CRLF file with its terminators unchanged.
3. **What refuses, and why**: the Editor terminal, the `local` MCP connectors
   (`console`, `fedit`) and every language server — each refused **in words**,
   naming target B, and where that sentence is visible (the hub card's error and
   the hub payload's note). State that this is a capability boundary chosen on
   purpose, not a bug: an unbounded agent surface is worse than a missing one.
4. **The two template differences from A-D7**: the Windows form of
   `lsp_connectors/lsp_servers.example.json`'s
   `${VOLTA_BIN:-/usr/local/bin}/pyright-langserver`
   (`${VOLTA_BIN:-.}\pyright-langserver.cmd`) and its `PATH` (`;`-separated) —
   and that `lsp_connectors/restricted-paths.txt` holds POSIX forms only, with
   the Windows values coming from target B. win-08 puts the pointer to *this*
   section inside those two files; this is the sentence they point at.

Then one line linking `docs/windows-support/windows-support-road_to_target.md`
and the three target plans, so the roadmap is findable.

Tone constraint from A-D8: *"Not a porting guide; a statement of the boundary."*
Keep it under ~35 lines, imperative, no version numbers, and do not churn the
Linux sections. Do not restate the A decisions — the docs folder does that.

## 2. The AGENTS.md clause (§7)

Exactly one bullet changes, and it **gains** rather than grows sideways:

- *"Path rules come from two levels, one verdict"* gains the sentence that **the
  canonical path form is `/`-joined on every platform** and that **rule
  comparison folds case only where the filesystem does**.
- Add the *"a refusal names its reason"* invariant **once**, not per surface
  (§7's words). The terminal row, the hub rows and the lock refusals are all the
  same rule; naming it once is what keeps the file from becoming a list of
  surfaces.

Hard constraints, straight from *Editing AGENTS.md* — this file is paid for on
every chat message:

- stay **under ~300 lines** (it is at 286 now, so the budget is roughly two or
  three lines net: write tight, and if a neighbouring paragraph can be shrunk to
  pay for the new sentences, shrink it);
- one idea per bullet, wrapped at ~100 columns;
- no counts, no member enumerations, no version numbers, nothing a routine code
  change would make wrong — state the **invariant**;
- describe the app **as it is now**: no "on Windows, previously…", no
  before/after narration;
- **no Windows section in AGENTS.md** — that belongs to the README (§7 says so
  explicitly).

## 3. Check the code's own sentences agree

win-05 makes `policy_glob.py:226`'s docstring honest and win-07 writes the hub
refusal sentences; this task only checks the README agrees with them. A doc that
contradicts a runtime sentence is exactly the drift AGENTS.md warns about. Adjust
the README to the code, never the code to the README — unless the runtime
sentence is genuinely unclear, in which case note it in the commit rather than
editing another task's module silently.

## Acceptance

- `wc -l AGENTS.md` is **≤ ~300** (report the number in the commit note), the
  file still has exactly one bullet about path rules, and a case-insensitive
  search for "windows" in it shows no new *section* (the two sentences may name
  the platform; a heading may not).
- The README's Windows section covers all four required contents and links the
  roadmap; searching the README for `windows-support` finds the link.
- Every claim in the section is true of the code **as it now stands**: check each
  one against the tree (does folder delete work? does the terminal still refuse
  with `CLOSE_CODE_UNSUPPORTED`? do the hub notes appear?). Anything not
  verifiable on Linux, phrase as the boundary rather than as a promise.
- No test, code, or `docs/windows-support/` file changed: `git diff --stat`
  shows exactly `README.md` and `AGENTS.md`.
- `pytest -q` and `cd web && npm test` still green — a doc change should not
  touch them; if a test asserts README/AGENTS text, that test is the thing to
  update, and say so.
- Move this file to `todos/done/` when the above is true. The lot is then
  complete; the manual Windows pass (A §6, last paragraph) is run by hand
  afterwards and is **not** part of this lot.
