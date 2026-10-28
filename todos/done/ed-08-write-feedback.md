# ed-08 — Feedback: where a write refusal lands, and how long it stays

Four tasks of write routes are in place; each one can answer 409/413/422/404 with
a reason the user has to be able to read. This task makes that answer **visible
and durable**, and gives every successful mutation a short confirmation. No new
route, no new policy rule — only the surface that reports them.

Depends on: ed-03 (save), ed-05 (delete), ed-06 (rename), ed-07 (the tree's
in-row error).

## 1. Why the existing message line is not enough

`useEditorFiles.state.message` is a good model for **a refused open**: one line
above the pane, naming the path, and cleared by the next click, the next tab
focus or the next close (`useEditorFiles.js:229`, `:253`). For a write that is
wrong in two ways:

* a failed save must stay visible while the user decides what to do about the
  dirty buffer — a click on another tab must not eat the reason;
* the user may be looking at the **tree** when a rename or delete fails, and the
  message line lives above the **pane**.

So: keep `message` exactly as it is (its tests and its semantics are fine), and
add a second channel.

## 2. `useEditorNotices.js` — the one write-feedback channel

A small new composable (singleton, like every other concern):

```
state.notices = []   // { id, tone: 'error'|'ok'|'warn', text, path, at }
notice(tone, text, path)   // auto-id, deduped
dismiss(id) / clear()
```

Rules:

* **dedupe** on `(tone, text, path)` within a short window, so a `Ctrl+S` held
  down does not stack five identical notices (the re-entrancy guard in ed-03
  already stops the requests — this stops the noise from anything else);
* **at most 3 visible**, oldest dropped first, and a notice that concerns the
  buffer the user just fixed is dismissed when that buffer's action succeeds
  (a save that now works removes its own earlier failure line);
* **`error` and `warn` never time out**; `ok` dismisses after ~4 s. A refusal that
  vanishes on a timer is the bug this task exists to avoid;
* `aria-live="polite"` on the container, so a screen reader hears a refusal that
  happened while focus was in the tree (the pane's `role="alert"` line stays as it
  is for the read case — do not make one element serve both).

Rendering: a fixed-position stack at the bottom-centre of the app
(`.editor-notice-stack`, styles in `theme.css` — this stylesheet, not a scoped
block), teleported to `<body>` the way the tree's context menu is, with an × per
notice like `.editor-raw-message-dismiss`. Bottom-centre because it must not cover
the tab strip, the composer, or the tree's right-click point.

## 3. Who reports what

| Action | Failure notice | Success notice |
|---|---|---|
| save | server `detail`, or the 409-stale wording ("the file changed on disk — resolve the conflict first") | none (the dirty dot disappearing *is* the feedback) |
| create | server `detail` (409-exists / 422-policy / 409-parent) | "Created `path`" |
| delete | server `detail` | "Deleted — kept in `logs/trash/<batch>` for N days" |
| rename | server `detail` (also rendered in-row by ed-07; the notice is the copy for a user whose eyes are on the pane) | "Renamed `from` → `to`" |
| probe failure (ed-02) | one `warn` the first time, then silence until it recovers — a 5 s poll must not produce a new notice every 5 s | none |

The **409-stale case is not a notice on its own** when it comes from a save: the
conflict dialog (ed-02) is the interaction, and a notice behind a modal dialog is
noise. It becomes a notice only when the same conflict is detected by the probe on
a buffer the user is not looking at — and there it is one line, not a stack.

## 4. Copy matters (this is the user's only text)

The server's `detail` strings are already specific because `path_policy` names the
entry (`"inside '.git', which connectors never write to"`). Rules for the client
text that wraps them:

* never rewrite or shorten a server reason — show it, prefixed by what was
  attempted ("Could not save `web/src/x.js`: …");
* never say "an error occurred", and never show a bare status code;
* name the path the operation was about, always (a refusal about a *renamed*
  path is confusing if the notice names the old one — use the path the request
  carried);
* keep one sentence. If the server's reason is two clauses, that is enough.

## 5. The read-only badge gets the last word

`useEditorMode.js` explains *why* the pane is read-only in its `title`. Two cases
are missing once writes exist, and both are cheap here:

* the pane is editable but the active buffer is dirty → mention the unsaved state
  (the badge is the only pane-wide indicator that survives a tab switch);
* a write was refused while a reply streamed → mention that saving is blocked
  until the reply ends, which is the answer to "why did my Ctrl+S do nothing?".

Also update the batch-mode gap here (D7): `editorMode` becomes
`chat.state.phase === 'idle' && !useBatch.state.active`, replacing the "known
edge, deliberate" comment at `useEditorMode.js:21-24`. The server-side gate stays
authoritative either way — this only stops the pane *inviting* an edit that will
be refused. `web/tests/useEditorMode.test.js` already exists and must be extended,
not replaced.

## 6. Tests

Vitest, `web/tests/useEditorNotices.test.js`:

1. an error notice persists (no timer), an `ok` notice expires;
2. the same failure twice within the window → one notice;
3. a cap of 3, oldest dropped;
4. a success dismisses the matching earlier failure;
5. `dismiss`/`clear` behave;
6. the text is the server's `detail` verbatim, with the attempt prefixed;
7. the probe failure produces one `warn`, not one per tick.

Extend `web/tests/useEditorMode.test.js`:

8. a batch running with `phase === 'idle'` → `read_only`;
9. an idle chat, no batch → `read_write` (the existing case, unchanged).

Playwright gets its own coverage in ed-09 — do not grow an e2e here.

## 7. Acceptance

* Rename a file to `.env` → one notice with the connector's wording; the in-row
  error also shows; clicking elsewhere keeps both.
* Save, then change the file from the terminal, then save again → the conflict
  dialog appears, and no stack of notices hides behind it.
* Delete a file → one short confirmation naming the trash path, gone in a few
  seconds.
* A save refused because a WorkBench reply is streaming: the notice says so, and
  so does the read-only badge's `title`.
* Hold `Ctrl+S` on a failing buffer → one notice.
* Batch mode between turns: the pane is read-only, and the badge says why.
* `cd web && npm test`, `pytest` green.
