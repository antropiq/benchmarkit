# Save / load a WorkBench session as JSON

Park a WorkBench (Chat tab) conversation on the user's hard drive as a plain JSON
file (`a-very-important-question.json`), and later re-open such a file to
**continue the conversation exactly where it stopped** — no re-generation.

Two buttons in the side-panel action row, next to Copy / Restart:

| Button | Behaviour |
|---|---|
| **Save session** | browser "Save as…" download, user-named |
| **Load session** | file picker → conversation restored, ready to continue |

Both disabled while a reply is streaming. Base64 image streams are **stripped**.

## Decisions taken (from the brief)

1. Load over a non-empty live conversation → **confirm, then replace**; never
   while the flow is running.
2. Image-only turns after stripping → **placeholder text**, so the history stays
   valid (step 3: an empty turn would 422 the next send).
3. Filename → the browser's native **Save-As** (no in-app name field).
4. **No** `stats` block: the file carries the messages only; tok/s and ctx badges
   are not restored.
5. **No** `mcpEnabled` / `lspEnabled` in the file: a loaded file never influences
   which connectors a conversation may reach.

## Verdict: not big

The heavy lifting is already done by the app's design: the client rebuilds the
whole history on **every** send (`useChat.js:606`,
`state.messages.map(messagePayload)`) and `POST /api/chat` (`routes.py:1127`) is
stateless — it takes any history and answers the next turn. So "continue a loaded
session" is *free*: **Phase 1 needs no backend change at all.**

| Phase | Scope | Size |
|---|---|---|
| **1** (the feature) | 2 buttons + serializer + loader, frontend only | ~130 lines, 3 files + 2 doc lines |
| **2** (optional: the `last_session_content.json` snapshot) | per-turn server-side copy | ~45 lines Python + 3 tests |

No frontend test runner exists, so Phase 1 is verified manually (checklist at the
end). Phase 2 is covered by `tests/test_chat.py`.

---

## The JSON file format (v1)

A thin envelope around the OpenAI message list, so the file is readable and
portable (its `messages` paste into any OpenAI-compatible client):

```json
{
  "format": "benchmarkit-chat-session",
  "version": 1,
  "messages": [
    { "role": "user", "content": "explain transformer attention" },
    { "role": "assistant", "content": "..." }
  ]
}
```

- Exactly the shape `POST /api/chat` accepts: roles limited to `user` /
  `assistant` (`ChatMessage`, `routes.py:235`), `content` a plain string.
- On load, unknown extra keys are ignored (forward-compatible), and a bare
  `[{role, content}, ...]` array is also accepted (3 lines of tolerance, so
  hand-written and third-party files work).
- Rejected on load, with a precise error: a turn whose role is neither `user` nor
  `assistant`, a turn carrying `tool_calls`, or a file with no usable message.
  Silently dropping a `system` turn would change the conversation's meaning, so an
  invalid file must fail loudly.

### Stripped on save — what is lost, stated honestly

| Item | Why |
|---|---|
| **Base64 images** | Required by the brief; `useChat.js:49` allows 10 MB per image, so a JSON transcript would balloon to tens of MB. |
| **`tool_calls` / `role:"tool"` turns** | The browser never receives them: it only gets `_preview`-capped summaries (200 chars of args, 400 of result — `chat.py:84`, emitted at `:385` / `:412`). Writing them back would be fabrication. They could not be re-posted either: `ChatMessage.role` is `Literal["user","assistant"]` and `extra="ignore"` drops `tool_calls` (`routes.py:233-236`). |
| **Reasoning / thinking text** | **Saved since v2** as an optional per-assistant-turn `reasoning` list (one entry per `WorkBenchToolPanel` block), restored into `message.blocks`. It sits *beside* `content`, never inside it (`chat.py` docstring: reasoning is never answer text) — `send()` builds its history through `messagePayload()`, which reads `raw`/`images` only, so a restored flow can never reach the model. |
| **System prompt + AGENTS.md instructions** | Rebuilt server-side at every send (`chat.py:157`, `agent_instructions_block()` at `routes.py:1171`) — always fresh, never stale in the file. |
| **Per-turn badges** (`tokens`, `took`, `contextTokens`, …) | Decision 4. Client-only render state anyway. |

**Consequence:** a saved session replays the *conversation text*, not the tool
loop. Continuing still works unchanged, because today's sends already re-post only
`state.messages.map(messagePayload)` — tool turns are nowhere in it. This is not a
regression introduced by the file format.

---

## Phase 1 — Save / Load (the feature)

### 1. `web/src/composables/useChat.js` — serialize + restore (~85 lines)

Add beside `conversationMarkdown` (`:354`), same convention (exported, pure):

```js
export const SESSION_FILE_FORMAT = 'benchmarkit-chat-session'
export const SESSION_FILE_VERSION = 1

export function conversationJson(messages) { /* the envelope above */ }
```

- Reuse `messagePayload(m)` (`:369`), which already strips the render-only
  `TOOL_LINE_MARKER` (`:372`) and returns `{role, content: string}` for text-only
  messages. Pass an images-stripped copy (`{...m, images: []}`) so the multimodal
  branch is never taken.
- Skip a message left with blank `content` after stripping.
- `stats` deliberately absent (decision 4); keep the loader tolerant in case a
  later version adds it back.

Then inside `useChat()`:

```js
function loadSession(parsed) -> { restored: number }
```

- Guard `state.phase !== 'idle'` first, mirroring `editLast()` (`:705`) and
  `reset()` (`:724`) — the "must not be running" rule enforced in the composable,
  not only on the button.
- `normalizeContent(content)` accepts a string **or** an OpenAI parts array: text
  parts joined, image parts replaced by
  `[image stripped from the saved session]` (decision 2).
- Rebuild the rich shape the view renders (`WorkBenchView.vue:81`):
  `{ role, raw, html: renderMarkdown(raw), streaming: false, images: [] }` for
  every turn, plus the assistant-only field set `send()` initialises (`:607`) —
  `tools: []`, `blocks: []`, `pendingStreamQuote: false`, `tokens: 0`, `took: 0`,
  `ttft: 0`, `decode: 0`, `contextTokens: 0`, `tokensEstimated: false`,
  `contextEstimated: false`. Fill them so the panel badges (which read `tokens` /
  `contextTokens`, `WorkBenchSidePanel.vue:50`, `:87`) never see `undefined`.
- Assign `state.messages = restored` (**replace**, not push), then set
  `state.mcp.locked = state.lsp.locked = true` when the file carried messages: a
  loaded conversation counts as started, so the toggles freeze exactly as they
  would have. Call `loadChatMcp()` / `loadChatLsp()` **only when unlocked** — and
  do *not* call `reset()`, it would wipe the messages just loaded.
- Clear `state.error` and `state.pendingImage`; leave `state.session.startedAt` at
  0 so the stopwatch reads 00:00:00.
- Return `loadSession` from `useChat()`.

### 2. `web/src/components/WorkBenchSidePanel.vue` — two buttons (~45 lines)

- **Save**: `new Blob([JSON.stringify(conversationJson(state.messages), null, 2)],
  { type: 'application/json' })` + `URL.createObjectURL` + a synthetic `click()` +
  `URL.revokeObjectURL`. No server round-trip, so the native Save-As dialog lets
  the user type `a-very-important-question.json`. Default name
  `chat-session-<YYYY-MM-DD-HHmm>.json`. Disabled when `!canCopy` (the Copy
  button's own predicate, `:144`) **or** `streaming` (prop, `:9`).
- **Load**: a second hidden `<input type="file" accept="application/json,.json">`
  next to the image input (`:194-202`; hidden by `.chat-file-input`,
  `theme.css:1531`), with the same `input.value = ''` reset afterwards so
  re-picking the same file still fires. Parse through `file.text()` + `JSON.parse`
  inside `try/catch`. Disabled when `streaming`.
- Confirm-and-replace (decision 1): when `state.messages.length > 0`, ask
  *"Replace the current conversation with the loaded session?"*.
- All failures land in `state.error`, the red alert line the view already renders
  (`WorkBenchView.vue:170`): unreadable file, malformed JSON, unknown `format`,
  unsupported role, no usable message. Also refuse a file over ~2 MB before
  parsing (a stray 10 MB base64 blob would freeze the tab).
- Icons: inline SVG 16×16, `stroke-width="2"`, lucide shapes like the neighbours
  (`:233`, `:256`) — download tray for Save, upload arrow for Load — each with
  `aria-label` + `title`.

### 3. Keep the stripped-image turn valid (`useChat.js`, 1 rule)

An image-only user turn carries **no** text part in `messagePayload` (`:375`), and
`ChatMessage._require_payload` / `ChatIn._require_turns` (`routes.py:238-251`,
`:279-287`) reject empty content with **HTTP 422**. So the placeholder of step 1 is
not cosmetic: without it, a loaded session breaks the *next* send with a confusing
error. `editLast()` (`:717`) re-attaches `images[0]` — after a load that is `null`,
which is correct and needs no change (the file holds no image).

### 4. Docs (2 lines)

- `AGENTS.md` (*Key patterns & conventions*): a session file is
  `{format, version, messages}` in OpenAI user/assistant form; images, reasoning
  and tool-loop turns are never saved, and loading one is client-side only.
- `README.md` (*Features* / *Technical details*): one short paragraph.

---

## Phase 2 (optional, the `last_session_content.json` idea)

A server-side copy of the conversation, refreshed as the session runs, overwritten
by the next conversation. **Not needed for Phase 1** — it buys durability, not the
save/load buttons. Read the caveats before deciding.

### How it would work (~45 lines)

1. `src/core/utils.py`, next to `RESULTS_PATH` (`:10-11`):
   `CHAT_SESSION_PATH = PROJECT_ROOT / "last_session_content.json"` plus
   `save_chat_session(messages, answer, path=CHAT_SESSION_PATH)`, mirroring
   `save_results()` (`:53`) — same patchable-constant convention the tests already
   follow (`tests/test_chat.py:392-393`).
2. `post_chat` (`routes.py:1127`): accumulate the reply inside the SSE pump
   (`event_stream`, `:1183-1202`) — the `token` frames already carry the answer
   text — and write the file **only when the `done` frame passes through**
   (`:1197`). Content = `body.messages` (already `{role, content}` user/assistant,
   no system message, no `tool_calls`) with `image_url` parts dropped, plus the
   accumulated answer as the trailing assistant turn. No `core/chat.py` change:
   verified that `stream_chat` *rebinds* `messages` (`:157-161`), so the route's
   list never sees the tool turns it appends (`:353`, `:404`) — reaching that
   transcript would require `stream_chat` to hand it back.
3. `GET /api/chat/session` → `FileResponse`, mirroring `/api/results`
   (`routes.py:1298-1303`, 404 when absent), and one `.gitignore` line.
4. Tests in `tests/test_chat.py`: a fake `stream_chat` (pattern at `:436`) posts two
   turns → the file holds both turns with the images stripped; a third turn from a
   *new* conversation overwrites it; a stopped turn leaves it untouched.

### Write only on `done` (the invariant that makes it safe)

Written at `done`, the file always equals *the conversation as it stands after the
last completed turn* — which matches the screen in every edge case:

- **Stop** discards the partial reply client-side (`useChat.js:689-695`) and emits
  no `done` → the previous snapshot stays correct.
- **Send failure / connection drop** likewise keeps the previous snapshot.
- A **page refresh or server crash** loses nothing beyond the in-flight turn.

### Caveats to accept (or skip Phase 2)

- **Two serializers.** The client builds the envelope in JS, the server in Python. A
  format change must touch both; the Phase-2 test asserting a saved file passes
  `ChatIn` is what protects the pair.
- **Not per-conversation.** Nothing enforces one chat at a time (`_CHAT_TASKS` is a
  set, `routes.py:292`), so two open tabs would take turns clobbering one file.
  Acceptable for a desktop single-window app; worth a code comment.
- **A Save button reading this file would be a trap**: after loading a file from
  disk, the snapshot is stale until the next send, so Save would re-download
  something that is not what is on screen. Hence **Save stays client-side (Phase
  1)** even if Phase 2 lands; the snapshot's real payoff is a future "Restore last
  session" button (`GET /api/chat/session` feeding the same `loadSession`).
- **Conversation text sits unencrypted in the project root**, rewritten on every
  completed turn: keep it out of git (`.gitignore`) and off every list endpoint —
  only `/api/chat/session` should serve it.

**Recommendation:** ship Phase 1 alone unless durability across a crash/refresh is
what actually motivated the request. If it is, Phase 2 is about half a day more, and
its first follow-up should be that "Restore last session" button — otherwise the
file has no consumer beyond a hypothetical one.

---

## Deliberately out of scope

- Server-side session library / `chat_sessions/` folder, session list,
  restore-on-boot, multi-session tabs.
- Faithful tool-loop persistence (`tool_calls` + full tool results). Needs
  `stream_chat` to return its mutated `messages` and the `done` frame to carry it
  (~15 lines in core), a client that re-renders tool badges from raw turns, **and**
  a widened `ChatMessage` model so those turns can be re-posted. Only if verbatim
  tool turns are truly required.
- Re-attaching images after a load; restoring badges. (Reasoning text was out of
  scope in v1 and is saved/restored from **v2** — see the table above. What is
  still lost is the *interleaving* of thinking with tool badges: only the
  thinking flows are in the file, so a restored timeline shows the flows
  without the tool badges between them.)

## Manual verification (no JS test runner)

1. Chat 3 turns (one with an image) → Save → the file has 6 messages, no
   `data:image`, and the image turn kept its text.
2. Restart the conversation, Load that file → 6 bubbles render (markdown intact);
   send a 7th turn → answered with full context, no 422.
3. Load with a live conversation → confirm dialog; cancelling leaves it untouched.
4. Save / Load while a reply streams → both buttons disabled (`streaming`).
5. Load `not json`, `{"format":"other"}`, and a file with `"role":"system"` → three
   distinct red error lines, conversation untouched.
6. After a load, the MCP/LSP toggles are frozen (locked) and the stopwatch reads
   00:00:00.

## Files touched

| File | Phase 1 | Phase 2 |
|---|---|---|
| `web/src/composables/useChat.js` | +~85 | — |
| `web/src/components/WorkBenchSidePanel.vue` | +~45 | — |
| `AGENTS.md`, `README.md` | +2 lines | +2 lines |
| `src/core/utils.py` | — | +~15 |
| `src/api/routes.py` | — | +~25 |
| `.gitignore` | — | +1 |
| `tests/test_chat.py` | — | +~35 |
