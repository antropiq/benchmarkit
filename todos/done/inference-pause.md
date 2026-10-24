# Pause the WorkBench inference flow (interrupt, keep context, comment)

Third control in the composer action row, next to **Stop generating**: a **Pause**
button that interrupts the *current* reply **without throwing anything away**, so
the user can type a comment/correction that rides on top of the interrupted turn.

Today the composer only offers (`WorkBenchInput.vue:358-400`):

| Button | Effect |
|---|---|
| **Restart conversation** (`:359`) | `stop()` + `reset()` — kills the reply *and* the conversation |
| **Send message** / **Stop generating** (`:373` / `:388`) | Send when idle; Stop aborts the fetch **and discards the partial reply** (`useChat.js:924-930`) |

Stop is destructive by design. Pause must be the non-destructive twin: same
interrupt, retained context.

## Flow requested (brief)

```
user clicks Pause
  → POST /api/chat/pause sets pause_requested = True on the in-flight reply, 200
  → the Pause button blinks, and is NOT clickable while it blinks
  → core/chat.py checks the flag at its next message boundary (chunk, tool call,
    next round) → close the upstream LLM stream, emit usage + pause_ack
  → on the ack: inference is over, Stop is hidden, Send appears, the Pause button
    leaves the row with it (it only exists while streaming); the partial reply
    stays in the thread
  → the user MUST type a message to start inference again — that comment is the
    only way back; there is no "resume" button
  → that send re-posts the whole history, partial turn included
```

**The pause is one-way.** Asking to interrupt means the user has something to say,
so the recovery is always a real user message. This removes the whole resume
machine (no ▶ button, no synthetic "continue" turn, no hidden-message plumbing)
and leaves Pause as exactly three moving parts: a flag, a terminal frame, and a
composer that reopens.

## Verdict: the hard half is already free

**Context retention needs no server state at all.** `POST /api/chat`
(`routes.py:1129`) is stateless: every send rebuilds the full history client-side
(`useChat.js:838`, `state.messages.map(messagePayload)`). This is the same
property that made session save/load free (`todos/done/session-saving.md:30-36`).
So Pause reduces to: *interrupt cleanly* + *keep the partial assistant message in
`state.messages` instead of splicing it out*.

| Part | Scope |
|---|---|
| Interrupt + flag plumbing | `core/chat.py` + one route + `RunState`-like registry |
| Protocol | one new terminal SSE frame `pause_ack` |
| Composer/UI | one button, one phase, textarea unlocked |

## Where the interrupt must land (two loops, not one)

`stream_chat` (`core/chat.py:108`) nests two loops, and both need a pause check:

1. **Chunk loop of one round** (`:248` `for chunk in response_generator`) — the
   check the brief describes ("before each iteration"). Put it next to the existing cancel check
   (`:249-251`), which already does the right cleanup shape:
   `close_llm_stream(response_generator)` then `return`.
2. **Tool-calling loop** (`:212` `while True`, tool execution at `:390-437`) —
   with MCP/LSP active, a pause during a 60 s Playwright call cannot land on the
   chunk loop. Needs a check *before each tool execution* (in the
   `for _, call in sorted(tool_calls.items())` body) and *between rounds*
   (next to the post-tool cancel check at `:438`).

Note on the brief's three boundaries ("assistant message, reasoning message,
tool call"): reasoning deltas are **not** a separate loop — they arrive as chunks
of the same loop (`:261-265`), so the per-chunk check covers them for free.

A pause that lands between rounds has already streamed a *complete* assistant
turn carrying `tool_calls`. Keeping it is actually the best case: re-posting that
turn plus a `role:"tool"` stub ("interrupted by the user") is a legal OpenAI
sequence — but note `ChatMessage.role` is `Literal["user","assistant"]`
(`routes.py:237`), so the client cannot re-post it today. **Decided out of scope
(D10)**: the client never holds a `tool_calls` turn, so this case can only be
*observed* server-side (the pause lands there, the fragment is dropped) — it
needs no re-posting support to be correct. The real usage also sits far from it:
the user watches the reasoning stream, sees the direction drift, and pauses
*before* any tool call runs — which is also what keeps unwanted tool side effects
out of the picture.

## Two implementation options

### Option A — client-only (no backend change at all)

A Pause button that calls `controller.abort()` like `stop()` (`useChat.js:935-938`)
but **skips the discard block** (`useChat.js:924-930`), so the partial assistant
message survives in `state.messages`. Because the history is re-posted from the
client, continuing works for free.

Costs of A: the server learns about the interrupt only through
`request.is_disconnected()` polled every 0.25 s (`routes.py:1189-1194`), so
llama.cpp keeps generating into a dead socket for up to a chunk loop's worth of
work; token/time badges fall back to the ~4 chars/token estimate
(`finish()` at `:778-785`, no exact `usage` frame arrives); and "paused" is
indistinguishable client-side from a network drop — no explicit protocol signal.

### Option B — the endpoint (the brief's design) ← recommended

`POST /api/chat/pause` sets the flag; the loop acks with a terminal `pause_ack`.
Buys: a clean server-side close (`close_llm_stream` → the generation slot is
aborted at the next chunk), an **exact** `usage` frame emitted before the ack
(`chat.py:365` already does this for the tool path), an unambiguous terminal
event, and the same semantics whether or not a tool is mid-flight.

Both options share the frontend half, so **Phase 2/3 below are valid for A as
well** — A can ship as a fallback while B is built.

---

## Phase 1 — backend (Option B)

### 1.1 Flag carrier: a pause event next to the cancel event

`threading.Event`, symmetric with `cancel_event`. `stream_chat` grows one
optional keyword (defaults `None`, like `cancel_event`, so the ~20 existing
`stream_chat` call sites in `tests/test_chat.py` keep working):

```python
# core/chat.py, stream_chat signature (after cancel_event)
cancel_event: Optional[threading.Event] = None,
pause_event: Optional[threading.Event] = None,
```

### 1.2 Registry: which reply is pausable

The pause route is a *different* HTTP request from the chat stream, so the flag
must live in a process-wide registry (precedent: `RunState` at `routes.py:416`
and the module-level `_CHAT_TASKS` set at `routes.py:294`).

**Targeting (decided: mono-user)** — the app is single-user/single-tab, so no
`chatId` is needed: one module-level *current* handle, exactly the shape `RunState`
already uses. Simpler than a dict, and it matches the invariant that a benchmark
run and a chat reply are each already exclusive (`routes.py:1138`).

```python
# routes.py, module level
class ChatHandle:
    """The one in-flight chat reply (mono-user app, so a single slot)."""
    pause_event: threading.Event
    cancel_event: threading.Event
    task: asyncio.Task

_CHAT_HANDLE: Optional[ChatHandle] = None
```

(`routes.py` imports no `dataclass` today — plain class or a 3-tuple keeps it
consistent with the file's current style.)

- `post_chat` sets the handle at the top of the endpoint and **always** clears it
  in the `event_stream` `finally` (`routes.py:1201`), so a finished/aborted reply
  can never be paused afterwards.
- New route (plain `200`, as the brief specifies):

```python
@router.post("/chat/pause", status_code=200)
async def post_chat_pause() -> Dict[str, Any]:
    global _CHAT_HANDLE
    handle = _CHAT_HANDLE
    if handle is None:
        raise HTTPException(status_code=409, detail="No chat reply is streaming")
    if handle.pause_event.is_set():
        return {"paused": "pending"}      # idempotent: the flag is already armed
    handle.pause_event.set()
    return {"paused": "requested"}
```

No request body at all: there is exactly one pausable reply.
`post_chat` must pass `pause_event` through `asyncio.to_thread(stream_chat, ...)`
(`routes.py:1159-1175`) — it is positional today, so append it after `cancel_event`
or switch that call to keywords (safer, matches how `agent_instructions` is
already passed).

## What the next send is (and what a pause is *not*)

Pausing calls `close_llm_stream()`, which **aborts the llama.cpp generation slot**
for that request. There is no server-side session to hand back to, so nothing can
be "decoding continues from the last token" — no inference engine supports that
across two HTTP requests here. And since the pause is one-way, the app never
pretends otherwise: the only continuation is an ordinary new `POST /api/chat`.

What that send does: it re-posts the history ending with the partial assistant
turn, and the model generates a **new** continuation on top of it. That is exactly
what the chat template does with a trailing assistant message (llama.cpp appends
the assistant header and lets the model keep going), which is why keeping the
fragment — plus the `[interrupted by the user]` marker, D4 — is what makes the
continuation land where the user expects.

Two consequences to keep visible while designing:

1. **The continuation is a fresh request**: sampling is re-drawn, the prompt is
   re-prefilled (the server's own prompt cache may absorb it — that is a
   llama.cpp concern, not the app's). Token/time badges of the new reply are
   independent numbers, like any other send.
2. **Nothing is left half-open.** Once `pause_ack` is out and the SSE closes, the
   reply is finished from both sides' point of view: no stream, no handle, no
   pending flag. The `finally` at `routes.py:1201` still sets `cancel_event`
   (stopping any tool still running), which is what makes that true.

### 1.3 The interrupt itself, in `core/chat.py`

One small helper, used from all three sites, so the cleanup shape stays single:

```python
def _emit_paused(emit, *, completion_tokens, decode_seconds, context_tokens,
                 context_estimated, cancel_event):
    emit({"event": "usage", ...})          # exact figures, as the tool path does
    emit({"event": "pause_ack", "partial": True})
```

Call sites:

| Site | today | with pause |
|---|---|---|
| chunk loop `:249` | `cancel_event` → close + return | `elif pause_event.is_set()` → `close_llm_stream` + `_emit_paused` + return |
| before tool exec `:390` | none | same, *before* executing the call |
| between rounds `:438` | `cancel_event` → return | same as above |

**Mandatory guard — the truncation check.** The block at `:301-309` fails the
round when `not saw_finish_reason(response_generator)`. A pause *itself* closed
that stream, so `received_finish_reason` is `None` by construction and every
pause would emit "The answer was cut short by the server…". The pause path must
`return` **before** that check (the cancel path already does), and the check
should additionally be skipped when `pause_event.is_set()` — that covers the
rare ordering where the stream dies a moment before the flag is observed.

### 1.4 Terminal-event contract

`chat.py` docstring (`:147`) promises "exactly one of `done` or `error`".
It becomes **`done` | `error` | `pause_ack`**, with precedence
**cancel > error > pause** (an error already emitted must not be overwritten by a
late pause ack; a cancel means the client is gone, so nothing is emitted — the
existing rule, unchanged).

`routes.py:1199` must learn the new terminal name:

```python
if item.get("event") in ("done", "error", "pause_ack"):
    return
```

Without that, the generator keeps polling a queue that will never yield again,
and `finally` sets `cancel_event` on an already-finished worker (harmless but the
SSE socket lingers until the next disconnect poll).

The `finally` at `:1201-1204` still sets `cancel_event` after the ack: that is
correct (it stops any tool still running) but it must happen *after*
`pause_ack` is flushed, which it does.

### 1.5 Backend tests (`tests/test_chat.py`)

Follow the existing style (`TestStreamChatTruncation` at `:37`, monkeypatched
`completion` returning a fake wrapper):

1. `pause_event` set → events end on `pause_ack`, no `done`, no `error`.
2. Same, asserting **no truncation error frame** (guards 1.3).
3. A generator of *many* chunks with the flag set after N → exactly N-ish tokens
   emitted, then the ack (proves the check is per-chunk).
4. `usage` frame precedes `pause_ack` with the exact cumulative tokens.
5. Tool loop: `tool_executor` that flips `pause_event` (or a fake generator that
   does) → the next tool call is **not** executed.
6. Route level: `POST /api/chat/pause` with no reply streaming → 409; while one
   streams → 200 and the stream terminates with `pause_ack` (stub `stream_chat`,
   as `:529`/`:550` do); a second call before the ack → 200 `{"paused":"pending"}`.

## Phase 2 — frontend: keep the context (shared by A and B)

### 2.1 `useChat.js` — a `paused` outcome that is not an abort

The one line that carries the feature: the `finally` block
(`:923-932`) splices the assistant message out when `userStopped`. Pause needs a
third outcome.

With the **endpoint** path there is nothing to guard: the server ends the SSE with
`pause_ack`, `readStream` returns normally, no `AbortError`, `userStopped` stays
false → `finish()` keeps the partial turn. The flag is only needed for the **Option
A fallback**, which does abort:

```js
let userStopped = false      // existing
let userPaused = false      // new: an abort that must NOT discard
```

```js
// finally, :924
if (userStopped && !userPaused) {   // pause: keep the fragment
```

In `applyEvent` (`:624`) add `case 'pause_ack'`:

- close the generation window (`closeGenerationWindow`, as `usage` does at `:695`);
- fold the exact figures onto the message (`tokens`, `decode`, `contextTokens`)
  so the badges read the server's numbers, not the estimate;
- mark the turn `assistant.paused = true` (the marker rendered like `truncated`,
  `:746`), and initialise `paused: false` in the message literal (see Phase 3);
- set `state.phase = 'idle'` — this is what re-enables the composer, hides Stop
  and shows Send;
- clear `state.pausePending = false` — the ⏸ stops blinking exactly when the
  stream does. `finish()` must clear it too (E10: a pause whose ack never comes
  because the server died).

`finish()` (`:769`) then behaves as for a normal end: it already keeps partial
text on a connection drop, so the retained fragment needs no extra work beyond
not splicing. Tools left `state: 'running'` already flip to `'error'` (`:791`).

`pause()` (new, exported next to `stop()` at `:935`). One module-level flag drives
the button: `state.pausePending` (requested, ack not yet received). There is no
`resume()` — the pause is one-way and the user's next **Send** is the recovery:

```js
async function pause() {
  if (state.phase === 'idle' || state.pausePending) return   // blinking = disabled
  state.pausePending = true
  try {
    await fetch('/api/chat/pause', { method: 'POST' })       // no body: one handle
  } catch {
    // Option A fallback: local abort that keeps the fragment (see above). The
    // abort ends readStream, so finish() runs and clears pausePending (E10).
    userPaused = true
    if (controller) controller.abort()
  }
}
```

### 2.2 The button is a 2-state button that leaves the row on the ack

`streaming` is `state.phase !== 'idle'` (`WorkBenchInput.vue:56`), and the ack sets
`phase = 'idle'` — so `v-if="streaming"` makes the Pause button **disappear at the
ack**, which is precisely what the brief asks for: Stop is hidden, Send appears
(the existing `v-if="!streaming"` / `v-else` pair at `:373`/`:388` already does
that), and the Pause button returns to no longer being there.

```html
<button v-if="streaming" class="icon-btn chat-pause" type="button"
        :disabled="state.pausePending"
        :class="{ 'is-paused-blink': state.pausePending }"
        :aria-label="state.pausePending ? 'Pausing…' : 'Pause — keep the partial reply, add a comment'"
        @click="pause()">
  <!-- ⏸ two bars -->
</button>
```

| State | trigger | look | clickable |
|---|---|---|---|
| arming | streaming, no request sent | ⏸ | yes |
| pending | after the POST, before `pause_ack` | ⏸ blinking (`animation: bi-pulse`) | **no** (`:disabled`) |
| — | after `pause_ack` | gone (`streaming` false) | n/a |

No `state.paused` flag is needed at all: after the ack the app is simply idle with
a retained fragment, which is a state the code already handles (a truncated reply
sits in the thread the same way). Restart and Stop stay available in the pending
state (Stop keeps its destructive meaning), and `restart()` must clear
`pausePending` — see Phase 3.

`send()` guard: it refuses unless `state.phase === 'idle'` (`:822`), which the ack
sets — so typing a comment and sending it works with **no change**, the history at
`:838` already holds the partial assistant turn.

### 2.3 Anchoring `[interrupted by the user]` in the re-posted history (decided)

The fragment must not read to the model as a finished turn. **Decided: append the
marker.** Where it is appended matters, and the code already shows the safe way:
`messagePayload` (`:369-372`) strips the render-only `TOOL_LINE_MARKER` sentinel
out of `raw` before posting. Same trick, one level up — keep the *displayed* text
clean and add the marker at payload-build time only:

```js
// messagePayload() — after the TOOL_LINE_MARKER strip
if (message.paused) raw += '\n\n[interrupted by the user]'
```

so `state.messages[i].raw` (the bubble the user reads) is untouched, and the model
sees the note. Alternative — writing it into `raw` — is one line shorter but shows
the marker inside the bubble; not recommended.

One consequence of appending it here rather than in `raw`: `conversationJson`
(`:435`) builds each saved turn through `messagePayload` too, so a **saved session
carries the marker inside `content`** and a reload makes it part of what the model
is shown again. That is acceptable — arguably desirable, the fragment stays marked
wherever the conversation travels — but it makes D8 slightly wider than it looks:
what is *not* saved is the `paused` **flag** (so the ⏸ caption disappears on
reload, as `truncated`'s does), not the marker text.

This also subsumes what today happens silently for truncated replies: those keep
posting their fragment as a finished turn (the docstring at `chat.py:297` calls
that out). Optionally reuse the same note for `truncated`, which is a separate
call (it changes the benchmark-adjacent behaviour of a cut reply).

### 2.4 The `paused` marker in the thread

`WorkBenchView.vue:122` shows the sticky "Cut short by the server" caption for
`message.truncated`. Add the sibling, with different wording (the user caused
this one, and the fragment is deliberately part of the context):

```html
<p v-if="message.paused" class="caption chat-paused" role="status">
  ⏸ Interrupted by you — the partial reply above is kept in the conversation.
</p>
```

`.chat-paused` beside `.chat-truncated` (`theme.css:1772`); the icon-button
styling can reuse `.chat-stop` (`theme.css:1545`) with its own accent.

---

## Phase 3 — the Pause button (`WorkBenchInput.vue`)

The markup and its two states are in **2.2**; it slots into the action row's right
group (`:358-400`), between Restart and the Send **xor** Stop pair
(`v-if="!streaming"` / `v-else`) — so at the ack, Send takes Stop's place and the
⏸ leaves the row with it.

Notes on the UI contract:

- The **textarea is not disabled today** (only the buttons are gated by `canSend`,
  `:57-60`), so "make the textarea available" is automatic once `phase` returns to
  `idle`; the user can even type during the stream, they just cannot send.
- Everything else that keys off `streaming` behaves correctly once the ack lands,
  because the conversation really is idle again: `pickSessionFile` (`:190`),
  attach (`:272`), `canSave` (`:161`), `editLast` (`useChat.js:941`), the MCP/LSP
  toggles (`:175`/`:209`), and the hub-restart guards (`routes.py:824`/`:960`/
  `:1044`). The toggles are an accepted side effect **(D9)**: a pause *un-freezes*
  them mid-conversation, which the sticky-lock design (`:910-915`) did not
  anticipate. Leave it — arguably a feature (swap a connector before writing the
  comment), and the next `send()` re-freezes them. No code change.
- `restart()` (`:76`) must clear `pausePending` (and `userPaused`) too — otherwise
  a Pause request in flight while the user hits Restart leaves a blinking ⏸ on a
  conversation that no longer streams.
- Blinking: `@keyframes bi-pulse` already exists (`theme.css:266`, used by
  `.chat-typing:1765`) — `animation: bi-pulse 1.2s ease-in-out infinite` on
  `.is-paused-blink` reuses it. Disabled + blinking is the brief's
  "clignote = non cliquable".
- `state.pausePending` must live in the `useChat` reactive `state` object (top of
  the file, `:6` area), not in a module-local `let`: the button's `:disabled` and
  its blink class both read it from `WorkBenchInput.vue`. The per-message `paused`
  flag lives on the message object instead (like `truncated`, `:870`).
- The `paused: false` field belongs in the assistant-message literal in `send()`
  (`:870`, next to `truncated: false`) so a paused fragment round-trips through the
  session-load shape too.

---

## Edge cases and races (each needs a decision or a test)

| # | Case | Proposed handling |
|---|---|---|
| E1 | Pause flag set, stream ends on its own a moment later | First terminal frame wins: `done` (the loop's own `break` path) — the client sees `done`, ignores a late `pause_ack` (guard `if (!assistant.streaming) return` in `applyEvent`). |
| E2 | Pause during tool execution (Playwright, 60 s) | The ack waits for the tool to return, then the between-rounds check fires. Accept it (Stop already covers the hard case). *Optional*: `asyncio.Task` cancellation as well — out of scope. |
| E3 | Pause already requested, user clicks Pause again | Unreachable by clicking: the button is `:disabled` while it blinks. The route stays idempotent (200 `{"paused":"pending"}`) for any second POST. |
| E4 | Pause requested, then the user clicks Stop | `stop()` aborts the fetch; server-side `cancel_event` wins over `pause_event` (check cancel **first** at every site). Client discards the partial reply (Stop semantics, unchanged). |
| E5 | Pause while the server is still prefilling (no chunk yet) | The chunk loop never runs, so the flag is only seen at the first chunk — the ack can take a few seconds while ⏸ blinks. Accepted (the button is disabled meanwhile, so nothing is inconsistent). Empty-turn handling: see E11. |
| E6 | Pause then the browser reloads before the ack | Server sees the disconnect → cancel path. Nothing to keep, consistent. |
| E7 | Pause accepted, user sends a comment *before* the ack lands | Cannot happen: `send()` requires `phase === 'idle'` (`:822`) and the phase only clears on `pause_ack`. The composer stays shut until the ack — which is exactly the "clignote = on attend" contract. |
| E7b | Paused, then Send with a comment (the only way forward) | An ordinary send: the partial turn is already in the history and `phase === 'idle'` satisfies the guard at `:822`. Nothing to clear — the only flag left over from the pause is `pausePending`, and the ack cleared it. |
| E7c | Paused, then Restart / Stop / session load / settings save | All allowed (the app is idle), all already behave correctly for a finished reply. Like `truncated`, `paused` is display state and is **not** written to the session JSON (`conversationJson` at `:435` serializes through `messagePayload`, so only `role`/`content`/`reasoning`/images survive) — a loaded session therefore loses the caption but keeps the fragment and its meaning, which is exactly `truncated`'s existing behaviour (D8). |
| E8 | Benchmark run in progress | Chat is refused then already (`routes.py:1138`), so Pause is unreachable. No work. |
| E9 | Re-posting the paused fragment | Solved by 2.3: the `[interrupted by the user]` note rides on the payload only, never on the displayed text. |
| E10 | Pause returned 200 but no ack ever arrives (server died mid-round) | The client's `finally` still runs `finish()` → `phase = 'idle'`, but `pausePending` would leave the ⏸ blinking forever. `finish()` (`:769`) must clear `state.pausePending` (the per-message `paused` flag stays as-is; it only drives a caption). |
| E11 | Pause with nothing generated yet (prefill) | `pause_ack` still terminates the reply, but `finish()` never removes an empty assistant turn — so the `pause_ack` handler needs the same guard the non-OK path already uses at `:904` (`if (!assistant.raw) splice`). Without it an empty assistant turn sits in the thread and gets re-posted (an empty turn is also what `routes.py:287` rejects, though only for user content). |

## Decisions (settled with the architect)

| # | Decision | Where it lands |
|---|---|---|
| D1 | Backend endpoint (Option B): `POST /api/chat/pause` → **200**; Option A's local-but-non-destructive abort stays as the `catch` fallback in `pause()`. | 1.2, 2.1 |
| D2 | **Single module-level handle**, no `chatId` — mono-user app, one tab. | 1.2 |
| D3 | **The pause is one-way**: no resume button, no second click. After the ack the only way back into inference is typing a message and pressing Send — asking to interrupt means having something to say. No synthetic turn, no `state.paused` flag. | 2.1, 2.2 |
| D4 | **`[interrupted by the user]` is appended** to the assistant fragment — at payload-build time only, so the bubble stays clean (the saved session does carry it, see 2.3). | 2.3 |
| D5 | **While it blinks the button is disabled**; Stop and Restart stay available and keep their meaning. | 2.2 |
| D6 | A pause during a **running tool** waits for that tool, then skips the next call. | E2 |
| D7 | **WorkBench only**; the benchmark run keeps `/run/stop`. | — |
| D8 | `paused` is display state: the *flag* is not written to the session JSON (like `truncated`), so the caption disappears on reload while the marker text rides on in `content`. | 2.3, 2.4 |
| D9 | **MCP/LSP toggles un-freeze after an ack — accepted, no code change.** Swapping a connector before writing the comment is legitimate, and the next `send()` re-freezes them. | Phase 3 notes |
| D10 | **A pause landing between tool-call rounds is out of scope** (the fragment is dropped; the client cannot re-post a `tool_calls` turn anyway). Not a real gap: the user pauses while watching the *reasoning* stream, i.e. before any tool call runs. | intro |
| D11 | Everything above is **settled** — no open question remains. Phase 1 (backend) can start in a code session on its own; Phases 2-3 depend on the `pause_ack` shape from 1.4. | — |

## Size estimate

| Phase | Files | Roughly |
|---|---|---|
| 1 — flag + route + 3 interrupt sites + contract | `src/core/chat.py`, `src/api/routes.py` | ~70 lines Python |
| 1.5 — tests | `tests/test_chat.py` | ~90 lines, 6 tests |
| 2 — retained-context outcome + `pause_ack` handling | `web/src/composables/useChat.js` | ~40 lines |
| 3 — pause button (2 states) + caption + css | `WorkBenchInput.vue`, `WorkBenchView.vue`, `theme.css` | ~35 lines |

No frontend test runner exists, so Phases 2-3 are verified with the manual
checklist below (same convention as `todos/done/session-saving.md`).

## Manual verification checklist

1. Long answer → Pause mid-stream → stream stops within ~1 chunk, partial text
   stays visible with the ⏸ marker, composer is usable, badges show exact tokens.
2. Send the comment immediately → the model receives the history *including* the
   partial turn (check the server log / the model's behaviour).
3. Pause during an MCP tool call (`console` `sleep`) → the ack lands when the tool
   returns; the tool badge does not stay "running".
4. Pause during a tool *loop* with several calls → the next call is not executed.
5. Pause then Restart (while it blinks) → `pausePending` cleared, conversation
   wiped, no ghost `pause_ack` (the guard in `applyEvent` also covers it).
6. Pause then Stop before the ack → partial reply discarded (Stop semantics).
7. Pause after the answer already ended → button is gone (`streaming` false).
8. Click the ⏸ while it blinks → nothing happens (disabled); Stop and Restart
   still work from that state.
9. No `truncated` caption ever appears next to a paused reply (guards E1/1.3).
10. After the ack, Send is enabled and Send-with-empty-draft is not (the
    `canSend` guard, `WorkBenchInput.vue:57-60`) — the user cannot post a blank
    turn, so the comment really is the resume.
11. The next send continues the cut sentence rather than restarting the answer
    (proves D4's marker reaches the model).
12. Kill the server mid-reply after a Pause → the ⏸ stops blinking (E10), the
    composer is usable.
13. `pytest` green with `BENCHMARKIT_DISABLE_MCP=1` / `_LSP=1`.

## Follow-ups (not in scope)

- Pausing a *benchmark run* to annotate a question (needs a per-question pause in
  `core/bench.py` and a different UI story).
- A resumable "Continue" that reuses the server-side KV context by keeping the
  llama.cpp slot alive — out of scope: the app has no server-side session, and the
  server manages its own cache.
- Hard-cancel of a *running* MCP tool on pause (D6 waits for it instead).

