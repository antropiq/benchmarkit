# Port benchmarkit: Streamlit → FastAPI + Vue 3 (Vite) + Pico.css

## Context

`benchmarkit` is a ~880-line Streamlit app that benchmarks LLM endpoints: it streams each model answer into a panel, verifies it with a second LLM, and tracks score/token-rate/elapsed time. The Streamlit rerun-loop + `session_state` model is replaced by an explicit async run job + SSE event stream. Full replacement: Streamlit is removed once parity is reached.

## Locked decisions

- **Backend**: FastAPI + uvicorn. Benchmark loop runs in a worker thread (litellm sync API stays), pushing events onto a thread-safe in-memory event bus. SSE (`text/event-stream`) is the only realtime transport; `Last-Event-ID` replays buffered events for reconnects.
- **Frontend**: Vue 3 + Vite (plain JS, no TS), built to `web/dist`, served by FastAPI `StaticFiles`. Pico.css + ~150 lines of custom CSS recreate the Streamlit look (palette taken from `.streamlit/config.toml`; fonts via `@fontsource/inter` + `@fontsource/jetbrains-mono`). Markdown answers rendered with `marked` + `dompurify`.
- **Job model**: single-user, single active run (mirrors today's behavior). `POST /api/run` while running → `409`. Run state lives in memory; a server restart mid-run is treated as an abandoned run (same failure mode as losing a Streamlit session). Results still saved once at the end to `results.json`.
- **Chronometer semantics preserved**: elapsed = cumulative LLM answer time only (`sum took`), ticking live while an answer streams, showing `Checking…` while the verifier runs, frozen during verify. Driven purely by SSE event boundaries; 250 ms client interval, same as `src/ui.py`.

## Target project structure

```
benchmarkit/
├── pyproject.toml            # deps: fastapi, uvicorn[standard], litellm, pydantic>=2, python-dotenv
│                             # remove: streamlit, watchdog. script: benchmarkit = "src.api.server:run"
├── src/
│   ├── core/                 # UI-free benchmark core (importable, streamlit-free)
│   │   ├── models.py         # moved as-is from src/models.py
│   │   ├── llm.py            # ex-llm.py: take an `emit` callable, no st.* calls
│   │   ├── bench.py          # ex-bench.py: run_benchmark(questions, config, job, emit)
│   │   ├── job.py            # replaces state.py: RunJob holds results/index/failed + stats
│   │   └── utils.py          # format_time, save_results(job, path) — no st.* calls
│   ├── api/
│   │   ├── server.py         # create_app(), uvicorn run() entry, StaticFiles("web/dist")
│   │   ├── routes.py         # REST + SSE endpoints
│   │   └── events.py         # EventBus: seq-numbered buffer, thread-safe publish, async subscribe
│   └── config.py             # load_dotenv + questions.json + EndpointsConfig.from_env (ex-main.py helpers)
├── web/                      # Vite + Vue 3 (JS)
│   ├── package.json          # vue, marked, dompurify, @picocss/pico, @fontsource/inter, @fontsource/jetbrains-mono
│   ├── vite.config.js        # dev proxy: /api → http://localhost:8000 ; build.outDir = dist
│   └── src/
│       ├── main.js
│       ├── App.vue           # brand row + [1,4] grid: stats column | stream panel (CSS grid)
│       ├── styles/           # pico overrides: theme tokens, metric card, badge, stream box (3px border)
│       ├── composables/
│       │   └── useBenchmark.js  # EventSource client, run state store (reactive)
│       └── components/
│           ├── MetricCard.vue     # question-in-progress, score, token/s, failed
│           ├── Chronometer.vue    # HH:MM:SS, live/frozen/checking modes
│           ├── StartButton.vue
│           ├── ProgressBar.vue
│           ├── StreamPanel.vue    # autoscroll container, model badges, run_finished
│           └── QuestionCard.vue   # question, streamed markdown answer, Correct/Incorrect badge, took·tokens
├── resources/questions.json  # unchanged
├── .env / .env.example       # unchanged
└── tests/                    # migrated (see Task 6)
```

Delete after parity: `src/main.py`, `src/bench.py`, `src/llm.py`, `src/state.py`, `src/utils.py`, `src/config.py`, `src/ui.py`, `.streamlit/` (keep the palette by copying tokens into `web/src/styles/` first).

## SSE event contract

Every event carries an incrementing `id:` (sequence number) for `Last-Event-ID` replay. The EventBus buffers all events of the current run in memory.

| event | payload | UI effect |
|---|---|---|
| `run_started` | `{runId, models[], totalTests, samplingSummary}` | enable UI, clock starts at 0 |
| `model_started` | `{model}` | violet badge in stream panel |
| `question_started` | `{index, number, model, uuid, question}` | open QuestionCard, clock live-ticks from `base + now - startedAt` |
| `token` | `{text}` | append to current card's raw buffer, render markdown |
| `answer_done` | `{took, tokens}` | freeze display at `base + took`, switch clock to `Checking…` |
| `question_done` | `{index, succeeded, took, tokens}` | Correct/Incorrect badge, footer caption, update score/failed cards |
| `run_finished` | `{totalRunningTime}` | success message, unlock Start, enable results download |
| `run_error` | `{message, index?}` | inline error banner (non-fatal errors during a question also emit `question_done` with `succeeded:false`) |

Reconnect: client GETs `/api/run` snapshot, then subscribes to `/api/run/events` with `Last-Event-ID`; buffer replay rebuilds state. Snapshot response = current event list + status (`idle | running | finished`).

## REST API

- `GET /api/config` → `{models[], samplingSummary, questionCount, totalTests, verifierConfigured}` (422-style error payload if `.env` invalid — mirrors today's `ValidationError` shown in UI)
- `POST /api/run` → `202 {runId}`; `409` if already running; `400` if no models/questions
- `GET /api/run` → status snapshot
- `GET /api/run/events` → SSE stream
- `GET /api/results` → `FileResponse(results.json)` as download; `404` if absent (replaces `st.download_button`)
- Everything under `/api/*`; all other paths → `StaticFiles(web/dist, html=True)` SPA fallback

## Tasks (ordered)

1. **Core extraction (streamlit-free).** Move `models.py` → `src/core/models.py` untouched. Rewrite `llm.py` `call_llm_and_stream` to accept `emit(event: dict)` and drop `st.error`/`slot.markdown` (emit `token`/`run_error` instead); `verify_response` raises/returns on failure instead of `st.error`. Rewrite `state.py` as `RunJob` class (plain attributes + `add_result`, `get_stats`, `get_answered_time`, `get_consumed_tokens`, failed list) with no `st` imports. `bench.py`: `run_benchmark(questions, config, job, emit)` — same loop, emits `question_started/answer_done/question_done`, calls `save_results(job)` at end. `utils.py`: `save_results(job, path)` returns the JSON string, no st.
2. **Event bus + API.** `src/api/events.py`: seq-numbered list buffer; `publish()` thread-safe via `loop.call_soon_threadsafe` into per-subscriber `asyncio.Queue`; `subscribe()` async generator; replay from index. `routes.py` per contract above; run launched via `asyncio.to_thread(run_benchmark, ...)` with a module-level `RunJob` + guard flag; `GET /api/config` uses `src/config.py` (loads `.env` at startup, real env vars take precedence).
3. **Server + packaging.** `create_app()` registers `/api` router first, then static mount; `run()` = `uvicorn.run`. Update `pyproject.toml` deps/scripts. `mkdir web` placeholders so server boots before frontend exists.
4. **Vue scaffold.** `npm create vite@latest web -- --template vue`; add deps (vue, pico, marked, dompurify, fontsource); Vite proxy; styles: map Streamlit tokens (light + dark via `prefers-color-scheme`) to Pico CSS vars; metric card, 3px-bordered stream box, badges as custom classes.
5. **Frontend wiring.** `useBenchmark.js`: fetch `/api/config` on mount, `EventSource` with auto-reconnect + `Last-Event-ID`, exposes reactive state (cards, stream entries, clock state machine `idle|ticking|checking|frozen`). Components per structure table. Start button disabled when `totalTests==0 || !models || running`. Autoscroll on token append (throttled with `requestAnimationFrame`, not `sleep(0.05)` — drop the artificial throttle entirely).
6. **Test migration.** `test_models.py` unchanged. `conftest.py`: drop streamlit stub; `session_state` fixture → `RunJob` fixture. `test_llm/test_state/test_utils` retargeted to `src.core.*` with `emit` captured into a list. New: `test_events.py` (buffer/replay/subscribe), `test_api.py` (`TestClient` + mocked `completion`: start/409/config/events/results). Verify existing `test_config.py` still covers env parsing.
7. **Delete Streamlit.** Remove old modules, `.streamlit/`, streamlit+watchdog deps. Update `AGENTS.md` + `README.md` (commands: `pip install -e . && benchmarkit`; dev: `uvicorn src.api.server:app --reload` + `npm run dev`).

## Validation

- `pytest` green.
- Manual parity pass against old app with a live llama.cpp endpoint (or mocked): start → token stream renders per question → Correct/Incorrect badges → clock excludes verifier time → score/failed/token-rate cards match → progress reaches 100% → `results.json` written with identical schema → download works.
- Reconnect test: kill network mid-run, reload page, verify state rebuilds from snapshot + replay.
- Browser check: no mixed markdown/HTML injection (dompurify), autoscroll doesn't fight manual scroll-up (nice-to-have: pause autoscroll when scrolled away).

## Out of scope

- Multi-user / concurrent runs, persistent job history, database, auth.
- Run cancellation (not in today's app).
- TypeScript, component testing (Vitest), CI pipeline.

## Risks

- litellm sync streaming inside `to_thread` blocks one thread per run — fine for single-user.
- SSE through proxies can buffer; disable buffering with `X-Accel-Buffering: no` header.
- Token-event flood in Vue: append to a raw string, render markdown at most once per animation frame.
