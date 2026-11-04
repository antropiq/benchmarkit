import { existsSync, readFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { defineConfig, devices } from '@playwright/test'

// Layer 2: end-to-end tests. Playwright drives a real Chromium against the
// real FastAPI app (built SPA + /api + SSE), never the Vite dev server, so
// what is exercised is what `benchmarkit` actually serves.
//
// Not 8000: that is where the developer's own `benchmarkit` usually sits, and
// sharing it would make a run reuse a server pointed at the *run* endpoint and
// whatever build happened to be on disk — passing specs proving nothing. A
// dedicated port also means e2e and a manually-run app can coexist.
const PORT = Number(process.env.E2E_PORT ?? 8123)
const BASE_URL = `http://127.0.0.1:${PORT}`

// Which Chromium to drive — a "which binary is this" question, so the OS is the
// honest thing to ask here; no capability probe answers it.
//
// Linux: reuse the system Chromium the project's own MCP connector already
// drives, so e2e needs no browser download of its own (the pip `mcp` extra
// already carries a playwright). Verified against this build; on a machine
// where the version gap bites, unset PLAYWRIGHT_EXECUTABLE_PATH and run
// `npx playwright install chromium` to get the browser @playwright/test pins.
//
// Windows: that path names nothing, and a launch failure about a missing file
// is not an explanation of why the suite could not run — so the default there
// is *unset*, which makes Playwright use the browser its own version pins: the
// one `npx playwright install chromium` downloads. `e2e/global-setup.js` checks
// that browser is installed before any spec runs and names the command when it
// is not. Do not "repair" a Windows run by putting a hard-coded path back in
// this file — the pinned browser is the supported one there, and an explicit
// PLAYWRIGHT_EXECUTABLE_PATH stays the escape hatch on every platform.
const EXECUTABLE_PATH =
  process.env.PLAYWRIGHT_EXECUTABLE_PATH ??
  (process.platform === 'win32' ? undefined : '/usr/bin/chromium')

// The env file that defines the LLM endpoint for the server under test.
// `.env.test` (repo root, gitignored, copied from `.env.test.example`) names the
// dedicated test server: it is the only endpoint a spec may reach, so the suite
// never depends on — or silently passes against — the developer's own `.env`.
// Resolved from this file's own location, not from `cwd`, so the run works from
// `web/` or root.
const ENV_FILE =
  process.env.E2E_ENV_FILE ??
  join(dirname(fileURLToPath(import.meta.url)), '..', '.env.test')

// Loud on purpose. Were this file missing and the flag dropped quietly, specs
// would pass against the real run endpoint — the one failure mode worth
// failing hard on, since everything such a run asserts came from a
// configuration no other machine can repeat.
if (!existsSync(ENV_FILE)) {
  throw new Error(
    `E2E env file not found: ${ENV_FILE}\n` +
      'It is the only LLM endpoint a e2e spec may use. Point E2E_ENV_FILE at a ' +
      'file with the same key set as .env.example, and make sure the server it ' +
      'names is reachable.',
  )
}

// Parse `.env.test` into the environment of the server under test, so that
// file — not dotenv's precedence rules — decides what a spec talks to.
//
// Reading it here rather than handing uvicorn `--env-file` is deliberate:
// dotenv never overrides a variable that already exists, and the app itself
// exports every SETTING_KEYS value into os.environ when a settings save
// happens. A suite launched from inside a running app's process tree (the
// agent case, and any wrapper that inherits that environment) therefore
// carries the developer's live SERVER_URL, which would silently beat a file
// uvicorn loads later. `webServer.env` is applied last when Playwright spawns
// the server, so these values win no matter what the parent already exported.
//
// Supported syntax is the subset the file itself uses — `KEY=value`, `#`
// comments, and one level of double quoting. A commented-out key is exported
// as empty, which is what clears an ambient value instead of leaving it set.
function readEnvFile(path) {
  const values = {}
  for (const raw of readFileSync(path, 'utf8').split('\n')) {
    const line = raw.trim()
    if (!line) continue
    const commented = line.startsWith('#')
    const body = commented ? line.slice(1).trim() : line
    const eq = body.indexOf('=')
    if (eq === -1 || !/^[A-Za-z_][A-Za-z0-9_]*$/.test(body.slice(0, eq))) continue
    const key = body.slice(0, eq)
    if (commented) {
      values[key] = ''
      continue
    }
    let value = body.slice(eq + 1).trim()
    if (value.startsWith('"') && value.endsWith('"') && value.length > 1) {
      value = value.slice(1, -1).replace(/\\"/g, '"').replace(/\\\\/g, '\\')
    }
    values[key] = value
  }
  return values
}

const TEST_ENV = readEnvFile(ENV_FILE)

export default defineConfig({
  testDir: './e2e',
  outputDir: './test-results',
  // The app streams for as long as a model takes to answer, so the defaults
  // (5 s) are far too tight: an assertion here waits for a *stream*, not for a
  // DOM mutation. Timeouts only bound the wait — what makes a spec *deterministic*
  // is asserting on app state the composables own (an event count, a run state),
  // never on a sleep.
  timeout: 120_000,
  expect: { timeout: 30_000 },
  // A run touches one global server and one active benchmark run at a time,
  // so specs must not fight over it.
  fullyParallel: false,
  workers: 1,
  forbidOnly: Boolean(process.env.CI),
  reporter: [['list'], ['html', { open: 'never' }]],
  use: {
    baseURL: BASE_URL,
    // Streaming UI is the hardest thing here to debug: the trace records which
    // SSE chunk arrived when.
    trace: 'on-first-retry',
    screenshot: 'only-on-failure',
    video: 'retain-on-failure',
    actionTimeout: 30_000,
    // Desktop browsers only — the app has no mobile layout, so a viewport
    // matrix would only add noise.
    viewport: { width: 1440, height: 900 },
  },
  projects: [
    {
      name: 'chromium',
      use: {
        ...devices['Desktop Chrome'],
        // A path, not `channel` — channels name branded installs like "chrome".
        launchOptions: { executablePath: EXECUTABLE_PATH },
      },
    },
  ],
  webServer: {
    // The app under test, served by FastAPI from `web/dist`. Requires
    // `npm run build` to have produced dist (see the static mount in
    // src/api/server.py) — `globalSetup` does that instead of trusting the
    // developer to remember.
    //
    // No `--env-file`: the endpoint arrives through `env` below, which
    // overrides anything the invoking shell (or a parent app process) exported.
    // An override is still possible without editing either file —
    // `E2E_ENV_FILE=other.test npm run test:e2e` points the whole suite at a
    // different file, and neither one may hold a secret.
    command: `python -m uvicorn src.api.server:app --host 127.0.0.1 --port ${PORT}`,
    cwd: '..',
    url: `${BASE_URL}/api/health`,
    reuseExistingServer: !process.env.CI,
    timeout: 60_000,
    // Hermetic by default, mirroring tests/conftest.py on the Python side:
    // the endpoints stay live but no connector subprocess and no language
    // server is spawned for a spec that does not ask for one. A spec that
    // wants them starts its own server rather than loosening this globally.
    //
    // The endpoint is a real server, not a fake: answers then vary with the
    // model's sampling, so a spec must assert on the *shape* of what streams
    // (chunk order, state transitions, an event-seq gap after reconnect), never
    // on the wording of an answer. A stubbed OpenAI-compatible stream is the
    // next step if any spec needs an exactly-known token count.
    env: {
      ...TEST_ENV,
      BENCHMARKIT_DISABLE_MCP: '1',
      BENCHMARKIT_DISABLE_LSP: '1',
    },
  },
  globalSetup: './e2e/global-setup.js',
})
