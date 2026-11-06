import { expect, test } from '@playwright/test'
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

// The working location is what every workspace-relative path in the app is
// measured from, so moving it invalidates all of them at once — including the
// file an Editor tab is showing. The bug this spec pins: after a switch the tab
// strip kept the README.md of the *previous* root while the tree had moved on,
// so the two halves of the Editor disagreed about which project was open. A
// switch now closes the tabs of the root it leaves.
//
// No LLM call is needed. The new root is a top-level scratch folder (gitignored
// and out of the way of the repo's own files), and the location is restored
// afterwards so the other specs still see the boot-time root. It is *not* under
// `logs/` any more: benchmarkit's own rules hide that tree from the Editor, so
// a folder in it could not be created through the API at all.
//
// The scratch root is removed with node's `fs`, not through `/api/fs/file`:
// switching into it seeds a `.nojae/read-write-policy.json` there (pol-05), and
// that folder is reserved — the Editor will not delete a subtree that contains
// it, by design. So the spec cleans up the way a user would, from outside the
// app.
const SCRATCH = 'e2e-ws-root'

async function create(request, path, dir = false) {
  return request.post('/api/fs/create', { data: { path, dir } })
}

async function currentRoot(request) {
  return (await (await request.get('/api/path-location')).json()).path
}

// Windows: the unlink of a folder that existed for seconds can fail EPERM
// while a handle to it is still closing — anything the app (or the OS) ran
// inside the switched-in root holds its current-directory handle until it is
// reaped: the `git` of a status query (`editorfs/gitstatus.py`), the indexer,
// the scan of the freshly seeded files. `maxRetries`/`retryDelay` is Node's
// bound for exactly this on a directory (EPERM/EBUSY/ENOTEMPTY). A folder
// that still will not go is not this spec's failure: it is gitignored, the
// next `beforeEach` tries again, and one genuinely in the way then fails
// loudly at that spec's own `create` (409) — which names the real problem.
function removeScratch(root) {
  try {
    fs.rmSync(path.join(root, SCRATCH), {
      recursive: true,
      force: true,
      maxRetries: 10,
      retryDelay: 300,
    })
  } catch {
    // Leftover; the next `beforeEach` retries, or `create` reports it.
  }
}

// The scratch root as a switch target: the boot root (`GET /api/path-location`,
// the app's own witness, so already canonical) plus the folder's name. Folded
// to `/` because the app normalises a `\`-shaped input anyway (win-03) — the
// value only has to *reach* the root, its spelling is the host's.
function expectedPath(root, rel) {
  return path.join(root, rel).replaceAll(path.sep, '/')
}

// The install root, from this file's own `../../..` — the way
// `playwright.config.js` locates the repo, and the same join that file uses to
// find the suite's settings fixture.
const REPO_ROOT = path.dirname(path.dirname(path.dirname(fileURLToPath(import.meta.url))))

// The suite's own model connection: the file the server under test is configured
// from, because `playwright.config.js` exports `BENCHMARKIT_TEST_SETTINGS` and
// refuses to start the suite without it. Read from disk with `node:fs` — the
// folder is reserved, so no Editor route can see it — and resolved the way the
// app resolves it: `BENCHMARKIT_PATH_LOCATION` when set, else the repo root.
//
// This is the assertion that pins "the suite talks to the fixture, never to
// wherever the app happens to be pointed": the dialog's `serverUrl` field is
// compared to *this file's* value, so a spec that started reaching the
// developer's workspace settings fails instead of passing.
const TEST_SETTINGS_FILE = path.join(
  (process.env.BENCHMARKIT_PATH_LOCATION || '').trim() || REPO_ROOT,
  '.nojae',
  'nojae-test-settings.json',
)
function testSettingsDocument() {
  return JSON.parse(fs.readFileSync(TEST_SETTINGS_FILE, 'utf8'))
}

// The app's own spelling of a path — `os.path.realpath`'s answer, which spells
// the host separator (win-01: the shape is the OS's, never a name check). A
// comparison with a `/`-shaped value folds both sides, which is what lets one
// assertion work whatever the host.
const fold = (p) => p.replaceAll('\\', '/')

// The switch as a user performs it: the app bar's working-location field, then
// its Set button. There is nothing to close afterwards — the field *is* the
// location, and Set is an apply, so the bar shows the new root as soon as the
// server has answered.
//
// `target` is always absolute: the route refuses anything else with 422, so a
// relative one would leave the location exactly where it was while the page
// looked like it had moved. The assertion folds the separators on both sides —
// the sentence these lines are here for is "the app bar moved to the root this
// switch asked for", the app's behaviour, and the host's spelling of a path may
// not decide it.
async function setRoot(page, target) {
  const field = page.locator('.appbar-location-input')
  await field.fill(target)
  const button = page.locator('.appbar-location-set')
  await expect(button).toBeEnabled()
  await button.click()
  await expect
    .poll(async () => fold(await field.inputValue()))
    .toContain(fold(target))
}

test.beforeEach(async ({ request }) => {
  removeScratch(await currentRoot(request))
})

// The root this server booted on, remembered so a spec that fails between its
// two switches cannot leave the whole suite pointing at a scratch folder.
let bootRoot = null

test.afterEach(async ({ request }) => {
  if (bootRoot === null) return
  // The *live* location decides, not a remembered flag: a spec that ended its
  // own switch back owes nothing, and one that died mid-way owes exactly one
  // restore. A switch restarts both hubs and costs seconds, so the check keeps
  // that one free — and it cannot drift out of sync the way a flag can.
  if ((await currentRoot(request)) !== bootRoot) {
    const res = await request.post('/api/path-location', { data: { path: bootRoot } })
    expect(res.ok(), `could not restore the working location: ${await res.text()}`).toBe(true)
  }
  removeScratch(bootRoot)
})

test('a switch closes the tabs of the root it leaves', async ({ page, request }) => {
  const original = await currentRoot(request)
  bootRoot = original
  expect((await create(request, SCRATCH, true)).ok()).toBe(true)
  // The same relative name on both sides of the switch, so a surviving tab
  // cannot be mistaken for a correctly re-opened one.
  expect((await create(request, `${SCRATCH}/README.md`)).ok()).toBe(true)
  const scratchAbs = expectedPath(original, SCRATCH)

  await page.goto('/')
  await page.getByRole('tab', { name: 'Editor' }).click()
  await expect(page.locator('.editor-nav-tree')).toBeVisible()

  await setRoot(page, scratchAbs)

  // The acceptance for the reserved folder's seeding, on the real switch and the
  // real disk: a brand-new workspace ends up with exactly the six files the app
  // seeds there — the policy (pol-05), the agent-modes file (am-03), the
  // prompt-helpers file (mbl-02/mbl-03), the MCP declaration (from the
  // committed template), the workspace settings file (ws-02, from
  // `nojae-settings.default.json`) and the LSP *template* only (the declaration
  // itself is hand-written, so it must not appear here) — and nothing else.
  expect(fs.readdirSync(path.join(scratchAbs, '.nojae')).sort()).toEqual([
    'agent-modes.json',
    'lsp-settings.template.json',
    'mcp-settings.json',
    'nojae-settings.json',
    'prompt-helpers.json',
    'read-write-policy.json',
  ])
  // And the seeded file is a real document, not a stub: the personas the loader
  // falls back to are the ones this workspace can now edit.
  const modes = JSON.parse(fs.readFileSync(path.join(scratchAbs, '.nojae', 'agent-modes.json'), 'utf8'))
  expect(Object.keys(modes.agents)).toEqual(['chat', 'code', 'architect'])

  // The tree re-rooted: the root row names the new folder and lists its own
  // README.md, and nothing of the previous root's listing is on screen.
  const rootRow = page.locator('.tree-row--root')
  // The name only, so the match cannot depend on how the host spells a path.
  await expect(rootRow).toHaveAttribute('title', new RegExp(SCRATCH))
  await page.locator('.tree-row[data-path="README.md"]').first().click()
  const tabs = page.locator('.editor-tab')
  await expect(tabs).toHaveCount(1)
  await expect(tabs.locator('.editor-tab-label')).toHaveText('README.md')

  await setRoot(page, original)

  // The bug this spec exists for: the tab used to survive this, still showing
  // the file of the root that was just left.
  await expect(tabs).toHaveCount(0)
  await expect(page.locator('.editor-body .stream-empty-title')).toHaveText('Open a file')
  // The old root is re-rooted too, and listed for real — the forget is not a
  // blank tree.
  await expect(rootRow).toHaveAttribute('title', new RegExp(path.basename(original)))

  // The scratch folder the spec created is back in the listing (it is a
  // top-level folder of this workspace, and no policy rule names it). What the
  // root can *not* list is `.nojae` — reserved, and holding this workspace's
  // policy, its agent-modes file (am-03) *and* its trash — and `logs`, which
  // benchmarkit's own rules hide as the app's backups.
  await expect(page.locator('.tree-row[data-path="e2e-ws-root"]')).toBeVisible()
  await expect(page.locator('.tree-row[data-path="logs"]')).toHaveCount(0)
  await expect(page.locator('.tree-row[data-path=".nojae"]')).toHaveCount(0)
})

// The nine prompt helpers behind the WorkBench bullets belong to the workspace,
// so a switch has to replace what the row shows — without the user leaving the
// chat tab and coming back, which until this feature was the row's only other
// fetch. The scratch root is given its own helper through the app's writer
// (`POST /api/prompt-helpers`, which can only address the active workspace) and
// the row is then re-read on a *later* switch back into that root, the tab never
// left in between: the only thing that can put that text on screen is the
// reload `setPathLocation()` triggers.
//
// The boot root's own file is only ever read here, never written — its bytes are
// compared at the end, so the test shows the write landed in the workspace it
// was pointed at. No LLM call either: the bullet row is fetched, not generated.
test('a switch replaces the prompt-helper bullets the composer offers', async ({
  page,
  request,
}) => {
  const original = await currentRoot(request)
  bootRoot = original
  expect((await create(request, SCRATCH, true)).ok()).toBe(true)
  const scratchAbs = expectedPath(original, SCRATCH)
  const bootHelpers = path.join(original, '.nojae', 'prompt-helpers.json')
  // Read from outside the app, because the folder is reserved: no Editor route
  // can see it, which is the point of mbl-06.
  const bootBytes = fs.existsSync(bootHelpers) ? fs.readFileSync(bootHelpers, 'utf8') : null

  // Open WorkBench and wait for the row's own first fetch, so what is read next
  // is the boot root's file and not the composable's nine-empty fallback.
  await page.goto('/')
  const firstLoad = page.waitForResponse(
    (response) => new URL(response.url()).pathname === '/api/prompt-helpers',
  )
  await page.getByRole('tab', { name: 'WorkBench' }).click()
  await firstLoad

  const bullets = page.locator('.chat-bullet-row .memory-bullet')
  await expect(bullets).toHaveCount(9)
  // A filled bullet carries its text in `title` and an empty one has no title at
  // all, which is how the row is read without reaching into the composable.
  const titles = () => bullets.evaluateAll((els) => els.map((el) => el.getAttribute('title')))
  const bootTitles = await titles()

  // The switch reloads the row itself, un-awaited, so `setRoot` can report the
  // new root while that GET is still in flight. Await the answer here: were the
  // write below to land first, the switch's own load would put the sentinel on
  // screen — and the "a write from outside is no reason to change the screen"
  // assertion would then be measuring the order of two requests.
  const switchLoad = page.waitForResponse(
    (response) => new URL(response.url()).pathname === '/api/prompt-helpers',
  )
  await setRoot(page, scratchAbs)
  await switchLoad

  // One recognisable helper, written by the app's own writer while the scratch
  // root is the active workspace (`POST /api/prompt-helpers` can only address
  // that workspace, which is the whole point of the move).
  const sentinel = 'the scratch workspace says hello'
  const res = await request.post('/api/prompt-helpers', {
    data: { index: 0, color: '#2ec27e', content: sentinel },
  })
  expect(res.ok(), await res.text()).toBe(true)
  // It landed in *that* workspace's reserved folder...
  const stored = JSON.parse(
    fs.readFileSync(path.join(scratchAbs, '.nojae', 'prompt-helpers.json'), 'utf8'),
  )
  expect(stored.helpers[0].content).toBe(sentinel)
  // ...and not in the boot root's file at all.
  if (bootBytes !== null) {
    expect(fs.readFileSync(bootHelpers, 'utf8')).toBe(bootBytes)
  }
  // And a write from outside the page is no reason for the screen to change:
  // only the switch itself reloads the row.
  await expect(page.locator(`.memory-bullet[title="${sentinel}"]`)).toHaveCount(0)

  // Away and back into that root, the tab never left: its own helper is on
  // screen, which only the reload a switch triggers can put there.
  const reloaded = page.waitForResponse(
    (response) => new URL(response.url()).pathname === '/api/prompt-helpers',
  )
  await setRoot(page, original)
  await reloaded
  const backToScratch = page.waitForResponse(
    (response) => new URL(response.url()).pathname === '/api/prompt-helpers',
  )
  await setRoot(page, scratchAbs)
  await backToScratch
  await expect(page.locator(`.memory-bullet[title="${sentinel}"]`)).toHaveCount(1)

  await setRoot(page, original)
  await expect(page.locator(`.memory-bullet[title="${sentinel}"]`)).toHaveCount(0)
  await expect
    .poll(async () => (await titles()).join('\n'))
    .toBe(bootTitles.join('\n'))
})

// The other half of "one value moves everything": the trash moves with the
// root too. Deleting inside a switched-in workspace must file its batch under
// *that* workspace's reserved folder — not under the install directory the app
// booted from, where a foreign project's deleted bytes have no business being.
test('a delete in a switched-in workspace trashes under that workspace', async ({
  page,
  request,
}) => {
  const original = await currentRoot(request)
  bootRoot = original
  expect((await create(request, SCRATCH, true)).ok()).toBe(true)
  expect((await create(request, `${SCRATCH}/gone.txt`)).ok()).toBe(true)
  const scratchAbs = expectedPath(original, SCRATCH)

  await page.goto('/')
  await setRoot(page, scratchAbs)

  const res = await request.delete('/api/fs/file?path=gone.txt')
  expect(res.ok(), await res.text()).toBe(true)
  const body = await res.json()
  // Working-location-relative, the digest-and-stamp batch name the delete
  // invents — and this time the location it is relative to is the new root.
  // `/`-joined on every host (A §7): a payload field never carries `\`.
  expect(body.trashedTo).toMatch(/^\.nojae\/trash\/[0-9a-f]{12}_/)
  // The batch lives under the *switched-in* root, layout and all, and the
  // install directory gained nothing.
  expect(fs.existsSync(path.join(scratchAbs, body.trashedTo))).toBe(true)
  expect(fs.existsSync(path.join(scratchAbs, body.trashedTo, 'gone.txt'))).toBe(true)
})

test('unsaved edits are reported when the switch closes their tab', async ({ page, request }) => {
  const original = await currentRoot(request)
  bootRoot = original
  expect((await create(request, SCRATCH, true)).ok()).toBe(true)
  expect((await create(request, `${SCRATCH}/note.txt`)).ok()).toBe(true)
  const scratchAbs = expectedPath(original, SCRATCH)

  await page.goto('/')
  await page.getByRole('tab', { name: 'Editor' }).click()
  await setRoot(page, scratchAbs)
  await page.locator('.tree-row[data-path="note.txt"]').first().click()
  await expect(page.locator('.editor-tab')).toHaveCount(1)

  await page.locator('.monaco-host .view-lines').first().click()
  await page.keyboard.type(' written here only')
  await expect(page.locator('.editor-tab-dirty')).toHaveCount(1)

  await setRoot(page, original)

  // Closing a dirty tab is the switch's doing, not the user's gesture, so the
  // loss gets a line rather than the confirm whose Save answer no longer has a
  // path to write to.
  await expect(page.locator('.editor-tab')).toHaveCount(0)
  const notice = page.locator('.editor-notice--warn').first()
  await expect(notice).toContainText('1 tab closed with unsaved changes')
})

// The headline behaviour of the workspace-scoped model connection, seen through
// the suite's own fixture: the Settings dialog means a *different* file per
// workspace, so it has to name a different path and answer with different values
// on either side of a switch — and give the original ones back on the way home.
//
// The server under test runs with `BENCHMARKIT_TEST_SETTINGS` exported, so the
// file in use at **both** roots is `nojae-test-settings.json`, and the boot
// root's values are compared to that file's own bytes rather than to themselves:
// the assertion that would have caught a suite silently talking to the
// developer's workspace settings. Nothing is saved — a group-1 save is refused
// in this mode, and the dialog is only ever read.
//
// The scratch root has no test settings file (the app never seeds one), so its
// dialog is the `absent` state with the reader's own sentence — the state
// production can never reach, because there the seed always creates the file.
test('the settings dialog follows the workspace it is opened in', async ({
  page,
  request,
}) => {
  const original = await currentRoot(request)
  bootRoot = original
  expect((await create(request, SCRATCH, true)).ok()).toBe(true)
  const scratchAbs = expectedPath(original, SCRATCH)
  // The path the dialog is expected to name, per root — the **test** file, which
  // is the one in use while the switch is on, in the host's own spelling.
  const settingsFile = (root) => path.join(root, '.nojae', 'nojae-test-settings.json')
  // The LLM section's own fields: the verifier section has a 'Server URL' of
  // its own, and `v-show` keeps it in the DOM, so an unscoped label is two
  // elements.
  const llmSection = page.locator('.settings-editor > section').first()
  const serverField = llmSection.getByLabel('Server URL')
  // The LLM section's own note (the workspace file's report), not the dialog's
  // top-level save errors — same scoping reason as the field above.
  // `.first()` because the section renders other error lines further down.
  const note = llmSection.locator('.alert-error').first()
  const hint = page.locator('.settings-hint')
  const openSettings = async () => {
    // The dialog renders from the payload it already holds and is filled by the
    // GET it fires on open, so the *response* is the moment the new workspace's
    // values are on screen — the same reason the bullets spec above awaits its
    // own reload. Asserting before it would read the previous workspace.
    const loaded = page.waitForResponse(
      (response) => new URL(response.url()).pathname === '/api/settings',
    )
    await page.getByRole('button', { name: 'Open settings' }).click()
    await loaded
    await expect(hint).toBeVisible()
  }

  await page.goto('/')

  // 1. At the boot root: the test file, named in the dialog, holding the values
  // read from that same file off disk — not compared to themselves, which is the
  // assertion that pins the suite onto its fixture.
  await openSettings()
  expect(fold(await hint.innerText())).toContain(fold(settingsFile(original)))
  expect(fold(await hint.innerText())).not.toContain(
    fold(path.join(original, '.nojae', 'nojae-settings.json')),
  )
  const bootServer = await serverField.inputValue()
  expect(bootServer).toBe(testSettingsDocument().serverUrl)
  // A configured fixture answers no note at all.
  await expect(note).toHaveCount(0)
  await page.locator('.settings-dialog').getByRole('button', { name: 'Cancel' }).click()

  // 2. In the scratch root: its own path, and the `absent` state — the app never
  // seeds a test settings file, so this workspace has none and the dialog says so
  // rather than showing empty fields with no explanation.
  await setRoot(page, scratchAbs)
  await openSettings()
  expect(fold(await hint.innerText())).toContain(fold(settingsFile(scratchAbs)))
  expect(fold(await hint.innerText())).not.toContain(fold(settingsFile(original)))
  await expect(serverField).toHaveValue('')
  await expect(note).toHaveCount(1)
  await expect(note).toContainText('there is no workspace settings file at')
  expect(fold(await note.innerText())).toContain(fold(settingsFile(scratchAbs)))
  await page.locator('.settings-dialog').getByRole('button', { name: 'Cancel' }).click()

  // 3. Home: the boot root's test file is named again and its own values are back.
  await setRoot(page, original)
  await openSettings()
  expect(fold(await hint.innerText())).toContain(fold(settingsFile(original)))
  await expect(serverField).toHaveValue(bootServer)
  await expect(note).toHaveCount(0)
  await page.locator('.settings-dialog').getByRole('button', { name: 'Cancel' }).click()
})

// A WorkBench reply at the boot root, so the suite is proven *configured* and
// not merely readable: with the switch on, a mispointed or malformed fixture
// refuses with the reader's own sentence naming the file, which is a far more
// honest failure than a timeout. Nothing is asserted about the wording — the
// endpoint is a real server, so only the shape of the stream is.
test('the boot root streams a reply from the test settings file', async ({
  page,
  request,
}) => {
  bootRoot = await currentRoot(request)

  await page.goto('/')
  await page.getByRole('tab', { name: 'WorkBench' }).click()
  await page.locator('.chat-textarea').fill('Reply with exactly one word.')
  await page.locator('.chat-send').click()

  // Both scoped to the WorkBench output column: tab views stay mounted through
  // `v-show`, and the Benchmark tab renders `.question-card`s of its own.
  const output = page.locator('.chat-output')
  const error = output.locator('.error-line')
  const replies = output.locator('.question-card')
  await expect(replies).toHaveCount(1)
  // Wait for the reply to **finish**: the composer swaps Send for Stop while one
  // streams, so Send being back is the stream's own end. Leaving it in flight
  // would make the next spec's working-location switch — or any other mutation —
  // answer 409 for a reason that has nothing to do with what it asserts.
  //
  // The bound is the test's own, not the 30 s expect default: this is the only
  // spec that streams, and it is the first thing to ask the model after the
  // server booted, so a cold server's first reply carries its load/warm-up time
  // — measured here at ~31 s against a 2 s warm one. Bounding that wait at 30 s
  // is a flake that costs the *next* spec too: the reply is still in flight, so
  // the shared afterEach's restore of the working location answers 409 and an
  // unrelated spec fails. Nothing is weakened — the stream must still end, and
  // the refusal below still fires well inside the bound. Kept under the test's
  // own 120 s budget (playwright.config.js), which is the wait that actually
  // fails the spec.
  await expect(page.locator('.chat-send:not(.chat-stop)')).toBeVisible({
    timeout: 90_000,
  })
  // A misconfigured fixture refuses with the reader's sentence on the error line
  // rather than streaming; a real reply lands in the card. Checking the refusal
  // first means a failure says which of the two happened.
  await expect(error).toHaveCount(0)
  const answer = replies.first().locator('.answer')
  await expect
    .poll(async () => (await answer.innerText()).trim().length)
    .toBeGreaterThan(0)
  // And the reply is the model's own, not a fragment of a cut stream.
  await expect(replies.locator('.chat-truncated')).toHaveCount(0)
})

