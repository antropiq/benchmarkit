import { expect, test } from '@playwright/test'
import fs from 'node:fs'
import path from 'node:path'

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
  // real disk: a brand-new workspace ends up with exactly the five files the
  // app seeds there — the policy (pol-05), the agent-modes file (am-03), the
  // prompt-helpers file (mbl-02/mbl-03), the MCP declaration (from the
  // committed template) and the LSP *template* only (the declaration itself is
  // hand-written, so it must not appear here) — and nothing else.
  expect(fs.readdirSync(path.join(scratchAbs, '.nojae')).sort()).toEqual([
    'agent-modes.json',
    'lsp-settings.template.json',
    'mcp-settings.json',
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
