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

function removeScratch(root) {
  fs.rmSync(path.join(root, SCRATCH), { recursive: true, force: true })
}

// The switch as a user performs it: Settings → Working directory → type → Set.
// The dialog stays open after `Set` (it is an apply, not a save), so the spec
// closes it the same way a user would before going back to the tree.
async function setRoot(page, path) {
  await page.getByRole('button', { name: 'Open settings' }).click()
  const dialog = page.locator('.settings-dialog')
  await expect(dialog).toBeVisible()
  await dialog.getByRole('button', { name: 'Working directory' }).click()
  const field = dialog.locator('.path-location-row input')
  await field.fill(path)
  await expect(dialog.locator('.path-location-set')).toBeEnabled()
  await dialog.locator('.path-location-set').click()
  await expect(page.locator('.appbar-workspace-path')).toHaveText(path)
  await dialog.getByRole('button', { name: 'Cancel' }).click()
  await expect(dialog).not.toBeVisible()
}

test.beforeEach(async ({ request }) => {
  removeScratch(await currentRoot(request))
})

// The root this server booted on, remembered so a spec that fails between its
// two switches cannot leave the whole suite pointing at a scratch folder.
let bootRoot = null

test.afterEach(async ({ request }) => {
  if (bootRoot === null) return
  const res = await request.post('/api/path-location', { data: { path: bootRoot } })
  expect(res.ok(), `could not restore the working location: ${await res.text()}`).toBe(true)
  removeScratch(bootRoot)
})

test('a switch closes the tabs of the root it leaves', async ({ page, request }) => {
  const original = await currentRoot(request)
  bootRoot = original
  expect((await create(request, SCRATCH, true)).ok()).toBe(true)
  // The same relative name on both sides of the switch, so a surviving tab
  // cannot be mistaken for a correctly re-opened one.
  expect((await create(request, `${SCRATCH}/README.md`)).ok()).toBe(true)
  const scratchAbs = `${original}/${SCRATCH}`

  await page.goto('/')
  await page.getByRole('tab', { name: 'Editor' }).click()
  await expect(page.locator('.editor-nav-tree')).toBeVisible()

  await setRoot(page, scratchAbs)

  // am-03's acceptance, on the real switch and the real disk: a brand-new
  // workspace ends up with exactly the two reserved-folder files — the policy
  // pol-05 seeds and the agent-modes file am-03 seeds — and nothing else.
  expect(fs.readdirSync(path.join(scratchAbs, '.nojae')).sort()).toEqual([
    'agent-modes.json',
    'read-write-policy.json',
  ])
  // And the seeded file is a real document, not a stub: the personas the loader
  // falls back to are the ones this workspace can now edit.
  const modes = JSON.parse(fs.readFileSync(path.join(scratchAbs, '.nojae', 'agent-modes.json'), 'utf8'))
  expect(Object.keys(modes.agents)).toEqual(['chat', 'code', 'architect'])

  // The tree re-rooted: the root row names the new folder and lists its own
  // README.md, and nothing of the previous root's listing is on screen.
  const rootRow = page.locator('.tree-row--root')
  await expect(rootRow).toHaveAttribute('title', new RegExp(scratchAbs))
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
  await expect(rootRow).toHaveAttribute('title', new RegExp(original))
  // The scratch folder the spec created is back in the listing (it is a
  // top-level folder of this workspace, and no policy rule names it). What the
  // root can *not* list is `.nojae` — reserved, and holding this workspace's
  // policy, its agent-modes file (am-03) *and* its trash — and `logs`, which
  // benchmarkit's own rules hide as the app's backups.
  await expect(page.locator('.tree-row[data-path="e2e-ws-root"]')).toBeVisible()
  await expect(page.locator('.tree-row[data-path="logs"]')).toHaveCount(0)
  await expect(page.locator('.tree-row[data-path=".nojae"]')).toHaveCount(0)
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
  const scratchAbs = `${original}/${SCRATCH}`

  await page.goto('/')
  await setRoot(page, scratchAbs)

  const res = await request.delete('/api/fs/file?path=gone.txt')
  expect(res.ok(), await res.text()).toBe(true)
  const body = await res.json()
  // Working-location-relative, the digest-and-stamp batch name the delete
  // invents — and this time the location it is relative to is the new root.
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
  const scratchAbs = `${original}/${SCRATCH}`

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
