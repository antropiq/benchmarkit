import { expect, test } from '@playwright/test'

// The working location is what every workspace-relative path in the app is
// measured from, so moving it invalidates all of them at once — including the
// file an Editor tab is showing. The bug this spec pins: after a switch the tab
// strip kept the README.md of the *previous* root while the tree had moved on,
// so the two halves of the Editor disagreed about which project was open. A
// switch now closes the tabs of the root it leaves.
//
// No LLM call is needed. The new root is a folder under `logs/` (already
// browsed, gitignored, outside every protected name), and the location is
// restored afterwards so the other specs still see the boot-time root.
const SCRATCH = 'logs/e2e-ws-root'

async function create(request, path, dir = false) {
  return request.post('/api/fs/create', { data: { path, dir } })
}

async function currentRoot(request) {
  return (await (await request.get('/api/path-location')).json()).path
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
  await request.delete(`/api/fs/file?path=${encodeURIComponent(SCRATCH)}`)
})

// The root this server booted on, remembered so a spec that fails between its
// two switches cannot leave the whole suite pointing at a scratch folder.
let bootRoot = null

test.afterEach(async ({ request }) => {
  if (bootRoot === null) return
  const res = await request.post('/api/path-location', { data: { path: bootRoot } })
  expect(res.ok(), `could not restore the working location: ${await res.text()}`).toBe(true)
  await request.delete(`/api/fs/file?path=${encodeURIComponent(SCRATCH)}`)
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
  await expect(page.locator('.tree-row[data-path="logs"]')).toBeVisible()
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
