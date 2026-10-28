import { expect, test } from '@playwright/test'
import { existsSync } from 'node:fs'
import { join } from 'node:path'

// The Editor's write loop as a human performs it: create, type, save, reload,
// rename, delete — and the refusals a user meets on the way.
//
// This is the cheapest end-to-end proof in the repo: every step is the app's
// own Editor surface and **no LLM call is needed**. The server is the witness
// for the result (the file is read back through the API, never inferred from
// what the DOM drew), and the server's *wording* is part of the assertion for
// the refusals: a connector rule a human cannot read is a rule that gets
// worked around.
//
// The scratch lives under `logs/` — gitignored, already part of what the tree
// browses, and outside every protected name. Each test starts by trashing it,
// because a leftover from an interrupted run would turn this run's create into
// a 409 and its rename into a collision with a collapsed row.
const SCRATCH = 'logs/e2e-write'

async function trash(request, path) {
  return request.delete(`/api/fs/file?path=${encodeURIComponent(path)}`)
}

async function create(request, path, dir = false) {
  return request.post('/api/fs/create', { data: { path, dir } })
}

async function diskStat(request, path) {
  const res = await request.get(`/api/fs/stat?path=${encodeURIComponent(path)}`)
  return (await res.json()).paths[0]
}

async function diskExists(request, path) {
  return (await diskStat(request, path)).exists
}

// A protected name is the one path `diskExists` cannot ask about: `/api/fs/stat`
// answers it with a 422 refusal rather than `exists: false`, so the probe can
// never be used to learn whether a secret file is there (and the tree omits
// denied siblings, so it cannot witness it either). So "nothing was created"
// for a denied name is checked on the filesystem, with the working location the
// server itself reports — which also makes it an independent witness, with no
// app code in the loop.
async function existsOnDisk(request, rel) {
  const res = await request.get('/api/path-location')
  const { path: root } = await res.json()
  if (!res.ok() || !root) {
    throw new Error(`cannot resolve the working location: ${await res.text()}`)
  }
  return existsSync(join(root, rel))
}

// Null when the route refuses the path, which is how "the old name is gone"
// reads in the assertions below: 404 from the read route, not a missing file
// the spec has to imagine.
async function diskText(request, path) {
  const res = await request.get(`/api/fs/file?path=${encodeURIComponent(path)}`)
  if (!res.ok()) return null
  return (await res.json()).text
}

async function openEditor(page) {
  await page.goto('/')
  await page.getByRole('tab', { name: 'Editor' }).click()
  await expect(page.locator('.editor-nav-tree')).toBeVisible()
}

// The tree is lazy and a folder row's click *toggles* it, so opening a folder
// means "click it unless it is already open" — clicking unconditionally would
// collapse the branch and the row below it would be gone.
async function openFolder(page, path) {
  const row = page.locator(`.tree-row[data-path="${path}"]`).first()
  if ((await row.getAttribute('aria-expanded')) !== 'true') await row.click()
}

async function openFolders(page, ...paths) {
  for (const path of paths) await openFolder(page, path)
}

// One item of a row's context menu, chosen by label — the menu is the feature
// under test, so the spec reaches every operation through it rather than
// through the keyboard chords (whose own coverage is the Vitest suite's).
async function useMenuItem(page, path, label) {
  await page.locator(`.tree-row[data-path="${path}"]`).first().click({ button: 'right' })
  await page.getByRole('menuitem', { name: label }).click()
}

async function menuItems(page, path) {
  await page.locator(`.tree-row[data-path="${path}"]`).first().click({ button: 'right' })
  const items = page.locator('.tree-context-menu [role="menuitem"]')
  await expect(items.first()).toBeVisible()
  const labels = await items.allTextContents()
  await page.keyboard.press('Escape')
  return labels
}

// Type at the end of the buffer. Monaco's caret target is zero-sized until the
// surface is focused, so the caret is placed with a click on the text and the
// keyboard — there is no input to fill, and this is the only way to put bytes
// in the pane from a browser.
async function typeAtEnd(page, text) {
  await page.locator('.monaco-host .view-lines').first().click()
  await page.keyboard.press('End')
  await page.keyboard.type(text)
}

test.beforeEach(async ({ request }) => {
  await trash(request, SCRATCH)
  const created = await create(request, SCRATCH, true)
  expect(created.ok(), `could not create the scratch folder: ${await created.text()}`).toBe(
    true,
  )
})

test.afterEach(async ({ request }) => {
  await trash(request, SCRATCH)
})

test('create, type, save, reload — the bytes are on disk, not just on screen', async ({
  page,
  request,
}) => {
  await openEditor(page)
  await openFolders(page, 'logs', SCRATCH)

  // A folder and a file inside it, both from the tree's own menu.
  await useMenuItem(page, SCRATCH, 'New folder…')
  await page.locator('.tree-name-input').fill('inner')
  await page.keyboard.press('Enter')
  const inner = `${SCRATCH}/inner`
  await expect(page.locator(`.tree-row[data-path="${inner}"]`)).toHaveAttribute(
    'aria-expanded',
    'true',
  )

  const file = `${inner}/note.txt`
  await useMenuItem(page, inner, 'New file…')
  await page.locator('.tree-name-input').fill('note.txt')
  await page.keyboard.press('Enter')

  // A created file arrives as a *clean* buffer: it is on disk, and the pane
  // agrees with it. The row and the disk both say so.
  const tab = page.locator('.editor-tab')
  await expect(tab).toHaveCount(1)
  await expect(tab.locator('.editor-tab-label')).toHaveText('note.txt')
  await expect(page.locator('.editor-tab-dirty')).toHaveCount(0)
  expect(await diskExists(request, file)).toBe(true)

  // Typing makes it dirty — the dot, the strip's title, the enabled Save.
  await typeAtEnd(page, 'written from e2e')
  await expect(page.locator('.editor-tab-dirty')).toHaveCount(1)
  await expect(page.locator('.editor-tabs')).toHaveAttribute('title', '1 unsaved')
  await expect(page.locator('.editor-save-button').first()).toBeEnabled()

  // Ctrl+S: the dot going away is what says the write landed.
  await page.keyboard.press('Control+s')
  await expect(page.locator('.editor-tab-dirty')).toHaveCount(0)
  await expect(page.locator('.editor-save-button').first()).toBeDisabled()
  await expect(page.locator('.editor-notice--error')).toHaveCount(0)
  expect(await diskText(request, file)).toBe('written from e2e')

  // A reload re-reads the same text, so the buffer was never the only copy.
  await page.reload()
  await openEditor(page)
  await openFolders(page, 'logs', SCRATCH, inner)
  await page.locator(`.tree-row[data-path="${file}"]`).first().click()
  await expect(page.locator('.editor-tab-label')).toHaveText('note.txt')
  await expect(page.locator('.editor-tab-dirty')).toHaveCount(0)
  expect(await diskText(request, file)).toBe('written from e2e')
})

test('rename from the menu — the row moves, the chip follows, the buffer saves to the new path', async ({
  page,
  request,
}) => {
  const from = `${SCRATCH}/renamed-me.txt`
  const to = `${SCRATCH}/moved-name.txt`
  expect((await create(request, from)).ok()).toBe(true)
  const seeded = await request.put('/api/fs/file', {
    data: { path: from, text: 'keep me', ifUnmodifiedSince: (await diskStat(request, from)).mtime },
  })
  expect(seeded.ok(), await seeded.text()).toBe(true)

  await openEditor(page)
  await openFolders(page, 'logs', SCRATCH)
  await page.locator(`.tree-row[data-path="${from}"]`).first().click()
  await expect(page.locator('.editor-tab')).toHaveCount(1)

  // Dirty it *first*: the proof is that a dirty buffer stays dirty across the
  // move, and its save token is still valid at the new path.
  await typeAtEnd(page, ' and edited here')
  await expect(page.locator('.editor-tab-dirty')).toHaveCount(1)

  await useMenuItem(page, from, 'Rename…')
  const input = page.locator('.tree-name-input')
  await expect(input).toBeFocused()
  await expect(input).toHaveValue('renamed-me.txt')
  await input.fill('moved-name.txt')
  await page.keyboard.press('Enter')

  await expect(page.locator(`.tree-row[data-path="${to}"]`)).toBeVisible()
  await expect(page.locator(`.tree-row[data-path="${from}"]`)).toHaveCount(0)
  expect(await diskExists(request, from)).toBe(false)
  expect(await diskExists(request, to)).toBe(true)

  // Same tab, new label, still dirty, still the only tab.
  const tab = page.locator('.editor-tab')
  await expect(tab).toHaveCount(1)
  await expect(tab.locator('.editor-tab-label')).toHaveText('moved-name.txt')
  await expect(page.locator('.editor-tab-dirty')).toHaveCount(1)

  await page.keyboard.press('Control+s')
  await expect(page.locator('.editor-tab-dirty')).toHaveCount(0)
  await expect(page.locator('.editor-notice--error')).toHaveCount(0)
  expect(await diskText(request, to)).toBe('keep me and edited here')
})

test('delete — the row is gone, and the notice names the trash folder the route used', async ({
  page,
  request,
}) => {
  const file = `${SCRATCH}/keep-me-anyway.txt`
  expect((await create(request, file)).ok()).toBe(true)

  await openEditor(page)
  await openFolders(page, 'logs', SCRATCH)

  await useMenuItem(page, file, 'Delete')
  const dialog = page.locator('.editor-delete-dialog')
  await expect(dialog).toBeVisible()
  await expect(dialog.locator('h2')).toHaveText('Delete keep-me-anyway.txt')
  // Cancel is the default answer of a confirm, so Esc keeps the file.
  await page.keyboard.press('Escape')
  await expect(dialog).not.toBeVisible()
  expect(await diskExists(request, file)).toBe(true)

  await useMenuItem(page, file, 'Delete')
  await expect(dialog).toBeVisible()
  // The notice quotes the route's answer, so the assertion compares the two
  // instead of pattern-matching a folder name the server invents per delete.
  const responsePromise = page.waitForResponse(
    (res) => res.url().includes('/api/fs/file') && res.request().method() === 'DELETE',
  )
  await dialog.locator('.editor-delete-confirm').click()
  const body = await (await responsePromise).json()
  expect(body.dir).toBe(false)

  await expect(page.locator(`.tree-row[data-path="${file}"]`)).toHaveCount(0)
  const notice = page.locator('.editor-notice--ok').first()
  await expect(notice).toContainText(file)
  await expect(notice).toContainText(body.trashedTo)

  // The batch is named repo-relative so a human can cd to it, it is real, and
  // it holds the *path* that was deleted — the layout that makes a restore a
  // plain mv. (The timestamp inside the name is deliberately not asserted on.)
  expect(body.trashedTo).toContain('logs/trash/')
  const branch = await (
    await request.get(`/api/fs/tree?path=${encodeURIComponent(body.trashedTo)}`)
  ).json()
  expect(branch.entries.map((entry) => entry.name)).toEqual(['logs'])
  const inside = await (
    await request.get(
      `/api/fs/tree?path=${encodeURIComponent(`${body.trashedTo}/logs/e2e-write`)}`,
    )
  ).json()
  expect(inside.entries.map((entry) => entry.name)).toEqual(['keep-me-anyway.txt'])
  expect(await diskExists(request, file)).toBe(false)
})

test('each row offers exactly what it can honour', async ({ page }) => {
  await openEditor(page)
  await openFolders(page, 'logs', SCRATCH)
  const file = `${SCRATCH}/menu.txt`
  expect((await create(page.request, file)).ok()).toBe(true)
  await useMenuItem(page, SCRATCH, 'Refresh')

  // The root lists its own actions but is never renamable or deletable; an
  // untracked file with no buffer offers neither View diff… nor Close.
  expect(await menuItems(page, '')).toEqual([
    'New file…',
    'New folder…',
    'Refresh',
    'Copy relative path',
  ])
  expect(await menuItems(page, SCRATCH)).toEqual([
    'New file…',
    'New folder…',
    'Rename…',
    'Delete',
    'Refresh',
    'Copy relative path',
  ])
  expect(await menuItems(page, file)).toEqual([
    'Open',
    'Rename…',
    'Delete',
    'Copy relative path',
  ])
})

test('a refusal reaches the DOM in the server\'s own words', async ({ page, request }) => {
  await openEditor(page)
  await openFolders(page, 'logs', SCRATCH)

  // Creating `.env` — a name the write gate refuses. The copy is the deny-list's
  // sentence: in the row the name was typed into, and in the notice stack for a
  // user whose eyes are elsewhere.
  await useMenuItem(page, SCRATCH, 'New file…')
  await page.locator('.tree-name-input').fill('.env')
  await page.keyboard.press('Enter')
  await expect(page.locator('.tree-name-input-error')).toContainText('connector path policy')
  await expect(page.locator('.tree-name-input-error')).toContainText('never write to it')
  await expect(page.locator('.editor-notice--error').first()).toContainText(
    `Could not create ${SCRATCH}/.env`,
  )
  expect(await existsOnDisk(request, `${SCRATCH}/.env`)).toBe(false)

  // A move into the connectors' own folder: an innocent source, a protected
  // target. The input stays open holding the typed value, so the fix is to type
  // again, and nothing moved.
  const file = `${SCRATCH}/innocent.txt`
  expect((await create(request, file)).ok()).toBe(true)
  // Dismiss the create input (it stays open on a refusal, by design) and let the
  // tree learn about the row the API just made — the same Refresh the menu offers
  // an already-open branch, so the row below can be right-clicked.
  await page.keyboard.press('Escape')
  await useMenuItem(page, SCRATCH, 'Refresh')
  await expect(page.locator(`.tree-row[data-path="${file}"]`)).toBeVisible()
  await useMenuItem(page, file, 'Rename…')
  const input = page.locator('.tree-name-input')
  await input.fill('mcp_connectors/policy_patch.py')
  await page.keyboard.press('Enter')
  await expect(page.locator('.tree-row-error')).toContainText('connector path policy')
  await expect(page.locator('.tree-row-error')).toContainText('mcp_connectors')
  await expect(input).toBeVisible()
  await expect(input).toHaveValue('mcp_connectors/policy_patch.py')
  expect(await diskExists(request, file)).toBe(true)
  await page.keyboard.press('Escape')
})

test('a folder renamed with F2 keeps its branch open under the new name', async ({
  page,
  request,
}) => {
  // Seeded through the API, so the spec does not depend on the previous test
  // having left anything behind.
  const inner = `${SCRATCH}/inner`
  const file = `${inner}/keep-me.txt`
  expect((await create(request, inner, true)).ok()).toBe(true)
  expect((await create(request, file)).ok()).toBe(true)

  await openEditor(page)
  await openFolders(page, 'logs', SCRATCH, inner)
  await page.locator(`.tree-row[data-path="${inner}"]`).first().focus()
  await page.keyboard.press('F2')
  await page.locator('.tree-name-input').fill('renamed-inner')
  await page.keyboard.press('Enter')

  // The branch the user was looking at is still open, under the new name, and
  // its children came with it.
  await expect(
    page.locator(`.tree-row[data-path="${SCRATCH}/renamed-inner"]`),
  ).toHaveAttribute('aria-expanded', 'true')
  await expect(
    page.locator(`.tree-row[data-path="${SCRATCH}/renamed-inner/keep-me.txt"]`),
  ).toBeVisible()
  expect(await diskExists(request, file)).toBe(false)
  expect(await diskExists(request, `${SCRATCH}/renamed-inner/keep-me.txt`)).toBe(true)
})

test('a dirty buffer is never closed by one gesture — the three answers all mean what they say', async ({
  page,
  request,
}) => {
  const file = `${SCRATCH}/closing.txt`
  expect((await create(request, file)).ok()).toBe(true)

  await openEditor(page)
  await openFolders(page, 'logs', SCRATCH)
  await page.locator(`.tree-row[data-path="${file}"]`).first().click()
  await typeAtEnd(page, 'work that exists nowhere else')
  await expect(page.locator('.editor-tab-dirty')).toHaveCount(1)

  // The close gesture asks, naming the file. Escape is Cancel, not Discard.
  await page.keyboard.press('Control+w')
  const dialog = page.locator('.editor-close-dialog')
  await expect(dialog).toBeVisible()
  await expect(dialog.locator('h2')).toHaveText('Close closing.txt?')
  await expect(dialog.locator('footer button')).toHaveText([
    'Cancel',
    'Discard — the changes are lost',
    'Save and close',
  ])
  await page.keyboard.press('Escape')
  await expect(dialog).not.toBeVisible()
  await expect(page.locator('.editor-tab')).toHaveCount(1)
  await expect(page.locator('.editor-tab-dirty')).toHaveCount(1)
  expect(await diskText(request, file)).toBe('')

  // The destructive answer is the second one, and it says so.
  await page.locator('.editor-tab-close').click()
  await dialog.getByRole('button', { name: 'Discard — the changes are lost' }).click()
  await expect(page.locator('.editor-tab')).toHaveCount(0)
  expect(await diskText(request, file)).toBe('')

  // Asked again with a disk that accepts it, save-and-close writes first.
  await page.locator(`.tree-row[data-path="${file}"]`).first().click()
  await typeAtEnd(page, 'this one lands')
  await page.locator('.editor-tab-close').click()
  await dialog.getByRole('button', { name: 'Save and close' }).click()
  await expect(page.locator('.editor-tab')).toHaveCount(0)
  expect(await diskText(request, file)).toBe('this one lands')
})

test('the badge says what the pane will do, including what is unsaved', async ({ page }) => {
  // Only the *idle* half is driven here: making the badge read "read-only"
  // needs a live streaming reply, and no spec in this file may depend on a
  // model answering. The server's own chat-stream gate on every write route is
  // asserted in tests/test_editorfs_write.py.
  const file = `${SCRATCH}/badge.txt`
  expect((await create(page.request, file)).ok()).toBe(true)

  await openEditor(page)
  await openFolders(page, 'logs', SCRATCH)
  await page.locator(`.tree-row[data-path="${file}"]`).first().click()
  await expect(page.locator('.editor-tab')).toHaveCount(1)

  const badge = page.locator('.editor-mode-badge')
  await expect(badge).toHaveText('read-write')
  await expect(badge).toHaveAttribute('title', /WorkBench is idle/)
  await expect(badge).toHaveAttribute('title', /not written to disk yet/)

  await typeAtEnd(page, 'a change')
  await expect(badge).toHaveAttribute('title', /Ctrl\+S writes them/)

  await page.keyboard.press('Control+s')
  await expect(page.locator('.editor-tab-dirty')).toHaveCount(0)
  await expect(badge).not.toHaveAttribute('title', /Ctrl\+S writes them/)
})
