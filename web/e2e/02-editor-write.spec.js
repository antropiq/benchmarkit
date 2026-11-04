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
// the refusals: a rule a human cannot read or change is a rule that gets
// worked around, so each refusal here is asserted by the sentence that names
// the level and the pattern.
//
// The scratch is a top-level folder — gitignored, and **visible to the pane**.
// It used to live under `logs/`, and that no longer works: benchmarkit's own
// rules hide `logs/**` from the tree, the read and the write routes, because
// that is where the app keeps its pre-save backups (the trash does not live
// there any more — it is each workspace's own `.nojae/trash`, reserved and
// hidden in every workspace). Each test starts by trashing the scratch,
// because a leftover from an interrupted run would turn this run's create into
// a 409 and its rename into a collision with a collapsed row.
const SCRATCH = 'e2e-write'
const TRASH_BATCH = '.nojae/trash'

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
// answers it with `exists: false` — indistinguishable from absence, which is
// what a hidden name means for anything that only looks (pol-00 D8) — so the
// probe cannot tell a hidden file from a missing one, in either direction.
// "Nothing was created" for a refused name is therefore checked on the
// filesystem, with the working location the server itself reports, which is also
// an independent witness with no app code in the loop.
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
  await openFolders(page, SCRATCH)

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
  await openFolders(page, SCRATCH, inner)
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
  await openFolders(page, SCRATCH)
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
  await openFolders(page, SCRATCH)

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

  // The batch is named **working-location**-relative so a human can cd to it,
  // it is real, and it holds the *path* that was deleted — the layout that
  // makes a restore a plain mv. (The timestamp inside the name is deliberately
  // not asserted on.) The trash now sits in the workspace's reserved folder,
  // which is hidden from the tree, the read and the write routes in *every*
  // workspace — so `existsOnDisk`, which joins the working location the server
  // itself reports, is the only witness left, and asserting on it is the point.
  expect(body.trashedTo).toContain(`${TRASH_BATCH}/`)
  expect(await existsOnDisk(request, body.trashedTo)).toBe(true)
  expect(await existsOnDisk(request, `${body.trashedTo}/${file}`)).toBe(true)
  expect(await diskExists(request, file)).toBe(false)

  // And the reserved folder that holds it is not a side door: the pane cannot
  // list or read it, whatever the batch inside.
  const trashedCopy = `${body.trashedTo}/${file}`
  expect((await request.get('/api/fs/tree?path=.nojae')).status()).toBe(404)
  expect(
    (await request.get(`/api/fs/file?path=${encodeURIComponent(trashedCopy)}`)).status(),
  ).toBe(404)
})

test('each row offers exactly what it can honour', async ({ page }) => {
  await openEditor(page)
  await openFolders(page, SCRATCH)
  const file = `${SCRATCH}/menu.txt`
  expect((await create(page.request, file)).ok()).toBe(true)
  // A markdown sibling, so the one difference between the two rows is the *name*
  // the row carries — `isMarkdown` is derived from it, before any read happens.
  const markdown = `${SCRATCH}/menu.md`
  expect((await create(page.request, markdown)).ok()).toBe(true)
  await useMenuItem(page, SCRATCH, 'Refresh')

  // The root lists its own actions but is never renamable or deletable; an
  // untracked file with no buffer offers neither View diff… nor Close.
  expect(await menuItems(page, '')).toEqual([
    'Search…',
    'New file…',
    'New folder…',
    'Refresh',
    'Copy relative path',
  ])
  expect(await menuItems(page, SCRATCH)).toEqual([
    'Search…',
    'New file…',
    'New folder…',
    'Rename…',
    'Delete',
    'Refresh',
    'Copy relative path',
  ])
  expect(await menuItems(page, file)).toEqual([
    'Open',
    'Search…',
    'Rename…',
    'Move to…',
    'Delete',
    'Copy relative path',
  ])
  // The `.md` row is the only difference, and it adds exactly Preview.
  expect(await menuItems(page, markdown)).toEqual([
    'Open',
    'Search…',
    'Preview',
    'Rename…',
    'Move to…',
    'Delete',
    'Copy relative path',
  ])
})

// ---------------------------------------------------------------------------
// The batch move — marked rows, one destination folder, one request. This is
// the only layer that can show what jsdom cannot: a `Ctrl`+click that marks two
// rows, marks that are *visible* (the row style, `aria-selected` and the
// navigator's count line), a `<dialog>` that survives a refusal with the typed
// value still in it (jsdom has no `showModal`), the exact text of a prefill,
// and two files that land on disk.
//
// Nothing here needs a model: every step is the Editor surface plus the write
// route. Success is asserted on the **server** (`/api/fs/stat`), the refusal on
// the **server's own sentence**; the DOM is only the evidence that the user was
// shown the row move.
//
// The scratch is a top-level folder for the same reason as everywhere else in
// this file, and it is never `todos/`: the lot in this repository is real work,
// and a test that moved one of its files would edit the lot.
const NESTED = `${SCRATCH}/nested`
const OTHER = `${SCRATCH}/other`

// Byte-exact seed through the write routes, with the mtime token the save guard
// demands — the way the rename test above seeds its known text.
async function seedFile(request, path, text) {
  expect((await create(request, path)).ok(), `could not create ${path}`).toBe(true)
  const res = await request.put('/api/fs/file', {
    data: { path, text, ifUnmodifiedSince: (await diskStat(request, path)).mtime },
  })
  expect(res.ok(), `could not seed ${path}: ${await res.text()}`).toBe(true)
}

// `<scratch>/nested/{a,b,c}.any` plus an empty `<scratch>/other/`: the three
// same-parent files a batch is made of, and the destination that exists.
async function seedScratchTree(request) {
  for (const dir of [NESTED, OTHER]) {
    expect((await create(request, dir, true)).ok(), `could not create ${dir}`).toBe(true)
  }
  for (const name of ['a.any', 'b.any', 'c.any']) {
    await seedFile(request, `${NESTED}/${name}`, `contents of ${name}`)
  }
}

// The move dialog is always mounted (no `v-if`, so a refusal keeps the field's
// caret and its half-typed value), so "closed" reads as `toBeHidden`.
const moveDialog = (page) => page.locator('.editor-move-dialog')
const moveField = (page) => page.locator('.editor-move-destination')
const moveError = (page) => page.locator('.editor-move-error')
const moveButton = (page) => page.locator('.editor-move-confirm')

// `Ctrl`+click *is* the marking gesture, so it is performed as the user
// performs it rather than by reaching into the composable.
async function markRow(page, path) {
  await page
    .locator(`.tree-row[data-path="${path}"]`)
    .first()
    .click({ modifiers: ['Control'] })
}

// "Before any request goes out" needs a counter, not a hope: the only honest
// form of the claim that a client-side refusal costs no round trip.
function countMoveRequests(page) {
  let count = 0
  page.on('request', (req) => {
    if (req.url().includes('/api/fs/move-batch')) count += 1
  })
  return () => count
}

test('a marked batch moves to another folder — rows, disk and a dirty buffer all follow', async ({
  page,
  request,
}) => {
  const a = `${NESTED}/a.any`
  const b = `${NESTED}/b.any`
  const c = `${NESTED}/c.any`
  await seedScratchTree(request)

  await openEditor(page)
  await openFolders(page, SCRATCH, NESTED, OTHER)

  // The marks, and their being *visible*: two rows in, one row out, each answer
  // read three ways — the row style, `aria-selected` (a multi-selectable tree
  // reports its selected set) and the navigator's count line.
  await markRow(page, a)
  await markRow(page, b)
  for (const path of [a, b]) {
    const row = page.locator(`.tree-row[data-path="${path}"]`).first()
    await expect(row).toHaveClass(/tree-row--marked/)
    await expect(row).toHaveAttribute('aria-selected', 'true')
  }
  const unmarked = page.locator(`.tree-row[data-path="${c}"]`).first()
  await expect(unmarked).not.toHaveClass(/tree-row--marked/)
  await expect(unmarked).toHaveAttribute('aria-selected', 'false')
  await expect(page.locator('.editor-nav-selected')).toHaveText('2 selected')

  // A marked row of a same-parent set: the subject is the whole set.
  await useMenuItem(page, a, 'Move to…')
  await expect(moveDialog(page)).toBeVisible()
  await expect(moveDialog(page).locator('h2')).toHaveText('Move 2 files')
  await expect(moveDialog(page).locator('.editor-move-list p')).toHaveText([a, b])
  await expect(moveField(page)).toHaveValue(NESTED)

  await moveField(page).fill(OTHER)
  await moveButton(page).click()

  // The tree shows them there, and the server says they are there — the second
  // is the witness, the first is only what the user was shown.
  await expect(page.locator(`.tree-row[data-path="${OTHER}/a.any"]`)).toBeVisible()
  await expect(page.locator(`.tree-row[data-path="${OTHER}/b.any"]`)).toBeVisible()
  await expect(page.locator(`.tree-row[data-path="${OTHER}/c.any"]`)).toHaveCount(0)
  await expect(page.locator(`.tree-row[data-path="${a}"]`)).toHaveCount(0)
  expect(await diskExists(request, a)).toBe(false)
  expect(await diskExists(request, b)).toBe(false)
  expect(await diskExists(request, `${OTHER}/a.any`)).toBe(true)
  expect(await diskExists(request, `${OTHER}/b.any`)).toBe(true)
  expect(await diskExists(request, c)).toBe(true)
  await expect(page.locator('.editor-notice--error')).toHaveCount(0)
  // **One** notice for the whole batch, naming the count and the destination.
  await expect(page.locator('.editor-notice--ok').first()).toContainText('Moved 2 files')
  await expect(page.locator('.editor-notice--ok').first()).toContainText(OTHER)

  // The rename test's proof at batch scale. Both rows are `Ctrl`+clicked, which
  // marks them *and* selects them, so the marks and the buffers arrive in one
  // gesture: both files are open in the strip, and the last row clicked is the
  // active one — that is what is typed into. The pair then moves back with the
  // edit still unwritten.
  await markRow(page, `${OTHER}/a.any`)
  await markRow(page, `${OTHER}/b.any`)
  await typeAtEnd(page, ' edited here')
  await expect(page.locator('.editor-tab-dirty')).toHaveCount(1)

  await useMenuItem(page, `${OTHER}/a.any`, 'Move to…')
  // Prefilled with the batch's own parent, spelled the way the tree spells it.
  await expect(moveField(page)).toHaveValue(OTHER)
  await moveField(page).fill(NESTED)
  await moveButton(page).click()

  await expect(page.locator('.editor-tab-dirty')).toHaveCount(1)
  // **Both** chips name the new paths: `repointPath` runs for every entry of the
  // batch, before any re-list. Waiting for this is waiting for the batch's own
  // reconciliation rather than for a pause — a Ctrl+S sent while the modal's
  // round trip is still running would post the pre-move path and come back
  // refused, which is the race the ordering of the flow exists to close.
  await expect(page.locator('.editor-tab', { hasText: 'a.any' })).toHaveAttribute(
    'title',
    `${NESTED}/a.any`,
  )
  await expect(page.locator('.editor-tab.active')).toHaveAttribute('title', b)
  // The buffer was re-pointed, not orphaned: the records move before any
  // re-list, so the freshness probe never sees a path that merely *left*.
  await expect(page.locator('.editor-tab-stale')).toHaveCount(0)
  await expect(page.locator(`.tree-row[data-path="${b}"]`).first()).toBeVisible()
  await page.keyboard.press('Control+s')
  await expect(page.locator('.editor-tab-dirty')).toHaveCount(0)
  await expect(page.locator('.editor-notice--error')).toHaveCount(0)
  expect(await diskText(request, `${OTHER}/b.any`)).toBe(null)
  expect(await diskText(request, b)).toBe('contents of b.any edited here')
})

test('a single right-click with no marks is the N=1 case, prefilled with the file’s own folder', async ({
  page,
  request,
}) => {
  const a = `${NESTED}/a.any`
  await seedScratchTree(request)

  await openEditor(page)
  await openFolders(page, SCRATCH, NESTED, OTHER)
  // A buffer first, so the same tab can be checked at the other end: the batch
  // re-points the records that name a moved path.
  await page.locator(`.tree-row[data-path="${a}"]`).first().click()
  await expect(page.locator('.editor-tab')).toHaveCount(1)

  await useMenuItem(page, a, 'Move to…')
  const prefill = await moveField(page).inputValue()
  // Exactly the tree's own spelling: not the bare `nested`, and not the
  // absolute working location (every path this app holds is relative).
  expect(prefill).toBe(NESTED)
  const { path: root } = await (await request.get('/api/path-location')).json()
  expect(root).toBeTruthy()
  expect(prefill).not.toContain(root)

  // The prefill is also the no-op case, and it explains itself: the sentence is
  // showing and the button is disabled, so the edit *is* the enable.
  await expect(moveError(page)).toHaveText('That is where they already are')
  await expect(moveButton(page)).toBeDisabled()

  await moveField(page).fill(OTHER)
  await expect(moveButton(page)).toBeEnabled()
  await expect(moveDialog(page).locator('h2')).toHaveText('Move 1 file')
  await moveButton(page).click()

  await expect(page.locator(`.tree-row[data-path="${OTHER}/a.any"]`)).toBeVisible()
  await expect(page.locator(`.tree-row[data-path="${a}"]`)).toHaveCount(0)
  expect(await diskExists(request, a)).toBe(false)
  expect(await diskExists(request, `${OTHER}/a.any`)).toBe(true)
  // The tab is the same tab, re-pointed: a chip's `title` is the record's path.
  await expect(page.locator('.editor-tab')).toHaveCount(1)
  await expect(page.locator('.editor-tab-label')).toHaveText('a.any')
  await expect(page.locator('.editor-tab')).toHaveAttribute('title', `${OTHER}/a.any`)
})

test('a refusal leaves the dialog open with what was typed, in the server’s words, and nothing moved', async ({
  page,
  request,
}) => {
  const a = `${NESTED}/a.any`
  const b = `${NESTED}/b.any`
  await seedScratchTree(request)

  await openEditor(page)
  await openFolders(page, SCRATCH, NESTED, OTHER)
  await markRow(page, a)
  await markRow(page, b)
  await useMenuItem(page, a, 'Move to…')
  await expect(moveDialog(page)).toBeVisible()

  const missing = `${SCRATCH}/nope`
  await moveField(page).fill(missing)
  await moveButton(page).click()

  // Still open, still holding the typed value: the fix is an edit, not a restart.
  await expect(moveDialog(page)).toBeVisible()
  await expect(moveField(page)).toHaveValue(missing)
  // The server's own sentence, naming the folder it looked for — the promise the
  // hint under the field made before anything was sent.
  await expect(moveError(page)).toContainText('Parent folder does not exist')
  await expect(moveError(page)).toContainText(missing)
  await expect(moveDialog(page).locator('.editor-move-hint')).toContainText(
    'The folder must already exist',
  )

  // Every source is still at home, on disk: one request, nothing applied.
  expect(await diskExists(request, a)).toBe(true)
  expect(await diskExists(request, b)).toBe(true)
  expect(await diskExists(request, `${OTHER}/a.any`)).toBe(false)
  await expect(page.locator('.editor-notice--error').first()).toContainText(
    `Could not move 2 files → ${missing}`,
  )
  // The marks survive the refusal too — the batch is still a batch, so the fix
  // is a new destination and not a fresh selection.
  await expect(page.locator(`.tree-row[data-path="${a}"]`).first()).toHaveClass(
    /tree-row--marked/,
  )
  await page.keyboard.press('Escape')
  await expect(moveDialog(page)).toBeHidden()
})

test('a destination the client can reject costs no request; a bare name reaches the server', async ({
  page,
  request,
}) => {
  const a = `${NESTED}/a.any`
  const b = `${NESTED}/b.any`
  await seedScratchTree(request)
  const moveRequests = countMoveRequests(page)

  await openEditor(page)
  await openFolders(page, SCRATCH, NESTED, OTHER)
  await markRow(page, a)
  await markRow(page, b)
  await useMenuItem(page, a, 'Move to…')

  // A leading slash is refused by the field's own shape check, in the client's
  // words — and the counter is what says *no request left the browser*.
  await moveField(page).fill('/tmp/outside')
  await expect(moveError(page)).toContainText('relative to the working location')
  await expect(moveButton(page)).toBeDisabled()
  // Enter is the submit gesture too, so it has to be refused by the same check:
  // the field answers, the dialog stays, and the counter has still seen nothing.
  await moveField(page).press('Enter')
  await expect(moveDialog(page)).toBeVisible()
  expect(moveRequests()).toBe(0)

  // A bare segment does not mean "relative to the sources' parent": it resolves
  // from the workspace root like every other path here, so it reaches the
  // server and comes back naming the folder that was looked for.
  await moveField(page).fill('nested')
  await expect(moveButton(page)).toBeEnabled()
  await moveButton(page).click()
  await expect(moveError(page)).toContainText('Parent folder does not exist')
  await expect(moveError(page)).toContainText('nested')
  expect(moveRequests()).toBe(1)

  await expect(moveDialog(page)).toBeVisible()
  expect(await diskExists(request, a)).toBe(true)
  expect(await diskExists(request, b)).toBe(true)
})

test('marks across parents narrow to the clicked row — in the dialog, before anything is sent', async ({
  page,
  request,
}) => {
  const markedFile = `${NESTED}/c.any`
  const otherFile = `${OTHER}/land.any`
  await seedScratchTree(request)
  await seedFile(request, otherFile, 'stays where it is')

  await openEditor(page)
  await openFolders(page, SCRATCH, NESTED, OTHER)

  // Marks across parents are legal, and the tree highlights both rows.
  await markRow(page, markedFile)
  await markRow(page, otherFile)
  await expect(page.locator('.editor-nav-selected')).toHaveText('2 selected')

  // The batch is one file, and the dialog says so twice — the title and the
  // list. That visible narrowing is what makes the gesture honest: the marks
  // stay highlighted, the list is the truth.
  await useMenuItem(page, markedFile, 'Move to…')
  await expect(moveDialog(page).locator('h2')).toHaveText('Move 1 file')
  await expect(moveDialog(page).locator('.editor-move-list p')).toHaveText([markedFile])
  await expect(moveField(page)).toHaveValue(NESTED)

  await moveField(page).fill(OTHER)
  await moveButton(page).click()

  await expect(page.locator(`.tree-row[data-path="${OTHER}/c.any"]`)).toBeVisible()
  expect(await diskExists(request, markedFile)).toBe(false)
  expect(await diskExists(request, `${OTHER}/c.any`)).toBe(true)
  // The other mark did **not** move: it was never in the batch.
  expect(await diskExists(request, otherFile)).toBe(true)
  expect(await diskText(request, otherFile)).toBe('stays where it is')
  expect(await diskExists(request, `${NESTED}/a.any`)).toBe(true)
})

// The markdown preview as the Editor performs it: the tree's Preview item opens
// a *rendered* panel in the same strip — no Monaco surface — and a plain Open of
// the same row afterwards gives a second chip that has one. Both are the pane's
// own surfaces; no LLM call is involved.
test('Preview renders a .md row in its own tab, beside the buffer', async ({ page, request }) => {
  const file = `${SCRATCH}/preview.md`
  expect((await create(request, file)).ok()).toBe(true)
  const seeded = await request.put('/api/fs/file', {
    data: {
      path: file,
      text: [
        '# Rendered heading',
        '',
        '| column | note |',
        '| --- | --- |',
        '| a | b |',
        '',
        '```python',
        'def greet(name):',
        '    return f"hi {name}"',
        '```',
        '',
      ].join('\n'),
      ifUnmodifiedSince: (await diskStat(request, file)).mtime,
    },
  })
  expect(seeded.ok(), await seeded.text()).toBe(true)

  await openEditor(page)
  await openFolders(page, SCRATCH)
  await useMenuItem(page, file, 'Preview')

  // One chip, named for what it is, and the panel renders through the app's
  // markdown path: an h1 and a real table, inside the tabpanel the chips point at.
  await expect(page.locator('.editor-tab-label')).toHaveText('preview.md (preview)')
  // `#editor-pane` *is* the preview panel here — exactly one element carries the
  // id at a time, which is what makes the chips' aria-controls honest.
  const panel = page.locator('#editor-pane.editor-preview')
  await expect(panel).toHaveCount(1)
  await expect(panel.locator('h1')).toContainText('Rendered heading')
  await expect(panel.locator('table')).toHaveCount(1)
  await expect(panel.locator('pre code .hljs-keyword').first()).toBeVisible()
  // The honesty line about the snapshot the pane is showing.
  await expect(page.locator('.editor-raw-note').first()).toContainText(
    'rendered from the copy read when this tab opened',
  )
  // No Monaco surface at all for a preview.
  await expect(page.locator('.monaco-host')).toHaveCount(0)

  // Opening the same row normally gives a second chip with the buffer, and the
  // preview's own panel gives way to Monaco — exactly one #editor-pane on screen.
  await page.locator(`.tree-row[data-path="${file}"]`).first().click()
  await expect(page.locator('.editor-tab-label')).toHaveText([
    'preview.md (preview)',
    'preview.md',
  ])
  await expect(page.locator('.monaco-host .view-lines').first()).toBeVisible()
  await expect(page.locator('.editor-preview')).toHaveCount(0)
  await expect(page.locator('#editor-pane.editor-monaco')).toHaveCount(1)

  // The preview chip is a tab like any other, so selecting it reveals its row —
  // a preview record carries a real `path`, which is all the reveal needs.
  await page.locator(`.tree-row[data-path="${SCRATCH}"]`).first().click()
  await expect(page.locator(`.tree-row[data-path="${file}"]`)).toHaveCount(0)
  await page.getByRole('tab', { name: 'preview.md (preview)' }).click()
  await expect(page.locator(`.tree-row[data-path="${file}"]`)).toBeVisible()
  // Revealing went through the tree as a *tab* echo, so it did not open a
  // buffer on top of the render: still exactly two chips.
  await expect(page.locator('.editor-tab')).toHaveCount(2)
  // The buffer is editable; the preview chip is a view, so switching back to it
  // makes the pane read-only rather than inviting an edit of a rendered document.
  await page.getByRole('tab', { name: 'preview.md', exact: true }).click()
  await expect(page.locator('.editor-mode-badge')).toHaveText('read-write')
  await page.getByRole('tab', { name: 'preview.md (preview)' }).click()
  await expect(page.locator('.editor-mode-badge')).toHaveText('read-only')
  await expect(page.locator('.editor-mode-badge')).toHaveAttribute(
    'title',
    /this tab shows the rendered Markdown of the file/,
  )
  await expect(page.locator('#editor-pane.editor-preview h1')).toBeVisible()

  // Dirty the *buffer*, then close the preview from its own chip: a view has
  // nothing to lose, so the gesture closes it with no prompt even while the
  // buffer of the same path holds unwritten work. (The dialog is always mounted,
  // so "no prompt" is `toBeHidden`, not "no count".)
  await page.getByRole('tab', { name: 'preview.md', exact: true }).click()
  await typeAtEnd(page, ' written nowhere')
  await expect(page.locator('.editor-tab-dirty')).toHaveCount(1)
  await page.getByRole('tab', { name: 'preview.md (preview)' }).click()
  await page.locator('.editor-tab').first().locator('.editor-tab-close').click()
  await expect(page.locator('.editor-close-dialog')).toBeHidden()
  await expect(page.locator('.editor-tab-label')).toHaveText('preview.md')
  await expect(page.locator('.editor-tab-dirty')).toHaveCount(1)
})

test('a refusal reaches the DOM in the server\'s own words', async ({ page, request }) => {
  await openEditor(page)
  await openFolders(page, SCRATCH)

  // Creating `.env` — the app's own file, hidden by a rule this app wrote. The
  // refusal names the **level** and the **pattern**, because "benchmarkit
  // regenerates that file from the settings template" is a different answer from
  // "your policy hides it", and only the second one can be edited away.
  await useMenuItem(page, SCRATCH, 'New file…')
  await page.locator('.tree-name-input').fill('.env')
  await page.keyboard.press('Enter')
  await expect(page.locator('.tree-name-input-error')).toContainText(
    "benchmarkit's own rules for this workspace",
  )
  await expect(page.locator('.tree-name-input-error')).toContainText('"**/.env**"')
  await expect(page.locator('.editor-notice--error').first()).toContainText(
    `Could not create ${SCRATCH}/.env`,
  )
  expect(await existsOnDisk(request, `${SCRATCH}/.env`)).toBe(false)

  // A move into a read-only folder: an innocent source, a locked target. The
  // input stays open holding the typed value, so the fix is to type again, and
  // nothing moved. The seeded file marks every `node_modules` read-only — the
  // destination does not even exist yet, and the policy answers before the
  // "parent is missing" 409 could.
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
  // A typed `/` makes it a move, resolved against the working location.
  await input.fill('web/node_modules/hand-edited.js')
  await page.keyboard.press('Enter')
  await expect(page.locator('.tree-row-error')).toContainText('workspace policy')
  await expect(page.locator('.tree-row-error')).toContainText('"**/node_modules"')
  await expect(page.locator('.tree-row-error')).toContainText('read-only')
  await expect(input).toBeVisible()
  await expect(input).toHaveValue('web/node_modules/hand-edited.js')
  expect(await diskExists(request, file)).toBe(true)
  expect(await existsOnDisk(request, 'web/node_modules/hand-edited.js')).toBe(false)
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
  await openFolders(page, SCRATCH, inner)
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
  await openFolders(page, SCRATCH)
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
  await openFolders(page, SCRATCH)
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

// The find widget's buttons are icons of Monaco's own font, `codicon`, which the
// face's own rule asks for — but the widget styles its buttons with a longer
// selector, and an inherited `font` shorthand resets the family, so they end up
// in the editor's UI font (the app's answer is the rule in monaco-pico.css). A
// button left in a font with no glyph for its private-use code point is not
// blank: the browser stamps the code point into a box, so it keeps looking like
// a button and stops saying what it does. Only a browser can see which family a
// mark actually drew with, which is why the whole check lives at this layer.
test('the find widget buttons show Monaco own icons', async ({ page }) => {
  const file = `${SCRATCH}/find.txt`
  expect((await create(page.request, file)).ok()).toBe(true)

  await openEditor(page)
  await openFolders(page, SCRATCH)
  await page.locator(`.tree-row[data-path="${file}"]`).first().click()
  await expect(page.locator('.monaco-host .view-lines').first()).toBeVisible()

  // Ctrl+F while the pane has focus — the gesture the report describes, not a
  // call into the editor's API.
  await page.locator('.monaco-host .view-lines').first().click()
  await page.keyboard.press('Control+f')
  const widget = page.locator('.monaco-editor .find-widget')
  await expect(widget).toBeVisible()

  const marks = widget.locator('.button.codicon')
  await expect(marks.first(), 'the widget is icon-driven').toBeVisible()

  // The mark's *resolved* family — the only question that matters and the only
  // one a stylesheet cannot answer: several rules claim these buttons and the
  // browser's own cascade decides, so the answer is read off the element.
  const families = await marks.evaluateAll((els) =>
    els.map((el) => getComputedStyle(el).fontFamily))
  expect(families.length, 'the widget has several marks').toBeGreaterThan(3)
  for (const [index, family] of families.entries()) {
    expect(family, `mark ${index} asks for the codicon face`).toContain('codicon')
  }
  // And the family is backed by a face that really loaded: a name with no font
  // behind it draws the same box. Any private-use code point of the face answers
  // for it, so this one is named rather than read off a mark.
  await expect
    .poll(() => page.evaluate(() => document.fonts.check('16px codicon', '\uea50')),
      'the codicon face is loaded')
    .toBe(true)
})
