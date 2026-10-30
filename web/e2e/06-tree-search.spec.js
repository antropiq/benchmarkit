import { expect, test } from '@playwright/test'

// Task search-09 — the workspace search as a human performs it, in a real
// browser. This is the only layer that can see the two things the jsdom suite
// cannot: a caret that *lands on a line* (jsdom has no Monaco viewport, so the
// reveal is proved here and nowhere else) and a modal that survives a re-open
// (`<dialog>.showModal()` does not exist in jsdom — the unit suite patches it).
//
// Like `02-editor-write.spec.js`, this is one of the cheapest proofs in the
// repo: **no LLM call is involved**. The config comes from `.env.test`, no
// connector and no language server is started (`BENCHMARKIT_DISABLE_MCP/_LSP`
// in playwright.config.js), and the whole feature is one blocking GET plus a
// table of rows.
//
// The scratch is a **top-level** folder, and the files are seeded through the
// API (`POST /api/fs/create` + `PUT /api/fs/file`) rather than typed into the
// pane: byte-exact content with a known line number is what a caret assertion
// needs, and typing is only the point of a case where the *typing* is the
// feature (the invalid pattern below). It is not under `logs/` — benchmarkit's
// own rules hide that tree, which would make a search find nothing for the
// wrong reason. Each test trashes the scratch first, so a leftover from an
// interrupted run cannot turn this run's create into a 409.
const SCRATCH = 'e2e-search'
const NEEDLE = 'needle'

// A token no visible file holds, spelled in pieces so that *this* file never
// carries it contiguously — the sentinel trick `tests/test_editorfs_search.py`
// uses for the same reason: a spec that searches a word it also contains is
// searching itself. It lets one case assert an honest empty answer.
const ABSENT_NEEDLE = 'scratch' + 'probe' + 'line'

// The Contract's key set (spec §4's map), spelled out once against the live
// route: the caps' values, the route list and the column layout belong to the
// code, but "these are the keys the dialog reads" is a cross-layer contract.
const PAYLOAD_KEYS = [
  'filesScanned',
  'footerNote',
  'isFile',
  'matchCount',
  'matches',
  'scope',
  'skipped',
  'truncated',
]

async function trash(request, path) {
  return request.delete(`/api/fs/file?path=${encodeURIComponent(path)}`)
}

async function create(request, path, dir = false) {
  return request.post('/api/fs/create', { data: { path, dir } })
}

// A scratch folder: created through the same route the tree's own menu uses.
async function mkdir(request, path) {
  expect((await create(request, path, true)).ok(), `could not create ${path}`).toBe(true)
}

async function diskStat(request, path) {
  const res = await request.get(`/api/fs/stat?path=${encodeURIComponent(path)}`)
  return (await res.json()).paths[0]
}

// Byte-exact seed through the write route, with the mtime token the save guard
// demands — the way `02-editor-write.spec.js` seeds known text.
async function seed(request, path, text) {
  expect((await create(request, path)).ok(), `could not create ${path}`).toBe(true)
  const res = await request.put('/api/fs/file', {
    data: { path, text, ifUnmodifiedSince: (await diskStat(request, path)).mtime },
  })
  expect(res.ok(), `could not seed ${path}: ${await res.text()}`).toBe(true)
}

async function openEditor(page) {
  await page.goto('/')
  await page.getByRole('tab', { name: 'Editor' }).click()
  await expect(page.locator('.editor-nav-tree')).toBeVisible()
}

// The tree is lazy and a folder row's click *toggles* it, so opening a folder
// means "click it unless it is already open".
async function openFolder(page, path) {
  const row = page.locator(`.tree-row[data-path="${path}"]`).first()
  if ((await row.getAttribute('aria-expanded')) !== 'true') await row.click()
}

async function openFolders(page, ...paths) {
  for (const path of paths) await openFolder(page, path)
}

// Right-click a row and take its menu item by label. The right-click is also
// what makes that row the tree's subject, so the scope the dialog names is the
// row the gesture was made on — the entry point under test, not a shortcut.
async function useMenuItem(page, path, label) {
  await page.locator(`.tree-row[data-path="${path}"]`).first().click({ button: 'right' })
  await page.getByRole('menuitem', { name: label }).click()
}

// The dialog itself is *always mounted* (no `v-if`, so re-opening keeps the
// field's caret and the results' scroll position): "closed" means hidden, and
// that is what `toBeHidden` says — the same shape `02-editor-write.spec.js`
// uses for the close confirmation.
const dialog = (page) => page.locator('.editor-search-dialog')
const scopeLabel = (page) => page.locator('.editor-search-scope-path')
const scopeKind = (page) => page.locator('.editor-search-scope-kind')
const queryField = (page) => page.locator('.editor-search-query')
const resultRows = (page) => page.locator('.editor-search-row')
const editorTabs = (page) => page.locator('.editor-tab-label')

// A hit's own button, addressed by the `title` the row carries — which is also
// the assertion that the payload's *line* reached the affordance the user
// clicks, not just the table.
const hitButton = (page, path, line) =>
  page.locator(`.editor-search-hit[title="Open ${path} at line ${line}"]`)

// One rendered row of the gutter's line numbers, marked as the caret's line.
// See `expectCaretOnLine` for why this is the honest way to read a line number.
const activeLineNumberRows = (page) =>
  page.locator('.monaco-editor .margin-view-overlays .line-numbers.active-line-number')

// Type a query in the dialog and wait for *that* search to answer. The busy
// marker going away is the app's own statement that its request landed, so what
// is asserted next is the payload of the search that is over, never a guess
// about when it finished.
async function searchFor(page, query) {
  const answered = page.waitForResponse(
    (res) => res.url().includes('/api/fs/search') && res.request().method() === 'GET',
  )
  await queryField(page).fill(query)
  await queryField(page).press('Enter')
  await answered
  await expect(page.locator('.editor-search-busy')).toBeHidden()
}

// The line the caret is on, read from the rendered pane.
//
// **How a line number is read here without Monaco's own model.** The reveal ends
// with `editor.setPosition(...)`, and Monaco's line-number gutter is a rendered
// mirror of that same selection state: `LineNumbersOverlay` handles
// `onCursorStateChanged`, keeps its `_activeModelLineNumber` at the cursor's
// *model* line, and re-renders that one row with the extra class
// `active-line-number`. So a row of `.margin-view-overlays` carries that class
// if and only if the caret is on the model line the row's own text names. A
// scroll does not move the mark (the number belongs to the line, not to the
// viewport), and a caret on another line produces a different row.
//
// That is what makes it deterministic, and why nothing here compares pixels or
// screenshots: it is one DOM fact the app cannot get right by accident, and it
// checks a *number the gutter shows* against the line the result row promised.
//
// Exactly one gutter row carries the class at a time, so the array below has one
// element. Monaco's overlay starts with the caret on line 1, so an answer of `1`
// would prove nothing — which is why every line the reveal assertions below name
// is line 2 or later, and the only `1` asserted anywhere is the *baseline* caret
// position before a reveal.
async function expectCaretOnLine(page, line) {
  // Playwright retries the whole locator, so this waits for Monaco's repaint
  // instead of sleeping for it. `allInnerTexts` is compared as an array because
  // the gutter pads its numbers with spaces to right-align them.
  await expect(activeLineNumberRows(page)).toHaveText([String(line)])
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

// 1. folder → hit → caret. The scope comes from the row the gesture was made
// on, and the click lands the caret on the line the row named.
test('a folder search names its scope, and its hit lands the caret on the matching line', async ({
  page,
  request,
}) => {
  // The scope is the *inner* folder and the decoy sits one level above it, still
  // inside the scratch tree: with the same word in both files, one row is the
  // proof that what got searched was the inner subtree.
  const inner = `${SCRATCH}/inner`
  const notes = `${inner}/notes.md`
  const decoy = `${SCRATCH}/decoy.txt`
  await mkdir(request, inner)
  await seed(request, notes, '# Search e2e\n\nA preamble line.\nneedle here, on line 4.\n')
  await seed(request, decoy, 'a preamble\nneedle one level above the searched folder\n')

  await openEditor(page)
  await openFolders(page, SCRATCH, inner)
  await useMenuItem(page, inner, 'Search…')

  await expect(dialog(page)).toBeVisible()
  await expect(scopeLabel(page)).toHaveText(inner)
  await expect(scopeKind(page)).toHaveText('(folder)')

  await searchFor(page, NEEDLE)

  await expect(resultRows(page)).toHaveCount(1)
  await expect(hitButton(page, notes, 4)).toBeVisible()

  await hitButton(page, notes, 4).click()

  // The click closes the popup (D13), and the buffer is open as a tab.
  await expect(dialog(page)).toBeHidden()
  await expect(editorTabs(page)).toHaveText(['notes.md'])
  await expect(page.locator('.editor-tab-dirty')).toHaveCount(0)

  // The caret is on line 4 — the line the payload's row named, not merely a
  // view that happens to be scrolled there.
  await expectCaretOnLine(page, 4)
  // And the pane really holds the seeded text, so the assertion above cannot be
  // reading the decoy file or an empty buffer.
  await expect(page.locator('.monaco-editor .view-lines').first()).toContainText('line 4')
})

// 2. File scope: the row decides the scope, and the search covers one file.
test('a search raised on a file row covers that file and nothing else', async ({
  page,
  request,
}) => {
  const notes = `${SCRATCH}/notes.md`
  const decoy = `${SCRATCH}/decoy.txt`
  await seed(request, notes, 'line one\nneedle on line 2 of the scoped file\n')
  await seed(request, decoy, 'needle in a file the scope excludes\nneedle again here\n')

  await openEditor(page)
  await openFolders(page, SCRATCH)
  await useMenuItem(page, notes, 'Search…')

  await expect(scopeLabel(page)).toHaveText(notes)
  await expect(scopeKind(page)).toHaveText('(file)')

  await searchFor(page, NEEDLE)

  // Every row is that one path — the sibling with the same word twice is out.
  await expect(resultRows(page)).toHaveCount(1)
  await expect(page.locator('.editor-search-cell-path')).toHaveText([notes])
  await expect(page.locator('.editor-search-cell-line')).toHaveText(['2'])

  await hitButton(page, notes, 2).click()
  await expect(editorTabs(page)).toHaveText(['notes.md'])
  await expectCaretOnLine(page, 2)
})

// 3. Root scope: the root row is `""` to the route, so the label has to name the
// workspace rather than render an empty string.
test('a search raised on the root row names the workspace, and reaches a nested file', async ({
  page,
  request,
}) => {
  const nested = `${SCRATCH}/inner/deep.txt`
  const file = `${SCRATCH}/top.txt`
  // `create` refuses a missing parent (409, no `mkdir -p` side effect), so the
  // intermediate folder is made by its own call — an API call, which does not
  // expand the tree: the branch stays collapsed for the search below.
  await mkdir(request, `${SCRATCH}/inner`)
  await seed(request, nested, 'a preamble\nneedle two folders down\n')
  await seed(request, file, 'needle at the top level\n')
  // `rootLabel()` is the basename of the working location, read from the app
  // rather than hardcoded: the same route `03-workspace-switch.spec.js` asks.
  const rootName = (await (await request.get('/api/path-location')).json()).path
    .replace(/[/\\]+$/, '')
    .split('/')
    .pop()

  await openEditor(page)
  // The root row is the tree's own first row (`data-path=""`). Nothing below it
  // is expanded: the scan walks collapsed branches, so a nested file is found
  // without the user ever opening its folder.
  await useMenuItem(page, '', 'Search…')

  const label = scopeLabel(page)
  await expect(label).toHaveText(rootName)
  // The lie this case exists to catch: the root row rendering an empty label.
  await expect(label).not.toHaveText('')
  await expect(scopeKind(page)).toHaveText('(whole workspace)')

  await searchFor(page, NEEDLE)

  // Both seeded files are in the answer — one row each, at the line each was
  // seeded with. Asserted per path rather than as a row count: the scope is the
  // whole checkout, so this spec's own file (and every other visible one) is
  // searched too, and only the *absence* of a hidden path is a countable claim.
  await expect(hitButton(page, nested, 2)).toHaveCount(1)
  await expect(hitButton(page, file, 1)).toHaveCount(1)

  await hitButton(page, nested, 2).click()
  await expect(editorTabs(page)).toHaveText(['deep.txt'])
  await expectCaretOnLine(page, 2)
})

// 4. An invalid regex is `re`'s own message, inline — and it costs no scan and
// destroys no answer (D5 and D14 meeting).
test('an invalid pattern errors inside the dialog and keeps the previous answer', async ({
  page,
  request,
}) => {
  const notes = `${SCRATCH}/notes.md`
  await seed(request, notes, 'line one\nneedle on line 2\n')

  await openEditor(page)
  await openFolders(page, SCRATCH)
  await useMenuItem(page, SCRATCH, 'Search…')
  await searchFor(page, NEEDLE)
  await expect(resultRows(page)).toHaveCount(1)

  await page.locator('.editor-search-switches label.checkbox', { hasText: 'Regex' }).click()
  await queryField(page).fill('(')
  const refused = page.waitForResponse(
    (res) => res.url().includes('/api/fs/search') && res.status() === 400,
  )
  await queryField(page).press('Enter')
  const body = await (await refused).json()

  // The inline line under the field — carrying `re`'s message verbatim rather
  // than a generic sentence, and the same sentence the payload carries, so the
  // dialog is echoing the server rather than inventing wording.
  await expect(page.locator('.editor-search-error')).toBeVisible()
  await expect(page.locator('.editor-search-error')).toContainText(body.detail)
  await expect(page.locator('.editor-search-error')).toContainText('unterminated subpattern')

  // The table still holds the *previous* answer, after the error.
  await expect(resultRows(page)).toHaveCount(1)
  await expect(hitButton(page, notes, 2)).toBeVisible()

  // The same needle again is answered normally: the bad pattern cost nothing
  // but its own scan, and the error line goes away with the next success.
  await queryField(page).fill(NEEDLE)
  await searchFor(page, NEEDLE)
  await expect(resultRows(page)).toHaveCount(1)
  await expect(page.locator('.editor-search-error')).toHaveCount(0)

  // No notice-stack entry: this refusal lives beside the field that typed it,
  // not in the bottom-centre toast stack.
  await expect(page.locator('.editor-notice')).toHaveCount(0)
})

// 5. The retention rule, on the network too: closing the popup is not a search,
// and re-opening asks the server nothing.
test('re-opening the dialog shows the same answer and sends no second request', async ({
  page,
  request,
}) => {
  const notes = `${SCRATCH}/notes.md`
  await seed(request, notes, 'line one\nneedle on line 2\n')

  await openEditor(page)
  await openFolders(page, SCRATCH)

  // The honest form of "without searching": count the requests.
  let searches = 0
  page.on('request', (req) => {
    if (req.url().includes('/api/fs/search')) searches += 1
  })

  await useMenuItem(page, notes, 'Search…')
  await searchFor(page, NEEDLE)
  expect(searches, 'the search itself should have gone out').toBe(1)
  await expect(resultRows(page)).toHaveCount(1)

  await page.keyboard.press('Escape')
  await expect(dialog(page)).toBeHidden()

  // Re-open from the same row, touching nothing else.
  await useMenuItem(page, notes, 'Search…')
  await expect(dialog(page)).toBeVisible()
  await expect(resultRows(page)).toHaveCount(1)
  await expect(queryField(page)).toHaveValue(NEEDLE)
  await expect(scopeLabel(page)).toHaveText(notes)
  expect(searches, 'a re-open must not ask the server again').toBe(1)
})

// 6. The lazy-tree trap: a hit inside a branch the user never expanded. The
// naive form (select `entryFor(path)`) answers null for a path the tree has no
// listing of, and the click then silently opens nothing — the common case for a
// workspace-wide search.
test('a hit inside a collapsed branch still opens its file', async ({ page, request }) => {
  const deep = `${SCRATCH}/middle/deep.txt`
  // Both folders come from the API, so the tree has no listing of either: this
  // is the case where `entryFor(path)` answers null.
  await mkdir(request, `${SCRATCH}/middle`)
  await seed(request, deep, 'a preamble\nneedle buried two folders deep\n')

  await openEditor(page)
  // Deliberately *not* opening `SCRATCH`, let alone `middle`: the only listing
  // the tree holds is the root's first level.
  await useMenuItem(page, '', 'Search…')
  await searchFor(page, NEEDLE)

  // The row exists even though neither folder was ever listed to the tree.
  await expect(hitButton(page, deep, 2)).toHaveCount(1)

  await hitButton(page, deep, 2).click()
  await expect(editorTabs(page)).toHaveText(['deep.txt'])
  // And the tree followed the buffer open: `reveal` fetches the listings on the
  // way down, so the row the tab belongs to is on screen afterwards.
  await expect(page.locator(`.tree-row[data-path="${deep}"]`)).toBeVisible()
  await expectCaretOnLine(page, 2)
})

// 7. The reveal beats a buffer swap (D11): the pane keeps *one* Monaco surface
// across tab switches, and the `model.setValue` that swaps its content clears
// the cursor. A reveal applied before that watcher has run is erased — which is
// what the surface's own queue exists to prevent, and jsdom cannot see it.
test('the reveal survives the buffer swap that brings the hit file on screen', async ({
  page,
  request,
}) => {
  const first = `${SCRATCH}/first.txt`
  const hitFile = `${SCRATCH}/hit-target.md`
  // The hit is on line 4, and the file is the one that is *not* open when the
  // row is clicked. `first.txt` holds no needle, so the table has one row.
  await seed(request, hitFile, 'line one\nline two\nline three\nneedle on line 4\n')
  await seed(request, first, 'the buffer that is open when the hit is clicked\n')

  await openEditor(page)
  await openFolders(page, SCRATCH)
  await page.locator(`.tree-row[data-path="${first}"]`).first().click()
  await expect(editorTabs(page)).toHaveText(['first.txt'])
  // The caret sits on line 1 before anything is searched — the baseline that
  // makes the answer below mean more than "this buffer is fresh".
  await expect(activeLineNumberRows(page)).toHaveText(['1'])

  // The one Monaco instance is mounted the whole time (the pane keeps it across
  // tab switches), so every reveal below lands *after* a `model.setValue`.
  await expect(page.locator('.monaco-host')).toHaveCount(1)

  // First pass: the hit's file becomes a new tab while the reveal is in flight.
  await useMenuItem(page, SCRATCH, 'Search…')
  await searchFor(page, NEEDLE)
  await expect(hitButton(page, hitFile, 4)).toHaveCount(1)
  await hitButton(page, hitFile, 4).click()
  await expect(editorTabs(page)).toHaveText(['first.txt', 'hit-target.md'])
  await expect(page.locator('.monaco-host')).toHaveCount(1)
  await expect(page.locator('.monaco-editor .view-lines').first()).toContainText(
    'needle on line 4',
  )
  await expectCaretOnLine(page, 4)

  // Second pass — the case D11 is about: the hit's buffer is *already* a tab, so
  // the click switches to it through `setValue` on a mounted surface, and the
  // reveal is applied by that watcher rather than by the mount.
  await page.getByRole('tab', { name: 'first.txt' }).click()
  await expect(activeLineNumberRows(page)).toHaveText(['1'])

  // Re-opened from the same row: the retained table is clicked again.
  await useMenuItem(page, SCRATCH, 'Search…')
  await expect(resultRows(page)).toHaveCount(1)
  await hitButton(page, hitFile, 4).click()
  await expect(editorTabs(page)).toHaveText(['first.txt', 'hit-target.md'])
  await expect(page.locator('.monaco-editor .view-lines').first()).toContainText(
    'needle on line 4',
  )
  await expectCaretOnLine(page, 4)
})

// 8. The policy invariant, against the live route: a search sees exactly what
// the tree lists, and its payload says nothing about what it did not see.
//
// HTTP only, against a name **level A already hides** — this run's working
// location is this checkout, so `logs/**` (the app's own pre-save backups) is
// hidden *and already full of files*. Nothing is seeded and nothing is written
// there. There is no needle that only a backup could hold, so the assertion is
// on the **shape**: a broad needle that the visible tree certainly answers, and
// a payload that never mentions the hidden subtree — not as a row, not as a
// quoted snippet, and not as a count.
//
// Do not extend this test by writing `.nojae/read-write-policy.json`: that is the
// developer's real policy file for this checkout, and a spec that mutates it
// changes what their pane hides afterwards. Level-B rules belong to the backend
// suite, which owns a `tmp_path` (`tests/test_editorfs_search.py`).
test('a hidden path is absent from the payload, unquoted and uncounted', async ({
  page,
  request,
}) => {
  const root = await (await request.get('/api/fs/search?path=&q=e')).json()

  // The Contract's key set, on the live route where the real policy applies.
  expect(Object.keys(root).sort()).toEqual(PAYLOAD_KEYS)
  expect(root.scope).toBe('')
  expect(root.isFile).toBe(false)

  // The scan ran, over the visible tree.
  expect(root.filesScanned).toBeGreaterThan(0)
  expect(root.matchCount).toBeGreaterThan(0)
  expect(root.matches.length).toBeGreaterThan(0)

  // Nothing from the hidden tree, in the three ways it could leak: as a row, as
  // a quoted line, or as a *generated* backup name (`<digest>_<name>_<stamp>`
  // under the hidden folder). The last pattern is the shape of a path the app
  // writes there — a plain `logs/` substring would match visible code that only
  // *talks about* the folder, which is not a leak and would make this red for
  // the wrong reason.
  for (const hit of root.matches) {
    expect(hit.path.startsWith('logs/'), `hidden path in a row: ${hit.path}`).toBe(false)
    expect(
      hit.snippet,
      `a generated backup path is quoted: ${hit.snippet}`,
    ).not.toMatch(/[0-9a-f]{12}_[^/]*_\d{8}-\d{6}/)
  }

  // There is no count of policy-skipped files at all — an observation of a
  // hidden name is still indistinguishable from absence (D2).
  expect(Object.keys(root.skipped).sort()).toEqual(['binary', 'oversized'])
  expect('policy' in root.skipped).toBe(false)
  expect('skippedPolicy' in root).toBe(false)
  for (const hit of root.matches) {
    expect(Object.keys(hit).sort()).toEqual([
      'column',
      'line',
      'matchLength',
      'path',
      'snippet',
    ])
    expect(hit.line).toBeGreaterThan(0)
    expect(hit.column).toBeGreaterThan(0)
  }

  // A hidden scope answers 404 — the same answer a folder that is not there
  // gives, so neither the status nor the payload can be used to probe for a
  // hidden name.
  expect((await request.get('/api/fs/search?path=logs&q=e')).status()).toBe(404)
  expect((await request.get('/api/fs/search?path=not-a-real-folder&q=e')).status()).toBe(404)

  // And the tree agrees: the hidden folder is not a row, so the dialog could
  // never have been raised on it. What the UI does with a needle no visible file
  // holds is the honest empty answer, not an error.
  await openEditor(page)
  await useMenuItem(page, '', 'Search…')
  await searchFor(page, ABSENT_NEEDLE)
  await expect(resultRows(page)).toHaveCount(0)
  await expect(page.locator('.editor-search-empty')).toHaveText('no matches')
})

// The second entry point of §2.9: the header's icon raises the same dialog on
// the row the tree is *on*, so its subject is the tree's focus — a row can be
// the subject without ever having been clicked.
test('the header icon searches the focused row', async ({ page, request }) => {
  const other = `${SCRATCH}/other.txt`
  await seed(request, other, 'a preamble\nneedle in the file the focus is on\n')

  await openEditor(page)
  await openFolders(page, SCRATCH)
  // Focus with the keyboard only: `other.txt` becomes the tree's subject, and no
  // buffer of it is open.
  await page.locator(`.tree-row[data-path="${other}"]`).first().focus()
  await page.locator('.editor-nav-search').click()

  await expect(dialog(page)).toBeVisible()
  await expect(scopeLabel(page)).toHaveText(other)
  await expect(scopeKind(page)).toHaveText('(file)')

  await searchFor(page, NEEDLE)
  await expect(resultRows(page)).toHaveCount(1)
  await expect(page.locator('.editor-search-cell-path')).toHaveText([other])

  // The highlight of the snippet is the *match*, sliced out of the string the
  // server sent (D12): the mark is the needle, not the line.
  await expect(page.locator('.editor-search-mark')).toHaveText([NEEDLE])
  await expect(page.locator('.editor-search-snippet')).toHaveText([
    'needle in the file the focus is on',
  ])
})

