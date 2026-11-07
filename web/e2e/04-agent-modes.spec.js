import { expect, test } from '@playwright/test'
import fs from 'node:fs'
import path from 'node:path'

// Task am-07 — level: UI state. The mode dropdown needs no LLM call and no run:
// read the workspace's own list from GET /api/agent-modes, and check that the
// composer offers it and that a mode whose flag is false turns the matching tool
// panel inert. As in 01-app-boot, every expected label is read from the endpoint
// rather than written here, so the spec follows the modes file it measures.
//
// Nothing here names the app-owned mode: the non-editable entry is found by its
// flag, which is the only way the client is allowed to recognise it (D9).
//
// What it measures therefore has to be a file the spec owns. That file is the
// user's — `.nojae/agent-modes.json`, edited outside the app — so its flags are
// settings, not fixtures: a persona switched to `"mcptools": true` leaves no
// mode to gate, and a spec reading the *boot* workspace fails on an app that is
// working correctly. So the spec moves the working location to a scratch folder
// it creates, where am-03's seeding left benchmarkit's own document, and reads
// the payload from there.
//
// The scratch is removed with node's `fs`, not `/api/fs/file`: switching into it
// seeds a `.nojae/` there, and that folder is reserved — the Editor will not
// delete a subtree that contains it. Same shape as 03-workspace-switch.
// The name avoids the file's own name: gitignore is held to ignoring the
// reserved *folder*, never a line per file in it (am-03's acceptance test).
const SCRATCH = 'e2e-modes-root'

// The root this server booted on, remembered so a spec that fails between its
// two switches cannot leave the whole suite pointing at a scratch folder.
let bootRoot = null
// Whether the working location is currently on the scratch root: the restore in
// `afterEach` only runs when it is, and a spec that died mid-way leaves this
// true so the next one cleans up after it.
let switched = false

async function currentRoot(request) {
  return (await (await request.get('/api/path-location')).json()).path
}

function removeScratch(root) {
  fs.rmSync(path.join(root, SCRATCH), { recursive: true, force: true })
}

test.beforeEach(async ({ request }) => {
  removeScratch(await currentRoot(request))
})

test.afterEach(async ({ request }) => {
  // Only a test that actually moved the location has to move it back — a switch
  // restarts both hubs, so restoring it needlessly is not free.
  if (bootRoot === null || !switched) return
  switched = false
  const res = await request.post('/api/path-location', { data: { path: bootRoot } })
  expect(res.ok(), `could not restore the working location: ${await res.text()}`).toBe(true)
  removeScratch(bootRoot)
})

// The scratch workspace on its own: create it and move the working location
// onto it. The switch is the app's own route — the one the Settings dialog
// uses — and it seeds the modes document as it goes.
async function switchToScratch(request) {
  bootRoot ??= await currentRoot(request)
  const created = await request.post('/api/fs/create', { data: { path: SCRATCH, dir: true } })
  expect(created.ok(), `could not create the scratch workspace: ${await created.text()}`).toBe(true)
  const moved = await request.post('/api/path-location', {
    data: { path: path.join(bootRoot, SCRATCH) },
  })
  expect(moved.ok(), `could not move the working location: ${await moved.text()}`).toBe(true)
  switched = true
  // The server canonicalises the root it accepted (realpath), so compare against
  // what it reports rather than what was sent.
  return (await moved.json()).path
}

// Switch to the scratch workspace and answer with the **dropdown's** payload.
// The editor's read is a second endpoint on purpose (D2: the two reads are
// never merged), so the editor spec below calls `switchToScratch` and reads
// `GET /api/agent-modes/document` itself.
async function seededModesPayload(request) {
  const scratchRoot = await switchToScratch(request)
  const res = await request.get('/api/agent-modes')
  expect(res.ok(), `GET /api/agent-modes should succeed, got ${res.status()}`).toBe(true)
  const payload = await res.json()
  // The payload came from the scratch root's seeded file and was understood as
  // this version understands it. That is what makes the flags below facts about
  // benchmarkit's own document rather than a coincidence of this machine.
  expect(payload.source.path).toBe(path.join(scratchRoot, '.nojae', 'agent-modes.json'))
  expect(payload.source.exists, 'the switch seeded the modes file').toBe(true)
  expect(payload.source.loaded, 'the seeded modes file loads cleanly').toBe(true)
  expect(payload.findings, 'the seeded modes file has no problems').toEqual([])
  return payload
}

// The modes file as the workspace holds it, read with node's `fs`: the folder is
// reserved, so no Editor route can see it — the same reason the scratch itself is
// removed with `fs` and not `/api/fs/file`. This is the spec's witness that a
// save wrote what the form showed, rather than a second reading of the route
// that wrote it.
function readModesFile(scratchRoot) {
  return fs.readFileSync(path.join(scratchRoot, '.nojae', 'agent-modes.json'), 'utf8')
}

// The dialog's tabs: `Common` plus one per entry the **file** declares, in the
// file's order. Scoped to the dialog's own nav — the app's tab strip renders
// buttons of its own, and a dialog is not a tab container to the a11y tree.
function modeTabs(page) {
  return page.locator('.agent-modes-nav .agent-modes-tab')
}

test.describe('Agent mode dropdown: the composer, the gate, the hint', () => {
  test('the seeded default is shown, and a mode with mcptools false gates the panel', async ({
    page,
    request,
  }) => {
    const modes = await seededModesPayload(request)
    expect(modes.modes.length, 'the seeded modes document offers modes').toBeGreaterThan(1)
    const defaultMode = modes.modes.find((mode) => mode.id === modes.default)
    expect(defaultMode, 'the default is one of the offered modes').toBeTruthy()
    // A mode that switches MCP off, and one that leaves it on: the pair the
    // gate is about. These hold of benchmarkit's own document, which is why
    // they are assertions here rather than assumptions about someone's file.
    const gated = modes.modes.find((mode) => mode.mcptools === false)
    const open = modes.modes.find((mode) => mode.mcptools === true)
    expect(gated, 'a seeded mode switches MCP off').toBeTruthy()
    expect(open, 'a seeded mode leaves MCP on').toBeTruthy()
    // The app-owned mode is last and the only non-editable one (D9) — the payload
    // the client must read instead of naming the id.
    expect(modes.modes.at(-1).editable, 'the last mode is the app-owned one').toBe(false)
    expect(modes.modes.slice(0, -1).every((mode) => mode.editable)).toBe(true)

    // The panel only exists when Settings declares connectors; without them
    // there is nothing to gate, and the spec says so instead of passing empty.
    const mcpRes = await page.request.get('/api/mcp')
    const mcpConnectors = (await mcpRes.json()).connectors.filter((c) => c.active)

    await page.goto('/')
    await page.getByRole('tab', { name: 'WorkBench' }).click()

    const select = page.getByLabel('Agent mode')
    await expect(select, 'the composer offers the mode dropdown').toBeVisible()
    await expect(select).toHaveValue(defaultMode.id)
    // The one sentence that explains the gating: the select's title names the
    // mode and the hubs it can reach.
    const hint = (await select.getAttribute('title')) ?? ''
    expect(hint).toContain(defaultMode.label)
    // …and it *names the tools*, not just the persona: what a mode can reach is
    // the part of the sentence the user acts on. The default is the mode that
    // reaches everything, so its hint says so. (am-08 §4.)
    expect(defaultMode.mcptools, 'the seeded default opens MCP').toBe(true)
    expect(hint).toContain('available')

    if (mcpConnectors.length) {
      await expect(page.locator('.chat-mcp-panel')).toBeVisible()
      await expect(page.locator('.chat-mcp-panel input[role="switch"]')).toHaveCount(
        mcpConnectors.length,
      )
    }

    await select.selectOption(gated.id)
    await expect(select).toHaveValue(gated.id)
    const gatedHint = (await select.getAttribute('title')) ?? ''
    expect(gatedHint).toContain(gated.label)
    // MCP off is the one clause the gated mode cannot be missing, whatever its
    // LSP flag says: the hint is what tells the user the panel went dark by
    // itself and not because Settings did.
    expect(gatedHint).not.toContain('MCP available')

    if (mcpConnectors.length) {
      // Disabled, never hidden: the toggles stay real (Settings owns them), and
      // the panel says which mode took the hub away. Scoped to the MCP panel —
      // the LSP panel renders its own line for its own flag.
      const gate = page.locator('.chat-mcp-panel .chat-hub-gate')
      await expect(gate).toBeVisible()
      await expect(gate).toContainText(gated.label)
      // Every switch inert, and still the same number of them.
      await expect(
        page.locator('.chat-mcp-panel input[role="switch"]:disabled'),
      ).toHaveCount(mcpConnectors.length)
    }

    await select.selectOption(open.id)
    if (mcpConnectors.length) {
      await expect(page.locator('.chat-mcp-panel .chat-hub-gate')).toHaveCount(0)
      await expect(page.locator('.chat-mcp-panel input[role="switch"]').first()).toBeEnabled()
    }
  })

  // The composer row is `auto minmax(0,1fr) auto`: the bullets stay centered
  // between the two button groups. The dropdown joins the *left* group, so this
  // pins that the row still lays out that way and the select keeps the
  // icon-button height it sits beside.
  test('the dropdown joins the left action group without moving the bullets', async ({
    page,
  }) => {
    await page.goto('/')
    await page.getByRole('tab', { name: 'WorkBench' }).click()

    const select = page.getByLabel('Agent mode')
    await expect(select).toBeVisible()
    const box = await select.boundingBox()
    const button = await page.locator('.chat-bullet-actions--left .icon-btn').first().boundingBox()
    const bullets = await page.locator('.chat-bullet-row .memory-bullets').boundingBox()
    const rightGroup = await page
      .locator('.chat-bullet-actions--right')
      .boundingBox()

    expect(Math.round(box.height)).toBe(Math.round(button.height))
    // Left of the bullets, which are still in the middle of the row.
    expect(box.x + box.width).toBeLessThanOrEqual(bullets.x + 1)
    expect(bullets.x + bullets.width).toBeLessThanOrEqual(rightGroup.x + 1)

    // The modes pencil (eds-07) joined that same row without moving anything:
    // it sits immediately right of the select, on the same line, at the same
    // height, and the select's right edge is still left of the bullets.
    const pencil = page.locator('.chat-mode-edit')
    await expect(pencil).toBeVisible()
    const pencilBox = await pencil.boundingBox()
    expect(Math.round(pencilBox.height)).toBe(Math.round(box.height))
    expect(pencilBox.x).toBeGreaterThanOrEqual(box.x + box.width)
    // Same line: the row centres both, and both are the icon-button height, so
    // their top edges coincide (rounded — the boxes are fractional).
    expect(Math.round(pencilBox.y)).toBe(Math.round(box.y))
    expect(pencilBox.x + pencilBox.width).toBeLessThanOrEqual(bullets.x + 1)
  })

  // A modes list that cannot be fetched must cost the user nothing: no control
  // that cannot be used, and the chat request keeps the shape it had before the
  // feature (no `agentMode` key), which is what lets the server apply its own
  // default. Driving it needs the endpoint to fail, so the route is stubbed —
  // and no LLM call is made either way.
  test('a failing /api/agent-modes leaves the composer as it was', async ({ page }) => {
    await page.route('/api/agent-modes', (route) =>
      route.fulfill({ status: 500, contentType: 'application/json', body: '{}' }),
    )
    await page.goto('/')
    await page.getByRole('tab', { name: 'WorkBench' }).click()

    await expect(page.getByLabel('Agent mode')).toHaveCount(0)
    // Everything else about the composer is still there.
    await expect(page.getByLabel('Send message')).toBeVisible()
    await expect(page.locator('.chat-textarea')).toBeVisible()

    // …including the modes editor, which is *not* gated on the dropdown's
    // verdict (D10): the card reads the file through its own route, so a list
    // that could not be loaded — a broken modes file — is exactly when it is
    // most useful. The stub above matches only the list URL, so this pins the
    // affordance itself (present and enabled); what the card does once open is
    // the dialog suite's business, and no click happens here.
    const pencil = page.locator('.chat-mode-edit')
    await expect(pencil).toBeVisible()
    await expect(pencil).toBeEnabled()

    // The hub panels are not gated by a mode nobody resolved.
    const mcpRes = await page.request.get('/api/mcp')
    const mcpConnectors = (await mcpRes.json()).connectors.filter((c) => c.active)
    if (mcpConnectors.length) {
      await expect(page.locator('.chat-mcp-panel input[role="switch"]').first()).toBeEnabled()
      await expect(page.locator('.chat-hub-gate')).toHaveCount(0)
    }
  })
})

// Task eds-08 — level: the modes editor, end to end. The pencil beside the
// dropdown opens the card that edits the workspace's modes **file**; this spec
// follows one edit all the way to the bytes on disk and back to the dropdown.
//
// It measures three things the spec file asks for, and nothing else:
//
//  1. the pencil sits beside the dropdown and opens the dialog, whose tabs are
//     `Common` plus one per mode the **file** declares, in the file's order —
//     and the app-owned mode is nowhere in it;
//  2. **Create agent** adds a tab, focuses it, and writes nothing: Cancel
//     leaves the file byte-identical;
//  3. Save writes the document, the dropdown reloads with the new mode
//     selectable and selected as the new default, and the file holds what the
//     form showed — a persona that round-trips through the loader with no
//     findings.
//
// What it deliberately does **not** measure: the pencil being inert while a
// reply streams or a lot runs. Reaching either state needs a model, and this
// suite makes no LLM call here; that rule belongs to the component tests.
//
// Nothing here names the app-owned mode either: its id and its label are read
// out of `GET /api/agent-modes` as the **non-editable** entry's data, and used
// only to assert the dialog does not contain them.
const NEW_LABEL = 'E2e Reviewer'
// The extra persona line this spec adds to a persona the file already holds, so
// the round-trip is a *change* the form made and not a copy of what was there.
const PERSONA_TAIL = 'Review the diff before it lands.'

// Which section of the card is showing, and what holds the focus: the dialog's
// own answer to "did Create put me on the new mode's tab, in its first field?".
// Read through `dataset` because an entry key is arbitrary text.
function focusProbe(page) {
  return page.evaluate(() => {
    const el = document.activeElement
    return {
      tag: el?.tagName ?? null,
      tab: el?.closest('section[data-tab]')?.dataset.tab ?? null,
    }
  })
}

test.describe('Agent modes editor: the pencil, the tabs, the save', () => {
  test('the dialog edits the file it reads, and the disk agrees', async ({ page, request }) => {
    // Every wait in here is for an app state or a response, never a sleep.
    const waitForRoute = (pathname) =>
      page.waitForResponse((response) => new URL(response.url()).pathname === pathname)
    const scratchRoot = await switchToScratch(request)

    // The editor's own read — the second endpoint, and the file's own verdict
    // (D2). Everything the assertions below expect is read from it: the tab
    // captions, the order, the caps, the readme, the persona that round-trips.
    const docRes = await request.get('/api/agent-modes/document')
    expect(docRes.ok(), `document read failed: ${await docRes.text()}`).toBe(true)
    const doc = await docRes.json()
    expect(doc.source.exists, 'the switch seeded the modes file').toBe(true)
    expect(doc.source.loaded, 'the seeded modes file loads cleanly').toBe(true)
    expect(doc.findings, 'the seeded modes file has no problems').toEqual([])
    expect(doc.entries.length, 'the seeded document declares modes').toBeGreaterThan(1)
    const captions = ['Common'].concat(
      doc.entries.map(
        (entry) =>
          (typeof entry.fields?.label === 'string' && entry.fields.label.trim()) || entry.key,
      ),
    )

    await page.goto('/')
    await page.getByRole('tab', { name: 'WorkBench' }).click()

    // 1. The pencil sits beside the dropdown, and opens the card.
    const pencil = page.locator('.chat-mode-edit')
    await expect(pencil, 'the pencil is beside the mode dropdown').toBeVisible()
    const select = page.getByLabel('Agent mode')
    // The select renders only once the list loaded, so its box is worth taking
    // after it is there rather than before.
    await expect(select).toBeVisible()
    const selectBox = await select.boundingBox()
    const pencilBox = await pencil.boundingBox()
    expect(pencilBox.x).toBeGreaterThanOrEqual(selectBox.x + selectBox.width)

    // Opening the card *is* the document read — the card fetches on mount, so
    // the read is what opening waits for.
    const openModesDialog = async () => {
      const read = waitForRoute('/api/agent-modes/document')
      await pencil.click()
      await read
    }
    await openModesDialog()
    const dialog = page.locator('.agent-modes-dialog')
    await expect(dialog).toBeVisible()

    // One tab per mode the file declares, in the file's order, plus Common —
    // counted against the **document**, not against the dropdown's list, which
    // is a different set (the built-in fallback and the app-owned mode).
    const tabs = modeTabs(page)
    await expect(tabs).toHaveCount(captions.length)
    for (let i = 0; i < captions.length; i += 1) {
      await expect(tabs.nth(i), `tab ${i} is the file's ${i}th entry`).toHaveText(captions[i])
    }

    // The app-owned mode appears nowhere in the card. It is found here by its
    // flag, and both of its identifiers are read from the server: the dialog
    // holds neither, and the count above is the same claim from the other side.
    const list = await (await request.get('/api/agent-modes')).json()
    const appOwned = list.modes.filter((mode) => mode.editable === false)
    expect(appOwned.length, 'exactly one mode is app-owned').toBe(1)
    expect(
      list.modes.length,
      'the dropdown offers the file entries plus the app-owned one',
    ).toBe(doc.entries.length + 1)
    // `textContent`, not `innerText`: the tabs other than the active one are
    // `v-show`n away and still part of the card, so a check that only read the
    // rendered text would pass on a dialog that hid the reserved entry instead
    // of never containing it.
    const dialogText = await dialog.evaluate((el) => el.textContent ?? '')
    expect(dialogText).not.toContain(appOwned[0].id)
    expect(dialogText).not.toContain(appOwned[0].label)

    // 2. Create agent: a tab appears, is focused, and nothing is written.
    const bytesBefore = readModesFile(scratchRoot)
    await dialog.locator('.agent-modes-create').getByLabel('New agent label').fill(NEW_LABEL)
    await dialog.locator('.agent-modes-create-button').click()

    await expect(tabs).toHaveCount(captions.length + 1)
    await expect(tabs.last()).toHaveText(NEW_LABEL)
    // The card is *on* the new tab: its section is the visible one, and its
    // heading is the caption the tab just took.
    const section = page.locator('.agent-modes-editor section:visible')
    await expect(section.locator('h3')).toHaveText(NEW_LABEL)
    // Focus landed in that section's first field — the new mode's Label, not the
    // first mode's, and *after* the tab switched (`v-show` keeps every section in
    // the DOM, so the section the focused field belongs to is what makes the
    // focus mean anything). Polled, because the card focuses on its next tick.

    // The id is the slug the composable derived from the label — read from the
    // form, judged against the caps the payload published, never spelled here.
    const idField = section.getByLabel('Id', { exact: true })
    await expect(idField, "a new mode's id is editable while it is new").toBeEditable()
    const newId = await idField.inputValue()
    expect(newId).toMatch(new RegExp(doc.caps.idPattern))
    expect(doc.entries.map((entry) => entry.key)).not.toContain(newId)
    await expect
      .poll(async () => {
        const focused = await focusProbe(page)
        return `${focused.tag ?? 'none'}:${focused.tab ?? 'none'}`
      })
      .toBe(`INPUT:${newId}`)

    await dialog.getByRole('button', { name: 'Cancel' }).click()
    await expect(page.locator('.agent-modes-dialog')).toHaveCount(0)
    expect(readModesFile(scratchRoot), 'Cancel writes nothing').toBe(bytesBefore)

    // 3. The same edit, saved.
    await openModesDialog()
    await dialog.locator('.agent-modes-create').getByLabel('New agent label').fill(NEW_LABEL)
    await dialog.locator('.agent-modes-create-button').click()
    const savedSection = page.locator('.agent-modes-editor section:visible')
    expect(await savedSection.getByLabel('Id', { exact: true }).inputValue()).toBe(newId)

    // A persona built from one the file already holds, so the round-trip is
    // measured against text the loader is known to accept in list form.
    const seededPersona = doc.entries
      .map((entry) => entry.fields?.systemMessage)
      .find((value) => Array.isArray(value) && value.length > 1)
    expect(seededPersona, 'the seeded document holds a multi-line persona').toBeTruthy()
    const persona = [...seededPersona, PERSONA_TAIL].join('\n')
    await savedSection.locator('textarea').fill(persona)
    // The counter under the field is the payload's cap and the draft's bytes —
    // no cap is written in this file either.
    await expect(savedSection.locator('.agent-modes-persona-count')).toContainText(
      `${Buffer.byteLength(persona, 'utf8')} / ${doc.caps.maxPersonaBytes} bytes`,
    )

    // The new mode becomes the default, on the Common tab.
    await tabs.first().click()
    await dialog.getByLabel('Default mode').selectOption(newId)

    const saved = waitForRoute('/api/agent-modes/save')
    await dialog.locator('.agent-modes-save').click()
    const savedResponse = await saved
    expect(savedResponse.status(), 'the save was accepted').toBe(200)
    await expect(page.locator('.agent-modes-dialog')).toHaveCount(0)

    // The file holds what the form showed: the entries the file had, in order,
    // the new one appended, the new default, and the readme the file already
    // had (D6 — the writer preserves it rather than re-rendering it).
    const written = JSON.parse(readModesFile(scratchRoot))
    expect(Object.keys(written.agents)).toEqual([...doc.entries.map((entry) => entry.key), newId])
    expect(written.default).toBe(newId)
    expect(written.readme).toEqual(doc.readme)
    const writtenEntry = written.agents[newId]
    expect(writtenEntry.label).toBe(NEW_LABEL)
    expect(writtenEntry.mcptools).toBe(false)
    expect(writtenEntry.lsptools).toBe(false)
    // No blank line, so the writer chose the list form — the form the loader
    // reads back line for line, which is what makes the text round-trip.
    expect(writtenEntry.systemMessage).toEqual([...seededPersona, PERSONA_TAIL])
    expect(writtenEntry.systemMessage.join('\n')).toBe(persona)

    // The loader's verdict on the bytes the dialog wrote: understood, and clean.
    const after = await (await request.get('/api/agent-modes')).json()
    expect(after.source.loaded, 'the saved file loads').toBe(true)
    expect(after.findings, 'the saved file has no findings').toEqual([])
    const offered = after.modes.find((mode) => mode.id === newId)
    expect(offered, 'the new mode is offered').toBeTruthy()
    expect(offered.editable, 'the new mode is an editable one').toBe(true)
    expect(offered.label).toBe(NEW_LABEL)
    expect(after.default).toBe(newId)

    // The dropdown reloaded from the save response: the new mode is an option,
    // and it is the selection.
    // By role, not by an `option[value=…]` selector: Vue binds an option's value
    // as a DOM *property*, so the attribute is not what carries it.
    await expect(select.getByRole('option', { name: NEW_LABEL })).toHaveCount(1)
    await expect(select, 'the selection moved to the newly set default').toHaveValue(newId)
    // …and it can be chosen, i.e. it is a real option and not a stray value.
    await select.selectOption(after.modes[0].id)
    await select.selectOption(newId)
    await expect(select).toHaveValue(newId)
  })
})
