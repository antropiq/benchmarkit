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

// Switch the server onto a scratch workspace that it seeded itself, and answer
// with that workspace's payload. The switch is the app's own route — the one the
// Settings dialog uses — and it seeds the modes document as it goes.
async function seededModesPayload(request) {
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
  const scratchRoot = (await moved.json()).path

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

    // The hub panels are not gated by a mode nobody resolved.
    const mcpRes = await page.request.get('/api/mcp')
    const mcpConnectors = (await mcpRes.json()).connectors.filter((c) => c.active)
    if (mcpConnectors.length) {
      await expect(page.locator('.chat-mcp-panel input[role="switch"]').first()).toBeEnabled()
      await expect(page.locator('.chat-hub-gate')).toHaveCount(0)
    }
  })
})
