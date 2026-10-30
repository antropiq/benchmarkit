import { expect, test } from '@playwright/test'

// Task am-07 — level: UI state. The mode dropdown needs no LLM call and no run:
// boot, read the workspace's own list from GET /api/agent-modes, and check that
// the composer offers it and that a mode whose flag is false turns the matching
// tool panel inert. As in 01-app-boot, every expected label is read from the
// endpoint rather than written here, so the spec follows the seeded file.
//
// Nothing here names the app-owned mode: the non-editable entry is found by its
// flag, which is the only way the client is allowed to recognise it (D9).
async function modesPayload(page) {
  const res = await page.request.get('/api/agent-modes')
  expect(res.ok(), `GET /api/agent-modes should succeed, got ${res.status()}`).toBe(true)
  return res.json()
}

test.describe('Agent mode dropdown: the composer, the gate, the hint', () => {
  test('boot shows the workspace default, and Chat-like mode disables the MCP panel', async ({
    page,
  }) => {
    const modes = await modesPayload(page)
    expect(modes.modes.length, 'the seeded modes file offers modes').toBeGreaterThan(1)
    const defaultMode = modes.modes.find((mode) => mode.id === modes.default)
    expect(defaultMode, 'the default is one of the offered modes').toBeTruthy()
    // A mode that switches MCP off, and one that leaves it on: the pair the
    // gate is about.
    const gated = modes.modes.find((mode) => mode.mcptools === false)
    const open = modes.modes.find((mode) => mode.mcptools === true)
    expect(gated, 'some mode switches MCP off').toBeTruthy()
    expect(open, 'some mode leaves MCP on').toBeTruthy()

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
