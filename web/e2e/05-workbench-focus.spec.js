import { expect, test } from '@playwright/test'

// Task wf-04 — level: real layout. This is the only layer that can answer
// "is it out of the flow?": jsdom reports a hidden column as present in the
// DOM, so what wf-01…wf-03 pinned there is the state and its class names, and
// the geometry itself — `display: none`, the width and height a state hands
// the reasoning timeline — is decided here.
//
// No LLM call, no benchmark run, no streamed answer: the reasoning panel is
// rendered unconditionally, so every box below is measurable on a cold, empty
// WorkBench screen. A spec that waited for an answer would depend on
// `.env.test` and turn flaky for reasons that have nothing to do with layout.
//
// The controls' accessible names belong to the app and are never typed here:
// they change with the state (the focus control reads "show" until it is
// pressed, then "restore"), so `pressControl` asks the DOM for the name the
// control carries *now* and clicks that name back through `getByLabel` —
// D9's contract, exercised on every click rather than asserted once. Which of
// the two stacked controls is meant is positional, because the stack is the
// design (focus on top, hide below, spec §4); the first case below asserts
// both names exist and differ, so a renamed, duplicated or missing control
// fails there instead of silently pressing the wrong one here.

const SHELL = '.chat-shell'
const GUTTER = '.chat-col-gutter'
const LEFT_COL = '.chat-col--left'
const RIGHT_COL = '.chat-col--right'
const PANEL = '.chat-tool-panel'
const DOCK = '.chat-tools-dock'
const CONVERSATION = '.chat-output'

const TOP_CONTROL = 0
const LOWER_CONTROL = 1

// `useSplitter`'s own key for the dock's dragged height (D3). A storage key is
// not an accessible name: naming it here is how the spec can prove no state
// transition rewrote it.
const DOCK_SIZE_KEY = 'benchmarkit:chatToolsColumn'

// The tools dock carries a body — and its bar exists at all — only when the
// hub declaration files have something to show, and those files are
// gitignored, machine-local and not this spec's subject. So the two hub
// snapshots are answered from here: entries that are declared but stopped,
// which is all the layout needs (a dock to size, a bar to press) and nothing
// that touches an LLM or a process. Stubbed rather than assumed, so the case
// says which state of the app it measures.
async function declareOneHubEach(page) {
  const hub = (kind, entries) =>
    page.route(`/api/${kind}`, (route) =>
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          installed: true,
          [kind === 'mcp' ? 'connectors' : 'servers']: entries,
          maxIterations: 25,
          [kind === 'mcp' ? 'externalsError' : 'serversError']: null,
        }),
      }),
    )
  await hub('mcp', [{ id: 'probe-mcp', label: 'Probe MCP', active: true, status: 'stopped' }])
  await hub('lsp', [{ id: 'probe-lsp', label: 'Probe LSP', active: true, status: 'stopped' }])
}

// Every case runs against that dock, so the geometry is measured on the screen
// the feature was designed for: a two-panel column with an open dock below the
// timeline. Measuring a folded strip would make `reasoning` look taller for a
// reason that has nothing to do with D5.
async function openWorkBench(page) {
  await declareOneHubEach(page)
  await page.goto('/')
  await page.getByRole('tab', { name: 'WorkBench' }).click()
  // Visible before anything is measured: `boundingBox()` on a column in a
  // `display: none` tab is null for the wrong reason.
  await expect(page.locator(SHELL)).toBeVisible()
}

function control(page, index) {
  return page.locator(GUTTER).getByRole('button').nth(index)
}

async function pressControl(page, index) {
  const label = await control(page, index).getAttribute('aria-label')
  expect(label, 'the control carries an accessible name').toBeTruthy()
  await page.getByLabel(label, { exact: true }).click()
}

async function stateClasses(page) {
  const cls = (await page.locator(SHELL).getAttribute('class')) ?? ''
  return cls.split(/\s+/).filter((name) => name.startsWith('chat-shell--'))
}

async function boxOf(page, selector) {
  return page.locator(selector).boundingBox()
}

// A box, or null when the element is out of the flow — which in this app means
// `display: none`, the verdict the whole design rests on (D2).
async function boxes(page, selectors) {
  const out = {}
  for (const [name, selector] of Object.entries(selectors)) {
    out[name] = await boxOf(page, selector)
  }
  return out
}

test.describe('WorkBench focus layout: three states, one gutter', () => {
  test('normal: both columns in the flow, the gutter between them', async ({
    page,
  }) => {
    await openWorkBench(page)

    // The shell's three children in document order — a flex item, not an
    // overlay (spec §3), read from the DOM the browser built.
    const children = await page
      .locator(SHELL)
      .evaluate((el) => Array.from(el.children).map((c) => c.className))
    // The columns' own variant classes (`chat-col--single`) ride along, so the
    // state is matched as a class among the others rather than as the whole.
    expect(children[0]).toMatch(/(^|\s)chat-col--left(\s|$)/)
    expect(children[1]).toBe('chat-col-gutter')
    expect(children[2]).toMatch(/(^|\s)chat-col--right(\s|$)/)

    // Neither state class: the columns keep their own rules.
    expect(await stateClasses(page)).toEqual([])

    // Both controls are labelled, and their names come from the app — the spec
    // only checks they exist, are unique and reach a visible element.
    const names = await page
      .locator(GUTTER)
      .getByRole('button')
      .evaluateAll((btns) => btns.map((b) => b.getAttribute('aria-label')))
    expect(names).toHaveLength(2)
    expect(new Set(names).size, 'the two controls are distinguishable').toBe(2)
    for (const name of names) {
      await expect(page.getByLabel(name, { exact: true })).toBeVisible()
    }

    // Everything measurable has a box: nothing is hidden in `normal`.
    const measured = await boxes(page, {
      left: LEFT_COL,
      right: RIGHT_COL,
      panel: PANEL,
      dock: DOCK,
      gutter: GUTTER,
    })
    for (const [name, box] of Object.entries(measured)) {
      expect(box, `${name} is in the flow`).not.toBeNull()
      expect(box.width, `${name} has a width`).toBeGreaterThan(0)
    }
  })

  test('reasoning: the conversation leaves the flow, the timeline owns the window', async ({
    page,
  }) => {
    await openWorkBench(page)
    const baseline = await boxes(page, { panel: PANEL, gutter: GUTTER })

    await pressControl(page, TOP_CONTROL)
    await expect(page.locator(SHELL)).toHaveClass(/chat-shell--reasoning/)

    // D2 in the only layer that can state it: Playwright's hidden verdict is
    // `display: none`, so the composer, the Stop button and the conversation
    // are gone from the flow — not shrunk (D6).
    await expect(page.locator(RIGHT_COL)).toBeHidden()
    expect(await boxOf(page, RIGHT_COL)).toBeNull()
    // D5: the dock and its bar leave with it, or the timeline would gain width
    // and still lose its height to the dock's minimum.
    await expect(page.locator(DOCK)).toBeHidden()
    expect(await boxOf(page, DOCK)).toBeNull()

    // The maximised sketch: wider *and* taller than the baseline, the height
    // being the part a dock strip left behind would fail.
    const panel = await boxOf(page, PANEL)
    expect(panel.width, 'the timeline takes the row').toBeGreaterThan(
      baseline.panel.width,
    )
    expect(panel.height, 'the timeline takes the column').toBeGreaterThan(
      baseline.panel.height,
    )

    // The way back is still on screen: the gutter is the one surface no state
    // hides, and both its controls stay clickable from inside the state.
    await expect(page.locator(GUTTER)).toBeVisible()
    for (const index of [TOP_CONTROL, LOWER_CONTROL]) {
      await expect(control(page, index)).toBeVisible()
      expect(await control(page, index).boundingBox()).not.toBeNull()
    }
  })

  test('the way back: the same control restores the baseline boxes', async ({
    page,
  }) => {
    await openWorkBench(page)
    const before = await boxes(page, {
      left: LEFT_COL,
      right: RIGHT_COL,
      panel: PANEL,
      dock: DOCK,
    })

    await pressControl(page, TOP_CONTROL)
    await pressControl(page, TOP_CONTROL)

    expect(await stateClasses(page)).toEqual([])
    const after = await boxes(page, {
      left: LEFT_COL,
      right: RIGHT_COL,
      panel: PANEL,
      dock: DOCK,
    })
    for (const [name, box] of Object.entries(after)) {
      expect(box, `${name} is back in the flow`).not.toBeNull()
      // A pixel of slack: sub-pixel layout rounding, nothing else.
      expect(Math.abs(box.x - before[name].x)).toBeLessThanOrEqual(1)
      expect(Math.abs(box.y - before[name].y)).toBeLessThanOrEqual(1)
      expect(Math.abs(box.width - before[name].width)).toBeLessThanOrEqual(1)
      expect(Math.abs(box.height - before[name].height)).toBeLessThanOrEqual(1)
    }
  })

  test('conversation: the column leaves the flow and the same control reaches it', async ({
    page,
  }) => {
    await openWorkBench(page)
    const baseline = await boxes(page, { left: LEFT_COL, conversation: CONVERSATION })

    await pressControl(page, LOWER_CONTROL)
    await expect(page.locator(LEFT_COL)).toBeHidden()
    expect(await boxOf(page, LEFT_COL)).toBeNull()
    // The panel and the dock go with their column, without a rule of their own.
    expect(await boxOf(page, PANEL)).toBeNull()
    expect(await boxOf(page, DOCK)).toBeNull()

    const conversation = await boxOf(page, CONVERSATION)
    expect(conversation.width, 'the conversation takes the row').toBeGreaterThan(
      baseline.conversation.width,
    )

    // spec §3's assertion, the one a control placed on the gauge bar would have
    // failed: the chevron that hid the column is still visible and still
    // clickable once the column is gone.
    await expect(control(page, LOWER_CONTROL)).toBeVisible()
    await pressControl(page, LOWER_CONTROL)

    const restored = await boxOf(page, LEFT_COL)
    expect(restored).not.toBeNull()
    expect(Math.abs(restored.x - baseline.left.x)).toBeLessThanOrEqual(1)
    expect(Math.abs(restored.width - baseline.left.width)).toBeLessThanOrEqual(1)
  })

  // D4's asymmetry, which only a reload can show: `conversation` is a
  // furnishing preference and comes back; `reasoning` is a reading mode and
  // does not — booting into it would leave a screen with no composer.
  test('a reload keeps conversation, and drops reasoning', async ({ page }) => {
    await openWorkBench(page)

    await pressControl(page, LOWER_CONTROL)
    await expect(page.locator(LEFT_COL)).toBeHidden()
    await page.reload()
    await page.getByRole('tab', { name: 'WorkBench' }).click()
    await expect(page.locator(LEFT_COL)).toBeHidden()
    await expect(page.locator(SHELL)).toHaveClass(/chat-shell--conversation/)

    await pressControl(page, LOWER_CONTROL)
    await pressControl(page, TOP_CONTROL)
    await expect(page.locator(SHELL)).toHaveClass(/chat-shell--reasoning/)
    await page.reload()
    await page.getByRole('tab', { name: 'WorkBench' }).click()
    expect(await stateClasses(page)).toEqual([])
    await expect(page.locator(LEFT_COL)).toBeVisible()
    await expect(page.locator(RIGHT_COL)).toBeVisible()
  })

  // D3 + D5, the pair that says a state is a flag and not a size: the dock is
  // given a height, `reasoning` takes it out of the flow entirely, and it
  // comes back at the height it had — the dragged number never touched.
  //
  // The height is set through the bar's own keyboard contract (ArrowUp grows
  // the dock, `role="separator"`, one step per press) rather than a pointer
  // drag: same handler, same clamp, same write-through — and it does not
  // depend on how fast the mouse moves. The storage half of the same rule
  // ("no state reads or writes it") is `useChatLayoutFocus.test.js`; what is
  // only visible here is that the *rendered* height survives.
  test('a dock height given before reasoning comes back after it', async ({
    page,
  }) => {
    await openWorkBench(page)

    const beforeDrag = await boxOf(page, DOCK)
    expect(beforeDrag).not.toBeNull()

    // The bar, by role and scoped to the column that owns it: the composer
    // column carries a separator of its own, and its label is the app's to
    // write — neither is copied here.
    const bar = page.locator(LEFT_COL).getByRole('separator')
    await expect(bar, 'exactly one bar sizes the left column').toHaveCount(1)
    await expect(bar, 'the left column carries its own bar').toBeVisible()
    await bar.focus()
    // Enough steps to clear the dock's CSS default basis (26% of the column)
    // by a margin a rounding difference cannot eat.
    for (let step = 0; step < 8; step += 1) {
      await page.keyboard.press('ArrowUp')
    }

    const dragged = await boxOf(page, DOCK)
    expect(dragged.height, 'the bar grew the dock').toBeGreaterThan(beforeDrag.height)
    // The height is the splitter's own stored value, so the number below is
    // the one a state could in principle destroy.
    const stored = await page.evaluate((key) => window.localStorage.getItem(key), DOCK_SIZE_KEY)
    expect(stored, 'the drag persisted').not.toBeNull()

    await pressControl(page, TOP_CONTROL)
    // Out of the flow, not resized: no box at all, and the same for its bar.
    expect(await boxOf(page, DOCK)).toBeNull()
    expect(await boxOf(page, '.chat-col-resizer')).toBeNull()
    expect(
      await page.evaluate((key) => window.localStorage.getItem(key), DOCK_SIZE_KEY),
      'reasoning did not touch the stored height',
    ).toBe(stored)

    await pressControl(page, TOP_CONTROL)
    const restored = await boxOf(page, DOCK)
    expect(restored).not.toBeNull()
    expect(Math.abs(restored.height - dragged.height)).toBeLessThanOrEqual(1)
    expect(
      await page.evaluate((key) => window.localStorage.getItem(key), DOCK_SIZE_KEY),
    ).toBe(stored)
  })

  // Both controls are real `<button type="button">`s (D9), which is what makes
  // them reachable without a mouse — chrome the user hits repeatedly has to be.
  test('the keyboard drives both controls', async ({ page }) => {
    await openWorkBench(page)

    await control(page, TOP_CONTROL).focus()
    await page.keyboard.press('Enter')
    await expect(page.locator(SHELL)).toHaveClass(/chat-shell--reasoning/)
    await expect(control(page, TOP_CONTROL)).toHaveAttribute('aria-pressed', 'true')

    await page.keyboard.press('Space')
    expect(await stateClasses(page)).toEqual([])
    await expect(control(page, TOP_CONTROL)).toHaveAttribute('aria-pressed', 'false')

    await control(page, LOWER_CONTROL).focus()
    await page.keyboard.press('Enter')
    await expect(page.locator(SHELL)).toHaveClass(/chat-shell--conversation/)
    await expect(page.locator(LEFT_COL)).toBeHidden()

    await page.keyboard.press('Space')
    expect(await stateClasses(page)).toEqual([])
    await expect(page.locator(LEFT_COL)).toBeVisible()
  })
})
