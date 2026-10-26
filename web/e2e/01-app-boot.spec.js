import { expect, test } from '@playwright/test'

// Task 01 — level: UI state. Nothing here streams: the spec only checks what
// the app tells the user at boot (endpoint/model, tabs, empty run state).
// Therefore every expected string is read from GET /api/config and
// GET /api/path-location instead of being written in the spec, so the spec
// follows the configuration (.env.test) instead of hardcoding it.
test.describe('App boot: displayed configuration, tabs, empty state', () => {
  // Opens the app and returns it to a *verified* idle state, plus the live
  // config/path-location the assertions below are built from.
  async function bootIdle(page) {
    const configRes = await page.request.get('/api/config')
    expect(
      configRes.ok(),
      `GET /api/config should succeed, got ${configRes.status()}`,
    ).toBe(true)
    const config = await configRes.json()

    const pathRes = await page.request.get('/api/path-location')
    expect(
      pathRes.ok(),
      `GET /api/path-location should succeed, got ${pathRes.status()}`,
    ).toBe(true)
    const pathLocation = (await pathRes.json()).path

    await page.goto('/')

    // The shared server may still carry the result of an earlier spec, so use
    // the app's own control instead of assuming a clean server: after a
    // finished or failed run the stop button *is* the reset.
    const pill = page.locator('.status-pill')
    await expect(pill, 'app bar should report a run status').toBeVisible()
    const status = (await pill.textContent())?.trim()
    if (status === 'finished' || status === 'failed') {
      await page.getByLabel('Reset to initial state').click()
      await expect(pill).toHaveText('idle', { timeout: 10_000 })
      await expect(page.locator('.stream-empty')).toBeVisible()
    } else {
      // A *running* run belongs to somebody else on this port; cancelling it
      // silently would hide the real problem.
      expect(
        status,
        `the app on this port has a live run (status "${status}") - e2e needs a free server, use E2E_PORT`,
      ).toBe('idle')
    }

    return { config, pathLocation }
  }

  test('app bar names the model and the server it will benchmark', async ({ page }) => {
    const { config } = await bootIdle(page)

    // Two `.brand-sub` lines: the model list, then "on <SERVER_URL>".
    await expect(page.locator('.brand-sub')).toHaveText([
      config.models.join(' · '),
      `on ${config.serverUrl}`,
    ])
  })

  test('three tabs exist, Benchmark selected', async ({ page }) => {
    await bootIdle(page)

    await expect(page.getByRole('tab', { name: 'Benchmark' })).toHaveAttribute(
      'aria-selected',
      'true',
    )
    await expect(page.getByRole('tab', { name: 'WorkBench' })).toBeVisible()
    await expect(page.getByRole('tab', { name: 'Editor' })).toBeVisible()
    expect(await page.getByRole('tab').count()).toBe(3)
  })

  test('stats column shows a run that has not happened yet', async ({ page }) => {
    const { config } = await bootIdle(page)

    await expect(page.locator('.chronometer')).toHaveText('00:00:00')

    const scoreCard = page
      .locator('.metric-card')
      .filter({ has: page.getByText('Score', { exact: true }) })
    await expect(scoreCard.locator('.metric-value')).toHaveText('0/0')

    const questionCard = page
      .locator('.metric-card')
      .filter({ has: page.getByText('Question in progress', { exact: true }) })
    await expect(questionCard.locator('.metric-value')).toHaveText('--')

    // Question count and sampling summary both come from the server payload.
    await expect(page.locator('.run-meta')).toHaveText(
      `${config.questionCount} questions · ${config.samplingSummary}`,
    )
  })

  test('empty stream panel offers a start and no reset', async ({ page }) => {
    await bootIdle(page)

    // `.stream-empty-title` also exists in the hidden WorkBench view (tab
    // views stay mounted via v-show), so scope it to the stream column.
    const stream = page.getByRole('region', { name: 'Answer stream' })
    await expect(stream.locator('.stream-empty-title')).toHaveText(
      'Ready to benchmark',
    )
    await expect(stream.locator('.stream-empty')).toBeVisible()
    await expect(page.getByLabel('Run benchmark')).toBeEnabled()
    await expect(page.getByLabel('Reset to initial state')).toBeDisabled()
  })

  test('app bar shows the active working location', async ({ page }) => {
    const { pathLocation } = await bootIdle(page)

    await expect(page.locator('.appbar-workspace-path')).toHaveText(pathLocation)
  })

  // The /api/config route is stubbed here on purpose: this test is about the
  // warning the app raises for an unusable configuration, not about the
  // configuration itself (the real one does have a verifier, checked below).
  test('a missing verifier is announced instead of silently failing answers', async ({
    page,
  }) => {
    const configRes = await page.request.get('/api/config')
    expect(configRes.ok(), 'GET /api/config should succeed').toBe(true)
    const config = await configRes.json()
    expect(
      config.verifierConfigured,
      'the real config is expected to have a verifier, otherwise the stub proves nothing',
    ).toBe(true)

    await page.goto('/')
    await expect(
      page.getByText('No verifier agent configured'),
      'the warning must be absent with the real config',
    ).toHaveCount(0)

    await page.route('/api/config', async (route) =>
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ ...config, verifierConfigured: false }),
      }),
    )
    await page.reload()

    await expect(page.getByRole('alert')).toContainText(
      'No verifier agent configured',
    )
  })
})
