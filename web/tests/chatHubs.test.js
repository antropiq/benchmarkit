import { describe, expect, it } from 'vitest'

import {
  anyHub,
  bothHubs,
  hubsSummary,
  mcpVisible,
} from '../src/lib/chat-hubs'

// `lib/chat-hubs` is the single source of truth for what the tools dock and
// the Chat left column's width key on. A mismatch would put the reasoning
// timeline and the dock out of alignment, so the predicates are worth pinning
// down rather than re-deriving in three components.
function hubState({ mcpLoaded = true, mcp = [], lspLoaded = true, lsp = [] }) {
  return {
    mcp: {
      loaded: mcpLoaded,
      connectors: mcp,
      enabled: Object.fromEntries(mcp.map((c) => [c.id, c.on])),
    },
    lsp: {
      loaded: lspLoaded,
      servers: lsp,
      enabled: Object.fromEntries(lsp.map((s) => [s.id, s.on])),
    },
  }
}

const MCP = [{ id: 'console', label: 'Console', on: true }]
const LSP = [{ id: 'pyright', label: 'Pyright', on: false }]

describe('hub visibility predicates', () => {
  it('hides a hub whose snapshot never arrived', () => {
    const state = hubState({ mcpLoaded: false, mcp: MCP })
    expect(mcpVisible(state)).toBe(false)
    expect(anyHub(state)).toBe(false)
  })

  it('hides a hub with no declared entry', () => {
    const state = hubState({})
    expect(anyHub(state)).toBe(false)
    expect(bothHubs(state)).toBe(false)
  })

  it('claims the wide column only when both hubs have entries', () => {
    expect(bothHubs(hubState({ mcp: MCP, lsp: LSP }))).toBe(true)
    expect(bothHubs(hubState({ mcp: MCP }))).toBe(false)
    expect(bothHubs(hubState({ lsp: LSP }))).toBe(false)
    expect(bothHubs(hubState({}))).toBe(false)
  })
})

describe('hubsSummary', () => {
  it('counts switched-on entries per hub', () => {
    const state = hubState({
      mcp: [...MCP, { id: 'playwright', label: 'Playwright', on: false }],
      lsp: LSP,
    })
    expect(hubsSummary(state)).toBe('MCP 1/2 · LSP 0/1')
  })

  it('omits a hub with nothing to show', () => {
    expect(hubsSummary(hubState({ mcp: MCP }))).toBe('MCP 1/1')
    expect(hubsSummary(hubState({ lsp: LSP }))).toBe('LSP 0/1')
  })

  it('is empty when neither hub is present', () => {
    expect(hubsSummary(hubState({}))).toBe('')
  })

  it('survives an entry missing from the enable map', () => {
    const state = hubState({ mcp: MCP })
    state.mcp.enabled = {}
    expect(hubsSummary(state)).toBe('MCP 0/1')
  })
})
