// Which of the two tool hubs have anything to show for this conversation.
//
// Four places read this (the MCP panel, the LSP panel, the tools dock's
// folded summary, and the Chat left column's width modifier), and they must
// agree: the column is only one panel wide when exactly one hub has entries,
// otherwise the timeline and the dock would not line up.

export function mcpVisible(state) {
  return state.mcp.loaded && state.mcp.connectors.length > 0
}

export function lspVisible(state) {
  return state.lsp.loaded && state.lsp.servers.length > 0
}

// The left column keeps its two-panel width only when both hubs have
// something to show; with one or none it is one panel wide (the same rule the
// old input-row toggle column and tool panel shared).
export function bothHubs(state) {
  return mcpVisible(state) && lspVisible(state)
}

// Nothing to switch at all (no connector and no language server declared):
// the dock has no body, so it stays a single strip and the timeline owns the
// column's height.
export function anyHub(state) {
  return mcpVisible(state) || lspVisible(state)
}

// `MCP 2/3 · LSP 1/2`: how many of each hub's tools are switched on, shown
// while the dock is folded (the switches themselves are only in the panels).
export function hubsSummary(state) {
  const parts = []
  for (const [label, hub, entries] of [
    ['MCP', state.mcp, state.mcp.connectors],
    ['LSP', state.lsp, state.lsp.servers],
  ]) {
    if (!entries.length) continue
    const on = entries.filter((entry) => hub.enabled[entry.id]).length
    parts.push(`${label} ${on}/${entries.length}`)
  }
  return parts.join(' · ')
}
