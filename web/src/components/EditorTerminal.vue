<script setup>
// The terminal strip's placeholder (ed-02). The header chrome below is final;
// the body is a label until the xterm.js front end (ed-07) mounts a real PTY
// session over `WS /api/terminal` (ed-06). Until then there is no session to
// reconnect and nothing to kill, so the dot reads "off" and the button is
// disabled rather than faking a state.
</script>

<template>
  <section class="editor-terminal" aria-label="Workspace terminal">
    <header class="editor-terminal-head">
      <span class="editor-terminal-title">Terminal</span>
      <span class="editor-terminal-state" role="status">
        <span class="status-dot editor-terminal-dot off" aria-hidden="true"></span>
        <span class="caption">no session</span>
      </span>
      <button
        class="icon-btn editor-terminal-kill"
        type="button"
        disabled
        aria-label="Kill the terminal session"
        title="Kills the shell and everything it started"
      >
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor"
          stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
          <path d="M18 6 6 18" />
          <path d="m6 6 12 12" />
        </svg>
      </button>
    </header>

    <div class="editor-terminal-body">
      <p class="caption">Terminal coming in ed-07.</p>
    </div>
  </section>
</template>
