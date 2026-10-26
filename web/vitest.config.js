import { mergeConfig } from 'vite'
import { defineConfig } from 'vitest/config'
import viteConfig from './vite.config.js'

// Layer 1: unit and component tests, no browser and no server.
//
// A `vitest.config.js` next to `vite.config.js` *replaces* it rather than
// adding to it, so the shared pieces (the Vue plugin above all) are merged
// in explicitly instead of duplicated here.
export default mergeConfig(
  viteConfig,
  defineConfig({
    test: {
      // jsdom, because the subjects of this layer are the composables and
      // `lib/` — state machines and pure functions. Components that need real
      // browser behaviour (Monaco, xterm, EventSource timing) belong in
      // `e2e/`, driven by Playwright against the real app.
      environment: 'jsdom',
      include: ['tests/**/*.test.js'],
      // Tests import { describe, it, expect } explicitly: no implicit globals
      // to collide with the app's own `window`/`document` access.
      globals: false,
      // Every test starts from untouched mocks and env, so a composable that
      // reads import.meta.env or window.location cannot leak into the next.
      restoreMocks: true,
      unstubEnvs: true,
      // CSS and font imports are stubbed (the default): the theme is verified
      // visually in e2e, not asserted here.
      css: false,
      // Worker threads keep a crash inside one file instead of ending the run.
      pool: 'threads',
      coverage: {
        provider: 'v8',
        // Only app sources count: without this the report is dominated by
        // vendor chunks (monaco, mermaid, katex) that a jsdom test can load.
        include: ['src/**'],
        exclude: [
          'src/main.js',
          'src/fonts/**',
          // Editor chrome needs a real browser (Monaco/xterm/EventSource), so
          // its coverage legitimately lives in e2e, not here.
          'src/components/MonacoEditor.vue',
          'src/components/EditorTerminal.vue',
        ],
        reporter: ['text', 'html'],
      },
    },
  }),
)
