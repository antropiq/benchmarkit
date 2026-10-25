# Step 5 — Frontend dependencies (install + bundle policy)

Add the xterm and CodeMirror packages to `web/package.json`, and set the rules
that keep the new weight out of the Benchmark tab. Small step, done **before**
ed-04 and ed-07 so those steps start from installed imports.

Depends on: nothing. Blocks ed-04 and ed-07.

## Packages

```bash
cd web
npm install @xterm/xterm @xterm/addon-fit @xterm/addon-web-links
npm install codemirror @codemirror/state @codemirror/view \
            @codemirror/lang-python @codemirror/lang-javascript \
            @codemirror/lang-vue @codemirror/lang-json @codemirror/lang-markdown
```

| Package | Version at time of writing | Notes |
|---|---|---|
| `@xterm/xterm` | `^6.0.0` | the scoped name (the old `xterm` package is deprecated) |
| `@xterm/addon-fit` | `^0.11.0` | recompute rows/cols on resize — required, the pane is splitter-sized |
| `@xterm/addon-web-links` | `^0.12.0` | URLs in terminal output become clickable, opened with `window.open(…, 'noopener')` |
| `codemirror` | `^6.0.2` | preset (`basicSetup`); pulls `commands`, `search`, `autocomplete`, `language`, `state`, `view` |
| `@codemirror/state`, `@codemirror/view` | `^6.x` | imported directly, per ed-04 |
| `@codemirror/lang-python` | `^6.2.1` | |
| `@codemirror/lang-javascript` | `^6.2.5` | covers TS too |
| `@codemirror/lang-vue` | `^0.1.3` | deps: `@lezer/lr`, `@lezer/common`, `@lezer/highlight`, `@codemirror/language`, `lang-html`, `lang-javascript` — all resolved by the packages above |
| `@codemirror/lang-json`, `@codemirror/lang-markdown` | `^6.x` | the two most common in this repo |

Two things **not** to add, with reasons, so nobody re-adds them:

* no `monaco-editor` / `@guolao/vue-monaco-editor` — needs a same-origin worker
  file, and the SPA is served by a plain `StaticFiles` mount (`server.py:98`);
* no `vite-plugin-monaco-editor`-style worker plugin for the same reason.

## Bundle policy (decision D7)

`web/dist` is already ~8 MB, most of it mermaid (one chunk at 629 kB) plus the
KaTeX/Inter/JetBrains-Mono fonts. Rules for the new code:

1. `EditorView.vue` is loaded through `defineAsyncComponent` (ed-02) — the
   default screen never fetches any of this;
2. language packages are dynamic imports keyed by the `language` id (ed-04);
3. xterm is imported **inside** `EditorTerminal.vue`, not at the top of
   `EditorView.vue`, so a user who opens the tab and only browses the tree does
   not download the emulator;
4. CSS is imported where it is needed, not in `main.js`:
   `@xterm/xterm/css/xterm.css` inside the terminal component; CodeMirror is
   styled through its own theme extension plus `theme.css` overrides, so no
   third-party CSS file joins the app bundle.

Expected after `npm run build`: one chunk for the Editor view (~60–90 kB), one
per language grammar (5–40 kB), one for xterm (~290 kB). The **Benchmark tab's
critical path must not grow at all** — that is the acceptance criterion, checked
in the browser network panel, not in a bundler report.

If a future audit decides the total matters: `build.manualChunks` in
`vite.config.js` can name `editor`, `terminal` and `codemirror-langs` groups.
Not done now — the existing build already relies on Vite's default chunking and
adding an early optimisation without a measured problem is how `dist/` got to 8
MB in the first place.

## Vite dev server — the one config change

The terminal's WebSocket must be proxied too (ed-06):

```js
server: {
  proxy: {
    '/api': {
      target: 'http://localhost:8000',
      changeOrigin: true,
      ws: true,          // forwards /api/terminal to uvicorn
    },
  },
},
```

`ws: true` is inert for the HTTP routes, so `npm run dev` keeps working for
everything else. In production the SPA and the API are the same origin, so no
proxy is involved at all.

## Node toolchain note

`typescript` is already a devDependency and the language servers need
`node_modules/typescript` for `--tsdk` (AGENTS.md gotcha). The new packages are
plain JS-consumable ES modules, so nothing about `--tsdk`, Volar or the
`jsconfig.json` setup changes. `web/src` stays plain JS (`plain JS` per
AGENTS.md) — the CodeMirror and xterm APIs are used from JavaScript, and the
`@types/*` that ship with them are only used by the language servers, not by a
build step.

## Acceptance

* `cd web && npm run build` succeeds; `FastAPI` serves the new `dist/`.
* Open the Benchmark tab with the network panel filtered to JS: no chunk whose
  name contains `xterm`, `codemirror`, `lang-` or `lezer`.
* Open the Editor tab: the Editor view chunk loads once; opening a `.py` file
  pulls the Python grammar only; the xterm chunk appears when the terminal pane
  initialises.
* `npm run dev` (Vite on :5173, API on :8000): `/api/*` HTTP routes still work,
  and (after ed-06) a WebSocket to `/api/terminal` upgrades through the proxy.
* `pytest` untouched — no Python dependency changes in this step.

## Files touched

```
web/package.json          dependencies added
web/package-lock.json     regenerated
web/vite.config.js        ws: true on the /api proxy
```
