# Step 3 — Click a file, see its content

Selection in the tree loads the file into the (still plain-`<pre>`) editor pane:
a read endpoint, a composable that owns open-file state, and the language
guess that ed-04 will feed to CodeMirror. Read-only: nothing here writes.

Depends on: ed-01 (tree + `editorfs`), ed-02 (the pane to fill).

## Endpoint: `GET /api/fs/file?path=<rel>`

```jsonc
// GET /api/fs/file?path=src/config.py
{
  "path": "src/config.py",
  "text": "…",                    // absent when binary
  "size": 25921,                  // bytes on disk
  "mtime": 1761371340.41,         // the write guard's token (ed-04)
  "language": "python",           // extension → editor language id
  "binary": false,
  "truncated": false,
  "encoding": "utf-8"
}
```

Rules, each one mirrored by a test:

* **Same gate as the tree**: `editorfs.policy.resolve_in_workspace(rel)` with
  `for_write=False`, so `deny_reason()` applies (`.env*`, `*.pem`, `id_ed25519`,
  `.git/…`, `mcp_connectors/…`). The UI never gets to ask for a file the
  connectors refuse — decision D3, one policy per surface that touches disk.
* **Size cap**: `editorfs.constants.MAX_EDITOR_BYTES = 1_000_000`, the same
  number and the same reasoning as `lsphub.constants.MAX_DOCUMENT_BYTES` ("a
  language server is not a way to stream a giant file into the context window"
  — the same argument applies to a browser tab). Above it: 413 with the size in
  the message, and the tree row for such a file carries `size` so the UI can
  say so before the click.
* **Binary sniff**: a NUL byte in the first 8 KiB, or a `UnicodeDecodeError`
  on the head — answered with `binary: true` and **no `text` key**, not an
  error. The pane shows "binary file, 2.1 MB".
* **Truncation over the cap mid-decode is not silently lossy**: `truncated:
  true` plus the byte offset, in the same honesty spirit as the `tokensEstimated`
  flag documented in AGENTS.md (never present a partial as a whole).
* **Newline preservation**: `text` is the file's exact characters, CRLF
  included. Never normalise on the server; CodeMirror keeps its own line-ending
  setting and re-writes what it received.
* Errors: 422 escape/deny, 404 missing, 400 directory.

`language` is computed server-side (one dict in `editorfs/files.py`) so the
tree, the read endpoint and the client cannot disagree about what a file *is*:

| Extension | Language id |
|---|---|
| `.py` `.pyi` | `python` |
| `.js` `.mjs` `.cjs` `.jsx` | `javascript` |
| `.ts` `.tsx` | `typescript` |
| `.vue` | `vue` |
| `.json` `.jsonc` | `json` |
| `.md` | `markdown` |
| `.html` `.css` `.scss` `.toml` `.yaml` `.yml` `.sh` | their own id, loaded in ed-08 if at all |
| anything else | `plaintext` |

Unknown extensions and dotfiles (`Dockerfile`, `Makefile`, `.gitignore`) fall
back to `plaintext`; `web/dist` is ignored by the tree, so no minified bundle
reaches this path in practice.

## Composable: `composables/useEditorFiles.js`

The singleton that owns what is open — the pattern every other screen follows
(state in a composable, components render it):

```js
state = {
  open: [],            // [{ path, text, mtime, language, size, binary, stale }]
  active: null,        // path of the visible tab
  loading: false,
  error: null,
}
openFile(path)  // fetch + append-or-focus; sets state.active
closeFile(path) // drop it
reloadFile(path)// re-GET and swap the text — driven by ed-09, not by a Save
```

`openFile` on a path that is already open **focuses** it. It does *not* refetch:
with a read-only pane a refetch would jump the scroll position for no benefit,
and the freshness question belongs to ed-09's probe rather than to every click.

Each record carries the `mtime` returned by the read, which is the value ed-09
compares against `GET /api/fs/stat`, and a `stale` flag it sets when the two
disagree. There is **no `dirty` field in this lot** — nothing is editable
(decision D4), so there is nothing to mark, no confirm on close, and no
`beforeunload` warning. Those belong to ed-10.

The tree's `selected` and the editor's `active` are deliberately separate
pieces of state: selecting a row in the tree highlights it; opening a file puts
it in the tab strip. They coincide in the common case and diverge when a click
is refused (binary, too big), which is exactly when a single shared
value would lie about what is on screen.

## UI in the editor pane

* Tab strip above the pane: one chip per open file — name, path in `title`, a
  `×` button. Middle-click and `Ctrl+W` close a tab; with a read-only pane
  there is nothing to confirm first (a dirty check arrives with the write layer
  of ed-10).
* The pane itself stays a `<pre class="editor-raw">` in this step, with
  `white-space: pre`, `tab-size: 4`, JetBrains Mono (already a `@fontsource`
  dependency and used by `.mono`), and a line-number gutter rendered from a
  `v-for` over the lines **only under ~2000 lines** — beyond that the raw
  preview shows a "large file, syntax view coming with the editor" note rather
  than hanging the tab. ed-04 replaces this component entirely.
* Load errors land in `.alert-error` inside the pane, with the file path
  echoed, so a policy refusal reads as "this file is protected", not as a
  broken tab.

## Tests

Extend `tests/test_editorfs.py`:

1. text read: `path`, `size`, `mtime`, `language` all correct for each row of
   the extension table;
2. `.env`, `id_ed25519`, `.git/config`, a `mcp_connectors/` script → 4xx, and
   the body names the rule;
3. `../` and absolute `path` → 422; symlink to `/etc/hosts` → 422;
4. `MAX_EDITOR_BYTES + 1` file → 413 (patch `editorfs.constants`);
5. a NUL-containing file → `binary: true`, no `text` key;
6. CRLF file → the returned text still contains `\r\n`;
7. missing → 404, directory → 400.

## Manual checklist

* Click `src/config.py`: content appears, tab chip added, no duplicate on a
  second click.
* Click a PNG: "binary file" message, no garbage, tab chip absent.
* Click a 4 MB `results.json`-sized file: refusal names the size.
* Reload with three tabs open: the tabs are not restored in this step (parked
  with persistence in ed-08) — noted so it is not mistaken for a bug.

## Delivered

| File | Change |
|---|---|
| `src/core/editorfs/files.py` | `read_file` (the payload above: cap → 413 naming the size, NUL-in-8 KiB / non-UTF-8 → `binary: true` with no `text` key, `truncated` + `truncatedAt`, byte-exact text) and `LANGUAGE_BY_EXTENSION` + `language_for`, patchable through `editorfs.constants` |
| `src/api/routes.py` | `GET /api/fs/file` — status mapping only, like the tree and the probe |
| `src/core/editorfs/__init__.py` | re-exports `read_file`; the package docstring's ed-03 note becomes the description of what is there |
| `web/src/composables/useEditorFiles.js` | new singleton: `open/active/loading/message`, `openFile` (append-or-focus, never refetch), `closeFile` (focuses the neighbour), `focusFile`, `reloadFile` (ed-09's entry point, bumps `reloadedAt`), `humanSize`, `MAX_GUTTER_LINES`; records carry `mtime`, `language`, `stale`, no `dirty` (D4) |
| `web/src/components/EditorArea.vue` | tab chips (`role="tablist"`, roving `←→`/`Home`/`End`, middle-click and `Ctrl+W` close), the `<pre class="editor-raw">` body with a sticky line-number gutter, the >2000-line note, and `.alert-error` / binary messages with the path echoed |
| `web/src/composables/useWorkspaceTree.js` | the `select()` comment now names ed-03 as the consumer (no behaviour change) |
| `web/src/styles/theme.css` | `.editor-tab` chips, `.editor-body-col`, `.editor-raw-scroll`, `.editor-gutter`, `.editor-raw` (`white-space: pre`, `tab-size: 4`, mono); `.editor-body` now `overflow: hidden` so the scroll is the text's |
| `tests/test_editorfs.py` | 45 new tests (26 → 71) — payload shape, all 19 extension rows + dotfiles, the deny-list naming its rule, the three escape locks, the symlink, 413, NUL, invalid UTF-8, CRLF, the cap race's `truncated`/`truncatedAt`, a cap cut mid-character, 404/400, and the connector-source boundary below |
| `AGENTS.md` | the `core/editorfs/` row now mentions the read half |

### What differs from the plan, and why

* **`mcp_connectors/…` is not a read refusal, and cannot be.** The plan listed
  it among the 4xx names; `path_policy.deny_reason(…, for_write=False)` puts
  that directory in `DENY_COMPONENTS` (write-only) rather than in
  `DENY_COMPONENTS_SECRET`, so the tree already *lists* `console_mcp.py` — and
  the read endpoint refusing what the tree offers is exactly the disagreement
  D3 forbids. A test now states the boundary: the connector *sources* read fine
  (they are this project's own code, and read-only anyway), while a secret
  *name* inside that folder (`mcp_connectors/id_ed25519`) still answers 422.
* **`binary` answers also cover non-UTF-8 text, and a cap cut is not binary.**
  The plan's sniff (NUL, decode error) is implemented verbatim; what the plan
  left open is the interaction with truncation. A slice that ends mid-character
  would flag a perfectly ordinary UTF-8 file as binary, so the tail fragment is
  dropped instead — `truncated: true` already says the rest is missing.
* **`size` is re-stated when the read ran past the cap.** The stat and the read
  are two calls, so a file growing between them leaves `size` below what was
  read; in that branch only, the size is re-stat'ed (bounded below by
  `cap + 1`, the number the read proved). No new code path runs on a normal
  read.
* **Language id keys on the asked-for name, not the realpath.** A symlink's
  target basename is another name on disk; the id should describe the row that
  was clicked. Dotfiles and dot-less names (`Makefile`, `Dockerfile`) fall to
  `plaintext` — `str.rfind('.') <= 0`, one condition covering both.
* **The pane's `error: null` is a `message` object.** One field answers both
  the refusals and the non-errors (binary, truncation) that ed-03 has to
  distinguish, and it carries the path the plan asks the pane to echo. A
  message does not blank an already-open buffer: the tree's selection and the
  active buffer are separate state, so a refused click leaves the previous
  file on screen with the reason above it.
