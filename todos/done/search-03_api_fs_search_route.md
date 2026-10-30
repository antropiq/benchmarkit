# search-03 — `GET /api/fs/search`: the read route and its allowlist entry

Put the core behind the API: one new `GET` under `/api/fs`, camelCase query params, and
the route that only re-raises the core's status. From here the endpoint is live and
usable from a browser — the frontend tasks still have their own tests to stub it.

Depends on: **search-02** (`editorfs.search_workspace`, `editorfs.SearchQueryError`).
Read `docs/tree-search-spec.md` §4's route section and the bullet about the write-surface
guard; decision D16 and *Traps* §4 come from `todos/search-00_big_picture.md`.

## Files

| file | change |
|---|---|
| `src/api/routes.py` | the new route, in the read half of the `/fs` block |
| `tests/test_editorfs.py` | the `/api/fs` verb map in `TestSurfaceShape` gains one entry |
| `tests/test_editorfs_search.py` | route-level rows: params, statuses, camelCase |

## What to deliver

### The route

```python
@router.get("/fs/search")
def get_fs_search(
    path: str = "",
    q: str = "",
    caseSensitive: bool = False,
    regex: bool = False,
) -> Dict[str, Any]:
```

- Placement: with the other reads (`/fs/tree`, `/fs/file`, `/fs/stat`, `/fs/status`,
  `/fs/diff`), **before** the `# -- the guarded write surface` divider. The divider is
  the file's own statement of which routes touch disk; this one is on the read side of
  it.
- `path` defaults to `""` — the whole-workspace scope, the same convention
  `get_fs_tree` uses. `q` defaults to `""` so an unset query reaches the core's 400
  rather than FastAPI's 422; the reason the *client* should hear ("empty query") is
  written by the core, in one sentence, not by a validation decorator.
- camelCase param names, matching `ifUnmodifiedSince` elsewhere on `/api/fs`. Python
  style wants snake_case — do not rename them and do not add aliases: the contract is
  camelCase on the wire, and `routes.py` is the one place that spells it.
- Body: `try: return editorfs.search_workspace(path, query=q, case_sensitive=caseSensitive, regex=regex)`
  then two `except` clauses, `SearchQueryError` and `WorkspacePathError`, each re-raising
  as `HTTPException(status_code=exc.status_code, detail=exc.detail)`. **No logic here.**
  Every rule — the walk, the verdicts, the caps, the footer — is already in the core;
  if this handler grows past the call and two re-raises, the rule is in the wrong file.
- Order of the `except` clauses matters only if one type subclasses the other; they do
  not. If you make `SearchQueryError` a subclass of `WorkspacePathError` instead of a
  sibling, say so in a comment and re-raise with one clause — but §4 says it follows
  `WorkspacePathError` *exactly*, which reads as sibling, so a sibling is the default
  and search-02 will have built it that way.
- Docstring in this file's voice: one paragraph saying where the rules live (the way
  `get_fs_tree`'s does) and one sentence naming the three statuses the core answers.
  Say plainly that **this route reads** — it is the sentence the neighbouring handlers
  carry, and the guard suite is built on the distinction.

### Write-surface hygiene (D16)

This is a `GET`, so it needs **no** entry in `ALLOWED_HOMES` and no change to
`tests/test_write_surface_isolation.py`'s lists. `GET /api/fs` is deliberately not one
of the three write markers (the test's own comment: chosen so a docstring naming a
route method stays out of the scan). What you must not do is *add* a marker-bearing
string to a nearby docstring — do not write a sentence like "unlike `PUT /api/fs/file`,
this route…" inside the new handler, because that string is scanned and the file is
allowlisted but the *sentence* would still be the reason a future reader wonders.
Refer to the write routes by name, not by verb-and-path spelling.

*(Optional, one small addition worth making: that file already guards
`/agent-modes` by parsing the handler's **verb** out of the AST —
`route_registrations(source, "/agent-modes")` asserts `["get"]`. A two-line sibling
guard for `/fs/search` would pin "search stays a read" the same way the verb map does,
and the helper already exists. Only do this if it stays under ten lines; the verb map
in `test_editorfs.py` already covers the same ground, so it is a judgement call, not a
requirement.)*

### The verb-map allowlist (this is a review, not a chore)

`tests/test_editorfs.py:677` asserts the whole `/api/fs` surface as an exact dict:

```python
assert fs == {
    "/api/fs/tree": ["get"],
    "/api/fs/file": ["delete", "get", "put"],
    "/api/fs/create": ["post"],
    "/api/fs/search": ["get"],        # <- add this
    "/api/fs/stat": ["get"],
    "/api/fs/status": ["get"],
    "/api/fs/diff": ["get"],
    "/api/fs/rename": ["post"],
}
```

That test's docstring says a new route "has to appear here on purpose". Adding the line
*is* the recorded review that search is a read — so add `["get"]` and nothing else, and
do not touch the mutation half of the dict. If the assertion ever wanted a second verb
here, that would be the lot breaking D16; refuse it.

## Tests — extend `tests/test_editorfs_search.py`

Use the same `client` fixture pattern as `tests/test_editorfs.py`
(`TestClient(create_app())` over the `tmp_path` workspace). The core's rules are already
tested directly; these rows are about the **transport**.

| test | asserts |
|---|---|
| happy path over HTTP | `GET /api/fs/search?path=src&q=…` answers 200 with the Contract's exact key set |
| defaults | `?q=verb` alone works: scope is the root (`scope: ""`), `caseSensitive` and `regex` both default false — a literal `(` in `q` does not 400, which is what proves `regex` defaulted to false rather than to true |
| camelCase is the wire form | `caseSensitive=true` and `regex=true` each change the answer. Then the negative: `case_sensitive=true` (snake_case) leaves the answer at the case-**insensitive** default, because FastAPI ignores an unknown query param rather than rejecting it — pin that, so a future rename of the parameter to snake_case is caught by a red test and not by a review |
| invalid pattern | 400 and `detail` carrying `re`'s message (e.g. `unterminated character set`) |
| empty query | 400 with the core's sentence, **not** a FastAPI validation error: `detail` is a string, whereas FastAPI's 422 `detail` is a list |
| path escapes | `?path=../elsewhere` ⇒ 422; `?path=/etc` ⇒ 422 |
| hidden scope | a `hidden` rule over a folder, then search that folder ⇒ **404**, and the body does not name the rule (pol-00 D8: an observation of a hidden path is indistinguishable from absence) |
| absent scope | a path with nothing there ⇒ 404 |
| no write verb | `POST`/`PUT`/`DELETE` on `/api/fs/search` ⇒ 405 |

## Acceptance

- `pytest tests/test_editorfs_search.py tests/test_editorfs.py tests/test_api.py
  tests/test_write_surface_isolation.py` green.
- `TestSurfaceShape` now lists `"/api/fs/search": ["get"]` and the rest of that dict is
  byte-identical to before.
- The handler body is the call plus two re-raises — count the lines; if it is longer,
  move the extra logic back to `search.py`.
- `cd web && npm test` still green (untouched by this task; it is the baseline that
  search-04 onward has to keep green).
- `git diff --stat` shows exactly three files: `src/api/routes.py`,
  `tests/test_editorfs.py`, `tests/test_editorfs_search.py` (+ at most the optional
  guard inside `tests/test_write_surface_isolation.py`). No frontend file, and
  `grep -rn "fs/search" web/src` finds nothing — the UI arrives in search-05..07.
- The feature is reachable by URL but not yet by any gesture in the app. That is
  expected; say so rather than wiring a button to prove it.
- Move this file to `todos/done/` when the above is true.
