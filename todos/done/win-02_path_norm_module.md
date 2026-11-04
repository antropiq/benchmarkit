# win-02 — `mcp_connectors/path_norm.py`: the one text layer every consumer shares

Create the shared, `src`-free module that the rest of the lot routes every
path-text decision through. **Pure text, no callers yet**: this task adds the
module and its unit tests and changes nothing that uses it. win-03 wires it
into the inbound lock; win-04 wires it into the consumers.

Depends on: win-01 (the markers exist before the tests are written).
Spec: `docs/windows-support/windows-support-target-A-no-crash.md` §4, first
piece (*"`mcp_connectors/path_norm.py` (new; shared, `src`-free)"*), and A-D1 /
A-D2.1 for the semantics the functions must implement.

## Files

| file | change |
|---|---|
| `mcp_connectors/path_norm.py` | **new** — `canonical_rel`, `is_qualified`, `path_parts` |
| `tests/test_path_norm.py` | **new** — the unit tests, run directly on Linux |

## 1. Why it lives in `mcp_connectors/` and not under `src`

Same reason as `policy_glob` (pol-02) and `policy_shared` (pol-08): the two
stdio connectors run as scripts launched by absolute path and cannot
`import src` — the wheel ships `src` only. Copy that module-docstring shape:
stdlib-only, no FastAPI, no `src` import, and a docstring that states *why* the
file is here (a reader who wants to "tidy it into `editorfs`" needs the reason
not to).

**No `sys.platform` / `os.name` branch anywhere in the module** (A §4: *"No
`sys.platform` branch in this module: it is pure text"*). Do not use
`os.path.normpath`, `os.path.sep` or `pathlib`'s OS-aware classes inside it
either — use `ntpath.splitdrive` only where noted below (it is a pure text
function, importable and correct on every platform).

## 2. The three functions

```python
def canonical_rel(text: str) -> str: ...
def is_qualified(text: str) -> bool: ...
def path_parts(text: str) -> tuple[str, ...]: ...
```

### `canonical_rel(text)` — both separators in, `/` out

Split on **both** `/` and `\`; drop `.` components and duplicate separators;
apply `..` **lexically**; re-join with `/`. Contract details:

- `..` that leaves the start is **kept** as a leading `..` component, so the
  caller's own traversal check keeps its meaning: `a/../b` → `b`, but `../a` →
  `../a` and `a/../../b` → `../b`. That is exactly today's
  "`normpath` collapses what stays inside, and only a surviving `..` is refused"
  behaviour (`policy.py:198-202`), which A-D1 keeps.
- `""` → `""`; `"."` → `""`; `"a/"` → `"a"`; `"/a"` → `"/a"` — **no translation
  of a leading separator**: qualification is a separate question (`is_qualified`
  answers it, and `normalise_rel` refuses before calling this). Do not silently
  strip what the lock is supposed to refuse.
- `src\config.py` → `src/config.py`; `a\\b` → `a/b`; `a\/b` → `a/b`.
- Whitespace is **not** touched (the lock already strips before calling).
- Idempotent: `canonical_rel(canonical_rel(x)) == canonical_rel(x)` for any
  input — worth a test, because `batch.py` may call it on already-normal text.

### `is_qualified(text)` — drive spec, UNC, or a leading separator

True when the text cannot be a workspace-relative path on **any** platform:

- a leading `/` or `\` (`/etc/x`, `\Users\vgr`) — on Windows
  `ntpath.isabs('/etc/x')` is `False`, which is the hole A-D2.1 closes;
- a non-empty `ntpath.splitdrive(text)[0]` (`C:x`, `C:\x`, `c:/x`) — note
  `isabs('C:x')` is **`False`** and `ntpath.join(root, 'C:x')` returns *the other
  drive's current directory*, so this must be its own check, not folded into a
  separator test;
- a UNC root (`//srv/share`, `\\srv\share`): covered by the **leading separator**
  rule, not by the drive rule. Verified on 3.13:
  `ntpath.splitdrive('//srv/share')` returns `('//srv/share', '')` — the whole
  share as the drive — but such a text also *begins* with a separator, so order
  the two checks so the separator rule catches it and add no third case.

**The single-letter constraint.** `ntpath.splitdrive` answers `('a:', 'b')` for
`a:b` and `('foo:', 'bar')` for `foo:bar` — a shape no Windows drive can be, and
a workspace may legitimately hold a file named `foo:bar`. So do **not**
refuse every non-empty `splitdrive`: the drive part must be **exactly one letter
followed by `:`** (an empty rest, `C:`, counts). Write that as its own test in
both directions — `C:x` qualified, `foo:bar/src` not — it is the single easiest
way this lot breaks Linux behaviour.

**The residue, decided here rather than left to chance:** a **leading**
one-letter-plus-colon component is a drive spec on both platforms, so `a:b` as a
whole path is refused even though it is a legal POSIX filename. That is a second
widening of accepted input beyond §7's `a\b` sentence — A-D2.1's own wording
(*"anything with a drive spec"*) forces it, and no rule can refuse `C:x` without
also refusing `a:b`. Pin it in a test **and** in the lock's refusal message, and
note it in the commit, so §7's claim can be checked against reality instead of
silently contradicted. A colon anywhere *after* the first component (`src/a:b`)
stays legal.

### `path_parts(text)` — one component splitter for every consumer

`canonical_rel(text).split("/")` as a tuple, so the ancestor walk and the
matcher cannot disagree by coincidence of separator. A **pre-normalised** input
is the contract (callers pass what `normalise_rel` returned); return `()` for
`""`. Keep it total — no raising; the lock's job is to raise.

## 3. Module docstring

State the three things a later contributor must not lose: (a) the canonical
form is `/`-joined workspace-relative text on every platform (roadmap §9 rule
1); (b) `\` is accepted **only at the inbound lock**, never emitted; (c) this
module answers *what a path text is*, never *whether it is allowed* — the
verdict belongs to `policy_modes.mode_for` and the locks to `editorfs.policy`.

## Tests (`tests/test_path_norm.py`)

Every case is plain text, so all of them run on Linux — no marker needed
(win-01's markers exist for other reasons). Cover at least:

1. the §6 test-1 inputs at this layer: `src\config.py`, `a\..\b`, `a//b`,
   `a\/b`, leading `\`;
2. `..` survival both ways (`a/../../b` → `../b`, `a/../b` → `b`) — the property
   `policy.py:201` depends on;
3. idempotence over a list of shapes;
4. `is_qualified`: the three §6 test-4 shapes (`/etc/x`, `//srv/share`, `C:x`)
   plus the rest of the qualified set — `C:\x`, `c:/x`, `\Users`, `C:`, `x:C`,
   `\\srv\share`, and (per *"the residue"* above) a leading one-letter colon such
   as `a:b` and `x:y/z`; and **not** qualified: `src/config.py`, `a\b`, ``, `.`,
   `..`, `src/a:b`, `foo:bar/src`, `http://host:8080/v1`-shaped text (a
   multi-letter first component with a colon — the case the single-letter
   constraint exists for). Assert both directions in one parametrised pair so a
   future loosening of the drive rule fails visibly.
5. `path_parts` of a normalised path equals `canonical_rel(x).split("/")` for
   the same inputs (the agreement win-04 relies on).

## Acceptance

- `pytest tests/test_path_norm.py -q` green; full `pytest -q` unchanged green.
- `mcp_connectors/path_norm.py` imports nothing outside the stdlib — prove it:
  `python -c "import ast,sys; …"` or simply read the import block. No
  `sys.platform`, no `os.sep`, no `os.path.normpath` in the file.
- **Nothing imports the new module yet**: `grep -rn "path_norm" src` is empty.
  If a `src/` file changed, this task overreached — win-03 does the wiring.
- Move this file to `todos/done/` when the above is true.
