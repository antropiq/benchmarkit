# fedit-mcp — issues found by live probing

Findings from a hands-on test session against the running `fedit` connector
(`mcp_connectors/fedit_mcp.py`, started with `--root-dir /mnt/bigfoot/dev/benchmarkit`).
Every item below was reproduced through the actual MCP tools, not read off the
source. Line numbers refer to the version probed.

Nothing here proposes loosening the path policy: the deny-lists were probed and
behave correctly (`~/.ssh`, `/etc/shadow`, a symlink pointing at `/etc/shadow`,
and `.env` reached through a non-root alias are all refused for read,
`preview_edit` and `apply_edit` alike; the symlink write is refused too). The
problems below are round-trip correctness, encoding, and error messages that
block ordinary work.

Legend: **[blocks me]** = hit while doing real work in this project.

---

## Status: all 13 items fixed

Every item below is implemented in `mcp_connectors/fedit_mcp.py`, each with a
test in `tests/test_fedit_policy.py` (31 tests now, up from 11). The connector
was re-probed live after each change, the same way this report was written.

| # | How it was settled |
|---|--------------------|
| F-1 | `write_atomic_bytes()` reads the target's mode and `chmod`s the temp before `os.replace`. |
| F-2 | `_LINE_END_RE` (`\r\n \| \n \| \r`) replaces `splitlines()`; `detect_terminator()` reuses the dominant style, `join_text()` writes it; mixed files are announced as a warning. |
| F-3 | `revert_edit` restores **bytes**; `read_file` shows non-UTF-8 with `encoding: NOT valid UTF-8` in the header; `apply_edit`/`preview_edit`/`make_diff` refuse with the offending byte and offset. |
| F-4 | `parse_search_replace` tokenises the whole payload and refuses unless it reads as whole SEARCH/DIVIDER/REPLACE triples — no more edit at the wrong place. |
| F-5 | `preview_edit`/`apply_edit` docstrings now say to pass `format="diff"` for marker-bearing files; an applied edit to such a file also *warns*, naming the same route. |
| F-6 | `read_file` answers `lines 0-0 of 0` for an empty file. |
| F-7 | `FUZZ_CEILING = 3` clamps both the default and a per-call request, and a failed hunk says its fuzz was reduced rather than inviting a bigger one. |
| F-8 | The blanket `'..'` refusal is gone: paths are normalised and *both* the normalised and the realpath form are held to the deny rules, so refusal follows where a path lands. |
| F-9 | The root bypass and the backup digest both key on the resolved path (`backup_key()`), so two spellings of one tree get one verdict and one backup chain. |
| F-10 | A missing parent says "does not exist. Create it first with console's mkdir". |
| F-11 | Over-cap answers end with the console route that still works (`sed -n 'X,Yp'`, `grep -n`). |
| F-12 | `binary_reason()` / `undecodable_reason()` sniff the whole payload (NUL anywhere, not just 8 KB) and answer `<not editable text>`; a UTF-8 BOM is stripped before matching and re-attached on write, so a copied line 1 round-trips. |
| F-13 | `prune_old_backups()` sweeps `*.orig` past `FEDIT_BACKUP_PRUNE_DAYS` (default 30) from the write path; `edit_policy()` and the "no backup" message state the window. |

One extra finding from the fixing session, folded in here because it is the same
class of bug: with `format="auto"`, a payload carrying *both* a `<<<<<<< SEARCH`
marker and an `@@ -l,s +l,s @@` header used to be routed to whichever engine is
checked first — so a diff quoting a SEARCH marker (editing this connector's own
docstrings) was sent to the SEARCH engine. `resolve_format()` now treats a
leading `--- ` / `+++ ` pair as decisive for diff, and refuses with
"Ambiguous edit: pass format explicitly" when both shapes match without it.

Also fixed while in there: `make_backup()` used `shutil.copy2`, which copies the
*source's* mtime onto the backup — so the F-13 sweep would have deleted a fresh
backup of a long-untouched file on sight. The backup is re-stamped with the time
of the edit.

The live re-probe used `logs/probe2/probe.py`; the `logs/probe/` kit below is
left as-is (its files are still 0600 from F-1).

---

## **DONE**: F-1. `apply_edit` silently changes the file mode to 0600 **[blocks me]**

Repro (all inside the root, so no policy involved):

    chmod 755 logs/probe/script.sh     ->  stat: 755
    apply_edit(script.sh, ...)         ->  stat: 600
    chmod 444 logs/probe/readonly.txt  ->  stat: 444
    apply_edit(readonly.txt, ...)      ->  stat: 600

Every file touched by an edit ends up `0600`: `plain.txt` 644 -> 600,
`script.sh` 755 -> 600, `readonly.txt` 444 -> 600.

Root cause: `write_atomic()` (l.774-788) creates the temp with
`tempfile.mkstemp()` (mode 0600) and `os.replace()`s it over the target, so the
temp's mode becomes the file's mode.

Why it matters: an edited shell script or any executable the repo runs loses
its `+x`; a `444` file becomes writable. It is invisible in the diff the tool
returns, so nobody notices until `git diff` shows `old mode 100755 / new mode
100644` or the script stops running.

Suggested fix: capture the mode before the replace and re-apply it to the temp.

    def write_atomic(resolved: Path, content: str) -> None:
        mode = None
        try:
            mode = resolved.stat().st_mode & 0o7777
        except OSError:
            pass
        handle, tmp_name = tempfile.mkstemp(...)
        try:
            ...write...
            if mode is not None:
                os.chmod(tmp_name, mode)
            os.replace(tmp_name, resolved)

Add a test in `tests/test_fedit_policy.py`: 755 and 444 both survive an edit.

---

## **DONE**: F-2. CRLF files are rewritten as LF, silently **[blocks me]**

Repro:

    printf 'one\r\ntwo\r\nthree\r\n' > crlf.txt   # 17 bytes, all CRLF
    read_file(crlf.txt)      ->  "one / two / three"   (the CR is invisible)
    apply_edit(crlf.txt, ...) ->  "Applied ... 3 -> 3 line(s)"
    xxd crlf.txt             ->  6f6e650a 54574f0a ...  (every CR gone, 14 bytes)

The returned diff looks like a two-line change; the real change is a rewrite of
every line ending. That poisons `git diff` (whole-file churn) and breaks
anything relying on CRLF.

Root cause: `split_text()` (l.191-193) uses `str.splitlines()`, which splits on
`\r\n` and discards the terminator, while `join_text()` (l.196-200) always
joins with `"\n"`. Only the *final* newline is modelled (`final_newline`), not
the terminator style.

Second face of the same bug: `splitlines()` also splits on `\x0b \x0c \x1c-\x1e
\x85 \u2028 \u2029`, so a file containing a form feed gets its line structure
changed on write. Verified:

    printf 'page one\n\x0cpage two\nend\n'   (3 real lines, 23 bytes)
    read_file ->  header claims "lines 1-4 of 4"  (it invents a blank line 2)
    apply_edit ->  bytes: page one \n \n PAGE TWO \n end \n   (the \x0c is gone)

Same for `\u2028`. Line counts reported by `read_file` therefore do not match
`wc -l` for those files.

Suggested fix: detect the terminator once and reuse it.

    def split_text(text):
        terminator = "\r\n" if text.splitlines()[:1] and "\r\n" in text else "\n"
        ...

At minimum: keep `\r\n` when the first line ends with `\r\n`, and restrict
splitting to `\r\n | \n | \r` so exotic separators stay inside the line text.

---

## F-3. Any non-UTF-8 file is unreachable — read, diff AND revert

    read_file(logs/probe/latin1.dat)
      -> Error: 'utf-8' codec can't decode byte 0xe9 in position 3: invalid continuation byte
    make_diff(nul.dat, latin1.dat)      ->  same error
    apply_edit(latin1.dat, ...)         ->  same error

Root cause: hardcoded `encoding="utf-8"` at l.819 (`read_file`), l.856
(`preview_edit`), l.935 (`apply_edit`), l.1017 (`revert_edit`), l.1049-1050
(`make_diff`).

`revert_edit` is the worst case: a backup of a Latin-1 file cannot be restored,
because l.1025 catches `UnicodeDecodeError` and returns an error. The backup was
made with `shutil.copy2` (a faithful byte copy) but is unusable through the
connector, so *recovery* becomes a dead end exactly when you need it.

Suggested fix, in escalating order:

1. `revert_edit` should restore **bytes** (`read_bytes` + a byte-level atomic
   write). A revert must never depend on decoding.
2. `read_file` / `make_diff` may decode with `errors="replace"` for display and
   flag it in the header (`encoding: not valid UTF-8`) so the agent knows the
   read is not authoritative.
3. `apply_edit` should refuse non-UTF-8 with an actionable message instead of a
   codec exception: "this file is not valid UTF-8 (byte 0xe9 at offset 3); fedit
   edits text files only — inspect/modify it through console".

---

## F-4. A file containing a `=======` line cannot be SEARCH/REPLACEd — and the tool edits the WRONG place instead of failing

This is the dangerous one, not merely a blocker. Repro:

    cat logs/probe/marker.txt
        Some doc text
        =======
        more text

    apply_edit(marker.txt, format="search_replace", SEARCH="Some doc text")
      -> "Applied to ...: 3 -> 7 line(s)"      # no error at all

The `=======` inside my SEARCH text was re-tokenised as the block divider, so
the payload was parsed as several blocks and the file came out scrambled. No
error, no warning: silent corruption.

Root cause: `is_marker()` (l.446-447) accepts any line equal to, or starting
with, `=======` as the divider and `parse_search_replace()` (l.457) stops
collecting SEARCH lines at the first one. There is no way to express a literal
divider.

Real exposure in this repo:

* **git conflict markers** — `<<<<<<< HEAD`, `=======`, `>>>>>>> branch`. Probed:
  an edit of a conflicted file applied, but its hunk boundaries came from the
  conflict markers, not from what I asked for.
* `AGENTS.md`, `mcp_connectors/fedit_mcp.py` and this very report document these
  markers, so editing documentation about the edit format with the edit format is
  a trap. I hit it writing this file: my first attempt to create
  `todos/fedit-issues.md` with SEARCH/REPLACE failed with "SEARCH block #1 has
  no '>>>>>>> REPLACE' terminator" because the body contained a `=======` line.

Suggested fix (cheapest first):

* Reject rather than guess. If a collected SEARCH or REPLACE body contains a line
  that is exactly a divider/marker, raise:
  `FeditError("the edit text contains a '=======' line; this file must be edited
  with format='diff'")`. Honest, and it routes the agent to the format that
  already handles this content correctly (see F-5).
* Or make the divider unambiguous (require the divider count to equal the SEARCH
  marker count before applying anything).

---

## F-5. The diff format handles conflict markers correctly, but nothing points the agent there

`apply_edit(hunkish.txt, format="diff")` on a file whose first line is literally
`@@ -1,2 +1,2 @@` applied correctly, and appends to an empty file work too (that
is how this report was written). So the diff engine is the safe path for
marker-bearing files. The gap is only guidance: both `preview_edit` (l.846) and
`apply_edit` (l.919) docstrings present SEARCH/REPLACE first, which is the
format that breaks.

Suggested fix: one line in both docstrings — "if the file contains `=======`,
`<<<<<<<` or `>>>>>>>` lines (git conflicts, or this connector's own
documentation), pass `format="diff"`; SEARCH/REPLACE cannot express them."

---

## F-6. `read_file` on an empty file is an error, not a read

    read_file(logs/probe/empty.txt)
      -> Error: start_line 1 is past the end of a 0-line file

Root cause: l.825-826, `if start > len(lines)` fires for `1 > 0`. The agent then
believes the path is unreadable rather than "this file is empty", and it becomes
hard to reach `apply_edit`'s legitimate empty-SEARCH-to-create path (l.493-500)
because you cannot first confirm the file is empty.

Suggested fix: special-case `len(lines) == 0` and return the header with an empty
body, e.g. `File: ... lines 0-0 of 0  0 bytes  EOF: no final newline`.

---

## F-7. `fuzz` is documented as `0..3` but is never clamped

Docstrings say `fuzz: Max context lines a hunk may ignore when repositioning
(0..3)` (l.846, l.919). The implementation is

    max_fuzz = MAX_FUZZ if fuzz is None else max(0, int(fuzz))     # l.342

— no upper bound. `fuzz=10` was accepted and applied a hunk against a 2-line
file while ignoring almost all of its context.

Why it matters: a large fuzz turns "reposition a hunk when the file shifted a
little" into "match this anywhere", which is how an edit lands in the wrong
function of a large file.

Suggested fix: `min(3, max(0, int(fuzz)))` at l.342, or drop the `(0..3)` from
the docstrings and let `edit_policy()` report the effective range. Clamping is
safer.

---

## F-8. `..` is refused outside the root even when the target stays inside the root

    read_file("/mnt/bigfoot/dev/benchmarkit/logs/probe/../../src/core/job.py")
      -> works (the root bypass at l.672 fires first)
    read_file("<non-root spelling>/../inside")
      -> FeditError: path ... contains '..'          (l.676)

The refusal is a real security control (it blocks `~/.ssh/../../etc` smuggling —
comment at l.654-658), but because it runs *after* the root bypass, identical
logical paths behave differently depending on how they were spelled.

Suggested fix: normalise with `os.path.normpath()` before the `'..' in parts`
test, and re-check the normpath as well as the abspath and the realpath against
every deny rule, instead of blanket-refusing the token. The three-way check is
already half there (l.680-687 loops over literal and real).

---

## F-9. Two spellings of the same file get different policies and separate backup chains **[blocks me]**

The workspace is reachable through two prefixes: `/mnt/bigfoot/dev/benchmarkit`
and `/home/vgr/dev/benchmarkit` are the same tree (`stat` reports
`dev=66309 ino=46137628` for both; `/mnt/bigfoot` is a mount point). The
connector's root is the `/mnt/...` spelling, so:

    read_file("/mnt/bigfoot/dev/benchmarkit/.env")    -> works (root bypass)
    read_file("/home/vgr/dev/benchmarkit/.env")       -> refused: "a secrets/config file"

    apply_edit("/home/vgr/dev/benchmarkit/logs/probe/plain.txt", ...) -> applied, backup written
    list_edits("/mnt/bigfoot/dev/benchmarkit/logs/probe/plain.txt")
      -> "No backup for ... (it has not been edited by this server)"

The second line is the practical damage: the same physical file is editable
under one name and protected under the other, and `list_edits`/`revert_edit`
lose the backup chain as soon as the agent uses the other spelling — which reads
as "this file was never edited" and destroys the recovery path.

Root cause: `check_target_path()` compares literal strings against `ROOT_DIR`
(l.672 via `_within_root`, l.582-586) and `backup_path_for()` hashes the literal
path (l.747-751, used again by `list_backups`, l.754-758).

Suggested fix: canonicalise once. Resolve the root at startup
(`ROOT_DIR = Path(os.path.realpath(args.root_dir))`, l.356 area) and compare the
already-computed `absolute.resolve()` (l.667) rather than the literal; hash
`real` instead of `resolved` in `backup_path_for()`/`list_backups()`. Keep the
literal check as well, so a symlink still cannot smuggle an outside path into the
bypass (that is what the pair at l.672 buys today).

---

## F-10. A missing parent directory is reported as a permission problem

    apply_edit("logs/probe/nosuchdir/deep.txt", empty SEARCH)
      -> Error: Directory .../logs/probe/nosuchdir is not writable

The directory does not exist; `os.access(parent, W_OK)` (l.714) answers "not
writable". The agent reads that as "I am not allowed" and either gives up or
reaches for `chmod`, instead of running `mkdir`. The behaviour is deliberate
(comment l.708-709: creating trees is console's job), but the message hides the
intent.

Suggested fix:

    if not resolved.parent.is_dir():
        raise FeditError(
            f"{resolved.parent} does not exist. Create it first with console's "
            f"mkdir; fedit only writes files into existing directories.")

---

## F-11. `MAX_FILE_BYTES` refuses with a dead end

    read_file(logs/probe/huge.bin)
      -> Error: .../huge.bin is 2500000 bytes, over the 2000000 limit

The files in this repo above the cap are all vendored artefacts
(`web/node_modules/**`: `typescript.js` 9.1 MB, `mermaid.js.map` 19 MB,
`.vite/deps/*`), so this rarely bites in practice — but when it does the message
offers no route forward even though `read_file` is already paginated.

Suggested fix: honour an explicit `start_line`/`max_lines` window past the cap
(reject only whole-file reads and edits), or end the message with "inspect a
slice with console: sed -n '10,40p' <path>".

---

## F-12. Binary content is displayed as if it were text

    printf 'a\x00b\n' > nul.dat
    read_file(nul.dat)          ->  "     1\ta b"      (the NUL became a space)
    apply_edit(nul.dat, SEARCH="a b")
      -> Error: SEARCH text not found ... closest content there: | a b

The tool shows the line, asks the agent to copy it verbatim, and then rejects
the verbatim copy — an unwinnable loop. It also invites editing binary files
that should be refused outright.

Suggested fix: sniff the first 8 KB for NUL bytes / invalid UTF-8 and answer
`<binary: not a text file, fedit cannot edit it>`, mirroring `file(1)`. This
also gives F-3's message something concrete to say.

---

## F-13. Minor: the backup directory is never pruned across files

`logs/fedit_backups/` held 537 `.orig` files at probe time. `make_backup()`
prunes to `BACKUP_KEEP` per file digest (l.766) which is correct, but nothing
collects backups of files that no longer exist, so a long-lived workspace
accumulates them forever. Not blocking, but worth an age-based sweep.

---

## What was probed and behaves correctly — do not "fix" these

Recorded so the next session does not mistake these for bugs:

* Deny-lists hold for every entry point. `read_file`, `preview_edit`,
  `apply_edit`, `make_diff` and `list_edits` all refuse `/etc/shadow`, `~/.ssh`,
  `~/.config/gh`, `.env` by name and `*.pem`/`*.key` by suffix. `preview_edit` is
  never more permissive than `apply_edit` (the issue #1 asymmetry stays closed).
* Reads are policy-checked *before* existence: asking for a file inside `~/.ssh`
  returns a denial, not "not found", so the store's contents are not disclosed.
* Symlink escapes are caught on both spellings: `logs/probe/shadowlink.txt ->
  /etc/shadow` is refused for read and for write, naming the real target.
* A symlink pointing at a benign outside file (`/etc/hostname`) is readable but
  never writable — write is refused with the real target in the message.
* Empty SEARCH on an existing non-empty file is refused (l.494-498); an
  ambiguous SEARCH (several matches) is refused with the list of lines.
* Diff application is atomic — one bad hunk writes nothing (l.369-370).
* Tolerating wrong `@@` counts and repositioning a mis-numbered hunk both work
  and emit an explicit warning naming where the hunk actually landed. Probed with
  deliberately wrong counts: the body won, as documented.
* Preserving "no final newline" and trailing whitespace across an edit both work
  (`noeol.txt`, `trail.txt`).
* `check_write_path` refusing to create directory trees is intentional; only the
  wording is the problem (F-10).

---

## Suggested fix order

| # | Issue | Risk if unfixed | Effort |
|---|-------|-----------------|--------|
| F-4 | `=======` in the file => wrong edit applied, silently | data corruption | S |
| F-1 | every edit leaves mode 0600 | breaks scripts, git mode churn | S |
| F-2 | CRLF -> LF and exotic line splits | whole-file rewrites | M |
| F-9 | alias path: split policy, lost backup chain | confusing denials, no revert | S |
| F-3 | non-UTF-8 unreadable and unrevertable | dead ends, lost recovery | M |
| F-7 | unclamped `fuzz` | edit applied in the wrong place | XS |
| F-12 | binary shown as text | invites corruption | S |
| F-6 | empty file read is an error | friction | XS |
| F-10 | missing dir reported as "not writable" | misleads the agent | XS |
| F-5 | docstring: use diff for conflict markers | leads into F-4 | XS |
| F-8 | `..` refused outside the root | friction | M |
| F-11 | size cap with no route forward | friction | S |
| F-13 | backup dir never pruned | slow disk creep | S |

Each fix should land with a test in `tests/test_fedit_policy.py` (11 tests
today). F-1, F-2, F-9 and F-12 are round-trip properties, so they fit that file's
existing "one tool must never be lossier than another" style.

---

## Repro kit used

All probes lived in `logs/probe/` (inside the root, so policy never interfered)
and were left in place for re-running. One caveat for the fixing session: because
of F-1 every probe file now has mode 0600, so recreate the 755 and 444 cases with
`chmod` before re-testing F-1.

    plain.txt      644 text, edited through both path spellings (F-9)
    crlf.txt       was CRLF, now LF (F-2)
    noeol.txt      no final newline (behaves correctly)
    trail.txt      trailing spaces (behaves correctly)
    formfeed.txt   contains \x0c (F-2, line count inflated)
    bom.txt        UTF-8 BOM, blocks a verbatim SEARCH (see F-12 note)
    latin1.dat     invalid UTF-8 (F-3)
    nul.dat        contains a NUL (F-12)
    huge.bin       2.5 MB, over MAX_FILE_BYTES (F-11)
    marker.txt     contains a ======= line (F-4, left scrambled on purpose)
    conflict.txt   git conflict markers (F-4)
    hunkish.txt    first line is a @@ header (F-5, diff format handled it)
    empty.txt      0 bytes (F-6)
    script.sh      was 755, now 600 (F-1)
    readonly.txt   was 444, now 600 (F-1)
    etclink.txt    symlink -> /etc/hostname (write refused, correct)
    shadowlink.txt symlink -> /etc/shadow (read+write refused, correct)

One extra symptom worth folding into F-12: a file with a UTF-8 BOM
(`bom.txt`) cannot be edited with a SEARCH block copied from `read_file`,
because `read_file` shows `line1` while the stored first line is
`\ufeffline1`; the "closest content" hint does show the difference, so the agent
can work it out, but the fix is the same sniff-and-report approach.

