# pol-08 — The connectors honour the workspace policy (and keep their own floor)

Give `fedit` and `console` the same answer the Editor pane gives, so a folder the
user marked `ro` is uneditable by an agent too — while the connectors' own
secret floor stays exactly where it is, for the paths outside the root where it
is the only thing that means anything.

Depends on: pol-02 (the matcher, in `mcp_connectors/`), pol-04 (the parser,
likewise), pol-05 (a workspace that reliably *has* a policy file).

## 1. What the connectors actually do today (read this before editing)

The floor is **not** applied inside the root:

* `fedit_mcp.py:895` — `if _within_root(real) and not _system_prefix_hit(absolute):
  return absolute`, with the reasoning at `:876-894`: *"Inside the root the policy
  is bypassed; the operator chose that tree."* The address is still checked
  against system prefixes, and a symlink out of the root is judged by its target.
* `console_mcp.py:505-507` — the same carve-out for the token scan.

And `--root-dir` is `${BENCHMARKIT_PATH_LOCATION}` in both `local` entries of
`externals.json`, i.e. the workspace. So **today an agent in this session can read
and edit the app's own `.env`, while the Editor pane cannot read it.** The pane is
the stricter surface. This task narrows the agent, in the workspace only; it adds
no new power anywhere and removes none.

Keep that sentence in the module docstrings of both connectors and in
`path_policy.py`. The "one list, both connectors" symmetry the file defends
(`path_policy.py:6-9`) survives *within* the connector pair; what changes is that
the browser now answers to a different, user-authored question. Say so, or the
next contributor "restores" the shared import pol-06 removed.

## 2. The one question, asked in the connector's own process

Both connectors already resolve to an absolute path before judging it. Add, after
the root bypass and before returning:

```python
def workspace_policy_reason(path: Path, *, for_write: bool) -> Optional[str]:
    """The workspace policy's refusal for `path`, or None.

    Lives in ``mcp_connectors/policy_file.py`` next to the parser: stdlib-only,
    no ``src`` import, using ``policy_glob.pattern_matches``.
    """
```

Steps, each of which is why it cannot be shorter:

1. Make `path` workspace-relative against `ROOT_DIR` (the connector's root *is*
   the workspace; `Path.relative_to` after a `resolve()`, or None when the path is
   outside the root — outside the root the workspace policy has nothing to say and
   the existing deny-list already spoke).
2. `.nojae` in any component → **refused, always** (`policy_shared.POLICY_DIRNAME`
   + `reserved_hit`). This is the one rule that must survive a corrupt file: an
   agent rewriting the policy that constrains it is the same accident
   `DENY_COMPONENTS` includes `mcp_connectors` to prevent (`path_policy.py:56-58`).
3. `load_policy_text()` on the file; **errors are returned as a refusal for
   writes** (the connector's equivalent of pol-04's `policy_blocking_error`) and
   ignored for reads, matching the app's D4 rather than inventing a second rule.
4. Level **B only**. Level A (`is_self_workspace()`, `.env**`, `logs/**`) is not
   applied here: it exists because the *pane* fights the settings dialog over a
   generated file, which is not a property of an agent's edit. Note that in the
   docstring — a reader will otherwise ask why the app has a rule the connector
   skips.
5. Verdict: `hidden` → refused for reads **and** writes; `ro` → refused for
   writes. The `hidden`-refuses-reads half is what makes `**/__pycache__`
   unsearchable-by-agent, so the message must name the pattern, and the connector
   tells the model about the *rule* so it can ask differently instead of giving up
   (the same reason `edit_policy()` exists).

Wire it in:

* **fedit** — inside `check_read_path` / `check_write_path` (the two callers of the
  resolution at `:912` / the write twin). The error type stays `PathDenied` with
  the same "call `edit_policy()` for the full list" tail.
* **console** — inside `protected_path_error` (`console_mcp.py:483`) for every path
  token, and its **write-side** counterpart where one exists. Console executes one
  argv with no shell, so a token scan is the whole check — the same trust model
  the C-6 comment at `:484-491` describes.

## 3. `edit_policy()` must stop lying

The tool currently prints `inside the root : no policy restrictions (paths fully
bypass the lists below)` (`fedit_mcp.py:1456`). After this task that is false.
Rewrite the line, and add the policy block:

```
  root dir        : /home/vgr/anotherproject
  inside the root : the workspace policy of that folder applies (see below);
                    the connector deny-list does not
  policy file     : /home/vgr/anotherproject/.nojae/read-write-policy.json (3 rules)
                    hidden: .git, **/__pycache__
                    ro:     **/node_modules
```

Three lines, same format, no JSON dump — the model acts on this text, so the
*shape* matters more than completeness, and the truncation must be labelled
(`+7 more`). A tool that reports a policy it does not enforce is worse than one
that reports nothing.

## 4. Refresh, and the hot-switch guarantee

A `POST /api/path-location` restarts both hubs (`routes.py:1221-1222`), so each
connector process is relaunched with the new `--root-dir` and re-reads its policy
at startup. Nothing to build — but assert it: the cache in the connector process
is per-process, keyed on `ROOT_DIR`, with `(mtime_ns, size)` of the file, and
pol-04's rule about the root being in the key applies here too (the value is
frozen per process, but the file can change while the process lives).

Do **not** add a `/api/fs`-style IPC to the app for this: connectors must keep
working when the app is not the one that launched their root.

## 5. Tests

Extend `tests/test_fedit_policy.py` and `tests/test_console_policy.py` — they
already drive the real connector functions, which is why this task is cheap:

1. a `tmp_path` workspace with a policy marking `secret/` `ro`: `apply_edit` /
   `preview_edit` inside it → `PathDenied` naming `secret/`; reading the same file
   → allowed;
2. the same, marked `hidden`: the read is refused too, with the pattern in the
   message;
3. carve-out: `**/node_modules` `ro` + `**/node_modules/**/README.md` `rw` → the
   README edits;
4. `.nojae/read-write-policy.json` → refused for read and write, and the refusal
   stands with a deliberately corrupt policy file in place (the reserved rule does
   not depend on the parse);
5. a corrupt policy file → writes inside the root refused with the parse message,
   reads inside the root still allowed;
6. the same paths **outside** the root: the workspace policy says nothing, the
   deny-list answer is unchanged (pin the two floors' domains apart);
7. `edit_policy()` output contains the rule list and no longer contains "no
   policy restrictions";
8. `console`: `cat sub/.env` where the policy hides `**/.env` → refused; where no
   policy mentions it → allowed inside the root, refused outside it. That last
   pair *documents the root asymmetry of §1* rather than hiding it — if pol-03's
   level A is ever applied to connectors, this test is what changes, deliberately.

Also extend `tests/test_write_surface_isolation.py` **without weakening it**: the
new connector refusal path must not become a way for a connector to reach a
`/api/fs` route. The assertion list gains the two new connector functions and
still asserts zero write routes reachable from them.

## 6. Acceptance

* Mark a folder `ro` in this repo's own policy; in WorkBench, ask the model to
  edit a file inside it. The refusal names the rule and the file. Reading it still
  works.
* `fedit`'s `edit_policy()` matches what the Editor's panel (pol-07) shows for the
  same workspace, in the same words for the same rule.
* `pytest` green, both connector suites included, and
  `tests/test_write_surface_isolation.py` unchanged in strictness.
