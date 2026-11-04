# csl-04 — The driver seam, the honest refusal, and the `windows` extra

## Deliver this

The terminal stops asking *"is this POSIX?"* and starts asking *"is a driver
available?"* — the same refusal shape, one named seam, and **no Windows driver
implementation yet** (that is csl-06). After this task the behaviour on Linux is
unchanged, and the behaviour on Windows is still a refusal, but a refusal that
names the real reason.

Land:

1. `src/core/termhub/drivers.py` — the seam's **selection** question, answered by
   capability, never by `sys.platform` / `os.name`: *which driver can this app
   actually run right now?* On POSIX the answer is `pty_posix`. Anywhere else the
   answer is a driver only if its dependency is importable and the platform can
   host a pseudo-console; otherwise "none", and the reason is kept for the
   sentence.
2. The refusal path keeps its shape and its words change only to name the truth:
   `TerminalUnsupported` from `spawn()`, an `error` frame then
   `CLOSE_CODE_UNSUPPORTED` (4501) from the WebSocket route, and the existing
   `useTerminal.js` rendering of that sentence. **No frontend edit.**
3. `pyproject.toml` gains the one dependency of this roadmap, in its own extra:
   a `windows` extra carrying the pseudo-console driver, marked so a Linux
   install never resolves it. The suite stays green with it **absent**.
4. The sentence for "no driver" is written **once** (in `drivers.py` or
   `termhub/constants.py`, per the constants convention) and is what both
   `spawn()` and the route say. C-D3: *"A's test for that refusal … is not
   deleted by C — it becomes the test for 'driver not installed'."*

## Files it touches

| file | change |
|---|---|
| `src/core/termhub/drivers.py` | **new** — the availability question, the driver lookup, the refusal sentence |
| `src/core/termhub/session.py` | edit — the guard becomes "is a driver available?"; the guard's *reason text* comes from the seam |
| `src/api/routes.py` | edit — the platform guard before accept becomes the seam's answer; the close code and the accept/error-frame order stay as they are |
| `src/core/termhub/constants.py` | the close codes stay; add the refusal sentence and (if the seam needs it) a patchable "driver availability" attribute |
| `pyproject.toml` | edit — the `windows` optional extra with a `sys_platform == 'win32'` marker |
| `tests/test_terminal.py` | edit — the driver-absent refusal test (forced by hiding the driver), plus the existing tests kept intact |
| `web/src/composables/useTerminal.js` | **no change** (it already renders an `error` frame and the 4501 close) |

## Rules it must not break (quoted)

* roadmap §9 rule 2: *"Platform differences are selected by capability probes,
  not by `sys.platform`, wherever a probe is possible … Where it is not possible
  (job objects, ConPTY), the difference sits behind one named seam, and the
  *other* platform's tests keep running unchanged."*
* roadmap §9 rule 3: *"A refusal is a tested behaviour, not a fallback. Every
  'unsupported' path … gets a test that asserts the sentence and the status/close
  code, so B and C cannot turn a refusal into a crash without failing something
  on Linux."*
* C-D3's fallback paragraph: *"if `pywinpty` is absent on Windows, or the
  pseudo-console cannot be created, `TerminalUnsupported` and close code 4501
  still happen, with a sentence naming the missing dependency."*
* C §7 on the close codes: *"the three close codes and what each mirrors (4000-4999
  … 403 policy, 429 capacity, 501 platform) — C makes 4501 mean 'no driver',
  which is still the 501 case."* The origin policy in `termhub/policy.py` and the
  session cap keep their order and their codes: **origin first, then driver, then
  capacity**, exactly as the route's current guard order.
* C-D1's rejected alternatives are still rejected: raw ConPTY through `ctypes`,
  plain pipes, a WSL bridge. Do not implement a partial substitute to make the
  refusal go away.
* C §3: *"no frontend change in C."* The frame contract
  (`input`/`resize`/`kill` in, `open`/`output`/`exit`/`error` out) is not widened.
* AGENTS.md: the terminal is *"a human surface (`core/termhub/`, one WebSocket
  per session), never an agent tool"* — nothing in this task may make a shell
  reachable from chat, a hub or a tool.
* AGENTS.md: *"`.env` is regenerated from a fixed template on every save"* — this
  task adds no managed key and no settings-panel field.
* A test that forces "no driver" must **restore** the seam's state; it must not
  leave the driver unavailable for later tests in the same session.

## Depends on

csl-03 (the seam's POSIX side must exist before selection makes sense).

## How to tell it is done

1. `pytest tests/test_terminal.py` on Linux: every test that ran before still
   runs and passes, **zero skips**, and the new refusal test passes too.
2. The new refusal test exists and runs **on Linux**: with the driver made
   unavailable through the seam's own patchable attribute, the WebSocket answers
   an `error` frame whose message names what was looked for, then closes 4501 —
   and `GET /api/terminal` still reports no sessions (nothing was spawned).
3. `grep` in `src/core/termhub/` and the terminal route shows **no** remaining
   `os.name != "posix"` decision; the remaining platform answer is the seam's.
   (`proctree`'s POSIX/job dispatch belongs to csl-02/csl-07, not here.)
4. `pyproject.toml` has a `windows` extra whose dependency carries a
   `sys_platform == 'win32'` marker, and a Linux install/resolve never picks it
   up. Show the resolve in your *Results* note.
5. On a Windows runner the terminal still refuses, and the sentence a user sees
   names the **missing dependency**, not "POSIX-only". Paste that sentence into
   the *Results* note (if you are not on Windows, record the sentence the code
   produces for that branch and mark it as unverified-on-Windows).
6. `pytest tests/test_api.py tests/test_write_surface_isolation.py` green;
   `cd web && npm test` green; `git diff --stat web/` empty.

## Results (landed 2026-11-02, on the Windows runner)

**The seam as it landed.** `src/core/termhub/drivers.py` answers the selection
question: `selected_driver() -> TerminalDriver | None` — it asks each driver
module's own `available()` probe (`pty_posix.available()` was added to that
module as one function reporting its runtime-import verdict; the Windows half
is looked up by name, `drivers.WINDOWS_DRIVER_MODULE = "pty_conpty"`, and an
`ImportError` or a dependency-load `OSError` there is "no driver", never a
crash) — and `driver_available() -> bool` for the route. `TerminalPty` /
`TerminalDriver` are `typing.Protocol`s pinning csl-03's landed surface —
csl-06 implements exactly that, and `session.py`'s `self.pty` is now typed
`drivers.TerminalPty`. `spawn()` asks at spawn time and raises
`TerminalUnsupported(constants.NO_DRIVER_MESSAGE)`.

**The sentence, written once** (`termhub.constants.NO_DRIVER_MESSAGE`, 103
bytes — inside the 123-byte close-reason limit):

> The terminal has no pseudo-console driver available (on Windows: install the
> 'windows' extra, pywinpty)

Verified **on this Windows runner** (criterion 5): with no `pywinpty` and no
`pty_conpty` installed, the WebSocket answers one `error` frame with exactly
that message, then closes **4501** with the same sentence as the reason;
`GET /api/terminal` still reports `{"sessions": []}` — that is
`TestDriverSeam::test_websocket_answers_the_sentence_then_closes_4501`, green
here. The refusal is now a *tested* behaviour on both platforms, forced
through `termhub.constants.DRIVER_ABSENT_FORCED` (the seam's patch-point;
`monkeypatch` restores it so later tests keep their driver).
`test_the_driver_guard_comes_before_the_capacity_guard` pins C §7's order
(origin → driver → capacity): with the cap forced to 0 the answer is still
4501, never 4429. `test_hiding_a_driver_probe_takes_that_driver_out_of_the_seam`
proves selection follows the probe, not a platform name, in both directions.

**Criterion 3:** `grep os.name` over `termhub/session.py`, `termhub/drivers.py`
and the terminal route leaves exactly one — `session.py`'s teardown guard
`os.name == "posix"` around `kill_session_tree`, which csl-07 owns and removes.
Both `os.name != "posix"` decisions (route and `spawn()`) are gone.

**Criterion 4 (the resolve):** `pip install --dry-run ".[windows]"` on this
win32 venv resolves `pywinpty-3.1.2`; `packaging.markers.Marker("sys_platform
== 'win32'")` evaluates **False** against a Linux environment dict, so a Linux
install never picks it up.

**Linux promise (criterion 1)** — *not re-runnable from this Windows machine*;
what ran instead: `pytest tests/test_terminal.py --collect-only` shows the 12
pre-existing tests (unchanged, unedited) + the 4 new seam tests; the 4 new
tests pass on this Windows runner; `TestSession` keeps its **pre-existing**
5 skips (`no shell at /bin/sh to spawn` — conftest's probe, re-gated by
csl-08); `TestGuards`' spawn-opening tests and `TestRegistry` keep the
pre-existing Windows hang documented by csl-02/csl-03's Results (the wedge was
the accept-then-close refusal on a driverless Windows; it still refuses, now
via the seam, and csl-06 removes the condition). The pump/decoder/framing path
was additionally driven end-to-end through a **fake driver planted in the
Windows slot** (scratch run, not a suite file): selection picked it, output
frames came per chunk with a multi-byte character joined across chunks, the
driver's `FarEndGone` ended the pump, `teardown()` closed the master — csl-06's
real driver will satisfy exactly this surface.

**Regressions:** `tests/test_write_surface_isolation.py` 20 passed.
`tests/test_api.py` 63 passed / 2 failed — both **pre-existing** and outside
this task's diff: the `NameError: helpers_block` at `routes.py:1418` (the
mid-merge checkout, already recorded by csl-03), and
`test_the_deny_list_compare_folds_for_the_root_being_switched_to`, which fails
because this machine's NTFS folds the *non-existent* spelling at the OS level
(`os.path.realpath` of `…\Project` returns the created `…\project`), so both
posts answer 422 — a filesystem fact of this runner, not a terminal change.
`cd web && npm test` 472 passed; `git diff --stat web/` empty.
