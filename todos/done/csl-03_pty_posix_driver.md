# csl-03 — `termhub/pty_posix.py`: the PTY moves out of `session.py`, unchanged

## Deliver this

A **pure refactor**, landed before any Windows code exists, so that csl-04 and
csl-06 can add a second driver without touching the one that works.

New module `src/core/termhub/pty_posix.py` owns the POSIX terminal mechanics that
today sit inline in `src/core/termhub/session.py`:

* opening the terminal pair (today's `pty.openpty()` in `spawn()`);
* the window-size ioctls (today's `_set_window_size` / `TIOCSWINSZ`, and
  `terminal_size()` / `TIOCGWINSZ`);
* the byte read and byte write against the master;
* the **slave-close rule** — *"while any process holds the slave open, a read of
  the master never reports the shell's death"* — and *why* it must happen in the
  spawning thread and right after the spawn;
* the `EIO` translation: an `OSError` from the read becomes the seam's answer to
  **"is the far end gone?"**, while any other error stays the error it was.

`session.py` keeps *everything else exactly as it is*: `spawn` / `pump` / `write`
/ `resize` / `terminal_size` / `teardown` / `run` / `status`, the `PTY_READ_BYTES`
framing, the incremental UTF-8 decoder, the publish-failure-means-client-left
path, `_finish_pump`'s fd-reuse reasoning, the shielded teardown and its comment,
`_reap_leader`, the session-cap behaviour and the close codes. It also keeps the
`os.name != "posix"` guard **for now** — replacing that guard is csl-04's job, and
doing it here would mix a behaviour change into a refactor.

## Files it touches

| file | change |
|---|---|
| `src/core/termhub/pty_posix.py` | **new** — the mechanics above, with the rationale comments moved with them |
| `src/core/termhub/session.py` | edit — asks the driver instead of calling `pty`/`fcntl`/`termios`; the `ImportError → None` import block moves out with the code it guarded |
| `src/core/termhub/constants.py` | only if a mechanic genuinely needs a new tunable — the constants-module convention stays (*"a test that exercises a cap or a grace period patches it here"*); no new constants without a test that patches them |
| `tests/test_terminal.py` | edit **only** where a test reached into a moved private detail; no assertion may be weakened |
| `tests/test_api.py` | untouched |

## The rule that decides every design choice here

**No observable behaviour change on POSIX.** Specifically:

* the frame sequence a client sees (`open` → `output`* → `exit`) is identical in
  shape, including the per-chunk framing;
* an `OSError` from the read still ends the pump and still leads to the reaped
  exit code being reported, unless `_closed` or `_transport_gone` — the current
  condition stays;
* `teardown` still runs `proctree.kill_session_tree` under
  `os.name == "posix"` with `STOP_GRACE_SECONDS` and `_reap_leader()`, in the same
  order (kill → drain the reader → close the master);
* the master is never closed while a read is blocked in a worker thread;
* the driver is a **mechanics** layer only: it decides no policy, owns no process,
  spawns no child and knows nothing about WebSockets or frames.

## Rules it must not break (quoted)

* C-D1, the reason for the seam: *"keep `termhub/session.py` as the single session
  class and the single frame contract, and put the pseudo-terminal behind one
  seam … The seam is deliberately tiny — open, read, write, set_size, close —
  because the valuable part of `session.py` is not the PTY, it is everything
  around it."*
* C §7: *"the frame contract (`input`/`resize`/`kill` ↔ `output`/`exit`), the
  per-`os.read`-chunk framing …, the incremental decoder that never splits a
  multi-byte character, the four-session cap, the three close codes … and the
  rule that a terminal session is a process tree, never a wrapper."*
* C §6, the Linux regression rule, verbatim: *"The POSIX terminal tests must not
  gain a single skip: if moving code into `pty_posix.py` costs a test its reach,
  the move is wrong."*
* C-D3's shutdown reasoning must stay written down somewhere and stay true:
  *"it is unblocked by the teardown, because once the shell and its children are
  gone the master answers `EIO`."*
* AGENTS.md: *"Core owns logic, the API owns transport"*; core takes an injected
  `emit` and never imports the API layer.
* AGENTS.md style: explicit annotations and `typing` imports; the module's
  tunables are patchable module attributes (the `constants.py` convention).

## Depends on

Nothing. Independent of csl-01 and csl-02, and it must land **before csl-04**.

## How to tell it is done

1. On Linux, **all twelve** tests in `tests/test_terminal.py` run and pass — the
   count is unchanged and none is marked skipped. (`pytest tests/test_terminal.py
   -rs` shows no skip reason.)
2. `pytest tests/test_api.py tests/test_write_surface_isolation.py` green.
3. `src/core/termhub/session.py` no longer imports `pty`, `fcntl` or `termios`
   (nor the `TYPE_CHECKING` fallback block), and contains no `TIOCSWINSZ` /
   `TIOCGWINSZ` / `openpty` call.
4. `tests/test_terminal.py`'s `test_resize_updates_the_window_size`,
   `test_exit_frame_reports_the_code`, `test_kill_frame_closes_session_and_kills_pid`
   and `test_disconnect_tears_down_the_whole_session_tree` are **unmodified in
   what they assert** (a diff may show a moved helper at most).
5. `cd web && npm test` green (it must be — nothing in `web/` was touched;
   `git diff --stat web/` is empty).
6. Append a *Results* note naming the seam's final function/attribute list, so
   csl-04 and csl-06 implement **that** surface rather than inventing one, and
   stating where the `EIO`-ends-the-read fact ended up.

## Results (landed 2026-11-02)

**The seam's surface — csl-04 and csl-06 implement exactly this, not an invented
one.** `src/core/termhub/pty_posix.py` answers:

| member | shape |
|---|---|
| `open_pair()` | `-> PosixPty` — the terminal pair (today's `pty.openpty()`); a failure raises the `OSError` it was |
| `PosixPty.slave_fd` | property `-> int` — the end the spawn hands the child as stdio |
| `PosixPty.set_size(cols, rows)` | `-> None` — `TIOCSWINSZ`; raises `OSError` (the session suppresses) |
| `PosixPty.get_size()` | `-> tuple[int, int]` as `(cols, rows)` — `TIOCGWINSZ`; raises `OSError` |
| `PosixPty.read(max_bytes)` | `-> bytes`, never decoded text — raises `FarEndGone` for `EIO`, re-raises any other `OSError` unchanged |
| `PosixPty.write(data: bytes)` | `-> None`; raises `OSError` (the session suppresses) |
| `PosixPty.close_slave()` | `-> None` — the slave-close rule and its *why* moved here with it |
| `PosixPty.close_master()` | `-> None` — the *when-it-is-safe* rule (never under a blocked read) stays in `session.py`, which owns the reader thread |
| `FarEndGone` | `OSError` subclass — the seam's answer to *"is the far end gone?"* |

**Where the `EIO`-ends-the-read fact ended up:** inside `PosixPty.read` — the
`errno.EIO` check re-raises it as `FarEndGone`, deliberately an `OSError`
subclass, so `session.py`'s `pump()` keeps its single `except OSError: break`
and the POSIX outcome is unchanged: the read's end breaks the loop, then the
reap and the `exit` frame follow exactly as before, and an unexpected error
keeps today's path. A ConPTY driver (csl-06) answers the same question by
raising this same exception (its EOF translation), and the pump needs no
second condition.

The `ImportError → None` block for `fcntl`/`pty`/`termios` moved to
`pty_posix.py` verbatim with its comment (the module still imports on Windows
with the three bound to `None`); `session.py` no longer imports `pty`, `fcntl`,
`termios` or `struct`, contains no `openpty`/`TIOCSWINSZ`/`TIOCGWINSZ`, and
keeps its `os.name != "posix"` guard and sentence untouched — replacing that
guard is csl-04. `spawn()` keeps HEAD's exact fd order and lifetime: size
before the spawn, the parent's slave closed in the spawn's `finally`, the
master closed only on the failure path, and `self.pty` assigned only after a
successful spawn (mirroring HEAD's trailing `self.master_fd = master_fd`).
`constants.py` is untouched — no mechanic needed a new tunable; `PTY_READ_BYTES`
and `STOP_GRACE_SECONDS` stay where tests patch them. `termhub/__init__.py`'s
module map gained the one-line entry for the new file.

**Test evidence on this runner (Windows).** `tests/test_terminal.py` and
`tests/test_api.py` are byte-unchanged (`git diff` empty for both).

* `pytest tests/test_write_surface_isolation.py` → 20 passed.
* `pytest tests/test_api.py` → 63 passed, 2 failed — the two are the pre-existing
  `NameError: name 'helpers_block' is not defined` at `routes.py:1417` (a file
  this task does not touch; the checkout is mid-merge with `AGENTS.md`
  unmerged). Reproduced **identically with HEAD's `session.py`** swapped in.
* `pytest tests/test_terminal.py` behaves **identically before and after the
  move** on this box: `.F` then a hang in `TestGuards`/`TestRegistry` (the
  accept-then-refuse wedge csl-02's Results note documents — `routes.py` still
  refuses on non-POSIX), and `TestSession` skips its five with the pre-existing
  probe sentence `no shell at /bin/sh to spawn`. Both verified against HEAD's
  `session.py` on the same runner, so the refactor costs nothing here.
  **The twelve-green-on-Linux baseline (criterion 1) was not runnable from this
  Windows machine** — what ran instead, at the seam itself: a fake driver
  through `pump()` reproduced the frame sequence `output* → exit` with
  per-chunk framing and the incremental decoder joining a multi-byte character
  split across two chunks, and `read()` was proved to translate `EIO` →
  `FarEndGone` (still an `OSError`) while `EBADF` stays the error it was.
* `cd web && npm test` → 472 passed; `git diff --stat -- web/` empty.
