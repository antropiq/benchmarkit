# csl-01 — Extract the Git-install probe into `mcp_connectors/gitbash.py`

## Deliver this

The Git-for-Windows discovery and search-path logic that the console connector
already runs (target B1 / G-D4) becomes **one shared, stdlib-only module** that
the terminal will consume in csl-05. The console connector delegates to it and
its behaviour — every verdict, every refusal sentence, every log line — is
unchanged.

New module `mcp_connectors/gitbash.py` answers four questions:

1. `discover_git_install(...)` — is there a Git install root whose
   `usr\bin\ls.exe` exists? (which-git walk-up → fixed candidates → env override
   → `None`.)
2. `git_search_paths(root)` — the command search order for a found root:
   `usr\bin` → `bin` → the tool tree that exists on this install (`mingw64\bin`
   or `ucrt64\bin` — this machine has only `ucrt64`, so the helper must answer
   from what is actually present rather than hard-coding one name) → the ambient
   PATH is the caller's business. `console_mcp.py` keeps its `GIT_SEARCH_PATHS`
   module attribute (its tests patch it), fed by this helper.
3. `git_shell_program(root)` — the interactive shell program for that root
   (`bin\bash.exe`, verified to exist), or `None`.
4. `git_child_env(base, root)` — a child environment for a shell or tool started
   under that root: Git's directories **prepended** to `PATH` in Windows form,
   keeping entries that contain spaces intact.

Every parameter is taken as an argument (candidates, the env override name, the
base environment) so a test forces either answer on one runner. Module-level
defaults live in `gitbash.py` and remain patchable, exactly as they are today in
`console_mcp.py`.

## Files it touches

| file | change |
|---|---|
| `mcp_connectors/gitbash.py` | **new** — the probe, the search path, the shell program, the child env |
| `mcp_connectors/console_mcp.py` | edit — `discover_git_install`, `git_search_paths`, `GIT_INSTALL_CANDIDATES`, `GIT_ENV_OVERRIDE`, `GIT_SEARCH_PATHS` become imports/delegations; the `Toolset` block, `select_toolset()`, `resolve_command()`, `no_toolset_refusal()` and `toolset_split()` stay where they are |
| `tests/test_console_gbash.py` | edit — the existing discovery/resolver tests keep passing; add tests for the two new answers (the shell program, the child env) and for "one probe, both callers" |
| `tests/test_console_policy.py` | **must stay untouched and green** — it is the Linux gate |

`src/` does not use the new module yet (csl-05 is the first `src` consumer).

## Rules it must not break (quoted)

* G-D4: the found root is *"always verified by `usr\bin\ls.exe`, so a stray
  `git.exe` elsewhere on PATH cannot select the mode"*; the resolution order is
  *"Git's trees first … then the ambient PATH"*; a search **never** falls back to
  `shell=True`.
* G-D4's env override is *"deliberately not a managed `.env` key, the same
  reasoning as `BENCHMARKIT_PATH_LOCATION`"*. Do not add it to `SETTING_KEYS`,
  `_render_env()` or `.env.example`.
* G-D1/G-D2: *"a policy rule changes in `console_mcp.py` / `path_policy.py` once,
  and both modes inherit it"* — moving the probe must not create a second copy of
  it, and must not move any **policy** code into `gitbash.py`. The probe answers
  *where a program is*, never *whether a command is allowed*: the resolver's
  verdict stays folded into `console_mcp.py`'s checks.
* G-D3: `bash`, `sh`, `cmd`, `powershell`, `pwsh` stay in
  `ALWAYS_BLOCKED_COMMANDS` with the no-shell sentence. This task adds a shell
  *lookup* for the terminal; it must not make the console connector able to run a
  shell, and no test may weaken those refusals.
* AGENTS.md: *"The connectors answer the workspace level for their own root … the
  app and the connector processes share one parser, one matcher and one refusal
  sentence."* A connector process cannot `import src`, so `gitbash.py` stays
  stdlib-only and importable from `mcp_connectors/` — the same reason
  `path_norm.py`, `policy_glob.py` and `fscase.py` live there.
* AGENTS.md: *"Tests stay hermetic"*. The candidates and the override name stay
  patchable module attributes; a test patches them, never a path it computed.
* Python style: explicit annotations and `typing` imports, no bare containers.

## Depends on

Nothing. It is independent of csl-02 and csl-03.

## How to tell it is done

1. `pytest tests/test_console_policy.py tests/test_console_gbash.py tests/test_path_policy.py`
   green, **zero skips on Linux**, and `test_console_policy.py`'s file content
   unchanged (`git diff` shows no edit to it).
2. `pytest tests/test_write_surface_isolation.py` green — the new module's name
   and contents must not trip that guard.
3. `tests/test_console_gbash.py` asserts, **on Linux**, by patching the seam:
   * a fake Git tree (a `tmp_path` with `usr\bin\ls.exe`) is discovered, and a
     `git.exe` with no such tree is *not*;
   * `git_shell_program` answers `bin\bash.exe` for that fake root and `None`
     when it is absent;
   * `git_child_env` prepends Git's directories to `PATH` in the runner's own
     path-separator form and an entry containing a separator-adjacent space
     survives verbatim; the base environment's other keys are untouched;
   * `console_mcp`'s discovery answers come from `gitbash` (a delegation test
     in the style of the existing `fscase` "one probe" test).
4. **Measured, then recorded in this file:** does a child given only the ambient
   `PATH` fail to resolve `stty` where `git_child_env` resolves it? Run the
   comparison with the runner's own shell and write the observed answer here (in
   the *Results* note you append at the end of this file), because csl-05 builds
   on it rather than on this sentence.
