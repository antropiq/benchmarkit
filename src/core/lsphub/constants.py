"""Static configuration of the LSP hub: paths, timeouts, wire constants.

The workspace's declaration file and the deny-list live here as *module
attributes* on purpose (same idea as ``mcphub.constants``): the code that
reads them resolves them through this module at call time, so pointing the
hub at another declaration file (the tests do exactly this) is one patch of
``lsphub.constants.LSP_SETTINGS_PATH`` instead of a hunt for every importer.

The declaration itself lives in the workspace — like the MCP one, it
describes *this* folder's servers, so ``lsp_settings_path()`` is built per
call out of the active working location. The install-level example beside
the deny-list (``lsp_connectors/lsp_servers.example.json``) is not read as a
declaration any more: it is the committed source of the template the seed
copies when a folder becomes the working location.
"""

from pathlib import Path
from typing import Optional

# The one reach-capability seam, re-exported through *this* module so both hubs
# answer the question with the same function and a test forces it the same way
# it redirects any other hub constant (win-07 / A-D4).
from src.core.mcphub.process import guardrails_speak_here  # noqa: F401 — hub attr

# The reserved folder's name comes from ``mcphub.constants`` (which takes it
# from ``mcp_connectors/policy_shared``): two constants for one folder name
# is how the folder stops being one folder.
from src.core.mcphub.constants import POLICY_DIRNAME
from src.config import path_location

PROJECT_ROOT = Path(__file__).resolve().parents[3]
LSP_CONNECTORS_DIR = PROJECT_ROOT / "lsp_connectors"
LSP_LOGS_DIR = PROJECT_ROOT / "logs" / "lsp"

# The workspace's language-server declaration, keyed by id, inside the
# reserved folder at the working location: the one and only source of
# language servers. The file may carry secrets in ``env`` and is therefore
# gitignored (via ``.nojae/``); its ``env`` values are never returned by the
# API. The ``enable`` toggles write back into it — the folder's "edited only
# outside the app" rule bends for it the way it does for the MCP declaration
# and the prompt-helper store.
LSP_SETTINGS_FILENAME = "lsp-settings.json"

# The template the seed leaves in the reserved folder when a folder becomes
# the working location. Unlike MCP there is **no seeded declaration**: a
# server's ``workspaceRoot`` and interpreter paths are per-machine, so the
# app hands out this template and the operator writes the declaration from
# it by hand. The hub never reads the template.
LSP_SETTINGS_TEMPLATE_FILENAME = "lsp-settings.template.json"

# The committed install-level example, copied byte for byte into the seeded
# template. Its ``${NAME}`` references resolve on every read
# (``mcphub.validation.expand_env``), which is what keeps one file correct
# on every machine and across every workspace switch.
LSP_SETTINGS_TEMPLATE = LSP_CONNECTORS_DIR / "lsp_servers.example.json"

# **A pin, not the default.** When not ``None`` it wins over the computed
# path below — the ``mcphub.constants.MCP_SETTINGS_PATH`` precedent: one
# patchable override keeps the tests' one-attribute redirect without
# re-introducing an import-time constant. Left at ``None``, the workspace's
# own file answers.
LSP_SETTINGS_PATH: Optional[Path] = None


def lsp_settings_path() -> Path:
    """The active workspace's ``.nojae/lsp-settings.json`` (may not exist).

    Built per call out of module attributes, so the pin above, a patched
    filename, or a switched working location each move the lookup the same
    way. A read of an absent file is ``([], [])`` — an undeclared workspace
    simply has no language servers until its declaration is written.
    """
    if LSP_SETTINGS_PATH is not None:
        return LSP_SETTINGS_PATH
    return Path(path_location()) / POLICY_DIRNAME / LSP_SETTINGS_FILENAME

# Deny-list of workspace roots a server may not be pointed at (same idea as
# mcp_connectors/restricted-fqdn.txt). One absolute prefix per line.
LSP_RESTRICTED_PATHS = LSP_CONNECTORS_DIR / "restricted-paths.txt"

# Per-request and handshake timeouts. Language servers are chatty and slow
# on cold start (tsserver indexes the whole project), so ``initialize`` gets
# far more room than an ordinary request.
REQUEST_TIMEOUT = 20.0
INIT_TIMEOUT = 60.0
# Grace given to a language server between shutdown/exit and SIGTERM/SIGKILL.
STOP_GRACE_SECONDS = 5.0
# How long ``lsp_*_diagnostics`` waits for a publishDiagnostics notification
# after opening/changing a document (servers push; there is no pull request
# unless the server advertises the 3.17 diagnosticProvider).
DIAGNOSTICS_WAIT_SECONDS = 5.0
# Documents bigger than this are refused: a language server is not a way to
# stream a giant file into the context window.
MAX_DOCUMENT_BYTES = 1_000_000
# Default per-server cap on reported diagnostics.
DEFAULT_MAX_DIAGNOSTICS = 40
# Default idle time after which a server is torn down and respawned on the
# next tool call (0 disables the reaper).
DEFAULT_IDLE_SHUTDOWN_SECONDS = 300
# How often the idle reaper wakes up on the hub thread.
REAPER_INTERVAL_SECONDS = 30.0
# A server left in ``starting`` for longer than the handshake plus its
# teardown grace is stuck, not slow. This is the one status no other path
# clears on its own, so the reaper turns it into ``error``: the panel then
# says so, and the settings toggle (which retries anything but ``running``)
# can start it again. The budget is deliberately above what a *successful*
# cold start can take, so a healthy server is never interrupted.
STARTING_STUCK_SECONDS = (
    INIT_TIMEOUT + 2 * STOP_GRACE_SECONDS + 15.0
)
# Result text cap, mirroring extract_tool_result_text's truncation idea.
MAX_RESULT_CHARS = 12_000

# Volar (``vue-language-server``) delegates every TypeScript semantic to a
# tsserver it expects its *client* to own, and asks for it over the private
# ``tsserver/request`` notification. Its ``getLanguageService()`` awaits the
# answer, so an unanswered relay hangs the session forever: every later
# request for that file queues behind the dangling promise and times out.
# Answering at least ``projectInfo`` is therefore what makes ``vue`` usable.
TSSERVER_REQUEST = "tsserver/request"
TSSERVER_RESPONSE = "tsserver/response"
# Volar prefixes every relayed command with ``_vue:``.
TSSERVER_COMMAND_PREFIX = "_vue:"
TSSERVER_PROJECT_INFO = "_vue:projectInfo"
# Config files a projectInfo answer may point at. With none found the hub
# answers ``null`` and Volar falls back to its inferred ("simple") project,
# which still resolves Vue semantics for a standalone workspace.
PROJECT_CONFIG_NAMES = ("tsconfig.json", "jsconfig.json")

LSP_MISSING_HINT = (
    'The "pygls" package is not installed — run: pip install -e ".[lsp]"'
)

# win-07 / A-D4: every declared language server is a stdio child whose path
# reach is bounded by ``restricted-paths.txt`` below plus the connector
# deny-prefixes. Both lists speak both platforms' path forms (B1 for the
# connector lists, B-D1 for this one, entries expanded from the environment at
# read time), and the child's process *tree* — tsserver, Volar's workers — is
# bounded on either platform too: the spawn puts it in a Job Object where there
# are no process groups, and the teardown is the one shared entry point (B-D5,
# ``proctree`` / ``proctree_jobs``). So the only thing left to refuse on is a
# list that matches nothing: where no entry can name this filesystem's paths,
# the app has a bound in words only and says so instead of claiming it
# (lspfinal-04 — that seam's platform half is gone, because the job wiring
# above is what replaced it). The wording is *the same string* as
# ``mcphub.constants``' copy (a test asserts it): the two hubs refuse local
# processes in one voice, and now for one reason.
#
# lspfinal-06 dropped the ``(see target B)`` pointer from the wording, in both
# hubs together: the target it named is finished, so it pointed a reader at a
# closed plan instead of at the thing to change. What the sentence says now is
# the cause itself — a guardrail list with no entry in this filesystem's path
# form — which is a file the operator edits.
LOCAL_REACH_REFUSAL = (
    "its reach cannot be bounded on this filesystem: no guardrail entry speaks "
    "this working location's path form"
)

# lspfinal-03 / B-D6: the sentences a declared command can end a start with.
#
# A language server declared the way a POSIX user writes it — the extensionless
# shim npm/volta put on PATH — is not an executable image on Windows: that file
# is a ``#!/bin/bash`` script, and CreateProcess answers ``[WinError 193] %1 is
# not a valid Win32 application``. So the hub resolves the declared command
# through ``shutil.which`` (which honours ``PATHEXT``) before asking pygls to
# spawn it, and what an operator reads when a start does not work is one of the
# sentences below — never a bare OS code, and never a ``shell=True`` fallback
# that would turn a declared command into a command line.
#
# Every one carries the same lead and names *both* the command and the folders
# that were searched, because that is the half the operator can act on.
SPAWN_FAILURE_LEAD = "Could not spawn language server: "

# The command and the folders are quoted rather than ``repr``-ed: they are paths
# an operator reads and pastes, and an escaped ``\\`` is not what to paste.
COMMAND_NOT_RESOLVED_NOTE = (
    "the declared command '{command}' is not a file and it resolves to nothing "
    "on the path that was searched ({searched}). Name it as that path sees it — "
    "an executable image, or a shim whose extension is in PATHEXT — or give the "
    "entry an ``env.PATH`` that carries it."
)

COMMAND_UNUSABLE_NOTE = (
    "the declared command '{command}' is a file, but the file found there "
    "(searched in {searched}) is not a program this OS can start: an "
    "extensionless script — the ``#!/bin/bash`` shim npm and volta install — "
    "carries no executable image. Point the declaration at the program itself, "
    "usually the PATHEXT shim beside it (e.g. ``pyright-langserver.cmd``)."
)

COMMAND_OS_ERROR_NOTE = (
    "the OS refused to start the declared command '{command}' ({exc}); the "
    "program it handed over was '{program}', searched in ({searched})."
)

# The OS answers a file that is a script rather than an executable image with
# these two codes, on the two Windows flavours of it (a text script answers 193,
# a file with a stale or 16-bit header answers 216). A bare code is not
# actionable, so it is translated into ``COMMAND_UNUSABLE_NOTE`` above; any
# other OS answer at the spawn keeps its own words, because those *are* the
# diagnosis for it.
UNEXECUTABLE_IMAGE_WINERRORS = (193, 216)
