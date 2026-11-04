"""Application configuration: .env loading, questions, and endpoint config."""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from dotenv import load_dotenv
from pydantic import ValidationError

from src.core.models import DEFAULT_LLM_SYSPROMPT, EndpointsConfig

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"
RESOURCES_DIR = PROJECT_ROOT / "resources"

QUESTION_SET_DEFAULT = "legacy"

# Cap on tool-calling iterations for one chat reply; shared by MCP connectors
# and LSP language servers alike (one knob, no split-brain).
DEFAULT_MAX_TOOL_ITERATION = 25

# Managed .env keys, in the order they are written to the file.
SETTING_KEYS: Tuple[str, ...] = (
    "SERVER_URL",
    "SERVER_URL_API_KEY",
    "ENDPOINT",
    "LLM_SYSPROMPT",
    "TEMPERATURE",
    "TOP_P",
    "TOP_K",
    "MIN_P",
    "REPEAT_PENALTY",
    "PRESENCE_PENALTY",
    "REASONING",
    "LITELLM_DEBUG",
    "SUPPORTS_VISION",
    "QUESTION_SET",
    "VERIFIER_AGENT_SYSPROMPT",
    "VERIFIER_AGENT_SERVERURL",
    "VERIFIER_AGENT_SERVERURL_API_KEY",
    "VERIFIER_AGENT_ENDPOINT",
    "MAX_TOOL_ITERATION",
    "LSP_ENABLED",
)

# Names an operating system already owns, which a managed key may therefore
# never take.
#
# This joins the rule AGENTS.md states about a new or renamed key — edit
# ``SETTING_KEYS``, ``_render_env()`` and ``.env.example`` together — with a
# fourth demand: the name must be free of OS variables on *every* supported
# platform. ``load_dotenv_once`` below deliberately keeps dotenv's "existing
# environment wins" precedence, so a key that names an OS variable does not set
# a value at all: the OS value survives and the app reads something it never
# wrote. That is the whole ``TEMP`` story — on Windows ``TEMP`` is the temp
# folder, so a sampling parameter became a directory path (and a stray folder
# appeared in the working location). The check lives on the key list rather
# than inside the loader, because that precedence is the point and not a bug to
# paper over: the collision is prevented by never choosing such a name.
_RESERVED_ENV_NAMES: frozenset[str] = frozenset(
    {
        "TEMP",
        "TMP",
        "PATH",
        "HOME",
        "USER",
        "USERNAME",
        "USERPROFILE",
        "SYSTEMROOT",
        "COMSPEC",
        "PROMPT",
        "OS",
        "APPDATA",
        "LOCALAPPDATA",
        "PROGRAMFILES",
        "COMPUTERNAME",
        "HOMEDRIVE",
        "HOMEPATH",
        "NUMBER_OF_PROCESSORS",
        "PATHEXT",
    }
)

# Keys always written to .env; the others are commented out when empty.
REQUIRED_SETTING_KEYS: Tuple[str, ...] = ("SERVER_URL", "ENDPOINT")

# Environment variable holding the *working location* the MCP connectors and
# language servers are pointed at. Both declaration files reference it
# (``--root-dir`` / ``workspaceRoot`` as ``${BENCHMARKIT_PATH_LOCATION}``), so
# one value moves every file-touching tool together.
#
# Deliberately NOT a managed ``SETTING_KEYS`` key: it is process state, not
# run configuration, and keeping it out of ``.env`` means ``save_settings``
# (which regenerates that file from a fixed template) can neither erase it nor
# resurrect a stale value at the next boot. The boot default below is the
# single persistence rule: a restart always lands on the install directory.
PATH_LOCATION_KEY = "BENCHMARKIT_PATH_LOCATION"


# Debug switch for the LiteLLM *client* library. When set, litellm's own
# loggers (``LiteLLM`` / ``LiteLLM Router`` / ``LiteLLM Proxy``) are raised to
# DEBUG, which is what makes a failed call legible: without it the exception
# handed to the ``except Exception`` in the core loops is only litellm's
# one-line summary, while the provider's real status/body stays unrendered.
#
# The value lives in ``.env`` (a managed ``SETTING_KEYS`` key) rather than in
# code so it can be flipped from the settings dialog, and it is applied to the
# live process at startup and on every save — no restart needed.
LITELLM_DEBUG_KEY = "LITELLM_DEBUG"

TRUTHY_VALUES = frozenset({"1", "true", "yes", "on"})

# The three loggers ``litellm._turn_on_debug()`` touches; the off path resets
# exactly these back to ``NOTSET`` (inherited from the root logger).
LITELLM_LOGGER_NAMES: Tuple[str, ...] = ("LiteLLM", "LiteLLM Router", "LiteLLM Proxy")


def env_flag(key: str) -> bool:
    """True when an env var is set to a truthy value (1/true/yes/on)."""
    return (os.environ.get(key) or "").strip().lower() in TRUTHY_VALUES


def normalize_env_flag(raw: Optional[str]) -> str:
    """Normalise a flag-ish submitted value to the ``.env`` form ("1" or "").

    Used by the settings API for the boolean-ish keys: any truthy spelling
    (1/true/yes/on) becomes the canonical ``1``, and anything else — including
    the explicit off spellings (0/false/no/off) and an empty field — becomes
    "", which ``save_settings`` writes as a commented-out line.
    """
    return "1" if (raw or "").strip().lower() in TRUTHY_VALUES else ""


def litellm_debug_enabled() -> bool:
    """True when LiteLLM's internal debug logging was requested."""
    return env_flag(LITELLM_DEBUG_KEY)


def apply_litellm_debug() -> bool:
    """Raise litellm's own loggers to DEBUG when ``LITELLM_DEBUG`` is set.

    Called at app startup (right after the .env is read) and again on every
    settings save, so flipping the switch takes effect without restarting the
    server. The off path is just as explicit: levels go back to ``NOTSET`` and
    litellm's own debug-info banner comes back, so a process never stays
    chatty (nor silent) after the flag is cleared.

    litellm's own log handler already accepts DEBUG records — the level it
    publishes under is gated by its *loggers*, which default to ``NOTSET`` and
    therefore inherit WARNING from the root logger. Raising those three is all
    that is needed; the records land on stderr like the rest of uvicorn's
    output. ``litellm`` is imported lazily on purpose: it is a heavy package,
    and this module is also imported by code paths that never call an LLM.

    With debug on, ``suppress_debug_info`` is set too: the
    "LiteLLM.Info: If you need to debug this error, use
    `litellm._turn_on_debug()'" banner litellm prints on every mapped
    exception would only repeat advice this function already followed, and it
    buries the real tracebacks. Returns whether the switch is now on.
    """
    import litellm  # noqa: PLC0415 — heavy dependency, imported only on demand

    enabled = litellm_debug_enabled()
    if enabled:
        # Private litellm API on purpose: it is the only switch that turns the
        # library's own debug logging on, and it is stable across the versions
        # this project pins.
        litellm._turn_on_debug()  # type: ignore[reportPrivateImportUsage]
        litellm.suppress_debug_info = True
    else:
        for name in LITELLM_LOGGER_NAMES:
            logging.getLogger(name).setLevel(logging.NOTSET)
        litellm.suppress_debug_info = False
    return enabled


# The provider prefix every LLM call in this project uses: ``core/llm.py`` and
# ``core/chat.py`` pass ``model=f"openai/{name}"`` to ``litellm.completion()``,
# so a cost entry must be keyed the same way to be found by the same lookup.
LITELLM_MODEL_PREFIX: str = "openai/"

# Cost-table entry litellm needs for a model it has never heard of.
#
# The shape mirrors what ``litellm.register_model`` documents and what the
# bundled ``model_prices_and_context_window.json`` stores per entry: a
# provider-qualified **key** plus ``litellm_provider`` / ``mode`` inside the
# value. Both halves matter:
#
# * Keying by the bare name (``qwen3.8-flash-next-iq3_s``) makes registration
#   itself go look the name up first — ``register_model`` calls
#   ``get_model_info`` → ``get_supported_openai_params`` → ``get_llm_provider``,
#   and with no prefix in the key that last one cannot infer a provider, so
#   litellm prints its red ``Provider List: https://docs.litellm.ai/...``
#   banner (twice per model, at startup) and the entry lands as
#   ``mode: None``. The prefix is what carries the provider here.
# * ``litellm_provider`` also decides which provider list the name is added to
#   (``open_ai_chat_completion_models`` for ``"openai"``), and without
#   ``"mode": "chat"`` the entry declares no modality.
#
# Prices are all zeros: benchmarkit talks to local / self-hosted endpoints and
# never reports a monetary cost, so the only correct price is 0 — and *having*
# an entry is what stops litellm from raising on every single call (see below).
# The two cache keys are part of the entry so ``register_model`` does not add a
# "no cache cost tracking" warning of its own.
ZERO_COST_MODEL_INFO: Dict[str, Any] = {
    "input_cost_per_token": 0,
    "output_cost_per_token": 0,
    "cache_read_input_token_cost": 0,
    "cache_creation_input_token_cost": 0,
    "litellm_provider": "openai",
    "mode": "chat",
}


def register_unmapped_models(model_names: Iterable[Optional[str]]) -> List[str]:
    """Give litellm a zero-cost entry for every model it cannot price.

    Names arrive from the settings payload, where a blank optional field can be
    ``None`` rather than ``""``, so the element type allows it: each entry is
    normalised and blanks are skipped.

    A quantised local model name (``qwen3.8-flash-next-iq3_s``) is by
    definition absent from litellm's ``model_prices_and_context_window.json``,
    so its per-call cost calculator raises ``ModelNotMappedError``. litellm
    catches that internally — the call itself succeeds — but with
    ``LITELLM_DEBUG=1`` it logs the whole exception *plus traceback* once per
    call, which buries the lines that debug mode exists to show (a real
    upstream failure hides among ~50 identical cost-table dumps).

    Registering a zero-cost entry removes that noise at the source. Only models
    litellm genuinely cannot price are registered: one already in its cost map
    keeps its real pricing, so this never rewrites a figure the library knows.

    Called at startup and on every settings save (next to
    ``apply_litellm_debug``), because ``ENDPOINT`` / ``VERIFIER_AGENT_ENDPOINT``
    are exactly what the settings dialog edits — but it deliberately does *not*
    read them through :func:`load_endpoints_config`: that runs on every API
    request, and mutating litellm's process-wide cost map per request is not
    something a read endpoint should do. Returns the names registered, which
    makes the effect observable in tests. Never raises: a cost table is pure
    log hygiene, and a failure here must not break app startup.
    """
    import litellm  # noqa: PLC0415 — heavy dependency, imported only on demand

    registered: List[str] = []
    for name in model_names:
        name = (name or "").strip()
        if not name or name in registered:
            continue
        try:
            # Same lookup the failing cost calculator performs (and the same
            # provider-qualified form ``core/llm.py`` calls ``completion()``
            # with), so the guard cannot disagree with it.
            litellm.get_model_info(
                model=f"{LITELLM_MODEL_PREFIX}{name}", custom_llm_provider="openai"
            )
        except Exception:
            # Not in the cost map (the ModelNotMappedError case) — teach it.
            try:
                litellm.register_model(
                    {f"{LITELLM_MODEL_PREFIX}{name}": dict(ZERO_COST_MODEL_INFO)}
                )
                registered.append(name)
            except Exception:  # noqa: BLE001 — hygiene only, never fatal
                pass
    return registered


def register_configured_models() -> List[str]:
    """Register the model names the current environment points at.

    Reads the two endpoint keys straight from ``os.environ`` rather than
    building an :class:`EndpointsConfig`: startup must survive an invalid
    ``.env`` (the API surfaces that error, the process must not die on it),
    and after ``save_settings`` the environment already carries the new
    values.
    """
    return register_unmapped_models(
        (os.environ.get("ENDPOINT", ""), os.environ.get("VERIFIER_AGENT_ENDPOINT", ""))
    )


def load_dotenv_once() -> None:
    """Load the .env file once; existing environment variables take precedence."""
    if ENV_PATH.exists():
        load_dotenv(ENV_PATH)


def default_path_location() -> str:
    """The install directory, used as the working-location boot default."""
    return str(PROJECT_ROOT)


def setup_path_location() -> None:
    """Ensure ``PATH_LOCATION_KEY`` is set, defaulting it to the install dir.

    Called at startup, *after* ``load_dotenv_once``: an exported variable (or
    one written in ``.env``) wins, which is what lets an operator pin the
    location without touching the code. Never overwrites, so it stays
    idempotent across the test suite and repeated app factories.
    """
    current = (os.environ.get(PATH_LOCATION_KEY) or "").strip()
    if not current:
        os.environ[PATH_LOCATION_KEY] = default_path_location()


def path_location() -> str:
    """Effective working location (the value the declaration files expand to).

    Reads ``os.environ`` at call time, so a value set through the API is what
    the next hub re-scan sees. Falls back to the boot default rather than
    returning empty, so the settings dialog always shows a real path.
    """
    return (os.environ.get(PATH_LOCATION_KEY) or "").strip() or default_path_location()


# Project agent instructions: when present at the root of the working
# location, their text is appended to the Chat system prompt so a new
# conversation starts already knowing the project's conventions.
AGENT_INSTRUCTIONS_FILENAME = "AGENTS.md"


def agent_instructions_filename() -> str:
    """Name of the agent-instructions file looked for at the location root.

    Read through a function (not the constant) so a test can point the lookup
    at another filename without rewriting the module.
    """
    return AGENT_INSTRUCTIONS_FILENAME


def load_agent_instructions(location: Optional[str] = None) -> str:
    """Text of the working location's ``AGENTS.md``, "" when absent/unreadable.

    Resolved against ``BENCHMARKIT_PATH_LOCATION`` (the same folder the
    file-touching tools are pointed at) at call time, so moving the location
    through the settings dialog also moves where the instructions are looked
    for. Read per chat request rather than cached: editing ``AGENTS.md`` then
    takes effect on the very next message.
    """
    root = Path(location) if location else Path(path_location())
    try:
        return (root / agent_instructions_filename()).read_text(encoding="utf-8").strip()
    except (OSError, ValueError):
        return ""


def agent_instructions_block(text: Optional[str] = None) -> str:
    """The instructions rendered as a system-prompt block ("" when no text).

    The framing tells the model these are the project's own instructions —
    they win over generic wording for anything project-specific — and pins
    the project root to the working location, which is what makes the paths
    the instructions mention (``src/``, ``resources/``, ``tests/``) and the
    tool UIDs they describe resolve to the real project.
    """
    body = load_agent_instructions() if text is None else (text or "").strip()
    if not body:
        return ""
    return (
        "The current project root is {root}\n\n"
        "The following are this project's own instructions (AGENTS.md at that "
        "root). Follow them for anything project-specific; they take "
        "precedence over generic wording.\n\n{body}"
    ).format(root=path_location(), body=body)


def questions_path() -> Path:
    """Resolve the questions file path from the QUESTION_SET env var.

    Defaults to ``legacy`` which reads ``resources/legacy.json``.
    Resolved at call time so ``.env`` (loaded at app startup) takes effect.
    """
    name = os.environ.get("QUESTION_SET", QUESTION_SET_DEFAULT)
    return RESOURCES_DIR / f"{name}.json"


def list_question_sets() -> List[Dict[str, Any]]:
    """Discover available question sets in ``resources/``.

    Each ``*.json`` file must contain a ``"questions"`` key. Returns a list of
    ``{id, label, count}`` dicts sorted by filename.
    """
    sets: List[Dict[str, Any]] = []
    for path in sorted(RESOURCES_DIR.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            questions = data.get("questions", [])
            if not isinstance(questions, list):
                continue
        except (json.JSONDecodeError, OSError, ValueError):
            continue
        sets.append(
            {
                "id": path.stem,
                "label": path.stem.capitalize(),
                "count": len(questions),
            }
        )
    return sets


def load_questions(path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Load the benchmark questions from the configured question set."""
    if path is None:
        path = questions_path()
    data: Dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data.get("questions", [])


def load_endpoints_config() -> Tuple[Optional[EndpointsConfig], Optional[ValidationError]]:
    """Build and validate the endpoints config from the environment.

    Returns ``(config, None)`` on success or ``(None, error)`` when the
    environment is invalid.
    """
    try:
        return EndpointsConfig.from_env(), None
    except ValidationError as exc:
        return None, exc


def load_settings() -> Dict[str, str]:
    """Return the effective value of every managed .env key (empty if unset).

    ``os.environ`` already reflects the .env file loaded at startup (real
    environment variables keep precedence) and is refreshed on every save.
    ``QUESTION_SET`` and ``LLM_SYSPROMPT`` resolve to their defaults when unset.
    """
    defaults = {
        "QUESTION_SET": QUESTION_SET_DEFAULT,
        "LLM_SYSPROMPT": DEFAULT_LLM_SYSPROMPT,
        "MAX_TOOL_ITERATION": str(DEFAULT_MAX_TOOL_ITERATION),
    }
    return {
        key: os.environ.get(key) or defaults.get(key, "")
        for key in SETTING_KEYS
    }


def save_settings(values: Dict[str, str]) -> None:
    """Persist managed settings to the .env file and refresh ``os.environ``.

    Empty optional values are written as commented-out lines and removed from
    the environment; required keys are always written. The file is regenerated
    from a fixed template mirroring ``.env.example``. ``QUESTION_SET`` always
    resolves to ``QUESTION_SET_DEFAULT`` when unset, so it is never written
    as a commented-out line.
    """
    clean: Dict[str, str] = {
        key: (values.get(key) or "").strip() for key in SETTING_KEYS
    }
    # QUESTION_SET always has a value; default to "legacy" when unset.
    if not clean["QUESTION_SET"]:
        clean["QUESTION_SET"] = QUESTION_SET_DEFAULT
    ENV_PATH.write_text(_render_env(clean), encoding="utf-8")
    for key in SETTING_KEYS:
        if clean[key]:
            os.environ[key] = clean[key]
        else:
            os.environ.pop(key, None)
    # Re-apply the LiteLLM debug switch from the values just written, so
    # ticking the box in the settings dialog takes effect on the *current*
    # process (the next call is what gets the extra logging) instead of
    # waiting for a server restart.
    apply_litellm_debug()
    # Same reason to redo it here: the endpoints are what just changed, so a
    # newly typed model name gets its zero-cost cost-table entry at once
    # instead of leaking one cost-table traceback per call until the restart.
    register_configured_models()


def _render_env(values: Dict[str, str]) -> str:
    """Render the .env file content from the managed values."""

    def line(key: str) -> str:
        value = values.get(key, "")
        if not value:
            return f"#{key}="
        return f"{key}={_encode_value(value)}"

    lines = [
        "# Benchmark'it configuration (managed by the settings dialog).",
        "# Real environment variables take precedence over the values defined here.",
        "",
        "# Target LLM endpoint",
        line("SERVER_URL"),
        line("SERVER_URL_API_KEY"),
        line("ENDPOINT"),
        "# Set to 1 when the model accepts image input (enables Chat image attach)",
        line("SUPPORTS_VISION"),
        line("LLM_SYSPROMPT"),
        "",
        "# Sampling parameters (optional)",
        line("TEMPERATURE"),
        line("TOP_P"),
        line("TOP_K"),
        line("MIN_P"),
        line("REPEAT_PENALTY"),
        line("PRESENCE_PENALTY"),
        line("REASONING"),
        line("QUESTION_SET"),
        "",
        "# LITELLM_DEBUG=1 raises the LiteLLM client's own loggers to DEBUG so",
        "# a failed call shows its real upstream status/body on the server",
        "# console instead of just litellm's one-line summary (it also silences",
        "# the 'use litellm._turn_on_debug()' hint, whose advice is redundant",
        "# here). Applies to the next saved settings / server start; it floods",
        "# the console with every request payload, so switch it back off once",
        "# the failure is diagnosed.",
        line("LITELLM_DEBUG"),
        "",
        "# Verifier agent (optional)",
        line("VERIFIER_AGENT_SYSPROMPT"),
        line("VERIFIER_AGENT_SERVERURL"),
        line("VERIFIER_AGENT_SERVERURL_API_KEY"),
        line("VERIFIER_AGENT_ENDPOINT"),
        "",
        "# MCP connectors (Chat tab only) are declared only in",
        "# mcp_connectors/externals.json: their per-entry ``enable`` flag is",
        "# the single source of truth, no .env key is involved.",
        "# LSP language servers (Chat tab only) — ids of the servers in",
        "# lsp_connectors/lsp_servers.json to activate on top of their",
        "# per-entry ``enable`` flag (both must agree).",
        line("LSP_ENABLED"),
        "# Cap on the tool loop of one chat reply, MCP + LSP together.",
        line("MAX_TOOL_ITERATION"),
        "",
    ]
    return "\n".join(lines)


def max_tool_iteration() -> int:
    """Tool-iteration cap for one chat reply; invalid or unset values fall
    back to ``DEFAULT_MAX_TOOL_ITERATION``.

    The key was renamed ``MAX_TOOL_ITERATION`` because the cap covers the
    whole tool loop of one chat reply — MCP connectors and LSP language
    servers share the single counter (and the UI's progress gauge).
    ``MAX_MCP_ITERATION`` is still honoured so an existing .env keeps working.
    """
    raw = (
        os.environ.get("MAX_TOOL_ITERATION", "")
        or os.environ.get("MAX_MCP_ITERATION", "")
    ).strip()
    if not raw:
        return DEFAULT_MAX_TOOL_ITERATION
    try:
        return max(1, int(raw))
    except ValueError:
        return DEFAULT_MAX_TOOL_ITERATION


def lsp_enabled_ids() -> List[str]:
    """Language-server ids activated for the Chat tab from ``LSP_ENABLED``.

    Comma-separated, deduplicated, order-preserving; empty entries ignored.
    A server is only actually started when this list contains its id *and*
    its ``enable`` flag is true in ``lsp_servers.json`` — the env key lets a
    whole prepared config file stay dormant.
    """
    ids: List[str] = []
    for part in os.environ.get("LSP_ENABLED", "").split(","):
        part = part.strip()
        if part and part not in ids:
            ids.append(part)
    return ids


def _encode_value(value: str) -> str:
    """Quote a .env value when it contains characters dotenv would misread."""
    if any(char in value for char in '#"$\\ \n\t') or "\n" in value:
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        escaped = escaped.replace("\n", "\\n").replace("\t", "\\t")
        return f'"{escaped}"'
    return value
