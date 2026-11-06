"""am-02: the words behind the modes — the seeded document, the app-owned mode.

am-01's :mod:`core.agent_modes` decides whether a modes document is well-formed;
this module holds the one document benchmarkit **seeds**, the one mode it never
lets anyone edit, and the batch-conventions block that mode carries. Nothing here
reads a file, opens a route or calls a model: it is vocabulary and rendering.

Why a Python literal and not a ``.json`` on disk (the ``editorfs/policy_seed.py``
argument, applied harder): the bytes the seeder writes (am-03) and the fallback
used when the file is broken are **the same object**, and a test asserts that
object round-trips through am-01's loader with zero findings. A file would let
the seed and the reader agree by luck.

The personas are **instructions to the model, in the second person**, never
descriptions of a role, and an editable one stays inside ~15 lines: this text is
paid for on every turn of the thread, and a mode's job is to remove options, not
to decorate. The byte cap is am-01's to enforce — every entry here sits an order
of magnitude under it, which the module's own self-check also holds.

Two rules the tests hold, because prose rots quietly:

* a persona names the other modes only through the label constants below, so a
  rename cannot leave a stale phrase in someone's prompt;
* a persona never claims to have read ``AGENTS.md`` — it says "the project
  instructions above", which is what :func:`config.agent_instructions_block`
  actually puts there, and every mode receives it.
"""

import json
from typing import Any, Dict, List, Tuple

from src.core import agent_modes, batch
from src.core.agent_modes import AgentMode

#: The three mode ids benchmarkit seeds, and their dropdown labels. Both are
#: constants because the personas quote the labels in their own first line (see
#: the module docstring): one name per mode, in the code and in the prose.
CHAT_MODE_ID = "chat"
CODE_MODE_ID = "code"
ARCHITECT_MODE_ID = "architect"

CHAT_LABEL = "Chat"
CODE_LABEL = "Code"
ARCHITECT_LABEL = "Architect"

#: The app-owned mode (am-00 D9): not in the seeded document, not editable, and
#: the only mode that carries generated blocks.
TODO_BUILDER_MODE_ID = "todoBuilder"
TODO_BUILDER_LABEL = "Todo builder"


# -- the seeded document --------------------------------------------------------

#: Prose for the human editing the file by hand. JSON has no comments, and this
#: file is edited by hand outside the app (D6), so it has to explain its own
#: vocabulary: the four things a first-time editor cannot infer from the keys.
AGENT_MODES_README: Tuple[str, ...] = (
    "benchmarkit reads this file to build the agent-mode dropdown.",
    "The keys in it are matched exactly: an unknown key is refused with an error,",
    "not ignored, so a misspelling is reported instead of quietly switching an",
    "agent off. The allowed top-level keys are version, readme, default and",
    "agents; the allowed keys of one mode are label, mcptools, lsptools and",
    "systemMessage.",
    "The entries under agents are kept in the order you wrote them, and that is",
    "the order of the dropdown.",
    'systemMessage is the persona sent to the model. Write it as a string, or as',
    "a list with one line per entry (readable inside JSON). An empty or missing",
    "systemMessage means: use LLM_SYSPROMPT from .env instead.",
    "mcptools and lsptools are JSON booleans, true or false - not 1/0, not",
    '"yes"/"no". A missing one means false. They decide whether that hub reaches',
    "the model at all; Settings still decides which of its connectors are on.",
    '"todoBuilder" is defined by benchmarkit and cannot be set here; an entry',
    "for it is ignored.",
    "benchmarkit seeds this file when a folder becomes the working location, then",
    "never writes it again. It is machine-local and gitignored, so each checkout",
    "edits its own agents.",
)


# -- the three editable personas ------------------------------------------------
#
# Second-person instructions, not role descriptions. ``code`` is what the app
# already does, plus the "you implement" framing that was missing; the others
# work by taking options away. Each first line names its own mode through the
# label constant, so a rename cannot leave a stale word in someone's prompt.

CHAT_PERSONA: Tuple[str, ...] = (
    "You are a helpful assistant. You can answer questions of every kind, even some not related to the current project or unrelated to software engineering.",
    "If you have tools, you can use them to read files, fetch information from Internet, edit files, etc but **you must not edit, create new files in the current projects directory**.",
)

CODE_PERSONA: Tuple[str, ...] = (
    "You are an experienced senior software engineering working on an active project workspace.",
    "You can use all available tools to accomplish the task that is given to you.",
    "* Always follow the given user instructions or the spec file provided.",
    "* Do not make design choice of your own. When a spec is provided, trust it and implement defined features. When there is a missing piece in user's provided instruction, ask for clarification but do not make up a design choice by your own.",
    "* Do not make up features on your own. You must only add code to handle the provided spec features or the set of instructions given by the user.",
    "* When using tools, report feedback to the user as soon as you can.",
    "* Do not **overthing** and to not **overengineering** solutions to be implemented. Follow the *KISS* principle.",
    "* If a file is bigger than 2000 lines, try to figure out how to split it by refactoring the existing code without breaking the logic.",
    "* Always write unit test cases for the code you have provided.",
    "* During the session, if you find a BUG not related to your task, report it at the end but do not try to fix it.",
    "* Definition of Done is: 1) The feature is implemented, 2) all tests are green.",
)

ARCHITECT_PERSONA: Tuple[str, ...] = (
    "You are an experienced senior software architect working on an active project workspace.",
    "You can use all available tools to accomplish the task that is given to you but:",
    "- Do not modify any source code. You only modify files related to documentation and specification witch are located in **docs** folder.",
    "> If the docs folder does not exist create it.",
    "* The user will provide a draft spec description about a new feature.",
    "* You will focus on it and:",
    "    * Provide guidance to refine the feature.",
    "    * Ask questions to clarify.",
    "    * Provide advise using best practices and common software engineering design patterns.",
    "* You will produce a clean and detailed documentation in markdown format.",
    "* You must not go too much into technical details. Keep focus on the functionnal need. It needs to be crystal clear.",
    "* Keep in mind that this specification file will later be declined into task items.",
)

#: The reserved mode's persona. The lot conventions are **not** in this text:
#: they are the generated block of :func:`lot_conventions_block`, because a
#: prompt that restates a rule the code enforces is a rule that can go stale.
TODO_BUILDER_PERSONA: Tuple[str, ...] = (
    f"You are working in {TODO_BUILDER_LABEL} mode: you turn a feature into a",
    "task lot that the batch runner will later execute one task at a time. You",
    "never implement any of it.",
    "- You must trust the given spec. It contains every thing you need.",
    "- All questions have been resolved in the spec. Do not ask any question.",
    "- Write the lot with the file tools, as markdown, following the lot",
    "  conventions below exactly: one big-picture file, then one per task,",
    "  numbered in execution order.",
    "- Task size is one session with 131K context.",
    "- Every task file says: what to deliver, which files it touches, the rules",
    "  it must not break (quote them from the project instructions), which task",
    "  numbers it depends on, and how to tell it is done. A task must be",
    "  executable by a reader who never saw this conversation.",
    "- Never write a task that is \"implement the feature\", and put no code in",
    "  the lot: a task says *what*, not *how* — except where the how is an",
    "  invariant of this project.",
    "- If the feature is too big for one lot, say so and ask for the big picture",
    "  before writing anything.",
    "- After writing, report the task numbers and titles, and say the lot is",
    "  ready to launch. Do not launch it yourself.",
)


#: The document am-03 seeds and am-01 falls back to — one object, two uses.
#: ``todoBuilder`` is deliberately absent from it (D9); :func:`resolve_modes`
#: appends the app-owned mode last, whatever a file says.
DEFAULT_AGENTS_DOCUMENT: Dict[str, Any] = {
    "version": agent_modes.AGENT_MODES_VERSION,
    "readme": list(AGENT_MODES_README),
    "default": CODE_MODE_ID,
    "agents": {
        CHAT_MODE_ID: {
            "label": CHAT_LABEL,
            "mcptools": False,
            "lsptools": False,
            "systemMessage": list(CHAT_PERSONA),
        },
        CODE_MODE_ID: {
            "label": CODE_LABEL,
            "mcptools": True,
            "lsptools": True,
            "systemMessage": list(CODE_PERSONA),
        },
        ARCHITECT_MODE_ID: {
            "label": ARCHITECT_LABEL,
            "mcptools": True,
            "lsptools": False,
            "systemMessage": list(ARCHITECT_PERSONA),
        },
    },
}


def default_modes() -> List[AgentMode]:
    """The built-in modes, run through am-01's validator (never hand-built).

    The fallback path and the seeded path then share one definition, which is the
    whole point: a key the seeder writes that the validator does not accept fails
    here, at import of the fallback, instead of showing up as findings in every
    workspace on the machine.

    The assertion is about *our own* content, so raising is right; am-01's
    never-raises promise is about the file on disk.
    """
    document = json.loads(json.dumps(DEFAULT_AGENTS_DOCUMENT))
    modes, declared_default, findings, usable = agent_modes._validated_document(
        document, "<built-in defaults>"
    )
    assert usable and not findings, f"the seeded modes document is invalid: {findings}"
    assert declared_default == DEFAULT_AGENTS_DOCUMENT["default"], (
        "the seeded default is not one of the seeded modes"
    )
    return modes


# -- the reserved mode ----------------------------------------------------------

def lot_conventions_block() -> str:
    """The conventions a written lot must follow, rendered from ``core/batch.py``.

    These are the runner's rules, and the runner enforces them in code — so the
    prompt quotes the **live constants** instead of restating them in a string
    that can go stale. Every number and folder name below comes from a named
    attribute of :mod:`core.batch` at call time, which is what lets a test patch
    one constant and see this text change (the drift guard).
    """
    separators = " or ".join(f"`{char}`" for char in batch.NUMBER_SEPARATORS)
    return (
        "The lot conventions below are enforced by benchmarkit, not suggestions:\n"
        "a file that does not match them is not a task.\n"
        f"- Lot folder: `{batch.DEFAULT_LOT_DIR}/`, relative to the working location.\n"
        "- File names: `<prefix><NN><sep><slug>.md`, where `<prefix>` matches\n"
        f"  {batch.PREFIX_PATTERN} and `<sep>` is {separators}. The number is a token\n"
        "  of its own: `<prefix>-1x` is a file name, not task 1.\n"
        f"- One file per task. At most {batch.MAX_BATCH_TASKS} tasks.\n"
        "- The NN == 0 file is the lot's big picture; its body is sent as extra\n"
        f"  context on every task turn, capped at {batch.MAX_BATCH_LOT_BYTES} bytes,\n"
        "  so it is the map, not a preamble.\n"
        f"- A task is done when its file has moved into `{batch.DONE_SUBFOLDER}/`. The\n"
        "  runner re-lists the folder and never reads task bodies: the agent opens\n"
        "  its own file. Write each task so a reader who only has that file and the\n"
        "  big picture can finish it."
    )


def todo_builder_persona() -> str:
    """The reserved mode's persona text, joined from its lines."""
    return "\n".join(TODO_BUILDER_PERSONA)


def todo_builder_mode() -> AgentMode:
    """The app-owned mode (D9): persona and instructions rendered from code.

    Never built from the modes file — :func:`core.agent_modes.resolve_modes`
    appends this last whatever the file declares, and an entry for the id there
    is answered with a note. ``instructions`` carries the generated lot
    conventions; am-04 renders them after AGENTS.md and before the lot's own big
    picture, so the text stays in the same order for a re-read.
    """
    return AgentMode(
        id=TODO_BUILDER_MODE_ID,
        label=TODO_BUILDER_LABEL,
        system_message=todo_builder_persona(),
        mcptools=True,
        lsptools=False,
        editable=False,
        instructions=(lot_conventions_block(),),
    )


def default_agent_modes_text() -> str:
    """The bytes the seeder writes (am-03), rendered from the literal.

    Same shape as ``policy_seed.default_policy_text()``: one ``json.dumps`` of the
    document, indented, with a trailing newline, so the file a workspace receives
    is readable by hand and is the object the loader accepts, byte for byte. It
    lives here rather than in the seeder so the seeded bytes have exactly one
    home: the fallback and the file cannot then disagree.

    ``ensure_ascii=False`` because this file is edited by hand outside the app
    (D6): the personas use real prose, and a file whose dashes arrive as
    ``\u2014`` is a file nobody wants to read or patch.
    """
    return json.dumps(DEFAULT_AGENTS_DOCUMENT, indent=2, ensure_ascii=False) + "\n"


# -- self-check ----------------------------------------------------------------

def _self_check() -> None:
    """Fail at import on the content rules this module states about itself.

    A persona typed over am-01's cap, a label that no longer matches the id the
    rest of the feature keys on, or a seeded ``default`` naming a mode the
    document does not declare are all mistakes a later edit can make silently —
    and every one of them costs someone their persona text on every turn.
    """
    # The persona checks come first: they are this module's own rules, and they
    # speak in this module's voice. The document round-trip below answers through
    # am-01's validator, whose findings are the same fact in the loader's words.
    personas = (
        (CHAT_MODE_ID, CHAT_PERSONA),
        (CODE_MODE_ID, CODE_PERSONA),
        (ARCHITECT_MODE_ID, ARCHITECT_PERSONA),
        (TODO_BUILDER_MODE_ID, TODO_BUILDER_PERSONA),
    )
    for mode_id, persona in personas:
        if not persona:
            raise ValueError(f"the {mode_id} persona is empty")
        size = len("\n".join(persona).encode("utf-8"))
        if size > agent_modes.MAX_MODE_SYSTEM_MESSAGE_BYTES // 2:
            raise ValueError(
                f"the {mode_id} persona is {size} bytes, over half the "
                f"{agent_modes.MAX_MODE_SYSTEM_MESSAGE_BYTES} cap"
            )
    declared = list(DEFAULT_AGENTS_DOCUMENT["agents"])
    if declared != [CHAT_MODE_ID, CODE_MODE_ID, ARCHITECT_MODE_ID]:
        raise ValueError(
            f"the seeded modes document declares {declared}, expected "
            f"{[CHAT_MODE_ID, CODE_MODE_ID, ARCHITECT_MODE_ID]}"
        )
    for reserved in agent_modes.RESERVED_MODE_IDS:
        if reserved in declared:
            raise ValueError(
                f"the reserved mode {reserved!r} must not be in the seeded document"
            )
    if TODO_BUILDER_MODE_ID not in agent_modes.RESERVED_MODE_IDS:
        raise ValueError(
            f"{TODO_BUILDER_MODE_ID!r} is not in agent_modes.RESERVED_MODE_IDS"
        )
    default_modes()


_self_check()

