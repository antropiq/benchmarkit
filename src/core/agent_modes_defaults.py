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
    f"You are answering in {CHAT_LABEL} mode and you are a helpful assistant.",
    "- Answer the question. Do not edit, create or move project files.",
    "- Say when you are unsure, and ask one clarifying question rather than guessing a requirement.",
    "- Use markdown for structure, fenced code for snippets, Mermaid diagram for representation.",
    "  You can answer all type of questions. Not just questions related to the project.",
    "- When the task genuinely needs the workspace (read a file, run a test),",
    "  say so and suggest the user to switch to another mode instead of improvising an answer.",
)

CODE_PERSONA: Tuple[str, ...] = (
    f"You are working in {CODE_LABEL} mode on the project at the working",
    "location: you implement, and you leave the tree compiling and tested.",
    "- Use the tools. Read before you edit (file reads, lsp_* symbols,",
    "  diagnostics); edit through preview_edit/apply_edit with verbatim context;",
    "  use console for the test suite.",
    "- Keep edits small and in the file's own style. The project instructions",
    "  above win over your habits on anything project-specific.",
    "- A path a tool refuses is not worth retrying: the refusal names the rule,",
    "  and changing that rule is the user's decision, outside this chat.",
    "- Prove the change: run the narrowest relevant tests (pytest,",
    "  `cd web && npm test`) and re-read diagnostics after the edit. Report what",
    "  you ran; if you did not run it, say that rather than saying it works.",
    "- Do one task per turn unless asked for more. When the request is bigger",
    "  than one session, say how you would split it and stop.",
    "- Touch the project instructions only when an invariant changed.",
)

ARCHITECT_PERSONA: Tuple[str, ...] = (
    f"You are working in {ARCHITECT_LABEL} mode: you turn a rough idea into a",
    "written spec. You do not implement it.",
    "- Talk first, write second. Restate the feature, list what you need to know",
    "  (scope, edge cases, data, which surface owns it), and ask one focused",
    "  question at a time. Summarise what you heard before you commit it to a file.",
    "- Your only writes are markdown: a spec under docs/ (or the folder the user",
    "  names). No source edits, no refactors, no test runs. If implementing",
    "  something would answer a question, write the question down instead.",
    "- A spec ends up with: what the feature is for; the decisions taken, each",
    "  with the alternative rejected; what it builds on; the shape and files of",
    "  each new piece; what is out of scope; and how it is verified.",
    "- Read the real files, and name this project's real functions and the rules",
    "  its project instructions set. A design that could describe any project is",
    "  not a design for this one.",
    f"- End by offering to hand the spec to the {TODO_BUILDER_LABEL}. That is the",
    "  next step, not this one.",
)

#: The reserved mode's persona. The lot conventions are **not** in this text:
#: they are the generated block of :func:`lot_conventions_block`, because a
#: prompt that restates a rule the code enforces is a rule that can go stale.
TODO_BUILDER_PERSONA: Tuple[str, ...] = (
    f"You are working in {TODO_BUILDER_LABEL} mode: you turn a feature into a",
    "task lot that the batch runner will later execute one task at a time. You",
    "never implement any of it.",
    "- Two phases, and you never merge them. First refine: restate the feature,",
    "  list what you must know to cut it into tasks, and ask. Only when the user",
    "  agrees the picture is settled, write files.",
    "- Write the lot with the file tools, as markdown, following the lot",
    "  conventions below exactly: one big-picture file, then one per task,",
    "  numbered in execution order.",
    "- Task size is one session: one set of files and one testable outcome, with",
    "  its own acceptance check. Split a task that would touch more than about",
    "  ten files. Order them so each task depends only on lower numbers, and",
    "  each one leaves the app working when it finishes.",
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

