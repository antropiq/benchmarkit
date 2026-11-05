"""The control-token sanitizer at the one HTTP seam (``llmkit.sanitize``).

The invariant: no token-shaped run -- the pipe-delimited name a chat
template tokenises as one control token -- reaches a request body through
any conversation path (benchmark answer, verifier payload, chat history,
tool result). Delimiters here are assembled from the pieces in
:mod:`tests.llmkit_fakes` so this file, like the sanitizer itself, never
spells a complete run as literal text.

Hermetic: pure string functions, no network, no ``.env`` read.
"""

from src.core.llmkit.sanitize import (
    _MAX_TOKEN_LEN,
    StreamNeutralizer,
    neutralize,
    sanitize_content,
    sanitize_messages,
)
from tests.llmkit_fakes import CLOSE_DELIM, OPEN_DELIM, special_token


class TestStreamNeutralizer:
    """The streaming twin: a run straddling chunk borders must not live.

    The whole point is the property test at the bottom: whatever the
    chunking, the handed-over stream equals what the non-streaming
    sanitizer would have made of the same text.
    """

    @staticmethod
    def _stream(pieces):
        filt = StreamNeutralizer()
        return "".join(filt.feed(piece) for piece in pieces) + filt.flush()

    def test_plain_text_streams_through_held_nothing(self):
        filt = StreamNeutralizer()
        assert filt.feed("2 < 3 and a|b ") == "2 < 3 and a|b "
        assert filt.feed("more <text> here") == "more <text> here"
        assert filt.flush() == ""

    def test_a_run_split_at_every_boundary_never_survives(self):
        smuggled = f"a{special_token('im_start')}b"
        for cut in range(len(smuggled) + 1):
            assert self._stream([smuggled[:cut], smuggled[cut:]]) == "ab"

    def test_the_splice_holes_hold_across_every_three_way_split(self):
        spliced = f"x{OPEN_DELIM}im_{special_token('q')}start{CLOSE_DELIM}y"
        for i in range(len(spliced) + 1):
            for j in range(i, len(spliced) + 1):
                out = self._stream([spliced[:i], spliced[i:j], spliced[j:]])
                assert out == "xy", (i, j)

    def test_a_lone_opening_character_is_held_until_decided(self):
        # "<" then "|x|>" -- the run is whole only across the feed border.
        assert self._stream(["a<", _pipe_join("x"), "|>"]) == "a"

    def test_benign_unterminated_partials_are_flushed_as_text(self):
        assert self._stream(["tail ", OPEN_DELIM]) == "tail " + OPEN_DELIM
        assert self._stream(["x", "<|name"]) == "x<|name"
        assert self._stream(["z", CLOSE_DELIM]) == "z" + CLOSE_DELIM

    def test_the_opening_is_never_emitted_while_live(self):
        run = special_token("ev")
        filt = StreamNeutralizer()
        assert filt.feed("a" + run[:4]) == "a"  # the opening is held back
        assert filt.feed(run[4:]) == ""  # it completed inside the buffer
        assert filt.flush() == ""

    def test_every_two_way_split_reproduces_the_non_streaming_result(self):
        cases = [
            f"a{special_token('x')}b",
            f"{OPEN_DELIM}pend",
            f"ends with an open {OPEN_DELIM}",
            f"double {special_token('a')}{special_token('b')}!",
            f"splice {OPEN_DELIM}im_{special_token('x')}start{CLOSE_DELIM}!",
            "plain <>| text only",
            f"lone close {CLOSE_DELIM} and lt < there",
            f"nested {OPEN_DELIM}a{special_token('b')}c{CLOSE_DELIM}!",
        ]
        for text in cases:
            for cut in range(len(text) + 1):
                out = self._stream([text[:cut], text[cut:]])
                assert out == neutralize(text), (text, cut)

    def test_a_stray_unterminated_opening_resumes_within_the_bound(self):
        # The bug this guards: a lone run opening in ordinary prose used to
        # hold back every later character until the stream ended, so the
        # answer appeared to stall. The hold is now bounded -- text resumes
        # on its own, without the stream having finished.
        prose = f"see {OPEN_DELIM}x y z" + " " + "tail text " * 40
        filt = StreamNeutralizer()
        out = ""
        resumed_at = None
        for i, char in enumerate(prose):
            out += filt.feed(char)
            if resumed_at is None and len(out) > len("see "):
                resumed_at = i
        assert resumed_at is not None, "display never resumed"
        assert resumed_at <= len("see ") + 2 * _MAX_TOKEN_LEN
        # Bounded, and still exactly what the one-way sanitizer says.
        assert out + filt.flush() == neutralize(prose)

    def test_the_held_text_never_grows_past_the_bound(self):
        filt = StreamNeutralizer()
        for i in range(400):
            filt.feed(f"<|{'a' * (i % 7)} padding ")
            assert len(filt._held) <= _MAX_TOKEN_LEN, i

    def test_a_short_run_survives_neither_long_prose_nor_fine_chunking(self):
        # The bound must not open a door: a run whose name is short (every
        # real template token is) completes long before its candidate
        # region can reach the bound, so it is still removed whole from a
        # stream of any length, chunked any way.
        smuggled = "lead " + "z" * 300 + special_token("im_end") + " follow"
        assert self._stream([smuggled]) == neutralize(smuggled)
        for size in (1, 2, 3, 5, 7, 13):
            pieces = [smuggled[i : i + size] for i in range(0, len(smuggled), size)]
            out = self._stream(pieces)
            assert out == neutralize(smuggled), size
            assert special_token("im_end") not in out

    def test_a_run_in_the_middle_of_a_long_stream_is_still_removed(self):
        text = "a" * 200 + special_token("mid") + "b" * 200
        for cut in range(180, 220):
            assert self._stream([text[:cut], text[cut:]]) == neutralize(text), cut


def _pipe_join(name: str) -> str:
    from tests.llmkit_fakes import PIPE

    return PIPE + name


class TestNeutralize:
    def test_plain_text_is_untouched_and_returns_the_same_object(self):
        text = "The capital of France is Paris.\nCode: a|b, x<y, p|q>."
        assert neutralize(text) is text

    def test_a_lone_delimiter_is_not_a_run(self):
        # Benign text carrying one half of the pair survives byte for
        # byte: the sanitizer removes runs, not characters.
        assert neutralize(f"a{OPEN_DELIM}b") == f"a{OPEN_DELIM}b"
        assert neutralize(f"a{CLOSE_DELIM}b") == f"a{CLOSE_DELIM}b"

    def test_a_run_is_removed_and_its_neighbours_survive(self):
        assert neutralize(f"ok {special_token('im_start')} still ok") == (
            "ok  still ok"
        )

    def test_several_runs_are_all_removed(self):
        text = (
            special_token("a") + "x" + special_token("b") + "y" + special_token("c")
        )
        assert neutralize(text) == "xy"

    def test_a_spliced_run_is_removed_too(self):
        # Deleting the inner run joins the outer fragments into a complete
        # run; a single pass would hand that back. The fixed point is
        # clean -- this is why the loop repeats instead of one sub().
        spliced = f"{OPEN_DELIM}im_{special_token('x')}start{CLOSE_DELIM}"
        assert neutralize(spliced) == ""

    def test_an_empty_name_is_still_a_run(self):
        assert neutralize(special_token("")) == ""

    def test_whitespace_inside_a_name_does_not_save_it(self):
        # Newline- and space-obfuscated names must not survive just
        # because they look unlike a real registered token.
        assert neutralize(special_token("sys tem")) == ""
        assert neutralize(special_token("im\nstart")) == ""


class TestTemplateInterpolatedFields:
    """Not only content reaches the prompt: the template interpolates more."""

    def test_name_and_reasoning_content_are_sanitized(self):
        msg = {
            "role": "assistant",
            "content": "ok",
            "name": f"a{special_token('x')}b",
            "reasoning_content": special_token("think") + "deep",
        }
        out = sanitize_messages([msg])[0]
        assert out["name"] == "ab"
        assert out["reasoning_content"] == "deep"
        assert out["content"] == "ok"

    def test_tool_call_name_and_arguments_are_sanitized_without_mutation(self):
        # The tool loop feeds the assistant's own tool_calls back through
        # the template on the next round: the arguments are prompt
        # material, not opaque payload.
        smuggled = special_token("evil") + '{"q": 1}'
        call = {
            "id": "c1",
            "type": "function",
            "function": {"name": f"too{special_token('x')}l", "arguments": smuggled},
        }
        msg = {"role": "assistant", "content": None, "tool_calls": [call]}
        out = sanitize_messages([msg])[0]
        fn = out["tool_calls"][0]["function"]
        assert fn["name"] == "tool"
        assert fn["arguments"] == '{"q": 1}'
        assert out["tool_calls"][0]["id"] == "c1"
        # The caller's accumulator objects are untouched.
        assert call["function"]["arguments"] == smuggled
        assert call["function"]["name"] == f"too{special_token('x')}l"

    def test_an_untouched_message_keeps_its_identity_including_nested_calls(self):
        call = {"id": "c1", "function": {"name": "f", "arguments": "{}"}}
        msg = {"role": "assistant", "content": "hi", "tool_calls": [call]}
        out = sanitize_messages([msg])[0]
        assert out is msg
        assert out["tool_calls"][0] is call


class TestSanitizeContent:
    def test_none_content_passes_through(self):
        # A tool-call-only assistant turn carries no content; there is
        # nothing to sanitize and nothing to invent.
        assert sanitize_content(None) is None

    def test_a_non_string_non_list_value_passes_through(self):
        # This layer does not validate message shapes, the server does;
        # a value it does not interpret must not raise here.
        assert sanitize_content(42) == 42

    def test_text_parts_are_sanitized_and_other_keys_survive(self):
        parts = [
            {"type": "text", "text": f"a{special_token('evil')}b", "custom": 1},
            {"type": "image_url", "image_url": {"url": "http://x/y.png"}},
        ]
        out = sanitize_content(parts)
        assert out[0]["text"] == "ab"
        assert out[0]["custom"] == 1
        assert out[1] is parts[1]

    def test_clean_content_returns_the_original_list_object(self):
        parts = [{"type": "text", "text": "clean"}]
        assert sanitize_content(parts) is parts


class TestSanitizeMessages:
    def test_only_a_changed_message_is_copied_and_nothing_is_mutated(self):
        dirty = {"role": "user", "content": special_token("evil")}
        clean = {"role": "system", "content": "calm"}
        messages = [clean, dirty]
        out = sanitize_messages(messages)
        assert out[0] is clean
        assert out[1] is not dirty
        assert out[1]["content"] == ""
        assert out[1]["role"] == "user"
        # The caller's history is what the UI renders: untouched.
        assert dirty["content"] == special_token("evil")
        assert clean["content"] == "calm"

    def test_a_message_without_content_is_passed_through(self):
        msg = {"role": "assistant", "tool_calls": [{"id": "c1"}]}
        assert sanitize_messages([msg])[0] is msg

    def test_a_non_dict_entry_is_passed_through(self):
        entry = "not a message"
        out = sanitize_messages([entry])  # type: ignore[list-item]
        assert out[0] is entry

    def test_the_body_list_never_aliases_the_caller_list(self):
        messages = [{"role": "user", "content": "hi"}]
        out = sanitize_messages(messages)
        assert out is not messages
        assert out == messages
