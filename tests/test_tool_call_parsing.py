"""Unit tests for the agent's tool-call parser and thinking splitter.

These target pure functions in app.agent — no model load, no GPU. Importing
app.agent pulls in app.model_manager (which imports torch/transformers) but
never instantiates or loads a model, so these run on CPU.

Run with:  python -m pytest tests/
"""

from app.agent import (
    _parse_tool_calls,
    _strip_tool_call_markup,
    _split_thinking,
)


# --- _parse_tool_calls -----------------------------------------------------


def test_call_inside_think_block():
    """A genuine tool call emitted inside a [THINK] block must still be parsed
    (the core item-5 regression: it used to fall through to 'final response')."""
    text = (
        "[THINK]The user wants a calculation, I should use the calculator.\n"
        '[TOOL_CALLS]calculator[ARGS]{"expression": "42 * 17"}[/THINK]'
    )
    calls = _parse_tool_calls(text)
    assert calls == [{"name": "calculator", "arguments": {"expression": "42 * 17"}}]


def test_call_followed_by_trailing_text():
    """The old primary regex required the JSON to be immediately followed by
    </s> or end-of-string. The reasoning model keeps generating afterwards, so
    trailing content must not defeat the match."""
    text = (
        '[TOOL_CALLS]read_file[ARGS]{"filename": "program_info.txt"}\n'
        "Now I'll wait for the result and continue reasoning."
    )
    calls = _parse_tool_calls(text)
    assert calls == [{"name": "read_file", "arguments": {"filename": "program_info.txt"}}]


def test_call_with_trailing_eos_and_whitespace():
    text = '[TOOL_CALLS]list_files[ARGS]{}</s>   \n'
    calls = _parse_tool_calls(text)
    assert calls == [{"name": "list_files", "arguments": {}}]


def test_nested_brace_arguments():
    """Balanced raw_decode must capture the whole object; the old non-greedy
    \\{.*?\\} stopped at the first closing brace."""
    text = (
        '[TOOL_CALLS]search[ARGS]'
        '{"query": "x", "options": {"limit": 5, "nested": {"deep": true}}} trailing'
    )
    calls = _parse_tool_calls(text)
    assert calls == [{
        "name": "search",
        "arguments": {"query": "x", "options": {"limit": 5, "nested": {"deep": True}}},
    }]


def test_multiple_calls():
    text = (
        '[TOOL_CALLS]list_files[ARGS]{} '
        '[TOOL_CALLS]read_file[ARGS]{"filename": "team_roster.txt"} done'
    )
    calls = _parse_tool_calls(text)
    assert calls == [
        {"name": "list_files", "arguments": {}},
        {"name": "read_file", "arguments": {"filename": "team_roster.txt"}},
    ]


def test_no_call_returns_empty():
    """A genuine prose response must not be mistaken for a tool call."""
    text = "The answer is 714. I used the calculator to multiply 42 by 17."
    assert _parse_tool_calls(text) == []


# --- _strip_tool_call_markup ----------------------------------------------


def test_strip_removes_call_markup():
    text = (
        "Let me check the files. "
        '[TOOL_CALLS]list_files[ARGS]{} '
        "I'll read the right one next."
    )
    stripped = _strip_tool_call_markup(text)
    assert "[TOOL_CALLS]" not in stripped
    assert "[ARGS]" not in stripped
    assert "Let me check the files." in stripped
    assert "I'll read the right one next." in stripped


def test_strip_leaves_text_without_calls_unchanged():
    text = "No tool calls here, just prose."
    assert _strip_tool_call_markup(text) == text


# --- _split_thinking -------------------------------------------------------


def test_split_think_tag():
    thinking, response = _split_thinking("<think>reasoning here</think>final answer")
    assert thinking == "reasoning here"
    assert response == "final answer"


def test_split_thinking_tag():
    """The model sometimes emits <thinking>...</thinking> despite the prompt
    asking for [THINK]...[/THINK]; both must be recognized."""
    thinking, response = _split_thinking("<thinking>reasoning here</thinking>final answer")
    assert thinking == "reasoning here"
    assert response == "final answer"


def test_split_bracket_think_marker():
    thinking, response = _split_thinking("[THINK]reasoning here[/THINK]final answer")
    assert thinking == "reasoning here"
    assert response == "final answer"


def test_split_no_thinking_returns_none():
    thinking, response = _split_thinking("just a plain answer")
    assert thinking is None
    assert response == "just a plain answer"


def test_tool_call_inside_thinking_tag_is_honored_and_stripped():
    """Combined: a call inside <thinking>...</thinking>. After stripping the
    call markup the thinking still extracts cleanly, and the call is parsed."""
    raw = (
        "<thinking>I need to multiply these numbers. "
        '[TOOL_CALLS]calculator[ARGS]{"expression": "100 / 5"}</thinking>'
    )
    calls = _parse_tool_calls(raw)
    assert calls == [{"name": "calculator", "arguments": {"expression": "100 / 5"}}]

    display = _strip_tool_call_markup(raw)
    thinking, _response = _split_thinking(display)
    assert thinking is not None
    assert "[TOOL_CALLS]" not in thinking
    assert "I need to multiply these numbers." in thinking
