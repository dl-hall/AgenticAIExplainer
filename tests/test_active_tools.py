"""Unit tests for per-tool enabling (bucket 05).

Like test_tool_call_parsing, these import app.agent / app.tools but never load
a model. Agent.__init__ only stores the model manager reference and seeds the
enabled-tool set, so Agent(None) is sufficient to exercise get_active_tools().

Run with:  python -m pytest tests/
"""

from app.agent import Agent
from app.tools import tool_names


def _names(tools):
    return [t["function"]["name"] for t in tools]


def test_tool_names_are_the_three_demo_tools():
    assert tool_names() == ["calculator", "read_file", "list_files"]


def test_all_tools_enabled_by_default():
    agent = Agent(None)
    assert _names(agent.get_active_tools()) == tool_names()


def test_disabling_calculator_drops_only_calculator():
    agent = Agent(None)
    agent.enabled_tools = {"read_file", "list_files"}
    active = _names(agent.get_active_tools())
    assert "calculator" not in active
    assert "read_file" in active
    assert "list_files" in active


def test_empty_set_returns_no_tools():
    agent = Agent(None)
    agent.enabled_tools = set()
    assert agent.get_active_tools() == []


def test_active_tools_preserve_definition_order():
    """get_active_tools filters TOOL_DEFINITIONS, so order follows the source
    of truth regardless of the set's iteration order."""
    agent = Agent(None)
    agent.enabled_tools = {"list_files", "calculator"}
    assert _names(agent.get_active_tools()) == ["calculator", "list_files"]
