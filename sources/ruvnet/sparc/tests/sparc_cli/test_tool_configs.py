from sparc_cli.tool_configs import (
    get_read_only_tools,
    get_research_tools,
    get_planning_tools,
    get_implementation_tools,
    get_chat_tools
)

def test_get_read_only_tools():
    """Test that get_read_only_tools returns expected tools."""
    tools = get_read_only_tools()
    assert isinstance(tools, list)
    assert len(tools) > 0

def test_get_read_only_tools_with_human():
    """Test that get_read_only_tools includes human interaction when enabled."""
    tools = get_read_only_tools(human_interaction=True)
    assert isinstance(tools, list)
    assert len(tools) > 0
    # Should have one more tool than without human interaction
    assert len(tools) == len(get_read_only_tools()) + 1

def test_get_research_tools():
    """Test that get_research_tools returns expected tools."""
    tools = get_research_tools()
    assert isinstance(tools, list)
    assert len(tools) > 0

def test_get_planning_tools():
    """Test that get_planning_tools returns expected tools."""
    tools = get_planning_tools()
    assert isinstance(tools, list)
    assert len(tools) > 0

def test_get_implementation_tools():
    """Test that get_implementation_tools returns expected tools."""
    tools = get_implementation_tools()
    assert isinstance(tools, list)
    assert len(tools) > 0

def test_get_chat_tools():
    """Test that get_chat_tools returns expected tools."""
    tools = get_chat_tools()
    assert isinstance(tools, list)
    assert len(tools) > 0
