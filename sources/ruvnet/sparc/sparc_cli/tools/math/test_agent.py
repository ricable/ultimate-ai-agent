"""
Tests for the ReActMathAgent implementation.
"""

import pytest
from unittest.mock import Mock, patch
from langchain.schema import BaseLanguageModel
from langchain.tools import BaseTool
from .agent import ReActMathAgent

# Fixtures
@pytest.fixture
def mock_llm():
    """Mock LLM for testing."""
    llm = Mock(spec=BaseLanguageModel)
    llm.predict.return_value = "Mock LLM response"
    return llm

@pytest.fixture
def mock_tools():
    """Mock mathematical tools for testing."""
    calculator = Mock(spec=BaseTool)
    calculator.name = "calculator"
    calculator.description = "Performs basic arithmetic operations"
    
    symbolic = Mock(spec=BaseTool)
    symbolic.name = "symbolic"
    symbolic.description = "Handles symbolic mathematics"
    
    matrix = Mock(spec=BaseTool)
    matrix.name = "matrix"
    matrix.description = "Performs matrix operations"
    
    return [calculator, symbolic, matrix]

@pytest.fixture
def math_agent(mock_llm, mock_tools):
    """Create ReActMathAgent instance with mock components."""
    return ReActMathAgent(llm=mock_llm, tools=mock_tools)

# Initialization Tests
def test_agent_initialization(math_agent, mock_tools):
    """Test proper agent initialization."""
    assert math_agent.llm is not None
    assert len(math_agent.tools) == len(mock_tools)
    assert hasattr(math_agent, 'analysis_chain')
    assert hasattr(math_agent, 'tool_selection_chain')
    assert hasattr(math_agent, 'reasoning_chain')

def test_prompt_setup(math_agent):
    """Test prompt template configuration."""
    assert math_agent.analysis_prompt is not None
    assert math_agent.tool_selection_prompt is not None
    assert math_agent.reasoning_prompt is not None
    assert "problem" in math_agent.analysis_prompt.input_variables

# Chain-of-Thought Tests
@pytest.mark.parametrize("problem,expected_steps", [
    ("2 + 2", 3),  # Simple arithmetic
    ("Solve x^2 + 2x + 1 = 0", 3),  # Algebraic equation
    ("Calculate determinant of [[1,2],[3,4]]", 3),  # Matrix operation
])
def test_reasoning_flow(math_agent, problem, expected_steps):
    """Test the complete reasoning flow with different problems."""
    result = math_agent.run(problem)
    assert len(result["steps"]) == expected_steps
    assert all(step["stage"] in ["analysis", "tool_selection", "reasoning"] 
              for step in result["steps"])

# Tool Usage Tests
def test_calculator_tool_usage(math_agent, mock_tools):
    """Test calculator tool invocation."""
    with patch.object(mock_tools[0], 'run') as mock_run:
        mock_run.return_value = "4"
        result = math_agent.run("2 + 2")
        assert "solution" in result
        mock_run.assert_called_once()

def test_symbolic_tool_usage(math_agent, mock_tools):
    """Test symbolic mathematics tool invocation."""
    with patch.object(mock_tools[1], 'run') as mock_run:
        mock_run.return_value = "x = -1"
        result = math_agent.run("Solve x + 1 = 0")
        assert "solution" in result
        mock_run.assert_called_once()

# Error Handling Tests
def test_invalid_tool_error(math_agent):
    """Test handling of invalid tool selection."""
    with patch.object(math_agent.agent_executor, 'run') as mock_run:
        mock_run.side_effect = ValueError("Invalid tool")
        result = math_agent.run("Use nonexistent tool")
        assert "error" in result
        assert result["confidence"] == 0.0

@pytest.mark.parametrize("error_type", [
    ValueError,
    RuntimeError,
    TypeError
])
def test_error_handling(math_agent, error_type):
    """Test various error scenarios."""
    with patch.object(math_agent.agent_executor, 'run') as mock_run:
        mock_run.side_effect = error_type("Test error")
        result = math_agent.run("Problematic input")
        assert "error" in result
        assert isinstance(result["error"], str)
        assert result["tools_used"] == []

# ReAct Framework Tests
def test_react_cycle_tracking(math_agent):
    """Test observation-thought-action cycle tracking."""
    result = math_agent.run("3 * 4")
    steps = result["steps"]
    
    # Verify analysis step
    assert steps[0]["stage"] == "analysis"
    
    # Verify tool selection step
    assert steps[1]["stage"] == "tool_selection"
    
    # Verify reasoning step
    assert steps[2]["stage"] == "reasoning"

def test_intermediate_steps_tracking(math_agent):
    """Test tracking of intermediate reasoning steps."""
    result = math_agent.run("5 - 3")
    assert "steps" in result
    assert isinstance(result["steps"], list)
    assert all("stage" in step for step in result["steps"])
    assert all("output" in step for step in result["steps"])

@pytest.mark.parametrize("confidence_threshold", [
    0.0,
    0.5,
    1.0
])
def test_confidence_scoring(math_agent, confidence_threshold):
    """Test confidence score calculation and thresholds."""
    with patch.object(math_agent.agent_executor, 'run') as mock_run:
        mock_run.return_value = {"confidence": confidence_threshold}
        result = math_agent.run("Test problem")
        assert "confidence" in result
        assert isinstance(result["confidence"], float)
        assert 0 <= result["confidence"] <= 1
