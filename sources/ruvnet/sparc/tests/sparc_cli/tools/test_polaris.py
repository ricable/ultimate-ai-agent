"""Tests for the PolarisOne tool."""

import pytest
from sparc_cli.tools.polaris import PolarisTool

def test_polaris_tool():
    """Test that the PolarisOne tool correctly processes text."""
    tool = PolarisTool()
    
    # Test basic query
    result = tool("What is the capital of France?")
    
    # Verify response structure
    assert isinstance(result, dict)
    assert "response" in result
    assert "token_weights" in result
    
    # Verify token weights
    token_weights = result["token_weights"]
    assert isinstance(token_weights, list)
    assert len(token_weights) > 0
    
    # Each token weight should be a (str, float) tuple
    for token, weight in token_weights:
        assert isinstance(token, str)
        assert isinstance(weight, float)
        assert 0 <= weight <= 1  # Weights should be between 0 and 1
