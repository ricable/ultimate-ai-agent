import pytest
from sparc_cli.text.processing import truncate_output

def test_truncate_output():
    """Test that truncate_output correctly truncates long strings."""
    # Test short string (no truncation)
    short = "Hello world"
    assert truncate_output(short) == short
    
    # Test long string (truncation)
    long = "x" * 10000
    truncated = truncate_output(long)
    assert len(truncated) < len(long)
    assert "..." in truncated
    
    # Test empty string
    assert truncate_output("") == ""
    
    # Test None
    assert truncate_output(None) == ""
