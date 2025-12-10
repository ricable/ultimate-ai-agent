import pytest
from sparc_cli.tools import list_directory_tree
from sparc_cli.tools.list_directory import load_gitignore_patterns, should_ignore

def test_list_directory_tree():
    """Test that list_directory_tree returns directory structure."""
    result = list_directory_tree(path=".")
    assert isinstance(result, dict)
    assert "tree" in result
    assert isinstance(result["tree"], str)

def test_load_gitignore_patterns():
    """Test loading of gitignore patterns."""
    patterns = load_gitignore_patterns()
    assert isinstance(patterns, list)
    # Should at least have common patterns
    assert len(patterns) > 0

def test_should_ignore():
    """Test gitignore pattern matching."""
    # Test node_modules
    assert should_ignore("node_modules") is True
    assert should_ignore("path/to/node_modules") is True
    
    # Test .git
    assert should_ignore(".git") is True
    assert should_ignore("path/to/.git") is True
    
    # Test regular directory
    assert should_ignore("src") is False
    assert should_ignore("path/to/src") is False

def test_list_directory_tree_with_path():
    """Test list_directory_tree with specific path."""
    result = list_directory_tree(path=".")
    assert isinstance(result, dict)
    assert "tree" in result
    assert isinstance(result["tree"], str)

def test_list_directory_tree_nonexistent():
    """Test list_directory_tree with nonexistent path."""
    result = list_directory_tree(path="nonexistent")
    assert isinstance(result, dict)
    assert "error" in result["tree"].lower()
