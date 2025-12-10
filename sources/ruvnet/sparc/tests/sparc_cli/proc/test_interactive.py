import pytest
from sparc_cli.proc.interactive import run_interactive_command

def test_run_interactive_command():
    """Test that run_interactive_command executes commands and returns output."""
    output, return_code = run_interactive_command(['echo', 'test'])
    assert output == b'test\n'
    assert return_code == 0

def test_run_interactive_command_error():
    """Test that run_interactive_command handles command errors."""
    output, return_code = run_interactive_command(['false'])
    assert return_code == 1

def test_run_interactive_command_invalid():
    """Test that run_interactive_command handles invalid commands."""
    with pytest.raises(FileNotFoundError):
        run_interactive_command(['nonexistentcommand'])
