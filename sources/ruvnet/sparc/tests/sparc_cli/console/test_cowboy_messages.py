from sparc_cli.console.cowboy_messages import get_cowboy_message, COWBOY_MESSAGES

import tempfile

def test_get_cowboy_message():
    """Test that get_cowboy_message returns a string from COWBOY_MESSAGES."""
    message = get_cowboy_message()
    assert isinstance(message, str)
    assert message in COWBOY_MESSAGES
