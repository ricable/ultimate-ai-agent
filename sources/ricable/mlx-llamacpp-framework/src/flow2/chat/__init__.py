"""
Chat Interface Module
====================

Interactive chat interfaces for both CLI and Web, supporting MLX and LlamaCpp frameworks.

Modules:
    interfaces: CLI and Web chat implementations
    common: Shared chat utilities and history management
"""

from .common.chat_history import create_chat_session

__all__ = [
    "create_chat_session",
]
EOF < /dev/null