# File: backend/websocket/__init__.py
"""
WebSocket optimization package for UAP platform.
Provides optimized WebSocket handling with connection pooling, message batching, and compression.
"""

from .optimized_handler import (
    OptimizedWebSocketHandler, ConnectionPool, MessageCompressor,
    WebSocketConfig, ConnectionMetrics, ConnectionState,
    optimized_websocket_handler
)

__all__ = [
    'OptimizedWebSocketHandler',
    'ConnectionPool',
    'MessageCompressor',
    'WebSocketConfig',
    'ConnectionMetrics',
    'ConnectionState',
    'optimized_websocket_handler'
]
