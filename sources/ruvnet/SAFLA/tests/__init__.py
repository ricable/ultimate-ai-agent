"""
SAFLA Test Suite

This package contains comprehensive tests for the SAFLA system,
including unit tests, integration tests, and installation validation.
"""

# Test configuration
import pytest

# Configure pytest for async tests
pytest_plugins = ["pytest_asyncio"]

# Test markers
pytestmark = [
    pytest.mark.asyncio,
]

__all__ = []