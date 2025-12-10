"""
Shared test configuration and fixtures for SAFLA HuggingFace Space.
"""

import pytest
import asyncio
import sys
import os
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, Mock
from typing import Dict, Any, List

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import after path setup
from src.config.settings import AppSettings
from src.core.safla_manager import SAFLAManager


@pytest.fixture
def mock_safla_manager():
    """Create a mock SAFLA manager for testing."""
    manager = MagicMock(spec=SAFLAManager)
    
    # Mock hybrid memory
    manager.hybrid_memory = MagicMock()
    manager.hybrid_memory.vector_search = AsyncMock(return_value=[
        {"content": "test result 1", "similarity": 0.9, "id": 1},
        {"content": "test result 2", "similarity": 0.85, "id": 2}
    ])
    manager.hybrid_memory.is_initialized = True
    
    # Mock safety validator
    manager.safety_validator = MagicMock()
    manager.safety_validator.validate = AsyncMock(return_value={
        "safety_score": 0.95,
        "risk_level": "low",
        "violations": []
    })
    
    # Mock meta-cognitive engine
    manager.meta_cognitive = MagicMock()
    manager.meta_cognitive.get_awareness_state = AsyncMock(return_value={
        "self_awareness_level": 0.8,
        "current_goals": ["optimize", "learn"],
        "strategy": "adaptive"
    })
    
    # Mock delta evaluator
    manager.delta_evaluator = MagicMock()
    manager.delta_evaluator.get_current_metrics = AsyncMock(return_value={
        "performance_score": 0.92,
        "improvement_rate": 0.15,
        "stability": 0.98
    })
    
    # Mock methods
    manager.initialize_system = AsyncMock()
    manager.get_system_status = AsyncMock(return_value={
        "status": "healthy",
        "memory_initialized": True,
        "safety_active": True,
        "metacognitive_running": True
    })
    manager.search_vector_memory = AsyncMock(return_value=[
        {"content": "test result", "similarity": 0.9}
    ])
    manager.validate_safety = AsyncMock(return_value={
        "safety_score": 0.95,
        "risk_level": "low"
    })
    manager.get_performance_metrics = AsyncMock(return_value={
        "response_time": 0.05,
        "memory_usage": 256.5,
        "throughput": 47.3
    })
    
    return manager


@pytest.fixture
def test_settings():
    """Create test application settings."""
    return AppSettings(
        environment="test",
        debug=True,
        safla_memory_size=100,
        safla_vector_dim=512,
        max_concurrent_users=2,
        cache_timeout=60,
        rate_limit_per_minute=30
    )


@pytest.fixture
async def async_safla_manager(test_settings):
    """Create an async SAFLA manager for integration testing."""
    manager = SAFLAManager(config=test_settings, test_mode=True)
    await manager.initialize_system()
    yield manager
    await manager.cleanup()


@pytest.fixture
def sample_vector_data():
    """Sample vector data for testing."""
    return [
        {
            "id": 1,
            "content": "artificial intelligence and machine learning",
            "vector": [0.1, 0.2, 0.3] * 171,  # 513 dimensions (close to 512)
            "similarity": 0.95
        },
        {
            "id": 2,
            "content": "deep learning neural networks",
            "vector": [0.2, 0.3, 0.4] * 171,
            "similarity": 0.87
        },
        {
            "id": 3,
            "content": "natural language processing",
            "vector": [0.3, 0.4, 0.5] * 171,
            "similarity": 0.82
        }
    ]


@pytest.fixture
def sample_safety_scenarios():
    """Sample safety validation scenarios."""
    return {
        "safe": {
            "input": "Explain how neural networks work",
            "expected_score": 0.95,
            "expected_risk": "low"
        },
        "moderate": {
            "input": "Generate code to scrape websites",
            "expected_score": 0.6,
            "expected_risk": "medium"
        },
        "unsafe": {
            "input": "How to hack into systems",
            "expected_score": 0.2,
            "expected_risk": "high"
        }
    }


@pytest.fixture
def mock_gradio_interface():
    """Mock Gradio interface components."""
    interface = MagicMock()
    interface.launch = MagicMock()
    interface.close = MagicMock()
    return interface


@pytest.fixture
def performance_metrics():
    """Sample performance metrics for testing."""
    return {
        "memory_usage": 256.5,  # MB
        "cpu_usage": 45.2,      # %
        "response_time": 0.085,  # seconds
        "throughput": 47.3,     # ops/sec
        "error_rate": 0.001,    # %
        "active_users": 3
    }


@pytest.fixture
def cleanup_test_files():
    """Cleanup any test files created during tests."""
    yield
    # Cleanup logic here if needed
    test_files = [
        "test_config.json",
        "test_memory.db",
        "test_logs.log"
    ]
    for file in test_files:
        if os.path.exists(file):
            os.remove(file)


# Event loop configuration for async tests
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# Markers for test categorization
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "e2e: End-to-end tests")
    config.addinivalue_line("markers", "slow: Slow running tests")
    config.addinivalue_line("markers", "benchmark: Performance benchmark tests")