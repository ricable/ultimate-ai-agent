"""
Unit tests for SAFLA Manager.
"""

import pytest
import asyncio
from unittest.mock import patch, AsyncMock, MagicMock
from datetime import datetime

from src.core.safla_manager import SAFLAManager, SAFLAError, SystemStatus
from src.config.settings import AppSettings


@pytest.mark.unit
class TestSAFLAManager:
    """Test suite for SAFLA Manager."""
    
    def test_initialization(self, test_settings):
        """Test SAFLA manager initialization."""
        manager = SAFLAManager(config=test_settings)
        
        assert manager.config == test_settings
        assert manager.test_mode is False
        assert manager.is_initialized is False
        assert manager._system_status.status == "uninitialized"
    
    def test_initialization_test_mode(self):
        """Test SAFLA manager initialization in test mode."""
        manager = SAFLAManager(test_mode=True)
        
        assert manager.test_mode is True
        assert manager.is_initialized is False
    
    def test_property_lazy_loading(self, test_settings):
        """Test lazy loading of components."""
        manager = SAFLAManager(config=test_settings)
        
        # Components should not be initialized yet
        assert manager._hybrid_memory is None
        assert manager._safety_validator is None
        assert manager._meta_cognitive is None
        assert manager._delta_evaluator is None
        
        # Access properties should create instances
        memory = manager.hybrid_memory
        assert memory is not None
        assert manager._hybrid_memory is not None
        
        safety = manager.safety_validator
        assert safety is not None
        assert manager._safety_validator is not None
    
    @pytest.mark.asyncio
    async def test_initialize_system_success(self, test_settings):
        """Test successful system initialization."""
        manager = SAFLAManager(config=test_settings)
        
        # Mock the hybrid memory initialize method
        with patch.object(manager.hybrid_memory, 'initialize', new_callable=AsyncMock) as mock_init:
            await manager.initialize_system()
            
            assert manager.is_initialized is True
            assert manager._system_status.status == "healthy"
            assert manager._system_status.memory_initialized is True
            assert manager._system_status.safety_active is True
            assert manager._system_status.metacognitive_running is True
            assert manager._system_status.performance_normal is True
            mock_init.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_initialize_system_failure(self, test_settings):
        """Test system initialization failure."""
        manager = SAFLAManager(config=test_settings)
        
        # Mock initialization to fail
        with patch.object(
            manager.hybrid_memory,
            'initialize',
            side_effect=Exception("Memory init failed")
        ):
            with pytest.raises(SAFLAError, match="System initialization failed"):
                await manager.initialize_system()
            
            assert manager.is_initialized is False
            assert manager._system_status.status == "error"
            assert "error" in manager._system_status.details
    
    @pytest.mark.asyncio
    async def test_get_system_status(self, test_settings):
        """Test getting system status."""
        manager = SAFLAManager(config=test_settings)
        
        # Before initialization
        status = await manager.get_system_status()
        assert status["status"] == "uninitialized"
        assert status["memory_initialized"] is False
        assert status["safety_active"] is False
        assert "last_updated" in status
        
        # After initialization
        with patch.object(manager.hybrid_memory, 'initialize', new_callable=AsyncMock):
            await manager.initialize_system()
            
        status = await manager.get_system_status()
        assert status["status"] == "healthy"
        assert status["memory_initialized"] is True
        assert status["safety_active"] is True
    
    @pytest.mark.asyncio
    async def test_search_vector_memory_valid(self, test_settings):
        """Test vector memory search with valid parameters."""
        manager = SAFLAManager(config=test_settings)
        
        # Initialize system
        with patch.object(manager.hybrid_memory, 'initialize', new_callable=AsyncMock):
            await manager.initialize_system()
        
        # Mock search results
        expected_results = [
            {"content": "result 1", "similarity": 0.9},
            {"content": "result 2", "similarity": 0.8}
        ]
        
        with patch.object(
            manager.hybrid_memory,
            'vector_search',
            return_value=expected_results
        ) as mock_search:
            results = await manager.search_vector_memory("test query", threshold=0.7)
            
            assert results == expected_results
            mock_search.assert_called_once_with(
                query="test query",
                similarity_threshold=0.7,
                max_results=10
            )
    
    @pytest.mark.asyncio
    async def test_search_vector_memory_invalid_threshold(self, test_settings):
        """Test vector search with invalid threshold."""
        manager = SAFLAManager(config=test_settings)
        
        with patch.object(manager.hybrid_memory, 'initialize', new_callable=AsyncMock):
            await manager.initialize_system()
        
        # Test threshold too low
        with pytest.raises(ValueError, match="Threshold must be between"):
            await manager.search_vector_memory("query", threshold=-0.1)
        
        # Test threshold too high
        with pytest.raises(ValueError, match="Threshold must be between"):
            await manager.search_vector_memory("query", threshold=1.5)
    
    @pytest.mark.asyncio
    async def test_search_vector_memory_not_initialized(self, test_settings):
        """Test vector search when system not initialized."""
        manager = SAFLAManager(config=test_settings)
        
        with pytest.raises(SAFLAError, match="System not initialized"):
            await manager.search_vector_memory("query")
    
    @pytest.mark.asyncio
    async def test_validate_safety_success(self, test_settings):
        """Test successful safety validation."""
        manager = SAFLAManager(config=test_settings)
        
        with patch.object(manager.hybrid_memory, 'initialize', new_callable=AsyncMock):
            await manager.initialize_system()
        
        expected_result = {
            "safety_score": 0.95,
            "risk_level": "low",
            "violations": []
        }
        
        with patch.object(
            manager.safety_validator,
            'validate',
            return_value=expected_result
        ) as mock_validate:
            result = await manager.validate_safety("test content")
            
            assert result == expected_result
            mock_validate.assert_called_once_with("test content")
    
    @pytest.mark.asyncio
    async def test_validate_safety_not_initialized(self, test_settings):
        """Test safety validation when system not initialized."""
        manager = SAFLAManager(config=test_settings)
        
        with pytest.raises(SAFLAError, match="System not initialized"):
            await manager.validate_safety("test content")
    
    @pytest.mark.asyncio
    async def test_get_performance_metrics(self, test_settings):
        """Test getting performance metrics."""
        manager = SAFLAManager(config=test_settings)
        
        with patch.object(manager.hybrid_memory, 'initialize', new_callable=AsyncMock):
            await manager.initialize_system()
        
        expected_metrics = {
            "performance_score": 0.92,
            "improvement_rate": 0.15,
            "stability": 0.98
        }
        
        with patch.object(
            manager.delta_evaluator,
            'get_current_metrics',
            return_value=expected_metrics
        ):
            metrics = await manager.get_performance_metrics()
            
            assert "performance_score" in metrics
            assert "memory_usage_mb" in metrics  # Added by manager
            assert "cpu_percent" in metrics
            assert "thread_count" in metrics
    
    @pytest.mark.asyncio
    async def test_get_meta_cognitive_state(self, test_settings):
        """Test getting meta-cognitive state."""
        manager = SAFLAManager(config=test_settings)
        
        with patch.object(manager.hybrid_memory, 'initialize', new_callable=AsyncMock):
            await manager.initialize_system()
        
        expected_state = {
            "self_awareness_level": 0.8,
            "current_goals": ["optimize", "learn"],
            "strategy": "adaptive"
        }
        
        with patch.object(
            manager.meta_cognitive,
            'get_awareness_state',
            return_value=expected_state
        ):
            state = await manager.get_meta_cognitive_state()
            
            assert state == expected_state
    
    @pytest.mark.asyncio
    async def test_store_memory_safe_content(self, test_settings):
        """Test storing safe content in memory."""
        manager = SAFLAManager(config=test_settings)
        
        with patch.object(manager.hybrid_memory, 'initialize', new_callable=AsyncMock):
            await manager.initialize_system()
        
        # Mock safety validation to pass
        with patch.object(
            manager.safety_validator,
            'validate',
            return_value={"safety_score": 0.9, "risk_level": "low"}
        ):
            result = await manager.store_memory("safe content", "vector")
            
            assert result["success"] is True
            assert result["memory_type"] == "vector"
            assert result["content_length"] == len("safe content")
            assert result["safety_score"] == 0.9
    
    @pytest.mark.asyncio
    async def test_store_memory_unsafe_content(self, test_settings):
        """Test storing unsafe content is rejected."""
        manager = SAFLAManager(config=test_settings)
        
        with patch.object(manager.hybrid_memory, 'initialize', new_callable=AsyncMock):
            await manager.initialize_system()
        
        # Mock safety validation to fail
        with patch.object(
            manager.safety_validator,
            'validate',
            return_value={"safety_score": 0.3, "risk_level": "high"}
        ):
            with pytest.raises(SAFLAError, match="Content failed safety validation"):
                await manager.store_memory("unsafe content", "vector")
    
    @pytest.mark.asyncio
    async def test_cleanup(self, test_settings):
        """Test system cleanup."""
        manager = SAFLAManager(config=test_settings)
        
        with patch.object(manager.hybrid_memory, 'initialize', new_callable=AsyncMock):
            await manager.initialize_system()
        
        assert manager.is_initialized is True
        
        await manager.cleanup()
        
        assert manager.is_initialized is False
        assert manager._system_status.status == "shutdown"
    
    def test_repr(self, test_settings):
        """Test string representation."""
        manager = SAFLAManager(config=test_settings, test_mode=True)
        
        repr_str = repr(manager)
        assert "SAFLAManager" in repr_str
        assert "initialized=False" in repr_str
        assert "status=uninitialized" in repr_str
        assert "mode=test" in repr_str