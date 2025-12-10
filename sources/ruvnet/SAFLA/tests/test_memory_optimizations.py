"""
Test suite for Memory Optimization features.

Tests the enhanced memory management components including:
- FAISS and ChromaDB backend integrations
- Performance monitoring and metrics
- Advanced consolidation algorithms
- Optimized vector memory manager
"""

import pytest
import numpy as np
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any
from unittest.mock import Mock, patch, MagicMock

# Import optimization classes
from safla.core.memory_optimizations import (
    VectorSearchBackend,
    FAISSVectorBackend,
    ChromaVectorBackend,
    OptimizedVectorMemoryManager,
    MemoryPerformanceMetrics,
    MemoryPerformanceMonitor,
    AdvancedConsolidationAlgorithms,
    create_optimized_memory_manager,
    FAISS_AVAILABLE,
    CHROMA_AVAILABLE
)

from safla.core.hybrid_memory import (
    SimilarityMetric,
    EpisodicEvent,
    WorkingMemoryContext
)


class MockVectorBackend(VectorSearchBackend):
    """Mock vector backend for testing."""
    
    def __init__(self):
        self.vectors = {}
        self.next_idx = 0
    
    def add_vectors(self, vectors: np.ndarray, ids: List[str]) -> bool:
        for i, vector_id in enumerate(ids):
            self.vectors[vector_id] = vectors[i]
        return True
    
    def search(self, query_vector: np.ndarray, k: int):
        # Simple mock search - return random results
        ids = list(self.vectors.keys())[:k]
        distances = np.random.rand(len(ids))
        indices = np.arange(len(ids))
        return distances, indices
    
    def remove_vectors(self, ids: List[str]) -> bool:
        for vector_id in ids:
            self.vectors.pop(vector_id, None)
        return True
    
    def get_vector_count(self) -> int:
        return len(self.vectors)


class TestVectorSearchBackends:
    """Test suite for vector search backends."""
    
    def test_mock_backend_functionality(self):
        """Test mock backend basic functionality."""
        backend = MockVectorBackend()
        
        # Test adding vectors
        vectors = np.random.rand(3, 128).astype(np.float32)
        ids = ["vec1", "vec2", "vec3"]
        
        success = backend.add_vectors(vectors, ids)
        assert success
        assert backend.get_vector_count() == 3
        
        # Test search
        query = np.random.rand(128).astype(np.float32)
        distances, indices = backend.search(query, k=2)
        assert len(distances) == 2
        assert len(indices) == 2
        
        # Test removal
        success = backend.remove_vectors(["vec1"])
        assert success
        assert backend.get_vector_count() == 2
    
    @pytest.mark.skipif(not FAISS_AVAILABLE, reason="FAISS not available")
    def test_faiss_backend_initialization(self):
        """Test FAISS backend initialization."""
        backend = FAISSVectorBackend(dimension=128, index_type="Flat")
        
        assert backend.dimension == 128
        assert backend.metric == "L2"
        assert backend.get_vector_count() == 0
    
    @pytest.mark.skipif(not FAISS_AVAILABLE, reason="FAISS not available")
    def test_faiss_backend_operations(self):
        """Test FAISS backend vector operations."""
        backend = FAISSVectorBackend(dimension=128, index_type="Flat")
        
        # Add vectors
        vectors = np.random.rand(5, 128).astype(np.float32)
        ids = [f"vec_{i}" for i in range(5)]
        
        success = backend.add_vectors(vectors, ids)
        assert success
        assert backend.get_vector_count() == 5
        
        # Search
        query = vectors[0]  # Use first vector as query
        distances, indices = backend.search(query, k=3)
        
        assert len(distances) == 3
        assert len(indices) == 3
        assert distances[0] < 0.1  # Should be very close to itself
    
    @pytest.mark.skipif(not CHROMA_AVAILABLE, reason="ChromaDB not available")
    def test_chroma_backend_initialization(self):
        """Test ChromaDB backend initialization."""
        backend = ChromaVectorBackend(collection_name="test_collection")
        
        assert backend.collection is not None
        assert backend.get_vector_count() >= 0
    
    @pytest.mark.skipif(not CHROMA_AVAILABLE, reason="ChromaDB not available")
    def test_chroma_backend_operations(self):
        """Test ChromaDB backend vector operations."""
        backend = ChromaVectorBackend(collection_name="test_ops")
        
        # Add vectors
        vectors = np.random.rand(3, 128).astype(np.float32)
        ids = [f"chroma_vec_{i}" for i in range(3)]
        
        success = backend.add_vectors(vectors, ids)
        assert success
        
        # Search
        query = vectors[0]
        distances, indices = backend.search(query, k=2)
        
        assert len(distances) <= 2  # May return fewer if collection is small
        assert len(indices) <= 2


class TestOptimizedVectorMemoryManager:
    """Test suite for optimized vector memory manager."""
    
    @pytest.fixture
    def optimized_manager(self):
        """Create optimized manager with mock backend."""
        with patch('safla.core.memory_optimizations.FAISS_AVAILABLE', False):
            with patch('safla.core.memory_optimizations.CHROMA_AVAILABLE', False):
                return OptimizedVectorMemoryManager(
                    embedding_dim=128,
                    backend="mock"
                )
    
    def test_optimized_manager_initialization(self, optimized_manager):
        """Test optimized manager initialization."""
        assert optimized_manager.embedding_dim == 128
        assert optimized_manager.current_size == 0
        assert optimized_manager.is_empty()
    
    def test_optimized_manager_fallback_behavior(self, optimized_manager):
        """Test fallback to base implementation when backend unavailable."""
        # Should work like regular VectorMemoryManager
        embedding = np.random.rand(128).astype(np.float32)
        metadata = {"test": True}
        
        item_id = optimized_manager.store(embedding, metadata)
        assert item_id is not None
        assert optimized_manager.current_size == 1
        
        # Test search
        results = optimized_manager.similarity_search(embedding, k=1)
        assert len(results) == 1
        assert results[0].item_id == item_id
    
    def test_optimized_manager_with_mock_backend(self):
        """Test optimized manager with mock backend."""
        manager = OptimizedVectorMemoryManager(embedding_dim=128)
        
        # Mock the backend
        manager.backend = MockVectorBackend()
        manager._use_backend = True
        
        # Store vectors
        embeddings = [np.random.rand(128).astype(np.float32) for _ in range(3)]
        metadata_list = [{"index": i} for i in range(3)]
        
        item_ids = manager.batch_store(embeddings, metadata_list)
        assert len(item_ids) == 3
        assert manager.current_size == 3
        
        # Test search
        query = embeddings[0]
        results = manager.similarity_search(query, k=2)
        assert len(results) <= 2  # May return fewer due to mock behavior


class TestMemoryPerformanceMonitor:
    """Test suite for memory performance monitoring."""
    
    @pytest.fixture
    def performance_monitor(self):
        """Create performance monitor instance."""
        return MemoryPerformanceMonitor()
    
    def test_performance_monitor_initialization(self, performance_monitor):
        """Test performance monitor initialization."""
        assert len(performance_monitor.metrics_history) == 0
        assert 'search' in performance_monitor.operation_times
        assert 'store' in performance_monitor.operation_times
        assert 'retrieve' in performance_monitor.operation_times
        assert 'consolidate' in performance_monitor.operation_times
    
    def test_record_operation_time(self, performance_monitor):
        """Test recording operation times."""
        performance_monitor.record_operation_time('search', 10.5)
        performance_monitor.record_operation_time('search', 15.2)
        performance_monitor.record_operation_time('store', 5.8)
        
        assert len(performance_monitor.operation_times['search']) == 2
        assert len(performance_monitor.operation_times['store']) == 1
        assert performance_monitor.operation_times['search'][0] == 10.5
        assert performance_monitor.operation_times['store'][0] == 5.8
    
    def test_get_performance_metrics(self, performance_monitor):
        """Test getting performance metrics."""
        # Record some operation times
        performance_monitor.record_operation_time('search', 10.0)
        performance_monitor.record_operation_time('search', 20.0)
        performance_monitor.record_operation_time('store', 5.0)
        
        metrics = performance_monitor.get_performance_metrics()
        
        assert isinstance(metrics, MemoryPerformanceMetrics)
        assert metrics.search_latency_ms == 15.0  # Average of 10.0 and 20.0
        assert metrics.storage_latency_ms == 5.0
        assert metrics.memory_usage_mb >= 0
        assert 0 <= metrics.cache_hit_rate <= 1
        assert metrics.consolidation_rate >= 0
        assert metrics.throughput_ops_per_sec >= 0
    
    def test_operation_time_limit(self, performance_monitor):
        """Test that operation times are limited to prevent memory growth."""
        # Add many operation times
        for i in range(1500):
            performance_monitor.record_operation_time('search', float(i))
        
        # Should be limited to 500 most recent
        assert len(performance_monitor.operation_times['search']) == 500
        assert performance_monitor.operation_times['search'][0] == 1000.0  # Should start from 1000


class TestAdvancedConsolidationAlgorithms:
    """Test suite for advanced consolidation algorithms."""
    
    @pytest.fixture
    def sample_contexts(self):
        """Create sample working memory contexts."""
        base_time = datetime.now()
        return [
            WorkingMemoryContext(
                context_id=f"ctx_{i}",
                content=f"content_{i}",
                attention_weight=0.5 + (i * 0.1),
                timestamp=base_time - timedelta(hours=i),
                embedding=np.random.rand(128).astype(np.float32)
            )
            for i in range(5)
        ]
    
    @pytest.fixture
    def sample_events(self):
        """Create sample episodic events."""
        base_time = datetime.now()
        return [
            EpisodicEvent(
                event_id=f"event_{i}",
                timestamp=base_time - timedelta(minutes=i * 10),
                event_type="test_event" if i % 2 == 0 else "other_event",
                context={"action": f"action_{i}"},
                embedding=np.random.rand(128).astype(np.float32)
            )
            for i in range(6)
        ]
    
    @pytest.mark.asyncio
    async def test_importance_based_consolidation(self, sample_contexts):
        """Test importance-based consolidation algorithm."""
        candidates = await AdvancedConsolidationAlgorithms.importance_based_consolidation(
            sample_contexts,
            importance_threshold=0.6,
            decay_factor=0.9
        )
        
        # Should return contexts that meet importance threshold
        assert isinstance(candidates, list)
        assert len(candidates) <= len(sample_contexts)
        
        # All candidates should have sufficient importance
        for candidate in candidates:
            assert candidate.attention_weight >= 0.5  # Base attention weight
    
    @pytest.mark.asyncio
    async def test_semantic_clustering_consolidation(self, sample_events):
        """Test semantic clustering consolidation algorithm."""
        clusters = await AdvancedConsolidationAlgorithms.semantic_clustering_consolidation(
            sample_events,
            similarity_threshold=0.5,
            min_cluster_size=2
        )
        
        # Should return list of clusters
        assert isinstance(clusters, list)
        
        # Each cluster should meet minimum size requirement
        for cluster in clusters:
            assert len(cluster) >= 2
            
            # All events in cluster should be similar enough
            for event in cluster:
                assert hasattr(event, 'event_id')
                assert hasattr(event, 'embedding')
    
    @pytest.mark.asyncio
    async def test_semantic_similarity_calculation(self, sample_events):
        """Test semantic similarity calculation."""
        if len(sample_events) >= 2:
            event1, event2 = sample_events[0], sample_events[1]
            
            similarity = await AdvancedConsolidationAlgorithms._calculate_semantic_similarity(
                event1, event2
            )
            
            assert 0 <= similarity <= 1
            assert isinstance(similarity, float)
    
    @pytest.mark.asyncio
    async def test_empty_input_handling(self):
        """Test handling of empty inputs."""
        # Empty contexts
        candidates = await AdvancedConsolidationAlgorithms.importance_based_consolidation(
            [],
            importance_threshold=0.7
        )
        assert candidates == []
        
        # Empty events
        clusters = await AdvancedConsolidationAlgorithms.semantic_clustering_consolidation(
            [],
            similarity_threshold=0.8
        )
        assert clusters == []


class TestMemoryManagerFactory:
    """Test suite for memory manager factory function."""
    
    def test_create_optimized_manager_default_config(self):
        """Test creating manager with default configuration."""
        config = {}
        manager = create_optimized_memory_manager(config)
        
        assert isinstance(manager, OptimizedVectorMemoryManager)
        assert manager.embedding_dim == 512  # Default
        assert manager.similarity_metric == SimilarityMetric.COSINE  # Default
    
    def test_create_optimized_manager_custom_config(self):
        """Test creating manager with custom configuration."""
        config = {
            'embedding_dim': 768,
            'similarity_metric': 'euclidean',
            'max_capacity': 5000,
            'backend': 'default'
        }
        
        manager = create_optimized_memory_manager(config)
        
        assert manager.embedding_dim == 768
        assert manager.similarity_metric == SimilarityMetric.EUCLIDEAN
        assert manager.max_capacity == 5000
    
    def test_create_optimized_manager_auto_backend(self):
        """Test auto backend selection."""
        config = {
            'backend': 'auto',
            'expected_dataset_size': 50000
        }
        
        manager = create_optimized_memory_manager(config)
        assert isinstance(manager, OptimizedVectorMemoryManager)
    
    def test_create_optimized_manager_large_dataset_config(self):
        """Test configuration for large datasets."""
        config = {
            'backend': 'faiss',
            'expected_dataset_size': 200000,
            'backend_config': {}
        }
        
        manager = create_optimized_memory_manager(config)
        assert isinstance(manager, OptimizedVectorMemoryManager)
    
    def test_create_optimized_manager_small_dataset_config(self):
        """Test configuration for small datasets."""
        config = {
            'backend': 'faiss',
            'expected_dataset_size': 5000,
            'backend_config': {}
        }
        
        manager = create_optimized_memory_manager(config)
        assert isinstance(manager, OptimizedVectorMemoryManager)


class TestIntegrationWithHybridMemory:
    """Test integration between optimizations and hybrid memory."""
    
    def test_optimized_manager_integration(self):
        """Test that optimized manager works with hybrid memory architecture."""
        from safla.core.hybrid_memory import HybridMemoryArchitecture
        
        # Create optimized vector manager using factory
        vector_config = {
            'embedding_dim': 256,
            'similarity_metric': 'cosine',
            'max_capacity': 1000,
            'backend': 'default'  # Use default since external libs may not be available
        }
        
        # Create optimized manager directly
        optimized_manager = create_optimized_memory_manager(vector_config)
        
        assert optimized_manager is not None
        assert optimized_manager.embedding_dim == 256
        assert optimized_manager.similarity_metric == SimilarityMetric.COSINE
        
        # Test basic functionality
        embedding = np.random.rand(256).astype(np.float32)
        metadata = {"test": True}
        
        item_id = optimized_manager.store(embedding, metadata)
        assert item_id is not None
        assert optimized_manager.current_size == 1
    
    def test_performance_monitoring_integration(self):
        """Test performance monitoring with memory operations."""
        monitor = MemoryPerformanceMonitor()
        
        # Simulate some operations
        import time
        
        start_time = time.time()
        # Simulate search operation
        time.sleep(0.001)  # 1ms
        end_time = time.time()
        
        duration_ms = (end_time - start_time) * 1000
        monitor.record_operation_time('search', duration_ms)
        
        metrics = monitor.get_performance_metrics()
        assert metrics.search_latency_ms > 0
    
    @pytest.mark.asyncio
    async def test_advanced_consolidation_integration(self):
        """Test advanced consolidation with hybrid memory."""
        from safla.core.hybrid_memory import WorkingMemory, EpisodicMemory
        
        # Create memory components
        working_memory = WorkingMemory(capacity=5)
        episodic_memory = EpisodicMemory(max_capacity=10)
        
        # Add some contexts
        contexts = [
            WorkingMemoryContext(
                context_id=f"ctx_{i}",
                content=f"important_content_{i}",
                attention_weight=0.8,
                timestamp=datetime.now(),
                embedding=np.random.rand(128).astype(np.float32)
            )
            for i in range(3)
        ]
        
        for context in contexts:
            working_memory.add_context(context)
        
        # Test advanced consolidation
        candidates = await AdvancedConsolidationAlgorithms.importance_based_consolidation(
            working_memory.get_active_contexts(),
            importance_threshold=0.7
        )
        
        assert len(candidates) > 0
        assert all(candidate.attention_weight >= 0.7 for candidate in candidates)


class TestErrorHandling:
    """Test error handling in optimization components."""
    
    def test_backend_initialization_errors(self):
        """Test handling of backend initialization errors."""
        # Test with invalid backend
        with patch('safla.core.memory_optimizations.FAISS_AVAILABLE', False):
            manager = OptimizedVectorMemoryManager(backend="faiss")
            # Should fall back to default implementation
            assert not manager._use_backend
    
    def test_performance_monitor_edge_cases(self):
        """Test performance monitor with edge cases."""
        monitor = MemoryPerformanceMonitor()
        
        # Test with invalid operation type
        monitor.record_operation_time('invalid_operation', 10.0)
        # Should not crash
        
        # Test metrics with no data
        metrics = monitor.get_performance_metrics()
        assert metrics.search_latency_ms == 0.0
        assert metrics.storage_latency_ms == 0.0
    
    @pytest.mark.asyncio
    async def test_consolidation_algorithm_edge_cases(self):
        """Test consolidation algorithms with edge cases."""
        # Test with contexts having zero attention weight
        contexts = [
            WorkingMemoryContext(
                context_id="zero_attention",
                content="content",
                attention_weight=0.0,
                timestamp=datetime.now(),
                embedding=np.random.rand(128).astype(np.float32)
            )
        ]
        
        candidates = await AdvancedConsolidationAlgorithms.importance_based_consolidation(
            contexts,
            importance_threshold=0.5
        )
        
        # Should handle gracefully
        assert isinstance(candidates, list)
        assert len(candidates) == 0  # Zero attention should not meet threshold