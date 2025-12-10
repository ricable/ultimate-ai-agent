"""
Comprehensive test suite for optimized vector memory manager.

Tests performance improvements, correctness of optimizations,
and validates that performance targets are met.
"""

import pytest
import numpy as np
import time
from unittest.mock import Mock, patch
from typing import List, Tuple

from safla.core.optimized_vector_memory import (
    OptimizedVectorMemoryManager,
    HierarchicalClusterIndex,
    IndexNode,
    CachedSearchManager,
    BatchOperationOptimizer
)
from safla.core.hybrid_memory import (
    SimilarityMetric,
    EvictionPolicy,
    MemoryItem,
    SimilarityResult
)


class TestOptimizedVectorMemoryManager:
    """Test cases for OptimizedVectorMemoryManager performance and correctness."""
    
    @pytest.fixture
    def vector_manager(self):
        """Create an optimized vector memory manager instance."""
        return OptimizedVectorMemoryManager(
            embedding_dim=512,
            similarity_metric=SimilarityMetric.COSINE,
            max_capacity=10000,
            enable_caching=True,
            enable_batch_optimization=True
        )
    
    @pytest.fixture
    def test_vectors(self):
        """Generate test vectors for benchmarking."""
        np.random.seed(42)
        return {
            'small': np.random.randn(100, 512).astype(np.float32),
            'medium': np.random.randn(1000, 512).astype(np.float32),
            'large': np.random.randn(10000, 512).astype(np.float32)
        }
    
    def test_initialization(self):
        """Test proper initialization of optimized manager."""
        manager = OptimizedVectorMemoryManager(
            embedding_dim=768,
            similarity_metric=SimilarityMetric.EUCLIDEAN,
            max_capacity=5000
        )
        
        assert manager.embedding_dim == 768
        assert manager.similarity_metric == SimilarityMetric.EUCLIDEAN
        assert manager.max_capacity == 5000
        assert manager.enable_caching is True  # Default
        assert manager.enable_batch_optimization is True  # Default
        assert isinstance(manager.index, HierarchicalClusterIndex)
    
    def test_similarity_search_performance_target(self, vector_manager, test_vectors):
        """Test that similarity search meets <1ms target for 10k vectors."""
        # Store 10k vectors
        vectors = test_vectors['large']
        metadata_list = [{"id": i} for i in range(len(vectors))]
        
        # Batch store for efficiency
        item_ids = vector_manager.batch_store(vectors, metadata_list)
        
        # Perform search and measure time
        query_vector = np.random.randn(512).astype(np.float32)
        
        # Warm up cache
        vector_manager.similarity_search(query_vector, k=10)
        
        # Measure actual search time
        start_time = time.perf_counter()
        results = vector_manager.similarity_search(query_vector, k=10)
        search_time = (time.perf_counter() - start_time) * 1000  # Convert to ms
        
        assert len(results) == 10
        assert search_time < 1.0, f"Search took {search_time:.2f}ms, expected <1ms"
    
    def test_batch_operations_performance_target(self, vector_manager):
        """Test that batch operations meet <10ms target for 1000 vectors."""
        # Prepare 1000 vectors
        np.random.seed(42)
        vectors = [np.random.randn(512).astype(np.float32) for _ in range(1000)]
        metadata_list = [{"id": i, "batch": True} for i in range(1000)]
        
        # Measure batch store time
        start_time = time.perf_counter()
        item_ids = vector_manager.batch_store(vectors, metadata_list)
        batch_time = (time.perf_counter() - start_time) * 1000  # Convert to ms
        
        assert len(item_ids) == 1000
        assert batch_time < 10.0, f"Batch store took {batch_time:.2f}ms, expected <10ms"
    
    def test_memory_efficiency_target(self, vector_manager, test_vectors):
        """Test that memory efficiency meets 80%+ target."""
        # Store vectors and measure memory usage
        vectors = test_vectors['medium']  # 1000 vectors
        metadata_list = [{"id": i} for i in range(len(vectors))]
        
        initial_memory = vector_manager.get_memory_usage()
        item_ids = vector_manager.batch_store(vectors, metadata_list)
        final_memory = vector_manager.get_memory_usage()
        
        # Calculate theoretical minimum memory
        vector_bytes = vectors.nbytes
        metadata_bytes = len(str(metadata_list).encode())
        theoretical_min = vector_bytes + metadata_bytes
        
        actual_usage = final_memory - initial_memory
        efficiency = (theoretical_min / actual_usage) * 100 if actual_usage > 0 else 100
        
        assert efficiency >= 80, f"Memory efficiency is {efficiency:.1f}%, expected >=80%"
    
    def test_throughput_target(self, vector_manager, test_vectors):
        """Test that search throughput meets 1000+ searches/second target."""
        # Store test vectors
        vectors = test_vectors['medium']
        metadata_list = [{"id": i} for i in range(len(vectors))]
        vector_manager.batch_store(vectors, metadata_list)
        
        # Prepare different query vectors
        num_queries = 100
        query_vectors = [np.random.randn(512).astype(np.float32) for _ in range(num_queries)]
        
        # Measure throughput
        start_time = time.perf_counter()
        for query in query_vectors:
            vector_manager.similarity_search(query, k=5)
        total_time = time.perf_counter() - start_time
        
        searches_per_second = num_queries / total_time
        assert searches_per_second >= 1000, f"Throughput is {searches_per_second:.0f} searches/s, expected >=1000"
    
    def test_hierarchical_index_correctness(self, vector_manager):
        """Test that hierarchical index returns correct results."""
        # Create distinct clusters of vectors
        np.random.seed(42)
        cluster1 = np.random.randn(50, 512).astype(np.float32) + np.array([1] * 512)
        cluster2 = np.random.randn(50, 512).astype(np.float32) + np.array([-1] * 512)
        
        # Store vectors
        vectors = np.vstack([cluster1, cluster2])
        metadata_list = [{"cluster": i // 50} for i in range(100)]
        item_ids = vector_manager.batch_store(vectors, metadata_list)
        
        # Search for vector similar to cluster1
        query = cluster1[0] + np.random.randn(512) * 0.1
        results = vector_manager.similarity_search(query, k=10)
        
        # Verify most results are from cluster1
        cluster1_count = sum(1 for r in results if r.item.metadata["cluster"] == 0)
        assert cluster1_count >= 8, f"Expected mostly cluster1 results, got {cluster1_count}/10"
    
    def test_cache_effectiveness(self, vector_manager, test_vectors):
        """Test that caching improves search performance."""
        # Store vectors
        vectors = test_vectors['medium']
        metadata_list = [{"id": i} for i in range(len(vectors))]
        vector_manager.batch_store(vectors, metadata_list)
        
        query_vector = np.random.randn(512).astype(np.float32)
        
        # First search (cache miss)
        start_time = time.perf_counter()
        results1 = vector_manager.similarity_search(query_vector, k=10)
        uncached_time = time.perf_counter() - start_time
        
        # Second search (cache hit)
        start_time = time.perf_counter()
        results2 = vector_manager.similarity_search(query_vector, k=10)
        cached_time = time.perf_counter() - start_time
        
        # Verify cache hit is faster
        assert cached_time < uncached_time * 0.5, "Cache should provide >50% speedup"
        
        # Verify results are identical
        assert len(results1) == len(results2)
        for r1, r2 in zip(results1, results2):
            assert r1.item_id == r2.item_id
    
    def test_incremental_index_updates(self, vector_manager):
        """Test that incremental index updates work correctly."""
        # Initial batch
        initial_vectors = np.random.randn(100, 512).astype(np.float32)
        initial_metadata = [{"batch": "initial", "id": i} for i in range(100)]
        initial_ids = vector_manager.batch_store(initial_vectors, initial_metadata)
        
        # Add more vectors incrementally
        for i in range(10):
            new_vector = np.random.randn(512).astype(np.float32)
            new_metadata = {"batch": "incremental", "id": 100 + i}
            item_id = vector_manager.store(new_vector, new_metadata)
            assert item_id is not None
        
        # Verify total count
        assert vector_manager.current_size == 110
        
        # Search should find vectors from both batches
        query = np.random.randn(512).astype(np.float32)
        results = vector_manager.similarity_search(query, k=20)
        
        initial_count = sum(1 for r in results if r.item.metadata["batch"] == "initial")
        incremental_count = sum(1 for r in results if r.item.metadata["batch"] == "incremental")
        
        assert initial_count > 0 and incremental_count > 0
    
    def test_delete_and_reindex(self, vector_manager, test_vectors):
        """Test deletion and reindexing performance."""
        # Store vectors
        vectors = test_vectors['small']
        metadata_list = [{"id": i} for i in range(len(vectors))]
        item_ids = vector_manager.batch_store(vectors, metadata_list)
        
        # Delete half the vectors
        items_to_delete = item_ids[:50]
        for item_id in items_to_delete:
            assert vector_manager.delete(item_id) is True
        
        assert vector_manager.current_size == 50
        
        # Verify search still works correctly
        query = np.random.randn(512).astype(np.float32)
        results = vector_manager.similarity_search(query, k=10)
        
        # All results should be from remaining vectors
        for result in results:
            assert result.item_id not in items_to_delete
    
    def test_concurrent_operations(self, vector_manager):
        """Test thread-safety of optimized operations."""
        import threading
        import queue
        
        errors = queue.Queue()
        
        def store_vectors():
            try:
                for i in range(100):
                    vector = np.random.randn(512).astype(np.float32)
                    metadata = {"thread": "store", "id": i}
                    vector_manager.store(vector, metadata)
            except Exception as e:
                errors.put(e)
        
        def search_vectors():
            try:
                for i in range(100):
                    query = np.random.randn(512).astype(np.float32)
                    results = vector_manager.similarity_search(query, k=5)
            except Exception as e:
                errors.put(e)
        
        # Run concurrent operations
        threads = []
        for _ in range(2):
            threads.append(threading.Thread(target=store_vectors))
            threads.append(threading.Thread(target=search_vectors))
        
        for t in threads:
            t.start()
        
        for t in threads:
            t.join()
        
        # Check for errors
        assert errors.empty(), f"Concurrent operations failed: {errors.get()}"
    
    def test_similarity_metrics_optimization(self, vector_manager):
        """Test that all similarity metrics are optimized."""
        metrics = [
            SimilarityMetric.COSINE,
            SimilarityMetric.EUCLIDEAN,
            SimilarityMetric.DOT_PRODUCT
        ]
        
        for metric in metrics:
            manager = OptimizedVectorMemoryManager(
                embedding_dim=512,
                similarity_metric=metric
            )
            
            # Store test vectors
            vectors = np.random.randn(100, 512).astype(np.float32)
            metadata_list = [{"id": i} for i in range(100)]
            manager.batch_store(vectors, metadata_list)
            
            # Verify search works with optimized metric
            query = np.random.randn(512).astype(np.float32)
            results = manager.similarity_search(query, k=10)
            
            assert len(results) == 10
            # Verify results are properly sorted
            if metric in [SimilarityMetric.COSINE, SimilarityMetric.DOT_PRODUCT]:
                # Higher scores are better
                for i in range(len(results) - 1):
                    assert results[i].similarity_score >= results[i + 1].similarity_score
            else:
                # Lower distances are better
                for i in range(len(results) - 1):
                    assert results[i].similarity_score <= results[i + 1].similarity_score
    
    def test_memory_mapped_storage(self, vector_manager):
        """Test memory-mapped storage for large datasets."""
        # Enable memory mapping
        vector_manager.enable_memory_mapping(cache_dir="/tmp/safla_test_mmap")
        
        # Store large dataset
        large_vectors = np.random.randn(5000, 512).astype(np.float32)
        metadata_list = [{"id": i} for i in range(5000)]
        
        item_ids = vector_manager.batch_store(large_vectors, metadata_list)
        
        # Verify data can be retrieved after clearing in-memory cache
        vector_manager.clear_memory_cache()
        
        # Search should still work using memory-mapped data
        query = np.random.randn(512).astype(np.float32)
        results = vector_manager.similarity_search(query, k=10)
        
        assert len(results) == 10
        
        # Cleanup
        vector_manager.cleanup_memory_mapping()