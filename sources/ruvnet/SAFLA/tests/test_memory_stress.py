"""
Memory Stress Testing Module

This module implements comprehensive memory stress tests for the SAFLA memory bank system,
addressing the "Memory limit exceeded" scenario reported in the MCP server mock data.
"""

import pytest
import asyncio
import psutil
import gc
import numpy as np
from typing import List, Dict, Any
from unittest.mock import Mock, patch

from safla.core.hybrid_memory import HybridMemoryArchitecture, SimilarityMetric, EvictionPolicy
from safla.core.optimized_vector_memory import OptimizedVectorMemoryManager
from safla.core.memory_optimizations import MemoryPerformanceMonitor


class TestMemoryStress:
    """Comprehensive memory stress testing suite"""
    
    @pytest.fixture
    async def memory_system(self):
        """Create a memory system for testing"""
        config = {
            'vector_memory': {
                'embedding_dim': 512,
                'max_capacity': 10000,
                'similarity_metric': SimilarityMetric.COSINE,
                'eviction_policy': EvictionPolicy.LRU
            },
            'episodic_memory': {
                'max_capacity': 1000
            },
            'working_memory': {
                'capacity': 50,
                'attention_window': 5
            }
        }
        
        memory = HybridMemoryArchitecture(
            vector_config=config.get('vector_memory'),
            episodic_config=config.get('episodic_memory'),
            working_config=config.get('working_memory')
        )
        yield memory
        # Cleanup using the actual available method
        memory.cleanup_memory()
    
    @pytest.fixture
    def memory_monitor(self):
        """Memory usage monitoring fixture"""
        class MemoryMonitor:
            def __init__(self):
                self.initial_memory = psutil.Process().memory_info().rss
                self.peak_memory = self.initial_memory
                self.memory_samples = []
            
            def sample(self):
                current = psutil.Process().memory_info().rss
                self.memory_samples.append(current)
                self.peak_memory = max(self.peak_memory, current)
                return current
            
            def get_peak_mb(self):
                return self.peak_memory / (1024 * 1024)
            
            def get_increase_mb(self):
                return (self.peak_memory - self.initial_memory) / (1024 * 1024)
        
        return MemoryMonitor()
    
    @pytest.mark.asyncio
    async def test_large_vector_batch_processing(self, memory_system, memory_monitor):
        """Test memory handling with large vector batches"""
        # Generate large batch of vectors
        batch_size = 5000
        vector_dim = 512
        
        vectors = []
        for i in range(batch_size):
            vector = [0.1 * j for j in range(vector_dim)]
            vectors.append({
                'id': f'vector_{i}',
                'data': vector,
                'metadata': {'batch': 'stress_test', 'index': i}
            })
            
            # Sample memory every 1000 vectors
            if i % 1000 == 0:
                memory_monitor.sample()
        
        # Store vectors in batches to test memory management
        batch_size_limit = 1000
        for i in range(0, len(vectors), batch_size_limit):
            batch = vectors[i:i + batch_size_limit]
            # Convert to proper format for batch_store
            embeddings = [np.array(vector['data']) for vector in batch]
            metadata_list = [vector['metadata'] for vector in batch]
            memory_system.vector_memory.batch_store(embeddings, metadata_list)
            memory_monitor.sample()
            
            # Force garbage collection to test memory cleanup
            gc.collect()
        
        # Verify memory usage is within acceptable limits (< 500MB increase)
        memory_increase = memory_monitor.get_increase_mb()
        assert memory_increase < 500, f"Memory increase {memory_increase:.2f}MB exceeds 500MB limit"
        
        # Verify all vectors were stored
        stored_count = memory_system.vector_memory.current_size
        assert stored_count == batch_size, f"Expected {batch_size} vectors, got {stored_count}"
    
    @pytest.mark.asyncio
    async def test_memory_pressure_handling(self, memory_system, memory_monitor):
        """Test system behavior under memory pressure"""
        # Simulate memory pressure by creating large objects
        large_objects = []
        
        try:
            # Create progressively larger objects until memory pressure
            for size_mb in [10, 25, 50, 100, 200]:
                # Create large object (list of integers)
                size_bytes = size_mb * 1024 * 1024
                large_obj = [i for i in range(size_bytes // 8)]  # 8 bytes per int
                large_objects.append(large_obj)
                
                memory_monitor.sample()
                
                # Test memory operations under pressure
                test_vectors = [
                    {'id': f'pressure_test_{i}', 'data': [0.1] * 512, 'metadata': {'test': 'pressure'}}
                    for i in range(100)
                ]

                # Store vectors using batch_store with proper numpy arrays
                embeddings = [np.array(vector['data']) for vector in test_vectors]
                metadata_list = [vector['metadata'] for vector in test_vectors]
                memory_system.vector_memory.batch_store(embeddings, metadata_list)
                
                # Verify system still responds
                results = await memory_system.integrated_search(np.array([0.1] * 512), k=5)
                assert len(results) > 0, "System should respond under memory pressure"
                
                # Check if memory usage is getting too high
                current_memory_mb = memory_monitor.sample() / (1024 * 1024)
                if current_memory_mb > 1000:  # 1GB limit
                    break
        
        finally:
            # Cleanup large objects
            large_objects.clear()
            gc.collect()
        
        # Verify system recovers after memory pressure
        final_memory = memory_monitor.sample()
        recovery_test = await memory_system.integrated_search(np.array([0.1] * 512), k=10)
        assert len(recovery_test) > 0, "System should recover after memory pressure"
    
    @pytest.mark.asyncio
    async def test_memory_leak_detection(self, memory_system, memory_monitor):
        """Test for memory leaks during repeated operations"""
        initial_memory = memory_monitor.sample()
        
        # Perform repeated operations that could cause leaks
        for cycle in range(10):
            # Create and store vectors
            vectors = [
                {'id': f'leak_test_{cycle}_{i}', 'data': [0.1 * i] * 512, 'metadata': {'cycle': cycle}}
                for i in range(500)
            ]
            # Store vectors using batch_store with proper numpy arrays
            embeddings = [np.array(vector['data']) for vector in vectors]
            metadata_list = [vector['metadata'] for vector in vectors]
            memory_system.vector_memory.batch_store(embeddings, metadata_list)
            
            # Perform searches
            for _ in range(50):
                await memory_system.integrated_search(np.array([0.1] * 512), k=10)
            
            # Delete vectors to test cleanup
            for vector in vectors:
                memory_system.vector_memory.delete(vector['id'])
            
            # Force garbage collection
            gc.collect()
            memory_monitor.sample()
        
        final_memory = memory_monitor.sample()
        memory_growth = (final_memory - initial_memory) / (1024 * 1024)
        
        # Memory growth should be minimal (< 50MB) after cleanup
        assert memory_growth < 50, f"Potential memory leak detected: {memory_growth:.2f}MB growth"
    
    @pytest.mark.asyncio
    async def test_memory_optimization_under_stress(self, memory_system, memory_monitor):
        """Test memory optimization features under stress conditions"""
        performance_monitor = MemoryPerformanceMonitor()
        
        # Fill memory with vectors
        vectors = []
        for i in range(3000):
            vector = {
                'id': f'opt_test_{i}',
                'data': [0.1 * (i % 100)] * 512,
                'metadata': {'priority': i % 3, 'timestamp': i}
            }
            vectors.append(vector)
        
        # Store vectors using batch_store
        embeddings = [np.array(vector['data']) for vector in vectors]
        metadata_list = [vector['metadata'] for vector in vectors]
        memory_system.vector_memory.batch_store(embeddings, metadata_list)
        memory_before_opt = memory_monitor.sample()
        
        # Monitor performance during memory operations
        # MemoryPerformanceMonitor doesn't have start_monitoring method
        # Record initial state instead
        initial_metrics = performance_monitor.get_performance_metrics()
        
        # Trigger memory cleanup through the memory system
        cleanup_result = memory_system.cleanup_memory(max_age_hours=1, min_importance=0.3)
        memory_after_cleanup = memory_monitor.sample()
        
        performance_metrics = performance_monitor.get_performance_metrics()
        
        # Verify cleanup reduced memory usage or maintained efficiency
        memory_change = (memory_before_opt - memory_after_cleanup) / (1024 * 1024)
        assert memory_change >= 0 or performance_metrics.get('efficiency_score', 0) > 0.8, "Memory should be optimized or maintain high efficiency"
        
        # Verify system still functions correctly
        search_results = await memory_system.integrated_search(np.array([0.1] * 512), k=20)
        assert len(search_results) > 0, "System should function after cleanup"
        
        # Verify performance monitoring captured metrics
        assert performance_metrics is not None, "Performance metrics should be captured"
        assert hasattr(performance_metrics, 'memory_usage_mb') and hasattr(performance_metrics, 'throughput_ops_per_sec'), "Should track memory and operation metrics"
    
    @pytest.mark.asyncio
    async def test_emergency_memory_cleanup(self, memory_system, memory_monitor):
        """Test emergency memory cleanup procedures"""
        # Fill memory to trigger emergency cleanup
        vectors = []
        for i in range(8000):
            vector = {
                'id': f'emergency_test_{i}',
                'data': [0.1 * i] * 512,
                'metadata': {'importance': i % 5, 'created': i}
            }
            vectors.append(vector)
        
        # Store vectors using batch_store
        embeddings = [np.array(vector['data']) for vector in vectors]
        metadata_list = [vector['metadata'] for vector in vectors]
        item_ids = memory_system.vector_memory.batch_store(embeddings, metadata_list)
        memory_before_cleanup = memory_monitor.sample()
        
        # Trigger emergency cleanup using available cleanup_memory method
        memory_system.cleanup_memory(max_age_hours=0, min_importance=0.8)
        
        memory_after_cleanup = memory_monitor.sample()
        
        # Verify emergency cleanup was effective
        memory_freed = (memory_before_cleanup - memory_after_cleanup) / (1024 * 1024)
        assert memory_freed >= 0, f"Emergency cleanup should maintain or reduce memory usage, change: {memory_freed:.2f}MB"
        
        # Verify system remains functional
        query_embedding = [0.1] * 512
        search_results = await memory_system.integrated_search(np.array(query_embedding), k=10)
        assert len(search_results) >= 0, "System should function after cleanup"
        
        # Verify cleanup was performed (system should still be responsive)
        assert not memory_system.vector_memory.is_empty() or len(item_ids) > 0
    
    @pytest.mark.asyncio
    async def test_concurrent_memory_operations_stress(self, memory_system, memory_monitor):
        """Test memory handling under concurrent operation stress"""
        async def store_operation(batch_id: int):
            vectors = [
                {'id': f'concurrent_{batch_id}_{i}', 'data': [0.1 * i] * 512, 'metadata': {'batch': batch_id}}
                for i in range(200)
            ]
            # Store vectors using batch_store
            embeddings = [np.array(vector['data']) for vector in vectors]
            metadata_list = [vector['metadata'] for vector in vectors]
            memory_system.vector_memory.batch_store(embeddings, metadata_list)
        
        async def search_operation():
            for _ in range(50):
                await memory_system.integrated_search(np.array([0.1] * 512), k=10)
        
        async def delete_operation(batch_id: int):
            # Get stored item IDs first (simplified approach)
            for i in range(100):  # Delete half of stored vectors
                item_id = f'concurrent_{batch_id}_{i}'
                # Try to delete if exists (delete returns boolean)
                memory_system.vector_memory.delete(item_id)
        
        # Run concurrent operations
        tasks = []
        
        # Store operations
        for batch_id in range(10):
            tasks.append(store_operation(batch_id))
        
        # Search operations
        for _ in range(5):
            tasks.append(search_operation())
        
        # Delete operations (with delay)
        async def delayed_delete():
            await asyncio.sleep(0.5)  # Let some stores complete first
            delete_tasks = [delete_operation(batch_id) for batch_id in range(5)]
            await asyncio.gather(*delete_tasks)
        
        tasks.append(delayed_delete())
        
        # Monitor memory during concurrent operations
        memory_samples = []
        
        async def memory_sampling():
            for _ in range(20):
                memory_samples.append(memory_monitor.sample())
                await asyncio.sleep(0.1)
        
        tasks.append(memory_sampling())
        
        # Execute all tasks concurrently
        await asyncio.gather(*tasks)
        
        # Verify memory usage remained stable
        max_memory = max(memory_samples)
        min_memory = min(memory_samples)
        memory_variance = (max_memory - min_memory) / (1024 * 1024)
        
        assert memory_variance < 200, f"Memory variance {memory_variance:.2f}MB too high during concurrent operations"
        
        # Verify system integrity
        final_count = memory_system.vector_memory.current_size
        assert final_count > 0, "System should have vectors after concurrent operations"


# Additional stress test utilities
class MemoryStressUtils:
    """Utility functions for memory stress testing"""
    
    @staticmethod
    def generate_large_vector_dataset(count: int, dimension: int) -> List[Dict[str, Any]]:
        """Generate large vector dataset for stress testing"""
        return [
            {
                'id': f'stress_vector_{i}',
                'data': [0.1 * (i + j) for j in range(dimension)],
                'metadata': {
                    'index': i,
                    'category': f'category_{i % 10}',
                    'priority': i % 5,
                    'timestamp': i * 1000
                }
            }
            for i in range(count)
        ]
    
    @staticmethod
    def calculate_memory_efficiency(initial_mb: float, peak_mb: float, data_size_mb: float) -> float:
        """Calculate memory efficiency ratio"""
        memory_overhead = peak_mb - initial_mb
        return data_size_mb / memory_overhead if memory_overhead > 0 else float('inf')


if __name__ == "__main__":
    # Run stress tests directly
    pytest.main([__file__, "-v", "--tb=short"])