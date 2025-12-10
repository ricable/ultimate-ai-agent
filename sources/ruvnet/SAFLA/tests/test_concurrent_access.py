"""
Concurrent Access Testing Module

This module implements comprehensive concurrent access tests for the SAFLA memory bank system,
addressing the "Race condition detected" scenario reported in the MCP server mock data.
"""

import pytest
import asyncio
import threading
import time
from typing import List, Dict, Any, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import Mock, patch
import random

from safla.core.hybrid_memory import HybridMemoryArchitecture, SimilarityMetric, EvictionPolicy
from safla.core.optimized_vector_memory import OptimizedVectorMemoryManager


class TestConcurrentAccess:
    """Comprehensive concurrent access testing suite"""
    
    @pytest.fixture
    async def memory_system(self):
        """Create a memory system for concurrent testing"""
        config = {
            'vector_memory': {
                'embedding_dim': 256,
                'max_capacity': 5000,
                'similarity_metric': SimilarityMetric.COSINE,
                'eviction_policy': EvictionPolicy.LRU
            },
            'episodic_memory': {
                'max_capacity': 500
            },
            'working_memory': {
                'capacity': 25,
                'attention_window': 5
            }
        }

        memory = HybridMemoryArchitecture(
            vector_config=config.get('vector_memory'),
            episodic_config=config.get('episodic_memory'),
            working_config=config.get('working_memory')
        )
        # Cleanup using the actual available method
        memory.cleanup_memory()
    
    @pytest.fixture
    def race_condition_detector(self):
        """Race condition detection utility"""
        class RaceConditionDetector:
            def __init__(self):
                self.operations = []
                self.lock = threading.Lock()
                self.conflicts = []
                self.timing_violations = []
            
            def record_operation(self, operation_type: str, resource_id: str, thread_id: int, timestamp: float):
                with self.lock:
                    self.operations.append({
                        'type': operation_type,
                        'resource_id': resource_id,
                        'thread_id': thread_id,
                        'timestamp': timestamp
                    })
            
            def detect_conflicts(self):
                """Detect potential race conditions"""
                # Group operations by resource
                resource_ops = {}
                for op in self.operations:
                    resource_id = op['resource_id']
                    if resource_id not in resource_ops:
                        resource_ops[resource_id] = []
                    resource_ops[resource_id].append(op)
                
                # Check for concurrent write operations
                for resource_id, ops in resource_ops.items():
                    write_ops = [op for op in ops if op['type'] in ['store', 'update', 'delete']]
                    if len(write_ops) > 1:
                        # Check if writes occurred within dangerous time window
                        for i, op1 in enumerate(write_ops):
                            for op2 in write_ops[i+1:]:
                                time_diff = abs(op1['timestamp'] - op2['timestamp'])
                                if time_diff < 0.01:  # 10ms window
                                    self.conflicts.append({
                                        'resource_id': resource_id,
                                        'operations': [op1, op2],
                                        'time_diff': time_diff
                                    })
                
                return len(self.conflicts) == 0
        
        return RaceConditionDetector()
    
    @pytest.mark.asyncio
    async def test_concurrent_vector_storage(self, memory_system, race_condition_detector):
        """Test concurrent vector storage operations"""
        async def store_vectors_batch(batch_id: int, start_index: int):
            thread_id = threading.get_ident()
            vectors = []
            
            for i in range(100):
                vector_id = f'concurrent_store_{batch_id}_{start_index + i}'
                vector_data = [0.1 * (start_index + i + j) for j in range(256)]
                
                vectors.append({
                    'id': vector_id,
                    'data': vector_data,
                    'metadata': {'batch': batch_id, 'thread': thread_id}
                })
                
                race_condition_detector.record_operation(
                    'store', vector_id, thread_id, time.time()
                )
            
            await memory_system.store_vectors(vectors)
            return len(vectors)
        
        # Launch concurrent storage operations
        tasks = []
        for batch_id in range(10):
            start_index = batch_id * 100
            tasks.append(store_vectors_batch(batch_id, start_index))
        
        results = await asyncio.gather(*tasks)
        
        # Verify all vectors were stored
        total_stored = sum(results)
        assert total_stored == 1000, f"Expected 1000 vectors, got {total_stored}"
        
        # Check for race conditions
        no_conflicts = race_condition_detector.detect_conflicts()
        assert no_conflicts, f"Race conditions detected: {race_condition_detector.conflicts}"
        
        # Verify data integrity
        stored_count = await memory_system.get_vector_count()
        assert stored_count >= 1000, f"Vector count mismatch: expected >= 1000, got {stored_count}"
    
    @pytest.mark.asyncio
    async def test_concurrent_read_write_operations(self, memory_system, race_condition_detector):
        """Test concurrent read and write operations"""
        # Pre-populate with test data
        initial_vectors = [
            {
                'id': f'rw_test_{i}',
                'data': [0.1 * i] * 256,
                'metadata': {'index': i, 'type': 'read_write_test'}
            }
            for i in range(500)
        ]
        await memory_system.store_vectors(initial_vectors)
        
        read_results = []
        write_results = []
        
        async def read_operation(reader_id: int):
            thread_id = threading.get_ident()
            results = []
            
            for i in range(50):
                query_vector = [0.1 * (reader_id + i)] * 256
                race_condition_detector.record_operation(
                    'read', f'query_{reader_id}_{i}', thread_id, time.time()
                )
                
                search_results = await memory_system.search_similar(query_vector, top_k=5)
                results.append(len(search_results))
            
            return results
        
        async def write_operation(writer_id: int):
            thread_id = threading.get_ident()
            vectors_written = 0
            
            for i in range(25):
                vector_id = f'concurrent_write_{writer_id}_{i}'
                vector_data = [0.1 * (writer_id * 100 + i)] * 256
                
                race_condition_detector.record_operation(
                    'store', vector_id, thread_id, time.time()
                )
                
                await memory_system.store_vector({
                    'id': vector_id,
                    'data': vector_data,
                    'metadata': {'writer': writer_id, 'concurrent': True}
                })
                vectors_written += 1
            
            return vectors_written
        
        async def update_operation(updater_id: int):
            thread_id = threading.get_ident()
            updates_performed = 0
            
            for i in range(20):
                vector_id = f'rw_test_{updater_id * 10 + i}'
                race_condition_detector.record_operation(
                    'update', vector_id, thread_id, time.time()
                )
                
                # Update vector metadata
                success = await memory_system.update_vector_metadata(
                    vector_id, {'updated_by': updater_id, 'update_time': time.time()}
                )
                if success:
                    updates_performed += 1
            
            return updates_performed
        
        # Launch concurrent operations
        tasks = []
        
        # Readers
        for reader_id in range(5):
            tasks.append(read_operation(reader_id))
        
        # Writers
        for writer_id in range(3):
            tasks.append(write_operation(writer_id))
        
        # Updaters
        for updater_id in range(2):
            tasks.append(update_operation(updater_id))
        
        results = await asyncio.gather(*tasks)
        
        # Analyze results
        read_results = results[:5]  # First 5 are read operations
        write_results = results[5:8]  # Next 3 are write operations
        update_results = results[8:]  # Last 2 are update operations
        
        # Verify read operations succeeded
        for read_result in read_results:
            assert all(count > 0 for count in read_result), "All read operations should return results"
        
        # Verify write operations succeeded
        total_written = sum(write_results)
        assert total_written == 75, f"Expected 75 writes, got {total_written}"
        
        # Verify update operations succeeded
        total_updated = sum(update_results)
        assert total_updated > 0, "Some update operations should succeed"
        
        # Check for race conditions
        no_conflicts = race_condition_detector.detect_conflicts()
        assert no_conflicts, f"Race conditions detected: {race_condition_detector.conflicts}"
    
    @pytest.mark.asyncio
    async def test_concurrent_delete_operations(self, memory_system, race_condition_detector):
        """Test concurrent delete operations and data consistency"""
        # Pre-populate with test data
        test_vectors = [
            {
                'id': f'delete_test_{i}',
                'data': [0.1 * i] * 256,
                'metadata': {'index': i, 'deletable': True}
            }
            for i in range(1000)
        ]
        await memory_system.store_vectors(test_vectors)
        
        deleted_ids = set()
        deletion_conflicts = []
        
        async def delete_operation(deleter_id: int, start_index: int, count: int):
            thread_id = threading.get_ident()
            local_deleted = []
            
            for i in range(count):
                vector_id = f'delete_test_{start_index + i}'
                race_condition_detector.record_operation(
                    'delete', vector_id, thread_id, time.time()
                )
                
                try:
                    success = await memory_system.delete_vector(vector_id)
                    if success:
                        local_deleted.append(vector_id)
                except Exception as e:
                    # Record potential conflict
                    deletion_conflicts.append({
                        'vector_id': vector_id,
                        'deleter_id': deleter_id,
                        'error': str(e)
                    })
            
            return local_deleted
        
        # Launch concurrent delete operations with overlapping ranges
        tasks = []
        tasks.append(delete_operation(1, 0, 200))    # Delete 0-199
        tasks.append(delete_operation(2, 150, 200))  # Delete 150-349 (overlap with deleter 1)
        tasks.append(delete_operation(3, 300, 200))  # Delete 300-499 (overlap with deleter 2)
        tasks.append(delete_operation(4, 450, 200))  # Delete 450-649 (overlap with deleter 3)
        
        results = await asyncio.gather(*tasks)
        
        # Collect all deleted IDs
        for deleted_list in results:
            deleted_ids.update(deleted_list)
        
        # Verify no duplicate deletions (race condition indicator)
        total_deletion_attempts = sum(len(result) for result in results)
        unique_deletions = len(deleted_ids)
        
        # Some overlap is expected, but excessive conflicts indicate race conditions
        conflict_ratio = (total_deletion_attempts - unique_deletions) / total_deletion_attempts
        assert conflict_ratio < 0.3, f"Too many deletion conflicts: {conflict_ratio:.2%}"
        
        # Verify system consistency
        remaining_count = await memory_system.get_vector_count()
        expected_remaining = 1000 - unique_deletions
        
        # Allow for some variance due to concurrent operations
        assert abs(remaining_count - expected_remaining) <= 10, \
            f"Vector count inconsistency: expected ~{expected_remaining}, got {remaining_count}"
        
        # Check for race conditions
        no_conflicts = race_condition_detector.detect_conflicts()
        assert no_conflicts, f"Race conditions detected: {race_condition_detector.conflicts}"
    
    @pytest.mark.asyncio
    async def test_concurrent_search_operations(self, memory_system):
        """Test concurrent search operations under load"""
        # Pre-populate with diverse test data
        test_vectors = []
        for category in range(10):
            for i in range(100):
                vector_data = [0.1 * category + 0.01 * i + 0.001 * j for j in range(256)]
                test_vectors.append({
                    'id': f'search_test_{category}_{i}',
                    'data': vector_data,
                    'metadata': {'category': category, 'index': i}
                })
        
        memory_system.vector_memory.batch_store([v['data'] for v in test_vectors], [v['metadata'] for v in test_vectors])
        
        search_results = []
        search_times = []
        
        async def search_operation(searcher_id: int, query_count: int):
            local_results = []
            local_times = []
            
            for i in range(query_count):
                # Generate diverse query vectors
                category = searcher_id % 10
                query_vector = [0.1 * category + 0.01 * i + 0.001 * j for j in range(256)]
                
                start_time = time.time()
                results = await memory_system.integrated_search(query_vector, k=10)
                end_time = time.time()
                
                local_results.append(len(results))
                local_times.append(end_time - start_time)
            
            return local_results, local_times
        
        # Launch concurrent search operations
        tasks = []
        for searcher_id in range(20):  # 20 concurrent searchers
            tasks.append(search_operation(searcher_id, 50))  # 50 searches each
        
        results = await asyncio.gather(*tasks)
        
        # Analyze search performance
        all_result_counts = []
        all_search_times = []
        
        for result_counts, times in results:
            all_result_counts.extend(result_counts)
            all_search_times.extend(times)
        
        # Verify search quality
        avg_results = sum(all_result_counts) / len(all_result_counts)
        assert avg_results >= 8, f"Average search results too low: {avg_results}"
        
        # Verify search performance
        avg_search_time = sum(all_search_times) / len(all_search_times)
        max_search_time = max(all_search_times)
        
        assert avg_search_time < 0.1, f"Average search time too high: {avg_search_time:.3f}s"
        assert max_search_time < 0.5, f"Maximum search time too high: {max_search_time:.3f}s"
        
        # Verify no search returned empty results (system availability)
        empty_searches = sum(1 for count in all_result_counts if count == 0)
        empty_ratio = empty_searches / len(all_result_counts)
        assert empty_ratio < 0.05, f"Too many empty searches: {empty_ratio:.2%}"
    
    @pytest.mark.asyncio
    async def test_deadlock_prevention(self, memory_system):
        """Test deadlock prevention in concurrent operations"""
        deadlock_detected = False
        operation_timeouts = []
        
        async def complex_operation(op_id: int):
            """Complex operation that could potentially cause deadlocks"""
            try:
                # Operation with multiple resource accesses
                start_time = time.time()
                
                # Step 1: Store vectors
                vectors = [
                    {
                        'id': f'deadlock_test_{op_id}_{i}',
                        'data': [0.1 * (op_id + i)] * 256,
                        'metadata': {'operation': op_id, 'step': 1}
                    }
                    for i in range(50)
                ]
                memory_system.vector_memory.batch_store([v['data'] for v in vectors], [v['metadata'] for v in vectors])
                
                # Step 2: Search for similar vectors
                query_vector = [0.1 * op_id] * 256
                search_results = await memory_system.integrated_search(query_vector, k=20)
                
                # Step 3: Update some vectors based on search results
                for result in search_results[:10]:
                    memory_system.vector_memory.update_metadata(
                        result['id'], {'updated_by_op': op_id, 'timestamp': time.time()}
                    )
                
                # Step 4: Delete some vectors
                for i in range(0, 25, 2):  # Delete every other vector
                    memory_system.vector_memory.delete(f'deadlock_test_{op_id}_{i}')
                
                end_time = time.time()
                operation_time = end_time - start_time
                
                return operation_time
                
            except asyncio.TimeoutError:
                return None  # Timeout indicates potential deadlock
        
        # Launch operations that could potentially deadlock
        timeout_duration = 30.0  # 30 second timeout
        
        tasks = []
        for op_id in range(8):  # 8 concurrent complex operations
            task = asyncio.wait_for(complex_operation(op_id), timeout=timeout_duration)
            tasks.append(task)
        
        # Execute with timeout to detect deadlocks
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Analyze results
            successful_ops = [r for r in results if isinstance(r, float)]
            timeout_ops = [r for r in results if isinstance(r, asyncio.TimeoutError)]
            
            # Check for deadlocks (indicated by timeouts)
            deadlock_ratio = len(timeout_ops) / len(results)
            assert deadlock_ratio < 0.25, f"Too many operations timed out (potential deadlock): {deadlock_ratio:.2%}"
            
            # Verify performance of successful operations
            if successful_ops:
                avg_time = sum(successful_ops) / len(successful_ops)
                max_time = max(successful_ops)
                
                assert avg_time < 10.0, f"Average operation time too high: {avg_time:.2f}s"
                assert max_time < 20.0, f"Maximum operation time too high: {max_time:.2f}s"
        
        except Exception as e:
            pytest.fail(f"Unexpected error in deadlock test: {e}")
    
    @pytest.mark.asyncio
    async def test_thread_safety_validation(self, memory_system):
        """Validate thread safety of memory operations"""
        shared_counter = {'value': 0}
        counter_lock = threading.Lock()
        thread_results = {}
        
        def thread_worker(thread_id: int, operation_count: int):
            """Worker function for thread-based testing"""
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                async def async_worker():
                    local_operations = 0
                    
                    for i in range(operation_count):
                        # Increment shared counter (test for race conditions)
                        with counter_lock:
                            shared_counter['value'] += 1
                            expected_value = shared_counter['value']
                        
                        # Perform memory operation
                        vector = {
                            'id': f'thread_test_{thread_id}_{i}',
                            'data': [0.1 * (thread_id + i)] * 256,
                            'metadata': {'thread': thread_id, 'operation': i}
                        }
                        
                        await memory_system.store_vector(vector)
                        local_operations += 1
                        
                        # Verify shared counter consistency
                        with counter_lock:
                            current_value = shared_counter['value']
                            assert current_value >= expected_value, \
                                f"Counter inconsistency: expected >= {expected_value}, got {current_value}"
                    
                    return local_operations
                
                result = loop.run_until_complete(async_worker())
                thread_results[thread_id] = result
                
            finally:
                loop.close()
        
        # Launch multiple threads
        threads = []
        thread_count = 5
        operations_per_thread = 100
        
        for thread_id in range(thread_count):
            thread = threading.Thread(
                target=thread_worker,
                args=(thread_id, operations_per_thread)
            )
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=30.0)  # 30 second timeout
            assert not thread.is_alive(), f"Thread {thread.ident} did not complete in time"
        
        # Verify results
        total_operations = sum(thread_results.values())
        expected_operations = thread_count * operations_per_thread
        
        assert total_operations == expected_operations, \
            f"Operation count mismatch: expected {expected_operations}, got {total_operations}"
        
        # Verify shared counter consistency
        expected_counter_value = thread_count * operations_per_thread
        actual_counter_value = shared_counter['value']
        
        assert actual_counter_value == expected_counter_value, \
            f"Shared counter inconsistency: expected {expected_counter_value}, got {actual_counter_value}"
        
        # Verify memory system state
        stored_vectors = await memory_system.get_vector_count()
        assert stored_vectors >= expected_operations, \
            f"Vector storage inconsistency: expected >= {expected_operations}, got {stored_vectors}"


# Additional concurrent testing utilities
class ConcurrentTestUtils:
    """Utility functions for concurrent testing"""
    
    @staticmethod
    async def stress_test_concurrent_operations(memory_system, operation_count: int, concurrency_level: int):
        """Generic stress test for concurrent operations"""
        async def operation_worker(worker_id: int):
            operations_completed = 0
            
            for i in range(operation_count // concurrency_level):
                # Mix of different operations
                if i % 4 == 0:
                    # Store operation
                    vector = {
                        'id': f'stress_{worker_id}_{i}',
                        'data': [0.1 * (worker_id + i)] * 256,
                        'metadata': {'worker': worker_id, 'type': 'stress'}
                    }
                    await memory_system.store_vector(vector)
                elif i % 4 == 1:
                    # Search operation
                    query = [0.1 * worker_id] * 256
                    await memory_system.search_similar(query, top_k=5)
                elif i % 4 == 2:
                    # Update operation
                    vector_id = f'stress_{worker_id}_{i-2}'
                    await memory_system.update_vector_metadata(
                        vector_id, {'updated': True, 'timestamp': time.time()}
                    )
                else:
                    # Delete operation
                    vector_id = f'stress_{worker_id}_{i-3}'
                    await memory_system.delete_vector(vector_id)
                
                operations_completed += 1
            
            return operations_completed
        
        # Launch concurrent workers
        tasks = [operation_worker(i) for i in range(concurrency_level)]
        results = await asyncio.gather(*tasks)
        
        return sum(results)
    
    @staticmethod
    def detect_race_conditions(operation_log: List[Dict]) -> List[Dict]:
        """Analyze operation log for potential race conditions"""
        race_conditions = []
        
        # Group operations by resource
        resource_operations = {}
        for op in operation_log:
            resource = op.get('resource_id')
            if resource not in resource_operations:
                resource_operations[resource] = []
            resource_operations[resource].append(op)
        
        # Check for concurrent writes
        for resource, ops in resource_operations.items():
            write_ops = [op for op in ops if op.get('type') in ['store', 'update', 'delete']]
            
            for i, op1 in enumerate(write_ops):
                for op2 in write_ops[i+1:]:
                    time_diff = abs(op1.get('timestamp', 0) - op2.get('timestamp', 0))
                    if time_diff < 0.01:  # 10ms window
                        race_conditions.append({
                            'resource': resource,
                            'operation1': op1,
                            'operation2': op2,
                            'time_difference': time_diff
                        })
        
        return race_conditions


if __name__ == "__main__":
    # Run concurrent access tests directly
    pytest.main([__file__, "-v", "--tb=short"])