"""
Performance Integration Tests for SAFLA
======================================

This module tests system performance under various load conditions and validates
that all components maintain acceptable performance when working together.

Performance testing areas:
1. Concurrent operations across all components
2. Memory pressure and optimization under load
3. Throughput and latency benchmarks
4. Resource utilization optimization
5. Scalability testing
6. Performance degradation detection and recovery

Following TDD principles: Red-Green-Refactor cycle.
"""

import pytest
import asyncio
import time
import statistics
import psutil
import threading
from typing import Dict, List, Any, Optional, Tuple
from unittest.mock import Mock, patch
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# Import SAFLA components
from safla.core.delta_evaluation import DeltaEvaluator, DeltaMetrics
from safla.core.meta_cognitive_engine import MetaCognitiveEngine
from safla.core.hybrid_memory import HybridMemoryArchitecture, MemoryItem
from safla.core.mcp_orchestration import MCPOrchestrator, Agent
from safla.core.safety_validation import SafetyValidationFramework
from safla.core.memory_optimizations import OptimizedVectorMemoryManager


@dataclass
class PerformanceMetrics:
    """Performance metrics for testing."""
    operation_type: str
    total_operations: int
    total_time: float
    operations_per_second: float
    average_latency: float
    p95_latency: float
    p99_latency: float
    error_rate: float
    memory_usage_mb: float
    cpu_usage_percent: float
    success_rate: float
    
    @property
    def throughput_score(self) -> float:
        """Calculate throughput score (0-1)."""
        # Normalize based on expected minimums
        return min(1.0, self.operations_per_second / 100.0)
    
    @property
    def latency_score(self) -> float:
        """Calculate latency score (0-1, higher is better)."""
        # Lower latency is better, normalize to 0-1 scale
        return max(0.0, 1.0 - (self.average_latency / 5.0))
    
    @property
    def overall_score(self) -> float:
        """Calculate overall performance score."""
        return (
            self.throughput_score * 0.3 +
            self.latency_score * 0.3 +
            self.success_rate * 0.4
        )


@dataclass
class LoadTestConfiguration:
    """Configuration for load testing."""
    name: str
    concurrent_workers: int
    operations_per_worker: int
    duration_seconds: float
    operation_mix: Dict[str, float]  # operation_type -> percentage
    ramp_up_seconds: float = 0.0
    expected_performance: Dict[str, float] = field(default_factory=dict)


class TestConcurrentOperations:
    """Test concurrent operations across all components."""
    
    @pytest.mark.asyncio
    async def test_mixed_concurrent_operations(self, integrated_system, performance_monitor):
        """Test mixed concurrent operations across all components."""
        context = integrated_system
        components = context.components
        
        # Define operation types and their weights
        operation_types = {
            'memory_store': 0.3,
            'delta_evaluation': 0.25,
            'mcp_task': 0.25,
            'meta_cognitive_event': 0.2
        }
        
        # Performance test configuration
        config = LoadTestConfiguration(
            name='mixed_concurrent_operations',
            concurrent_workers=20,
            operations_per_worker=50,
            duration_seconds=30.0,
            operation_mix=operation_types,
            expected_performance={
                'min_ops_per_second': 50,
                'max_average_latency': 2.0,
                'min_success_rate': 0.95
            }
        )
        
        # Track performance metrics
        operation_times = {op_type: [] for op_type in operation_types.keys()}
        operation_errors = {op_type: 0 for op_type in operation_types.keys()}
        operation_successes = {op_type: 0 for op_type in operation_types.keys()}
        
        async def worker(worker_id: int) -> Dict[str, Any]:
            """Worker function for concurrent operations."""
            worker_metrics = {
                'operations_completed': 0,
                'errors_encountered': 0,
                'operation_times': {op_type: [] for op_type in operation_types.keys()}
            }
            
            for i in range(config.operations_per_worker):
                # Select operation type based on mix
                import random
                rand_val = random.random()
                cumulative = 0.0
                selected_op = None
                
                for op_type, weight in operation_types.items():
                    cumulative += weight
                    if rand_val <= cumulative:
                        selected_op = op_type
                        break
                
                if selected_op is None:
                    selected_op = list(operation_types.keys())[0]
                
                # Execute operation
                start_time = time.time()
                try:
                    if selected_op == 'memory_store':
                        node = MemoryNode(
                            id=f"perf_test_{worker_id}_{i}",
                            content=f"Performance test content {worker_id}-{i}",
                            embedding=[0.1 * ((worker_id + i) % 10)] * 128,
                            metadata={'worker': worker_id, 'iteration': i}
                        )
                        await components['memory_system'].store_node(node)
                    
                    elif selected_op == 'delta_evaluation':
                        metrics = DeltaMetrics(
                            performance=0.1 + (i * 0.001),
                            efficiency=0.05 + (i * 0.0005),
                            stability=0.02,
                            capability=0.08,
                            confidence=0.9
                        )
                        await components['delta_evaluator'].evaluate(metrics)
                    
                    elif selected_op == 'mcp_task':
                        task = AgentTask(
                            id=f"perf_task_{worker_id}_{i}",
                            type="performance_test",
                            payload={'worker': worker_id, 'iteration': i},
                            priority=1
                        )
                        await components['mcp_orchestrator'].submit_task(task)
                    
                    elif selected_op == 'meta_cognitive_event':
                        event = {
                            'type': 'performance_test_event',
                            'worker_id': worker_id,
                            'iteration': i,
                            'timestamp': time.time()
                        }
                        await components['meta_engine'].process_system_event(event)
                    
                    end_time = time.time()
                    operation_time = end_time - start_time
                    
                    worker_metrics['operation_times'][selected_op].append(operation_time)
                    worker_metrics['operations_completed'] += 1
                    
                    operation_times[selected_op].append(operation_time)
                    operation_successes[selected_op] += 1
                    
                except Exception as e:
                    worker_metrics['errors_encountered'] += 1
                    operation_errors[selected_op] += 1
                    logging.warning(f"Worker {worker_id} operation {selected_op} failed: {e}")
                
                # Small delay to prevent overwhelming
                await asyncio.sleep(0.01)
            
            return worker_metrics
        
        # Execute concurrent load test
        start_time = time.time()
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        initial_cpu = psutil.cpu_percent()
        
        # Create and run workers
        workers = [worker(i) for i in range(config.concurrent_workers)]
        worker_results = await asyncio.gather(*workers, return_exceptions=True)
        
        end_time = time.time()
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        final_cpu = psutil.cpu_percent()
        
        total_time = end_time - start_time
        
        # Calculate performance metrics for each operation type
        performance_results = {}
        
        for op_type in operation_types.keys():
            times = operation_times[op_type]
            successes = operation_successes[op_type]
            errors = operation_errors[op_type]
            total_ops = successes + errors
            
            if times:
                metrics = PerformanceMetrics(
                    operation_type=op_type,
                    total_operations=total_ops,
                    total_time=total_time,
                    operations_per_second=total_ops / total_time,
                    average_latency=statistics.mean(times),
                    p95_latency=statistics.quantiles(times, n=20)[18] if len(times) > 20 else max(times),
                    p99_latency=statistics.quantiles(times, n=100)[98] if len(times) > 100 else max(times),
                    error_rate=errors / total_ops if total_ops > 0 else 0,
                    memory_usage_mb=final_memory - initial_memory,
                    cpu_usage_percent=final_cpu,
                    success_rate=successes / total_ops if total_ops > 0 else 0
                )
                performance_results[op_type] = metrics
        
        # Verify performance benchmarks
        total_operations = sum(len(times) for times in operation_times.values())
        overall_ops_per_second = total_operations / total_time
        overall_error_rate = sum(operation_errors.values()) / total_operations
        
        # Performance assertions
        assert overall_ops_per_second >= config.expected_performance['min_ops_per_second'], \
            f"Operations per second {overall_ops_per_second} below minimum {config.expected_performance['min_ops_per_second']}"
        
        assert overall_error_rate <= (1 - config.expected_performance['min_success_rate']), \
            f"Error rate {overall_error_rate} too high, success rate below {config.expected_performance['min_success_rate']}"
        
        # Verify each operation type meets performance standards
        for op_type, metrics in performance_results.items():
            assert metrics.average_latency <= config.expected_performance['max_average_latency'], \
                f"{op_type} average latency {metrics.average_latency} exceeds maximum {config.expected_performance['max_average_latency']}"
            
            assert metrics.success_rate >= config.expected_performance['min_success_rate'], \
                f"{op_type} success rate {metrics.success_rate} below minimum {config.expected_performance['min_success_rate']}"
        
        # Log performance results
        context.log_event('performance_test_completed', {
            'config': config.__dict__,
            'results': {op_type: metrics.__dict__ for op_type, metrics in performance_results.items()},
            'overall_metrics': {
                'total_operations': total_operations,
                'total_time': total_time,
                'operations_per_second': overall_ops_per_second,
                'error_rate': overall_error_rate,
                'memory_increase_mb': final_memory - initial_memory,
                'cpu_usage': final_cpu
            }
        })
    
    @pytest.mark.asyncio
    async def test_memory_pressure_performance(self, integrated_system, integration_test_helpers):
        """Test system performance under memory pressure."""
        context = integrated_system
        components = context.components
        
        # Phase 1: Establish baseline performance
        baseline_ops = 100
        baseline_times = []
        
        for i in range(baseline_ops):
            start_time = time.time()
            node = MemoryNode(
                id=f"baseline_{i}",
                content=f"Baseline content {i}",
                embedding=[0.1] * 128
            )
            await components['memory_system'].store_node(node)
            baseline_times.append(time.time() - start_time)
        
        baseline_avg_time = statistics.mean(baseline_times)
        baseline_memory = await components['memory_system'].get_memory_usage()
        
        # Phase 2: Create memory pressure by storing large amounts of data
        memory_pressure_nodes = 2000  # Significantly more than baseline
        pressure_times = []
        
        for i in range(memory_pressure_nodes):
            start_time = time.time()
            node = MemoryNode(
                id=f"pressure_{i}",
                content=f"Memory pressure content {i}" * 10,  # Larger content
                embedding=[0.1 * (i % 10)] * 128,
                metadata={'pressure_test': True, 'size': 'large'}
            )
            await components['memory_system'].store_node(node)
            pressure_times.append(time.time() - start_time)
            
            # Check if optimization was triggered
            if i % 100 == 0:
                current_usage = await components['memory_system'].get_memory_usage()
                if current_usage > 0.9:  # High memory usage
                    break
        
        # Phase 3: Measure performance under pressure
        pressure_avg_time = statistics.mean(pressure_times)
        pressure_memory = await components['memory_system'].get_memory_usage()
        
        # Phase 4: Verify optimization was triggered
        optimization_history = components['memory_optimizer'].get_optimization_history()
        assert len(optimization_history) > 0, "Memory optimization should have been triggered"
        
        # Phase 5: Test performance after optimization
        await asyncio.sleep(1.0)  # Allow optimization to complete
        
        post_optimization_ops = 100
        post_opt_times = []
        
        for i in range(post_optimization_ops):
            start_time = time.time()
            node = MemoryNode(
                id=f"post_opt_{i}",
                content=f"Post optimization content {i}",
                embedding=[0.1] * 128
            )
            await components['memory_system'].store_node(node)
            post_opt_times.append(time.time() - start_time)
        
        post_opt_avg_time = statistics.mean(post_opt_times)
        post_opt_memory = await components['memory_system'].get_memory_usage()
        
        # Performance assertions
        # Performance should not degrade more than 3x under pressure
        assert pressure_avg_time <= baseline_avg_time * 3, \
            f"Performance degraded too much under pressure: {pressure_avg_time} vs baseline {baseline_avg_time}"
        
        # Memory optimization should improve performance
        assert post_opt_avg_time <= pressure_avg_time * 1.2, \
            f"Performance did not improve after optimization: {post_opt_avg_time} vs pressure {pressure_avg_time}"
        
        # Memory usage should be reduced after optimization
        assert post_opt_memory < pressure_memory, \
            f"Memory usage not reduced after optimization: {post_opt_memory} vs {pressure_memory}"
        
        # System should remain stable
        system_state = await components['meta_engine'].get_system_state()
        assert system_state['status'] in ['operational', 'degraded'], \
            f"System not stable after memory pressure test: {system_state['status']}"
        
        # Log memory pressure test results
        context.log_event('memory_pressure_test', {
            'baseline_avg_time': baseline_avg_time,
            'pressure_avg_time': pressure_avg_time,
            'post_opt_avg_time': post_opt_avg_time,
            'baseline_memory': baseline_memory,
            'pressure_memory': pressure_memory,
            'post_opt_memory': post_opt_memory,
            'performance_degradation': pressure_avg_time / baseline_avg_time,
            'optimization_improvement': pressure_avg_time / post_opt_avg_time,
            'memory_reduction': (pressure_memory - post_opt_memory) / pressure_memory
        })
    
    @pytest.mark.asyncio
    async def test_throughput_scaling(self, integrated_system):
        """Test system throughput scaling with increasing load."""
        context = integrated_system
        components = context.components
        
        # Test different load levels
        load_levels = [10, 25, 50, 100, 200]
        throughput_results = []
        
        for load_level in load_levels:
            # Configure load test
            operations_count = load_level * 10  # 10 operations per load unit
            concurrent_workers = min(load_level, 50)  # Cap workers at 50
            
            # Execute load test
            start_time = time.time()
            operation_times = []
            errors = 0
            
            async def load_worker(worker_id: int, ops_per_worker: int):
                worker_errors = 0
                worker_times = []
                
                for i in range(ops_per_worker):
                    op_start = time.time()
                    try:
                        # Mix of operations
                        if i % 3 == 0:
                            # Memory operation
                            node = MemoryNode(
                                id=f"scale_test_{load_level}_{worker_id}_{i}",
                                content=f"Scaling test content {load_level}-{worker_id}-{i}",
                                embedding=[0.1] * 128
                            )
                            await components['memory_system'].store_node(node)
                        elif i % 3 == 1:
                            # Delta evaluation
                            metrics = DeltaMetrics(
                                performance=0.1, efficiency=0.05, stability=0.02, capability=0.08
                            )
                            await components['delta_evaluator'].evaluate(metrics)
                        else:
                            # MCP task
                            task = AgentTask(
                                id=f"scale_task_{load_level}_{worker_id}_{i}",
                                type="scaling_test",
                                payload={'load_level': load_level, 'worker': worker_id}
                            )
                            await components['mcp_orchestrator'].submit_task(task)
                        
                        worker_times.append(time.time() - op_start)
                    except Exception as e:
                        worker_errors += 1
                        logging.warning(f"Load worker error at level {load_level}: {e}")
                
                return worker_times, worker_errors
            
            # Distribute operations across workers
            ops_per_worker = operations_count // concurrent_workers
            remaining_ops = operations_count % concurrent_workers
            
            workers = []
            for i in range(concurrent_workers):
                worker_ops = ops_per_worker + (1 if i < remaining_ops else 0)
                workers.append(load_worker(i, worker_ops))
            
            # Execute workers
            worker_results = await asyncio.gather(*workers, return_exceptions=True)
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # Aggregate results
            all_times = []
            total_errors = 0
            
            for result in worker_results:
                if isinstance(result, tuple):
                    times, worker_errors = result
                    all_times.extend(times)
                    total_errors += worker_errors
                else:
                    total_errors += 1  # Worker failed completely
            
            # Calculate metrics
            successful_ops = len(all_times)
            throughput = successful_ops / total_time
            avg_latency = statistics.mean(all_times) if all_times else float('inf')
            error_rate = total_errors / operations_count
            
            throughput_results.append({
                'load_level': load_level,
                'operations_count': operations_count,
                'successful_operations': successful_ops,
                'throughput': throughput,
                'average_latency': avg_latency,
                'error_rate': error_rate,
                'total_time': total_time
            })
            
            # Brief pause between load levels
            await asyncio.sleep(0.5)
        
        # Analyze scaling characteristics
        throughputs = [r['throughput'] for r in throughput_results]
        latencies = [r['average_latency'] for r in throughput_results]
        error_rates = [r['error_rate'] for r in throughput_results]
        
        # Verify scaling properties
        # Throughput should generally increase with load (up to a point)
        max_throughput = max(throughputs)
        max_throughput_index = throughputs.index(max_throughput)
        
        # Should achieve reasonable throughput
        assert max_throughput >= 50, f"Maximum throughput {max_throughput} too low"
        
        # Error rates should remain reasonable
        max_error_rate = max(error_rates)
        assert max_error_rate <= 0.1, f"Maximum error rate {max_error_rate} too high"
        
        # Latency should not increase exponentially
        min_latency = min(latencies)
        max_latency = max(latencies)
        latency_increase_factor = max_latency / min_latency
        assert latency_increase_factor <= 10, f"Latency increased too much: {latency_increase_factor}x"
        
        # Log scaling test results
        context.log_event('throughput_scaling_test', {
            'load_levels': load_levels,
            'results': throughput_results,
            'max_throughput': max_throughput,
            'max_throughput_at_load': load_levels[max_throughput_index],
            'latency_increase_factor': latency_increase_factor,
            'max_error_rate': max_error_rate
        })


class TestResourceUtilization:
    """Test resource utilization optimization."""
    
    @pytest.mark.asyncio
    async def test_cpu_utilization_optimization(self, integrated_system, integration_test_helpers):
        """Test CPU utilization optimization under different workloads."""
        context = integrated_system
        components = context.components
        
        # Monitor CPU usage during different operation types
        cpu_measurements = {}
        
        # Test 1: CPU-intensive delta evaluations
        cpu_start = psutil.cpu_percent(interval=None)
        start_time = time.time()
        
        delta_tasks = []
        for i in range(100):
            metrics = DeltaMetrics(
                performance=0.1 + (i * 0.001),
                efficiency=0.05 + (i * 0.0005),
                stability=0.02,
                capability=0.08,
                confidence=0.9
            )
            delta_tasks.append(components['delta_evaluator'].evaluate(metrics))
        
        await asyncio.gather(*delta_tasks)
        
        delta_time = time.time() - start_time
        delta_cpu = psutil.cpu_percent(interval=0.1)
        cpu_measurements['delta_evaluation'] = {
            'cpu_usage': delta_cpu - cpu_start,
            'operations': 100,
            'time': delta_time,
            'cpu_efficiency': 100 / (delta_cpu - cpu_start) if delta_cpu > cpu_start else float('inf')
        }
        
        # Test 2: Memory-intensive operations
        cpu_start = psutil.cpu_percent(interval=None)
        start_time = time.time()
        
        memory_tasks = []
        for i in range(200):
            node = MemoryNode(
                id=f"cpu_test_{i}",
                content=f"CPU test content {i}" * 20,  # Larger content
                embedding=[0.1 * (i % 10)] * 128
            )
            memory_tasks.append(components['memory_system'].store_node(node))
        
        await asyncio.gather(*memory_tasks)
        
        memory_time = time.time() - start_time
        memory_cpu = psutil.cpu_percent(interval=0.1)
        cpu_measurements['memory_operations'] = {
            'cpu_usage': memory_cpu - cpu_start,
            'operations': 200,
            'time': memory_time,
            'cpu_efficiency': 200 / (memory_cpu - cpu_start) if memory_cpu > cpu_start else float('inf')
        }
        
        # Test 3: Mixed workload with optimization
        cpu_start = psutil.cpu_percent(interval=None)
        start_time = time.time()
        
        # Enable CPU optimization
        await components['meta_engine'].enable_cpu_optimization()
        
        mixed_tasks = []
        for i in range(150):
            if i % 3 == 0:
                metrics = DeltaMetrics(performance=0.1, efficiency=0.05, stability=0.02, capability=0.08)
                mixed_tasks.append(components['delta_evaluator'].evaluate(metrics))
            elif i % 3 == 1:
                node = MemoryNode(id=f"mixed_{i}", content=f"Mixed content {i}", embedding=[0.1] * 128)
                mixed_tasks.append(components['memory_system'].store_node(node))
            else:
                task = AgentTask(id=f"mixed_task_{i}", type="cpu_test", payload={'iteration': i})
                mixed_tasks.append(components['mcp_orchestrator'].submit_task(task))
        
        await asyncio.gather(*mixed_tasks)
        
        mixed_time = time.time() - start_time
        mixed_cpu = psutil.cpu_percent(interval=0.1)
        cpu_measurements['mixed_optimized'] = {
            'cpu_usage': mixed_cpu - cpu_start,
            'operations': 150,
            'time': mixed_time,
            'cpu_efficiency': 150 / (mixed_cpu - cpu_start) if mixed_cpu > cpu_start else float('inf')
        }
        
        # Verify CPU optimization effectiveness
        delta_efficiency = cpu_measurements['delta_evaluation']['cpu_efficiency']
        memory_efficiency = cpu_measurements['memory_operations']['cpu_efficiency']
        mixed_efficiency = cpu_measurements['mixed_optimized']['cpu_efficiency']
        
        # Mixed optimized workload should be more efficient than individual workloads
        assert mixed_efficiency >= min(delta_efficiency, memory_efficiency), \
            f"Mixed optimized efficiency {mixed_efficiency} not better than individual efficiencies"
        
        # CPU usage should be reasonable
        for workload, measurements in cpu_measurements.items():
            assert measurements['cpu_usage'] <= 90, \
                f"CPU usage too high for {workload}: {measurements['cpu_usage']}%"
        
        # Log CPU utilization results
        context.log_event('cpu_utilization_test', {
            'measurements': cpu_measurements,
            'optimization_effectiveness': mixed_efficiency / max(delta_efficiency, memory_efficiency)
        })
    
    @pytest.mark.asyncio
    async def test_memory_utilization_optimization(self, integrated_system):
        """Test memory utilization optimization."""
        context = integrated_system
        components = context.components
        
        # Phase 1: Baseline memory usage
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        system_memory_usage = await components['memory_system'].get_memory_usage()
        
        # Phase 2: Create memory load
        memory_nodes = []
        for i in range(1000):
            node = MemoryNode(
                id=f"memory_util_test_{i}",
                content=f"Memory utilization test content {i}" * 5,
                embedding=[0.1 * (i % 10)] * 128,
                metadata={'test': 'memory_utilization', 'size': 'medium'}
            )
            memory_nodes.append(node)
            await components['memory_system'].store_node(node)
            
            # Check memory usage periodically
            if i % 100 == 0:
                current_usage = await components['memory_system'].get_memory_usage()
                if current_usage > 0.8:  # High memory usage
                    break
        
        peak_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        peak_system_usage = await components['memory_system'].get_memory_usage()
        
        # Phase 3: Trigger memory optimization
        optimization_result = await components['memory_optimizer'].optimize_memory_usage()
        
        assert optimization_result['success'] is True
        assert optimization_result['nodes_optimized'] > 0
        
        # Phase 4: Measure post-optimization memory
        await asyncio.sleep(1.0)  # Allow optimization to complete
        
        post_opt_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        post_opt_system_usage = await components['memory_system'].get_memory_usage()
        
        # Verify memory optimization
        memory_reduction = peak_memory - post_opt_memory
        system_usage_reduction = peak_system_usage - post_opt_system_usage
        
        assert memory_reduction > 0, "Memory optimization should reduce memory usage"
        assert system_usage_reduction > 0, "System memory usage should be reduced"
        
        # Memory reduction should be significant
        reduction_percentage = memory_reduction / peak_memory
        assert reduction_percentage >= 0.1, f"Memory reduction {reduction_percentage:.2%} not significant enough"
        
        # System should remain functional after optimization
        test_node = MemoryNode(
            id="post_optimization_test",
            content="Test after optimization",
            embedding=[0.1] * 128
        )
        storage_result = await components['memory_system'].store_node(test_node)
        assert storage_result.success is True
        
        # Log memory utilization results
        context.log_event('memory_utilization_test', {
            'initial_memory_mb': initial_memory,
            'peak_memory_mb': peak_memory,
            'post_opt_memory_mb': post_opt_memory,
            'memory_reduction_mb': memory_reduction,
            'reduction_percentage': reduction_percentage,
            'initial_system_usage': system_memory_usage,
            'peak_system_usage': peak_system_usage,
            'post_opt_system_usage': post_opt_system_usage,
            'system_usage_reduction': system_usage_reduction,
            'optimization_result': optimization_result
        })


class TestPerformanceDegradationDetection:
    """Test detection and recovery from performance degradation."""
    
    @pytest.mark.asyncio
    async def test_performance_degradation_detection_and_recovery(self, integrated_system, error_injector):
        """Test system's ability to detect and recover from performance degradation."""
        context = integrated_system
        components = context.components
        
        # Phase 1: Establish baseline performance
        baseline_operations = 50
        baseline_times = []
        
        for i in range(baseline_operations):
            start_time = time.time()
            
            # Mixed operations for baseline
            if i % 2 == 0:
                node = MemoryNode(id=f"baseline_{i}", content=f"Baseline {i}", embedding=[0.1] * 128)
                await components['memory_system'].store_node(node)
            else:
                metrics = DeltaMetrics(performance=0.1, efficiency=0.05, stability=0.02, capability=0.08)
                await components['delta_evaluator'].evaluate(metrics)
            
            baseline_times.append(time.time() - start_time)
        
        baseline_avg_time = statistics.mean(baseline_times)
        baseline_p95 = statistics.quantiles(baseline_times, n=20)[18] if len(baseline_times) > 20 else max(baseline_times)
        
        # Phase 2: Inject performance degradation
        # Inject delays to simulate performance issues
        def slow_operation(*args, **kwargs):
            time.sleep(0.1)  # Add 100ms delay
            return original_store_node(*args, **kwargs)
        
        original_store_node = components['memory_system'].store_node
        components['memory_system'].store_node = slow_operation
        
        # Phase 3: Monitor for degradation detection
        degraded_operations = 30
        degraded_times = []
        
        for i in range(degraded_operations):
            start_time = time.time()
            
            node = MemoryNode(id=f"degraded_{i}", content=f"Degraded {i}", embedding=[0.1] * 128)
            await components['memory_system'].store_node(node)
            
            degraded_times.append(time.time() - start_time)
        
        degraded_avg_time = statistics.mean(degraded_times)
        
        # Phase 4: Verify degradation was detected
        performance_alerts = await components['meta_engine'].get_performance_alerts()
        degradation_alerts = [alert for alert in performance_alerts if 'degradation' in alert.get('type', '').lower()]
        
        assert len(degradation_alerts) > 0, "Performance degradation should have been detected"
        
        # Verify degradation metrics
        degradation_factor = degraded_avg_time / baseline_avg_time
        assert degradation_factor > 2, f"Degradation factor {degradation_factor} should be significant"
        
        # Phase 5: Test automatic recovery
        recovery_actions = await components['meta_engine'].get_recovery_actions()
        performance_recoveries = [action for action in recovery_actions if 'performance' in action.get('type', '').lower()]
        
        assert len(performance_recoveries) > 0, "Recovery actions should have been initiated"
        
        # Restore original function (simulating recovery)
        components['memory_system'].store_node = original_store_node
        
        # Phase 6: Verify recovery effectiveness
        recovery_operations = 30
        recovery_times = []
        
        for i in range(recovery_operations):
            start_time = time.time()
            
            node = MemoryNode(id=f"recovery_{i}", content=f"Recovery {i}", embedding=[0.1] * 128)
            await components['memory_system'].store_node(node)
            
            recovery_times.append(time.time() - start_time)
        
        recovery_avg_time = statistics.mean(recovery_times)
        
        # Recovery should bring performance back close to baseline
        recovery_improvement = degraded_avg_time / recovery_avg_time
        assert recovery_improvement > 2, f"Recovery improvement {recovery_improvement} not sufficient"
        
        # Performance should be within 50% of baseline after recovery
        performance_ratio = recovery_avg_time / baseline_avg_time
        assert performance_ratio <= 1.5, f"Performance ratio {performance_ratio} still too degraded after recovery"
        
        # Phase 7: Verify learning from degradation incident
        incident_learning = await components['meta_engine'].get_incident_learning()
        performance_incidents = [incident for incident in incident_learning if 'performance' in incident.get('type', '').lower()]
        
        assert len(performance_incidents) > 0, "System should have learned from performance incident"
        
        # Log degradation detection and recovery results
        context.log_event('performance_degradation_test', {
            'baseline_avg_time': baseline_avg_time,
            'baseline_p95': baseline_p95,
            'degraded_avg_time': degraded_avg_time,
            'recovery_avg_time': recovery_avg_time,
            'degradation_factor': degradation_factor,
            'recovery_improvement': recovery_improvement,
            'final_performance_ratio': performance_ratio,
            'alerts_triggered': len(degradation_alerts),
            'recovery_actions': len(performance_recoveries),
            'incidents_learned': len(performance_incidents)
        })


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-m", "not slow"])