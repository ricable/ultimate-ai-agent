"""
Core Benchmarking Framework
===========================

This module provides the core benchmarking infrastructure for SAFLA,
including benchmark definitions, execution, and result management.
"""

import time
import statistics
import psutil
import asyncio
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime
from abc import ABC, abstractmethod
import logging
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkMetrics:
    """Performance metrics for a benchmark."""
    execution_time: float  # seconds
    memory_usage: float    # MB
    cpu_usage: float      # percentage
    throughput: Optional[float] = None  # operations per second
    latency: Optional[float] = None     # milliseconds
    error_rate: Optional[float] = None  # percentage
    custom_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Result of a benchmark execution."""
    benchmark_name: str
    timestamp: datetime
    metrics: BenchmarkMetrics
    success: bool
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for storage."""
        return {
            'benchmark_name': self.benchmark_name,
            'timestamp': self.timestamp.isoformat(),
            'execution_time': self.metrics.execution_time,
            'memory_usage': self.metrics.memory_usage,
            'cpu_usage': self.metrics.cpu_usage,
            'throughput': self.metrics.throughput,
            'latency': self.metrics.latency,
            'error_rate': self.metrics.error_rate,
            'custom_metrics': self.metrics.custom_metrics,
            'success': self.success,
            'error_message': self.error_message,
            'metadata': self.metadata
        }


class Benchmark(ABC):
    """Abstract base class for benchmarks."""
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.setup_completed = False
        
    @abstractmethod
    async def setup(self) -> None:
        """Setup benchmark prerequisites."""
        pass
    
    @abstractmethod
    async def run(self) -> Dict[str, Any]:
        """Execute the benchmark and return results."""
        pass
    
    @abstractmethod
    async def teardown(self) -> None:
        """Cleanup after benchmark execution."""
        pass
    
    def validate_result(self, result: Dict[str, Any]) -> bool:
        """Validate benchmark result."""
        return True


class BenchmarkRunner:
    """Executes benchmarks and collects performance metrics."""
    
    def __init__(self, warmup_iterations: int = 3, measurement_iterations: int = 10):
        self.warmup_iterations = warmup_iterations
        self.measurement_iterations = measurement_iterations
        self.process = psutil.Process()
        
    async def run_benchmark(self, benchmark: Benchmark) -> BenchmarkResult:
        """Run a single benchmark and collect metrics."""
        logger.info(f"Starting benchmark: {benchmark.name}")
        
        try:
            # Setup
            await benchmark.setup()
            benchmark.setup_completed = True
            
            # Warmup
            logger.debug(f"Warming up benchmark: {benchmark.name}")
            for _ in range(self.warmup_iterations):
                await benchmark.run()
            
            # Measure performance
            logger.debug(f"Measuring benchmark: {benchmark.name}")
            execution_times = []
            memory_usages = []
            cpu_usages = []
            results = []
            
            for i in range(self.measurement_iterations):
                # Collect initial metrics
                initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB
                initial_cpu = self.process.cpu_percent()
                
                # Execute benchmark
                start_time = time.perf_counter()
                result = await benchmark.run()
                end_time = time.perf_counter()
                
                # Collect final metrics
                final_memory = self.process.memory_info().rss / 1024 / 1024  # MB
                final_cpu = self.process.cpu_percent()
                
                # Calculate metrics
                execution_time = end_time - start_time
                memory_usage = max(final_memory - initial_memory, 0)
                cpu_usage = max(final_cpu - initial_cpu, 0)
                
                execution_times.append(execution_time)
                memory_usages.append(memory_usage)
                cpu_usages.append(cpu_usage)
                results.append(result)
                
                # Validate result
                if not benchmark.validate_result(result):
                    raise ValueError(f"Benchmark result validation failed for iteration {i}")
            
            # Calculate aggregate metrics
            avg_execution_time = statistics.mean(execution_times)
            avg_memory_usage = statistics.mean(memory_usages)
            avg_cpu_usage = statistics.mean(cpu_usages)
            
            # Calculate derived metrics
            throughput = None
            latency = None
            error_rate = 0.0
            
            # Extract custom metrics from results
            custom_metrics = {}
            if results and isinstance(results[0], dict):
                # Aggregate custom metrics
                for key in results[0].keys():
                    if key not in ['throughput', 'latency', 'error_rate']:
                        values = [r.get(key) for r in results if key in r and isinstance(r[key], (int, float))]
                        if values:
                            custom_metrics[key] = statistics.mean(values)
                
                # Extract standard metrics
                throughput_values = [r.get('throughput') for r in results if 'throughput' in r]
                if throughput_values:
                    throughput = statistics.mean(throughput_values)
                
                latency_values = [r.get('latency') for r in results if 'latency' in r]
                if latency_values:
                    latency = statistics.mean(latency_values)
                
                error_values = [r.get('error_rate', 0) for r in results if 'error_rate' in r]
                if error_values:
                    error_rate = statistics.mean(error_values)
            
            # Create metrics object
            metrics = BenchmarkMetrics(
                execution_time=avg_execution_time,
                memory_usage=avg_memory_usage,
                cpu_usage=avg_cpu_usage,
                throughput=throughput,
                latency=latency,
                error_rate=error_rate,
                custom_metrics=custom_metrics
            )
            
            # Create result
            result = BenchmarkResult(
                benchmark_name=benchmark.name,
                timestamp=datetime.now(),
                metrics=metrics,
                success=True,
                metadata={
                    'description': benchmark.description,
                    'warmup_iterations': self.warmup_iterations,
                    'measurement_iterations': self.measurement_iterations,
                    'execution_times': execution_times,
                    'memory_usages': memory_usages,
                    'cpu_usages': cpu_usages
                }
            )
            
            logger.info(f"Benchmark completed: {benchmark.name} - {avg_execution_time:.3f}s")
            return result
            
        except Exception as e:
            logger.error(f"Benchmark failed: {benchmark.name} - {str(e)}")
            logger.debug(traceback.format_exc())
            
            # Create failure result
            return BenchmarkResult(
                benchmark_name=benchmark.name,
                timestamp=datetime.now(),
                metrics=BenchmarkMetrics(
                    execution_time=0.0,
                    memory_usage=0.0,
                    cpu_usage=0.0
                ),
                success=False,
                error_message=str(e),
                metadata={'description': benchmark.description}
            )
        
        finally:
            # Cleanup
            if benchmark.setup_completed:
                try:
                    await benchmark.teardown()
                except Exception as e:
                    logger.warning(f"Benchmark teardown failed: {benchmark.name} - {str(e)}")


class BenchmarkSuite:
    """Collection of benchmarks that can be executed together."""
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.benchmarks: List[Benchmark] = []
        self.runner = BenchmarkRunner()
        
    def add_benchmark(self, benchmark: Benchmark) -> None:
        """Add a benchmark to the suite."""
        self.benchmarks.append(benchmark)
        
    def remove_benchmark(self, benchmark_name: str) -> bool:
        """Remove a benchmark from the suite."""
        for i, benchmark in enumerate(self.benchmarks):
            if benchmark.name == benchmark_name:
                del self.benchmarks[i]
                return True
        return False
        
    async def run_all(self, parallel: bool = False) -> List[BenchmarkResult]:
        """Run all benchmarks in the suite."""
        logger.info(f"Running benchmark suite: {self.name} ({len(self.benchmarks)} benchmarks)")
        
        if parallel:
            return await self._run_parallel()
        else:
            return await self._run_sequential()
    
    async def _run_sequential(self) -> List[BenchmarkResult]:
        """Run benchmarks sequentially."""
        results = []
        for benchmark in self.benchmarks:
            result = await self.runner.run_benchmark(benchmark)
            results.append(result)
        return results
    
    async def _run_parallel(self) -> List[BenchmarkResult]:
        """Run benchmarks in parallel."""
        tasks = []
        for benchmark in self.benchmarks:
            task = asyncio.create_task(self.runner.run_benchmark(benchmark))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Benchmark failed: {self.benchmarks[i].name} - {str(result)}")
                final_results.append(BenchmarkResult(
                    benchmark_name=self.benchmarks[i].name,
                    timestamp=datetime.now(),
                    metrics=BenchmarkMetrics(
                        execution_time=0.0,
                        memory_usage=0.0,
                        cpu_usage=0.0
                    ),
                    success=False,
                    error_message=str(result)
                ))
            else:
                final_results.append(result)
        
        return final_results
    
    async def run_specific(self, benchmark_names: List[str]) -> List[BenchmarkResult]:
        """Run specific benchmarks by name."""
        results = []
        for name in benchmark_names:
            benchmark = next((b for b in self.benchmarks if b.name == name), None)
            if benchmark:
                result = await self.runner.run_benchmark(benchmark)
                results.append(result)
            else:
                logger.warning(f"Benchmark not found: {name}")
        return results