"""
Simplified unit tests for the SAFLA benchmark framework.

This module contains basic tests for the benchmark components to verify
the framework is working correctly.
"""

import pytest
import asyncio
import tempfile
import os
import json
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta

# Import benchmark modules
from benchmarks import (
    Benchmark, BenchmarkRunner, BenchmarkSuite, BenchmarkMetrics, 
    BenchmarkResult, BenchmarkDatabase, BenchmarkConfig
)
from benchmarks.cli_benchmarks import (
    CLICommandBenchmark, CLIHelpBenchmark, CLIValidateBenchmark,
    CLIInfoBenchmark, CLIStatusBenchmark, CLIConfigInitBenchmark,
    CLIBatchCommandBenchmark, CLIMemoryUsageBenchmark,
    CLIBenchmarkSuite
)
from benchmarks.utils import PerformanceMetrics, BenchmarkAnalyzer


class SimpleBenchmark(Benchmark):
    """Simple benchmark implementation for testing purposes"""
    
    def __init__(self, name="simple_benchmark", should_fail=False, execution_time=0.1):
        super().__init__(name, "Simple benchmark for unit testing")
        self.should_fail = should_fail
        self.execution_time = execution_time
        self.setup_called = False
        self.teardown_called = False
    
    async def setup(self):
        """Setup test benchmark"""
        self.setup_called = True
        await asyncio.sleep(0.01)  # Simulate setup time
    
    async def run(self) -> dict:
        """Run test benchmark"""
        await asyncio.sleep(self.execution_time)
        
        if self.should_fail:
            raise Exception("Simple benchmark failure")
        
        return {
            "operations_completed": 100,
            "data_processed": 1024,
            "custom_metric": 42.5
        }
    
    async def teardown(self):
        """Teardown test benchmark"""
        self.teardown_called = True
        await asyncio.sleep(0.01)  # Simulate teardown time


class TestBenchmarkMetrics:
    """Test BenchmarkMetrics dataclass"""
    
    def test_benchmark_metrics_creation(self):
        """Test creating BenchmarkMetrics with all fields"""
        metrics = BenchmarkMetrics(
            execution_time=1.5,
            memory_usage=128.0,
            cpu_usage=75.5,
            throughput=100.0,
            latency=50.0,
            error_rate=0.1,
            custom_metrics={"test": "value"}
        )
        
        assert metrics.execution_time == 1.5
        assert metrics.memory_usage == 128.0
        assert metrics.cpu_usage == 75.5
        assert metrics.throughput == 100.0
        assert metrics.latency == 50.0
        assert metrics.error_rate == 0.1
        assert metrics.custom_metrics["test"] == "value"
    
    def test_benchmark_metrics_defaults(self):
        """Test BenchmarkMetrics with default values"""
        metrics = BenchmarkMetrics(execution_time=1.0, memory_usage=64.0, cpu_usage=50.0)
        
        assert metrics.execution_time == 1.0
        assert metrics.memory_usage == 64.0
        assert metrics.cpu_usage == 50.0
        assert metrics.throughput is None
        assert metrics.latency is None
        assert metrics.error_rate is None
        assert metrics.custom_metrics == {}


class TestBenchmarkResult:
    """Test BenchmarkResult dataclass"""
    
    def test_benchmark_result_success(self):
        """Test successful benchmark result"""
        metrics = BenchmarkMetrics(execution_time=1.0, memory_usage=64.0, cpu_usage=50.0)
        result = BenchmarkResult(
            benchmark_name="test",
            timestamp=datetime.now(),
            metrics=metrics,
            success=True
        )
        
        assert result.benchmark_name == "test"
        assert result.success is True
        assert result.metrics.execution_time == 1.0
        assert result.error_message is None
    
    def test_benchmark_result_to_dict(self):
        """Test converting BenchmarkResult to dictionary"""
        metrics = BenchmarkMetrics(execution_time=1.0, memory_usage=64.0, cpu_usage=50.0)
        timestamp = datetime.now()
        result = BenchmarkResult(
            benchmark_name="test",
            timestamp=timestamp,
            metrics=metrics,
            success=True
        )
        
        result_dict = result.to_dict()
        
        assert result_dict["benchmark_name"] == "test"
        assert result_dict["success"] is True
        assert result_dict["execution_time"] == 1.0
        assert result_dict["memory_usage"] == 64.0
        assert result_dict["timestamp"] == timestamp.isoformat()


class TestBenchmarkRunner:
    """Test BenchmarkRunner functionality"""
    
    @pytest.mark.asyncio
    async def test_run_successful_benchmark(self):
        """Test running a successful benchmark"""
        benchmark = SimpleBenchmark("success_test", should_fail=False, execution_time=0.1)
        runner = BenchmarkRunner()
        
        result = await runner.run_benchmark(benchmark)
        
        assert result.success is True
        assert result.benchmark_name == "success_test"
        assert result.metrics.execution_time >= 0.1
        assert result.metrics.custom_metrics["operations_completed"] == 100
        assert benchmark.setup_called is True
        assert benchmark.teardown_called is True
    
    @pytest.mark.asyncio
    async def test_run_failing_benchmark(self):
        """Test running a failing benchmark"""
        benchmark = SimpleBenchmark("failure_test", should_fail=True)
        runner = BenchmarkRunner()
        
        result = await runner.run_benchmark(benchmark)
        
        assert result.success is False
        assert result.benchmark_name == "failure_test"
        assert "Simple benchmark failure" in result.error_message


class TestBenchmarkSuite:
    """Test BenchmarkSuite functionality"""
    
    def test_suite_creation(self):
        """Test creating a benchmark suite"""
        suite = BenchmarkSuite("Test Suite", "Test description")
        
        assert suite.name == "Test Suite"
        assert suite.description == "Test description"
        assert len(suite.benchmarks) == 0
    
    def test_add_benchmark(self):
        """Test adding benchmarks to suite"""
        suite = BenchmarkSuite("Test Suite", "Test description")
        benchmark = SimpleBenchmark("test1")
        
        suite.add_benchmark(benchmark)
        
        assert len(suite.benchmarks) == 1
        assert suite.benchmarks[0].name == "test1"
    
    @pytest.mark.asyncio
    async def test_run_all_sequential(self):
        """Test running all benchmarks sequentially"""
        suite = BenchmarkSuite("Test Suite", "Test description")
        suite.add_benchmark(SimpleBenchmark("test1", execution_time=0.1))
        suite.add_benchmark(SimpleBenchmark("test2", execution_time=0.1))
        
        results = await suite.run_all(parallel=False)
        
        assert len(results) == 2
        assert all(result.success for result in results)


class TestBenchmarkConfig:
    """Test BenchmarkConfig functionality"""
    
    def test_config_defaults(self):
        """Test default configuration values"""
        config = BenchmarkConfig()
        
        assert config.database_path == "benchmarks/results.db"
        assert config.parallel_execution is False
        assert config.store_results is True
        assert config.timeout_seconds == 30.0
    
    def test_config_creation(self):
        """Test creating config with custom values"""
        config = BenchmarkConfig(
            database_path="custom.db",
            parallel_execution=True,
            timeout_seconds=600.0,
            warmup_iterations=2
        )
        
        assert config.database_path == "custom.db"
        assert config.parallel_execution is True
        assert config.timeout_seconds == 600.0
        assert config.warmup_iterations == 2


class TestPerformanceMetrics:
    """Test PerformanceMetrics utility class"""
    
    def test_calculate_statistics(self):
        """Test statistical calculations"""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        stats = PerformanceMetrics.calculate_statistics(data)
        
        assert stats["mean"] == 3.0
        assert stats["median"] == 3.0
        assert stats["min"] == 1.0
        assert stats["max"] == 5.0
        assert abs(stats["stdev"] - 1.5811) < 0.001  # Standard deviation
        assert "p95" in stats
        assert "p99" in stats
    
    def test_calculate_improvement(self):
        """Test improvement calculation"""
        # Test latency improvement (lower is better)
        improvement = PerformanceMetrics.calculate_improvement(2.0, 1.0, "latency")
        assert improvement["improvement_ratio"] == 2.0
        assert improvement["improvement_percent"] == 50.0
        
        # Test throughput improvement (higher is better)
        improvement = PerformanceMetrics.calculate_improvement(100.0, 150.0, "throughput")
        assert improvement["improvement_ratio"] == 1.5
        assert improvement["improvement_percent"] == 50.0
    
    def test_detect_regression(self):
        """Test regression detection"""
        # Stable performance
        stable_data = [1.0, 1.1, 0.9, 1.0, 1.05]
        result = PerformanceMetrics.detect_performance_regression(stable_data, 1.0)
        assert not result['regression_detected']
        
        # Clear regression (increasing times)
        regression_data = [1.0, 1.5, 2.0, 2.5]
        result = PerformanceMetrics.detect_performance_regression(regression_data, 3.0)
        assert result['regression_detected']


class TestCLIBenchmarks:
    """Test CLI-specific benchmark implementations"""
    
    def test_cli_benchmark_suite(self):
        """Test CLI benchmark suite creation"""
        suite = CLIBenchmarkSuite()
        
        assert len(suite.benchmarks) > 0
        
        # Check that expected benchmarks are included
        benchmark_names = [b.name for b in suite.benchmarks]
        expected_benchmarks = [
            "cli_help_command",
            "cli_info_command",
            "cli_status_command",
            "cli_config_init_command",
            "cli_validate_command",
            "cli_batch_basic_commands",
            "cli_info_memory_usage",
            "cli_validate_memory_usage"
        ]
        
        for expected in expected_benchmarks:
            assert expected in benchmark_names


class TestBenchmarkDatabase:
    """Test BenchmarkDatabase functionality"""
    
    def setup_method(self):
        """Setup test database"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test.db")
        self.db = BenchmarkDatabase(self.db_path)
    
    def teardown_method(self):
        """Cleanup test database"""
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)
        os.rmdir(self.temp_dir)
    
    def test_store_result(self):
        """Test storing a single benchmark result"""
        metrics = BenchmarkMetrics(execution_time=1.5, memory_usage=128.0, cpu_usage=75.0)
        result = BenchmarkResult(
            benchmark_name="test_benchmark",
            timestamp=datetime.now(),
            metrics=metrics,
            success=True
        )
        
        result_id = self.db.store_result(result)
        assert result_id > 0
    
    def test_get_latest_results(self):
        """Test retrieving latest results for a benchmark"""
        # Store some test results
        for i in range(3):
            metrics = BenchmarkMetrics(
                execution_time=1.0 + i * 0.1, 
                memory_usage=64.0, 
                cpu_usage=50.0
            )
            result = BenchmarkResult(
                benchmark_name="latest_test",
                timestamp=datetime.now(),
                metrics=metrics,
                success=True
            )
            self.db.store_result(result)
        
        results = self.db.get_latest_results("latest_test", limit=2)
        assert len(results) == 2


if __name__ == "__main__":
    pytest.main([__file__])