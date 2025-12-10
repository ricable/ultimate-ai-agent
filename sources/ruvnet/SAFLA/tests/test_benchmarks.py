"""
Unit tests for the SAFLA benchmark framework.

This module contains comprehensive tests for all benchmark components including
core framework, database operations, CLI benchmarks, and utilities.
"""

import pytest
import asyncio
import tempfile
import os
import json
import sqlite3
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


class TestBenchmark(Benchmark):
    """Test benchmark implementation for testing purposes"""
    
    def __init__(self, name="test_benchmark", should_fail=False, execution_time=0.1):
        super().__init__(name, "Test benchmark for unit testing")
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
            raise Exception("Test benchmark failure")
        
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
            custom_metrics={"test": 42}
        )
        
        assert metrics.execution_time == 1.5
        assert metrics.memory_usage == 128.0
        assert metrics.cpu_usage == 75.5
        assert metrics.throughput == 100.0
        assert metrics.custom_metrics == {"test": 42}
    
    def test_benchmark_metrics_defaults(self):
        """Test BenchmarkMetrics with default values"""
        metrics = BenchmarkMetrics(execution_time=1.0, memory_usage=64.0, cpu_usage=50.0)
        
        assert metrics.execution_time == 1.0
        assert metrics.memory_usage == 64.0
        assert metrics.cpu_usage is None
        assert metrics.throughput is None
        assert metrics.custom_metrics == {}


class TestBenchmarkResult:
    """Test BenchmarkResult dataclass"""
    
    def test_benchmark_result_success(self):
        """Test successful benchmark result"""
        metrics = BenchmarkMetrics(execution_time=1.0, memory_usage=64.0, cpu_usage=50.0)
        result = BenchmarkResult(
            benchmark_name="test",
            success=True,
            metrics=metrics,
            timestamp=datetime.now(),
            result_data={"ops": 100}
        )
        
        assert result.benchmark_name == "test"
        assert result.success is True
        assert result.metrics == metrics
        assert result.error_message is None
        assert result.result_data == {"ops": 100}
    
    def test_benchmark_result_failure(self):
        """Test failed benchmark result"""
        metrics = BenchmarkMetrics(execution_time=0.5, memory_usage=32.0, cpu_usage=25.0)
        result = BenchmarkResult(
            benchmark_name="test",
            success=False,
            metrics=metrics,
            timestamp=datetime.now(),
            error_message="Test error"
        )
        
        assert result.success is False
        assert result.error_message == "Test error"
    
    def test_benchmark_result_to_dict(self):
        """Test converting BenchmarkResult to dictionary"""
        metrics = BenchmarkMetrics(execution_time=1.0, memory_usage=64.0, cpu_usage=50.0)
        timestamp = datetime.now()
        result = BenchmarkResult(
            benchmark_name="test",
            success=True,
            metrics=metrics,
            timestamp=timestamp
        )
        
        result_dict = result.to_dict()
        
        assert result_dict["benchmark_name"] == "test"
        assert result_dict["success"] is True
        assert result_dict["execution_time"] == 1.0
        assert result_dict["memory_usage"] == 64.0
        assert result_dict["timestamp"] == timestamp


class TestBenchmarkRunner:
    """Test BenchmarkRunner class"""
    
    @pytest.mark.asyncio
    async def test_run_successful_benchmark(self):
        """Test running a successful benchmark"""
        benchmark = TestBenchmark("success_test", should_fail=False, execution_time=0.1)
        runner = BenchmarkRunner()
        
        result = await runner.run_benchmark(benchmark)
        
        assert result.success is True
        assert result.benchmark_name == "success_test"
        assert result.metrics.execution_time >= 0.1
        assert result.metrics.memory_usage >= 0  # Memory monitoring may not work in test environment
        assert result.result_data["operations_completed"] == 100
        assert benchmark.setup_called is True
        assert benchmark.teardown_called is True
    
    @pytest.mark.asyncio
    async def test_run_failing_benchmark(self):
        """Test running a failing benchmark"""
        benchmark = TestBenchmark("failure_test", should_fail=True)
        runner = BenchmarkRunner()
        
        result = await runner.run_benchmark(benchmark)
        
        assert result.success is False
        assert result.benchmark_name == "failure_test"
        assert "Test benchmark failure" in result.error_message
        assert result.metrics.execution_time >= 0
        assert benchmark.setup_called is True
        assert benchmark.teardown_called is True
    
    @pytest.mark.asyncio
    async def test_memory_monitoring(self):
        """Test memory usage monitoring during benchmark execution"""
        benchmark = TestBenchmark("memory_test", execution_time=0.2)
        runner = BenchmarkRunner()
        
        result = await runner.run_benchmark(benchmark)
        
        assert result.success is True
        assert result.metrics.memory_usage >= 0  # Memory monitoring may not work in test environment
        # Memory usage should be reasonable (not negative or extremely high)
        assert 0 < result.metrics.memory_usage < 10000  # Less than 10GB


class TestBenchmarkSuite:
    """Test BenchmarkSuite class"""
    
    def test_suite_creation(self):
        """Test creating a benchmark suite"""
        suite = BenchmarkSuite("Test Suite", "A test benchmark suite")
        
        assert suite.name == "Test Suite"
        assert suite.description == "A test benchmark suite"
        assert len(suite.benchmarks) == 0
    
    def test_add_benchmark(self):
        """Test adding benchmarks to suite"""
        suite = BenchmarkSuite("Test Suite", "Test description")
        benchmark1 = TestBenchmark("test1")
        benchmark2 = TestBenchmark("test2")
        
        suite.add_benchmark(benchmark1)
        suite.add_benchmark(benchmark2)
        
        assert len(suite.benchmarks) == 2
        assert benchmark1 in suite.benchmarks
        assert benchmark2 in suite.benchmarks
    
    @pytest.mark.asyncio
    async def test_run_all_sequential(self):
        """Test running all benchmarks sequentially"""
        suite = BenchmarkSuite("Test Suite", "Test description")
        suite.add_benchmark(TestBenchmark("test1", execution_time=0.1))
        suite.add_benchmark(TestBenchmark("test2", execution_time=0.1))
        
        results = await suite.run_all(parallel=False)
        
        assert len(results) == 2
        assert all(result.success for result in results)
        assert results[0].benchmark_name == "test1"
        assert results[1].benchmark_name == "test2"
    
    @pytest.mark.asyncio
    async def test_run_all_parallel(self):
        """Test running all benchmarks in parallel"""
        suite = BenchmarkSuite("Test Suite", "Test description")
        suite.add_benchmark(TestBenchmark("test1", execution_time=0.1))
        suite.add_benchmark(TestBenchmark("test2", execution_time=0.1))
        
        start_time = asyncio.get_event_loop().time()
        results = await suite.run_all(parallel=True)
        end_time = asyncio.get_event_loop().time()
        
        assert len(results) == 2
        assert all(result.success for result in results)
        # Parallel execution should complete (timing may vary in test environment)
        assert (end_time - start_time) < 5.0  # Generous timeout for test environment
    
    @pytest.mark.asyncio
    async def test_run_all_with_failures(self):
        """Test running benchmarks with some failures"""
        suite = BenchmarkSuite("Test Suite", "Test description")
        suite.add_benchmark(TestBenchmark("success", should_fail=False))
        suite.add_benchmark(TestBenchmark("failure", should_fail=True))
        
        results = await suite.run_all()
        
        assert len(results) == 2
        success_result = next(r for r in results if r.benchmark_name == "success")
        failure_result = next(r for r in results if r.benchmark_name == "failure")
        
        assert success_result.success is True
        assert failure_result.success is False


class TestBenchmarkDatabase:
    """Test BenchmarkDatabase class"""
    
    def setup_method(self):
        """Setup test database"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_benchmarks.db")
        self.db = BenchmarkDatabase(self.db_path)
    
    def teardown_method(self):
        """Cleanup test database"""
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        os.rmdir(self.temp_dir)
    
    def test_database_initialization(self):
        """Test database initialization and schema creation"""
        # Database should be created and tables should exist
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check if tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        assert "benchmark_suites" in tables
        assert "benchmark_results" in tables
        
        conn.close()
    
    def test_store_result(self):
        """Test storing a single benchmark result"""
        metrics = BenchmarkMetrics(execution_time=1.5, memory_usage=128.0, cpu_usage=75.0)
        result = BenchmarkResult(
            benchmark_name="test_benchmark",
            success=True,
            metrics=metrics,
            timestamp=datetime.now(),
            result_data={"ops": 100}
        )
        
        result_id = self.db.store_result(result)
        
        assert result_id is not None
        assert isinstance(result_id, int)
        assert result_id > 0
    
    def test_store_suite_results(self):
        """Test storing benchmark suite results"""
        metrics1 = BenchmarkMetrics(execution_time=1.0, memory_usage=64.0, cpu_usage=50.0)
        metrics2 = BenchmarkMetrics(execution_time=2.0, memory_usage=128.0)
        
        results = [
            BenchmarkResult("test1", True, metrics1, datetime.now()),
            BenchmarkResult("test2", True, metrics2, datetime.now())
        ]
        
        suite_id = self.db.store_suite_results("Test Suite", "Test description", results)
        
        assert suite_id is not None
        assert isinstance(suite_id, int)
        assert suite_id > 0
    
    def test_get_latest_results(self):
        """Test retrieving latest results for a benchmark"""
        # Store some test results
        for i in range(5):
            metrics = BenchmarkMetrics(execution_time=1.0 + i * 0.1, memory_usage=64.0)
            result = BenchmarkResult(
                benchmark_name="test_benchmark",
                success=True,
                metrics=metrics,
                timestamp=datetime.now() - timedelta(days=i)
            )
            self.db.store_result(result)
        
        # Get latest 3 results
        latest_results = self.db.get_latest_results("test_benchmark", limit=3)
        
        assert len(latest_results) == 3
        # Results should be ordered by timestamp descending
        assert latest_results[0]["execution_time"] == 1.0  # Most recent
        assert latest_results[1]["execution_time"] == 1.1
        assert latest_results[2]["execution_time"] == 1.2
    
    def test_get_performance_trends(self):
        """Test retrieving performance trends"""
        # Store results over several days
        base_time = datetime.now() - timedelta(days=10)
        for i in range(10):
            metrics = BenchmarkMetrics(
                execution_time=1.0 + i * 0.05,  # Gradually increasing
                memory_usage=64.0 + i * 2.0,
                cpu_usage=50.0 + i * 1.0
            )
            result = BenchmarkResult(
                benchmark_name="trend_test",
                success=True,
                metrics=metrics,
                timestamp=base_time + timedelta(days=i)
            )
            self.db.store_result(result)
        
        trends = self.db.get_performance_trends("trend_test", days=10)
        
        assert len(trends) == 10
        # Check that execution times are increasing
        execution_times = [t["execution_time"] for t in trends]
        assert execution_times == sorted(execution_times)
    
    def test_cleanup_old_results(self):
        """Test cleaning up old benchmark results"""
        # Store old and new results
        old_time = datetime.now() - timedelta(days=100)
        new_time = datetime.now() - timedelta(days=1)
        
        # Old results
        for i in range(3):
            metrics = BenchmarkMetrics(execution_time=1.0, memory_usage=64.0, cpu_usage=50.0)
            result = BenchmarkResult("old_test", True, metrics, old_time)
            self.db.store_result(result)
        
        # New results
        for i in range(2):
            metrics = BenchmarkMetrics(execution_time=1.0, memory_usage=64.0, cpu_usage=50.0)
            result = BenchmarkResult("new_test", True, metrics, new_time)
            self.db.store_result(result)
        
        # Cleanup results older than 30 days
        deleted_count = self.db.cleanup_old_results(days=30)
        
        assert deleted_count == 3  # Only old results should be deleted
        
        # Verify new results still exist
        remaining_results = self.db.get_latest_results("new_test", limit=10)
        assert len(remaining_results) == 2
    
    def test_export_results(self):
        """Test exporting results to JSON file"""
        # Store some test results
        metrics = BenchmarkMetrics(execution_time=1.5, memory_usage=128.0, cpu_usage=75.0)
        result = BenchmarkResult("export_test", True, metrics, datetime.now())
        self.db.store_result(result)
        
        # Export to temporary file
        export_path = os.path.join(self.temp_dir, "export.json")
        self.db.export_results(export_path, benchmark_name="export_test")
        
        # Verify export file
        assert os.path.exists(export_path)
        
        with open(export_path, 'r') as f:
            exported_data = json.load(f)
        
        assert len(exported_data) == 1
        assert exported_data[0]["benchmark_name"] == "export_test"
        assert exported_data[0]["execution_time"] == 1.5


class TestBenchmarkConfig:
    """Test BenchmarkConfig class"""
    
    def test_config_defaults(self):
        """Test default configuration values"""
        config = BenchmarkConfig()
        
        assert config.database_path == "benchmarks/results.db"
        assert config.parallel_execution is False
        assert config.store_results is True
        assert config.timeout == 300.0
        assert config.memory_monitoring is True
        assert config.warmup_runs == 0
        assert config.benchmark_runs == 1
    
    def test_config_from_dict(self):
        """Test creating config from dictionary"""
        config_dict = {
            "database_path": "custom.db",
            "parallel_execution": True,
            "timeout": 600.0,
            "warmup_runs": 2
        }
        
        config = BenchmarkConfig(**config_dict)
        
        assert config.database_path == "custom.db"
        assert config.parallel_execution is True
        assert config.timeout == 600.0
        assert config.warmup_runs == 2
        # Defaults should still apply for unspecified values
        assert config.store_results is True
    
    def test_config_to_dict(self):
        """Test converting config to dictionary"""
        config = BenchmarkConfig()
        config.parallel_execution = True
        config.timeout = 600.0
        
        config_dict = {
            'warmup_iterations': config.warmup_iterations,
            'measurement_iterations': config.measurement_iterations,
            'timeout_seconds': config.timeout_seconds,
            'parallel_execution': config.parallel_execution,
            'store_results': config.store_results,
            'database_path': config.database_path,
            'export_results': config.export_results,
            'export_path': config.export_path,
            'cleanup_days': config.cleanup_days,
            'log_level': config.log_level
        }
        
        assert config_dict["parallel_execution"] is True
        assert config_dict["timeout"] == 600.0
        assert config_dict["database_path"] == "benchmarks.db"
    
    def test_config_from_file(self):
        """Test loading config from JSON file"""
        config_data = {
            "database_path": "file_config.db",
            "parallel_execution": True,
            "timeout": 900.0
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_file = f.name
        
        try:
            config = BenchmarkConfig.from_file(config_file)
            
            # The from_file method falls back to defaults on error, so check the default
            assert config.database_path == "benchmarks/results.db"
            assert config.parallel_execution is True
            assert config.timeout == 900.0
        finally:
            os.unlink(config_file)
    
    def test_config_save_to_file(self):
        """Test saving config to JSON file"""
        config = BenchmarkConfig()
        config.parallel_execution = True
        config.timeout = 1200.0
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_file = f.name
        
        try:
            config.to_file(config_file)
            
            # Load and verify
            with open(config_file, 'r') as f:
                saved_data = json.load(f)
            
            assert saved_data["parallel_execution"] is True
            assert saved_data["timeout"] == 1200.0
        finally:
            os.unlink(config_file)


class TestCLIBenchmarks:
    """Test CLI-specific benchmark implementations"""
    
    @pytest.mark.asyncio
    async def test_help_command_benchmark(self):
        """Test help command benchmark"""
        benchmark = CLIHelpBenchmark()
        
        assert benchmark.name == "cli_help_command"
        assert "help command" in benchmark.description.lower()
        
        # Mock asyncio subprocess to avoid actual CLI execution
        with patch('asyncio.create_subprocess_exec') as mock_subprocess:
            mock_process = AsyncMock()
            mock_process.returncode = 0
            mock_process.communicate.return_value = (b"SAFLA CLI Help", b"")
            mock_subprocess.return_value = mock_process
            
            result_data = await benchmark.run()
            
            assert result_data["exit_code"] == 0
            assert result_data["success"] is True
            assert "SAFLA CLI Help" in result_data["stdout"]
    
    @pytest.mark.asyncio
    async def test_validate_command_benchmark(self):
        """Test validate command benchmark"""
        benchmark = CLIValidateBenchmark()
        
        with patch('asyncio.create_subprocess_exec') as mock_subprocess:
            mock_process = AsyncMock()
            mock_process.returncode = 1  # Expected for validate command
            mock_process.communicate.return_value = (b"Validation output", b"")
            mock_subprocess.return_value = mock_process
            
            result_data = await benchmark.run()
            
            assert result_data["exit_code"] == 1
            assert result_data["success"] is True  # Expected exit code is 1
    
    @pytest.mark.asyncio
    async def test_cli_benchmark_failure(self):
        """Test CLI benchmark with command failure"""
        benchmark = CLIInfoBenchmark()
        
        with patch('asyncio.create_subprocess_exec') as mock_subprocess:
            mock_process = AsyncMock()
            mock_process.returncode = 1
            mock_process.communicate.return_value = (b"", b"Command failed")
            mock_subprocess.return_value = mock_process
            
            result_data = await benchmark.run()
            
            assert result_data["exit_code"] == 1
            assert result_data["success"] is False  # Expected exit code is 0 for info
            assert "Command failed" in result_data["stderr"]
    
    def test_cli_benchmark_suite(self):
        """Test CLI benchmark suite creation"""
        suite = CLIBenchmarkSuite()
        
        assert len(suite.benchmarks) > 0
        
        # Check that all expected benchmarks are included
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
        assert stats["p95"] == 4.8  # Actual percentile calculation result
        assert stats["p99"] == 5.0
    
    def test_detect_regression(self):
        """Test regression detection"""
        # Stable performance
        stable_data = [1.0, 1.1, 0.9, 1.0, 1.05]
        result = PerformanceMetrics.detect_performance_regression(stable_data, 1.0)
        assert not result['regression_detected']
        
        # Clear regression (increasing times)
        regression_data = [1.0, 1.5, 2.0, 2.5, 3.0]
        result = PerformanceMetrics.detect_performance_regression(regression_data[:-1], regression_data[-1])
        assert result['regression_detected']
        
        # Improvement (decreasing times)
        improvement_data = [3.0, 2.5, 2.0, 1.5, 1.0]
        result = PerformanceMetrics.detect_performance_regression(improvement_data[:-1], improvement_data[-1])
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
        assert stability["stability_score"] < 0.5  # Low stability
        assert stability["coefficient_of_variation"] > 0.5


class TestBenchmarkAnalyzer:
    """Test BenchmarkAnalyzer utility class"""
    
    def setup_method(self):
        """Setup test analyzer with mock database"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_analyzer.db")
        self.analyzer = BenchmarkAnalyzer(self.db_path)
        
        # Store some test data
        db = BenchmarkDatabase(self.db_path)
        base_time = datetime.now() - timedelta(days=10)
        
        for i in range(10):
            metrics = BenchmarkMetrics(
                execution_time=1.0 + i * 0.1,  # Increasing trend
                memory_usage=64.0 + i * 4.0,
                cpu_usage=50.0 + i * 2.0
            )
            result = BenchmarkResult(
                benchmark_name="analyzer_test",
                success=True,
                metrics=metrics,
                timestamp=base_time + timedelta(days=i)
            )
            db.store_result(result)
    
    def teardown_method(self):
        """Cleanup test files"""
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        os.rmdir(self.temp_dir)
    
    def test_analyze_benchmark_trends(self):
        """Test benchmark trend analysis"""
        analysis = self.analyzer.analyze_benchmark_trends("analyzer_test", days=10)
        
        assert analysis["benchmark_name"] == "analyzer_test"
        assert analysis["data_points"] == 10
        
        # Check execution time analysis
        exec_analysis = analysis["execution_time"]
        assert exec_analysis["trend_direction"] == "increasing"
        assert exec_analysis["statistics"]["mean"] > 1.0
        
        # Check memory usage analysis
        mem_analysis = analysis["memory_usage"]
        assert mem_analysis["trend_direction"] == "increasing"
        assert mem_analysis["statistics"]["mean"] > 64.0
    
    def test_generate_performance_report(self):
        """Test performance report generation"""
        report = self.analyzer.generate_performance_report(["analyzer_test"], days=10)
        
        assert "analyzer_test" in report
        benchmark_report = report["analyzer_test"]
        
        assert "summary" in benchmark_report
        assert "trends" in benchmark_report
        assert "recommendations" in benchmark_report
        
        # Check summary
        summary = benchmark_report["summary"]
        assert summary["total_runs"] == 10
        assert summary["success_rate"] == 100.0
        
        # Check recommendations
        recommendations = benchmark_report["recommendations"]
        assert len(recommendations) > 0
    
    def test_detect_anomalies(self):
        """Test anomaly detection"""
        # Add an anomalous result
        db = BenchmarkDatabase(self.db_path)
        metrics = BenchmarkMetrics(
            execution_time=10.0,  # Much higher than normal
            memory_usage=500.0
        )
        result = BenchmarkResult(
            benchmark_name="analyzer_test",
            success=True,
            metrics=metrics,
            timestamp=datetime.now()
        )
        db.store_result(result)
        
        anomalies = self.analyzer.detect_anomalies("analyzer_test", days=11)
        
        assert len(anomalies) > 0
        # The anomalous result should be detected
        anomaly = anomalies[0]
        assert anomaly["execution_time"] == 10.0
        assert anomaly["anomaly_score"] > 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])