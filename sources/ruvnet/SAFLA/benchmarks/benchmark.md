# SAFLA Benchmark Framework Documentation

## Overview

The SAFLA Benchmark Framework is a comprehensive performance testing and analysis system designed to monitor, evaluate, and optimize the performance of the SAFLA CLI system. It provides automated benchmarking capabilities with persistent storage, statistical analysis, and performance trend monitoring.

## Features

### Core Components

1. **Benchmark Framework** (`core.py`)
   - Abstract base classes for creating custom benchmarks
   - Asynchronous benchmark execution with timeout support
   - Memory and CPU usage monitoring
   - Comprehensive metrics collection

2. **Database Storage** (`database.py`)
   - SQLite-based persistent storage for benchmark results
   - Performance trend tracking over time
   - Data export capabilities
   - Automatic cleanup of old results

3. **CLI-Specific Benchmarks** (`cli_benchmarks.py`)
   - Pre-built benchmarks for SAFLA CLI commands
   - Memory usage monitoring for CLI operations
   - Batch command execution testing
   - Command validation and help system benchmarks

4. **Analysis Utilities** (`utils.py`)
   - Statistical analysis of benchmark results
   - Performance regression detection
   - Trend analysis and stability scoring
   - Configuration management

## Architecture

### Benchmark Classes

```python
# Abstract base class for all benchmarks
class Benchmark:
    async def setup(self) -> None
    async def run(self) -> dict
    async def teardown(self) -> None

# Benchmark execution and monitoring
class BenchmarkRunner:
    async def run_benchmark(benchmark: Benchmark) -> BenchmarkResult

# Collection of related benchmarks
class BenchmarkSuite:
    async def run_all(parallel: bool = False) -> List[BenchmarkResult]
```

### Data Models

```python
@dataclass
class BenchmarkMetrics:
    execution_time: float
    memory_usage: float
    cpu_usage: float
    throughput: Optional[float] = None
    latency: Optional[float] = None
    error_rate: Optional[float] = None
    custom_metrics: dict = field(default_factory=dict)

@dataclass
class BenchmarkResult:
    benchmark_name: str
    timestamp: datetime
    metrics: BenchmarkMetrics
    success: bool
    error_message: Optional[str] = None
    metadata: dict = field(default_factory=dict)
```

## Usage

### Running Benchmarks

#### CLI Commands

```bash
# Run all benchmarks
python -m safla benchmark run --suite all --store

# Run only CLI benchmarks
python -m safla benchmark run --suite cli --store

# Run benchmarks in parallel
python -m safla benchmark run --parallel --store

# Export results to file
python -m safla benchmark run --output results.json
```

#### View Results

```bash
# Show all recent results
python -m safla benchmark results

# Show results for specific benchmark
python -m safla benchmark results --benchmark cli_help_command

# Show results with limit
python -m safla benchmark results --limit 10
```

#### Analyze Performance

```bash
# Analyze specific benchmark
python -m safla benchmark analyze --benchmark cli_info_command

# Analyze trends over time period
python -m safla benchmark analyze --benchmark cli_info_command --days 7
```

#### Data Management

```bash
# Export results to JSON
python -m safla benchmark export --output benchmark_data.json

# Clean up old results (older than 90 days)
python -m safla benchmark cleanup --days 90
```

### Programmatic Usage

#### Creating Custom Benchmarks

```python
from benchmarks import Benchmark
import asyncio

class CustomBenchmark(Benchmark):
    def __init__(self):
        super().__init__("custom_test", "Custom benchmark example")
    
    async def setup(self):
        # Initialize test environment
        self.test_data = [i for i in range(1000)]
    
    async def run(self) -> dict:
        # Perform the actual benchmark
        result = sum(self.test_data)
        return {
            "sum_result": result,
            "data_size": len(self.test_data)
        }
    
    async def teardown(self):
        # Clean up resources
        del self.test_data

# Run the benchmark
async def main():
    from benchmarks import BenchmarkRunner
    
    benchmark = CustomBenchmark()
    runner = BenchmarkRunner()
    result = await runner.run_benchmark(benchmark)
    
    print(f"Execution time: {result.metrics.execution_time}s")
    print(f"Success: {result.success}")

asyncio.run(main())
```

#### Using the Database

```python
from benchmarks import BenchmarkDatabase
from datetime import datetime, timedelta

# Initialize database
db = BenchmarkDatabase("benchmarks/results.db")

# Get recent results
results = db.get_latest_results("cli_help_command", limit=10)

# Get performance trends
trends = db.get_performance_trends("cli_info_command", "execution_time", days=30)

# Store custom result
from benchmarks import BenchmarkResult, BenchmarkMetrics

metrics = BenchmarkMetrics(
    execution_time=1.5,
    memory_usage=128.0,
    cpu_usage=75.0
)

result = BenchmarkResult(
    benchmark_name="custom_test",
    timestamp=datetime.now(),
    metrics=metrics,
    success=True
)

result_id = db.store_result(result)
```

## Available Benchmarks

### CLI Benchmarks

1. **cli_help_command**
   - Tests the `--help` command performance
   - Measures command response time
   - Validates help output format

2. **cli_info_command**
   - Tests the `info` command performance
   - Measures system information gathering time
   - Validates output completeness

3. **cli_status_command**
   - Tests the `status` command performance
   - Measures status check execution time
   - Validates status reporting accuracy

4. **cli_config_init_command**
   - Tests configuration initialization
   - Measures config file creation time
   - Validates configuration structure

5. **cli_validate_command**
   - Tests validation functionality
   - Measures validation execution time
   - Tests various validation scenarios

6. **cli_batch_basic_commands**
   - Tests multiple commands in sequence
   - Measures batch execution performance
   - Calculates throughput metrics

7. **cli_info_memory_usage**
   - Monitors memory usage during info command
   - Tracks memory allocation patterns
   - Detects memory leaks

8. **cli_validate_memory_usage**
   - Monitors memory usage during validation
   - Tracks memory efficiency
   - Validates memory cleanup

## Performance Metrics

### Execution Time
- **Description**: Time taken to complete the benchmark
- **Unit**: Seconds
- **Lower is better**: Yes
- **Typical Range**: 0.1s - 10s for CLI commands

### Memory Usage
- **Description**: Peak memory consumption during execution
- **Unit**: Megabytes (MB)
- **Lower is better**: Yes
- **Monitoring**: Real-time memory tracking

### CPU Usage
- **Description**: Average CPU utilization during execution
- **Unit**: Percentage (%)
- **Range**: 0-100%
- **Monitoring**: Periodic sampling

### Throughput
- **Description**: Operations completed per second
- **Unit**: Operations/second
- **Higher is better**: Yes
- **Applicable**: Batch operations

### Error Rate
- **Description**: Percentage of failed operations
- **Unit**: Percentage (%)
- **Lower is better**: Yes
- **Target**: 0% for stable operations

## Statistical Analysis

### Trend Detection
- **Stable**: Performance remains consistent over time
- **Improving**: Performance is getting better
- **Degrading**: Performance is declining
- **Insufficient Data**: Not enough data points for analysis

### Regression Detection
- Automatic detection of performance regressions
- Configurable thresholds for regression alerts
- Statistical significance testing
- Baseline comparison capabilities

### Stability Scoring
- **Score Range**: 0.0 - 1.0
- **1.0**: Perfectly stable performance
- **0.8-0.9**: Good stability
- **0.6-0.7**: Moderate stability
- **<0.6**: Poor stability, investigation needed

## Configuration

### Benchmark Configuration

```python
from benchmarks import BenchmarkConfig

config = BenchmarkConfig(
    warmup_iterations=3,           # Number of warmup runs
    measurement_iterations=10,     # Number of measurement runs
    timeout_seconds=30.0,          # Timeout per benchmark
    parallel_execution=False,      # Run benchmarks in parallel
    store_results=True,           # Store results in database
    database_path="benchmarks/results.db",
    export_results=False,         # Export to file
    export_path=None,             # Export file path
    cleanup_days=90,              # Days to keep results
    log_level="INFO"              # Logging level
)
```

### Configuration File

Create a JSON configuration file:

```json
{
    "warmup_iterations": 3,
    "measurement_iterations": 10,
    "timeout_seconds": 30.0,
    "parallel_execution": false,
    "store_results": true,
    "database_path": "benchmarks/results.db",
    "export_results": false,
    "cleanup_days": 90,
    "log_level": "INFO"
}
```

Load configuration:

```bash
python -m safla benchmark run --config benchmark_config.json
```

## Database Schema

### benchmark_results Table
```sql
CREATE TABLE benchmark_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    benchmark_name TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    execution_time REAL NOT NULL,
    memory_usage REAL NOT NULL,
    cpu_usage REAL NOT NULL,
    throughput REAL,
    latency REAL,
    error_rate REAL,
    custom_metrics TEXT,  -- JSON string
    success BOOLEAN NOT NULL,
    error_message TEXT,
    metadata TEXT,  -- JSON string
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
```

### benchmark_suites Table
```sql
CREATE TABLE benchmark_suites (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    suite_name TEXT NOT NULL,
    description TEXT,
    total_benchmarks INTEGER NOT NULL,
    successful_benchmarks INTEGER NOT NULL,
    total_execution_time REAL NOT NULL,
    timestamp TEXT NOT NULL,
    metadata TEXT,  -- JSON string
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
```

### performance_trends Table
```sql
CREATE TABLE performance_trends (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    benchmark_name TEXT NOT NULL,
    metric_name TEXT NOT NULL,
    metric_value REAL NOT NULL,
    timestamp TEXT NOT NULL,
    optimization_tag TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
```

## Best Practices

### Benchmark Design
1. **Keep benchmarks focused**: Each benchmark should test one specific aspect
2. **Use realistic data**: Test with data similar to production scenarios
3. **Include setup/teardown**: Properly initialize and clean up test environments
4. **Handle errors gracefully**: Implement proper error handling and reporting
5. **Document expectations**: Clearly document what each benchmark measures

### Performance Monitoring
1. **Regular execution**: Run benchmarks regularly to catch regressions early
2. **Baseline establishment**: Establish performance baselines for comparison
3. **Trend analysis**: Monitor trends over time, not just individual results
4. **Alert thresholds**: Set up alerts for significant performance changes
5. **Historical data**: Maintain historical data for long-term analysis

### Data Management
1. **Regular cleanup**: Remove old benchmark data to manage storage
2. **Data export**: Regularly export important benchmark data
3. **Backup strategy**: Implement backup strategy for benchmark databases
4. **Data validation**: Validate benchmark data integrity regularly

## Troubleshooting

### Common Issues

#### Memory Monitoring Not Working
- **Symptom**: Memory usage shows 0.0MB
- **Cause**: Memory monitoring may not work in all environments
- **Solution**: This is expected in containerized environments; focus on execution time

#### Benchmarks Taking Too Long
- **Symptom**: Benchmarks exceed timeout
- **Cause**: System load or inefficient operations
- **Solution**: Increase timeout or optimize the operations being tested

#### Database Errors
- **Symptom**: Failed to store results
- **Cause**: Database permissions or corruption
- **Solution**: Check file permissions and database integrity

#### Inconsistent Results
- **Symptom**: High variance in benchmark results
- **Cause**: System load or external factors
- **Solution**: Run benchmarks during low-load periods, increase iterations

### Performance Optimization

#### Identifying Bottlenecks
1. Use the analysis tools to identify slow benchmarks
2. Compare execution times across different runs
3. Look for trends indicating performance degradation
4. Focus on benchmarks with high variance

#### Optimization Strategies
1. **Code optimization**: Optimize slow code paths identified by benchmarks
2. **Resource management**: Improve memory and CPU usage efficiency
3. **Caching**: Implement caching for frequently accessed data
4. **Parallel processing**: Use parallel execution where appropriate

## Integration with CI/CD

### GitHub Actions Example

```yaml
name: Performance Benchmarks

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  benchmark:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.9
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
    
    - name: Run benchmarks
      run: |
        python -m safla benchmark run --suite all --store
    
    - name: Analyze results
      run: |
        python -m safla benchmark analyze --benchmark cli_info_command
    
    - name: Export results
      run: |
        python -m safla benchmark export --output benchmark_results.json
    
    - name: Upload results
      uses: actions/upload-artifact@v2
      with:
        name: benchmark-results
        path: benchmark_results.json
```

## Future Enhancements

### Planned Features
1. **Web Dashboard**: Real-time performance monitoring dashboard
2. **Alert System**: Automated alerts for performance regressions
3. **Comparison Tools**: Compare performance across different versions
4. **Load Testing**: Extended load testing capabilities
5. **Custom Metrics**: Enhanced support for domain-specific metrics

### Contributing
To contribute new benchmarks or improvements:

1. Follow the existing benchmark patterns
2. Include comprehensive tests
3. Document the benchmark purpose and expected results
4. Ensure proper error handling
5. Add appropriate logging

## Conclusion

The SAFLA Benchmark Framework provides a robust foundation for performance monitoring and optimization. By regularly running benchmarks and analyzing trends, teams can maintain high performance standards and quickly identify potential issues.

For questions or support, please refer to the main SAFLA documentation or open an issue in the project repository.