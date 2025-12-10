# Testing Guide

This comprehensive guide covers testing strategies, methodologies, and best practices for SAFLA (Self-Aware Feedback Loop Algorithm). Our testing approach follows Test-Driven Development (TDD) principles and includes unit tests, integration tests, performance benchmarks, and safety validation.

## Table of Contents

1. [Testing Philosophy](#testing-philosophy)
2. [Test Structure Overview](#test-structure-overview)
3. [Unit Testing](#unit-testing)
4. [Integration Testing](#integration-testing)
5. [Performance Testing](#performance-testing)
6. [Safety Testing](#safety-testing)
7. [Test-Driven Development](#test-driven-development)
8. [Running Tests](#running-tests)
9. [Test Coverage](#test-coverage)
10. [Continuous Integration](#continuous-integration)
11. [Testing Best Practices](#testing-best-practices)

## Testing Philosophy

SAFLA's testing strategy is built on several core principles:

### Red-Green-Refactor Cycle
Following TDD principles, our tests follow the Red-Green-Refactor cycle:
- **Red**: Write failing tests that define expected behavior
- **Green**: Write minimal implementation to make tests pass
- **Refactor**: Improve code while maintaining test coverage

### Comprehensive Coverage
Our testing covers multiple dimensions:
- **Functional Testing**: Verifies component behavior
- **Performance Testing**: Ensures performance targets are met
- **Integration Testing**: Validates component interactions
- **Safety Testing**: Confirms safety constraints are enforced
- **Resilience Testing**: Tests error handling and recovery

### Baseline-Driven Testing
Performance tests establish baselines and track improvements:
```python
# Example from test_performance_benchmarks.py
def test_vector_search_latency_baseline(self, vector_memory_manager, test_vectors_10k):
    """Test baseline vector search latency (RED phase - should fail initially)."""
    # Store baseline for comparison
    pytest.baseline_vector_search_latency = avg_latency
    
    # For now, we expect this to fail until optimization is implemented
    if not result.meets_target:
        pytest.skip(f"Baseline test - measured latency {avg_latency:.3f}ms exceeds target {benchmark.target_value}ms")
```

## Test Structure Overview

### Test Directory Structure
```
tests/
├── test_adaptive_safety_boundaries.py      # Safety boundary testing
├── test_auto_scaling_infrastructure.py     # Auto-scaling tests
├── test_delta_evaluation.py                # Delta evaluation unit tests
├── test_distributed_mcp_orchestration.py   # Distributed MCP tests
├── test_federated_learning_integration.py  # Federated learning tests
├── test_hybrid_memory.py                   # Memory system tests
├── test_mcp_orchestration.py               # MCP orchestration tests
├── test_memory_optimizations.py            # Memory optimization tests
├── test_meta_cognitive_engine.py           # Meta-cognitive engine tests
├── test_ml_neural_embedding_engine.py      # Neural embedding tests
├── test_ml_rl_optimizer.py                 # RL optimizer tests
├── test_performance_benchmarks.py          # Performance benchmarks
├── test_safety_validation.py               # Safety validation tests
└── integration/                            # Integration test suite
    ├── conftest.py                         # Shared test configuration
    ├── test_deployment_readiness.py        # Deployment readiness tests
    ├── test_end_to_end_workflows.py        # End-to-end workflow tests
    ├── test_performance_integration.py     # Performance integration tests
    ├── test_safety_integration.py          # Safety integration tests
    └── test_system_integration.py          # System integration tests
```

### Test Categories

#### 1. Unit Tests
Test individual components in isolation:
- **Delta Evaluation**: [`test_delta_evaluation.py`](../../tests/test_delta_evaluation.py)
- **Memory System**: [`test_hybrid_memory.py`](../../tests/test_hybrid_memory.py)
- **Safety Validation**: [`test_safety_validation.py`](../../tests/test_safety_validation.py)
- **MCP Orchestration**: [`test_mcp_orchestration.py`](../../tests/test_mcp_orchestration.py)

#### 2. Integration Tests
Test component interactions and system workflows:
- **System Integration**: [`test_system_integration.py`](../../tests/integration/test_system_integration.py)
- **End-to-End Workflows**: [`test_end_to_end_workflows.py`](../../tests/integration/test_end_to_end_workflows.py)
- **Performance Integration**: [`test_performance_integration.py`](../../tests/integration/test_performance_integration.py)

#### 3. Performance Tests
Benchmark system performance with specific targets:
- **Performance Benchmarks**: [`test_performance_benchmarks.py`](../../tests/test_performance_benchmarks.py)

## Unit Testing

### Component Testing Strategy

#### Delta Evaluation Testing
Tests for the delta evaluation system focus on accuracy and performance:

```python
# Example test structure
class TestDeltaEvaluationBenchmarks:
    DELTA_EVALUATION_BENCHMARKS = [
        PerformanceBenchmark(
            name="delta_evaluation_throughput",
            target_value=500.0,  # 50% improvement over baseline
            target_unit="ops/sec",
            description="Delta evaluation pipeline throughput"
        )
    ]
    
    def test_delta_evaluation_throughput_baseline(self, delta_evaluator, test_delta_data):
        # Measure throughput and compare against targets
        pass
```

#### Memory System Testing
Memory system tests cover storage, retrieval, and optimization:

```python
# Vector similarity search performance targets
VECTOR_SEARCH_BENCHMARKS = [
    PerformanceBenchmark(
        name="vector_search_latency_10k",
        target_value=1.0,  # <1ms
        target_unit="ms",
        description="Vector similarity search latency for 10k vectors"
    )
]
```

#### Safety Validation Testing
Safety tests ensure constraints are properly enforced:

```python
SAFETY_VALIDATION_BENCHMARKS = [
    PerformanceBenchmark(
        name="safety_validation_latency",
        target_value=10.0,  # <10ms real-time performance
        target_unit="ms",
        description="Safety validation pipeline latency"
    )
]
```

### Test Fixtures and Utilities

#### Common Test Data
```python
@pytest.fixture
def test_vectors_10k(self):
    """Generate 10k test vectors for benchmarking."""
    np.random.seed(42)  # Reproducible results
    vectors = []
    metadata_list = []
    
    for i in range(10000):
        vector = np.random.normal(0, 1, 512).astype(np.float32)
        vector = vector / np.linalg.norm(vector)  # Normalize
        vectors.append(vector)
        
        metadata_list.append({
            'id': f'test_vector_{i}',
            'category': f'category_{i % 100}',
            'timestamp': time.time() + i
        })
    
    return vectors, metadata_list
```

#### Performance Benchmark Framework
```python
@dataclass
class PerformanceBenchmark:
    """Performance benchmark specification."""
    name: str
    target_value: float
    target_unit: str
    tolerance: float = 0.1  # 10% tolerance by default
    description: str = ""
    
    def meets_target(self, measured_value: float) -> bool:
        """Check if measured value meets the target."""
        if "latency" in self.name.lower() or "time" in self.name.lower():
            # For latency/time metrics, lower is better
            return measured_value <= self.target_value * (1 + self.tolerance)
        else:
            # For throughput/performance metrics, higher is better
            return measured_value >= self.target_value * (1 - self.tolerance)
```

## Integration Testing

### System Integration Testing

Integration tests validate that components work together correctly. The main integration test suite is in [`test_system_integration.py`](../../tests/integration/test_system_integration.py).

#### Integration Test Context
```python
@dataclass
class IntegrationTestContext:
    """Context for integration tests with shared state and configuration."""
    test_id: str
    temp_dir: Path
    config: Dict[str, Any] = field(default_factory=dict)
    components: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)
    
    def log_event(self, event_type: str, data: Dict[str, Any]):
        """Log an event for analysis."""
        self.events.append({
            'timestamp': time.time(),
            'type': event_type,
            'data': data
        })
```

#### Component Integration Tests
Test interactions between specific components:

```python
@pytest.mark.asyncio
async def test_delta_evaluation_meta_cognitive_integration(self, integrated_system):
    """Test integration between delta evaluation and meta-cognitive engine."""
    components = integrated_system['components']
    
    # Create test metrics that should trigger adaptation
    metrics = DeltaMetrics(
        performance=0.25,  # Significant improvement
        efficiency=0.15,
        stability=0.05,
        capability=0.20,
        confidence=0.9
    )
    
    # Evaluate delta - should trigger meta-cognitive adaptation
    result = await delta_evaluator.evaluate(metrics)
    
    # Verify integration
    assert result.total_delta > context.config['meta_cognitive']['adaptation_threshold']
```

### End-to-End Workflow Testing

Complete workflow tests validate the entire SAFLA feedback loop:

```python
@pytest.mark.asyncio
async def test_complete_self_improvement_cycle(self, integrated_system):
    """Test complete self-improvement cycle: observation → evaluation → decision → modification → validation."""
    
    # Phase 1: Observation - System observes its current state
    initial_state = await components['meta_cognitive_engine'].observe_system_state()
    
    # Phase 2: Evaluation - Delta evaluation of current performance
    evaluation_result = await components['delta_evaluator'].evaluate(delta_metrics)
    
    # Phase 3: Decision - Meta-cognitive engine decides on adaptation
    decision = await components['meta_cognitive_engine'].make_adaptation_decision(evaluation_result)
    
    # Phase 4: Modification - Apply the decided modifications
    if decision['should_adapt']:
        modification_result = await components['meta_cognitive_engine'].apply_adaptation(decision['strategy'])
        
        # Phase 5: Validation - Validate the modifications are safe and effective
        validation_result = await components['safety_validator'].validate_modification(modification_result)
        
        assert validation_result.is_safe
        assert validation_result.effectiveness_score > 0.5
```

### Cross-Component Communication Testing

Test event-driven communication between components:

```python
@pytest.mark.asyncio
async def test_event_propagation_across_components(self, integrated_system):
    """Test that events propagate correctly across all components."""
    
    # Create an event that should propagate through multiple components
    initial_event = {
        'type': 'performance_degradation',
        'severity': 'high',
        'affected_components': ['memory_system', 'mcp_orchestrator'],
        'metrics': {'performance_drop': 0.25, 'error_rate_increase': 0.15}
    }
    
    # Trigger event in meta-cognitive engine
    await components['meta_cognitive_engine'].process_system_event(initial_event)
    
    # Verify event was received by relevant components
    memory_events = await components['hybrid_memory'].get_received_events()
    mcp_events = await components['mcp_orchestrator'].get_received_events()
    
    assert any(e['type'] == 'performance_degradation' for e in memory_events)
    assert any(e['type'] == 'performance_degradation' for e in mcp_events)
```

## Performance Testing

### Performance Benchmarks

Performance tests establish specific targets and measure system performance against them. Key performance targets include:

#### Vector Similarity Search
- **Latency**: <1ms for 10k vectors
- **Throughput**: 1000 searches/second
- **Batch Storage**: <10ms for 1000 vectors
- **Memory Efficiency**: 80% storage efficiency

#### Delta Evaluation Pipeline
- **Throughput**: 50% improvement over baseline
- **Latency**: <2ms per evaluation
- **Batch Processing**: 100 evaluations per batch

#### MCP Communication
- **Round-trip Latency**: <5ms
- **Concurrent Requests**: 100 concurrent requests
- **Message Throughput**: 1000 messages/second

#### Safety Validation
- **Validation Latency**: <10ms real-time performance
- **Throughput**: 100 validations/second
- **Rule Evaluation**: 1000 rules/second

### Performance Test Implementation

```python
def test_vector_search_latency_baseline(self, vector_memory_manager, test_vectors_10k):
    """Test baseline vector search latency (RED phase - should fail initially)."""
    vectors, metadata_list = test_vectors_10k
    benchmark = self.VECTOR_SEARCH_BENCHMARKS[0]  # vector_search_latency_10k
    
    # Store vectors
    for vector, metadata in zip(vectors, metadata_list):
        vector_memory_manager.store(vector, metadata)
    
    # Measure search latency
    query_vector = np.random.normal(0, 1, 512).astype(np.float32)
    query_vector = query_vector / np.linalg.norm(query_vector)
    
    # Warm up
    for _ in range(10):
        vector_memory_manager.similarity_search(query_vector, k=10)
    
    # Benchmark search latency
    search_times = []
    for _ in range(100):
        start_time = time.perf_counter()
        results = vector_memory_manager.similarity_search(query_vector, k=10)
        end_time = time.perf_counter()
        
        search_times.append((end_time - start_time) * 1000)  # Convert to ms
        assert len(results) <= 10
    
    avg_latency = statistics.mean(search_times)
    p95_latency = statistics.quantiles(search_times, n=20)[18]
    
    # Store baseline for comparison
    pytest.baseline_vector_search_latency = avg_latency
```

### Concurrent Performance Testing

Test system performance under concurrent load:

```python
@pytest.mark.asyncio
async def test_concurrent_operations_performance(self, integrated_system):
    """Test system performance with concurrent operations across all components."""
    
    async def memory_operations():
        """Concurrent memory operations."""
        tasks = []
        for i in range(50):
            node = MemoryNode(
                id=f"perf_test_{i}",
                content=f"Performance test content {i}",
                embedding=[0.1 * (i % 10)] * 128,
                metadata={'test': 'performance', 'batch': i // 10}
            )
            tasks.append(components['memory_system'].store_node(node))
        
        await asyncio.gather(*tasks)
        return len(tasks)
    
    # Run concurrent operations and measure performance
    start_time = time.time()
    memory_count, delta_count, mcp_count = await asyncio.gather(
        memory_operations(),
        delta_evaluations(),
        mcp_tasks()
    )
    end_time = time.time()
    
    # Verify performance benchmarks
    total_operations = memory_count + delta_count + mcp_count
    operations_per_second = total_operations / (end_time - start_time)
    
    assert operations_per_second > 10  # Minimum 10 ops/sec
```

## Safety Testing

### Safety Constraint Testing

Safety tests ensure that safety constraints are properly enforced across all components:

```python
@pytest.mark.asyncio
async def test_safety_constraints_across_components(self, integrated_system):
    """Test that safety constraints are enforced across all components."""
    
    # Define safety constraints
    safety_constraints = [
        SafetyConstraint(
            name="max_modification_rate",
            threshold=0.1,
            component="meta_engine",
            action="halt_modifications"
        ),
        SafetyConstraint(
            name="memory_corruption_prevention",
            threshold=0.05,
            component="memory_system", 
            action="rollback_changes"
        )
    ]
    
    # Register constraints and test enforcement
    for constraint in safety_constraints:
        await components['safety_validator'].register_constraint(constraint)
```

### Emergency Stop Testing

Test emergency stop propagation across all components:

```python
@pytest.mark.asyncio
async def test_emergency_safety_stop_propagation(self, integrated_system):
    """Test that emergency safety stops propagate across all components."""
    
    # Create critical safety violation
    critical_violation = {
        'type': 'system_instability',
        'severity': 'critical',
        'affected_components': 'all',
        'metrics': {
            'stability_score': 0.1,  # Very low stability
            'error_rate': 0.8,       # Very high error rate
            'performance_drop': 0.9   # Severe performance drop
        }
    }
    
    # Trigger emergency stop
    await components['safety_validator'].trigger_emergency_stop(critical_violation)
    
    # Verify all components received emergency stop
    assert await components['meta_engine'].is_emergency_stopped()
    assert await components['memory_system'].is_emergency_stopped()
    assert await components['mcp_orchestrator'].is_emergency_stopped()
```

## Test-Driven Development

### TDD Workflow in SAFLA

Our TDD approach follows these steps:

1. **Write Failing Tests (Red)**
   - Define expected behavior through tests
   - Tests should fail initially (no implementation)
   - Focus on one behavior at a time

2. **Implement Minimal Code (Green)**
   - Write just enough code to make tests pass
   - Don't optimize prematurely
   - Focus on functionality, not performance

3. **Refactor and Optimize (Refactor)**
   - Improve code structure and performance
   - Maintain test coverage
   - Add performance optimizations

### Example TDD Cycle

```python
# RED: Write failing test
def test_vector_search_performance_target():
    """Test that vector search meets performance target."""
    # This test will fail initially
    assert search_latency < 1.0  # Target: <1ms

# GREEN: Implement basic functionality
def similarity_search(self, query_vector, k=10):
    """Basic implementation that passes test."""
    # Simple linear search (slow but correct)
    similarities = []
    for vector in self.vectors:
        similarity = cosine_similarity(query_vector, vector)
        similarities.append(similarity)
    return sorted(similarities, reverse=True)[:k]

# REFACTOR: Optimize for performance
def similarity_search(self, query_vector, k=10):
    """Optimized implementation with indexing."""
    # Use FAISS or similar for fast similarity search
    distances, indices = self.index.search(query_vector, k)
    return [(self.vectors[i], distances[i]) for i in indices]
```

### TDD Best Practices

1. **Test First**: Always write tests before implementation
2. **Small Steps**: Make small, incremental changes
3. **Fast Feedback**: Keep test execution time low
4. **Clear Intent**: Tests should clearly express expected behavior
5. **Refactor Regularly**: Improve code structure continuously

## Running Tests

### Test Execution Commands

#### Run All Tests
```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run with coverage
pytest --cov=safla --cov-report=html
```

#### Run Specific Test Categories
```bash
# Run unit tests only
pytest tests/test_*.py

# Run integration tests only
pytest tests/integration/

# Run performance benchmarks
pytest tests/test_performance_benchmarks.py

# Run safety tests
pytest tests/test_safety_validation.py tests/integration/test_safety_integration.py
```

#### Run Specific Test Classes or Methods
```bash
# Run specific test class
pytest tests/test_performance_benchmarks.py::TestVectorSimilaritySearchBenchmarks

# Run specific test method
pytest tests/test_performance_benchmarks.py::TestVectorSimilaritySearchBenchmarks::test_vector_search_latency_baseline

# Run tests matching pattern
pytest -k "vector_search"
```

### Test Configuration

#### pytest.ini Configuration
```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    --strict-markers
    --strict-config
    --verbose
    --tb=short
    --cov=safla
    --cov-report=term-missing
    --cov-report=html:htmlcov
    --cov-fail-under=80
markers =
    unit: Unit tests
    integration: Integration tests
    performance: Performance tests
    safety: Safety tests
    slow: Slow running tests
```

#### Test Environment Setup
```python
# conftest.py
import pytest
import asyncio
import tempfile
from pathlib import Path

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)
```

## Test Coverage

### Coverage Targets

- **Overall Coverage**: >80%
- **Core Components**: >90%
- **Safety-Critical Code**: >95%
- **Performance-Critical Code**: >85%

### Coverage Analysis

```bash
# Generate coverage report
pytest --cov=safla --cov-report=html

# View coverage in browser
open htmlcov/index.html

# Generate coverage badge
coverage-badge -o coverage.svg
```

### Coverage Configuration
```ini
# .coveragerc
[run]
source = safla
omit = 
    */tests/*
    */venv/*
    */virtualenv/*
    setup.py

[report]
exclude_lines =
    pragma: no cover
    def __repr__
    raise AssertionError
    raise NotImplementedError
```

## Continuous Integration

### GitHub Actions Workflow

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, 3.10, 3.11]
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v3
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements-test.txt
        pip install -e .
    
    - name: Run unit tests
      run: pytest tests/test_*.py -v
    
    - name: Run integration tests
      run: pytest tests/integration/ -v
    
    - name: Run performance benchmarks
      run: pytest tests/test_performance_benchmarks.py -v
    
    - name: Generate coverage report
      run: |
        pytest --cov=safla --cov-report=xml
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
```

### Pre-commit Hooks

```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: pytest-check
        name: pytest-check
        entry: pytest
        language: system
        pass_filenames: false
        always_run: true
        args: [tests/test_*.py, -x, -v]
```

## Testing Best Practices

### 1. Test Organization
- **Group Related Tests**: Use test classes to group related functionality
- **Clear Test Names**: Use descriptive names that explain what is being tested
- **Logical Structure**: Organize tests to match code structure

### 2. Test Data Management
- **Use Fixtures**: Create reusable test data with pytest fixtures
- **Reproducible Data**: Use fixed seeds for random data generation
- **Realistic Data**: Use data that represents real-world scenarios

### 3. Assertion Best Practices
- **Specific Assertions**: Use specific assertions that clearly indicate what failed
- **Multiple Assertions**: Group related assertions in single tests
- **Custom Assertions**: Create custom assertion helpers for complex validations

### 4. Performance Testing
- **Baseline Measurements**: Establish performance baselines
- **Statistical Significance**: Use multiple measurements and statistical analysis
- **Environment Consistency**: Control test environment variables

### 5. Integration Testing
- **Realistic Scenarios**: Test real-world usage patterns
- **Error Conditions**: Test error handling and recovery
- **Resource Management**: Properly clean up resources after tests

### 6. Safety Testing
- **Boundary Conditions**: Test at safety constraint boundaries
- **Failure Scenarios**: Test system behavior during failures
- **Recovery Testing**: Verify system can recover from safety violations

### 7. Test Maintenance
- **Regular Updates**: Keep tests updated with code changes
- **Refactor Tests**: Apply refactoring principles to test code
- **Documentation**: Document complex test scenarios and setups

## Troubleshooting Common Test Issues

### Performance Test Failures
```python
# Common issue: Inconsistent performance measurements
# Solution: Use warm-up runs and multiple measurements
def test_with_warmup():
    # Warm up
    for _ in range(10):
        function_under_test()
    
    # Actual measurements
    times = []
    for _ in range(100):
        start = time.perf_counter()
        function_under_test()
        end = time.perf_counter()
        times.append(end - start)
    
    avg_time = statistics.mean(times)
    assert avg_time < target_time
```

### Integration Test Flakiness
```python
# Common issue: Race conditions in async tests
# Solution: Use proper synchronization
@pytest.mark.asyncio
async def test_with_proper_sync():
    # Start background task
    task = asyncio.create_task(background_operation())
    
    # Wait for specific condition
    await wait_for_condition(lambda: system.is_ready())
    
    # Perform test
    result = await system.operation()
    
    # Cleanup
    task.cancel()
    await task
```

### Memory Test Issues
```python
# Common issue: Memory leaks in tests
# Solution: Proper cleanup and monitoring
def test_with_memory_monitoring():
    initial_memory = get_memory_usage()
    
    try:
        # Test operations
        perform_memory_intensive_operation()
        
        # Verify memory usage
        current_memory = get_memory_usage()
        assert current_memory - initial_memory < memory_threshold
        
    finally:
        # Cleanup
        cleanup_resources()
```

This comprehensive testing guide provides the foundation for maintaining high-quality, reliable, and performant SAFLA implementations. Regular testing ensures that the system continues to meet its performance targets while maintaining safety and reliability standards.