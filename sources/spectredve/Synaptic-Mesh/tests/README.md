# Synaptic Neural Mesh Test Suite

This directory contains comprehensive test suites for the Synaptic Neural Mesh project, ensuring reliability, performance, and correctness across all system components.

## Test Structure

### 1. Unit Tests (`unit/`)
- **Rust Crates**: Individual module testing for QuDAG, ruv-FANN, DAA
- **JavaScript/TypeScript**: Module-level tests for claude-flow and ruv-swarm
- **WASM Integration**: WebAssembly compilation and binding tests
- **Coverage Target**: >95% code coverage

### 2. Integration Tests (`integration/`)
- **Cross-Component**: JS-Rust WASM integration testing
- **MCP Protocol**: Model Context Protocol implementation validation
- **Neural Mesh**: Coordination between neural components
- **DAG Consensus**: Distributed consensus mechanism testing
- **Memory Persistence**: Cross-session state management

### 3. Performance Tests (`performance/`)
- **Neural Inference**: <100ms response time targets
- **Swarm Coordination**: Large-scale agent coordination
- **Memory Usage**: <50MB per agent target
- **Concurrent Handling**: 1000+ agents simultaneously
- **SWE-Bench**: >84.8% solve rate benchmarking

### 4. Stress Tests (`stress/`)
- **High Load**: 10,000+ operations/second capacity
- **Fault Tolerance**: System recovery testing
- **Memory Leaks**: Long-running stability analysis
- **Edge Cases**: Error condition handling
- **24-Hour**: Extended stability validation

### 5. End-to-End Tests (`e2e/`)
- **Full Workflow**: Complete system integration scenarios
- **Real-World**: Production-like environment testing
- **User Journey**: Complete development workflows
- **Cross-Platform**: Multi-environment compatibility

## Test Coverage Targets

| Component | Coverage Target | Critical Path | Public API |
|-----------|----------------|---------------|------------|
| QuDAG Core | >95% | >99% | 100% |
| ruv-FANN | >95% | >99% | 100% |
| DAA Service | >95% | >99% | 100% |
| Claude Flow | >95% | >99% | 100% |
| ruv-swarm | >95% | >99% | 100% |
| WASM Bindings | >90% | >99% | 100% |

## Performance Benchmarks

| Metric | Target | Measurement |
|--------|--------|-------------|
| Neural Inference | <100ms | Average response time |
| Memory per Agent | <50MB | Peak memory usage |
| Concurrent Agents | 1000+ | Simultaneous active agents |
| Swarm Coordination | <1s | Full swarm sync time |
| SWE-Bench Score | >84.8% | Problem solving accuracy |
| System Throughput | 10,000 ops/s | Peak operations per second |

## Running Tests

### All Tests
```bash
# Run comprehensive test suite
./run-all-tests.sh

# Run with coverage analysis
./run-tests-with-coverage.sh
```

### By Category
```bash
# Unit tests
npm run test:unit
cargo test --workspace

# Integration tests
npm run test:integration
cargo test --workspace --features test-integration

# Performance tests
npm run test:performance
./performance/run-benchmarks.sh

# Stress tests
./stress/run-stress-tests.sh
```

### By Component
```bash
# JavaScript/TypeScript components
cd src/js && npm test

# Rust components
cd src/rs && cargo test --workspace

# WASM integration
./test-wasm-integration.sh
```

## Test Configuration

### Jest Configuration (JavaScript/TypeScript)
- Located in `jest.config.js`
- Supports ES modules and WASM
- Coverage reporting enabled
- Parallel test execution

### Cargo Configuration (Rust)
- Workspace-level testing
- Feature-gated tests
- Benchmark support
- Property-based testing

### Environment Setup
- Docker containers for isolation
- CI/CD integration
- Cross-platform testing
- Automated regression detection

## Continuous Integration

The test suite integrates with CI/CD pipelines to ensure:
- Automated testing on all commits
- Performance regression detection
- Coverage trend monitoring
- Cross-platform compatibility
- Security vulnerability scanning

## Test Reports

Test results and coverage reports are generated in:
- `coverage/` - Code coverage analysis
- `performance/` - Benchmark results
- `reports/` - Comprehensive test summaries
- `artifacts/` - Test artifacts and logs