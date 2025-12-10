# QUIC Transport Test Suite

Comprehensive test suite for QUIC protocol implementation in agentic-flow.

## Test Structure

```
tests/
├── transport/          # Unit tests for QUIC transport layer
│   └── quic.test.ts   # Connection, streams, multiplexing, migration
├── integration/        # Integration tests for QUIC proxy
│   └── quic-proxy.test.ts  # Agent communication, fallback, config
├── e2e/               # End-to-end workflow tests
│   └── quic-workflow.test.ts  # Full SPARC workflows, orchestration
├── vitest.config.ts   # Test configuration
├── setup.ts           # Global test setup
└── README.md          # This file
```

## Running Tests

### Run all tests
```bash
npm test
```

### Run specific test suites
```bash
# Unit tests only
npm test tests/transport

# Integration tests only
npm test tests/integration

# E2E tests only
npm test tests/e2e
```

### Run with coverage
```bash
npm run test:coverage
```

### Watch mode
```bash
npm run test:watch
```

## Coverage Requirements

Target: 90%+ coverage across all metrics:
- Lines: 90%
- Functions: 90%
- Branches: 90%
- Statements: 90%

## Test Categories

### Unit Tests (tests/transport/quic.test.ts)

Tests for core QUIC transport functionality:
- ✅ Connection establishment (0-RTT, 1-RTT)
- ✅ Stream multiplexing (bidirectional, unidirectional)
- ✅ Error handling (connection loss, stream errors)
- ✅ Connection migration (network changes)
- ✅ Performance characteristics
- ✅ Edge cases and memory management
- ✅ Stream prioritization

**Coverage:** Core transport layer implementation

### Integration Tests (tests/integration/quic-proxy.test.ts)

Tests for QUIC proxy integration:
- ✅ Agent communication (spawn, execute, terminate)
- ✅ Multi-agent scenarios (concurrent operations)
- ✅ HTTP/2 fallback mechanisms
- ✅ Configuration loading (environment variables)
- ✅ Error recovery (retry logic, stream isolation)
- ✅ Memory operations (dedicated streams)
- ✅ Concurrency scenarios (100+ concurrent ops)

**Coverage:** Proxy layer and agent integration

### E2E Tests (tests/e2e/quic-workflow.test.ts)

Tests for complete workflows:
- ✅ Swarm initialization (mesh, hierarchical topologies)
- ✅ Task orchestration (parallel, sequential execution)
- ✅ Agent lifecycle (spawn → execute → terminate)
- ✅ Full SPARC workflow (5 phases)
- ✅ Stress testing (50-agent, 100-step workflows)
- ✅ Real-world scenarios (code review, deployment, data pipelines)

**Coverage:** End-to-end system integration

## Performance Benchmarks

The test suite includes performance validation based on research targets:

| Metric | Target | Test Location |
|--------|--------|---------------|
| 0-RTT connection | < 50ms | transport/quic.test.ts |
| Stream creation | < 10ms | transport/quic.test.ts |
| 10-agent spawn | < 200ms | integration/quic-proxy.test.ts |
| Connection migration | < 50ms | transport/quic.test.ts |
| 100-stream concurrency | < 1000ms | transport/quic.test.ts |
| QUIC vs HTTP/2 speedup | > 1.5x | integration/quic-proxy.test.ts |

## Test Architecture

Based on the QUIC research document architecture:

```
┌────────────────────────────────────────────────────────┐
│              QUIC TEST ARCHITECTURE                    │
├────────────────────────────────────────────────────────┤
│                                                        │
│  E2E Tests (Workflow Orchestration)                    │
│  ┌──────────────────────────────────────────────┐    │
│  │  • Full SPARC workflows                      │    │
│  │  • Multi-agent orchestration                 │    │
│  │  • Real-world scenarios                      │    │
│  └──────────────────────────────────────────────┘    │
│                    ↕                                   │
│  Integration Tests (Proxy Layer)                       │
│  ┌──────────────────────────────────────────────┐    │
│  │  • Agent communication                        │    │
│  │  • Fallback mechanisms                        │    │
│  │  • Configuration loading                      │    │
│  └──────────────────────────────────────────────┘    │
│                    ↕                                   │
│  Unit Tests (Transport Layer)                          │
│  ┌──────────────────────────────────────────────┐    │
│  │  • QUIC connections                           │    │
│  │  • Stream multiplexing                        │    │
│  │  • Connection migration                       │    │
│  └──────────────────────────────────────────────┘    │
└────────────────────────────────────────────────────────┘
```

## Mock Implementation

Tests use mock implementations of QUIC primitives:
- `MockQuicConnection`: Simulates QUIC connections with 0-RTT support
- `MockQuicSendStream`: Simulates send streams with priority and flow control
- `MockQuicRecvStream`: Simulates receive streams
- `MockQuicConnectionPool`: Simulates connection pooling and management

These mocks provide realistic timing and behavior based on QUIC research.

## Test Data

Tests use realistic scenarios:
- Agent types: researcher, coder, tester, reviewer, architect
- Task types: analyze, code, test, review, deploy
- Stream priorities: Critical (0), High (64), Medium (128), Low (192)
- Flow control limits: 1MB to 100MB based on priority

## Continuous Integration

Tests are designed to run in CI environments:
- Fast execution (< 5 minutes for full suite)
- Deterministic results
- No external dependencies
- Parallel test execution
- Comprehensive coverage reporting

## Future Enhancements

When QUIC implementation is available:
1. Replace mocks with actual QUIC client/server
2. Add performance regression tests
3. Add network condition simulation (latency, packet loss)
4. Add TLS certificate validation tests
5. Add connection migration stress tests
6. Add QLOG debugging tests

## Related Documentation

- [QUIC Research Document](../../agentic-flow/docs/plans/quic-research.md)
- [Implementation Roadmap](../../agentic-flow/docs/plans/quic-research.md#7-integration-roadmap)
- [Performance Projections](../../agentic-flow/docs/plans/quic-research.md#6-performance-projections)
