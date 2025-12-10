# Attention Mechanism Testing Guide

## Overview

This document describes the comprehensive testing strategy for AgentDB's attention mechanism integration, including how to run tests, interpret results, and contribute new tests.

## Table of Contents

1. [Test Suites](#test-suites)
2. [Running Tests](#running-tests)
3. [Test Coverage](#test-coverage)
4. [Performance Benchmarks](#performance-benchmarks)
5. [Browser Tests](#browser-tests)
6. [CI/CD Integration](#cicd-integration)
7. [Writing New Tests](#writing-new-tests)

## Test Suites

### 1. Integration Tests

**Location**: `tests/integration/attention-integration.test.ts`

**Coverage**:
- Self-attention mechanisms
- Cross-attention mechanisms
- Multi-head attention
- Memory controller integrations
- CLI commands
- MCP tools
- Browser WASM loading

**Run Command**:
```bash
cd packages/agentdb
npx vitest tests/integration/attention-integration.test.ts --run
```

**Expected Results**:
- All user-facing APIs tested
- End-to-end workflows validated
- Integration with existing AgentDB features verified

### 2. Regression Tests

**Location**: `tests/regression/attention-regression.test.ts`

**Coverage**:
- Backward compatibility (attention disabled)
- Feature flag behavior (attention enabled)
- API stability
- Performance regression checks
- Database migration
- Error handling stability

**Run Command**:
```bash
cd packages/agentdb
npx vitest tests/regression/attention-regression.test.ts --run
```

**Expected Results**:
- All existing functionality unchanged
- No breaking changes detected
- Performance within acceptable range (<2x degradation)

### 3. Performance Benchmarks

**Location**: `benchmarks/attention/attention-benchmarks.ts`

**Coverage**:
- Throughput (queries/second)
- Latency (P50, P95, P99)
- Memory usage
- NAPI vs WASM comparison
- Scalability tests
- Concurrency tests

**Run Command**:
```bash
cd packages/agentdb
tsx benchmarks/attention/attention-benchmarks.ts
```

**Expected Results**:
- Throughput: >100 queries/second (1000 items)
- Latency P95: <100ms
- Memory usage: <50MB for 1000 items
- Scalability: Linear or better

### 4. Browser Tests

**Location**: `tests/browser/attention-browser.test.js`

**Coverage**:
- WASM module loading
- Lazy loading behavior
- Fallback mechanisms
- Cross-browser compatibility (Chrome, Firefox, Safari)
- Progressive enhancement
- Bundle size optimization

**Run Command**:
```bash
cd packages/agentdb
npx playwright test tests/browser/attention-browser.test.js
```

**Expected Results**:
- Works in all major browsers
- WASM loads successfully
- Fallback to JavaScript when WASM unavailable
- Bundle size <100KB (with lazy loading)

## Running Tests

### Quick Start

Run all tests:
```bash
npm test
```

Run specific test suites:
```bash
# Integration tests only
npm run test:integration

# Regression tests only
npm run test:regression

# Browser tests only
npm run test:browser

# Performance benchmarks
npm run benchmark:attention
```

### With Coverage

Generate coverage reports:
```bash
npm run test:coverage
```

View coverage report:
```bash
open coverage/index.html
```

### Development Mode

Watch mode for TDD:
```bash
npx vitest tests/integration/attention-integration.test.ts
```

### Environment Variables

Configure test behavior:
```bash
# Enable garbage collection for memory tests
NODE_OPTIONS=--expose-gc npm test

# Increase memory limit for large datasets
NODE_OPTIONS=--max-old-space-size=4096 npm test

# Enable/disable attention features
AGENTDB_ATTENTION_ENABLED=true npm test
```

## Test Coverage

### Coverage Requirements

- **Statements**: >85%
- **Branches**: >75%
- **Functions**: >85%
- **Lines**: >85%

### Coverage Reports

After running tests with coverage:

```bash
npm run test:coverage
```

View detailed coverage:
- **Terminal**: Inline summary after tests
- **HTML**: `coverage/index.html`
- **JSON**: `coverage/coverage-summary.json`
- **LCOV**: `coverage/lcov.info` (for CI tools)

### Uncovered Code

Intentionally excluded from coverage:
- Test files (`**/*.test.ts`)
- Benchmark files (`benchmarks/**`)
- Type definitions (`**/*.d.ts`)
- Build scripts (`scripts/**`)
- Fallback implementations (`**/db-fallback.ts`)

## Performance Benchmarks

### Benchmark Suite

The attention benchmarks test:

1. **Self-Attention Performance**
   - Data sizes: 100, 500, 1000, 5000 items
   - Query counts: 100, 200, 500
   - Embedding dimensions: 128, 256, 512

2. **Cross-Attention Performance**
   - Context sizes: 100, 500, 1000
   - Query patterns: Single, batch, concurrent

3. **Multi-Head Attention Performance**
   - Head counts: 4, 8, 16
   - Aggregation strategies: Average, max, concat

4. **Scalability Tests**
   - Dataset growth: 100 → 5000 items
   - Expected: Linear or sub-linear scaling

5. **Concurrency Tests**
   - Concurrency levels: 1, 5, 10, 20, 50
   - Expected: Linear throughput scaling

### Running Benchmarks

Run all benchmarks:
```bash
npm run benchmark:attention
```

Results are saved to:
```
benchmarks/attention/benchmark-results.json
```

### Interpreting Results

Example output:
```json
{
  "name": "Self-Attention (data=1000, queries=200, dim=128)",
  "throughput": 156.3,
  "latency": {
    "p50": 45.2,
    "p95": 87.6,
    "p99": 124.8,
    "mean": 52.1
  },
  "memory": {
    "initial": 28.5,
    "peak": 42.3,
    "final": 30.1
  },
  "duration": 1.28
}
```

**Key Metrics**:
- **Throughput**: Queries per second (higher is better)
- **Latency P95**: 95th percentile latency in ms (lower is better)
- **Memory Peak**: Maximum memory usage in MB (lower is better)
- **Duration**: Total benchmark time in seconds

### Baseline Comparison

Compare with baseline:
```bash
# Save current results as baseline
cp benchmarks/attention/benchmark-results.json \
   benchmarks/attention/benchmark-baseline.json

# Future runs will compare against this baseline
```

CI/CD automatically fails if:
- Throughput drops >20%
- Latency P95 increases >50%
- Memory usage increases >100%

## Browser Tests

### Supported Browsers

- **Chrome**: Latest stable
- **Firefox**: Latest stable
- **Safari**: Latest stable
- **Edge**: Latest stable

### Test Scenarios

1. **WASM Loading**
   - Module initialization
   - Lazy loading
   - Error handling

2. **Fallback Behavior**
   - JavaScript fallback when WASM unavailable
   - Partial WASM support
   - Feature detection

3. **Performance**
   - Query throughput in browser
   - Memory efficiency
   - Bundle size impact

4. **Compatibility**
   - IndexedDB persistence
   - Web Workers
   - Progressive enhancement

### Running Browser Tests

Local testing:
```bash
npx playwright test tests/browser/attention-browser.test.js
```

Specific browser:
```bash
npx playwright test --browser=chromium
npx playwright test --browser=firefox
npx playwright test --browser=webkit
```

Debug mode:
```bash
npx playwright test --debug
```

## CI/CD Integration

### GitHub Actions Workflows

**Workflow**: `.github/workflows/test-agentdb-attention.yml`

**Jobs**:

1. **test-attention-integration**
   - Runs on: Ubuntu, macOS, Windows
   - Node versions: 18.x, 20.x, 22.x
   - Tests: Integration test suite

2. **test-attention-regression**
   - Tests: Regression test suite
   - Configurations: Attention enabled/disabled

3. **test-attention-performance**
   - Runs: Performance benchmarks
   - Compares: Against baseline
   - Fails if: Performance degrades >20%

4. **test-browser-attention**
   - Browsers: Chromium, Firefox, WebKit
   - Tests: Browser compatibility

5. **test-coverage-attention**
   - Generates: Coverage reports
   - Enforces: Coverage thresholds
   - Posts: Coverage comment on PRs

### Workflow Triggers

Runs on:
- Push to `main` or `mcp-dev` branches
- Pull requests to `main`
- Changes in `packages/agentdb/**`

### Viewing Results

1. **GitHub Actions Tab**: See all workflow runs
2. **PR Checks**: See test status in PR
3. **Artifacts**: Download test results and coverage reports
4. **PR Comments**: Coverage metrics automatically commented

## Writing New Tests

### Test Structure

Follow the existing pattern:

```typescript
import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { AgentDB } from '../../src/index';

describe('Feature Name', () => {
  let db: AgentDB;

  beforeEach(async () => {
    db = new AgentDB({ /* config */ });
    await db.initialize();
  });

  afterEach(async () => {
    await db.close();
    // Cleanup
  });

  it('should do something specific', async () => {
    // Arrange
    const input = /* setup */;

    // Act
    const result = await db.someMethod(input);

    // Assert
    expect(result).toBeDefined();
    expect(result).toHaveProperty('expected');
  });
});
```

### Test Documentation

Include JSDoc comments:

```typescript
/**
 * @test Feature Name
 * @description What this test validates
 * @prerequisites
 *   - Required setup
 *   - Dependencies
 * @steps
 *   1. Setup
 *   2. Execute
 *   3. Verify
 * @expected Expected outcome
 */
```

### Best Practices

1. **Test Isolation**: Each test should be independent
2. **Cleanup**: Always cleanup resources in `afterEach`
3. **Descriptive Names**: Use clear test names
4. **One Assertion**: Focus on one behavior per test
5. **Edge Cases**: Test boundaries, errors, empty states
6. **Performance**: Include timing assertions where relevant

### Adding to CI/CD

1. Create test file in appropriate directory
2. Follow naming convention: `*.test.ts` or `*.test.js`
3. Tests are automatically discovered by Vitest
4. No CI/CD changes needed unless:
   - New test suite requires special setup
   - New dependencies needed
   - New browser features required

### Performance Test Guidelines

When adding benchmarks:

```typescript
it('should perform operation efficiently', async () => {
  const start = performance.now();

  // Operation
  await someOperation();

  const duration = performance.now() - start;

  expect(duration).toBeLessThan(100); // ms
});
```

Memory tests:
```typescript
it('should manage memory efficiently', async () => {
  const initial = process.memoryUsage().heapUsed;

  // Operation
  await largeOperation();

  global.gc && global.gc();
  const final = process.memoryUsage().heapUsed;
  const increase = (final - initial) / (1024 * 1024);

  expect(increase).toBeLessThan(50); // MB
});
```

## Troubleshooting

### Common Issues

**Tests fail with "out of memory"**:
```bash
NODE_OPTIONS=--max-old-space-size=4096 npm test
```

**Browser tests fail to launch**:
```bash
npx playwright install --with-deps
```

**Coverage thresholds not met**:
- Check `coverage/index.html` for uncovered lines
- Add tests for untested code paths
- Update thresholds if intentional (requires justification)

**Performance benchmarks timeout**:
- Increase timeout in test configuration
- Check for memory leaks
- Optimize test data size

### Getting Help

1. Check existing issues: https://github.com/ruvnet/agentic-flow/issues
2. Review test output and error messages
3. Run tests in debug mode: `npx vitest --debug`
4. Ask in discussions: https://github.com/ruvnet/agentic-flow/discussions

## Contributing

When contributing new tests:

1. Follow existing test patterns
2. Ensure all tests pass locally
3. Update this documentation if adding new test categories
4. Include test coverage for new features
5. Add performance benchmarks for performance-critical code

## References

- [Vitest Documentation](https://vitest.dev/)
- [Playwright Documentation](https://playwright.dev/)
- [AgentDB Documentation](../../README.md)
- [CI/CD Workflows](../../../.github/workflows/)
