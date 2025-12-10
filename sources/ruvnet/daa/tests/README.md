# DAA NAPI Bindings Test Suite

Comprehensive test suite for DAA (Distributed Agentic Architecture) NAPI-rs bindings and SDK functionality.

## 📁 Test Structure

```
tests/
├── unit/                          # Unit tests for individual modules
│   ├── qudag-crypto.test.js      # ML-KEM-768, ML-DSA, BLAKE3 tests
│   ├── qudag-vault.test.js       # Password vault tests
│   ├── qudag-exchange.test.js    # rUv token exchange tests
│   └── sdk-platform-detection.test.js  # Platform detection tests
├── integration/                   # Integration tests
│   ├── qudag-full-workflow.test.js     # Complete QuDAG workflows
│   └── platform-comparison.test.js     # Native vs WASM parity
├── e2e/                          # End-to-end tests
│   └── full-daa-workflow.test.js # Complete DAA system tests
├── benchmarks/                    # Performance benchmarks
│   └── crypto-performance.bench.js     # Crypto operation benchmarks
├── utils/                        # Test utilities
│   ├── test-helpers.js          # Common test utilities
│   └── mock-loader.js           # Dynamic module loading
└── README.md                     # This file
```

## 🚀 Running Tests

### Run All Tests

```bash
# Run all tests with Node.js built-in test runner
npm test

# Or directly with node
node --test tests/**/*.test.js
```

### Run Specific Test Suites

```bash
# Unit tests only
node --test tests/unit/**/*.test.js

# Integration tests
node --test tests/integration/**/*.test.js

# End-to-end tests
node --test tests/e2e/**/*.test.js

# Benchmarks
node --test tests/benchmarks/**/*.bench.js
```

### Run with Coverage

```bash
# Install c8 for coverage
npm install --save-dev c8

# Run tests with coverage
npm run test:coverage

# Or directly
npx c8 node --test tests/**/*.test.js
```

## 📊 Test Coverage

### Current Coverage Targets

- **Unit Tests**: >90% code coverage
- **Integration Tests**: All major workflows
- **E2E Tests**: Complete user scenarios
- **Performance**: Baseline benchmarks established

### Coverage Areas

#### QuDAG Crypto Operations
- ✅ ML-KEM-768 key generation
- ✅ ML-KEM-768 encapsulation
- ✅ ML-KEM-768 decapsulation
- ✅ ML-DSA signing
- ✅ ML-DSA verification
- ✅ BLAKE3 hashing
- ✅ Quantum fingerprinting
- ✅ Error handling and validation

#### Password Vault
- ✅ Vault creation with master password
- ✅ Unlock with correct/incorrect password
- ✅ Store and retrieve values
- ✅ Delete operations
- ✅ List all keys
- ✅ Multiple vault instances

#### rUv Token Exchange
- ✅ Transaction creation
- ✅ Transaction signing with ML-DSA
- ✅ Transaction verification
- ✅ Transaction submission
- ✅ Batch transaction processing
- ✅ High-volume scenarios

#### Platform Detection & SDK
- ✅ Node.js environment detection
- ✅ Platform information retrieval
- ✅ Native vs WASM loading
- ✅ Feature parity verification

#### Integration Workflows
- ✅ Secure key exchange with vault storage
- ✅ End-to-end secure transactions
- ✅ Multi-party key exchange
- ✅ Vault-backed transaction signing
- ✅ Hybrid encryption workflows
- ✅ Key rotation procedures

#### End-to-End Scenarios
- ✅ Complete DAA initialization
- ✅ Agent authentication flows
- ✅ Orchestrator lifecycle management
- ✅ Multi-agent coordination
- ✅ Distributed agent networks
- ✅ Fault tolerance and recovery

#### Performance Benchmarks
- ✅ ML-KEM-768 operations (keygen, encap, decap)
- ✅ ML-DSA operations (sign, verify)
- ✅ BLAKE3 hashing (various data sizes)
- ✅ End-to-end workflows
- ✅ Throughput measurements

## 🎯 Performance Targets

### Expected Performance (Native NAPI-rs)

| Operation | Target | WASM Baseline | Speedup |
|-----------|--------|---------------|---------|
| ML-KEM Keygen | 1.8ms | 5.2ms | 2.9x |
| ML-KEM Encapsulate | 1.1ms | 3.1ms | 2.8x |
| ML-KEM Decapsulate | 1.3ms | 3.8ms | 2.9x |
| ML-DSA Sign | 1.5ms | 4.5ms | 3.0x |
| ML-DSA Verify | 1.3ms | 3.8ms | 2.9x |
| BLAKE3 (1MB) | 2.1ms | 8.2ms | 3.9x |

### Benchmark Execution

```bash
# Run performance benchmarks
node --test tests/benchmarks/crypto-performance.bench.js

# With detailed output
NODE_OPTIONS="--test-reporter=spec" node --test tests/benchmarks/
```

## 🧪 Test Utilities

### Test Helpers (`tests/utils/test-helpers.js`)

```javascript
const {
  randomBuffer,
  createMockKeypair,
  measureTime,
  benchmark,
  createMockQuDAG,
  createMockVault,
  createMockExchange
} = require('./utils/test-helpers');

// Use in tests
const keypair = createMockKeypair();
const stats = benchmark(() => someFunction(), 100);
```

### Mock Loader (`tests/utils/mock-loader.js`)

```javascript
const { loadQuDAG, detectAvailableBindings } = require('./utils/mock-loader');

// Dynamically load native or WASM bindings
const qudag = await loadQuDAG();

// Check what's available
const available = detectAvailableBindings();
```

## 📝 Writing New Tests

### Unit Test Template

```javascript
const { test } = require('node:test');
const assert = require('node:assert/strict');
const { createMockQuDAG } = require('../utils/test-helpers');

test('Feature: Description', (t) => {
  const qudag = createMockQuDAG();

  // Arrange
  const input = /* setup test data */;

  // Act
  const result = /* call function */;

  // Assert
  assert.equal(result, expected, 'Description of assertion');
});
```

### Integration Test Template

```javascript
const { test } = require('node:test');
const assert = require('node:assert/strict');

test('Integration: Multi-step workflow', async (t) => {
  // Step 1
  const step1Result = await doFirstThing();
  assert.ok(step1Result, 'Step 1 should complete');

  // Step 2
  const step2Result = await doSecondThing(step1Result);
  assert.ok(step2Result, 'Step 2 should complete');

  // Verify end result
  assert.equal(finalState, expected, 'Final state should match');
});
```

### Benchmark Template

```javascript
const { test } = require('node:test');
const { benchmark } = require('../utils/test-helpers');

test('Benchmark: Operation name', (t) => {
  const stats = benchmark(() => {
    // Operation to benchmark
    return performOperation();
  }, 100); // 100 iterations

  console.log(`Average: ${stats.avg.toFixed(3)}ms`);
  console.log(`Median: ${stats.median.toFixed(3)}ms`);

  assert.ok(stats.avg < targetMs, 'Should meet performance target');
});
```

## 🔧 Test Configuration

### package.json Scripts

Add these scripts to your `package.json`:

```json
{
  "scripts": {
    "test": "node --test tests/**/*.test.js",
    "test:unit": "node --test tests/unit/**/*.test.js",
    "test:integration": "node --test tests/integration/**/*.test.js",
    "test:e2e": "node --test tests/e2e/**/*.test.js",
    "test:bench": "node --test tests/benchmarks/**/*.bench.js",
    "test:coverage": "c8 node --test tests/**/*.test.js",
    "test:watch": "node --test --watch tests/**/*.test.js"
  }
}
```

### Coverage Configuration (.c8rc.json)

```json
{
  "all": true,
  "include": [
    "qudag/qudag-napi/src/**/*.rs",
    "packages/daa-sdk/src/**/*.ts",
    "packages/daa-sdk/src/**/*.js"
  ],
  "exclude": [
    "tests/**",
    "node_modules/**",
    "**/dist/**"
  ],
  "reporter": ["text", "html", "lcov"],
  "check-coverage": true,
  "lines": 90,
  "functions": 90,
  "branches": 85
}
```

## 🐛 Debugging Tests

### Run Single Test File

```bash
node --test tests/unit/qudag-crypto.test.js
```

### Run with Verbose Output

```bash
NODE_OPTIONS="--test-reporter=spec" node --test tests/unit/qudag-crypto.test.js
```

### Debug with Chrome DevTools

```bash
node --inspect-brk --test tests/unit/qudag-crypto.test.js
# Then open chrome://inspect
```

## 📈 Continuous Integration

### GitHub Actions Example

```yaml
name: Test Suite

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: 20

      - name: Install dependencies
        run: npm install

      - name: Run tests
        run: npm test

      - name: Run coverage
        run: npm run test:coverage

      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          files: ./coverage/lcov.info
```

## 📚 Additional Resources

- [Node.js Test Runner Docs](https://nodejs.org/api/test.html)
- [NAPI-rs Documentation](https://napi.rs/)
- [QuDAG Documentation](../qudag/README.md)
- [DAA SDK Documentation](../packages/daa-sdk/README.md)
- [NAPI Integration Plan](../docs/napi-rs-integration-plan.md)

## 🤝 Contributing

When adding new tests:

1. Follow the existing test structure
2. Use descriptive test names
3. Include both positive and negative test cases
4. Add performance benchmarks for critical paths
5. Update this README if adding new test categories
6. Ensure >90% code coverage for new features

## 📄 License

MIT License - Same as DAA ecosystem
