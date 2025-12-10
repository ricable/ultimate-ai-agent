# Contributing to DAA Test Suite

Thank you for contributing to the DAA (Distributed Agentic Architecture) test suite!

## 📋 Test Writing Guidelines

### Test Structure

All tests should follow this structure:

```javascript
const { test } = require('node:test');
const assert = require('node:assert/strict');

test('Module: Feature description', (t) => {
  // Arrange - Set up test data
  const input = setupTestData();

  // Act - Execute the function
  const result = functionUnderTest(input);

  // Assert - Verify the result
  assert.equal(result, expected, 'Description of what should happen');
});
```

### Test Categories

#### 1. Unit Tests (`tests/unit/`)
- Test individual functions and classes
- Use mocks for external dependencies
- Fast execution (< 100ms per test)
- High coverage (>90%)

```javascript
test('MlKem768: Generate keypair', (t) => {
  const mlkem = new MlKem768();
  const { publicKey, secretKey } = mlkem.generateKeypair();

  assert.equal(publicKey.length, 1184);
  assert.equal(secretKey.length, 2400);
});
```

#### 2. Integration Tests (`tests/integration/`)
- Test interaction between modules
- Minimal mocking
- Focus on workflows
- Medium execution time (< 1s per test)

```javascript
test('Integration: Secure key exchange with vault', async (t) => {
  const mlkem = new MlKem768();
  const vault = new PasswordVault('password');

  const { publicKey, secretKey } = mlkem.generateKeypair();
  const { sharedSecret } = mlkem.encapsulate(publicKey);

  await vault.store('secret', sharedSecret.toString('hex'));
  const retrieved = await vault.retrieve('secret');

  assert.ok(retrieved);
});
```

#### 3. End-to-End Tests (`tests/e2e/`)
- Test complete user scenarios
- No mocking (use real bindings when available)
- Focus on real-world workflows
- Longer execution time acceptable (< 10s per test)

```javascript
test('E2E: Complete agent authentication flow', async (t) => {
  const daa = new DAA();
  await daa.init();

  const mlkem = daa.crypto.mlkem();
  const keys = mlkem.generateKeypair();

  const vault = daa.vault;
  await vault.store('identity', keys.secretKey.toString('hex'));

  const retrieved = await vault.retrieve('identity');
  assert.ok(retrieved);
});
```

#### 4. Benchmarks (`tests/benchmarks/`)
- Measure performance
- Compare native vs WASM
- Track regressions
- Output detailed metrics

```javascript
test('Benchmark: ML-KEM keypair generation', (t) => {
  const mlkem = new MlKem768();

  const stats = benchmark(() => {
    return mlkem.generateKeypair();
  }, 100);

  console.log(`Average: ${stats.avg.toFixed(3)}ms`);
  assert.ok(stats.avg < 5, 'Should be under 5ms');
});
```

### Naming Conventions

#### Test Files
- Unit tests: `module-name.test.js`
- Integration tests: `workflow-name.test.js`
- E2E tests: `scenario-name.test.js`
- Benchmarks: `operation-name.bench.js`

#### Test Names
Use the format: `Category: Description`

```javascript
// Good
test('MlKem768: Generate keypair with correct sizes', ...)
test('Integration: Multi-party key exchange', ...)
test('E2E: Complete DAA workflow from init to shutdown', ...)

// Bad
test('test1', ...)
test('it works', ...)
test('keypair', ...)
```

### Assertions

Always use descriptive assertion messages:

```javascript
// Good
assert.equal(result.length, 32, 'Shared secret should be 32 bytes');
assert.ok(vault.unlock('password'), 'Vault should unlock with correct password');

// Bad
assert.equal(result.length, 32);
assert.ok(vault.unlock('password'));
```

### Async Tests

Use `async` for asynchronous operations:

```javascript
test('Vault: Store and retrieve', async (t) => {
  const vault = new PasswordVault('password');

  await vault.store('key', 'value');
  const retrieved = await vault.retrieve('key');

  assert.equal(retrieved, 'value');
});
```

### Error Handling

Test both success and failure cases:

```javascript
test('MlKem768: Invalid key length throws error', (t) => {
  const mlkem = new MlKem768();
  const invalidKey = Buffer.alloc(100);

  assert.throws(
    () => mlkem.encapsulate(invalidKey),
    /Invalid public key length/,
    'Should throw on invalid key length'
  );
});
```

## 🧪 Using Test Utilities

The `tests/utils/` directory provides helpful utilities:

### test-helpers.js

```javascript
const {
  randomBuffer,
  createMockKeypair,
  measureTime,
  benchmark,
  createMockQuDAG
} = require('./utils/test-helpers');

// Generate random data
const data = randomBuffer(1024);

// Measure execution time
const { result, timeMs } = await measureTime(async () => {
  return await someOperation();
});

// Run benchmarks
const stats = benchmark(() => operation(), 100);
```

### mock-loader.js

```javascript
const { loadQuDAG, detectAvailableBindings } = require('./utils/mock-loader');

// Load native or WASM bindings (or fall back to mock)
const qudag = await loadQuDAG();

// Check what's available
const available = detectAvailableBindings();
if (available.native) {
  // Use native bindings
}
```

## 📊 Code Coverage

### Running Coverage

```bash
# Generate coverage report
npm run test:coverage

# Generate HTML report
npm run test:coverage:html
```

### Coverage Targets

- Lines: 90%
- Functions: 90%
- Branches: 85%
- Statements: 90%

### Improving Coverage

1. Check coverage report: `coverage/index.html`
2. Identify uncovered lines
3. Add tests for uncovered code
4. Test edge cases and error paths

## 🚀 Performance Testing

### Writing Benchmarks

```javascript
const { benchmark } = require('./utils/test-helpers');

test('Benchmark: Operation name', (t) => {
  const stats = benchmark(() => {
    // Operation to benchmark
    return performOperation();
  }, 100); // 100 iterations

  console.log(`\nPerformance Metrics:`);
  console.log(`  Average: ${stats.avg.toFixed(3)}ms`);
  console.log(`  Median:  ${stats.median.toFixed(3)}ms`);
  console.log(`  Min:     ${stats.min.toFixed(3)}ms`);
  console.log(`  Max:     ${stats.max.toFixed(3)}ms`);
  console.log(`  P95:     ${stats.p95.toFixed(3)}ms`);

  // Assert performance target
  assert.ok(stats.avg < targetMs, 'Should meet performance target');
});
```

### Performance Targets

Refer to `tests/README.md` for specific performance targets for each operation.

## 🔍 Debugging Tests

### Run Single Test

```bash
node --test tests/unit/qudag-crypto.test.js
```

### Verbose Output

```bash
NODE_OPTIONS="--test-reporter=spec" node --test tests/unit/qudag-crypto.test.js
```

### Debug with Chrome DevTools

```bash
node --inspect-brk --test tests/unit/qudag-crypto.test.js
# Open chrome://inspect in Chrome
```

### Add Debug Logging

```javascript
test('Feature with debug logging', (t) => {
  console.log('Test input:', input);

  const result = operation(input);

  console.log('Test result:', result);
  assert.ok(result);
});
```

## 📝 Pull Request Checklist

Before submitting a PR with tests:

- [ ] All tests pass locally
- [ ] Code coverage meets targets (>90%)
- [ ] Test names are descriptive
- [ ] Tests follow project structure
- [ ] Performance benchmarks included for new features
- [ ] Both positive and negative cases tested
- [ ] Documentation updated if needed
- [ ] No console.log statements (except in benchmarks)
- [ ] Tests are isolated and don't depend on execution order

## 🎯 Best Practices

### Do's

✅ Test one thing per test
✅ Use descriptive test names
✅ Test edge cases and error conditions
✅ Keep tests fast and isolated
✅ Use mocks appropriately
✅ Add performance benchmarks for critical paths
✅ Update tests when functionality changes

### Don'ts

❌ Don't test implementation details
❌ Don't make tests depend on each other
❌ Don't use setTimeout or fixed delays
❌ Don't skip tests without good reason
❌ Don't commit commented-out tests
❌ Don't test external APIs directly (use mocks)
❌ Don't write tests that are slower than necessary

## 📚 Resources

- [Node.js Test Runner Docs](https://nodejs.org/api/test.html)
- [c8 Coverage Tool](https://github.com/bcoe/c8)
- [Test Structure Best Practices](https://kentcdodds.com/blog/common-testing-mistakes)
- [DAA Test Suite README](./README.md)

## 🤝 Getting Help

If you need help writing tests:

1. Check existing tests for examples
2. Review this contributing guide
3. Look at test utilities in `tests/utils/`
4. Open an issue on GitHub
5. Ask in the project Discord

## 📄 License

All test code is licensed under MIT, same as the DAA project.
