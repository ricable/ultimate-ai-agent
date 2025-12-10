# How to Run Agent Booster Tests

## Prerequisites

### For Rust Tests
```bash
# Install Rust (if not already installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Verify installation
rustc --version
cargo --version
```

### For Node.js Tests
```bash
# Verify Node.js installation (requires 14+)
node --version

# Install mocha globally (optional)
npm install -g mocha
```

## Running Tests

### 1. Rust Unit Tests

```bash
cd /workspaces/agentic-flow/agent-booster

# Run all tests
cargo test --all

# Run specific test file
cargo test --test parser_tests
cargo test --test similarity_tests
cargo test --test merge_tests

# Run specific test function
cargo test test_parse_javascript

# Run with verbose output
cargo test -- --nocapture

# Run in release mode (faster, but less debug info)
cargo test --release
```

### 2. Rust Integration Tests

```bash
cd /workspaces/agentic-flow/agent-booster

# Run integration tests
cargo test --test complete_flow_test

# Run with fixtures
cargo test test_end_to_end
```

### 3. Node.js Integration Tests

```bash
cd /workspaces/agentic-flow/agent-booster

# Run with Node.js directly
node tests/integration/npm_integration_test.js

# Or with Mocha
npx mocha tests/integration/npm_integration_test.js

# With verbose output
npx mocha tests/integration/npm_integration_test.js --reporter spec
```

### 4. NPM SDK Tests

```bash
cd /workspaces/agentic-flow/agent-booster/npm/agent-booster

# Install dependencies
npm install

# Run tests (if package.json has test script)
npm test

# Or run the test file directly
node ../../tests/integration/npm_integration_test.js
```

## Quick Test Commands

### Run Everything
```bash
cd /workspaces/agentic-flow/agent-booster

# Rust tests
cargo test --all

# Node.js tests
node tests/integration/npm_integration_test.js
```

### Run Specific Categories

```bash
# Parser tests only
cargo test parser

# Similarity tests only
cargo test similarity

# Merge tests only
cargo test merge

# Integration tests only
cargo test --test complete_flow_test
```

## Test Output Examples

### Successful Test Run
```
running 58 tests
test tests::test_simple_function_replacement ... ok
test tests::test_class_method_edit ... ok
test tests::test_typescript_interface ... ok
...
test result: ok. 58 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
```

### Failed Test
```
test tests::test_parse_javascript ... FAILED

failures:
    tests::test_parse_javascript

test result: FAILED. 57 passed; 1 failed; 0 ignored
```

## Troubleshooting

### Rust Tests Won't Run

**Problem:** `cargo: command not found`
```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```

**Problem:** `error: could not compile`
```bash
# Clean and rebuild
cargo clean
cargo build
cargo test
```

### Node.js Tests Won't Run

**Problem:** `Cannot find module 'agent-booster'`
```bash
# Build the native addon or WASM first
cd npm/agent-booster
npm install
npm run build  # if build script exists
```

**Problem:** `TypeError: AgentBooster.applyEdit is not a function`
```bash
# Verify the module structure
cd npm/agent-booster
node -e "console.log(require('./index.js'))"
```

### Fixture Files Not Found

```bash
# Verify fixtures exist
ls -la tests/fixtures/

# If missing, they should contain:
# - sample_javascript.js
# - sample_typescript.ts
```

## Test Coverage

### Generate Coverage Report (Rust)

```bash
# Install cargo-tarpaulin
cargo install cargo-tarpaulin

# Generate coverage
cargo tarpaulin --out Html --output-dir coverage

# Open coverage report
open coverage/index.html
```

### Expected Coverage
- Statements: >85%
- Branches: >80%
- Functions: >90%

## Continuous Integration

### GitHub Actions Example

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2

      - name: Setup Rust
        uses: actions-rs/toolchain@v1
        with:
          toolchain: stable

      - name: Run Rust tests
        run: cargo test --all

      - name: Setup Node.js
        uses: actions/setup-node@v2
        with:
          node-version: '18'

      - name: Run Node.js tests
        run: node tests/integration/npm_integration_test.js
```

## Performance Testing

### Benchmark Tests

```bash
# Run with timing
cargo test -- --nocapture | grep "test result"

# Profile specific test
cargo test test_performance_benchmark -- --nocapture
```

### Expected Performance
- Unit tests: <100ms each
- Integration tests: <1s each
- Full test suite: <30s

## Debugging Tests

### Rust Debugging

```bash
# Run single test with backtrace
RUST_BACKTRACE=1 cargo test test_name -- --nocapture

# Run with debug logging
RUST_LOG=debug cargo test
```

### Node.js Debugging

```bash
# Run with Node debugger
node --inspect-brk tests/integration/npm_integration_test.js

# Run with console output
node tests/integration/npm_integration_test.js --verbose
```

## Test Maintenance

### Adding New Tests

1. Create test file in appropriate directory
2. Follow naming convention: `*_tests.rs` or `*_test.js`
3. Run tests to verify
4. Update documentation

### Updating Fixtures

1. Edit files in `tests/fixtures/`
2. Ensure they're realistic examples
3. Run integration tests to verify

## Getting Help

If tests fail unexpectedly:

1. Check the error message carefully
2. Verify all dependencies are installed
3. Ensure fixtures are present
4. Try cleaning and rebuilding
5. Check for environment-specific issues

For persistent issues, refer to:
- TEST_PLAN.md for test structure
- README.md for project overview
- GitHub issues for known problems
