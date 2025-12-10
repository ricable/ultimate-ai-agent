# Test Suite Installation Guide

## Prerequisites

- Node.js 18+ (LTS)
- npm or yarn

## Installation

### 1. Install Test Dependencies

```bash
cd tests
npm install
```

This will install:
- `c8` - Code coverage tool
- `eslint` - Linting
- `prettier` - Code formatting

### 2. Verify Installation

```bash
# Run a simple test
node --test unit/qudag-crypto.test.js

# Should output test results
```

### 3. Run Full Test Suite

```bash
# All tests
npm test

# With coverage
npm run test:coverage
```

## Usage

### Run Tests by Category

```bash
# Unit tests only
npm run test:unit

# Integration tests
npm run test:integration

# End-to-end tests
npm run test:e2e

# Performance benchmarks
npm run benchmark
```

### Watch Mode

```bash
npm run test:watch
```

### Coverage Report

```bash
# Generate coverage report
npm run test:coverage

# Open HTML report
open coverage/index.html  # macOS
xdg-open coverage/index.html  # Linux
start coverage/index.html  # Windows
```

## Configuration

All configuration is in the `tests/` directory:
- `.c8rc.json` - Coverage configuration
- `test-runner.config.js` - Test runner settings
- `package.json` - Scripts and dependencies

## Troubleshooting

### Tests Not Running

```bash
# Make sure you're in the tests directory
cd tests

# Verify Node.js version
node --version  # Should be 18+
```

### Coverage Tool Not Found

```bash
# Install c8 globally
npm install -g c8

# Or install locally
npm install
```

### Module Not Found Errors

The tests use mocks by default. When NAPI bindings are built:

```bash
# Build native bindings first
cd ../qudag/qudag-napi
npm run build

# Then run tests
cd ../../tests
npm test
```

## CI/CD Integration

### GitHub Actions

```yaml
- name: Run tests
  run: |
    cd tests
    npm install
    npm run test:coverage
```

### GitLab CI

```yaml
test:
  script:
    - cd tests
    - npm install
    - npm run test:coverage
  coverage: '/Lines\s*:\s*(\d+\.\d+)%/'
```

## Documentation

- [README.md](./README.md) - Full test documentation
- [CONTRIBUTING.md](./CONTRIBUTING.md) - Contribution guidelines
- [TEST_SUITE_SUMMARY.md](./TEST_SUITE_SUMMARY.md) - Implementation summary

## Support

For issues or questions:
1. Check the documentation
2. Review existing tests for examples
3. Open an issue on GitHub
