# AgentDB Alpha Package Validation

Docker-based testing environment for validating the published `agentdb@alpha` package.

## Quick Start

### Build and Run Tests

```bash
# Build the Docker image
docker build -f tests/docker/Dockerfile.alpha-test -t agentdb-alpha-test .

# Run validation script
docker run --rm agentdb-alpha-test bash tests/docker/validate-alpha.sh

# Interactive testing
docker run -it --rm agentdb-alpha-test bash
```

## What Gets Tested

### 1. Package Installation (4 tests)
- ✅ npx agentdb@alpha installation
- ✅ npm install agentdb@alpha
- ✅ Version verification (2.0.0-alpha.1)
- ✅ Package file integrity

### 2. CLI Commands (5 tests)
- ✅ Help command
- ✅ Init command
- ✅ Simulate list command
- ✅ Reflexion commands
- ✅ Skill commands

### 3. Programmatic Usage (3 tests)
- ✅ Node.js import
- ✅ AgentDB instance creation
- ✅ TypeScript declaration files

### 4. Simulations (3 tests)
- ✅ Scenario listing (8+ scenarios expected)
- ✅ Simulation wizard availability
- ✅ Scenario file presence

### 5. MCP Integration (2 tests)
- ✅ MCP command availability
- ✅ MCP integration files

### 6. Documentation (3 tests)
- ✅ README.md presence
- ✅ Quick Start section
- ✅ Tutorial section

### 7. Performance & Dependencies (3 tests)
- ✅ Package size check
- ✅ npm audit (security vulnerabilities)
- ✅ Peer dependencies

### 8. Backward Compatibility (1 test)
- ✅ v1.x API compatibility

## Test Output

The script generates:
1. **Console output**: Color-coded test results
2. **Report file**: `/tmp/alpha-validation-report.txt`

### Exit Codes
- `0`: All tests passed or only minor warnings
- `1`: Critical failures detected

## Manual Testing

```bash
# Start interactive shell
docker run -it --rm agentdb-alpha-test bash

# Test npx installation
npx agentdb@alpha --version

# List scenarios
npx agentdb@alpha simulate list

# Initialize test project
npx agentdb@alpha init --name test --dimensions 384

# Run simulation
npx agentdb@alpha simulate run hnsw --nodes 10000

# Test programmatic usage
node -e "const {AgentDB} = require('agentdb'); console.log(new AgentDB({name:'test',dimensions:384}))"
```

## CI/CD Integration

```yaml
# .github/workflows/alpha-validation.yml
name: Alpha Package Validation

on:
  release:
    types: [published]
  workflow_dispatch:

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Build test image
        run: docker build -f tests/docker/Dockerfile.alpha-test -t agentdb-alpha-test .

      - name: Run validation
        run: docker run --rm agentdb-alpha-test bash tests/docker/validate-alpha.sh

      - name: Upload report
        uses: actions/upload-artifact@v3
        with:
          name: validation-report
          path: /tmp/alpha-validation-report.txt
```

## Expected Results

### Passing Criteria
- **Critical**: 25+ tests passed
- **Warnings**: ≤5 warnings acceptable for alpha
- **Failures**: 0 critical failures

### Known Alpha Limitations
- Some analyzer/benchmark commands disabled (coming in beta)
- MCP tools require separate server setup
- Some advanced features may have warnings

## Troubleshooting

### "Package not found"
```bash
# Verify npm registry
npm view agentdb@alpha version

# Clear npm cache
npm cache clean --force
```

### "Permission denied"
```bash
# Run as non-root user (included in Dockerfile)
docker run --user $(id -u):$(id -g) ...
```

### "Import failed"
```bash
# Check Node.js version
node --version  # Should be 20.x

# Verify package installation
npm ls agentdb
```

## Report Issues

If validation fails, create an issue with:
1. Full validation report (`/tmp/alpha-validation-report.txt`)
2. Docker version (`docker --version`)
3. Steps to reproduce
4. Expected vs actual behavior
