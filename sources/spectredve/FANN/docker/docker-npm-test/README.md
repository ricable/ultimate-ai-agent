# Docker NPM Testing Environment for ruv-swarm

This directory contains a complete Docker-based testing environment for validating the ruv-swarm npm package installation in clean, isolated environments.

## 🎯 Purpose

- Test npm package installation in clean environments
- Validate compatibility across different Node.js versions (18, 20, 22)
- Ensure all package features work correctly after installation
- Identify any missing dependencies or configuration issues

## 📁 Structure

```
docker-npm-test/
├── Dockerfile              # Multi-stage Docker build file
├── docker-compose.yml      # Orchestration for multiple test scenarios
├── build-and-test.sh      # Build and test automation script
├── validate-npm-install.js # Comprehensive validation script
└── README.md              # This file
```

## 🚀 Quick Start

### Test NPM Registry Package

```bash
# Run basic test
./build-and-test.sh test

# Run all test configurations
./build-and-test.sh test-all
```

### Test Local Package Build

```bash
# Test the local .tgz package
./build-and-test.sh test-local
```

### Generate Test Report

```bash
# Run validation and generate report
./build-and-test.sh report
```

## 🔧 Available Commands

### build-and-test.sh Commands

- `build [target]` - Build Docker image for specific target
- `test [target]` - Run tests for specific target
- `test-all` - Run all test configurations with docker-compose
- `test-local` - Test local package build
- `interactive` - Start interactive shell in container
- `validate` - Run validation script
- `clean` - Clean up Docker resources
- `report` - Generate comprehensive test report

### Docker Targets

- `npm-test` - Test installation from NPM registry
- `local-test` - Test installation from local .tgz file
- `comprehensive-test` - Run full test suite with frameworks

## 📋 Test Coverage

The validation script checks:

1. ✅ Package installation success
2. ✅ Main entry point accessibility
3. ✅ Exported classes (RuvSwarm, Agent, Neural)
4. ✅ CLI command availability
5. ✅ WASM file presence
6. ✅ Swarm instance creation
7. ✅ MCP tools availability
8. ✅ Package metadata integrity
9. ✅ Executable scripts presence
10. ✅ MCP server functionality

## 🐳 Docker Compose Services

- `node18-test` - Test with Node.js 18 LTS
- `node20-test` - Test with Node.js 20 LTS
- `node22-test` - Test with Node.js 22 Latest
- `comprehensive-test` - Full test suite with Jest/Mocha
- `local-test` - Test local package builds

## 📊 Test Reports

Test results are saved to:
- `test-results/test-output.log` - Raw test output
- `test-results/validation-report.json` - Structured validation report

## 🔍 Debugging

### Interactive Shell

```bash
# Start interactive shell for debugging
./build-and-test.sh interactive
```

### View Logs

```bash
# View docker-compose logs
docker-compose logs -f
```

### Check Container Status

```bash
# List running containers
docker ps

# Inspect specific container
docker inspect ruv-swarm-node20-test
```

## ⚠️ Common Issues

### WASM Loading Errors
- The package includes WebAssembly files that may require specific Node.js flags
- Use `NODE_OPTIONS="--experimental-wasm-modules"` if needed

### Permission Issues
- The Dockerfile creates a non-root user to avoid permission problems
- Volume mounts should use appropriate permissions

### Network Issues
- Ensure Docker can access npm registry
- Use `--network host` if behind proxy

## 🛠️ Customization

### Testing Different Versions

Edit `docker-compose.yml` to test specific package versions:

```yaml
command: |
  sh -c '
    npm install ruv-swarm@1.0.5  # Specific version
    npm test
  '
```

### Adding Test Cases

Add custom tests to `validate-npm-install.js` or create new test files in the container.

## 📈 Performance Notes

- Docker layer caching speeds up subsequent builds
- Use `--no-cache` flag to force fresh builds
- Multi-stage builds minimize final image size

## 🤝 Integration with CI/CD

This setup can be integrated into CI/CD pipelines:

```yaml
# Example GitHub Actions
- name: Test NPM Package
  run: |
    cd docker-npm-test
    ./build-and-test.sh test-all
```

## 📝 License

This testing environment is part of the ruv-swarm project.