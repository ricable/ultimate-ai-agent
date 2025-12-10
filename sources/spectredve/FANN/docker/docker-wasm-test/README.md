# Docker WASM Test Suite for ruv-swarm

This comprehensive test suite verifies that the ruv-swarm npm package correctly loads and uses WebAssembly (WASM) modules without falling back to placeholder functionality.

## 🎯 Test Objectives

1. **Verify WASM Loading**: Ensure the actual WASM binary is loaded, not a placeholder
2. **Test NPX Integration**: Confirm all npx commands work with WASM
3. **Cross-Platform Validation**: Test on multiple Node.js versions
4. **Performance Verification**: Ensure WASM provides expected performance
5. **Installation Methods**: Test npm local, global, and production installations

## 🏗️ Test Structure

```
docker-wasm-test/
├── Dockerfile              # Main multi-stage test Dockerfile
├── Dockerfile.node18       # Node.js 18 compatibility test
├── Dockerfile.node22       # Node.js 22 compatibility test
├── docker-compose.yml      # Orchestrates all tests
├── build-and-test.sh      # Main test runner script
├── tests/
│   └── validate-wasm-functionality.js  # WASM-specific validator
└── results/               # Test outputs and reports
```

## 🚀 Running Tests

### Quick Start
```bash
# Run all tests
./build-and-test.sh

# Run specific test
docker-compose run npm-test

# Run with different Node version
docker-compose run node18-test
```

### Individual Tests

1. **NPM Installation Test**
   ```bash
   docker-compose run npm-test
   ```
   - Installs from npm registry
   - Verifies WASM files present
   - Checks binary format
   - Tests functionality

2. **Global Installation Test**
   ```bash
   docker-compose run global-test
   ```
   - Tests global npm install
   - Verifies CLI commands work globally

3. **Production Simulation**
   ```bash
   docker-compose run production
   ```
   - Simulates production environment
   - Minimal dependencies
   - Verifies core functionality

## 🔍 WASM-Specific Checks

The test suite performs these WASM-specific validations:

1. **Binary Format Validation**
   - Checks for `\0asm` magic number
   - Verifies WASM version (should be 1)
   - Ensures file size > 10KB

2. **Module Loading**
   - Tests WasmModuleLoader initialization
   - Verifies no placeholder fallback
   - Checks memory allocation

3. **Export Verification**
   - Ensures WASM exports are available
   - Tests specific WASM functions
   - Validates memory usage

4. **Performance Testing**
   - Runs benchmarks using WASM
   - Compares against expected performance
   - Detects placeholder usage via timing

## 📊 Test Results

Results are saved in the `results/` directory:

- `summary.txt`: Quick pass/fail summary
- `docker-test-report.md`: Detailed markdown report
- `wasm-test-results.json`: WASM-specific test data
- `wasm-validation-report.json`: Comprehensive validation results

## 🐞 Debugging

If tests fail, check:

1. **WASM Files Missing**
   ```bash
   docker-compose run npm-test ls -la node_modules/ruv-swarm/wasm/
   ```

2. **Module Loading Issues**
   ```bash
   docker-compose run npm-test node -e "
     const { WasmModuleLoader } = require('ruv-swarm/src/wasm-loader.js');
     new WasmModuleLoader().initialize('progressive').then(console.log);
   "
   ```

3. **Binary Inspection**
   ```bash
   docker-compose run npm-test xxd -l 16 node_modules/ruv-swarm/wasm/ruv_swarm_wasm_bg.wasm
   ```

## 🔧 Customization

### Environment Variables
- `NODE_ENV`: Set to 'test' for testing
- `DEBUG`: Set to 'ruv-swarm:*' for debug output
- `RUV_SWARM_WASM_PATH`: Override WASM directory location

### Adding New Tests

1. Create test file in `tests/`
2. Add new stage to Dockerfile
3. Add service to docker-compose.yml
4. Update build-and-test.sh

## 📈 Expected Results

A successful test run should show:
- ✅ All WASM files present
- ✅ Binary format valid
- ✅ Module loads without placeholder
- ✅ Memory usage > 0
- ✅ All npx commands functional
- ✅ Performance within expected range

## 🤝 Contributing

When adding new WASM functionality:
1. Add corresponding tests here
2. Ensure no regression to placeholder
3. Update validation criteria
4. Document any new WASM exports