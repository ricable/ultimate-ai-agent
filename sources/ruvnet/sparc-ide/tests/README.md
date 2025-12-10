# SPARC IDE Tests

This directory contains comprehensive tests for the SPARC IDE implementation, focusing on:

1. Build scripts testing
2. Roo Code integration validation
3. UI configuration and customization testing
4. Branding modification verification

## Test Structure

- `build-scripts/` - Tests for build scripts (setup-build-environment.sh, build-sparc-ide.sh)
- `roo-code/` - Tests for Roo Code integration
- `ui-config/` - Tests for UI configuration and customization
- `branding/` - Tests for branding modification functionality
- `helpers/` - Test helper functions and utilities

## Running Tests

To run all tests:

```bash
./run-tests.sh
```

To run a specific test category:

```bash
./run-tests.sh build-scripts
./run-tests.sh roo-code
./run-tests.sh ui-config
./run-tests.sh branding
```

## Test Reports

Test reports are generated in the `test-reports/` directory.