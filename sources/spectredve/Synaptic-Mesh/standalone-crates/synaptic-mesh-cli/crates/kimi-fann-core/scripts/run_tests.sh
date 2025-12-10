#!/bin/bash
#
# Comprehensive Test Runner for Kimi-FANN Core
# Runs all test types: unit, integration, property-based, benchmarks, and WASM tests
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
CARGO_FLAGS="--release"
BENCH_FLAGS="--output-format html"
WASM_PACK_FLAGS="--target web --dev"

echo -e "${BLUE}🧪 Kimi-FANN Core Comprehensive Test Suite${NC}"
echo "=========================================="

# Function to run tests with timing
run_test_suite() {
    local suite_name="$1"
    local command="$2"
    
    echo -e "\n${YELLOW}📋 Running $suite_name...${NC}"
    start_time=$(date +%s)
    
    if eval "$command"; then
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        echo -e "${GREEN}✅ $suite_name completed in ${duration}s${NC}"
        return 0
    else
        echo -e "${RED}❌ $suite_name failed${NC}"
        return 1
    fi
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
echo -e "\n${BLUE}🔍 Checking prerequisites...${NC}"
prerequisites_ok=true

if ! command_exists cargo; then
    echo -e "${RED}❌ cargo not found${NC}"
    prerequisites_ok=false
fi

if ! command_exists wasm-pack; then
    echo -e "${YELLOW}⚠️  wasm-pack not found, WASM tests will be skipped${NC}"
    SKIP_WASM=true
else
    echo -e "${GREEN}✅ wasm-pack found${NC}"
fi

if ! command_exists node; then
    echo -e "${YELLOW}⚠️  node not found, some WASM tests may fail${NC}"
fi

if [ "$prerequisites_ok" = false ]; then
    echo -e "${RED}❌ Missing required prerequisites${NC}"
    exit 1
fi

# Set up environment
export RUST_BACKTRACE=1
export RUST_LOG=debug

# Create output directories
mkdir -p target/test-results
mkdir -p target/coverage
mkdir -p target/benchmarks

# 1. Unit Tests
run_test_suite "Unit Tests" "cargo test --lib $CARGO_FLAGS"

# 2. Integration Tests
run_test_suite "Integration Tests" "cargo test --test integration_tests $CARGO_FLAGS"

# 3. Neural Network Tests
run_test_suite "Neural Network Tests" "cargo test --test neural_network_tests $CARGO_FLAGS"

# 4. Router Tests
run_test_suite "Router Tests" "cargo test --test router_tests $CARGO_FLAGS"

# 5. End-to-End Tests
run_test_suite "End-to-End Tests" "cargo test --test end_to_end_tests $CARGO_FLAGS"

# 6. Property-Based Tests
run_test_suite "Property-Based Tests" "cargo test --test property_tests $CARGO_FLAGS"

# 7. Examples Tests
run_test_suite "Examples Tests" "cargo test --examples $CARGO_FLAGS"

# 8. Documentation Tests
run_test_suite "Documentation Tests" "cargo test --doc $CARGO_FLAGS"

# 9. Performance Benchmarks
if command_exists cargo-criterion || cargo install --list | grep -q criterion; then
    run_test_suite "Performance Benchmarks" "cargo bench $BENCH_FLAGS"
else
    echo -e "${YELLOW}⚠️  cargo-criterion not found, installing...${NC}"
    cargo install cargo-criterion
    run_test_suite "Performance Benchmarks" "cargo bench $BENCH_FLAGS"
fi

# 10. WASM Tests (if available)
if [ "$SKIP_WASM" != true ]; then
    echo -e "\n${YELLOW}🌐 Building WASM package...${NC}"
    if wasm-pack build $WASM_PACK_FLAGS; then
        echo -e "${GREEN}✅ WASM build successful${NC}"
        
        # Run WASM tests
        run_test_suite "WASM Tests" "wasm-pack test --node --release"
        
        # Test WASM in browser (headless)
        if command_exists chrome || command_exists chromium || command_exists google-chrome; then
            run_test_suite "WASM Browser Tests" "wasm-pack test --headless --chrome --release"
        else
            echo -e "${YELLOW}⚠️  Chrome not found, skipping browser WASM tests${NC}"
        fi
    else
        echo -e "${RED}❌ WASM build failed${NC}"
    fi
fi

# 11. Linting and Code Quality
echo -e "\n${YELLOW}🔍 Running code quality checks...${NC}"

# Clippy lints
if cargo clippy --version >/dev/null 2>&1; then
    run_test_suite "Clippy Lints" "cargo clippy -- -D warnings"
else
    echo -e "${YELLOW}⚠️  Clippy not found, installing...${NC}"
    rustup component add clippy
    run_test_suite "Clippy Lints" "cargo clippy -- -D warnings"
fi

# Formatting check
if cargo fmt --version >/dev/null 2>&1; then
    run_test_suite "Format Check" "cargo fmt -- --check"
else
    echo -e "${YELLOW}⚠️  rustfmt not found, installing...${NC}"
    rustup component add rustfmt
    run_test_suite "Format Check" "cargo fmt -- --check"
fi

# 12. Security Audit (if available)
if command_exists cargo-audit; then
    run_test_suite "Security Audit" "cargo audit"
else
    echo -e "${YELLOW}⚠️  cargo-audit not found, skipping security audit${NC}"
fi

# 13. Coverage Report (if available)
if command_exists cargo-tarpaulin; then
    run_test_suite "Coverage Report" "cargo tarpaulin --out html --output-dir target/coverage"
    echo -e "${BLUE}📊 Coverage report generated in target/coverage/tarpaulin-report.html${NC}"
elif command_exists cargo-llvm-cov; then
    run_test_suite "Coverage Report" "cargo llvm-cov --html --output-dir target/coverage"
    echo -e "${BLUE}📊 Coverage report generated in target/coverage/index.html${NC}"
else
    echo -e "${YELLOW}⚠️  No coverage tool found, skipping coverage report${NC}"
fi

# Test Report Summary
echo -e "\n${BLUE}📊 Test Summary${NC}"
echo "==============="

# Count test results
total_tests=$(cargo test --list 2>/dev/null | grep -c "test " || echo "Unknown")
echo -e "${BLUE}Total tests: $total_tests${NC}"

# Check if all tests passed
if cargo test --quiet $CARGO_FLAGS >/dev/null 2>&1; then
    echo -e "${GREEN}✅ All tests passed!${NC}"
    test_status=0
else
    echo -e "${RED}❌ Some tests failed${NC}"
    test_status=1
fi

# Performance summary
if [ -f "target/criterion/index.html" ]; then
    echo -e "${BLUE}📈 Performance report: target/criterion/index.html${NC}"
fi

# Final recommendations
echo -e "\n${BLUE}🎯 Recommendations${NC}"
echo "=================="

if [ $test_status -eq 0 ]; then
    echo -e "${GREEN}✅ All tests are passing - ready for deployment!${NC}"
else
    echo -e "${RED}❌ Fix failing tests before deployment${NC}"
fi

echo -e "${BLUE}💡 Next steps:${NC}"
echo "  - Review benchmark results for performance optimization"
echo "  - Check coverage report for untested code paths"
echo "  - Run WASM tests in different browsers for compatibility"
echo "  - Consider adding more property-based tests for edge cases"

# Archive test results
echo -e "\n${BLUE}📦 Archiving test results...${NC}"
timestamp=$(date +"%Y%m%d_%H%M%S")
mkdir -p "test-archives/$timestamp"

# Copy test artifacts
cp -r target/criterion "test-archives/$timestamp/" 2>/dev/null || true
cp -r target/coverage "test-archives/$timestamp/" 2>/dev/null || true
cp -r pkg "test-archives/$timestamp/" 2>/dev/null || true

echo -e "${GREEN}✅ Test results archived in test-archives/$timestamp/${NC}"

exit $test_status