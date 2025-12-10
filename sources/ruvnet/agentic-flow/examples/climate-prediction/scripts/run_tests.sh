#!/bin/bash

# Comprehensive test runner script with coverage reporting

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "🧪 Climate Prediction Test Suite Runner"
echo "========================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test counters
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

run_test_suite() {
    local suite_name=$1
    local test_command=$2

    echo ""
    echo "📦 Running $suite_name..."
    echo "----------------------------------------"

    if eval "$test_command"; then
        echo -e "${GREEN}✅ $suite_name passed${NC}"
        PASSED_TESTS=$((PASSED_TESTS + 1))
    else
        echo -e "${RED}❌ $suite_name failed${NC}"
        FAILED_TESTS=$((FAILED_TESTS + 1))
    fi

    TOTAL_TESTS=$((TOTAL_TESTS + 1))
}

# 1. Unit Tests
run_test_suite "Unit Tests" "cargo test --lib --all-features --verbose"

# 2. Integration Tests
run_test_suite "Data Ingestion Tests" "cargo test --test data_ingestion_test --all-features --verbose"
run_test_suite "Model Inference Tests" "cargo test --test model_inference_test --all-features --verbose"
run_test_suite "API Endpoints Tests" "cargo test --test api_endpoints_test --all-features --verbose"
run_test_suite "End-to-End Tests" "cargo test --test end_to_end_test --all-features --verbose"

# 3. Property Tests
run_test_suite "Property-Based Tests" "cargo test --test property_tests --all-features --verbose"

# 4. Code Quality Checks
echo ""
echo "🔍 Running Code Quality Checks..."
echo "----------------------------------------"

if cargo fmt -- --check; then
    echo -e "${GREEN}✅ Code formatting check passed${NC}"
    PASSED_TESTS=$((PASSED_TESTS + 1))
else
    echo -e "${RED}❌ Code formatting check failed${NC}"
    FAILED_TESTS=$((FAILED_TESTS + 1))
fi
TOTAL_TESTS=$((TOTAL_TESTS + 1))

if cargo clippy --all-targets --all-features -- -D warnings; then
    echo -e "${GREEN}✅ Clippy linting passed${NC}"
    PASSED_TESTS=$((PASSED_TESTS + 1))
else
    echo -e "${RED}❌ Clippy linting failed${NC}"
    FAILED_TESTS=$((FAILED_TESTS + 1))
fi
TOTAL_TESTS=$((TOTAL_TESTS + 1))

# 5. Coverage Report
echo ""
echo "📊 Generating Coverage Report..."
echo "----------------------------------------"

if command -v cargo-tarpaulin &> /dev/null; then
    cargo tarpaulin --all-features --workspace --timeout 120 --out Html --output-dir coverage

    COVERAGE=$(cargo tarpaulin --all-features --workspace --timeout 120 | grep -oP '\d+\.\d+(?=%)' || echo "0")

    echo ""
    echo "📈 Coverage: $COVERAGE%"

    if (( $(echo "$COVERAGE >= 80" | bc -l 2>/dev/null || echo "0") )); then
        echo -e "${GREEN}✅ Coverage meets 80% threshold${NC}"
        PASSED_TESTS=$((PASSED_TESTS + 1))
    else
        echo -e "${YELLOW}⚠️  Coverage below 80% threshold${NC}"
        FAILED_TESTS=$((FAILED_TESTS + 1))
    fi
    TOTAL_TESTS=$((TOTAL_TESTS + 1))

    echo ""
    echo "📄 HTML coverage report generated at: coverage/index.html"
else
    echo -e "${YELLOW}⚠️  cargo-tarpaulin not installed. Skipping coverage.${NC}"
    echo "Install with: cargo install cargo-tarpaulin"
fi

# 6. Benchmark Compilation
echo ""
echo "⚡ Checking Benchmark Compilation..."
echo "----------------------------------------"

if cargo bench --no-run; then
    echo -e "${GREEN}✅ Benchmarks compile successfully${NC}"
    PASSED_TESTS=$((PASSED_TESTS + 1))
else
    echo -e "${RED}❌ Benchmark compilation failed${NC}"
    FAILED_TESTS=$((FAILED_TESTS + 1))
fi
TOTAL_TESTS=$((TOTAL_TESTS + 1))

# Summary
echo ""
echo "========================================"
echo "📊 Test Summary"
echo "========================================"
echo "Total Test Suites: $TOTAL_TESTS"
echo -e "Passed: ${GREEN}$PASSED_TESTS${NC}"
echo -e "Failed: ${RED}$FAILED_TESTS${NC}"

if [ $FAILED_TESTS -eq 0 ]; then
    echo ""
    echo -e "${GREEN}🎉 All tests passed!${NC}"
    exit 0
else
    echo ""
    echo -e "${RED}❌ Some tests failed${NC}"
    exit 1
fi
