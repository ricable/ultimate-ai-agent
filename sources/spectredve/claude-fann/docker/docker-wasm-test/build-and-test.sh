#!/bin/bash
set -e

echo "🚀 ruv-swarm Docker WASM Test Suite"
echo "==================================="
echo "Testing WASM loading and functionality in isolated Docker environments"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Function to print colored output
print_status() {
    local status=$1
    local message=$2
    case $status in
        "success") echo -e "${GREEN}✅ ${message}${NC}" ;;
        "error") echo -e "${RED}❌ ${message}${NC}" ;;
        "warning") echo -e "${YELLOW}⚠️  ${message}${NC}" ;;
        *) echo "$message" ;;
    esac
}

# Create results directory
mkdir -p results

# Build the npm package if needed (for local testing)
if [ -f "../ruv-swarm/npm/package.json" ]; then
    print_status "info" "Building local npm package..."
    cd ../ruv-swarm/npm
    npm pack
    cp ruv-swarm-*.tgz ../../docker-wasm-test/
    cd ../../docker-wasm-test
    print_status "success" "Local package built"
fi

# Function to run a test
run_test() {
    local test_name=$1
    local service=$2
    
    echo -e "\n📋 Running test: $test_name"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    if docker-compose run --rm $service; then
        print_status "success" "$test_name completed successfully"
        echo "$test_name: PASSED" >> results/summary.txt
    else
        print_status "error" "$test_name failed"
        echo "$test_name: FAILED" >> results/summary.txt
    fi
}

# Clean up any existing containers
docker-compose down --remove-orphans 2>/dev/null || true

# Initialize summary
echo "Test Summary - $(date)" > results/summary.txt
echo "========================" >> results/summary.txt

# Run tests
print_status "info" "Starting Docker tests..."

# 1. NPM Installation Test
run_test "NPM Installation Test" "npm-test"

# 2. Global Installation Test  
run_test "Global Installation Test" "global-test"

# 3. Production Simulation
run_test "Production Simulation" "production"

# 4. Combined Test Runner
run_test "All Tests Combined" "test-runner"

# Collect results
echo -e "\n📊 Test Results Summary"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check for WASM-specific results
if docker-compose run --rm npm-test cat wasm-test-results.json 2>/dev/null; then
    cp $(docker-compose run --rm npm-test sh -c "pwd")/wasm-test-results.json results/ 2>/dev/null || true
fi

# Count results
PASSED=$(grep -c "PASSED" results/summary.txt || echo 0)
FAILED=$(grep -c "FAILED" results/summary.txt || echo 0)
TOTAL=$((PASSED + FAILED))

echo "Total Tests: $TOTAL"
echo "Passed: $PASSED"
echo "Failed: $FAILED"

# Generate detailed report
cat > results/docker-test-report.md <<EOF
# Docker WASM Test Report

**Date:** $(date)
**Total Tests:** $TOTAL
**Passed:** $PASSED  
**Failed:** $FAILED

## Test Details

$(cat results/summary.txt)

## WASM Verification

The tests specifically verify:
1. WASM files are present in the npm package
2. WASM binary has correct magic number (\0asm)
3. WASM module loads without falling back to placeholder
4. Memory usage indicates real WASM is running
5. All npx commands work correctly

## Recommendations

EOF

if [ $FAILED -eq 0 ]; then
    echo "✅ All tests passed! WASM is loading correctly." >> results/docker-test-report.md
    print_status "success" "All tests passed! 🎉"
else
    echo "⚠️ Some tests failed. Check individual test outputs for details." >> results/docker-test-report.md
    print_status "warning" "Some tests failed. Review the results."
fi

# Clean up
docker-compose down

echo -e "\n📄 Full report saved to: results/docker-test-report.md"
echo "🗂️  Test outputs saved in: results/"

# Exit with appropriate code
exit $([ $FAILED -eq 0 ] && echo 0 || echo 1)