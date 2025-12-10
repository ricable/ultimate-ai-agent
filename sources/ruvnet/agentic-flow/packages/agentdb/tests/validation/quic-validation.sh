#!/bin/bash

# QUIC Synchronization End-to-End Validation Script
# This script performs comprehensive validation of QUIC sync functionality

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test results
TESTS_PASSED=0
TESTS_FAILED=0
TESTS_SKIPPED=0

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[PASS]${NC} $1"
    ((TESTS_PASSED++))
}

log_error() {
    echo -e "${RED}[FAIL]${NC} $1"
    ((TESTS_FAILED++))
}

log_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_skip() {
    echo -e "${YELLOW}[SKIP]${NC} $1"
    ((TESTS_SKIPPED++))
}

print_header() {
    echo ""
    echo "=================================================="
    echo "$1"
    echo "=================================================="
    echo ""
}

# Validate test environment
validate_environment() {
    print_header "Validating Test Environment"

    # Check Node.js
    if command -v node &> /dev/null; then
        NODE_VERSION=$(node --version)
        log_success "Node.js found: $NODE_VERSION"
    else
        log_error "Node.js not found"
        return 1
    fi

    # Check npm
    if command -v npm &> /dev/null; then
        NPM_VERSION=$(npm --version)
        log_success "npm found: $NPM_VERSION"
    else
        log_error "npm not found"
        return 1
    fi

    # Check if in correct directory
    if [ ! -f "$PROJECT_DIR/package.json" ]; then
        log_error "package.json not found in $PROJECT_DIR"
        return 1
    fi
    log_success "Project directory validated: $PROJECT_DIR"

    # Check if test files exist
    if [ ! -d "$PROJECT_DIR/tests" ]; then
        log_error "Tests directory not found"
        return 1
    fi
    log_success "Tests directory found"

    return 0
}

# Run unit tests
run_unit_tests() {
    print_header "Running Unit Tests"

    cd "$PROJECT_DIR"

    log_info "Running QUICServer unit tests..."
    if npm run test:unit -- tests/unit/quic-server.test.ts 2>&1; then
        log_success "QUICServer unit tests passed"
    else
        log_error "QUICServer unit tests failed"
    fi

    log_info "Running QUICClient unit tests..."
    if npm run test:unit -- tests/unit/quic-client.test.ts 2>&1; then
        log_success "QUICClient unit tests passed"
    else
        log_error "QUICClient unit tests failed"
    fi

    log_info "Running SyncCoordinator unit tests..."
    if npm run test:unit -- tests/unit/sync-coordinator.test.ts 2>&1; then
        log_success "SyncCoordinator unit tests passed"
    else
        log_error "SyncCoordinator unit tests failed"
    fi
}

# Run integration tests
run_integration_tests() {
    print_header "Running Integration Tests"

    cd "$PROJECT_DIR"

    log_info "Running QUIC synchronization integration tests..."
    if npm run test:unit -- tests/integration/quic-sync.test.ts 2>&1; then
        log_success "QUIC integration tests passed"
    else
        log_error "QUIC integration tests failed"
    fi
}

# Test coverage report
generate_coverage_report() {
    print_header "Generating Coverage Report"

    cd "$PROJECT_DIR"

    log_info "Running tests with coverage..."
    if npm run test:unit -- --coverage tests/unit/*.test.ts tests/integration/*.test.ts 2>&1 | tee coverage.log; then
        log_success "Coverage report generated"

        # Extract coverage summary if available
        if [ -f "coverage.log" ]; then
            log_info "Coverage Summary:"
            grep -A 10 "Coverage" coverage.log || true
            rm coverage.log
        fi
    else
        log_warning "Coverage report generation failed (may not be configured)"
    fi
}

# Validate QUIC implementation readiness
validate_quic_readiness() {
    print_header "Validating QUIC Implementation Readiness"

    # Check for required dependencies
    log_info "Checking for QUIC-related dependencies..."

    cd "$PROJECT_DIR"

    # Note: Actual QUIC implementation would require specific libraries
    # This is a placeholder for when implementation begins
    log_skip "QUIC implementation not yet present (tests use mocks)"

    # Check if sync directory structure exists
    if [ -d "$PROJECT_DIR/src/sync" ]; then
        log_success "Sync directory structure exists"
    else
        log_skip "Sync directory not created yet (expected for mock tests)"
    fi
}

# Test performance characteristics
test_performance() {
    print_header "Performance Testing"

    cd "$PROJECT_DIR"

    log_info "Running performance tests..."
    if npm run test:unit -- tests/integration/quic-sync.test.ts -t "Performance" 2>&1; then
        log_success "Performance tests passed"
    else
        log_warning "Performance tests not executed or failed"
    fi
}

# Test error handling
test_error_handling() {
    print_header "Error Handling Tests"

    cd "$PROJECT_DIR"

    log_info "Running error handling tests..."
    if npm run test:unit -- tests/unit/*.test.ts -t "Error" 2>&1; then
        log_success "Error handling tests passed"
    else
        log_error "Error handling tests failed"
    fi
}

# Test concurrent operations
test_concurrent_operations() {
    print_header "Concurrent Operations Tests"

    cd "$PROJECT_DIR"

    log_info "Running concurrent operation tests..."
    if npm run test:unit -- tests/unit/*.test.ts -t "Concurrent" 2>&1; then
        log_success "Concurrent operation tests passed"
    else
        log_error "Concurrent operation tests failed"
    fi
}

# Generate test report
generate_test_report() {
    print_header "Test Summary Report"

    TOTAL_TESTS=$((TESTS_PASSED + TESTS_FAILED + TESTS_SKIPPED))

    echo ""
    echo "Total Test Suites Run: $TOTAL_TESTS"
    echo -e "${GREEN}Passed: $TESTS_PASSED${NC}"
    echo -e "${RED}Failed: $TESTS_FAILED${NC}"
    echo -e "${YELLOW}Skipped: $TESTS_SKIPPED${NC}"
    echo ""

    if [ $TESTS_FAILED -eq 0 ]; then
        echo -e "${GREEN}✓ All tests passed!${NC}"
        return 0
    else
        echo -e "${RED}✗ Some tests failed${NC}"
        return 1
    fi
}

# Cleanup
cleanup() {
    log_info "Cleaning up..."
    cd "$PROJECT_DIR"
    rm -f coverage.log
}

# Main execution
main() {
    print_header "QUIC Synchronization Validation Suite"

    log_info "Starting validation at $(date)"

    # Run validation steps
    if ! validate_environment; then
        log_error "Environment validation failed"
        exit 1
    fi

    run_unit_tests
    run_integration_tests
    test_error_handling
    test_concurrent_operations
    test_performance
    generate_coverage_report
    validate_quic_readiness

    # Generate final report
    echo ""
    generate_test_report
    TEST_RESULT=$?

    cleanup

    log_info "Validation completed at $(date)"

    exit $TEST_RESULT
}

# Run main function
main "$@"
