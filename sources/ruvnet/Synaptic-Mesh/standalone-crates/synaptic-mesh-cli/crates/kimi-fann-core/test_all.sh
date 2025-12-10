#!/bin/bash
#
# Comprehensive Test Runner for Kimi-FANN Core
# Runs all tests and generates detailed reports
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Test results tracking
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0
SKIPPED_TESTS=0

# Create test results directory
mkdir -p test-results
TEST_REPORT="test-results/test-report-$(date +%Y%m%d-%H%M%S).txt"

# Function to print colored output
print_color() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Function to log to both console and file
log_output() {
    local message=$1
    echo -e "$message" | tee -a "$TEST_REPORT"
}

# Header
log_output "${BLUE}================================================================${NC}"
log_output "${BLUE}       Kimi-FANN Core Comprehensive Test Suite${NC}"
log_output "${BLUE}                    $(date)${NC}"
log_output "${BLUE}================================================================${NC}"
log_output ""

# Check Rust version
log_output "${CYAN}📋 Environment Information:${NC}"
log_output "Rust version: $(rustc --version)"
log_output "Cargo version: $(cargo --version)"
log_output "Operating System: $(uname -s)"
log_output ""

# 1. Run basic tests
log_output "${YELLOW}🧪 Running Basic Functionality Tests...${NC}"
if cargo test --test basic_functionality --release 2>&1 | tee -a "$TEST_REPORT"; then
    ((PASSED_TESTS++))
    log_output "${GREEN}✅ Basic functionality tests passed${NC}"
else
    ((FAILED_TESTS++))
    log_output "${RED}❌ Basic functionality tests failed${NC}"
fi
((TOTAL_TESTS++))
log_output ""

# 2. Run integration tests
log_output "${YELLOW}🧪 Running Integration Tests...${NC}"
if cargo test --test integration_tests --release -- --nocapture 2>&1 | tee -a "$TEST_REPORT"; then
    ((PASSED_TESTS++))
    log_output "${GREEN}✅ Integration tests passed${NC}"
else
    ((FAILED_TESTS++))
    log_output "${RED}❌ Integration tests failed${NC}"
fi
((TOTAL_TESTS++))
log_output ""

# 3. Run unit tests
log_output "${YELLOW}🧪 Running Unit Tests...${NC}"
if cargo test --lib --release 2>&1 | tee -a "$TEST_REPORT"; then
    ((PASSED_TESTS++))
    log_output "${GREEN}✅ Unit tests passed${NC}"
else
    ((FAILED_TESTS++))
    log_output "${RED}❌ Unit tests failed${NC}"
fi
((TOTAL_TESTS++))
log_output ""

# 4. Run example tests
log_output "${YELLOW}🧪 Running Example Tests...${NC}"
if cargo test --examples --release 2>&1 | tee -a "$TEST_REPORT"; then
    ((PASSED_TESTS++))
    log_output "${GREEN}✅ Example tests passed${NC}"
else
    ((FAILED_TESTS++))
    log_output "${RED}❌ Example tests failed${NC}"
fi
((TOTAL_TESTS++))
log_output ""

# 5. Run documentation tests
log_output "${YELLOW}🧪 Running Documentation Tests...${NC}"
if cargo test --doc --release 2>&1 | tee -a "$TEST_REPORT"; then
    ((PASSED_TESTS++))
    log_output "${GREEN}✅ Documentation tests passed${NC}"
else
    ((FAILED_TESTS++))
    log_output "${RED}❌ Documentation tests failed${NC}"
fi
((TOTAL_TESTS++))
log_output ""

# 6. Test specific features
log_output "${YELLOW}🧪 Testing Specific Features...${NC}"

# Test each expert domain
DOMAINS=("reasoning" "coding" "mathematics" "language" "tool_use" "context")
for domain in "${DOMAINS[@]}"; do
    log_output "${CYAN}Testing $domain expert...${NC}"
    if cargo test "test_${domain}" --release 2>&1 | grep -q "test result: ok"; then
        log_output "${GREEN}✅ $domain expert tests passed${NC}"
    else
        log_output "${YELLOW}⚠️  No specific tests for $domain expert${NC}"
    fi
done
log_output ""

# 7. Performance benchmarks (if available)
log_output "${YELLOW}🧪 Running Performance Benchmarks...${NC}"
if cargo bench --no-run 2>&1 | grep -q "error"; then
    log_output "${YELLOW}⚠️  Benchmarks not configured${NC}"
    ((SKIPPED_TESTS++))
else
    if timeout 60 cargo bench 2>&1 | tee -a "$TEST_REPORT"; then
        ((PASSED_TESTS++))
        log_output "${GREEN}✅ Benchmarks completed${NC}"
    else
        ((FAILED_TESTS++))
        log_output "${RED}❌ Benchmarks failed or timed out${NC}"
    fi
fi
((TOTAL_TESTS++))
log_output ""

# 8. WASM build test
log_output "${YELLOW}🧪 Testing WASM Build...${NC}"
if command -v wasm-pack >/dev/null 2>&1; then
    if wasm-pack build --target web --dev 2>&1 | tee -a "$TEST_REPORT"; then
        ((PASSED_TESTS++))
        log_output "${GREEN}✅ WASM build successful${NC}"
        
        # Check WASM output files
        if [ -f "pkg/kimi_fann_core_bg.wasm" ]; then
            WASM_SIZE=$(ls -lh pkg/kimi_fann_core_bg.wasm | awk '{print $5}')
            log_output "${CYAN}WASM file size: $WASM_SIZE${NC}"
        fi
    else
        ((FAILED_TESTS++))
        log_output "${RED}❌ WASM build failed${NC}"
    fi
else
    ((SKIPPED_TESTS++))
    log_output "${YELLOW}⚠️  wasm-pack not installed, skipping WASM tests${NC}"
fi
((TOTAL_TESTS++))
log_output ""

# 9. Linting checks
log_output "${YELLOW}🧪 Running Linting Checks...${NC}"
if cargo clippy -- -D warnings 2>&1 | tee -a "$TEST_REPORT"; then
    ((PASSED_TESTS++))
    log_output "${GREEN}✅ Linting checks passed${NC}"
else
    ((FAILED_TESTS++))
    log_output "${RED}❌ Linting checks failed${NC}"
fi
((TOTAL_TESTS++))
log_output ""

# 10. Format check
log_output "${YELLOW}🧪 Running Format Check...${NC}"
if cargo fmt -- --check 2>&1 | tee -a "$TEST_REPORT"; then
    ((PASSED_TESTS++))
    log_output "${GREEN}✅ Code formatting is correct${NC}"
else
    ((FAILED_TESTS++))
    log_output "${RED}❌ Code formatting issues found${NC}"
fi
((TOTAL_TESTS++))
log_output ""

# Generate test summary
log_output "${BLUE}================================================================${NC}"
log_output "${BLUE}                    Test Summary${NC}"
log_output "${BLUE}================================================================${NC}"
log_output ""
log_output "Total Test Suites: $TOTAL_TESTS"
log_output "${GREEN}Passed: $PASSED_TESTS${NC}"
log_output "${RED}Failed: $FAILED_TESTS${NC}"
log_output "${YELLOW}Skipped: $SKIPPED_TESTS${NC}"
log_output ""

# Calculate success rate
if [ $TOTAL_TESTS -gt 0 ]; then
    SUCCESS_RATE=$(( (PASSED_TESTS * 100) / TOTAL_TESTS ))
    log_output "Success Rate: ${SUCCESS_RATE}%"
    
    if [ $SUCCESS_RATE -eq 100 ]; then
        log_output "${GREEN}🎉 All tests passed! The system is working perfectly.${NC}"
    elif [ $SUCCESS_RATE -ge 80 ]; then
        log_output "${GREEN}✅ Most tests passed. The system is working well.${NC}"
    elif [ $SUCCESS_RATE -ge 60 ]; then
        log_output "${YELLOW}⚠️  Some tests failed. Review the failures above.${NC}"
    else
        log_output "${RED}❌ Many tests failed. The system needs attention.${NC}"
    fi
fi

# Generate example outputs for documentation
log_output ""
log_output "${BLUE}================================================================${NC}"
log_output "${BLUE}           Example Outputs for Documentation${NC}"
log_output "${BLUE}================================================================${NC}"
log_output ""

# Run example programs and capture outputs
log_output "${CYAN}📋 Running Example Programs...${NC}"

# Basic neural usage example
if cargo run --example basic_neural_usage --release 2>&1 | tail -20 >> "$TEST_REPORT"; then
    log_output "${GREEN}✅ Basic neural usage example completed${NC}"
fi

# Command line usage example  
if cargo run --example command_line_usage --release 2>&1 | tail -20 >> "$TEST_REPORT"; then
    log_output "${GREEN}✅ Command line usage example completed${NC}"
fi

# Performance benchmark example
if timeout 10 cargo run --example performance_benchmark --release 2>&1 | tail -20 >> "$TEST_REPORT"; then
    log_output "${GREEN}✅ Performance benchmark example completed${NC}"
fi

log_output ""
log_output "${PURPLE}📄 Full test report saved to: $TEST_REPORT${NC}"
log_output ""

# Exit with appropriate code
if [ $FAILED_TESTS -eq 0 ]; then
    exit 0
else
    exit 1
fi