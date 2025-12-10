#!/bin/bash

# Comprehensive Swarm Intelligence Test Runner
# Executes all swarm intelligence tests in the correct order

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Test configuration
TEST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$TEST_DIR")"
REPORTS_DIR="$TEST_DIR/reports"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Ensure reports directory exists
mkdir -p "$REPORTS_DIR"

# Initialize test results
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0
TEST_RESULTS=()

echo -e "${CYAN}🧪 Synaptic Neural Mesh - Swarm Intelligence Test Suite${NC}"
echo -e "${CYAN}================================================================${NC}"
echo -e "Start Time: $(date)"
echo -e "Test Directory: $TEST_DIR"
echo -e "Reports Directory: $REPORTS_DIR"
echo ""

# Function to run a test and capture results
run_test() {
    local test_name="$1"
    local test_command="$2"
    local test_description="$3"
    
    echo -e "${BLUE}🔬 Running: $test_description${NC}"
    echo -e "${BLUE}Command: $test_command${NC}"
    echo ""
    
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    
    # Create log file for this test
    local log_file="$REPORTS_DIR/${test_name}_${TIMESTAMP}.log"
    
    # Run the test and capture output
    if eval "$test_command" 2>&1 | tee "$log_file"; then
        echo -e "${GREEN}✅ PASSED: $test_description${NC}"
        PASSED_TESTS=$((PASSED_TESTS + 1))
        TEST_RESULTS+=("PASS: $test_description")
    else
        echo -e "${RED}❌ FAILED: $test_description${NC}"
        FAILED_TESTS=$((FAILED_TESTS + 1))
        TEST_RESULTS+=("FAIL: $test_description")
    fi
    
    echo ""
    echo -e "${YELLOW}─────────────────────────────────────────────────────────────${NC}"
    echo ""
}

# Function to check dependencies
check_dependencies() {
    echo -e "${PURPLE}🔍 Checking Dependencies...${NC}"
    
    # Check Node.js
    if ! command -v node &> /dev/null; then
        echo -e "${RED}❌ Node.js is required but not installed.${NC}"
        exit 1
    fi
    
    local node_version=$(node --version)
    echo -e "${GREEN}✅ Node.js: $node_version${NC}"
    
    # Check npm packages are installed
    if [ ! -d "$PROJECT_ROOT/node_modules" ]; then
        echo -e "${YELLOW}⚠️ Node modules not found. Installing dependencies...${NC}"
        cd "$PROJECT_ROOT"
        npm install
    fi
    
    # Check test files exist
    local required_files=(
        "swarm-components-unit.test.js"
        "swarm-intelligence-integration.test.js"
        "swarm-performance-benchmark.test.js"
    )
    
    for file in "${required_files[@]}"; do
        if [ ! -f "$TEST_DIR/$file" ]; then
            echo -e "${RED}❌ Required test file not found: $file${NC}"
            exit 1
        fi
    done
    
    echo -e "${GREEN}✅ All dependencies satisfied${NC}"
    echo ""
}

# Function to generate summary report
generate_summary_report() {
    local summary_file="$REPORTS_DIR/swarm_test_summary_${TIMESTAMP}.md"
    
    cat > "$summary_file" << EOF
# Swarm Intelligence Test Suite Summary

**Execution Date:** $(date)  
**Total Tests:** $TOTAL_TESTS  
**Passed:** $PASSED_TESTS  
**Failed:** $FAILED_TESTS  
**Success Rate:** $(( (PASSED_TESTS * 100) / TOTAL_TESTS ))%  

## Test Results

EOF

    for result in "${TEST_RESULTS[@]}"; do
        if [[ $result == PASS:* ]]; then
            echo "- ✅ ${result#PASS: }" >> "$summary_file"
        else
            echo "- ❌ ${result#FAIL: }" >> "$summary_file"
        fi
    done
    
    cat >> "$summary_file" << EOF

## Performance Targets

The swarm intelligence system should meet these performance targets:

- **Agent Selection Time:** < 100ms
- **Consensus Latency:** < 5000ms  
- **Evolution Cycle Time:** < 3000ms
- **Memory per Agent:** < 10MB
- **Throughput:** > 100 ops/second
- **Scalability Factor:** > 0.9

## Next Steps

EOF

    if [ $FAILED_TESTS -eq 0 ]; then
        cat >> "$summary_file" << EOF
🚀 **EXCELLENT!** All tests passed. The swarm intelligence system is ready for production integration.

Recommended actions:
1. Proceed with production deployment
2. Monitor performance in real-world scenarios
3. Consider advanced optimization strategies
EOF
    elif [ $FAILED_TESTS -le 2 ]; then
        cat >> "$summary_file" << EOF
👍 **GOOD!** Most tests passed with only minor issues.

Recommended actions:
1. Review and fix failing tests
2. Re-run test suite to verify fixes
3. Consider production deployment after fixes
EOF
    else
        cat >> "$summary_file" << EOF
⚠️ **NEEDS ATTENTION!** Several tests failed and require investigation.

Recommended actions:
1. Review detailed test logs in reports directory
2. Address failing components systematically
3. Re-run tests after fixes
4. Consider additional debugging and optimization
EOF
    fi
    
    echo -e "${CYAN}📄 Summary report generated: $summary_file${NC}"
}

# Function to print final results
print_final_results() {
    echo -e "${CYAN}🎯 FINAL TEST RESULTS${NC}"
    echo -e "${CYAN}================================================================${NC}"
    echo -e "Total Tests: $TOTAL_TESTS"
    echo -e "${GREEN}Passed: $PASSED_TESTS${NC}"
    echo -e "${RED}Failed: $FAILED_TESTS${NC}"
    
    if [ $TOTAL_TESTS -gt 0 ]; then
        local success_rate=$(( (PASSED_TESTS * 100) / TOTAL_TESTS ))
        echo -e "Success Rate: ${success_rate}%"
        
        if [ $FAILED_TESTS -eq 0 ]; then
            echo -e "\n${GREEN}🚀 EXCELLENT! All swarm intelligence tests passed!${NC}"
            echo -e "${GREEN}The system is ready for production integration.${NC}"
        elif [ $success_rate -ge 80 ]; then
            echo -e "\n${YELLOW}👍 GOOD! Most tests passed with minor issues.${NC}"
            echo -e "${YELLOW}Review failed tests and consider fixes before production.${NC}"
        else
            echo -e "\n${RED}⚠️ NEEDS ATTENTION! Multiple test failures detected.${NC}"
            echo -e "${RED}Significant issues require resolution before production.${NC}"
        fi
    fi
    
    echo -e "\nEnd Time: $(date)"
    echo -e "Test logs available in: $REPORTS_DIR"
}

# Main test execution
main() {
    echo -e "${PURPLE}Starting Swarm Intelligence Test Suite...${NC}"
    echo ""
    
    # Check dependencies first
    check_dependencies
    
    # Change to test directory
    cd "$TEST_DIR"
    
    # 1. Unit Tests - Test individual components
    run_test "unit_tests" \
        "node swarm-components-unit.test.js" \
        "Swarm Components Unit Tests"
    
    # 2. Integration Tests - Test component interactions
    run_test "integration_tests" \
        "node swarm-intelligence-integration.test.js" \
        "Swarm Intelligence Integration Tests"
    
    # 3. Performance Benchmarks - Test performance characteristics
    run_test "performance_benchmarks" \
        "node swarm-performance-benchmark.test.js" \
        "Swarm Performance Benchmark Tests"
    
    # 4. Run existing comprehensive QA if available
    if [ -f "comprehensive-qa-runner.js" ]; then
        run_test "comprehensive_qa" \
            "node comprehensive-qa-runner.js" \
            "Comprehensive QA Test Suite"
    fi
    
    # 5. Run swarm intelligence demo as validation test
    if [ -f "../examples/swarm_intelligence_demo.js" ]; then
        run_test "swarm_demo" \
            "node ../examples/swarm_intelligence_demo.js" \
            "Swarm Intelligence Demonstration"
    fi
    
    # Generate reports and summary
    generate_summary_report
    print_final_results
    
    # Exit with appropriate code
    if [ $FAILED_TESTS -eq 0 ]; then
        exit 0
    else
        exit 1
    fi
}

# Handle interruption gracefully
trap 'echo -e "\n${YELLOW}🛑 Test suite interrupted by user${NC}"; exit 130' INT TERM

# Parse command line arguments
VERBOSE=false
QUICK=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--verbose)
            VERBOSE=true
            echo -e "${PURPLE}Verbose mode enabled${NC}"
            shift
            ;;
        -q|--quick)
            QUICK=true
            echo -e "${PURPLE}Quick mode enabled (skipping performance benchmarks)${NC}"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "OPTIONS:"
            echo "  -v, --verbose    Enable verbose output"
            echo "  -q, --quick      Quick mode (skip performance benchmarks)"
            echo "  -h, --help       Show this help message"
            echo ""
            echo "EXAMPLES:"
            echo "  $0                 # Run all tests"
            echo "  $0 --quick         # Run tests without benchmarks"
            echo "  $0 --verbose       # Run with detailed output"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Run the main test suite
main "$@"