#!/bin/bash
# Script to run neuro-divergent stress tests

echo "🚀 Neuro-Divergent Stress Test Runner"
echo "===================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to run a test category
run_test_category() {
    local category=$1
    local test_name=$2
    echo -e "\n${YELLOW}Running $test_name...${NC}"
    
    if cargo test --release --test "$category" -- --test-threads=1 --nocapture; then
        echo -e "${GREEN}✅ $test_name passed${NC}"
    else
        echo -e "${RED}❌ $test_name failed${NC}"
    fi
}

# Check if we're in the right directory
if [ ! -f "Cargo.toml" ]; then
    echo -e "${RED}Error: Must run from neuro-divergent directory${NC}"
    exit 1
fi

# Parse command line arguments
INCLUDE_IGNORED=false
CATEGORY=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --ignored)
            INCLUDE_IGNORED=true
            shift
            ;;
        --category)
            CATEGORY="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --ignored       Include ignored tests (resource intensive)"
            echo "  --category NAME Run only specific test category"
            echo "                  (large_dataset, edge_case, resource_limit,"
            echo "                   concurrent_usage, failure_recovery, fuzz, memory)"
            echo "  --help          Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Build the test command
TEST_CMD="cargo test --release"
if [ "$INCLUDE_IGNORED" = true ]; then
    TEST_CMD="$TEST_CMD -- --ignored"
fi

# Run tests based on category
if [ -n "$CATEGORY" ]; then
    case $CATEGORY in
        large_dataset)
            run_test_category "large_dataset_tests" "Large Dataset Tests"
            ;;
        edge_case)
            run_test_category "edge_case_tests" "Edge Case Tests"
            ;;
        resource_limit)
            run_test_category "resource_limit_tests" "Resource Limit Tests"
            ;;
        concurrent_usage)
            run_test_category "concurrent_usage_tests" "Concurrent Usage Tests"
            ;;
        failure_recovery)
            run_test_category "failure_recovery_tests" "Failure Recovery Tests"
            ;;
        fuzz)
            run_test_category "fuzz_tests" "Fuzz Tests"
            ;;
        memory)
            run_test_category "memory_stress_tests" "Memory Stress Tests"
            ;;
        *)
            echo -e "${RED}Unknown category: $CATEGORY${NC}"
            exit 1
            ;;
    esac
else
    # Run all stress tests
    echo "Running all stress test categories..."
    echo "===================================="
    
    # Non-ignored tests first
    echo -e "\n${YELLOW}Running standard stress tests...${NC}"
    $TEST_CMD --test '*stress*'
    
    if [ "$INCLUDE_IGNORED" = true ]; then
        echo -e "\n${YELLOW}Running resource-intensive tests...${NC}"
        echo -e "${RED}WARNING: These tests may use significant resources!${NC}"
        read -p "Continue? (y/N) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            # Run each category with ignored tests
            for category in large_dataset edge_case resource_limit concurrent_usage failure_recovery fuzz memory; do
                run_test_category "${category}_tests" "${category} Tests (with ignored)"
            done
        fi
    fi
fi

echo -e "\n${GREEN}Stress test run complete!${NC}"
echo "See neuro-divergent/docs/STRESS_TEST_REPORT.md for detailed results."