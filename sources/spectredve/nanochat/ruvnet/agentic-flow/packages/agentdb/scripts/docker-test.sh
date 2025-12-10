#!/bin/bash
# AgentDB v2.0.0-alpha.1 - Docker Test Runner
# Comprehensive testing in Docker environment

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}================================${NC}"
echo -e "${GREEN}AgentDB v2.0.0-alpha.1${NC}"
echo -e "${GREEN}Docker Test Suite${NC}"
echo -e "${GREEN}================================${NC}"
echo ""

# Function to run a specific Docker build target
run_test_stage() {
    local stage=$1
    local description=$2

    echo -e "${YELLOW}Running: ${description}${NC}"
    if docker build --target ${stage} -t agentdb-${stage} . ; then
        echo -e "${GREEN}✅ ${description} PASSED${NC}"
        return 0
    else
        echo -e "${RED}❌ ${description} FAILED${NC}"
        return 1
    fi
}

# Track failures
FAILED_TESTS=0

# Run all test stages
echo ""
echo "1. Building base dependencies..."
run_test_stage "base" "Base Dependencies" || ((FAILED_TESTS++))

echo ""
echo "2. Building TypeScript..."
run_test_stage "builder" "TypeScript Build" || ((FAILED_TESTS++))

echo ""
echo "3. Running test suite..."
run_test_stage "test" "Test Suite" || ((FAILED_TESTS++))

echo ""
echo "4. Validating npm package..."
run_test_stage "package-test" "Package Validation" || ((FAILED_TESTS++))

echo ""
echo "5. Testing CLI commands..."
run_test_stage "cli-test" "CLI Validation" || ((FAILED_TESTS++))

echo ""
echo "6. Testing MCP server..."
run_test_stage "mcp-test" "MCP Server" || ((FAILED_TESTS++))

echo ""
echo "7. Building production image..."
run_test_stage "production" "Production Runtime" || ((FAILED_TESTS++))

echo ""
echo "8. Generating test report..."
run_test_stage "test-report" "Test Report" || ((FAILED_TESTS++))

# Final summary
echo ""
echo -e "${GREEN}================================${NC}"
echo -e "${GREEN}Test Summary${NC}"
echo -e "${GREEN}================================${NC}"

if [ $FAILED_TESTS -eq 0 ]; then
    echo -e "${GREEN}✅ All tests PASSED${NC}"
    echo ""
    echo "AgentDB v2.0.0-alpha.1 is ready for npm publish!"
    exit 0
else
    echo -e "${RED}❌ ${FAILED_TESTS} test(s) FAILED${NC}"
    echo ""
    echo "Please fix failures before publishing."
    exit 1
fi
