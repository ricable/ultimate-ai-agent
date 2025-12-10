#!/bin/bash
# Comprehensive Docker-based validation script for AgentDB v1.5.0

set -e

echo "🐳 AgentDB Docker Validation Suite"
echo "==================================="
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Counters
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# Results
RESULTS_DIR="./docker-validation-results"
mkdir -p "$RESULTS_DIR"

# Function to run test
run_test() {
    local test_name="$1"
    local command="$2"

    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    echo -e "${BLUE}[TEST $TOTAL_TESTS]${NC} $test_name"

    if eval "$command" > "$RESULTS_DIR/test_$TOTAL_TESTS.log" 2>&1; then
        echo -e "${GREEN}✅ PASS${NC}"
        PASSED_TESTS=$((PASSED_TESTS + 1))
        return 0
    else
        echo -e "${RED}❌ FAIL${NC}"
        echo "See $RESULTS_DIR/test_$TOTAL_TESTS.log for details"
        FAILED_TESTS=$((FAILED_TESTS + 1))
        return 1
    fi
}

# Build Docker image
echo -e "${YELLOW}Building validation Docker image...${NC}"
run_test "Docker image build" \
    "docker build -f Dockerfile.validation -t agentdb-validation:latest --target base ."

# Test 1: Security validation
echo ""
echo -e "${YELLOW}=== Security Tests ===${NC}"
run_test "Security fixes validation" \
    "docker build -f Dockerfile.validation --target security-test -t agentdb-security ."

# Test 2: CLI validation
echo ""
echo -e "${YELLOW}=== CLI Tests ===${NC}"
run_test "CLI commands and init" \
    "docker build -f Dockerfile.validation --target cli-test -t agentdb-cli ."

# Test 3: Database init creates file
echo -e "${BLUE}[Detailed Test]${NC} Verifying database file creation..."
docker run --rm agentdb-cli sh -c "ls -lh /tmp/test1.db && sqlite3 /tmp/test1.db 'SELECT COUNT(*) FROM sqlite_master WHERE type=\"table\";'" > "$RESULTS_DIR/db_verification.log" 2>&1
if [ $? -eq 0 ]; then
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    PASSED_TESTS=$((PASSED_TESTS + 1))
    echo -e "${GREEN}✅ Database file created successfully${NC}"
    cat "$RESULTS_DIR/db_verification.log"
else
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    FAILED_TESTS=$((FAILED_TESTS + 1))
    echo -e "${RED}❌ Database file creation failed${NC}"
fi

# Test 4: MCP server
echo ""
echo -e "${YELLOW}=== MCP Server Tests ===${NC}"
run_test "MCP server startup" \
    "docker build -f Dockerfile.validation --target mcp-test -t agentdb-mcp ."

# Test 5: Test suite
echo ""
echo -e "${YELLOW}=== Test Suite ===${NC}"
run_test "Full test suite execution" \
    "docker build -f Dockerfile.validation --target test-suite -t agentdb-tests ."

# Extract test results
echo -e "${BLUE}[Analysis]${NC} Extracting test results..."
docker run --rm agentdb-tests cat /tmp/test-results.log > "$RESULTS_DIR/full_test_results.log" 2>&1 || true

# Test 6: Regression tests
echo ""
echo -e "${YELLOW}=== Regression Tests ===${NC}"
run_test "Regression validation" \
    "docker build -f Dockerfile.validation --target regression-test -t agentdb-regression ."

# Test 7: Performance tests
echo ""
echo -e "${YELLOW}=== Performance Tests ===${NC}"
run_test "Performance benchmarks" \
    "docker build -f Dockerfile.validation --target perf-test -t agentdb-perf ." || true

# Test 8: Package integrity
echo ""
echo -e "${YELLOW}=== Package Integrity ===${NC}"
run_test "Package structure verification" \
    "docker run --rm agentdb-validation:latest sh -c 'ls -la dist/ && ls -la dist/cli/ && ls -la dist/mcp/'"

# Test 9: Dependencies check
run_test "Dependencies installation" \
    "docker run --rm agentdb-validation:latest sh -c 'npm list --depth=0'"

# Test 10: Build artifacts
run_test "Build artifacts verification" \
    "docker run --rm agentdb-validation:latest sh -c 'test -f dist/cli/agentdb-cli.js && test -f dist/mcp/agentdb-mcp-server.js && test -f dist/agentdb.min.js'"

# Generate summary report
echo ""
echo "==================================="
echo -e "${YELLOW}📊 Validation Summary${NC}"
echo "==================================="
echo ""
echo "Total Tests: $TOTAL_TESTS"
echo -e "${GREEN}Passed: $PASSED_TESTS${NC}"
echo -e "${RED}Failed: $FAILED_TESTS${NC}"
echo ""

# Calculate pass rate
if [ $TOTAL_TESTS -gt 0 ]; then
    PASS_RATE=$((PASSED_TESTS * 100 / TOTAL_TESTS))
    echo "Pass Rate: $PASS_RATE%"
    echo ""
fi

# Create detailed report
cat > "$RESULTS_DIR/VALIDATION_REPORT.md" << EOF
# AgentDB Docker Validation Report

**Date:** $(date)
**Version:** v1.5.0
**Total Tests:** $TOTAL_TESTS
**Passed:** $PASSED_TESTS
**Failed:** $FAILED_TESTS
**Pass Rate:** $PASS_RATE%

## Test Categories

### Security Tests
- SQL injection prevention
- Input validation
- eval() removal

### CLI Tests
- Database initialization
- Nested directory creation
- Command execution
- Error handling

### MCP Server Tests
- Server startup
- Stability (3-second test)
- Tool availability

### Test Suite
- Unit tests
- Integration tests
- Security tests

### Regression Tests
- Backend compatibility (better-sqlite3 / sql.js)
- Package exports
- Database creation

### Package Integrity
- Build artifacts
- Dependencies
- File structure

## Detailed Results

See individual log files in: \`$RESULTS_DIR/\`

## Recommendations

EOF

# Add recommendations based on results
if [ $FAILED_TESTS -eq 0 ]; then
    echo -e "${GREEN}✅ All tests passed! Ready for publishing.${NC}"
    echo "✅ **READY FOR PUBLISHING** - All validation tests passed." >> "$RESULTS_DIR/VALIDATION_REPORT.md"
elif [ $PASS_RATE -ge 80 ]; then
    echo -e "${YELLOW}⚠️ Some tests failed, but pass rate >= 80%. Review failures before publishing.${NC}"
    echo "⚠️ **REVIEW REQUIRED** - $PASS_RATE% pass rate. Review failures before publishing." >> "$RESULTS_DIR/VALIDATION_REPORT.md"
else
    echo -e "${RED}❌ Too many failures ($PASS_RATE% pass rate). DO NOT publish.${NC}"
    echo "❌ **NOT READY** - Only $PASS_RATE% pass rate. Critical issues must be fixed." >> "$RESULTS_DIR/VALIDATION_REPORT.md"
fi

echo ""
echo "Full report saved to: $RESULTS_DIR/VALIDATION_REPORT.md"
echo ""

# Return exit code based on results
if [ $FAILED_TESTS -eq 0 ]; then
    exit 0
else
    exit 1
fi
