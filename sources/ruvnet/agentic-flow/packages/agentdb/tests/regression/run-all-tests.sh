#!/bin/bash
# AgentDB v1.6.0 Comprehensive Regression Test Suite Runner
# Runs all regression tests and generates detailed report

set -e

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

echo -e "${BOLD}${CYAN}"
echo "╔════════════════════════════════════════════════════════════╗"
echo "║   AgentDB v1.6.0 Comprehensive Regression Test Suite      ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

REPORT_FILE="./tests/regression/test-report-$(date +%Y%m%d-%H%M%S).md"
PASSED=0
FAILED=0
TOTAL=0

# Create report header
cat > "$REPORT_FILE" << EOF
# AgentDB v1.6.0 Regression Test Report
**Generated:** $(date)
**Version:** 1.6.0

## Test Summary

EOF

echo -e "${BOLD}${BLUE}📋 Test Configuration${NC}"
echo "  Report file: $REPORT_FILE"
echo ""

# Function to run test suite
run_test_suite() {
    local name="$1"
    local command="$2"
    local description="$3"

    echo -e "${BOLD}${BLUE}═══════════════════════════════════════════════════════${NC}"
    echo -e "${BOLD}${CYAN}🧪 Running: $name${NC}"
    echo -e "${BLUE}Description: $description${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
    echo ""

    # Run test and capture output
    if eval "$command" > /tmp/test-output.log 2>&1; then
        echo -e "${GREEN}✅ PASSED: $name${NC}"
        echo ""
        echo "### ✅ $name - PASSED" >> "$REPORT_FILE"
        echo "" >> "$REPORT_FILE"
        echo "**Description:** $description" >> "$REPORT_FILE"
        echo "" >> "$REPORT_FILE"
        echo '```' >> "$REPORT_FILE"
        cat /tmp/test-output.log >> "$REPORT_FILE"
        echo '```' >> "$REPORT_FILE"
        echo "" >> "$REPORT_FILE"
        PASSED=$((PASSED + 1))
    else
        echo -e "${RED}❌ FAILED: $name${NC}"
        echo -e "${YELLOW}Output:${NC}"
        cat /tmp/test-output.log
        echo ""
        echo "### ❌ $name - FAILED" >> "$REPORT_FILE"
        echo "" >> "$REPORT_FILE"
        echo "**Description:** $description" >> "$REPORT_FILE"
        echo "" >> "$REPORT_FILE"
        echo '```' >> "$REPORT_FILE"
        cat /tmp/test-output.log >> "$REPORT_FILE"
        echo '```' >> "$REPORT_FILE"
        echo "" >> "$REPORT_FILE"
        FAILED=$((FAILED + 1))
    fi
    TOTAL=$((TOTAL + 1))
    echo ""
}

# 1. Build Tests
echo -e "${BOLD}${YELLOW}🔨 Phase 1: Build & Compilation Tests${NC}"
echo ""

run_test_suite \
    "TypeScript Build" \
    "npm run build" \
    "Verify TypeScript compilation with zero errors"

run_test_suite \
    "Type Checking" \
    "npx tsc --noEmit" \
    "Verify all types are correct and no type errors exist"

# 2. Unit Tests
echo -e "${BOLD}${YELLOW}🧪 Phase 2: Unit Tests${NC}"
echo ""

run_test_suite \
    "Core Features Unit Tests" \
    "npx vitest run tests/regression/core-features.test.ts" \
    "Test reflexion memory, skill library, and causal memory graph"

run_test_suite \
    "Build Validation Tests" \
    "npx vitest run tests/regression/build-validation.test.ts" \
    "Test import resolution, package structure, and dependencies"

# 3. v1.6.0 Feature Tests
echo -e "${BOLD}${YELLOW}🆕 Phase 3: v1.6.0 New Features${NC}"
echo ""

run_test_suite \
    "Vector Search Tests" \
    "npx vitest run tests/regression/v1.6.0-features.test.ts -t 'Vector Search'" \
    "Test vector-search command with all distance metrics"

run_test_suite \
    "Export/Import Tests" \
    "npx vitest run tests/regression/v1.6.0-features.test.ts -t 'Export/Import'" \
    "Test database export and import functionality"

run_test_suite \
    "Stats Command Tests" \
    "npx vitest run tests/regression/v1.6.0-features.test.ts -t 'Stats'" \
    "Test database statistics reporting"

run_test_suite \
    "Enhanced Init Tests" \
    "npx vitest run tests/regression/v1.6.0-features.test.ts -t 'Init'" \
    "Test enhanced init command with --dimension, --preset, --in-memory flags"

# 4. Integration Tests
echo -e "${BOLD}${YELLOW}🔗 Phase 4: Integration Tests${NC}"
echo ""

run_test_suite \
    "Full Workflow Integration" \
    "npx vitest run tests/regression/integration.test.ts -t 'Full Workflow'" \
    "Test complete init → store → export → import → verify workflow"

run_test_suite \
    "Memory Persistence Tests" \
    "npx vitest run tests/regression/integration.test.ts -t 'Memory Persistence'" \
    "Test data persistence across operations and restarts"

run_test_suite \
    "Error Handling Tests" \
    "npx vitest run tests/regression/integration.test.ts -t 'Error Handling'" \
    "Test graceful handling of invalid inputs and edge cases"

run_test_suite \
    "Causal Recall Integration" \
    "npx vitest run tests/regression/integration.test.ts -t 'Causal Recall'" \
    "Test causal recall with certificate generation"

run_test_suite \
    "Pattern Discovery Tests" \
    "npx vitest run tests/regression/integration.test.ts -t 'Nightly Learner'" \
    "Test automated pattern discovery from episodes"

run_test_suite \
    "Skill Consolidation Tests" \
    "npx vitest run tests/regression/integration.test.ts -t 'Skill Consolidation'" \
    "Test episode consolidation into skills with pattern extraction"

run_test_suite \
    "Concurrent Operations Tests" \
    "npx vitest run tests/regression/integration.test.ts -t 'Concurrent'" \
    "Test concurrent episode storage and queries"

# 5. CLI Tests
echo -e "${BOLD}${YELLOW}⌨️  Phase 5: CLI Command Tests${NC}"
echo ""

run_test_suite \
    "CLI Command Suite" \
    "bash tests/cli-test-suite.sh" \
    "Test all CLI commands including reflexion, skill, causal, recall, learner"

# 6. MCP Server Tests (if available)
echo -e "${BOLD}${YELLOW}🔌 Phase 6: MCP Server Tests${NC}"
echo ""

if [ -f "tests/mcp-tools.test.ts" ]; then
    run_test_suite \
        "MCP Server Tests" \
        "npx vitest run tests/mcp-tools.test.ts" \
        "Test MCP server tools and integration"
else
    echo -e "${YELLOW}⏭️  Skipping MCP tests (file not found)${NC}"
    echo ""
fi

# Generate summary
echo -e "${BOLD}${BLUE}═══════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}${CYAN}📊 Test Results Summary${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
echo ""
echo -e "Total Tests:  ${BOLD}$TOTAL${NC}"
echo -e "✅ Passed:     ${GREEN}${BOLD}$PASSED${NC}"
echo -e "❌ Failed:     ${RED}${BOLD}$FAILED${NC}"

if [ $TOTAL -gt 0 ]; then
    SUCCESS_RATE=$((PASSED * 100 / TOTAL))
    echo -e "Success Rate: ${BOLD}${SUCCESS_RATE}%${NC}"
fi

echo ""

# Add summary to report
cat >> "$REPORT_FILE" << EOF

---

## Summary Statistics

- **Total Test Suites:** $TOTAL
- **Passed:** $PASSED ✅
- **Failed:** $FAILED ❌
- **Success Rate:** $((PASSED * 100 / TOTAL))%

## Test Coverage

### Core Features ✅
- Reflexion Memory (store, retrieve, prune)
- Skill Library (create, search, consolidate)
- Causal Memory Graph (edges, experiments, observations)
- Database persistence

### v1.6.0 Features 🆕
- Vector search with cosine/euclidean/dot metrics
- Export/import database to JSON
- Stats command
- Enhanced init with --dimension, --preset, --in-memory

### Build Validation ✅
- TypeScript compilation
- Import resolution
- Package structure
- Dependencies

### Integration ✅
- Full workflows
- Memory persistence
- Error handling
- Concurrent operations

### CLI Commands ✅
- All reflexion commands
- All skill commands
- All causal commands
- Recall with certificate
- Learner discovery

## Regression Status

EOF

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}${BOLD}✅ All regression tests passed! No regressions detected.${NC}"
    echo "**✅ NO REGRESSIONS DETECTED** - All functionality working as expected." >> "$REPORT_FILE"
    EXIT_CODE=0
else
    echo -e "${RED}${BOLD}❌ Some tests failed. Regressions detected.${NC}"
    echo "**⚠️ REGRESSIONS DETECTED** - $FAILED test suite(s) failed." >> "$REPORT_FILE"
    EXIT_CODE=1
fi

echo ""
echo -e "${BOLD}${CYAN}📄 Full report saved to: $REPORT_FILE${NC}"
echo ""

# Open report in less if available
if command -v less &> /dev/null; then
    echo -e "${YELLOW}Press any key to view report, or Ctrl+C to exit...${NC}"
    read -n 1 -s
    less "$REPORT_FILE"
fi

exit $EXIT_CODE
