#!/bin/bash
# Run All Validation Tests
set -e

echo "=========================================="
echo "AgentDB v2 - Comprehensive Validation Suite"
echo "=========================================="
echo "Testing backward compatibility, CLI, MCP tools, and migration"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

PASSED=0
FAILED=0
WARNINGS=0

run_test() {
  local test_name="$1"
  local test_script="$2"

  echo ""
  echo "================================================"
  echo "Running: $test_name"
  echo "================================================"

  if bash "$test_script" 2>&1; then
    echo -e "${GREEN}✅ PASSED: $test_name${NC}"
    ((PASSED++))
  else
    local exit_code=$?
    if [ $exit_code -eq 2 ]; then
      echo -e "${YELLOW}⚠️  WARNING: $test_name${NC}"
      ((WARNINGS++))
    else
      echo -e "${RED}❌ FAILED: $test_name${NC}"
      ((FAILED++))
    fi
  fi
}

# Run all validation tests
run_test "v1 API Compatibility" "/test/validation/01-test-v1-compatibility.sh"
run_test "CLI Commands" "/test/validation/02-test-cli-commands.sh"
run_test "v2 New Features" "/test/validation/03-test-v2-features.sh"
run_test "MCP Tools Integration" "/test/validation/04-test-mcp-tools.sh"
run_test "v1 to v2 Migration" "/test/validation/05-test-migration.sh"

# Summary
echo ""
echo "=========================================="
echo "VALIDATION SUMMARY"
echo "=========================================="
echo -e "${GREEN}Passed: $PASSED${NC}"
echo -e "${YELLOW}Warnings: $WARNINGS${NC}"
echo -e "${RED}Failed: $FAILED${NC}"
echo ""

if [ $FAILED -eq 0 ]; then
  echo -e "${GREEN}✅ ALL VALIDATIONS PASSED${NC}"
  echo "AgentDB v2 is ready for production!"
  exit 0
else
  echo -e "${RED}❌ SOME VALIDATIONS FAILED${NC}"
  echo "Please review failed tests before proceeding."
  exit 1
fi
