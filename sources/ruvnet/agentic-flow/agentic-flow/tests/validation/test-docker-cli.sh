#!/bin/bash
# Validation script for Docker CLI functionality
# Tests different ways to run agents in Docker

set -e

echo "🧪 Docker CLI Validation Test"
echo "================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test counter
TESTS_PASSED=0
TESTS_FAILED=0

# Helper function to test
test_command() {
  local test_name="$1"
  local command="$2"
  local expected_pattern="$3"

  echo -e "${YELLOW}Testing: ${test_name}${NC}"
  echo "Command: $command"

  if eval "$command" 2>&1 | grep -q "$expected_pattern"; then
    echo -e "${GREEN}✅ PASS${NC}"
    ((TESTS_PASSED++))
  else
    echo -e "${RED}❌ FAIL${NC}"
    ((TESTS_FAILED++))
  fi
  echo ""
}

echo "📦 Test 1: List Agents"
test_command \
  "List all available agents" \
  "docker run claude-agents:cli --list" \
  "Available Agents"

echo "🤖 Test 2: Run Goal Planner Agent"
test_command \
  "Run goal-planner with simple task" \
  "docker run --env-file ../../.env claude-agents:cli --agent goal-planner --task 'Plan test'" \
  "Step"

echo "💻 Test 3: Run Coder Agent"
test_command \
  "Run coder with simple task" \
  "docker run --env-file ../../.env claude-agents:cli --agent coder --task 'Write hello function'" \
  "function\|def\|const"

echo "📋 Test 4: Help Flag"
test_command \
  "Show help message" \
  "docker run claude-agents:cli --help" \
  "USAGE"

echo "🎯 Test 5: Verify Agent Count"
test_command \
  "Verify agents loaded" \
  "docker run claude-agents:cli --list" \
  "60\|65\|70"

echo ""
echo "================================"
echo "📊 Test Results"
echo "================================"
echo -e "Passed: ${GREEN}${TESTS_PASSED}${NC}"
echo -e "Failed: ${RED}${TESTS_FAILED}${NC}"
echo -e "Total:  $((TESTS_PASSED + TESTS_FAILED))"
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
  echo -e "${GREEN}✅ All tests passed!${NC}"
  exit 0
else
  echo -e "${RED}❌ Some tests failed${NC}"
  exit 1
fi
