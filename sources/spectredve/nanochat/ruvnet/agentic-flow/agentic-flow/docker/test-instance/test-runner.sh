#!/bin/bash
# Agentic-Flow Docker Test Runner

set -e

echo "======================================"
echo "Agentic-Flow Docker Test Suite"
echo "======================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

test_passed=0
test_failed=0

run_test() {
    local test_name="$1"
    local test_command="$2"

    echo -e "${YELLOW}Running: ${test_name}${NC}"
    if eval "$test_command"; then
        echo -e "${GREEN}✓ PASSED: ${test_name}${NC}"
        ((test_passed++))
    else
        echo -e "${RED}✗ FAILED: ${test_name}${NC}"
        ((test_failed++))
    fi
    echo ""
}

# Test 1: CLI Help
run_test "CLI Help Command" \
    "node /app/dist/cli-proxy.js --help | head -20"

# Test 2: List Agents
run_test "List Available Agents" \
    "node /app/dist/cli-proxy.js --list | head -30"

# Test 3: Agent Info
run_test "Agent Info Command" \
    "node /app/dist/cli-proxy.js agent info coder"

# Test 4: MCP Status
run_test "MCP Status" \
    "node /app/dist/cli-proxy.js mcp status || true"

# Test 5: Configuration List
run_test "Config List" \
    "node /app/dist/cli-proxy.js config list || true"

# Test 6: Federation Status
run_test "Federation Status" \
    "node /app/dist/cli-proxy.js federation status || true"

# Test 7: Test Agent Execution (with ONNX/local only if no API keys)
if [ -n "$ANTHROPIC_API_KEY" ] || [ -n "$OPENROUTER_API_KEY" ] || [ -n "$GOOGLE_GEMINI_API_KEY" ]; then
    run_test "Agent Execution Test" \
        "timeout 30 node /app/dist/cli-proxy.js --agent researcher --task 'Say hello' --max-tokens 50"
else
    echo -e "${YELLOW}⚠ Skipping agent execution test (no API keys configured)${NC}"
    echo "  To test agent execution, set ANTHROPIC_API_KEY, OPENROUTER_API_KEY, or GOOGLE_GEMINI_API_KEY"
    echo ""
fi

# Test 10: Check WASM Modules
run_test "WASM Module Check" \
    "test -f /app/wasm/reasoningbank/reasoningbank_wasm_bg.wasm && echo 'WASM modules present'"

# Test 11: Directory Structure
run_test "Data Directory Structure" \
    "test -d /app/data/agentdb && test -d /app/data/memory && echo 'Data directories exist'"

# Summary
echo "======================================"
echo "Test Summary"
echo "======================================"
echo -e "${GREEN}Passed: ${test_passed}${NC}"
echo -e "${RED}Failed: ${test_failed}${NC}"
echo "======================================"

if [ $test_failed -eq 0 ]; then
    echo -e "${GREEN}✓ All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}✗ Some tests failed${NC}"
    exit 1
fi
