#!/bin/bash
# AgentDB CLI Test Suite
# Comprehensive validation of all CLI commands

set -e

echo "🧪 AgentDB CLI Test Suite"
echo "========================="
echo ""

CLI="node dist/cli/agentdb-cli.js"

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test counter
PASSED=0
FAILED=0

# Test helper function
test_command() {
    local name="$1"
    local command="$2"
    local expected_pattern="$3"

    echo -n "Testing: $name... "

    if output=$(eval "$command" 2>&1); then
        if echo "$output" | grep -q "$expected_pattern"; then
            echo -e "${GREEN}✅ PASSED${NC}"
            PASSED=$((PASSED + 1))
            return 0
        else
            echo -e "${RED}❌ FAILED${NC} (unexpected output)"
            echo "Expected pattern: $expected_pattern"
            echo "Got: $output"
            FAILED=$((FAILED + 1))
            return 1
        fi
    else
        # Some commands might exit with non-zero but still work
        if echo "$output" | grep -q "$expected_pattern"; then
            echo -e "${GREEN}✅ PASSED${NC}"
            PASSED=$((PASSED + 1))
            return 0
        else
            echo -e "${RED}❌ FAILED${NC} (command failed)"
            echo "Output: $output"
            FAILED=$((FAILED + 1))
            return 1
        fi
    fi
}

# Setup test database
export AGENTDB_PATH="./test-agentdb.db"
rm -f "$AGENTDB_PATH"

echo "📦 Testing Help & Info Commands"
echo "--------------------------------"
test_command "Help command" "$CLI --help" "AgentDB CLI"
test_command "MCP in help" "$CLI --help" "MCP COMMANDS"
echo ""

echo "💾 Testing Database Commands"
echo "----------------------------"
test_command "Database stats" "$CLI db stats" "Database Statistics"
echo ""

echo "💭 Testing Reflexion Commands"
echo "-----------------------------"
test_command "Store episode" "$CLI reflexion store 'test-session' 'test_task' 0.9 true 'test critique'" "Stored episode"
test_command "Retrieve episodes" "$CLI reflexion retrieve 'test_task' 5" "Retrieved.*episodes"
test_command "Critique summary" "$CLI reflexion critique-summary 'test_task'" "Past Lessons"
echo ""

echo "🎯 Testing Skill Commands"
echo "-------------------------"
test_command "Create skill" "$CLI skill create 'test_skill' 'Test description' 'console.log(\"test\");'" "Created skill"
test_command "Search skills" "$CLI skill search 'test' 3" "Found.*skills"
test_command "Skill consolidate" "$CLI skill consolidate 2 0.5 14 true" "Created.*skills"
echo ""

echo "📊 Testing Causal Commands"
echo "--------------------------"
test_command "Add causal edge" "$CLI causal add-edge 'cause1' 'effect1' 0.3 0.9 50" "Added causal edge"
test_command "Create experiment" "$CLI causal experiment create 'test-exp' 'treatment' 'outcome'" "Created experiment"
test_command "Add observation" "$CLI causal experiment add-observation 1 true 0.8" "Recorded.*observation"
test_command "Calculate uplift" "$CLI causal experiment calculate 1" "Uplift"
test_command "Query edges" "$CLI causal query 'cause1' 'effect1' 0.7 0.1 10" "causal edges"
echo ""

echo "🔍 Testing Recall Commands"
echo "--------------------------"
test_command "Causal recall" "$CLI recall with-certificate 'test query' 5" "Causal Recall"
echo ""

echo "🌙 Testing Learner Commands"
echo "---------------------------"
test_command "Learner run" "$CLI learner run 2 0.5 0.6" "Discovered.*causal edges"
echo ""

echo "🚀 Testing MCP Command"
echo "----------------------"
# MCP command starts a server, so we just test that it's recognized
test_command "MCP start (timeout)" "timeout 1 $CLI mcp start 2>&1 || true" "AgentDB MCP Server"
echo ""

# Cleanup
rm -f "$AGENTDB_PATH"
rm -f "${AGENTDB_PATH}-shm"
rm -f "${AGENTDB_PATH}-wal"

echo "================================"
echo "📊 Test Results"
echo "================================"
echo -e "Passed: ${GREEN}$PASSED${NC}"
echo -e "Failed: ${RED}$FAILED${NC}"
echo "Total: $((PASSED + FAILED))"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✅ All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}❌ Some tests failed${NC}"
    exit 1
fi
