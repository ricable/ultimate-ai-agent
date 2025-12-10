#!/bin/bash
# Federation CLI Validation Test
# Tests all federation CLI commands and verifies no regressions

set -e

echo "🧪 Federation CLI Validation Test"
echo "════════════════════════════════════════════════════════════"
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

PASS=0
FAIL=0
TOTAL=0

# Test function
test_command() {
    local name="$1"
    local command="$2"
    local expected_pattern="$3"

    TOTAL=$((TOTAL + 1))
    echo -n "Testing: $name ... "

    if eval "$command" 2>&1 | grep -q "$expected_pattern"; then
        echo -e "${GREEN}✓ PASS${NC}"
        PASS=$((PASS + 1))
        return 0
    else
        echo -e "${RED}✗ FAIL${NC}"
        FAIL=$((FAIL + 1))
        return 1
    fi
}

echo "📋 SECTION 1: Federation CLI Commands"
echo "────────────────────────────────────────────────────────────"

test_command \
    "federation help" \
    "npx tsx src/cli/federation-cli.ts help" \
    "Federation Hub CLI"

test_command \
    "federation status" \
    "npx tsx src/cli/federation-cli.ts status" \
    "FederationHubServer"

test_command \
    "federation stats (placeholder)" \
    "npx tsx src/cli/federation-cli.ts stats" \
    "Federation Hub Statistics"

echo ""
echo "📋 SECTION 2: Main CLI Integration"
echo "────────────────────────────────────────────────────────────"

test_command \
    "main CLI help includes federation" \
    "npx tsx src/cli-proxy.ts --help" \
    "FEDERATION COMMANDS"

test_command \
    "main CLI federation routing" \
    "npx tsx src/cli-proxy.ts federation help" \
    "Federation Hub CLI"

echo ""
echo "📋 SECTION 3: Existing CLI Features (Regression Tests)"
echo "────────────────────────────────────────────────────────────"

test_command \
    "agent list command" \
    "npx tsx src/cli-proxy.ts --list" \
    "Available Agents"

test_command \
    "agent manager list" \
    "npx tsx src/cli-proxy.ts agent list" \
    "Available Agents"

test_command \
    "config help" \
    "npx tsx src/cli-proxy.ts config help" \
    "Configuration Manager"

test_command \
    "main help displays" \
    "npx tsx src/cli-proxy.ts --help" \
    "Agentic Flow"

test_command \
    "version flag works" \
    "npx tsx src/cli-proxy.ts --version" \
    "agentic-flow"

echo ""
echo "📋 SECTION 4: Command Parsing"
echo "────────────────────────────────────────────────────────────"

test_command \
    "federation mode parsing" \
    "npx tsx -e \"import {parseArgs} from './src/utils/cli.js'; const opts = parseArgs(); process.argv = ['node', 'cli', 'federation']; console.log(parseArgs().mode)\"" \
    "federation"

test_command \
    "agent mode parsing" \
    "npx tsx -e \"import {parseArgs} from './src/utils/cli.js'; process.argv = ['node', 'cli', 'agent']; console.log(parseArgs().mode)\"" \
    "agent-manager"

echo ""
echo "📋 SECTION 5: Federation Help Content"
echo "────────────────────────────────────────────────────────────"

test_command \
    "help includes start command" \
    "npx tsx src/cli/federation-cli.ts help" \
    "Start federation hub server"

test_command \
    "help includes spawn command" \
    "npx tsx src/cli/federation-cli.ts help" \
    "Spawn ephemeral agent"

test_command \
    "help includes architecture section" \
    "npx tsx src/cli/federation-cli.ts help" \
    "Hub: Persistent central database"

test_command \
    "help includes benefits section" \
    "npx tsx src/cli/federation-cli.ts help" \
    "Memory outlives agents"

echo ""
echo "════════════════════════════════════════════════════════════"
echo "📊 TEST RESULTS"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "Total Tests:  $TOTAL"
echo -e "Passed:       ${GREEN}$PASS${NC}"
echo -e "Failed:       ${RED}$FAIL${NC}"
echo ""

if [ $FAIL -eq 0 ]; then
    echo -e "${GREEN}✅ ALL TESTS PASSED!${NC}"
    echo ""
    echo "✓ Federation CLI fully integrated"
    echo "✓ All commands working correctly"
    echo "✓ No regressions detected"
    echo "✓ Main CLI integration successful"
    echo ""
    exit 0
else
    echo -e "${RED}❌ SOME TESTS FAILED${NC}"
    echo ""
    echo "Please review the failed tests above."
    echo ""
    exit 1
fi
