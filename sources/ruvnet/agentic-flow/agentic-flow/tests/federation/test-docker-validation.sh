#!/bin/bash
# Docker-based Federation CLI Validation
# Validates federation CLI in clean Docker environment

set -e

echo "🐳 Docker-based Federation CLI Validation"
echo "════════════════════════════════════════════════════════════"
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

PASS=0
FAIL=0

test_cmd() {
    local name="$1"
    local cmd="$2"
    local pattern="$3"

    echo -n "Testing: $name ... "
    if eval "$cmd" 2>&1 | grep -q "$pattern"; then
        echo -e "${GREEN}✓ PASS${NC}"
        PASS=$((PASS + 1))
        return 0
    else
        echo -e "${RED}✗ FAIL${NC}"
        FAIL=$((FAIL + 1))
        return 1
    fi
}

echo "📋 Federation CLI Commands"
echo "────────────────────────────────────────────────────────────"

test_cmd "federation help" \
    "npx tsx src/cli/federation-cli.ts help" \
    "Federation Hub CLI"

test_cmd "federation status" \
    "npx tsx src/cli/federation-cli.ts status" \
    "FederationHubServer"

test_cmd "federation stats" \
    "npx tsx src/cli/federation-cli.ts stats" \
    "Federation Hub Statistics"

echo ""
echo "📋 Main CLI Integration"
echo "────────────────────────────────────────────────────────────"

test_cmd "main help" \
    "npx tsx src/cli-proxy.ts --help" \
    "FEDERATION COMMANDS"

test_cmd "federation routing" \
    "npx tsx src/cli-proxy.ts federation help" \
    "Federation Hub CLI"

echo ""
echo "📋 Regression Tests (Existing Features)"
echo "────────────────────────────────────────────────────────────"

test_cmd "agent list" \
    "npx tsx src/cli-proxy.ts --list" \
    "Available Agents"

test_cmd "agent manager" \
    "npx tsx src/cli-proxy.ts agent list" \
    "Available Agents"

test_cmd "config help" \
    "npx tsx src/cli-proxy.ts config help" \
    "Configuration Manager"

test_cmd "version" \
    "npx tsx src/cli-proxy.ts --version" \
    "agentic-flow"

echo ""
echo "════════════════════════════════════════════════════════════"
echo "📊 RESULTS"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "Total:  $((PASS + FAIL))"
echo -e "Passed: ${GREEN}$PASS${NC}"
echo -e "Failed: ${RED}$FAIL${NC}"
echo ""

if [ $FAIL -eq 0 ]; then
    echo -e "${GREEN}✅ ALL TESTS PASSED${NC}"
    echo ""
    echo "✓ Federation CLI working"
    echo "✓ No regressions detected"
    echo "✓ Ready for production"
    echo ""
    exit 0
else
    echo -e "${RED}❌ $FAIL TEST(S) FAILED${NC}"
    exit 1
fi
