#!/bin/bash
# Supabase Integration Validation Script
# Runs comprehensive tests and generates report

set -e

echo "🧪 Supabase Integration Validation"
echo "═══════════════════════════════════════════════════════════"
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if running in CI
if [ -n "$CI" ]; then
    echo "🤖 Running in CI mode"
    CI_MODE=true
else
    CI_MODE=false
fi

# Check for Supabase credentials
if [ -n "$SUPABASE_URL" ] && [ -n "$SUPABASE_ANON_KEY" ]; then
    echo -e "${GREEN}✅ Supabase credentials detected${NC}"
    echo "   URL: $SUPABASE_URL"
    echo "   Mode: LIVE"
    TEST_MODE="live"
else
    echo -e "${YELLOW}⚠️  No Supabase credentials detected${NC}"
    echo "   Mode: MOCK"
    TEST_MODE="mock"
fi

echo ""
echo "─────────────────────────────────────────────────────────"
echo "📋 Pre-flight Checks"
echo "─────────────────────────────────────────────────────────"

# Check Node.js version
NODE_VERSION=$(node --version)
echo "Node.js: $NODE_VERSION"

# Check npm packages
if npm list @supabase/supabase-js >/dev/null 2>&1; then
    echo -e "${GREEN}✅ @supabase/supabase-js installed${NC}"
else
    echo -e "${RED}❌ @supabase/supabase-js not installed${NC}"
    echo "   Installing..."
    npm install @supabase/supabase-js
fi

# Check TypeScript executor
if command -v tsx &> /dev/null; then
    echo -e "${GREEN}✅ tsx available${NC}"
else
    echo -e "${RED}❌ tsx not available${NC}"
    echo "   Installing..."
    npm install -g tsx
fi

echo ""
echo "─────────────────────────────────────────────────────────"
echo "🧪 Running Tests"
echo "─────────────────────────────────────────────────────────"
echo ""

# Run tests
if npx tsx tests/supabase/test-integration.ts; then
    TEST_RESULT="PASSED"
    TEST_EXIT_CODE=0
else
    TEST_RESULT="FAILED"
    TEST_EXIT_CODE=1
fi

echo ""
echo "─────────────────────────────────────────────────────────"
echo "📊 Validation Report"
echo "─────────────────────────────────────────────────────────"
echo ""
echo "Test Mode:     $TEST_MODE"
echo "Test Result:   $TEST_RESULT"
echo "Timestamp:     $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
echo ""

if [ "$TEST_RESULT" = "PASSED" ]; then
    echo -e "${GREEN}✅ VALIDATION SUCCESSFUL${NC}"
    echo ""

    if [ "$TEST_MODE" = "mock" ]; then
        echo "ℹ️  Tests ran in MOCK mode"
        echo ""
        echo "To run LIVE tests, set these environment variables:"
        echo "  export SUPABASE_URL=\"https://your-project.supabase.co\""
        echo "  export SUPABASE_ANON_KEY=\"your-anon-key\""
        echo "  export SUPABASE_SERVICE_ROLE_KEY=\"your-service-role-key\""
    else
        echo -e "${GREEN}🎉 Live Supabase integration validated!${NC}"
    fi
else
    echo -e "${RED}❌ VALIDATION FAILED${NC}"
    echo ""
    echo "Check the test output above for details."
fi

echo ""
echo "─────────────────────────────────────────────────────────"
echo "📚 Resources"
echo "─────────────────────────────────────────────────────────"
echo ""
echo "Documentation:"
echo "  • Quickstart:    docs/supabase/QUICKSTART.md"
echo "  • Full Guide:    docs/supabase/SUPABASE-REALTIME-FEDERATION.md"
echo "  • Test Details:  tests/supabase/README.md"
echo ""
echo "Examples:"
echo "  • Working code:  examples/realtime-federation-example.ts"
echo ""

exit $TEST_EXIT_CODE
