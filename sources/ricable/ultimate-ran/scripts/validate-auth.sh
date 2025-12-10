#!/bin/bash
# OAuth Authentication Validation Script
# SUBSCRIPTION-ONLY MODE - Validates Claude Code and Google AI subscription setup
# Rejects pay-per-token API keys

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ENV_FILE="$PROJECT_ROOT/config/.env"

echo "========================================"
echo "TITAN SUBSCRIPTION-ONLY AUTH VALIDATOR"
echo "========================================"
echo ""
echo "Mode: SUBSCRIPTION ONLY (no pay-per-token API keys allowed)"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Load environment variables if .env exists
if [ -f "$ENV_FILE" ]; then
    echo "Loading environment from: $ENV_FILE"
    set -a
    source "$ENV_FILE"
    set +a
    echo ""
else
    echo -e "${RED}ERROR: config/.env not found${NC}"
    echo "   Run: cp config/.env.template config/.env"
    echo ""
    exit 1
fi

# Track validation status
ERRORS=0
WARNINGS=0

# ============================================
# SUBSCRIPTION-ONLY ENFORCEMENT
# ============================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "BLOCKED API KEYS CHECK"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Default to subscription-only mode (true unless explicitly set to false)
SUBSCRIPTION_ONLY="${SUBSCRIPTION_ONLY_MODE:-true}"
REJECT_API_KEYS="${REJECT_API_KEY_AUTH:-true}"

if [ "$SUBSCRIPTION_ONLY" != "false" ] || [ "$REJECT_API_KEYS" != "false" ]; then
    echo -e "${GREEN}✓${NC} Subscription-only mode: ENABLED (default)"

    # Check for blocked ANTHROPIC_API_KEY
    if [ -n "$ANTHROPIC_API_KEY" ]; then
        echo -e "${RED}✗ BLOCKED: ANTHROPIC_API_KEY detected!${NC}"
        echo "   Pay-per-token API keys are NOT allowed."
        echo "   Remove ANTHROPIC_API_KEY from your .env file."
        ERRORS=$((ERRORS + 1))
    else
        echo -e "${GREEN}✓${NC} No ANTHROPIC_API_KEY (correct - using subscription)"
    fi

    # Check for blocked GOOGLE_AI_API_KEY
    if [ -n "$GOOGLE_AI_API_KEY" ]; then
        echo -e "${RED}✗ BLOCKED: GOOGLE_AI_API_KEY detected!${NC}"
        echo "   Pay-per-token API keys are NOT allowed."
        echo "   Remove GOOGLE_AI_API_KEY and use OAuth instead."
        ERRORS=$((ERRORS + 1))
    else
        echo -e "${GREEN}✓${NC} No GOOGLE_AI_API_KEY (correct - using OAuth)"
    fi

    # Note: OPENROUTER_API_KEY is ALLOWED (optional, for additional model access)
    if [ -n "$OPENROUTER_API_KEY" ]; then
        echo -e "${GREEN}✓${NC} OPENROUTER_API_KEY configured (allowed)"
    else
        echo -e "${BLUE}ℹ${NC}  OPENROUTER_API_KEY not set (optional)"
    fi
else
    echo -e "${YELLOW}⚠${NC}  Subscription-only mode: DISABLED (not recommended)"
    echo "   Set SUBSCRIPTION_ONLY_MODE=true to enforce"
    echo "   Or remove the setting to use default (enabled)"
    WARNINGS=$((WARNINGS + 1))
fi

# Check for E2B (allowed)
if [ -n "$E2B_API_KEY" ] && [ "$E2B_API_KEY" != "e2b_your-key-here" ]; then
    echo -e "${GREEN}✓${NC} E2B_API_KEY configured (allowed)"
else
    echo -e "${BLUE}ℹ${NC}  E2B_API_KEY not set (optional - 100 hrs/month free)"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "1. CLAUDE CODE SUBSCRIPTION (OPUS 4.5)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check Claude Code OAuth
if command -v claude &> /dev/null; then
    CLAUDE_VERSION=$(claude --version 2>/dev/null || echo "unknown")
    echo -e "${GREEN}✓${NC} Claude Code CLI: $CLAUDE_VERSION"

    # Check if running within Claude Code (CLAUDECODE env var)
    if [ "$CLAUDECODE" = "1" ]; then
        echo -e "${GREEN}✓${NC} Running within Claude Code CLI (subscription active)"
        echo -e "${BLUE}ℹ${NC}  Authentication handled automatically"
    elif [ -n "$CLAUDE_CODE_OAUTH_TOKEN" ]; then
        echo -e "${GREEN}✓${NC} CLAUDE_CODE_OAUTH_TOKEN configured"
        echo -e "${BLUE}ℹ${NC}  Using OAuth subscription (no per-token costs)"
    else
        echo -e "${YELLOW}⚠${NC}  No OAuth token detected"
        echo "   If running outside Claude Code CLI:"
        echo "   1. Run: claude login"
        echo "   2. Run: claude setup-token"
        echo "   3. Set: CLAUDE_CODE_OAUTH_TOKEN=<your-token>"
        WARNINGS=$((WARNINGS + 1))
    fi

    # Check model configuration
    MODEL="${ANTHROPIC_MODEL:-claude-opus-4-5-20251101}"
    if [[ "$MODEL" == "claude-opus-4-5-20251101" ]]; then
        echo -e "${GREEN}✓${NC} Model: $MODEL (Opus 4.5 - Latest)"
    else
        echo -e "${YELLOW}⚠${NC}  Model: $MODEL"
        echo "   Recommended: claude-opus-4-5-20251101 (Opus 4.5)"
        WARNINGS=$((WARNINGS + 1))
    fi
else
    echo -e "${RED}✗${NC} Claude Code CLI not installed"
    echo "   Install: npm install -g @anthropic-ai/claude-code"
    ERRORS=$((ERRORS + 1))
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "2. GOOGLE AI PRO SUBSCRIPTION (GEMINI 3 PRO)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check Google AI OAuth (subscription required)
if [ -n "$GOOGLE_OAUTH_CLIENT_ID" ] && [ "$GOOGLE_OAUTH_CLIENT_ID" != "YOUR_CLIENT_ID.apps.googleusercontent.com" ]; then
    if [ -n "$GOOGLE_OAUTH_CLIENT_SECRET" ] && [ "$GOOGLE_OAUTH_CLIENT_SECRET" != "YOUR_CLIENT_SECRET" ]; then
        echo -e "${GREEN}✓${NC} Google OAuth credentials configured"
        echo -e "${BLUE}ℹ${NC}  Using AI Pro subscription (no per-token costs)"
        echo "   Account: cedricable@gmail.com"
    else
        echo -e "${RED}✗${NC} GOOGLE_OAUTH_CLIENT_SECRET not set"
        echo "   Complete OAuth setup in Google Cloud Console"
        ERRORS=$((ERRORS + 1))
    fi
else
    echo -e "${RED}✗${NC} Google OAuth credentials NOT configured"
    echo ""
    echo "   Setup Required:"
    echo "   1. Go to: https://console.cloud.google.com/"
    echo "   2. Create project: TITAN-RAN"
    echo "   3. Enable: Generative Language API"
    echo "   4. Configure OAuth consent screen"
    echo "   5. Create OAuth 2.0 Client ID (Desktop app)"
    echo "   6. Add to config/.env:"
    echo "      GOOGLE_OAUTH_CLIENT_ID=<your-client-id>"
    echo "      GOOGLE_OAUTH_CLIENT_SECRET=<your-secret>"
    echo ""
    echo "   Full guide: docs/AUTH.md"
    ERRORS=$((ERRORS + 1))
fi

# Check model configuration
GOOGLE_MODEL="${GOOGLE_AI_MODEL:-gemini-3-pro-preview}"
if [[ "$GOOGLE_MODEL" == "gemini-3-pro-preview" ]]; then
    echo -e "${GREEN}✓${NC} Model: $GOOGLE_MODEL (Gemini 3 Pro - Latest)"
else
    echo -e "${YELLOW}⚠${NC}  Model: $GOOGLE_MODEL"
    echo "   Recommended: gemini-3-pro-preview (Gemini 3 Pro)"
    WARNINGS=$((WARNINGS + 1))
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "3. MULTI-PROVIDER STRATEGY"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

STRATEGY="${AGENTIC_FLOW_STRATEGY:-consensus}"
PRIMARY="${AGENTIC_FLOW_PRIMARY_PROVIDER:-anthropic}"
SECONDARY="${AGENTIC_FLOW_SECONDARY_PROVIDER:-google}"

echo "Strategy: $STRATEGY"
echo "Primary: $PRIMARY"
echo "Secondary: $SECONDARY"

case "$STRATEGY" in
    "consensus")
        echo -e "${GREEN}✓${NC} Consensus mode (95%+ confidence)"
        echo -e "${BLUE}ℹ${NC}  Both Claude & Gemini must agree (production recommended)"
        ;;
    "claude_primary")
        echo -e "${BLUE}ℹ${NC}  Claude primary mode (85-90% confidence)"
        ;;
    "gemini_primary")
        echo -e "${BLUE}ℹ${NC}  Gemini primary mode (80-85% confidence)"
        ;;
    "single")
        echo -e "${BLUE}ℹ${NC}  Single provider mode ($PRIMARY only)"
        ;;
    "parallel")
        echo -e "${BLUE}ℹ${NC}  Parallel mode (70-80% confidence, fastest)"
        ;;
    *)
        echo -e "${YELLOW}⚠${NC}  Unknown strategy: $STRATEGY"
        WARNINGS=$((WARNINGS + 1))
        ;;
esac

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "4. ADDITIONAL SERVICES"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check E2B API (free tier, not pay-per-token)
if [ -n "$E2B_API_KEY" ] && [ "$E2B_API_KEY" != "e2b_your-key-here" ]; then
    echo -e "${GREEN}✓${NC} E2B API configured (sandbox execution)"
else
    echo -e "${YELLOW}⚠${NC}  E2B API not configured"
    echo "   Free tier: 100 hours/month"
    echo "   Get key: https://e2b.dev/docs"
    WARNINGS=$((WARNINGS + 1))
fi

# Runtime mode
RUNTIME="${RUNTIME_MODE:-local}"
echo -e "${BLUE}ℹ${NC}  Runtime mode: $RUNTIME"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "VALIDATION SUMMARY"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ $ERRORS -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo -e "${GREEN}PERFECT! All subscriptions configured correctly${NC}"
    echo ""
    echo "Cost Summary:"
    echo "  Claude Pro/Max: \$20-40/month (subscription)"
    echo "  Google AI Pro:  \$19.99/month (subscription)"
    echo "  Per-token API:  \$0 (BLOCKED)"
    echo ""
    echo "Next steps:"
    echo "  1. Build: npm run build"
    echo "  2. Test: npm run test:integration"
    echo "  3. Start: npm run start:local"
    exit 0
elif [ $ERRORS -eq 0 ]; then
    echo -e "${YELLOW}WARNINGS: $WARNINGS${NC}"
    echo "Configuration will work but review warnings above"
    echo ""
    echo "Next steps:"
    echo "  1. Fix warnings (optional)"
    echo "  2. Build: npm run build"
    echo "  3. Start: npm run start:local"
    exit 0
else
    echo -e "${RED}ERRORS: $ERRORS${NC}"
    if [ $WARNINGS -gt 0 ]; then
        echo -e "${YELLOW}WARNINGS: $WARNINGS${NC}"
    fi
    echo ""
    echo "Fix errors before starting TITAN"
    echo ""
    echo "Required for subscription-only mode:"
    echo "  1. Remove any API keys (ANTHROPIC_API_KEY, GOOGLE_AI_API_KEY)"
    echo "  2. Claude: claude login (run within Claude Code)"
    echo "  3. Google: Configure OAuth in Cloud Console"
    echo "  4. See: docs/AUTH.md for full setup guide"
    exit 1
fi
