#!/bin/bash
# Interactive OAuth Setup Script for TITAN
# Guides users through Claude Code and Google AI subscription setup

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ENV_FILE="$PROJECT_ROOT/config/.env"
ENV_TEMPLATE="$PROJECT_ROOT/config/.env.template"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

clear
echo -e "${CYAN}"
cat << "EOF"
╔════════════════════════════════════════════════════════════╗
║                                                            ║
║          TITAN OAuth Setup Assistant                       ║
║      Subscription-Based AI Authentication                  ║
║                                                            ║
╚════════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

echo "This script will help you set up subscription-based authentication"
echo "for Claude Opus 4.5 and Gemini 3 Pro (no per-token costs)."
echo ""

# Check if .env exists
if [ ! -f "$ENV_FILE" ]; then
    echo -e "${YELLOW}⚠️  config/.env not found${NC}"
    echo ""
    read -p "Create from template? (y/n): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        cp "$ENV_TEMPLATE" "$ENV_FILE"
        echo -e "${GREEN}✓${NC} Created config/.env from template"
    else
        echo -e "${RED}✗${NC} Setup cancelled"
        exit 1
    fi
else
    echo -e "${GREEN}✓${NC} Found existing config/.env"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 1: Claude Code Subscription (Opus 4.5)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check if Claude CLI is installed
if ! command -v claude &> /dev/null; then
    echo -e "${RED}✗${NC} Claude Code CLI not found"
    echo ""
    echo "Install Claude Code CLI:"
    echo -e "${CYAN}npm install -g @anthropic-ai/claude-code${NC}"
    echo ""
    read -p "Press Enter after installing Claude Code..."

    # Re-check after installation
    if ! command -v claude &> /dev/null; then
        echo -e "${RED}✗${NC} Claude Code CLI still not found. Please install and re-run."
        exit 1
    fi
fi

echo -e "${GREEN}✓${NC} Claude Code CLI found"

# Check if already logged in by trying to get account info
echo ""
echo "Checking Claude Code authentication status..."

# Try to run claude setup-token to check if logged in
OAUTH_TOKEN=""
if claude setup-token --help &> /dev/null 2>&1; then
    echo -e "${BLUE}ℹ${NC}  Attempting to generate OAuth token automatically..."
    echo ""

    # Try to generate token automatically
    # This will fail if not logged in, which is fine
    OAUTH_TOKEN=$(claude setup-token 2>/dev/null || echo "")

    if [ -n "$OAUTH_TOKEN" ] && [ "$OAUTH_TOKEN" != "" ]; then
        echo -e "${GREEN}✓${NC} OAuth token generated automatically!"
    else
        echo -e "${YELLOW}⚠${NC}  Could not generate token automatically."
        echo ""
        echo "Please login to Claude Code first:"
        echo -e "${CYAN}claude login${NC}"
        echo ""
        read -p "Press Enter after completing 'claude login'..."

        # Try again after login
        echo ""
        echo "Generating OAuth token..."
        OAUTH_TOKEN=$(claude setup-token 2>/dev/null || echo "")

        if [ -z "$OAUTH_TOKEN" ]; then
            echo -e "${YELLOW}⚠${NC}  Auto-generation failed. Please generate manually:"
            echo -e "${CYAN}claude setup-token${NC}"
            echo ""
            read -p "Paste your OAuth token here: " OAUTH_TOKEN
        fi
    fi
else
    echo -e "${YELLOW}⚠${NC}  claude setup-token not available"
    echo "Please ensure you have the latest Claude Code CLI"
    echo ""
    echo "Manual steps:"
    echo "  1. Run: ${CYAN}claude login${NC}"
    echo "  2. Run: ${CYAN}claude setup-token${NC}"
    echo "  3. Copy the token"
    echo ""
    read -p "Paste your OAuth token here: " OAUTH_TOKEN
fi

# Save the token if we got one
if [ -n "$OAUTH_TOKEN" ] && [ "$OAUTH_TOKEN" != "" ]; then
    # Update .env file - remove any existing entries first
    sed -i.bak '/^CLAUDE_CODE_OAUTH_TOKEN=/d' "$ENV_FILE"
    sed -i.bak '/^# CLAUDE_CODE_OAUTH_TOKEN=/d' "$ENV_FILE"

    # Add the new token
    echo "" >> "$ENV_FILE"
    echo "# Claude Code OAuth Token (auto-generated $(date +%Y-%m-%d))" >> "$ENV_FILE"
    echo "CLAUDE_CODE_OAUTH_TOKEN=$OAUTH_TOKEN" >> "$ENV_FILE"

    # Update model to Opus 4.5 (latest)
    sed -i.bak "s|^ANTHROPIC_MODEL=.*|ANTHROPIC_MODEL=claude-opus-4-5-20251101|" "$ENV_FILE"
    if ! grep -q "^ANTHROPIC_MODEL=" "$ENV_FILE"; then
        echo "ANTHROPIC_MODEL=claude-opus-4-5-20251101" >> "$ENV_FILE"
    fi

    echo -e "${GREEN}✓${NC} Claude Code OAuth token saved to config/.env"
    echo -e "${GREEN}✓${NC} Model set to: claude-opus-4-5-20251101 (Opus 4.5)"
    rm -f "$ENV_FILE.bak"
else
    echo -e "${YELLOW}⚠${NC}  No OAuth token configured"
    echo "   You can set it later by running this script again"
    echo "   Or manually: CLAUDE_CODE_OAUTH_TOKEN=<your-token>"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 2: Google AI Pro Subscription (Gemini 3 Pro)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo "For Google AI Pro/Ultra subscription (cedricable@gmail.com):"
echo ""
echo "1. Subscribe at: ${CYAN}https://gemini.google/subscriptions/${NC}"
echo "   • AI Pro: \$19.99/month"
echo "   • AI Ultra: \$124.99/month"
echo ""
echo "2. Setup OAuth:"
echo "   a) Visit: ${CYAN}https://ai.google.dev/gemini-api/docs/oauth${NC}"
echo "   b) Enable Gemini API in Google Cloud Console"
echo "   c) Configure OAuth consent screen"
echo "   d) Add cedricable@gmail.com as test user"
echo "   e) Create OAuth 2.0 Client ID"
echo ""

read -p "Do you have OAuth Client ID and Secret? (y/n): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    read -p "OAuth Client ID: " GOOGLE_CLIENT_ID
    read -p "OAuth Client Secret: " GOOGLE_CLIENT_SECRET

    if [ -n "$GOOGLE_CLIENT_ID" ] && [ -n "$GOOGLE_CLIENT_SECRET" ]; then
        # Update .env file
        if grep -q "GOOGLE_OAUTH_CLIENT_ID" "$ENV_FILE"; then
            sed -i.bak "s|^# GOOGLE_OAUTH_CLIENT_ID=.*|GOOGLE_OAUTH_CLIENT_ID=$GOOGLE_CLIENT_ID|" "$ENV_FILE"
            sed -i.bak "s|^GOOGLE_OAUTH_CLIENT_ID=.*|GOOGLE_OAUTH_CLIENT_ID=$GOOGLE_CLIENT_ID|" "$ENV_FILE"
        else
            echo "" >> "$ENV_FILE"
            echo "GOOGLE_OAUTH_CLIENT_ID=$GOOGLE_CLIENT_ID" >> "$ENV_FILE"
        fi

        if grep -q "GOOGLE_OAUTH_CLIENT_SECRET" "$ENV_FILE"; then
            sed -i.bak "s|^# GOOGLE_OAUTH_CLIENT_SECRET=.*|GOOGLE_OAUTH_CLIENT_SECRET=$GOOGLE_CLIENT_SECRET|" "$ENV_FILE"
            sed -i.bak "s|^GOOGLE_OAUTH_CLIENT_SECRET=.*|GOOGLE_OAUTH_CLIENT_SECRET=$GOOGLE_CLIENT_SECRET|" "$ENV_FILE"
        else
            echo "GOOGLE_OAUTH_CLIENT_SECRET=$GOOGLE_CLIENT_SECRET" >> "$ENV_FILE"
        fi

        # Update model
        if grep -q "GOOGLE_AI_MODEL" "$ENV_FILE"; then
            sed -i.bak "s|^GOOGLE_AI_MODEL=.*|GOOGLE_AI_MODEL=gemini-3-pro-preview|" "$ENV_FILE"
        else
            echo "GOOGLE_AI_MODEL=gemini-3-pro-preview" >> "$ENV_FILE"
        fi

        echo -e "${GREEN}✓${NC} Google OAuth credentials saved"
        rm -f "$ENV_FILE.bak"
    fi
else
    echo -e "${YELLOW}⚠${NC}  You can set up Google OAuth later"
    echo "   See: docs/QUICK-START.md for detailed instructions"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 3: Strategy Configuration"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo "Choose AI strategy:"
echo "  1) consensus      - Both providers agree (95%+ confidence, recommended)"
echo "  2) claude_primary - Claude leads (85-90% confidence, fast)"
echo "  3) gemini_primary - Gemini leads (80-85% confidence)"
echo "  4) parallel       - Both run independently (70-80% confidence, fastest)"
echo ""

read -p "Select strategy (1-4) [1]: " STRATEGY_CHOICE

case "$STRATEGY_CHOICE" in
    2) STRATEGY="claude_primary" ;;
    3) STRATEGY="gemini_primary" ;;
    4) STRATEGY="parallel" ;;
    *) STRATEGY="consensus" ;;
esac

# Update strategy in .env
if grep -q "AGENTIC_FLOW_STRATEGY" "$ENV_FILE"; then
    sed -i.bak "s|^AGENTIC_FLOW_STRATEGY=.*|AGENTIC_FLOW_STRATEGY=$STRATEGY|" "$ENV_FILE"
else
    echo "" >> "$ENV_FILE"
    echo "AGENTIC_FLOW_STRATEGY=$STRATEGY" >> "$ENV_FILE"
fi

rm -f "$ENV_FILE.bak"

echo -e "${GREEN}✓${NC} Strategy set to: $STRATEGY"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 4: Enforcing Subscription-Only Mode"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Ensure subscription-only mode is enabled
if ! grep -q "^SUBSCRIPTION_ONLY_MODE=true" "$ENV_FILE"; then
    echo "SUBSCRIPTION_ONLY_MODE=true" >> "$ENV_FILE"
fi
if ! grep -q "^REJECT_API_KEY_AUTH=true" "$ENV_FILE"; then
    echo "REJECT_API_KEY_AUTH=true" >> "$ENV_FILE"
fi

# CRITICAL: Remove BLOCKED API keys (Anthropic and Google only)
echo "Removing blocked pay-per-token API keys..."
sed -i.bak '/^ANTHROPIC_API_KEY=/d' "$ENV_FILE"
sed -i.bak '/^GOOGLE_AI_API_KEY=/d' "$ENV_FILE"
# Note: OPENROUTER_API_KEY and E2B_API_KEY are ALLOWED
rm -f "$ENV_FILE.bak"

echo -e "${GREEN}✓${NC} Subscription-only mode enabled"
echo -e "${GREEN}✓${NC} Blocked API keys removed (ANTHROPIC_API_KEY, GOOGLE_AI_API_KEY)"
echo -e "${BLUE}ℹ${NC}  E2B_API_KEY and OPENROUTER_API_KEY are allowed"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Setup Complete!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo -e "${GREEN}IMPORTANT:${NC} This project now uses subscription-only authentication."
echo "Pay-per-token API keys are BLOCKED and will cause errors."
echo ""

echo "Next steps:"
echo ""
echo "1. Validate configuration:"
echo -e "   ${CYAN}npm run auth:validate${NC}"
echo ""
echo "2. Start TITAN:"
echo -e "   ${CYAN}npm run start:local${NC}"
echo ""
echo "3. Test integration:"
echo -e "   ${CYAN}npm run test:integration${NC}"
echo ""
echo "4. Run UI demo:"
echo -e "   ${CYAN}npm run ui:integration${NC}"
echo ""

echo "Configuration saved to: $ENV_FILE"
echo ""
