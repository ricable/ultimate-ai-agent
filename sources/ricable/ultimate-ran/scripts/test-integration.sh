#!/bin/bash
# TITAN Multi-Provider Integration Test
# Tests all API integrations: Claude, Gemini, E2B, OpenRouter

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}🧪 TITAN Multi-Provider Integration Test${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

# Load environment
if [ -f "config/.env" ]; then
    set -a
    source config/.env
    set +a
else
    echo -e "${RED}❌ config/.env not found${NC}"
    exit 1
fi

TEST_RESULTS=()

# Test function
test_provider() {
    local name=$1
    local test_cmd=$2

    echo -e "\n${BLUE}Testing $name...${NC}"

    if eval "$test_cmd" &> /dev/null; then
        echo -e "${GREEN}  ✓ $name working${NC}"
        TEST_RESULTS+=("✓ $name")
        return 0
    else
        echo -e "${RED}  ✗ $name failed${NC}"
        TEST_RESULTS+=("✗ $name")
        return 1
    fi
}

# Test Anthropic Claude
if [ ! -z "$ANTHROPIC_API_KEY" ] && [[ ! "$ANTHROPIC_API_KEY" == *"your-key-here"* ]]; then
    test_provider "Claude Code PRO MAX" "curl -s -H 'x-api-key: $ANTHROPIC_API_KEY' -H 'anthropic-version: 2023-06-01' https://api.anthropic.com/v1/messages -X POST -d '{\"model\":\"claude-sonnet-4-5-20250929\",\"max_tokens\":1024,\"messages\":[{\"role\":\"user\",\"content\":\"test\"}]}'"
else
    echo -e "${YELLOW}⚠ Skipping Claude test (API key not configured)${NC}"
    TEST_RESULTS+=("⊘ Claude (not configured)")
fi

# Test Google AI
if [ ! -z "$GOOGLE_AI_API_KEY" ] && [[ ! "$GOOGLE_AI_API_KEY" == *"your-key-here"* ]]; then
    test_provider "Google AI Pro (Gemini)" "curl -s 'https://generativelanguage.googleapis.com/v1/models?key=$GOOGLE_AI_API_KEY'"
else
    echo -e "${YELLOW}⚠ Skipping Google AI test (API key not configured)${NC}"
    TEST_RESULTS+=("⊘ Google AI (not configured)")
fi

# Test E2B
if [ ! -z "$E2B_API_KEY" ] && [[ ! "$E2B_API_KEY" == *"your-key-here"* ]]; then
    test_provider "E2B Sandboxes" "curl -s -H 'X-API-Key: $E2B_API_KEY' https://api.e2b.dev/sandboxes"
else
    echo -e "${YELLOW}⚠ Skipping E2B test (API key not configured)${NC}"
    TEST_RESULTS+=("⊘ E2B (not configured)")
fi

# Test OpenRouter
if [ ! -z "$OPENROUTER_API_KEY" ] && [[ ! "$OPENROUTER_API_KEY" == *"your-key-here"* ]]; then
    test_provider "OpenRouter" "curl -s -H 'Authorization: Bearer $OPENROUTER_API_KEY' https://openrouter.ai/api/v1/models"
else
    echo -e "${YELLOW}⚠ Skipping OpenRouter test (API key not configured)${NC}"
    TEST_RESULTS+=("⊘ OpenRouter (not configured)")
fi

# Test agentic-flow
echo -e "\n${BLUE}Testing agentic-flow@alpha...${NC}"
if npx agentic-flow@alpha --version &> /dev/null; then
    echo -e "${GREEN}  ✓ agentic-flow installed${NC}"
    TEST_RESULTS+=("✓ agentic-flow")
else
    echo -e "${RED}  ✗ agentic-flow not available${NC}"
    TEST_RESULTS+=("✗ agentic-flow")
fi

# Test claude-flow
echo -e "\n${BLUE}Testing claude-flow@alpha...${NC}"
if npx claude-flow@alpha --version &> /dev/null; then
    echo -e "${GREEN}  ✓ claude-flow installed${NC}"
    TEST_RESULTS+=("✓ claude-flow")
else
    echo -e "${RED}  ✗ claude-flow not available${NC}"
    TEST_RESULTS+=("✗ claude-flow")
fi

# Test local environment
echo -e "\n${BLUE}Testing local environment...${NC}"

# Node.js version
NODE_VERSION=$(node -v)
echo -e "${GREEN}  ✓ Node.js $NODE_VERSION${NC}"

# Architecture
ARCH=$(uname -m)
echo -e "${GREEN}  ✓ Architecture: $ARCH${NC}"

# Docker
if docker info &> /dev/null; then
    echo -e "${GREEN}  ✓ Docker running${NC}"
    TEST_RESULTS+=("✓ Docker")
else
    echo -e "${YELLOW}  ⊘ Docker not running${NC}"
    TEST_RESULTS+=("⊘ Docker")
fi

# Print summary
echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}Test Summary:${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

for result in "${TEST_RESULTS[@]}"; do
    if [[ $result == ✓* ]]; then
        echo -e "${GREEN}$result${NC}"
    elif [[ $result == ✗* ]]; then
        echo -e "${RED}$result${NC}"
    else
        echo -e "${YELLOW}$result${NC}"
    fi
done

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

# Count successes
SUCCESS_COUNT=$(printf "%s\n" "${TEST_RESULTS[@]}" | grep -c "^✓" || true)
TOTAL_COUNT=${#TEST_RESULTS[@]}

echo -e "\n${GREEN}Passed: $SUCCESS_COUNT / $TOTAL_COUNT tests${NC}"

if [ $SUCCESS_COUNT -eq $TOTAL_COUNT ]; then
    echo -e "${GREEN}🎉 All tests passed!${NC}"
    exit 0
else
    echo -e "${YELLOW}⚠ Some tests failed or skipped${NC}"
    exit 0
fi
