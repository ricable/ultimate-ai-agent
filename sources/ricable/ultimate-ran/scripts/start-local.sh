#!/bin/bash
# TITAN Local Startup Script for Mac Silicon
# Supports both direct execution and DevPod modes

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 TITAN RAN Platform - Local Startup${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

# Detect architecture
ARCH=$(uname -m)
if [[ "$ARCH" == "arm64" ]]; then
    echo -e "${GREEN}✓ Detected Mac Silicon (ARM64)${NC}"
    export PLATFORM_ARCH=arm64
else
    echo -e "${YELLOW}⚠ Detected x86_64 architecture${NC}"
    export PLATFORM_ARCH=amd64
fi

# Check for .env file
if [ ! -f "config/.env" ]; then
    echo -e "${YELLOW}⚠ No .env file found. Creating from template...${NC}"
    cp config/.env.template config/.env
    echo -e "${RED}❌ Please edit config/.env with your API keys and restart${NC}"
    exit 1
fi

# Load environment variables
set -a
source config/.env
set +a

echo -e "\n${BLUE}📋 Configuration Check:${NC}"

# Check API keys
check_api_key() {
    local key_name=$1
    local key_value=$2

    if [ -z "$key_value" ] || [[ "$key_value" == *"your-key-here"* ]]; then
        echo -e "${RED}  ✗ $key_name not configured${NC}"
        return 1
    else
        echo -e "${GREEN}  ✓ $key_name configured${NC}"
        return 0
    fi
}

KEYS_OK=true
check_api_key "ANTHROPIC_API_KEY" "$ANTHROPIC_API_KEY" || KEYS_OK=false
check_api_key "GOOGLE_AI_API_KEY" "$GOOGLE_AI_API_KEY" || KEYS_OK=false
check_api_key "E2B_API_KEY" "$E2B_API_KEY" || KEYS_OK=false
check_api_key "OPENROUTER_API_KEY" "$OPENROUTER_API_KEY" || KEYS_OK=false

if [ "$KEYS_OK" = false ]; then
    echo -e "\n${YELLOW}⚠ Some API keys are missing. Continue anyway? (y/N)${NC}"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check Node.js version
NODE_VERSION=$(node -v | cut -d'v' -f2 | cut -d'.' -f1)
if [ "$NODE_VERSION" -lt 18 ]; then
    echo -e "${RED}❌ Node.js 18+ required (found: $(node -v))${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Node.js $(node -v)${NC}"

# Set runtime mode
export RUNTIME_MODE=${RUNTIME_MODE:-local}
echo -e "${GREEN}✓ Runtime mode: $RUNTIME_MODE${NC}"

# Install dependencies if needed
if [ ! -d "node_modules" ]; then
    echo -e "\n${BLUE}📦 Installing dependencies...${NC}"
    npm install
fi

# Build TypeScript
echo -e "\n${BLUE}🔨 Building TypeScript...${NC}"
npm run build

# Start services based on mode
echo -e "\n${BLUE}🎯 Starting TITAN services...${NC}"

if [ "$RUNTIME_MODE" == "devpod" ]; then
    echo -e "${BLUE}  Starting DevPod mode with Docker Compose...${NC}"
    docker-compose -f config/docker-compose.devpod.yml up -d

    echo -e "\n${GREEN}✓ DevPod services started!${NC}"
    echo -e "${BLUE}  View logs: docker-compose -f config/docker-compose.devpod.yml logs -f${NC}"

else
    echo -e "${BLUE}  Starting local mode...${NC}"

    # Start AgentDB if available
    if command -v npx &> /dev/null; then
        echo -e "${BLUE}  Starting AgentDB...${NC}"
        npx agentdb@alpha start &
        AGENTDB_PID=$!
    fi

    # Start AG-UI server
    echo -e "${BLUE}  Starting AG-UI server...${NC}"
    npm run agui:start &
    AGUI_PID=$!

    # Wait for services to be ready
    sleep 3

    # Start main orchestrator
    echo -e "${BLUE}  Starting TITAN orchestrator...${NC}"
    npm start &
    TITAN_PID=$!

    # Trap cleanup
    cleanup() {
        echo -e "\n${YELLOW}🛑 Shutting down TITAN...${NC}"
        [ ! -z "$AGENTDB_PID" ] && kill $AGENTDB_PID 2>/dev/null || true
        [ ! -z "$AGUI_PID" ] && kill $AGUI_PID 2>/dev/null || true
        [ ! -z "$TITAN_PID" ] && kill $TITAN_PID 2>/dev/null || true
        echo -e "${GREEN}✓ Shutdown complete${NC}"
        exit 0
    }

    trap cleanup SIGINT SIGTERM

    echo -e "\n${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}✓ TITAN is running!${NC}"
    echo -e "${BLUE}  UI Dashboard:  http://localhost:3000${NC}"
    echo -e "${BLUE}  AG-UI Server:  http://localhost:3001${NC}"
    echo -e "${BLUE}  QUIC Port:     4433${NC}"
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}Press Ctrl+C to stop${NC}\n"

    # Wait for processes
    wait
fi
