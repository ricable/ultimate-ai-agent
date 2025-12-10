#!/bin/bash

###########################################
# Distributed Cluster Setup Script
# Configures node for distributed operation
###########################################

set -e

echo "=========================================="
echo "Distributed Cluster Setup"
echo "=========================================="

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

# Step 1: Check prerequisites
echo -e "\n${BLUE}Step 1: Checking Prerequisites${NC}"

if ! command -v node &> /dev/null; then
    echo -e "${RED}Node.js not found. Please install Node.js 18+${NC}"
    exit 1
fi

if ! command -v npm &> /dev/null; then
    echo -e "${RED}npm not found. Please install npm 9+${NC}"
    exit 1
fi

if ! command -v docker &> /dev/null; then
    echo -e "${YELLOW}Warning: Docker not found. Install for sandbox support${NC}"
fi

echo -e "${GREEN}✓ Prerequisites OK${NC}"

# Step 2: Install dependencies
echo -e "\n${BLUE}Step 2: Installing Dependencies${NC}"

npm install

echo -e "${GREEN}✓ Dependencies installed${NC}"

# Step 3: Check for Redis
echo -e "\n${BLUE}Step 3: Checking Redis${NC}"

if ! command -v redis-cli &> /dev/null; then
    echo -e "${YELLOW}Redis not found locally${NC}"
    echo "Would you like to run Redis in Docker? (y/n)"
    read -r response

    if [[ "$response" =~ ^[Yy]$ ]]; then
        echo "Starting Redis container..."
        docker run -d \
            --name redis-cluster \
            -p 6379:6379 \
            redis:alpine

        echo -e "${GREEN}✓ Redis started in Docker${NC}"
    else
        echo -e "${YELLOW}You'll need to provide a Redis endpoint in .env${NC}"
    fi
else
    if redis-cli ping &> /dev/null; then
        echo -e "${GREEN}✓ Redis is running${NC}"
    else
        echo -e "${YELLOW}Redis installed but not running${NC}"
        echo "Start Redis with: redis-server"
    fi
fi

# Step 4: Detect hardware
echo -e "\n${BLUE}Step 4: Detecting Hardware${NC}"

PLATFORM=$(uname -s | tr '[:upper:]' '[:lower:]')
ARCH=$(uname -m)
RAM_GB=$(free -g 2>/dev/null | awk '/^Mem:/{print $2}' || sysctl -n hw.memsize 2>/dev/null | awk '{print int($1/1024/1024/1024)}')

echo "Platform: $PLATFORM"
echo "Architecture: $ARCH"
echo "RAM: ${RAM_GB}GB"

# Determine hardware type
HARDWARE_TYPE=""

if [[ "$PLATFORM" == "darwin" ]]; then
    if [[ "$RAM_GB" -ge 100 ]]; then
        HARDWARE_TYPE="macbook-m3-max"
    elif [[ "$RAM_GB" -ge 50 ]]; then
        HARDWARE_TYPE="mac-studio-m1"
    else
        HARDWARE_TYPE="mac-studio-m1"
    fi
elif [[ "$PLATFORM" == "linux" ]]; then
    if [[ "$ARCH" == "aarch64" || "$ARCH" == "armv7l" ]]; then
        if [[ "$RAM_GB" -lt 8 ]]; then
            HARDWARE_TYPE="raspberry-pi"
        else
            HARDWARE_TYPE="intel-nuc"
        fi
    else
        HARDWARE_TYPE="intel-nuc"
    fi
fi

echo -e "${GREEN}Detected: $HARDWARE_TYPE${NC}"

# Step 5: Configure node
echo -e "\n${BLUE}Step 5: Configuring Node${NC}"

# Copy .env.example if .env doesn't exist
if [ ! -f .env ]; then
    cp .env.example .env
    echo -e "${GREEN}✓ Created .env file${NC}"
fi

# Load hardware-specific config
CONFIG_FILE="config/hardware/${HARDWARE_TYPE}.json"

if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}Hardware config not found: $CONFIG_FILE${NC}"
    exit 1
fi

# Extract recommended settings from config
MODEL=$(node -p "JSON.parse(require('fs').readFileSync('$CONFIG_FILE')).inference.models.recommended[0].name")
CONTEXT=$(node -p "JSON.parse(require('fs').readFileSync('$CONFIG_FILE')).inference.models.maxContextSize")
MEMORY_LIMIT=$(node -p "JSON.parse(require('fs').readFileSync('$CONFIG_FILE')).orchestration.limits.maxMemoryPerTask")

echo "Recommended model: $MODEL"
echo "Max context: $CONTEXT"

# Update .env with hardware-specific settings
# (Optional: could automate this with sed, but manual review is safer)

echo -e "${YELLOW}Please review .env and adjust settings for your hardware${NC}"

# Step 6: Node name
echo -e "\n${BLUE}Step 6: Node Configuration${NC}"

echo "Enter a name for this node (default: auto-generated):"
read -r NODE_NAME

if [ -z "$NODE_NAME" ]; then
    NODE_NAME="node-$(hostname)"
fi

echo "NODE_NAME=$NODE_NAME" >> .env
echo -e "${GREEN}✓ Node name: $NODE_NAME${NC}"

# Step 7: Redis connection
echo -e "\n${BLUE}Step 7: Redis Connection${NC}"

echo "Redis host (default: localhost):"
read -r REDIS_HOST
REDIS_HOST=${REDIS_HOST:-localhost}

echo "Redis port (default: 6379):"
read -r REDIS_PORT
REDIS_PORT=${REDIS_PORT:-6379}

echo "REDIS_HOST=$REDIS_HOST" >> .env
echo "REDIS_PORT=$REDIS_PORT" >> .env

echo -e "${GREEN}✓ Redis configured${NC}"

# Step 8: Test connection
echo -e "\n${BLUE}Step 8: Testing Cluster Connection${NC}"

echo "Attempting to connect to Redis at $REDIS_HOST:$REDIS_PORT..."

if command -v redis-cli &> /dev/null; then
    if redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" ping &> /dev/null; then
        echo -e "${GREEN}✓ Successfully connected to Redis${NC}"
    else
        echo -e "${YELLOW}⚠ Could not connect to Redis${NC}"
        echo "Make sure Redis is running and accessible"
    fi
else
    echo -e "${YELLOW}redis-cli not available, skipping connection test${NC}"
fi

# Step 9: Summary
echo -e "\n=========================================="
echo -e "${GREEN}Setup Complete!${NC}"
echo -e "==========================================\n"

echo "Configuration:"
echo "  Node Type: $HARDWARE_TYPE"
echo "  Node Name: $NODE_NAME"
echo "  Model: $MODEL"
echo "  Redis: $REDIS_HOST:$REDIS_PORT"

echo -e "\nNext steps:"
echo -e "  1. Review and adjust ${BLUE}.env${NC} file"
echo -e "  2. Set up inference:"
echo "     - GaiaNet: ${BLUE}./scripts/setup-gaianet.sh${NC}"
echo "     - LlamaEdge: ${BLUE}./scripts/setup-llamaedge.sh${NC}"
echo -e "  3. Join cluster: ${BLUE}npm run cluster:init${NC}"
echo -e "  4. Check status: ${BLUE}npm run cluster:status${NC}"
echo -e "  5. Start orchestrator: ${BLUE}npm run start:quad${NC}"

echo ""
