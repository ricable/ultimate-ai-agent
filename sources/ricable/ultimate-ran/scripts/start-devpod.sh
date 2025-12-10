#!/bin/bash
# TITAN DevPod Startup Script
# Sets up and starts DevPod environment with Docker provider

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}🔷 TITAN DevPod Setup${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

# Check if devpod is installed
if ! command -v devpod &> /dev/null; then
    echo -e "${YELLOW}⚠ DevPod not found. Installing...${NC}"

    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS installation
        if command -v brew &> /dev/null; then
            brew install devpod
        else
            echo -e "${RED}❌ Homebrew not found. Install from: https://devpod.sh${NC}"
            exit 1
        fi
    else
        echo -e "${RED}❌ Please install DevPod from: https://devpod.sh${NC}"
        exit 1
    fi
fi

echo -e "${GREEN}✓ DevPod installed: $(devpod version)${NC}"

# Check if Docker is running
if ! docker info &> /dev/null; then
    echo -e "${RED}❌ Docker is not running. Please start Docker Desktop.${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Docker is running${NC}"

# Load environment variables
if [ ! -f "config/.env" ]; then
    echo -e "${YELLOW}⚠ Creating .env from template...${NC}"
    cp config/.env.template config/.env
    echo -e "${RED}❌ Please edit config/.env with your API keys and restart${NC}"
    exit 1
fi

# Check if Docker provider is configured
if ! devpod provider list | grep -q "docker"; then
    echo -e "${BLUE}📦 Adding Docker provider...${NC}"
    devpod provider add docker
fi

# Set Docker as default provider
echo -e "${BLUE}🔧 Setting Docker as default provider...${NC}"
devpod provider use docker

# Configure Docker provider options
echo -e "${BLUE}⚙️  Configuring Docker provider for Mac Silicon...${NC}"
devpod provider set-options docker \
    --option PLATFORM=linux/arm64 \
    --option CPUS=4 \
    --option MEMORY=8G

# Create or update workspace
WORKSPACE_NAME="titan-ran"

if devpod list | grep -q "$WORKSPACE_NAME"; then
    echo -e "${YELLOW}⚠ Workspace '$WORKSPACE_NAME' exists. Updating...${NC}"
    devpod delete "$WORKSPACE_NAME" --force || true
fi

echo -e "${BLUE}🏗️  Creating DevPod workspace...${NC}"
devpod up . \
    --id "$WORKSPACE_NAME" \
    --provider docker \
    --devcontainer-path config/devpod.yaml \
    --dotfiles-url "$(pwd)/config/.env" \
    --ide vscode

echo -e "\n${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}✓ DevPod workspace created!${NC}"
echo -e "${BLUE}  Name: $WORKSPACE_NAME${NC}"
echo -e "${BLUE}  Provider: Docker (ARM64)${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

# Start the workspace
echo -e "\n${BLUE}🚀 Starting workspace...${NC}"
devpod up "$WORKSPACE_NAME" --ide vscode

echo -e "\n${GREEN}✓ TITAN DevPod environment is ready!${NC}"
echo -e "\n${BLUE}Useful commands:${NC}"
echo -e "  ${YELLOW}devpod ssh $WORKSPACE_NAME${NC}         - SSH into workspace"
echo -e "  ${YELLOW}devpod stop $WORKSPACE_NAME${NC}        - Stop workspace"
echo -e "  ${YELLOW}devpod delete $WORKSPACE_NAME${NC}      - Delete workspace"
echo -e "  ${YELLOW}devpod list${NC}                        - List all workspaces"
