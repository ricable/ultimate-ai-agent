#!/bin/bash

###########################################
# GaiaNet Node Setup Script
# Installs and configures GaiaNet for monetization
###########################################

set -e

echo "======================================"
echo "GaiaNet Node Installation"
echo "======================================"

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

# Check if running on macOS
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo -e "${GREEN}✓ Detected macOS${NC}"
else
    echo -e "${YELLOW}Detected Linux${NC}"
fi

echo ""
echo "Step 1: Installing GaiaNet..."

# Install GaiaNet using official installer
if ! command -v gaianet &> /dev/null; then
    echo "Downloading and installing GaiaNet..."
    curl -sSfL 'https://github.com/GaiaNet-AI/gaianet-node/releases/latest/download/install.sh' | bash

    # Source the environment
    if [ -f "$HOME/.bashrc" ]; then
        source "$HOME/.bashrc"
    fi
    if [ -f "$HOME/.zshrc" ]; then
        source "$HOME/.zshrc"
    fi
else
    echo -e "${GREEN}✓ GaiaNet already installed${NC}"
fi

echo ""
echo "Step 2: Node Configuration"
echo ""
echo "Select configuration:"
echo "  1) Qwen 2.5 Coder 7B (16GB RAM minimum)"
echo "  2) Qwen 2.5 Coder 14B (24GB RAM minimum)"
echo "  3) Qwen 2.5 Coder 32B (48GB RAM minimum)"
echo "  4) Custom configuration"
echo ""
read -p "Enter choice [1-4]: " config_choice

case $config_choice in
    1)
        CONFIG_URL="https://raw.githubusercontent.com/GaiaNet-AI/node-configs/main/qwen2.5-coder-7b-instruct/config.json"
        MODEL_SIZE="7B"
        ;;
    2)
        CONFIG_URL="https://raw.githubusercontent.com/GaiaNet-AI/node-configs/main/qwen2.5-coder-14b-instruct/config.json"
        MODEL_SIZE="14B"
        ;;
    3)
        CONFIG_URL="https://raw.githubusercontent.com/GaiaNet-AI/node-configs/main/qwen2.5-coder-32b-instruct/config.json"
        MODEL_SIZE="32B"
        ;;
    4)
        read -p "Enter custom config URL: " CONFIG_URL
        MODEL_SIZE="Custom"
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac

echo ""
echo "Step 3: Initializing node with Qwen 2.5 Coder $MODEL_SIZE..."

gaianet init --config "$CONFIG_URL"

echo -e "${GREEN}✓ Node initialized${NC}"
echo ""

# Configure prompt template for Qwen
echo "Step 4: Configuring prompt template..."
gaianet config --prompt-template chatml

echo -e "${GREEN}✓ Prompt template set to ChatML${NC}"
echo ""

# Optional: Adjust context size
echo "Step 5: Performance tuning..."
read -p "Set context size (default 32768, press Enter to use default): " ctx_size
ctx_size=${ctx_size:-32768}

gaianet config --ctx-size "$ctx_size"

echo -e "${GREEN}✓ Context size set to $ctx_size${NC}"
echo ""

# Display node information
echo "======================================"
echo -e "${GREEN}GaiaNet Node Setup Complete!${NC}"
echo "======================================"
echo ""
echo "Node configuration:"
gaianet config --show
echo ""
echo "Next steps:"
echo ""
echo "1. Start your node:"
echo "   ${BLUE}gaianet start${NC}"
echo ""
echo "2. Check node status:"
echo "   ${BLUE}gaianet info${NC}"
echo ""
echo "3. View logs:"
echo "   ${BLUE}gaianet log${NC}"
echo ""
echo "4. Stop node:"
echo "   ${BLUE}gaianet stop${NC}"
echo ""
echo "Your local API endpoint will be:"
echo "   ${GREEN}http://localhost:8080/v1${NC}"
echo ""
echo "To register with GaiaNet network for monetization:"
echo "   Visit: https://www.gaianet.ai/register"
echo ""
echo "Monitor earnings:"
echo "   Dashboard: https://www.gaianet.ai/dashboard"
echo ""
