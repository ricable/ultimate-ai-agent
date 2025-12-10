#!/bin/bash

set -e  # Exit on error

# Colors
BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Setup flyctl environment
export FLYCTL_INSTALL="/home/$USER/.fly"
export PATH="$FLYCTL_INSTALL/bin:$PATH"

# Verify flyctl is available
if ! command -v flyctl &> /dev/null; then
    echo "Error: flyctl not found. Please run ./install.sh first"
    exit 1
fi

echo -e "${BLUE}🚀 SPARC CLI${NC}"
echo "============"

# Check if config exists
if [ ! -f "$(dirname "$0")/config.sh" ]; then
    echo "Error: config.sh not found. Please run install.sh first."
    exit 1
fi

# Source configuration
source "$(dirname "$0")/config.sh"

# Check if app exists
if ! flyctl apps list | grep -q "^$FLY_APP_NAME"; then
    echo "Error: Application $FLY_APP_NAME not found"
    echo "Please run install.sh and deploy.sh first"
    exit 1
fi

# Function to run SPARC CLI command
run_sparc() {
    local mode=$1
    echo -e "\n${GREEN}Connecting...${NC} Use Ctrl+C to exit\n"
    flyctl ssh console --app "$FLY_APP_NAME"
}

# Provide options for access
echo -e "\n${YELLOW}Modes:${NC}"
echo "1) Chat      sparc --chat"
echo "2) Research  sparc --research-only"
echo "3) Cowboy    sparc --cowboy-mode"
echo "4) Console   sparc --help"
echo "5) Logs"
echo "6) Exit"
echo

read -p "Select [1-6]: " choice

case $choice in
    1)
        echo -e "\n${GREEN}Chat Mode${NC}"
        echo "Run: sparc --chat"
        run_sparc "chat"
        ;;
    2)
        echo -e "\n${GREEN}Research Mode${NC}"
        echo "Run: sparc --research-only"
        run_sparc "research"
        ;;
    3)
        echo -e "\n${GREEN}Cowboy Mode${NC}"
        echo "Run: sparc --cowboy-mode"
        run_sparc "cowboy"
        ;;
    4)
        echo -e "\n${GREEN}Console Mode${NC}"
        echo "Commands:"
        echo "  sparc --chat"
        echo "  sparc --research-only"
        echo "  sparc --cowboy-mode"
        echo "  sparc --help"
        run_sparc "console"
        ;;
    5)
        echo -e "\n${GREEN}Logs${NC}"
        flyctl logs --app "$FLY_APP_NAME"
        ;;
    6)
        echo -e "\n${GREEN}Goodbye!${NC}"
        exit 0
        ;;
    *)
        echo -e "\n${RED}Invalid option${NC}"
        exit 1
        ;;
esac
