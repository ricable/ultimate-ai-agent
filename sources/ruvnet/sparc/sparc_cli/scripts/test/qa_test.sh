#!/bin/bash

set -e  # Exit on error

# Colors
BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Setup flyctl environment
export FLYCTL_INSTALL="/home/$USER/.fly"
export PATH="$FLYCTL_INSTALL/bin:$PATH"

# Verify flyctl is available
if ! command -v flyctl &> /dev/null; then
    echo "Error: flyctl not found. Please run ./install.sh first"
    exit 1
fi

# Check if config exists
if [ ! -f "$(dirname "$0")/../fly/config.sh" ]; then
    echo "Error: config.sh not found. Please run install.sh first."
    exit 1
fi

# Source configuration
source "$(dirname "$0")/../fly/config.sh"

echo -e "${BLUE}🚀 SPARC CLI QA Test${NC}"
echo "============"

# QA scenarios
SCENARIOS=(
    "Test Generator"
    "Performance Tester"
    "Security Tester"
    "UI/UX Tester"
    "Integration Tester"
    "Exit"
)

PROMPTS=(
    "Create a test generator that:
     1. Analyzes code for test cases
     2. Generates unit tests
     3. Creates integration tests
     4. Adds property-based tests
     5. Measures test coverage"

    "Create a performance tester that:
     1. Measures response times
     2. Tests under load
     3. Identifies bottlenecks
     4. Generates performance reports
     5. Suggests optimizations"

    "Create a security tester that:
     1. Scans for vulnerabilities
     2. Tests authentication
     3. Checks for common exploits
     4. Validates input handling
     5. Generates security reports"

    "Create a UI/UX tester that:
     1. Tests user flows
     2. Validates accessibility
     3. Checks responsiveness
     4. Tests cross-browser
     5. Generates UI reports"

    "Create an integration tester that:
     1. Tests API integrations
     2. Validates data flow
     3. Tests error handling
     4. Checks system boundaries
     5. Generates test reports"
)

DEPENDENCIES=(
    "python3-pip"  # Test Generator
    "python3-pip apache2-utils"  # Performance
    "python3-pip"  # Security
    "python3-pip chromium-driver"  # UI/UX
    "python3-pip"  # Integration
)

PACKAGES=(
    "pytest hypothesis coverage"  # Test Generator
    "locust pytest-benchmark"  # Performance
    "bandit safety owasp-zap-api-python"  # Security
    "selenium pytest-selenium axe-selenium-python"  # UI/UX
    "pytest requests pytest-asyncio"  # Integration
)

# Generate unique app name with timestamp
app_name="sparc-qa-$(date +%s)"

# Create new app
echo -e "\n${GREEN}Creating app: $app_name${NC}"
if ! flyctl apps create "$app_name" --machines; then
    echo -e "${RED}Failed to create app${NC}"
    exit 1
fi

# Set up secrets
echo -e "\n${GREEN}Setting up secrets...${NC}"
flyctl secrets set \
    ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY" \
    OPENAI_API_KEY="$OPENAI_API_KEY" \
    --app "$app_name"

# Get the image from the main app
echo -e "\n${GREEN}Getting machine info...${NC}"
machines_output=$(flyctl machines list --app "sparc-cli-test")

# Extract the image name using awk to handle variable whitespace
image=$(echo "$machines_output" | awk '/started/ {for(i=1;i<=NF;i++) if($i ~ /sparc-cli-test:deployment/) print $i}')

if [ -z "$image" ]; then
    echo -e "${RED}Error: Could not get image from sparc-cli-test${NC}"
    exit 1
fi

# Show scenario menu
echo -e "\n${YELLOW}Available QA Scenarios:${NC}"
for i in "${!SCENARIOS[@]}"; do
    echo "$((i+1))) ${SCENARIOS[$i]}"
done
echo

read -p "Select scenario [1-${#SCENARIOS[@]}]: " choice

if [ "$choice" -eq "${#SCENARIOS[@]}" ]; then
    echo -e "\n${GREEN}Goodbye!${NC}"
    flyctl apps destroy "$app_name" --yes
    exit 0
fi

if [ "$choice" -lt 1 ] || [ "$choice" -gt "$((${#SCENARIOS[@]}-1))" ]; then
    echo -e "\n${RED}Invalid option${NC}"
    flyctl apps destroy "$app_name" --yes
    exit 1
fi

idx=$((choice-1))
scenario="${SCENARIOS[$idx]}"
prompt="${PROMPTS[$idx]}"
deps="${DEPENDENCIES[$idx]}"
packages="${PACKAGES[$idx]}"

# Deploy using the existing image
echo -e "\n${GREEN}Starting machine...${NC}"
flyctl machine run "registry.fly.io/$image" \
    --app "$app_name" \
    --env PYTHONPATH=/opt/sparc \
    --env PYTHONUNBUFFERED=1 \
    --env TERM=xterm \
    --env ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY" \
    --env OPENAI_API_KEY="$OPENAI_API_KEY" \
    --metadata fly_process_group=app \
    --region lax \
    --memory 2048 \
    --cpus 2

# Setup environment and run QA
echo -e "\n${GREEN}Setting up environment and starting $scenario...${NC}"
flyctl ssh console --app "$app_name" --command "/bin/bash -c 'cd /home/sparc_user && \
    apt-get update && \
    apt-get install -y $deps && \
    pip install $packages && \
    mkdir -p tests/{unit,integration,performance,reports} && \
    sparc --cowboy-mode -m \"$prompt\"'"

# Get machine ID
machine_id=$(flyctl machines list --app "$app_name" --json | grep -o '"id":"[^"]*"' | head -n1 | cut -d'"' -f4)

# Cleanup
echo -e "\n${GREEN}Cleaning up...${NC}"
if [ ! -z "$machine_id" ]; then
    flyctl machine stop "$machine_id" --app "$app_name"
fi
flyctl apps destroy "$app_name" --yes

echo -e "\n${GREEN}QA test complete${NC}"
