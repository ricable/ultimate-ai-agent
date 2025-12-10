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

echo -e "${BLUE}🚀 SPARC CLI LionAGI Test${NC}"
echo "============"


# LionAGI scenarios
SCENARIOS=(
    "Chain Builder"
    "Agent System"
    "Memory Manager"
    "Tool Integration"
    "Workflow Automation"
    "Exit"
)

PROMPTS=(
    "Create a LionAGI chain that:
     1. Processes multiple inputs
     2. Uses sequential reasoning
     3. Implements error handling
     4. Manages state
     5. Generates detailed output"

    "Create a LionAGI agent system that:
     1. Handles multiple tasks
     2. Uses tool integration
     3. Manages memory
     4. Implements planning
     5. Adapts to feedback"

    "Create a memory management system that:
     1. Stores conversation history
     2. Implements retrieval
     3. Handles context windows
     4. Manages memory cleanup
     5. Optimizes storage"

    "Create a tool integration system that:
     1. Loads external tools
     2. Manages tool execution
     3. Handles tool output
     4. Implements error recovery
     5. Coordinates tool chains"

    "Create a workflow automation that:
     1. Defines workflow steps
     2. Manages dependencies
     3. Handles parallel execution
     4. Implements monitoring
     5. Generates reports"
)

DEPENDENCIES=(
    "python3-pip curl"  # Chain Builder
    "python3-pip git"   # Agent System
    "python3-pip"       # Memory Manager
    "python3-pip"       # Tool Integration
    "python3-pip"       # Workflow
)

PACKAGES=(
    "lionagi"  # Chain Builder
    "lionagi requests"  # Agent System
    "lionagi pandas"  # Memory Manager
    "lionagi python-dotenv"  # Tool Integration
    "lionagi pyyaml"  # Workflow
)

# Generate unique app name with timestamp
app_name="sparc-lionagi-$(date +%s)"

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
    echo -e "${RED}Error: No running machines found in sparc-cli-test. Please deploy first:${NC}"
    echo "cd sparc_cli/scripts/fly && ./deploy.sh"
    flyctl apps destroy "$app_name" --yes
    exit 1
fi

# Show scenario menu
echo -e "\n${YELLOW}Available LionAGI Scenarios:${NC}"
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

# Setup environment and run LionAGI test
echo -e "\n${GREEN}Setting up environment and starting $scenario...${NC}"
flyctl ssh console --app "$app_name" --command "/bin/bash -c 'cd /home/sparc_user && \
    apt-get update && \
    apt-get install -y $deps && \
    pip install $packages && \
    mkdir -p lionagi/{chains,agents,memory,tools} && \
    sparc --cowboy-mode -m \"$prompt\"'"

# Get machine ID
machine_id=$(flyctl machines list --app "$app_name" --json | grep -o '"id":"[^"]*"' | head -n1 | cut -d'"' -f4)

# Cleanup
echo -e "\n${GREEN}Cleaning up...${NC}"
if [ ! -z "$machine_id" ]; then
    flyctl machine stop "$machine_id" --app "$app_name"
fi
flyctl apps destroy "$app_name" --yes

echo -e "\n${GREEN}LionAGI test complete${NC}"
