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

echo -e "${BLUE}🚀 SPARC CLI Agent Test${NC}"
echo "============"

# Agent system options
SYSTEMS=(
    "Multi-Agent Development Team"
    "Self-Modifying Code System"
    "Emergent Behavior Network"
    "Autonomous System Designer"
    "Code Evolution Environment"
    "Exit"
)

PROMPTS=(
    "Create a multi-agent development system where:
     1. Each agent has a specialized role (architect, developer, tester, etc.)
     2. Agents communicate and coordinate through a shared workspace
     3. System includes code review and improvement mechanisms
     4. Agents can request help from other agents
     5. Final output is a complete, tested project"

    "Create a self-modifying code system that:
     1. Analyzes its own source code
     2. Identifies areas for improvement
     3. Generates and tests modifications
     4. Maintains a history of changes
     5. Ensures system stability through validation"

    "Create an emergent behavior network where:
     1. Multiple agents interact with shared resources
     2. Behaviors emerge from simple rule sets
     3. System adapts to changing conditions
     4. Patterns are identified and documented
     5. New behaviors can be integrated"

    "Create an autonomous system designer that:
     1. Takes high-level requirements
     2. Breaks down into components
     3. Generates architecture and code
     4. Tests and validates solutions
     5. Documents design decisions"

    "Create a code evolution environment that:
     1. Starts with basic code structures
     2. Applies evolutionary algorithms
     3. Tests fitness through execution
     4. Maintains successful variations
     5. Documents evolution process"
)

DEPENDENCIES=(
    "python3-pip git"  # Multi-Agent
    "python3-pip git"  # Self-Modifying
    "python3-pip git redis-server"  # Emergent
    "python3-pip git graphviz"  # System Designer
    "python3-pip git"  # Evolution
)

PACKAGES=(
    "networkx matplotlib pyyaml"  # Multi-Agent
    "astroid pylint black"  # Self-Modifying
    "redis-py networkx matplotlib"  # Emergent
    "graphviz pydot pyyaml"  # System Designer
    "deap networkx matplotlib"  # Evolution
)

# Generate unique app name with timestamp
app_name="sparc-agent-$(date +%s)"

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

# Show system menu
echo -e "\n${YELLOW}Available Agent Systems:${NC}"
for i in "${!SYSTEMS[@]}"; do
    echo "$((i+1))) ${SYSTEMS[$i]}"
done
echo

read -p "Select system [1-${#SYSTEMS[@]}]: " choice

if [ "$choice" -eq "${#SYSTEMS[@]}" ]; then
    echo -e "\n${GREEN}Goodbye!${NC}"
    flyctl apps destroy "$app_name" --yes
    exit 0
fi

if [ "$choice" -lt 1 ] || [ "$choice" -gt "$((${#SYSTEMS[@]}-1))" ]; then
    echo -e "\n${RED}Invalid option${NC}"
    flyctl apps destroy "$app_name" --yes
    exit 1
fi

idx=$((choice-1))
system="${SYSTEMS[$idx]}"
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

# Setup environment and run system
echo -e "\n${GREEN}Setting up environment and starting $system...${NC}"
flyctl ssh console --app "$app_name" --command "/bin/bash -c 'cd /home/sparc_user && \
    apt-get update && \
    apt-get install -y $deps && \
    pip install $packages && \
    git init && \
    git config --global user.email \"agent@example.com\" && \
    git config --global user.name \"SPARC Agent\" && \
    sparc --cowboy-mode -m \"$prompt\"'"

# Get machine ID
machine_id=$(flyctl machines list --app "$app_name" --json | grep -o '"id":"[^"]*"' | head -n1 | cut -d'"' -f4)

# Cleanup
echo -e "\n${GREEN}Cleaning up...${NC}"
if [ ! -z "$machine_id" ]; then
    flyctl machine stop "$machine_id" --app "$app_name"
fi
flyctl apps destroy "$app_name" --yes

echo -e "\n${GREEN}Agent system test complete${NC}"
