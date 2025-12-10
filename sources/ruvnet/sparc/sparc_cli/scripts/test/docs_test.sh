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

echo -e "${BLUE}🚀 SPARC CLI Documentation Test${NC}"
echo "============"

# Documentation scenarios
SCENARIOS=(
    "API Documentation Generator"
    "Code Documentation Builder"
    "Knowledge Base Creator"
    "Tutorial Generator"
    "Architecture Documenter"
    "Exit"
)

PROMPTS=(
    "Create an API documentation generator that:
     1. Analyzes API endpoints
     2. Generates OpenAPI specs
     3. Creates interactive docs
     4. Includes usage examples
     5. Tests API endpoints"

    "Create a code documentation builder that:
     1. Analyzes source code
     2. Extracts docstrings
     3. Generates class diagrams
     4. Creates function references
     5. Builds searchable docs"

    "Create a knowledge base creator that:
     1. Organizes information
     2. Creates categories
     3. Generates FAQs
     4. Builds search index
     5. Includes version control"

    "Create a tutorial generator that:
     1. Creates step-by-step guides
     2. Includes code examples
     3. Generates screenshots
     4. Adds interactive elements
     5. Tests tutorials"

    "Create an architecture documenter that:
     1. Analyzes system structure
     2. Creates architecture diagrams
     3. Documents dependencies
     4. Maps data flows
     5. Generates tech specs"
)

DEPENDENCIES=(
    "python3-pip"  # API Docs
    "python3-pip graphviz"  # Code Docs
    "python3-pip"  # Knowledge Base
    "python3-pip"  # Tutorial
    "python3-pip graphviz plantuml"  # Architecture
)

PACKAGES=(
    "sphinx fastapi pyyaml"  # API Docs
    "sphinx autodoc graphviz"  # Code Docs
    "mkdocs mkdocs-material"  # Knowledge Base
    "sphinx sphinx-gallery"  # Tutorial
    "sphinx graphviz pyyaml"  # Architecture
)

# Generate unique app name with timestamp
app_name="sparc-docs-$(date +%s)"

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
echo -e "\n${YELLOW}Available Documentation Scenarios:${NC}"
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

# Setup environment and run documentation
echo -e "\n${GREEN}Setting up environment and starting $scenario...${NC}"
flyctl ssh console --app "$app_name" --command "/bin/bash -c 'cd /home/sparc_user && \
    apt-get update && \
    apt-get install -y $deps && \
    pip install $packages && \
    mkdir -p docs/{source,build,assets} && \
    sparc --cowboy-mode -m \"$prompt\"'"

# Get machine ID
machine_id=$(flyctl machines list --app "$app_name" --json | grep -o '"id":"[^"]*"' | head -n1 | cut -d'"' -f4)

# Cleanup
echo -e "\n${GREEN}Cleaning up...${NC}"
if [ ! -z "$machine_id" ]; then
    flyctl machine stop "$machine_id" --app "$app_name"
fi
flyctl apps destroy "$app_name" --yes

echo -e "\n${GREEN}Documentation test complete${NC}"
