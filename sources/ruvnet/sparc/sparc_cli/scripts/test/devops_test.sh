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

echo -e "${BLUE}🚀 SPARC CLI DevOps Test${NC}"
echo "============"

# DevOps scenarios
SCENARIOS=(
    "CI/CD Pipeline Builder"
    "Infrastructure Automator"
    "Monitoring System"
    "Deployment Manager"
    "Container Orchestrator"
    "Exit"
)

PROMPTS=(
    "Create a CI/CD pipeline builder that:
     1. Sets up build automation
     2. Configures testing stages
     3. Implements deployment steps
     4. Adds quality gates
     5. Generates pipeline reports"

    "Create an infrastructure automator that:
     1. Generates IaC templates
     2. Manages cloud resources
     3. Handles configuration
     4. Implements scaling
     5. Monitors resources"

    "Create a monitoring system that:
     1. Collects metrics
     2. Sets up alerts
     3. Creates dashboards
     4. Handles logging
     5. Generates reports"

    "Create a deployment manager that:
     1. Handles versioning
     2. Manages environments
     3. Implements rollbacks
     4. Controls releases
     5. Tracks deployments"

    "Create a container orchestrator that:
     1. Manages containers
     2. Handles scaling
     3. Configures networking
     4. Manages storage
     5. Monitors health"
)

DEPENDENCIES=(
    "python3-pip git"  # CI/CD
    "python3-pip ansible"  # Infrastructure
    "python3-pip"  # Monitoring
    "python3-pip"  # Deployment
    "python3-pip docker.io"  # Container
)

PACKAGES=(
    "jenkins-job-builder pytest-ci"  # CI/CD
    "ansible boto3 terraform-provider-aws"  # Infrastructure
    "prometheus-client grafana-api"  # Monitoring
    "fabric python-semantic-release"  # Deployment
    "docker-compose kubernetes"  # Container
)

# Generate unique app name with timestamp
app_name="sparc-devops-$(date +%s)"

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
echo -e "\n${YELLOW}Available DevOps Scenarios:${NC}"
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

# Setup environment and run DevOps
echo -e "\n${GREEN}Setting up environment and starting $scenario...${NC}"
flyctl ssh console --app "$app_name" --command "/bin/bash -c 'cd /home/sparc_user && \
    apt-get update && \
    apt-get install -y $deps && \
    pip install $packages && \
    mkdir -p devops/{pipelines,config,scripts,logs} && \
    sparc --cowboy-mode -m \"$prompt\"'"

# Get machine ID
machine_id=$(flyctl machines list --app "$app_name" --json | grep -o '"id":"[^"]*"' | head -n1 | cut -d'"' -f4)

# Cleanup
echo -e "\n${GREEN}Cleaning up...${NC}"
if [ ! -z "$machine_id" ]; then
    flyctl machine stop "$machine_id" --app "$app_name"
fi
flyctl apps destroy "$app_name" --yes

echo -e "\n${GREEN}DevOps test complete${NC}"
