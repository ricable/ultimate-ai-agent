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

echo -e "${BLUE}🚀 SPARC CLI Orchestrator Test${NC}"
echo "============"

# Orchestration scenarios
SCENARIOS=(
    "Microservices Ecosystem"
    "Data Pipeline System"
    "Event-Driven Architecture"
    "API Gateway Network"
    "Service Mesh Platform"
    "Exit"
)

PROMPTS=(
    "Create a microservices ecosystem that:
     1. Implements multiple interconnected services (auth, user, content, etc.)
     2. Uses FastAPI for service endpoints
     3. Implements service discovery and health checks
     4. Includes load balancing and failover
     5. Provides monitoring and logging"

    "Create a data pipeline system that:
     1. Ingests data from multiple sources
     2. Implements ETL processes
     3. Uses Apache Airflow for orchestration
     4. Includes data validation and cleaning
     5. Generates processing reports"

    "Create an event-driven architecture that:
     1. Uses Redis for pub/sub messaging
     2. Implements event sourcing
     3. Handles async processing
     4. Includes retry mechanisms
     5. Provides event monitoring"

    "Create an API gateway network that:
     1. Routes requests to microservices
     2. Implements rate limiting
     3. Handles authentication/authorization
     4. Provides API documentation
     5. Includes usage analytics"

    "Create a service mesh platform that:
     1. Implements service-to-service communication
     2. Uses Envoy for proxying
     3. Includes circuit breaking
     4. Provides distributed tracing
     5. Monitors service health"
)

DEPENDENCIES=(
    "python3-pip redis-server"  # Microservices
    "python3-pip postgresql"  # Data Pipeline
    "python3-pip redis-server"  # Event-Driven
    "python3-pip nginx"  # API Gateway
    "python3-pip docker.io"  # Service Mesh
)

PACKAGES=(
    "fastapi uvicorn redis prometheus-client"  # Microservices
    "apache-airflow pandas numpy"  # Data Pipeline
    "redis aioredis asyncio"  # Event-Driven
    "fastapi uvicorn pyjwt"  # API Gateway
    "istio opentelemetry-api"  # Service Mesh
)

# Generate unique app name with timestamp
app_name="sparc-orch-$(date +%s)"

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
echo -e "\n${YELLOW}Available Orchestration Scenarios:${NC}"
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

# Setup environment and run orchestration
echo -e "\n${GREEN}Setting up environment and starting $scenario...${NC}"
flyctl ssh console --app "$app_name" --command "/bin/bash -c 'cd /home/sparc_user && \
    apt-get update && \
    apt-get install -y $deps && \
    pip install $packages && \
    mkdir -p services config logs && \
    sparc --cowboy-mode -m \"$prompt\"'"

# Get machine ID
machine_id=$(flyctl machines list --app "$app_name" --json | grep -o '"id":"[^"]*"' | head -n1 | cut -d'"' -f4)

# Cleanup
echo -e "\n${GREEN}Cleaning up...${NC}"
if [ ! -z "$machine_id" ]; then
    flyctl machine stop "$machine_id" --app "$app_name"
fi
flyctl apps destroy "$app_name" --yes

echo -e "\n${GREEN}Orchestration test complete${NC}"
