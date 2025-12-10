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

echo -e "${BLUE}🚀 SPARC CLI Security Test${NC}"
echo "============"

# Security scenarios
SCENARIOS=(
    "Code Security Analyzer"
    "Penetration Testing System"
    "Security Hardening Tool"
    "Compliance Checker"
    "Zero Trust Implementation"
    "Exit"
)

PROMPTS=(
    "Create a code security analyzer that:
     1. Scans code for common vulnerabilities
     2. Checks for OWASP Top 10 issues
     3. Analyzes dependency security
     4. Suggests security improvements
     5. Generates detailed security reports"

    "Create a penetration testing system that:
     1. Tests API endpoints for vulnerabilities
     2. Checks authentication mechanisms
     3. Tests for injection attacks
     4. Analyzes network security
     5. Provides remediation steps"

    "Create a security hardening tool that:
     1. Implements secure coding practices
     2. Adds input validation
     3. Improves error handling
     4. Implements secure configurations
     5. Documents security measures"

    "Create a compliance checker that:
     1. Verifies security standards
     2. Checks GDPR compliance
     3. Validates data handling
     4. Tests access controls
     5. Generates compliance reports"

    "Create a zero trust implementation that:
     1. Implements identity verification
     2. Sets up micro-segmentation
     3. Adds least privilege access
     4. Monitors all transactions
     5. Implements continuous validation"
)

DEPENDENCIES=(
    "python3-pip git"  # Security Analyzer
    "python3-pip git curl"  # Penetration Testing
    "python3-pip git"  # Security Hardening
    "python3-pip git"  # Compliance Checker
    "python3-pip git"  # Zero Trust
)

PACKAGES=(
    "bandit safety pylint"  # Security Analyzer
    "requests owasp-zap-api-python"  # Penetration Testing
    "cryptography pyjwt"  # Security Hardening
    "cerberus pyyaml"  # Compliance Checker
    "authlib cryptography"  # Zero Trust
)

# Generate unique app name with timestamp
app_name="sparc-sec-$(date +%s)"

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
echo -e "\n${YELLOW}Available Security Scenarios:${NC}"
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

# Setup environment and run security test
echo -e "\n${GREEN}Setting up environment and starting $scenario...${NC}"
flyctl ssh console --app "$app_name" --command "/bin/bash -c 'cd /home/sparc_user && \
    apt-get update && \
    apt-get install -y $deps && \
    pip install $packages && \
    mkdir -p security/{reports,tools,tests} && \
    sparc --cowboy-mode -m \"$prompt\"'"

# Get machine ID
machine_id=$(flyctl machines list --app "$app_name" --json | grep -o '"id":"[^"]*"' | head -n1 | cut -d'"' -f4)

# Cleanup
echo -e "\n${GREEN}Cleaning up...${NC}"
if [ ! -z "$machine_id" ]; then
    flyctl machine stop "$machine_id" --app "$app_name"
fi
flyctl apps destroy "$app_name" --yes

echo -e "\n${GREEN}Security test complete${NC}"
