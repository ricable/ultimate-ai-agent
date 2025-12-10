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

echo -e "${BLUE}🚀 SPARC CLI Framework Test${NC}"
echo "============"

# System dependencies for each framework
SYSTEM_DEPS=(
    "unzip curl"  # Deno
    "curl python3-pip"  # UV
    "nodejs npm"  # Vite.js
    "python3-pip"  # Streamlit
    "python3-pip"  # Gradio
    "python3-pip"  # PyTorch
    "python3-pip"  # TensorFlow
    "python3-pip"  # Hugging Face
    "python3-pip"  # ML basics
)

# Framework installation commands
INSTALL_COMMANDS=(
    "curl -fsSL https://deno.land/x/install/install.sh | sh && export DENO_INSTALL=\"/root/.deno\" && export PATH=\"\$DENO_INSTALL/bin:\$PATH\""  # Deno
    "curl -sS https://webi.sh/uv | sh && . ~/.config/envman/PATH.env"  # UV
    "npm create vite@latest my-app -- --template react --yes && cd my-app && npm install"  # Vite.js
    "pip install streamlit"  # Streamlit
    "pip install gradio"  # Gradio
    "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"  # PyTorch
    "pip install tensorflow"  # TensorFlow
    "pip install transformers"  # Hugging Face
    "pip install scikit-learn pandas numpy matplotlib"  # ML basics
)

FRAMEWORK_NAMES=(
    "Deno"
    "UV Package Manager"
    "Vite.js"
    "Streamlit"
    "Gradio"
    "PyTorch"
    "TensorFlow"
    "Hugging Face Transformers"
    "ML Basics (scikit-learn, pandas, numpy)"
)

TEST_PROMPTS=(
    "Create a simple Deno HTTP server that responds with Hello World"
    "Create a Python project using UV to manage dependencies"
    "Create a basic Vite.js React app with a Hello World component"
    "Create a Streamlit app that displays Hello World"
    "Create a Gradio interface that echoes text input"
    "Create a PyTorch tensor and perform a simple operation"
    "Create a TensorFlow constant and perform a simple operation"
    "Use a Hugging Face model to perform sentiment analysis"
    "Create a simple scikit-learn classification example"
)

# Generate unique app name with timestamp
app_name="sparc-framework-$(date +%s)"

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

# Show framework menu
echo -e "\n${YELLOW}Available Frameworks:${NC}"
for i in "${!FRAMEWORK_NAMES[@]}"; do
    echo "$((i+1))) ${FRAMEWORK_NAMES[$i]}"
done
echo "$((${#FRAMEWORK_NAMES[@]}+1))) Exit"
echo

read -p "Select framework [1-$((${#FRAMEWORK_NAMES[@]}+1))]: " choice

if [ "$choice" -eq "$((${#FRAMEWORK_NAMES[@]}+1))" ]; then
    echo -e "\n${GREEN}Goodbye!${NC}"
    flyctl apps destroy "$app_name" --yes
    exit 0
fi

if [ "$choice" -lt 1 ] || [ "$choice" -gt "${#FRAMEWORK_NAMES[@]}" ]; then
    echo -e "\n${RED}Invalid option${NC}"
    flyctl apps destroy "$app_name" --yes
    exit 1
fi

idx=$((choice-1))
framework="${FRAMEWORK_NAMES[$idx]}"
system_deps="${SYSTEM_DEPS[$idx]}"
install_cmd="${INSTALL_COMMANDS[$idx]}"
test_prompt="${TEST_PROMPTS[$idx]}"

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
    --memory 512 \
    --cpus 1

# Install framework and run test
echo -e "\n${GREEN}Installing $framework and running test...${NC}"
flyctl ssh console --app "$app_name" --command "/bin/bash -c 'cd /home/sparc_user && apt-get update && apt-get install -y $system_deps && $install_cmd && sparc --cowboy-mode -m \"$test_prompt\"'"

# Get machine ID
machine_id=$(flyctl machines list --app "$app_name" --json | grep -o '"id":"[^"]*"' | head -n1 | cut -d'"' -f4)

# Cleanup
echo -e "\n${GREEN}Cleaning up...${NC}"
if [ ! -z "$machine_id" ]; then
    flyctl machine stop "$machine_id" --app "$app_name"
fi
flyctl apps destroy "$app_name" --yes

echo -e "\n${GREEN}Test complete${NC}"
