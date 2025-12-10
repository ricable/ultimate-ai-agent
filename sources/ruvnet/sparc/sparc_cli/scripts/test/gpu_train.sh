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

echo -e "${BLUE}🚀 SPARC CLI GPU Training${NC}"
echo "============"

# Training options
MODELS=(
    "Self-Improving CNN"
    "Dynamic Transformer"
    "Neural Architecture Search"
    "Meta-Learning Model"
    "Exit"
)

PROMPTS=(
    "Create a self-improving CNN model that:
     1. Uses CUDA for GPU training
     2. Implements dynamic architecture modification
     3. Has automated hyperparameter tuning
     4. Includes performance monitoring and visualization
     5. Saves checkpoints and training history"

    "Create a dynamic transformer model that:
     1. Uses GPU-accelerated attention mechanisms
     2. Has adaptive layer creation/pruning
     3. Implements gradient accumulation
     4. Monitors and optimizes memory usage
     5. Generates architecture evolution reports"

    "Create a neural architecture search system that:
     1. Uses GPU for parallel architecture evaluation
     2. Implements evolutionary optimization
     3. Has performance-based selection
     4. Includes automated benchmarking
     5. Generates architecture visualizations"

    "Create a meta-learning model that:
     1. Uses GPU for multi-task training
     2. Implements few-shot learning
     3. Has dynamic task adaptation
     4. Includes performance tracking
     5. Generates learning curves and analysis"
)

# Generate unique app name with timestamp
app_name="sparc-gpu-$(date +%s)"

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

# Show model menu
echo -e "\n${YELLOW}Available Models:${NC}"
for i in "${!MODELS[@]}"; do
    echo "$((i+1))) ${MODELS[$i]}"
done
echo

read -p "Select model [1-${#MODELS[@]}]: " choice

if [ "$choice" -eq "${#MODELS[@]}" ]; then
    echo -e "\n${GREEN}Goodbye!${NC}"
    flyctl apps destroy "$app_name" --yes
    exit 0
fi

if [ "$choice" -lt 1 ] || [ "$choice" -gt "$((${#MODELS[@]}-1))" ]; then
    echo -e "\n${RED}Invalid option${NC}"
    flyctl apps destroy "$app_name" --yes
    exit 1
fi

idx=$((choice-1))
model="${MODELS[$idx]}"
prompt="${PROMPTS[$idx]}"

# Deploy using the existing image with GPU
echo -e "\n${GREEN}Starting machine with GPU...${NC}"
flyctl machine run "registry.fly.io/$image" \
    --app "$app_name" \
    --env PYTHONPATH=/opt/sparc \
    --env PYTHONUNBUFFERED=1 \
    --env TERM=xterm \
    --env ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY" \
    --env OPENAI_API_KEY="$OPENAI_API_KEY" \
    --metadata fly_process_group=app \
    --region lax \
    --vm-size performance-8x \
    --vm-memory 16384 \
    --vm-gpu-kind a100-pcie-40gb \
    --vm-gpus 1

# Install dependencies and run training
echo -e "\n${GREEN}Setting up GPU environment and starting training...${NC}"
flyctl ssh console --app "$app_name" --command "/bin/bash -c 'cd /home/sparc_user && \
    apt-get update && \
    apt-get install -y python3-pip && \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 && \
    pip install numpy matplotlib pandas tensorboard scikit-learn tqdm && \
    sparc --cowboy-mode -m \"$prompt\"'"

# Get machine ID
machine_id=$(flyctl machines list --app "$app_name" --json | grep -o '"id":"[^"]*"' | head -n1 | cut -d'"' -f4)

# Cleanup
echo -e "\n${GREEN}Cleaning up...${NC}"
if [ ! -z "$machine_id" ]; then
    flyctl machine stop "$machine_id" --app "$app_name"
fi
flyctl apps destroy "$app_name" --yes

echo -e "\n${GREEN}Training complete${NC}"
