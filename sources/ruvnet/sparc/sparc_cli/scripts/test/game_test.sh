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

echo -e "${BLUE}🚀 SPARC CLI Game Test${NC}"
echo "============"

# Game development scenarios
SCENARIOS=(
    "Text Adventure Engine"
    "2D Platformer Creator"
    "Strategy Game Builder"
    "Interactive Story System"
    "Puzzle Game Generator"
    "Exit"
)

PROMPTS=(
    "Create a text adventure engine that:
     1. Implements a rich command parser
     2. Has dynamic room generation
     3. Includes inventory management
     4. Features NPC interactions
     5. Saves game progress"

    "Create a 2D platformer creator that:
     1. Uses Pygame for graphics
     2. Implements physics and collision
     3. Has level generation
     4. Includes sprite animation
     5. Features sound effects"

    "Create a strategy game builder that:
     1. Implements turn-based mechanics
     2. Has resource management
     3. Features AI opponents
     4. Includes tech trees
     5. Has save/load functionality"

    "Create an interactive story system that:
     1. Has branching narratives
     2. Includes character development
     3. Features dynamic choices
     4. Tracks player decisions
     5. Generates story variations"

    "Create a puzzle game generator that:
     1. Creates random puzzles
     2. Has difficulty scaling
     3. Implements scoring system
     4. Features multiple game modes
     5. Includes level editor"
)

DEPENDENCIES=(
    "python3-pip"  # Text Adventure
    "python3-pip python3-pygame"  # 2D Platformer
    "python3-pip"  # Strategy Game
    "python3-pip"  # Interactive Story
    "python3-pip python3-tk"  # Puzzle Game
)

PACKAGES=(
    "prompt_toolkit colorama"  # Text Adventure
    "pygame numpy"  # 2D Platformer
    "numpy pandas"  # Strategy Game
    "pyyaml jinja2"  # Interactive Story
    "pygame numpy pillow"  # Puzzle Game
)

# Generate unique app name with timestamp
app_name="sparc-game-$(date +%s)"

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
echo -e "\n${YELLOW}Available Game Development Scenarios:${NC}"
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

# Setup environment and run game development
echo -e "\n${GREEN}Setting up environment and starting $scenario...${NC}"
flyctl ssh console --app "$app_name" --command "/bin/bash -c 'cd /home/sparc_user && \
    apt-get update && \
    apt-get install -y $deps && \
    pip install $packages && \
    mkdir -p game/{assets,src,levels} && \
    sparc --cowboy-mode -m \"$prompt\"'"

# Get machine ID
machine_id=$(flyctl machines list --app "$app_name" --json | grep -o '"id":"[^"]*"' | head -n1 | cut -d'"' -f4)

# Cleanup
echo -e "\n${GREEN}Cleaning up...${NC}"
if [ ! -z "$machine_id" ]; then
    flyctl machine stop "$machine_id" --app "$app_name"
fi
flyctl apps destroy "$app_name" --yes

echo -e "\n${GREEN}Game development test complete${NC}"
