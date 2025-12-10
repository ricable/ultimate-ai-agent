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

echo -e "${BLUE}🚀 SPARC CLI Data Science Test${NC}"
echo "============"

# Data science scenarios
SCENARIOS=(
    "Data Explorer"
    "Time Series Analyzer"
    "Pattern Recognition System"
    "Data Visualization Studio"
    "Predictive Analytics Tool"
    "Exit"
)

PROMPTS=(
    "Create a data exploration system that:
     1. Loads and cleans various data formats
     2. Performs statistical analysis
     3. Identifies data patterns
     4. Generates summary reports
     5. Creates interactive visualizations"

    "Create a time series analyzer that:
     1. Processes temporal data
     2. Identifies trends and seasonality
     3. Makes forecasts
     4. Handles missing data
     5. Visualizes predictions"

    "Create a pattern recognition system that:
     1. Implements clustering algorithms
     2. Finds data correlations
     3. Detects anomalies
     4. Generates pattern reports
     5. Visualizes relationships"

    "Create a data visualization studio that:
     1. Creates multiple chart types
     2. Supports interactive plots
     3. Handles large datasets
     4. Customizes visualizations
     5. Exports to various formats"

    "Create a predictive analytics tool that:
     1. Builds prediction models
     2. Evaluates accuracy
     3. Handles feature selection
     4. Cross-validates results
     5. Visualizes predictions"
)

DEPENDENCIES=(
    "python3-pip"  # Data Explorer
    "python3-pip"  # Time Series
    "python3-pip"  # Pattern Recognition
    "python3-pip python3-tk"  # Visualization
    "python3-pip"  # Predictive Analytics
)

PACKAGES=(
    "pandas numpy scipy matplotlib seaborn"  # Data Explorer
    "pandas numpy statsmodels prophet"  # Time Series
    "pandas numpy scikit-learn matplotlib"  # Pattern Recognition
    "pandas plotly dash bokeh"  # Visualization
    "pandas scikit-learn xgboost lightgbm"  # Predictive Analytics
)

# Generate unique app name with timestamp
app_name="sparc-data-$(date +%s)"

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
echo -e "\n${YELLOW}Available Data Science Scenarios:${NC}"
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

# Setup environment and run data science
echo -e "\n${GREEN}Setting up environment and starting $scenario...${NC}"
flyctl ssh console --app "$app_name" --command "/bin/bash -c 'cd /home/sparc_user && \
    apt-get update && \
    apt-get install -y $deps && \
    pip install $packages && \
    mkdir -p data/{raw,processed,visualizations,models} && \
    sparc --cowboy-mode -m \"$prompt\"'"

# Get machine ID
machine_id=$(flyctl machines list --app "$app_name" --json | grep -o '"id":"[^"]*"' | head -n1 | cut -d'"' -f4)

# Cleanup
echo -e "\n${GREEN}Cleaning up...${NC}"
if [ ! -z "$machine_id" ]; then
    flyctl machine stop "$machine_id" --app "$app_name"
fi
flyctl apps destroy "$app_name" --yes

echo -e "\n${GREEN}Data science test complete${NC}"
