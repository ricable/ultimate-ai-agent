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

echo -e "${BLUE}🚀 SPARC CLI Web Scraper Test${NC}"
echo "============"

# Scraping scenarios
SCENARIOS=(
    "News Aggregator"
    "E-commerce Monitor"
    "Social Media Analyzer"
    "Research Assistant"
    "Content Curator"
    "Exit"
)

PROMPTS=(
    "Create a news aggregator that:
     1. Scrapes multiple news sources
     2. Extracts article content
     3. Categorizes news topics
     4. Identifies trending stories
     5. Generates daily summaries"

    "Create an e-commerce monitor that:
     1. Tracks product prices
     2. Monitors stock levels
     3. Compares across sites
     4. Detects price changes
     5. Generates alerts"

    "Create a social media analyzer that:
     1. Gathers social posts
     2. Analyzes sentiment
     3. Identifies trends
     4. Tracks engagement
     5. Creates reports"

    "Create a research assistant that:
     1. Searches academic papers
     2. Extracts key findings
     3. Organizes references
     4. Summarizes content
     5. Generates citations"

    "Create a content curator that:
     1. Gathers relevant content
     2. Filters by quality
     3. Organizes by topic
     4. Creates summaries
     5. Schedules sharing"
)

DEPENDENCIES=(
    "python3-pip"  # News Aggregator
    "python3-pip chromium-driver"  # E-commerce
    "python3-pip"  # Social Media
    "python3-pip"  # Research
    "python3-pip"  # Content Curator
)

PACKAGES=(
    "beautifulsoup4 requests newspaper3k"  # News Aggregator
    "selenium beautifulsoup4 requests"  # E-commerce
    "tweepy beautifulsoup4 textblob"  # Social Media
    "scholarly beautifulsoup4 bibtexparser"  # Research
    "beautifulsoup4 requests feedparser"  # Content Curator
)

# Generate unique app name with timestamp
app_name="sparc-scrape-$(date +%s)"

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
echo -e "\n${YELLOW}Available Scraping Scenarios:${NC}"
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

# Setup environment and run scraper
echo -e "\n${GREEN}Setting up environment and starting $scenario...${NC}"
flyctl ssh console --app "$app_name" --command "/bin/bash -c 'cd /home/sparc_user && \
    apt-get update && \
    apt-get install -y $deps && \
    pip install $packages && \
    mkdir -p scraper/{data,cache,output} && \
    sparc --cowboy-mode -m \"$prompt\"'"

# Get machine ID
machine_id=$(flyctl machines list --app "$app_name" --json | grep -o '"id":"[^"]*"' | head -n1 | cut -d'"' -f4)

# Cleanup
echo -e "\n${GREEN}Cleaning up...${NC}"
if [ ! -z "$machine_id" ]; then
    flyctl machine stop "$machine_id" --app "$app_name"
fi
flyctl apps destroy "$app_name" --yes

echo -e "\n${GREEN}Web scraper test complete${NC}"
