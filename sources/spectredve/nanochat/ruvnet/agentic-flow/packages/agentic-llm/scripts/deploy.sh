#!/bin/bash
# Fly.io Deployment Script for Agentic LLM Training System

set -e

echo "🚀 Agentic LLM Phi-4 Deployment Script"
echo "======================================"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if flyctl is installed
if ! command -v flyctl &> /dev/null; then
    echo -e "${RED}Error: flyctl is not installed${NC}"
    echo "Install it from: https://fly.io/docs/hands-on/install-flyctl/"
    exit 1
fi

echo -e "${GREEN}✓ flyctl found${NC}"

# Check if logged in
if ! flyctl auth whoami &> /dev/null; then
    echo -e "${YELLOW}Not logged in to Fly.io${NC}"
    echo "Please run: flyctl auth login"
    exit 1
fi

echo -e "${GREEN}✓ Authenticated with Fly.io${NC}"

# Navigate to project directory
cd "$(dirname "$0")/.."

# Check if app exists
APP_NAME="agentic-llm-phi4"

if flyctl apps list | grep -q "$APP_NAME"; then
    echo -e "${YELLOW}App $APP_NAME already exists${NC}"
    read -p "Do you want to update the existing app? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 0
    fi
    DEPLOY_CMD="deploy"
else
    echo -e "${GREEN}Creating new app: $APP_NAME${NC}"
    DEPLOY_CMD="launch --now"
fi

# Deploy first (create app)
echo "🚢 Deploying to Fly.io..."

if [ "$DEPLOY_CMD" = "launch --now" ]; then
    flyctl launch --now --name "$APP_NAME" --region ord --no-deploy
else
    flyctl deploy
fi

# Create volumes after app exists
echo "📦 Setting up persistent volumes..."

if ! flyctl volumes list -a "$APP_NAME" 2>/dev/null | grep -q "phi4_models"; then
    echo "Creating phi4_models volume..."
    flyctl volumes create phi4_models --region ord --size 100 -a "$APP_NAME" --yes || true
fi

if ! flyctl volumes list -a "$APP_NAME" 2>/dev/null | grep -q "phi4_checkpoints"; then
    echo "Creating phi4_checkpoints volume..."
    flyctl volumes create phi4_checkpoints --region ord --size 50 -a "$APP_NAME" --yes || true
fi

echo -e "${GREEN}✓ Volumes configured${NC}"

# Set secrets
echo "🔐 Configuring secrets..."

read -p "Do you want to set/update secrets? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    read -p "Enter Hugging Face token (or press Enter to skip): " HF_TOKEN
    if [ ! -z "$HF_TOKEN" ]; then
        flyctl secrets set HF_TOKEN="$HF_TOKEN" -a "$APP_NAME"
    fi

    read -p "Enter WandB API key (or press Enter to skip): " WANDB_API_KEY
    if [ ! -z "$WANDB_API_KEY" ]; then
        flyctl secrets set WANDB_API_KEY="$WANDB_API_KEY" -a "$APP_NAME"
    fi
fi

echo -e "${GREEN}✓ Secrets configured${NC}"

# Now deploy with volumes
if [ "$DEPLOY_CMD" = "launch --now" ]; then
    echo "🚢 Deploying app with volumes..."
    flyctl deploy
fi

echo -e "${GREEN}✓ Deployment complete!${NC}"

# Show status
echo ""
echo "📊 Application Status:"
flyctl status

echo ""
echo "📝 Logs (last 100 lines):"
flyctl logs --lines 100

echo ""
echo -e "${GREEN}Deployment successful!${NC}"
echo ""
echo "Useful commands:"
echo "  flyctl logs           - View logs"
echo "  flyctl ssh console    - SSH into the container"
echo "  flyctl status         - Check app status"
echo "  flyctl scale count 1  - Scale to 1 instance"
echo "  flyctl scale vm a100-40gb - Change VM size"
