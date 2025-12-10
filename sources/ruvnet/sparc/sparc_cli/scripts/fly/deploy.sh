#!/bin/bash

set -e  # Exit on error

# Setup flyctl environment
export FLYCTL_INSTALL="/home/$USER/.fly"
export PATH="$FLYCTL_INSTALL/bin:$PATH"

# Verify flyctl is available
if ! command -v flyctl &> /dev/null; then
    echo "Error: flyctl not found. Please run ./install.sh first"
    exit 1
fi

echo "Deploying SPARC CLI to Fly.io..."
echo "================================"

# Function to show usage
show_usage() {
    echo "Usage: $0 [--remote]"
    echo
    echo "Options:"
    echo "  --remote    Deploy from a remote repository"
    echo "  --help      Show this help message"
    echo
    echo "By default, deploys from the current repository."
}

# Parse command line arguments
USE_REMOTE=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --remote)
            USE_REMOTE=true
            shift
            ;;
        --help)
            show_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Check if config exists
if [ ! -f "$(dirname "$0")/config.sh" ]; then
    echo "Error: config.sh not found. Please run install.sh first."
    exit 1
fi

# Source configuration
source "$(dirname "$0")/config.sh"

# Create temporary directory for deployment
TEMP_DIR=$(mktemp -d)
echo "Created temporary directory: $TEMP_DIR"

cleanup() {
    echo "Cleaning up temporary directory..."
    rm -rf "$TEMP_DIR"
}
trap cleanup EXIT

if [ "$USE_REMOTE" = true ]; then
    # Get repository URL
    read -p "Enter GitHub repository URL (e.g., https://github.com/username/repo.git): " GIT_REPO_URL
    if [ -z "$GIT_REPO_URL" ]; then
        echo "Error: Repository URL is required"
        exit 1
    fi

    # Get branch
    read -p "Enter branch name [main]: " GIT_BRANCH
    GIT_BRANCH=${GIT_BRANCH:-main}

    echo "Cloning repository from $GIT_REPO_URL branch $GIT_BRANCH..."
    if ! git clone --branch "$GIT_BRANCH" "$GIT_REPO_URL" "$TEMP_DIR/repo"; then
        echo "Error: Failed to clone repository"
        exit 1
    fi
else
    echo "Using current repository..."
    # Get the repository root (parent of sparc_cli/scripts/fly)
    REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
    echo "Repository root: $REPO_ROOT"
    cp -r "$REPO_ROOT/." "$TEMP_DIR/repo/"
fi

cd "$TEMP_DIR/repo"

# Verify repository structure
if [ ! -d "sparc_cli" ]; then
    echo "Error: Invalid repository structure"
    echo "Expected 'sparc_cli' directory containing:"
    echo "  - Python package files"
    echo
    echo "Current directory contents:"
    ls -la
    exit 1
fi

# Set up secrets first
echo "Setting up application secrets..."
flyctl secrets set \
    ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY" \
    --app "$FLY_APP_NAME"

# Create fly.toml
echo "Creating fly.toml configuration..."
cat > fly.toml << EOL
app = "$FLY_APP_NAME"
primary_region = "$FLY_REGION"

[build]
  dockerfile = "sparc_cli/scripts/fly/Dockerfile"

[env]
  PYTHONUNBUFFERED = "1"
  TERM = "xterm-256color"

[[vm]]
  cpu_kind = "shared"
  cpus = 1
  memory_mb = 1024
EOL

# Deploy to fly.io
echo "Deploying application..."
if ! flyctl deploy \
    --app "$FLY_APP_NAME" \
    --vm-memory 1024 \
    --vm-size "$DEPLOY_MACHINE_SIZE" \
    --regions "$FLY_REGION" \
    --yes; then
    echo "Error: Deployment failed"
    exit 1
fi

echo "Deployment complete!"
echo "Run ./access.sh to connect to your deployment"
