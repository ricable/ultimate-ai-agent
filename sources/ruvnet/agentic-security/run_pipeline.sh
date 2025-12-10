#!/bin/bash

# Exit on error, undefined variables, and pipe failures
set -euo pipefail

# Colors for output
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

# Function for error handling
handle_error() {
    echo -e "${RED}[✗] Error: $1${NC}"
    exit 1
}

# Function for status messages
status_msg() {
    echo -e "${CYAN}[>] $1${NC}"
}

# Function for success messages
success_msg() {
    echo -e "${GREEN}[✓] $1${NC}"
}

# Verify required tools
command -v docker >/dev/null 2>&1 || handle_error "Docker is required but not installed"
command -v python3 >/dev/null 2>&1 || handle_error "Python 3 is required but not installed"

# Environment setup
status_msg "Setting up environment..."
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}$(pwd)/src"
if [ -f ".env" ]; then
    source .env
else
    handle_error ".env file not found"
fi

# Verify required environment variables
[[ -z "${OPENAI_API_KEY:-}" ]] && handle_error "OPENAI_API_KEY is not set"
[[ -z "${ANTHROPIC_API_KEY:-}" ]] && handle_error "ANTHROPIC_API_KEY is not set"
[[ -z "${SLACK_WEBHOOK:-}" ]] && handle_error "SLACK_WEBHOOK is not set"

# Create cache directory if it doesn't exist
mkdir -p .security_cache

# Pull required Docker images
status_msg "Pulling required Docker images..."
docker pull owasp/zap2docker-stable >/dev/null || handle_error "Failed to pull OWASP ZAP image"

# Run security pipeline
status_msg "Running security pipeline..."
if python3 -m agentic_security.security_cli run --config config.yml; then
    success_msg "Pipeline execution completed successfully!"
else
    handle_error "Pipeline execution failed"
fi
