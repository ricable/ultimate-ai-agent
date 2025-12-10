#!/bin/bash

# Fly.io Configuration
export FLY_APP_NAME="your-app-name"  # The name of your fly.io application
export FLY_REGION="lax"              # Your preferred fly.io region
export FLY_ORG="personal"            # Your fly.io organization

# Git Repository Configuration
export GIT_REPO_URL="https://github.com/yourusername/sparc.git"
export GIT_BRANCH="main"

# API Keys (same as local .sparc_exports)
export ANTHROPIC_API_KEY=""          # Required
export OPENAI_API_KEY=""             # Optional
export OPENROUTER_KEY=""             # Optional
export ENCRYPTION_KEY=""             # Optional
export GEMINI_API_KEY=""             # Optional
export VERTEXAI_PROJECT=""           # Optional
export VERTEXAI_LOCATION=""          # Optional

# Deployment Configuration
export DEPLOY_MACHINE_SIZE="shared-cpu-1x"  # Fly.io machine size
export DEPLOY_MEMORY="256"                  # Memory in MB
