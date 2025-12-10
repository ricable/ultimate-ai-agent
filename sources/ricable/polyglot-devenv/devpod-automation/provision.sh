#!/bin/bash
# DevPod workspace provisioning script - Creates new workspace each time

set -e

# Check if a language parameter is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <language>"
    echo "Example: $0 go"
    exit 1
fi

LANGUAGE=$1
WORKSPACE_SOURCE="/Users/cedric/dev/github.com/polyglot-devenv/${LANGUAGE}-env"

# Generate unique workspace name with timestamp
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
WORKSPACE_NAME="polyglot-${LANGUAGE}-devpod-$TIMESTAMP"

echo "🐳 Provisioning new ${LANGUAGE} DevPod workspace..."
echo "📦 Creating workspace: $WORKSPACE_NAME"

# Always create a new workspace
devpod up "$WORKSPACE_SOURCE" \
    --id "$WORKSPACE_NAME" \
    --ide vscode \
    --provider docker

echo "✅ ${LANGUAGE} DevPod workspace ready!"
echo "🏷️  Workspace name: $WORKSPACE_NAME"
echo "💡 Use 'devpod list' to see all workspaces"
echo "💡 Use 'devpod stop $WORKSPACE_NAME' to stop this workspace"
echo "🧹 Use 'bash devpod-automation/scripts/provision-all.sh clean-all' to clean up old workspaces"