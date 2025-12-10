#!/bin/bash
# DevPod Rust workspace provisioning script - Creates new workspace each time

set -e

# Generate unique workspace name with timestamp (DevPod allows only lowercase, numbers, dashes)
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
WORKSPACE_NAME="polyglot-rust-devpod-$TIMESTAMP"
WORKSPACE_SOURCE="/Users/cedric/dev/github.com/polyglot-devenv/dev-env/rust"

echo "🐳 Provisioning new Rust DevPod workspace..."
echo "📦 Creating workspace: $WORKSPACE_NAME"

# Always create a new workspace
devpod up "$WORKSPACE_SOURCE" \
    --id "$WORKSPACE_NAME" \
    --ide vscode \
    --provider docker

echo "✅ Rust DevPod workspace ready!"
echo "🏷️  Workspace name: $WORKSPACE_NAME"
echo "💡 Use 'devpod list' to see all workspaces"
echo "💡 Use 'devpod stop $WORKSPACE_NAME' to stop this workspace"
echo "🧹 Use 'bash devpod-automation/scripts/provision-all.sh clean-all' to clean up old workspaces"