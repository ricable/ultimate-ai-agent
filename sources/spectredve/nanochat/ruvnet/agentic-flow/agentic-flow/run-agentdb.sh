#!/bin/bash
# Quick launcher for AgentDB CLI from anywhere in the repo

# Get the script's directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Run the CLI from the correct location
node "$SCRIPT_DIR/dist/agentdb/cli/agentdb-cli.js" "$@"
