#!/bin/bash
set -e

VCS_TYPE=${1:-git}  # git or jj
REPO_PATH=${2:-/repos/test-repo}
AGENT_COUNT=${AGENT_COUNT:-10}

echo "=== Setting up ${VCS_TYPE} test repository ==="
echo "Repository path: ${REPO_PATH}"
echo "Agent count: ${AGENT_COUNT}"

# Create repository directory
mkdir -p "${REPO_PATH}"
cd "${REPO_PATH}"

if [ "$VCS_TYPE" = "git" ]; then
    echo "Initializing Git repository..."
    git init

    # Create initial commit
    echo "# Test Repository" > README.md
    git add README.md
    git commit -m "Initial commit"

    # Create test files
    mkdir -p src tests docs
    for i in $(seq 1 10); do
        echo "// Test file $i" > "src/file_$i.rs"
        echo "// Test file $i" > "tests/test_$i.rs"
    done

    git add .
    git commit -m "Add initial files"

    echo "Git repository initialized successfully"

elif [ "$VCS_TYPE" = "jj" ]; then
    echo "Initializing Jujutsu repository..."

    # Check if co-located mode requested
    if [ "${JJ_COLOCATED:-false}" = "true" ]; then
        echo "Using co-located mode (Git + Jujutsu)"
        git init
        jj git init --colocate
    else
        echo "Using pure Jujutsu mode"
        jj git init
    fi

    # Create initial commit
    echo "# Test Repository" > README.md
    jj describe -m "Initial commit"

    # Create test files
    mkdir -p src tests docs
    for i in $(seq 1 10); do
        echo "// Test file $i" > "src/file_$i.rs"
        echo "// Test file $i" > "tests/test_$i.rs"
    done

    jj describe -m "Add initial files"

    # Set up workspaces for agents if shared-log mode
    if [ "${JJ_WORKSPACE_MODE}" = "shared-log" ]; then
        echo "Setting up ${AGENT_COUNT} agent workspaces (shared operation log)..."

        for i in $(seq 1 $AGENT_COUNT); do
            workspace_path="/repos/workspaces/agent-${i}"
            mkdir -p "${workspace_path}"

            # Create workspace
            jj workspace add "${workspace_path}" 2>/dev/null || true

            echo "Created workspace: agent-${i}"
        done
    fi

    echo "Jujutsu repository initialized successfully"
fi

# Verify setup
echo ""
echo "=== Repository Setup Complete ==="
if [ "$VCS_TYPE" = "git" ]; then
    git log --oneline | head -5
else
    jj log --limit 5 --no-graph
fi
