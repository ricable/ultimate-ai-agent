#!/bin/bash
# Edge-Native AI SaaS - Mac Silicon (ARM64 Darwin) Setup Script
# Installs all @ruvector packages optimized for Apple Silicon M1/M2/M3/M4

set -e

echo "=========================================="
echo "Edge-Native AI SaaS - Mac Silicon Setup"
echo "=========================================="
echo ""

# Check if running on Mac Silicon
if [[ "$(uname -s)" != "Darwin" ]] || [[ "$(uname -m)" != "arm64" ]]; then
    echo "Warning: This script is optimized for Mac Silicon (ARM64 Darwin)"
    echo "Current system: $(uname -s) $(uname -m)"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check Node.js version
echo "Checking Node.js version..."
NODE_VERSION=$(node -v 2>/dev/null || echo "none")
if [[ "$NODE_VERSION" == "none" ]]; then
    echo "Node.js not found. Please install Node.js 18+ first."
    echo "  brew install node"
    exit 1
fi
echo "Node.js version: $NODE_VERSION"

# Check npm version
NPM_VERSION=$(npm -v 2>/dev/null)
echo "npm version: $NPM_VERSION"

echo ""
echo "Installing @ruvector packages for Mac Silicon ARM64..."
echo ""

# Core RuVector packages
echo "[1/8] Installing ruvector core..."
npm install ruvector@latest

echo "[2/8] Installing ruvector-core-darwin-arm64 (Native ARM64 bindings)..."
npm install ruvector-core-darwin-arm64@latest

echo "[3/8] Installing ruvector-extensions..."
npm install ruvector-extensions@latest

echo "[4/8] Installing @ruvector/gnn (Graph Neural Networks)..."
npm install @ruvector/gnn@latest

echo "[5/8] Installing @ruvector/gnn-darwin-arm64 (Native ARM64 GNN)..."
npm install @ruvector/gnn-darwin-arm64@latest

echo "[6/8] Installing @ruvector/graph-node (Hypergraph Database)..."
npm install @ruvector/graph-node@latest

echo "[7/8] Installing @ruvector/graph-node-darwin-arm64 (Native ARM64 Graph)..."
npm install @ruvector/graph-node-darwin-arm64@latest

echo "[8/8] Installing @ruvector/agentic-synth-examples..."
npm install @ruvector/agentic-synth-examples@latest

echo ""
echo "Installing Agent Orchestration packages..."
echo ""

# Claude Flow and Agentic Flow
echo "[1/3] Installing claude-flow..."
npm install claude-flow@latest

echo "[2/3] Installing agentic-flow..."
npm install agentic-flow@latest

echo "[3/3] Installing agentdb..."
npm install agentdb@latest

echo ""
echo "Running tool initialization..."
echo ""

# Initialize tools
echo "Initializing claude-flow..."
npx claude-flow init --force 2>/dev/null || echo "claude-flow init skipped"

echo "Initializing agentdb..."
npx agentdb init --dimension 384 --backend ruvector 2>/dev/null || echo "agentdb init skipped"

echo "Checking ruvector..."
npx ruvector info

echo ""
echo "=========================================="
echo "Installation Complete!"
echo "=========================================="
echo ""
echo "Installed packages:"
echo "  - ruvector (Vector Database with HNSW)"
echo "  - ruvector-core-darwin-arm64 (Native ARM64)"
echo "  - @ruvector/gnn (Graph Neural Networks)"
echo "  - @ruvector/graph-node (Hypergraph Database)"
echo "  - claude-flow (Agent Orchestration)"
echo "  - agentic-flow (Multi-Agent Swarms)"
echo "  - agentdb (AI Agent Memory)"
echo ""
echo "Quick Start:"
echo "  npx ruvector --help          # Vector database CLI"
echo "  npx claude-flow --help       # Agent orchestration"
echo "  npx agentic-flow --list      # List available agents"
echo "  npx agentdb status           # Check database status"
echo ""
echo "Start the full stack:"
echo "  docker compose up -d"
echo ""
