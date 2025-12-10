#!/bin/bash

###############################################################################
# Medical Analysis System - Quick Start Script
# Starts the CLI and API components with proper configuration
###############################################################################

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SRC_DIR="$PROJECT_ROOT/src"

echo "🏥 Medical AI Analysis System - Quick Start"
echo "=========================================="
echo ""

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js 18+ first."
    exit 1
fi

echo "✓ Node.js version: $(node --version)"
echo ""

# Check if dependencies are installed
if [ ! -d "$SRC_DIR/node_modules" ]; then
    echo "📦 Installing dependencies..."
    cd "$SRC_DIR"
    npm install
    echo "✓ Dependencies installed"
else
    echo "✓ Dependencies already installed"
fi

# Create data directory for AgentDB
DATA_DIR="$PROJECT_ROOT/data"
if [ ! -d "$DATA_DIR" ]; then
    echo "📁 Creating data directory..."
    mkdir -p "$DATA_DIR"
    echo "✓ Data directory created: $DATA_DIR"
fi

# Build TypeScript
echo ""
echo "🔨 Building TypeScript..."
cd "$SRC_DIR"
npm run build
echo "✓ Build complete"

# Display menu
echo ""
echo "Choose an option:"
echo "1. Start API Server"
echo "2. Run CLI (Interactive)"
echo "3. Run CLI (Example Analysis)"
echo "4. Run Tests"
echo "5. Exit"
echo ""
read -p "Enter choice [1-5]: " choice

case $choice in
    1)
        echo ""
        echo "🚀 Starting Medical Analysis API Server..."
        echo "=========================================="
        echo ""
        echo "API will be available at: http://localhost:3000"
        echo "WebSocket endpoint: ws://localhost:3000"
        echo ""
        echo "Press Ctrl+C to stop the server"
        echo ""
        cd "$SRC_DIR"
        npm start
        ;;
    2)
        echo ""
        echo "🖥️  Starting CLI in Interactive Mode..."
        echo "=========================================="
        echo ""
        cd "$SRC_DIR"
        npm run cli -- analyze --interactive
        ;;
    3)
        echo ""
        echo "🖥️  Running Example Analysis..."
        echo "=========================================="
        echo ""
        cd "$SRC_DIR"
        npm run cli -- analyze "Type 2 Diabetes" \
            -s "increased thirst" "frequent urination" "fatigue" \
            --age 45 \
            --gender male \
            --high-confidence
        ;;
    4)
        echo ""
        echo "🧪 Running Tests..."
        echo "=========================================="
        echo ""
        cd "$SRC_DIR"
        npm test
        ;;
    5)
        echo "Goodbye!"
        exit 0
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac
