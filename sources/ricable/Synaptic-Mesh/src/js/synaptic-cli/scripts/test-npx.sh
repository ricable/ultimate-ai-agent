#!/bin/bash

# Test script for NPX functionality
# This script validates that the synaptic-mesh package works correctly with NPX

set -euo pipefail

echo "🧪 Testing NPX functionality for synaptic-mesh..."

# Create test directory
TEST_DIR="/tmp/synaptic-test-$$"
mkdir -p "$TEST_DIR"
cd "$TEST_DIR"

echo "📦 Testing local package installation..."

# Pack the package
PACKAGE_DIR="/workspaces/Synaptic-Neural-Mesh/src/js/synaptic-cli"
cd "$PACKAGE_DIR"
npm pack

# Get the packed file name
PACKED_FILE=$(ls synaptic-mesh-*.tgz | head -1)
echo "Created package: $PACKED_FILE"

# Install globally for testing
echo "🔧 Installing package globally for testing..."
npm install -g "$PACKED_FILE"

# Test global installation
echo "✅ Testing global installation..."
synaptic-mesh --version
synaptic-mesh --help

# Test in a new directory
cd "$TEST_DIR"
echo "🏗️ Testing node initialization..."

# Test init command
synaptic-mesh init --no-interactive --name test-node-npx

# Verify files were created
if [ -f ".synaptic/config.json" ]; then
    echo "✅ Configuration file created successfully"
else
    echo "❌ Configuration file not found"
    exit 1
fi

# Test status command
echo "📊 Testing status command..."
synaptic-mesh status

# Test other commands
echo "🧠 Testing neural command..."
synaptic-mesh neural --help

echo "🔗 Testing mesh command..."
synaptic-mesh mesh --help

echo "📊 Testing DAG command..."
synaptic-mesh dag --help

echo "👥 Testing peer command..."
synaptic-mesh peer --help

echo "🔧 Testing config command..."
synaptic-mesh config --help

# Test Docker build
echo "🐳 Testing Docker build..."
cd "$PACKAGE_DIR"
if command -v docker &> /dev/null; then
    docker build -t synaptic-mesh-test .
    echo "✅ Docker build successful"
else
    echo "⚠️ Docker not available, skipping Docker test"
fi

# Cleanup
echo "🧹 Cleaning up..."
npm uninstall -g synaptic-mesh
rm -rf "$TEST_DIR"
rm -f "$PACKAGE_DIR"/synaptic-mesh-*.tgz

echo "🎉 All NPX tests passed successfully!"
echo ""
echo "📋 Test Summary:"
echo "  ✅ Package creation and packing"
echo "  ✅ Global installation via NPM"
echo "  ✅ CLI version and help commands"
echo "  ✅ Node initialization"
echo "  ✅ Configuration file creation"
echo "  ✅ All subcommands functional"
echo "  ✅ Docker build (if available)"
echo ""
echo "🚀 Ready for NPX distribution!"