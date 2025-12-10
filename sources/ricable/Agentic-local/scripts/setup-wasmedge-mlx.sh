#!/bin/bash

###########################################
# WasmEdge + MLX Setup Script for Mac Silicon
# Builds WasmEdge with WASI-NN MLX backend support
###########################################

set -e

echo "======================================"
echo "WasmEdge MLX Setup for Apple Silicon"
echo "======================================"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo -e "${RED}Error: This script is designed for macOS only${NC}"
    exit 1
fi

# Check if running on Apple Silicon
ARCH=$(uname -m)
if [[ "$ARCH" != "arm64" ]]; then
    echo -e "${YELLOW}Warning: This script is optimized for Apple Silicon (M1/M2/M3)${NC}"
    echo -e "${YELLOW}You are running on: $ARCH${NC}"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo -e "${GREEN}✓ System check passed${NC}"
echo ""

# Step 1: Check and install Homebrew dependencies
echo "Step 1: Installing dependencies..."

if ! command -v brew &> /dev/null; then
    echo -e "${RED}Homebrew not found. Please install Homebrew first:${NC}"
    echo "/bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
    exit 1
fi

echo "Installing build dependencies via Homebrew..."
brew install cmake ninja llvm boost

echo -e "${GREEN}✓ Dependencies installed${NC}"
echo ""

# Step 2: Install MLX framework
echo "Step 2: Installing Apple MLX framework..."

if ! pip3 show mlx &> /dev/null; then
    echo "Installing mlx via pip..."
    pip3 install mlx mlx-lm
else
    echo -e "${GREEN}✓ MLX already installed${NC}"
fi

echo ""

# Step 3: Clone or update WasmEdge repository
echo "Step 3: Setting up WasmEdge source..."

WASMEDGE_DIR="$HOME/.wasmedge-build"
mkdir -p "$WASMEDGE_DIR"
cd "$WASMEDGE_DIR"

if [ -d "WasmEdge" ]; then
    echo "Updating existing WasmEdge repository..."
    cd WasmEdge
    git fetch origin
    git checkout master
    git pull
else
    echo "Cloning WasmEdge repository..."
    git clone https://github.com/WasmEdge/WasmEdge.git
    cd WasmEdge
fi

echo -e "${GREEN}✓ WasmEdge source ready${NC}"
echo ""

# Step 4: Build WasmEdge with MLX support
echo "Step 4: Building WasmEdge with WASI-NN MLX backend..."
echo "This may take 15-30 minutes..."

# Clean previous build
rm -rf build
mkdir -p build

# Configure with CMake
echo "Configuring build with MLX backend..."
cmake -GNinja -Bbuild \
    -DCMAKE_BUILD_TYPE=Release \
    -DWASMEDGE_PLUGIN_WASI_NN_BACKEND="mlx" \
    -DWASMEDGE_BUILD_PLUGINS=ON \
    -DCMAKE_INSTALL_PREFIX="$HOME/.wasmedge" \
    .

# Build
echo "Building (this will take a while)..."
cmake --build build --parallel $(sysctl -n hw.ncpu)

echo -e "${GREEN}✓ Build completed${NC}"
echo ""

# Step 5: Install WasmEdge
echo "Step 5: Installing WasmEdge..."

cmake --install build

# Add to PATH if not already present
SHELL_RC="$HOME/.zshrc"
if [[ -f "$HOME/.bashrc" ]]; then
    SHELL_RC="$HOME/.bashrc"
fi

if ! grep -q ".wasmedge/bin" "$SHELL_RC"; then
    echo "" >> "$SHELL_RC"
    echo "# WasmEdge" >> "$SHELL_RC"
    echo "export PATH=\"\$HOME/.wasmedge/bin:\$PATH\"" >> "$SHELL_RC"
    echo "export WASMEDGE_PLUGIN_PATH=\"\$HOME/.wasmedge/lib/wasmedge\"" >> "$SHELL_RC"
    echo -e "${GREEN}✓ Added WasmEdge to PATH in $SHELL_RC${NC}"
else
    echo -e "${GREEN}✓ WasmEdge already in PATH${NC}"
fi

echo ""

# Step 6: Verify installation
echo "Step 6: Verifying installation..."

export PATH="$HOME/.wasmedge/bin:$PATH"
export WASMEDGE_PLUGIN_PATH="$HOME/.wasmedge/lib/wasmedge"

if command -v wasmedge &> /dev/null; then
    echo -e "${GREEN}✓ WasmEdge installed successfully${NC}"
    wasmedge --version
else
    echo -e "${RED}Error: WasmEdge installation verification failed${NC}"
    exit 1
fi

# Check for MLX plugin
if [ -f "$HOME/.wasmedge/lib/wasmedge/libwasmedgePluginWasiNN.dylib" ]; then
    echo -e "${GREEN}✓ WASI-NN plugin installed${NC}"
    echo "Plugin location: $HOME/.wasmedge/lib/wasmedge/libwasmedgePluginWasiNN.dylib"
else
    echo -e "${YELLOW}Warning: WASI-NN plugin not found at expected location${NC}"
fi

echo ""
echo "======================================"
echo -e "${GREEN}Installation Complete!${NC}"
echo "======================================"
echo ""
echo "Next steps:"
echo "1. Restart your terminal or run: source $SHELL_RC"
echo "2. Install LlamaEdge: ./scripts/setup-llamaedge.sh"
echo "3. Download a model: ./scripts/download-qwen-coder.sh"
echo ""
echo "To verify MLX backend is working:"
echo "  wasmedge --version"
echo "  ls \$WASMEDGE_PLUGIN_PATH"
echo ""
