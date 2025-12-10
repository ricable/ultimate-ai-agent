#!/bin/bash

###########################################
# LlamaEdge Setup Script
# Installs LlamaEdge binaries for local inference
###########################################

set -e

echo "======================================"
echo "LlamaEdge Installation"
echo "======================================"

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Check WasmEdge is installed
if ! command -v wasmedge &> /dev/null; then
    echo -e "${RED}Error: WasmEdge not found. Please run setup-wasmedge-mlx.sh first${NC}"
    exit 1
fi

echo -e "${GREEN}✓ WasmEdge detected${NC}"
echo ""

# Create installation directory
LLAMAEDGE_DIR="$HOME/.llamaedge"
mkdir -p "$LLAMAEDGE_DIR/bin"

echo "Step 1: Downloading LlamaEdge binaries..."

# Download the latest LlamaEdge API server
LLAMAEDGE_VERSION="0.14.9"
API_SERVER_URL="https://github.com/LlamaEdge/LlamaEdge/releases/download/${LLAMAEDGE_VERSION}/llama-api-server.wasm"

echo "Downloading llama-api-server.wasm..."
curl -L "$API_SERVER_URL" -o "$LLAMAEDGE_DIR/bin/llama-api-server.wasm"

echo -e "${GREEN}✓ Downloaded LlamaEdge API server${NC}"
echo ""

# Create convenience wrapper script
echo "Step 2: Creating launcher scripts..."

cat > "$LLAMAEDGE_DIR/bin/llamaedge" << 'EOF'
#!/bin/bash
# LlamaEdge launcher script

LLAMAEDGE_DIR="$HOME/.llamaedge"
WASM_FILE="$LLAMAEDGE_DIR/bin/llama-api-server.wasm"

if [ ! -f "$WASM_FILE" ]; then
    echo "Error: llama-api-server.wasm not found"
    exit 1
fi

# Default configuration
MODEL_PATH="${MODEL_PATH:-$HOME/.llamaedge/models/default.gguf}"
CONTEXT_SIZE="${CONTEXT_SIZE:-32768}"
PORT="${PORT:-8080}"
PROMPT_TEMPLATE="${PROMPT_TEMPLATE:-chatml}"

exec wasmedge \
    --dir .:. \
    --nn-preload default:GGML:AUTO:"$MODEL_PATH" \
    "$WASM_FILE" \
    --model-name "$(basename $MODEL_PATH .gguf)" \
    --ctx-size "$CONTEXT_SIZE" \
    --prompt-template "$PROMPT_TEMPLATE" \
    --port "$PORT" \
    "$@"
EOF

chmod +x "$LLAMAEDGE_DIR/bin/llamaedge"

echo -e "${GREEN}✓ Created llamaedge launcher${NC}"
echo ""

# Add to PATH
SHELL_RC="$HOME/.zshrc"
if [[ -f "$HOME/.bashrc" ]]; then
    SHELL_RC="$HOME/.bashrc"
fi

if ! grep -q ".llamaedge/bin" "$SHELL_RC"; then
    echo "" >> "$SHELL_RC"
    echo "# LlamaEdge" >> "$SHELL_RC"
    echo "export PATH=\"\$HOME/.llamaedge/bin:\$PATH\"" >> "$SHELL_RC"
    echo -e "${GREEN}✓ Added LlamaEdge to PATH${NC}"
fi

echo ""
echo "======================================"
echo -e "${GREEN}LlamaEdge Installed Successfully!${NC}"
echo "======================================"
echo ""
echo "Installation directory: $LLAMAEDGE_DIR"
echo ""
echo "Next steps:"
echo "1. Restart terminal or run: source $SHELL_RC"
echo "2. Download a model: ./scripts/download-qwen-coder.sh"
echo "3. Start server: llamaedge"
echo ""
echo "Configuration via environment variables:"
echo "  MODEL_PATH     - Path to GGUF model file"
echo "  CONTEXT_SIZE   - Context window (default: 32768)"
echo "  PORT           - Server port (default: 8080)"
echo "  PROMPT_TEMPLATE - Template format (default: chatml)"
echo ""
