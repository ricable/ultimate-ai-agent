#!/bin/bash

###########################################
# Qwen 2.5 Coder Model Download Script
# Downloads optimized GGUF models for Apple Silicon
###########################################

set -e

echo "======================================"
echo "Qwen 2.5 Coder Model Downloader"
echo "======================================"

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Model directory
MODELS_DIR="$HOME/.llamaedge/models"
mkdir -p "$MODELS_DIR"

echo ""
echo "Select model size:"
echo "  1) Qwen 2.5 Coder 7B (Recommended for 16GB RAM)"
echo "  2) Qwen 2.5 Coder 14B (Recommended for 24GB RAM)"
echo "  3) Qwen 2.5 Coder 32B (Recommended for 48GB+ RAM)"
echo ""
read -p "Enter choice [1-3]: " choice

case $choice in
    1)
        MODEL_NAME="Qwen2.5-Coder-7B-Instruct"
        MODEL_FILE="qwen2.5-coder-7b-instruct-q5_k_m.gguf"
        MODEL_URL="https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct-GGUF/resolve/main/qwen2.5-coder-7b-instruct-q5_k_m.gguf"
        ;;
    2)
        MODEL_NAME="Qwen2.5-Coder-14B-Instruct"
        MODEL_FILE="qwen2.5-coder-14b-instruct-q5_k_m.gguf"
        MODEL_URL="https://huggingface.co/Qwen/Qwen2.5-Coder-14B-Instruct-GGUF/resolve/main/qwen2.5-coder-14b-instruct-q5_k_m.gguf"
        ;;
    3)
        MODEL_NAME="Qwen2.5-Coder-32B-Instruct"
        MODEL_FILE="qwen2.5-coder-32b-instruct-q5_k_m.gguf"
        MODEL_URL="https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct-GGUF/resolve/main/qwen2.5-coder-32b-instruct-q5_k_m.gguf"
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac

echo ""
echo "Downloading $MODEL_NAME..."
echo "This may take a while depending on your connection."
echo ""

MODEL_PATH="$MODELS_DIR/$MODEL_FILE"

# Check if already downloaded
if [ -f "$MODEL_PATH" ]; then
    echo -e "${YELLOW}Model already exists at: $MODEL_PATH${NC}"
    read -p "Re-download? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Using existing model."
        exit 0
    fi
fi

# Download with progress
echo "Downloading from HuggingFace..."
curl -L --progress-bar "$MODEL_URL" -o "$MODEL_PATH"

echo ""
echo -e "${GREEN}✓ Model downloaded successfully${NC}"
echo ""
echo "Model location: $MODEL_PATH"
echo ""

# Create symlink to default model
ln -sf "$MODEL_PATH" "$MODELS_DIR/default.gguf"
echo -e "${GREEN}✓ Set as default model${NC}"

echo ""
echo "======================================"
echo "Model ready!"
echo "======================================"
echo ""
echo "To start the inference server:"
echo "  MODEL_PATH=$MODEL_PATH llamaedge"
echo ""
echo "Or use the default:"
echo "  llamaedge"
echo ""
echo "The server will be available at:"
echo "  http://localhost:8080/v1"
echo ""
echo "OpenAI-compatible endpoint for agents:"
echo "  Base URL: http://localhost:8080/v1"
echo "  Model: $MODEL_NAME"
echo ""
