#!/bin/bash
# WASM Production Build Optimization Script

set -e

echo "🚀 Optimizing WASM Module for Production..."

# Navigate to WASM directory
cd "$(dirname "$0")/../wasm"

echo "📦 Installing wasm-pack if needed..."
if ! command -v wasm-pack &> /dev/null; then
  echo "Installing wasm-pack..."
  curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh
fi

echo "📦 Installing wasm-opt if needed..."
if ! command -v wasm-opt &> /dev/null; then
  echo "Installing binaryen (wasm-opt)..."
  if [[ "$OSTYPE" == "darwin"* ]]; then
    brew install binaryen
  elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    sudo apt-get update && sudo apt-get install -y binaryen
  fi
fi

echo "🔧 Building WASM with optimizations..."
wasm-pack build . \
  --target nodejs \
  --release \
  --scope agentdb \
  -- --features "simd"

echo "⚡ Running wasm-opt with maximum optimizations..."
wasm-opt -O4 -c --enable-simd \
  --enable-bulk-memory \
  --enable-mutable-globals \
  --enable-nontrapping-float-to-int \
  pkg/attention_bg.wasm \
  -o pkg/attention_bg.wasm

echo "📏 Compressing WASM bundle..."
gzip -k -f pkg/attention_bg.wasm

echo "📊 Build statistics:"
echo "Original WASM:"
ls -lh pkg/attention_bg.wasm
echo "Compressed:"
ls -lh pkg/attention_bg.wasm.gz

echo "✅ WASM optimization complete!"
echo ""
echo "Performance improvements:"
echo "  - O4 optimization: Maximum performance"
echo "  - SIMD enabled: 2x faster vector ops"
echo "  - Bulk memory: Faster array operations"
echo "  - Compressed: Smaller bundle size"
