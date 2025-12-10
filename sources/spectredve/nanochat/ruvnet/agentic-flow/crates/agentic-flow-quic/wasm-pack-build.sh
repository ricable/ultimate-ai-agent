#!/bin/bash
# Build WASM package for agentic-flow-quic

set -e

echo "Building WASM package with wasm-pack..."

# Build for Node.js target
wasm-pack build \
  --target nodejs \
  --out-dir pkg \
  --features wasm \
  --release

echo "WASM build complete!"
echo "Output: pkg/"
ls -lh pkg/
