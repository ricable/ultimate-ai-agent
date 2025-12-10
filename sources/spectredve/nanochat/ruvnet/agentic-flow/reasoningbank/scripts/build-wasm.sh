#!/bin/bash
# Build WASM packages for ReasoningBank
#
# Usage: ./scripts/build-wasm.sh [target]
#   target: web, nodejs, bundler (default: all)

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
WASM_CRATE="$PROJECT_ROOT/crates/reasoningbank-wasm"

TARGET="${1:-all}"

echo "🦀 Building ReasoningBank WASM..."
echo "Target: $TARGET"
echo "Crate: $WASM_CRATE"

cd "$WASM_CRATE"

build_for_target() {
    local target=$1
    local out_dir="$2"

    echo "Building for $target..."
    wasm-pack build \
        --target "$target" \
        --out-dir "$out_dir" \
        --release \
        --scope reasoningbank \
        -- --features wasm-storage

    echo "✅ Built $target -> $out_dir"
}

if [ "$TARGET" = "all" ]; then
    build_for_target "web" "pkg-web"
    build_for_target "nodejs" "pkg-node"
    build_for_target "bundler" "pkg-bundler"
elif [ "$TARGET" = "web" ] || [ "$TARGET" = "nodejs" ] || [ "$TARGET" = "bundler" ]; then
    build_for_target "$TARGET" "pkg-$TARGET"
else
    echo "❌ Unknown target: $TARGET"
    echo "Available targets: web, nodejs, bundler, all"
    exit 1
fi

# Optimize WASM files
echo "🔧 Optimizing WASM binaries..."
for dir in pkg-*/; do
    if [ -f "$dir"/*.wasm ]; then
        echo "Optimizing $dir..."
        wasm-opt -O4 --enable-simd "$dir"/*.wasm -o "$dir"/*.wasm.opt
        mv "$dir"/*.wasm.opt "$dir"/*.wasm
    fi
done

# Copy to agentic-flow npm package
AGENTIC_FLOW_WASM="$PROJECT_ROOT/../agentic-flow/wasm/reasoningbank"
mkdir -p "$AGENTIC_FLOW_WASM/node" "$AGENTIC_FLOW_WASM/web"

if [ -d "pkg-node" ]; then
    echo "📦 Copying Node.js package to $AGENTIC_FLOW_WASM/node"
    cp -r pkg-node/* "$AGENTIC_FLOW_WASM/node/"
fi

if [ -d "pkg-web" ]; then
    echo "📦 Copying Web package to $AGENTIC_FLOW_WASM/web"
    cp -r pkg-web/* "$AGENTIC_FLOW_WASM/web/"
fi

echo ""
echo "✨ WASM build complete!"
echo ""
echo "Packages:"
ls -lh pkg-*/reasoningbank_wasm_bg.wasm 2>/dev/null || echo "No WASM files found"
echo ""
echo "Next steps:"
echo "  1. cd ../agentic-flow"
echo "  2. npm run build"
echo "  3. npm test"
