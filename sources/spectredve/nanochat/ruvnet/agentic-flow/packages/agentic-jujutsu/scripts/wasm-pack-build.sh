#!/bin/bash
# Build WASM for all targets
# Usage: ./scripts/wasm-pack-build.sh [--release]

set -e

PROFILE=${1:-dev}
BUILD_FLAG=""

if [ "$PROFILE" = "--release" ]; then
    BUILD_FLAG="--release"
    echo "🚀 Building WASM in RELEASE mode..."
else
    echo "🔧 Building WASM in DEV mode..."
fi

# Clean previous builds
echo "🧹 Cleaning previous builds..."
rm -rf pkg

# Build for web (browser with ES modules)
echo "📦 Building for web target..."
wasm-pack build --target web --out-dir pkg/web $BUILD_FLAG

# Build for Node.js
echo "📦 Building for nodejs target..."
wasm-pack build --target nodejs --out-dir pkg/node $BUILD_FLAG

# Build for bundler (webpack, rollup, etc.)
echo "📦 Building for bundler target..."
wasm-pack build --target bundler --out-dir pkg/bundler $BUILD_FLAG

# Build for deno
echo "📦 Building for deno target..."
wasm-pack build --target deno --out-dir pkg/deno $BUILD_FLAG

echo ""
echo "✅ All WASM builds completed successfully!"
echo ""
echo "📊 Bundle sizes:"
echo "───────────────────────────────────────"
for target in web node bundler deno; do
    if [ -f "pkg/$target/agentic_jujutsu_bg.wasm" ]; then
        size=$(ls -lh "pkg/$target/agentic_jujutsu_bg.wasm" | awk '{print $5}')
        echo "  $target: $size"
    fi
done

# Run wasm-opt if available (for release builds)
if [ "$PROFILE" = "--release" ] && command -v wasm-opt &> /dev/null; then
    echo ""
    echo "🔧 Running wasm-opt for size optimization..."
    for target in web node bundler deno; do
        if [ -f "pkg/$target/agentic_jujutsu_bg.wasm" ]; then
            wasm-opt -Oz -o "pkg/$target/agentic_jujutsu_bg.wasm.opt" "pkg/$target/agentic_jujutsu_bg.wasm"
            mv "pkg/$target/agentic_jujutsu_bg.wasm.opt" "pkg/$target/agentic_jujutsu_bg.wasm"
        fi
    done
    echo "✅ Optimization complete!"
    echo ""
    echo "📊 Optimized bundle sizes:"
    echo "───────────────────────────────────────"
    for target in web node bundler deno; do
        if [ -f "pkg/$target/agentic_jujutsu_bg.wasm" ]; then
            size=$(ls -lh "pkg/$target/agentic_jujutsu_bg.wasm" | awk '{print $5}')
            echo "  $target: $size"
        fi
    done
fi

echo ""
echo "📁 Output directories:"
echo "  pkg/web/     - Browser with ES modules"
echo "  pkg/node/    - Node.js CommonJS"
echo "  pkg/bundler/ - Webpack/Rollup/Vite"
echo "  pkg/deno/    - Deno runtime"
