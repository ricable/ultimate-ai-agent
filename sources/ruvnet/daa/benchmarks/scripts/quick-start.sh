#!/bin/bash
# Quick start script for DAA benchmarks
# Checks dependencies and runs initial benchmarks

set -e

echo "🚀 DAA Benchmark Quick Start"
echo ""

# Check Node.js
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js 18+ first."
    exit 1
fi

NODE_VERSION=$(node -v | cut -d'v' -f2 | cut -d'.' -f1)
if [ "$NODE_VERSION" -lt 18 ]; then
    echo "❌ Node.js version is too old. Please upgrade to Node.js 18+ (current: $(node -v))"
    exit 1
fi

echo "✓ Node.js $(node -v) detected"

# Check Rust
if ! command -v cargo &> /dev/null; then
    echo "⚠️  Rust is not installed. Native benchmarks will be skipped."
    echo "   Install from: https://rustup.rs/"
    RUST_AVAILABLE=false
else
    echo "✓ Rust $(rustc --version | cut -d' ' -f2) detected"
    RUST_AVAILABLE=true
fi

# Check wasm-pack
if ! command -v wasm-pack &> /dev/null; then
    echo "⚠️  wasm-pack is not installed. WASM benchmarks may fail."
    echo "   Install with: curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh"
fi

echo ""
echo "📦 Installing dependencies..."
npm install

echo ""
echo "🔨 Building WASM bindings..."
if [ -d "../qudag/qudag-wasm" ]; then
    cd ../qudag/qudag-wasm
    if wasm-pack build --target nodejs --out-dir pkg-node; then
        echo "✓ WASM bindings built successfully"
    else
        echo "⚠️  WASM build failed. Some benchmarks may not run."
    fi
    cd ../../benchmarks
else
    echo "⚠️  qudag-wasm directory not found. Skipping WASM build."
fi

echo ""
echo "🔨 Building native bindings (optional)..."
if [ -d "../qudag/qudag-napi" ] && [ "$RUST_AVAILABLE" = true ]; then
    cd ../qudag/qudag-napi
    if npm run build 2>/dev/null; then
        echo "✓ Native bindings built successfully"
    else
        echo "⚠️  Native build failed. Comparison benchmarks will be limited."
    fi
    cd ../../benchmarks
else
    echo "⚠️  qudag-napi directory not found or Rust not available. Skipping native build."
fi

echo ""
echo "📊 Creating reports directory..."
mkdir -p reports

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Run crypto benchmarks:      npm run bench:crypto"
echo "  2. Run comparison:             npm run bench:crypto-compare"
echo "  3. Run all benchmarks:         npm run bench:compare"
echo "  4. Generate HTML report:       npm run report:html"
echo "  5. View report:                open reports/benchmark_report.html"
echo ""
echo "For more information, see README.md"
echo ""
