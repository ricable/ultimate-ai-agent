#!/bin/bash
set -e

# Build script for Synaptic Neural Mesh Rust components
echo "🦀 Building Synaptic Neural Mesh Rust Components"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the right directory
if [ ! -f "Cargo.toml" ]; then
    print_error "Must be run from src/rs directory"
    exit 1
fi

# Build type (default: release)
BUILD_TYPE=${1:-release}
FEATURES=${2:-""}

print_status "Building in $BUILD_TYPE mode"
if [ -n "$FEATURES" ]; then
    print_status "Features: $FEATURES"
fi

# Build QuDAG Core
print_status "Building QuDAG Core (quantum-resistant DAG networking)..."
cd qudag-core
if [ "$BUILD_TYPE" = "release" ]; then
    cargo build --release $FEATURES
else
    cargo build $FEATURES
fi
print_success "QuDAG Core built successfully"
cd ..

# Build ruv-FANN WASM
print_status "Building ruv-FANN WASM (neural networks with WASM optimization)..."
cd ruv-fann-wasm
if [ "$BUILD_TYPE" = "release" ]; then
    cargo build --release --target wasm32-unknown-unknown $FEATURES
    wasm-pack build --target web --out-dir pkg-web
    wasm-pack build --target nodejs --out-dir pkg-node
else
    cargo build --target wasm32-unknown-unknown $FEATURES
fi
print_success "ruv-FANN WASM built successfully"
cd ..

# Build Neural Mesh
print_status "Building Neural Mesh (distributed cognition layer)..."
cd neural-mesh
if [ "$BUILD_TYPE" = "release" ]; then
    cargo build --release $FEATURES
else
    cargo build $FEATURES
fi
print_success "Neural Mesh built successfully"
cd ..

# Build DAA Swarm
print_status "Building DAA Swarm (dynamic agent architecture)..."
cd daa-swarm
if [ "$BUILD_TYPE" = "release" ]; then
    cargo build --release $FEATURES
else
    cargo build $FEATURES
fi
print_success "DAA Swarm built successfully"
cd ..

# Build CLI
print_status "Building Synaptic Mesh CLI..."
cd synaptic-mesh-cli
if [ "$BUILD_TYPE" = "release" ]; then
    cargo build --release $FEATURES
else
    cargo build $FEATURES
fi
print_success "CLI built successfully"
cd ..

# Run tests
print_status "Running tests..."
cargo test --workspace
print_success "All tests passed"

# Check formatting
print_status "Checking code formatting..."
cargo fmt --check
print_success "Code formatting is correct"

# Run clippy
print_status "Running Clippy lints..."
cargo clippy --workspace -- -D warnings
print_success "Clippy checks passed"

# Build documentation
print_status "Building documentation..."
cargo doc --workspace --no-deps
print_success "Documentation built"

# Generate size report for WASM
if [ "$BUILD_TYPE" = "release" ]; then
    print_status "Generating WASM size report..."
    cd ruv-fann-wasm
    if [ -f "pkg-web/ruv_fann_wasm_bg.wasm" ]; then
        SIZE=$(wc -c < pkg-web/ruv_fann_wasm_bg.wasm)
        SIZE_KB=$((SIZE / 1024))
        if [ $SIZE_KB -lt 2048 ]; then
            print_success "WASM bundle size: ${SIZE_KB}KB (target: <2MB)"
        else
            print_warning "WASM bundle size: ${SIZE_KB}KB (exceeds 2MB target)"
        fi
    fi
    cd ..
fi

print_success "🎉 Build completed successfully!"

echo ""
echo "📦 Built components:"
echo "  ✓ QuDAG Core - Quantum-resistant DAG networking"
echo "  ✓ ruv-FANN WASM - Neural networks with WASM+SIMD"
echo "  ✓ Neural Mesh - Distributed cognition layer"  
echo "  ✓ DAA Swarm - Dynamic agent architecture"
echo "  ✓ CLI - Command-line interface"
echo ""
echo "🚀 To run: target/$BUILD_TYPE/synaptic-mesh-cli --help"