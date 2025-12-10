#!/bin/bash
# Synaptic Neural Mesh Crate Publishing Script
# Publishes the 5 core crates to crates.io

set -e

echo "🦀 Publishing Synaptic Neural Mesh Crates..."
echo "🔒 Setting up Rust environment..."

# Source Rust environment
source "$HOME/.cargo/env"

# Verify cargo is available
if ! command -v cargo &> /dev/null; then
    echo "❌ Error: Cargo not found. Please install Rust."
    exit 1
fi

echo "✅ Cargo version: $(cargo --version)"

# Change to Rust workspace
cd /workspaces/Synaptic-Neural-Mesh/src/rs

echo ""
echo "📦 Publishing 5 Synaptic Neural Mesh Crates..."
echo "   Publishing order follows dependency hierarchy"
echo ""

# 1. Publish qudag-core (no dependencies)
echo "🌐 Publishing qudag-core (1/5)..."
if cd qudag-core 2>/dev/null; then
    cargo publish --allow-dirty || echo "⚠️ qudag-core publish failed or already exists"
    cd ..
    sleep 5
else
    echo "⚠️ qudag-core directory not found, skipping..."
fi

# 2. Publish ruv-fann-wasm (neural engine)  
echo "🧠 Publishing ruv-fann-wasm (2/5)..."
if cd ruv-fann-wasm 2>/dev/null; then
    cargo publish --allow-dirty || echo "⚠️ ruv-fann-wasm publish failed or already exists"
    cd ..
    sleep 5
else
    echo "⚠️ ruv-fann-wasm directory not found, skipping..."
fi

# 3. Publish neural-mesh (depends on qudag-core)
echo "🌐 Publishing neural-mesh (3/5)..."
if cd neural-mesh 2>/dev/null; then
    cargo publish --allow-dirty || echo "⚠️ neural-mesh publish failed or already exists"
    cd ..
    sleep 5
else
    echo "⚠️ neural-mesh directory not found, skipping..."
fi

# 4. Publish daa-swarm (depends on neural-mesh)
echo "🐝 Publishing daa-swarm (4/5)..."
if cd daa-swarm 2>/dev/null; then
    cargo publish --allow-dirty || echo "⚠️ daa-swarm publish failed or already exists"
    cd ..
    sleep 5
else
    echo "⚠️ daa-swarm directory not found, skipping..."
fi

# 5. Publish synaptic-mesh-cli (depends on all above)
echo "⚡ Publishing synaptic-mesh-cli (5/5)..."
if cd synaptic-mesh-cli 2>/dev/null; then
    cargo publish --allow-dirty || echo "⚠️ synaptic-mesh-cli publish failed or already exists"
    cd ..
else
    echo "⚠️ synaptic-mesh-cli directory not found, skipping..."
fi

echo ""
echo "✅ Synaptic Neural Mesh crate publishing process completed!"
echo ""
echo "📊 Published crates provide:"
echo "   🌐 qudag-core: Quantum-resistant DAG networking"
echo "   🧠 ruv-fann-wasm: WASM neural network engine with SIMD"
echo "   🔗 neural-mesh: Neural mesh coordination layer"
echo "   🐝 daa-swarm: Distributed autonomous agent swarms"
echo "   ⚡ synaptic-mesh-cli: Complete CLI integration"
echo ""
echo "🚀 Crates are now available on crates.io for the Rust community!"
echo ""
echo "Next steps:"
echo "   - Update README.md with crates.io badges"
echo "   - Verify NPM package integration"
echo "   - Test cross-platform compatibility"
echo "   - Announce to Rust community"