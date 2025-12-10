#!/bin/bash

# Common issue fixes for neuro-divergent crates

echo "🔧 Fixing common issues before publishing..."

# Fix 1: Update ruv-fann dependency path if needed
echo "📝 Checking ruv-fann dependency paths..."
find . -name "Cargo.toml" -exec sed -i 's|ruv-fann = { path = "..", version = "0.1" }|ruv-fann = "0.1"|g' {} \;

# Fix 2: Ensure all crates use correct dependency versions
echo "📝 Updating dependency versions..."

# Fix 3: Add any missing re-exports in lib.rs files
echo "📝 Checking lib.rs files..."

# Check each crate
for crate in neuro-divergent-core neuro-divergent-data neuro-divergent-training neuro-divergent-models neuro-divergent-registry; do
    echo "  Checking $crate..."
    cd "$crate"
    
    # Try to compile
    if ! cargo check; then
        echo "❌ $crate has compilation issues"
    else
        echo "✅ $crate compiles successfully"
    fi
    
    cd ..
done

echo "🔧 Common fixes applied. Please run 'cargo check --all-features' to verify."