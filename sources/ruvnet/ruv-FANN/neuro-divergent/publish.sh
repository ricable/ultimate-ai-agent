#!/bin/bash

# Neuro-Divergent Crate Publishing Script
# Publishes all crates in correct dependency order with path dependency handling

set -e  # Exit on any error

echo "🚀 Publishing Neuro-Divergent Crates to crates.io"
echo "=================================================="

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Backup original Cargo.toml files
backup_tomls() {
    echo -e "${BLUE}📋 Backing up Cargo.toml files...${NC}"
    find . -name "Cargo.toml" -exec cp {} {}.backup \;
}

# Restore original Cargo.toml files
restore_tomls() {
    echo -e "${BLUE}📋 Restoring Cargo.toml files...${NC}"
    find . -name "Cargo.toml.backup" | while read -r backup; do
        original="${backup%.backup}"
        mv "$backup" "$original"
    done
}

# Replace path dependencies with version dependencies for publishing
prepare_for_publishing() {
    local crate_path=$1
    echo -e "${BLUE}🔧 Preparing $crate_path for publishing...${NC}"
    
    cd "$crate_path"
    
    # Replace path dependencies with version dependencies
    sed -i.tmp 's/{ version = "0.1.0", path = "[^"]*" }/{ version = "0.1.0" }/g' Cargo.toml
    rm -f Cargo.toml.tmp
    
    cd - > /dev/null
}

# Function to publish a crate
publish_crate() {
    local crate_name=$1
    local crate_path=$2
    
    echo -e "${YELLOW}📦 Publishing $crate_name...${NC}"
    
    # Prepare for publishing (handle path dependencies)
    prepare_for_publishing "$crate_path"
    
    cd "$crate_path"
    
    # Run dry-run first
    echo "  🔍 Running dry-run..."
    if ! cargo publish --dry-run; then
        echo -e "${RED}❌ Dry-run failed for $crate_name${NC}"
        cd - > /dev/null
        restore_tomls
        exit 1
    fi
    
    # Actual publish
    echo "  📤 Publishing to crates.io..."
    if ! cargo publish; then
        echo -e "${RED}❌ Publishing failed for $crate_name${NC}"
        cd - > /dev/null
        restore_tomls
        exit 1
    fi
    
    echo -e "${GREEN}✅ Successfully published $crate_name${NC}"
    
    # Wait for availability on crates.io
    if [ "$crate_name" != "neuro-divergent" ]; then
        echo "  ⏳ Waiting for crate to be available on crates.io..."
        sleep 120  # Increased wait time for crates.io indexing
    fi
    
    cd - > /dev/null
}

# Cleanup function for error handling
cleanup() {
    echo -e "${YELLOW}🧹 Cleaning up...${NC}"
    restore_tomls
}

# Set up cleanup trap
trap cleanup EXIT

# Pre-publishing checks
echo "🔍 Pre-publishing verification..."

# Check if logged into cargo
if ! cargo login --help > /dev/null 2>&1; then
    echo -e "${RED}❌ Cargo not available${NC}"
    exit 1
fi

# Verify we're in the right directory
if [ ! -f "Cargo.toml" ] || [ ! -d "neuro-divergent-core" ]; then
    echo -e "${RED}❌ Not in neuro-divergent root directory${NC}"
    exit 1
fi

# Check that all crates compile
echo "  🔧 Checking compilation..."
if ! cargo check --all-features; then
    echo -e "${RED}❌ Compilation check failed${NC}"
    echo "Please fix compilation errors before publishing"
    exit 1
fi

echo -e "${GREEN}✅ Pre-publishing checks passed${NC}"
echo ""

# Backup all Cargo.toml files before making changes
backup_tomls

# Publish in correct dependency order
echo "📦 Publishing crates in dependency order..."
echo ""

# 1. Core (foundation, no internal dependencies)
publish_crate "neuro-divergent-core" "neuro-divergent-core"

# 2. Registry (no internal dependencies)
publish_crate "neuro-divergent-registry" "neuro-divergent-registry"

# 3. Models (no internal dependencies, only ruv-fann)
publish_crate "neuro-divergent-models" "neuro-divergent-models"

# 4. Data (depends on core)
publish_crate "neuro-divergent-data" "neuro-divergent-data"

# 5. Training (depends on core)  
publish_crate "neuro-divergent-training" "neuro-divergent-training"

# 6. Main crate (depends on all others)
publish_crate "neuro-divergent" "."

echo ""
echo -e "${GREEN}🎉 All neuro-divergent crates published successfully!${NC}"
echo ""
echo "📋 Published crates:"
echo "  • neuro-divergent-core"
echo "  • neuro-divergent-registry"
echo "  • neuro-divergent-models"
echo "  • neuro-divergent-data"
echo "  • neuro-divergent-training"
echo "  • neuro-divergent"
echo ""
echo "🔗 Available at: https://crates.io/search?q=neuro-divergent"
echo ""
echo "📚 Next steps:"
echo "  1. Update documentation"
echo "  2. Create GitHub release"
echo "  3. Announce to community"
echo "  4. Update examples and tutorials"
echo ""
echo "✨ Publication completed successfully!"

# The cleanup trap will restore the original Cargo.toml files