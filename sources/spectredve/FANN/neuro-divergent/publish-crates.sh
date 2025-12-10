#!/bin/bash

# Neuro-Divergent Crate Publishing Script
# Publishes all crates in correct dependency order with fixes

set -e  # Exit on any error

echo "🚀 Publishing Neuro-Divergent Crates to crates.io"
echo "=================================================="

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to check if we're logged into cargo
check_cargo_auth() {
    echo -e "${BLUE}🔐 Checking cargo authentication...${NC}"
    if ! cargo login --help > /dev/null 2>&1; then
        echo -e "${RED}❌ Cargo not available${NC}"
        exit 1
    fi
    echo -e "${GREEN}✅ Cargo authentication ready${NC}"
}

# Function to fix dependencies for publishing
fix_dependencies() {
    echo -e "${BLUE}🔧 Fixing dependencies for publishing...${NC}"
    
    # Replace path dependencies with version dependencies for publishing
    echo "  Updating internal crate dependencies..."
    
    # Fix path dependencies in each crate for publishing
    for crate in neuro-divergent-core neuro-divergent-data neuro-divergent-training neuro-divergent-models neuro-divergent-registry .; do
        if [ -f "$crate/Cargo.toml" ]; then
            echo "    Fixing $crate/Cargo.toml..."
            # Create backup
            cp "$crate/Cargo.toml" "$crate/Cargo.toml.backup"
            
            # Replace path dependencies with version dependencies
            sed -i.tmp 's|neuro-divergent-core = { version = "0.1.0", path = "../neuro-divergent-core" }|neuro-divergent-core = "0.1.0"|g' "$crate/Cargo.toml"
            sed -i.tmp 's|neuro-divergent-data = { version = "0.1.0", path = "../neuro-divergent-data" }|neuro-divergent-data = "0.1.0"|g' "$crate/Cargo.toml"
            sed -i.tmp 's|neuro-divergent-training = { version = "0.1.0", path = "../neuro-divergent-training" }|neuro-divergent-training = "0.1.0"|g' "$crate/Cargo.toml"
            sed -i.tmp 's|neuro-divergent-models = { version = "0.1.0", path = "../neuro-divergent-models" }|neuro-divergent-models = "0.1.0"|g' "$crate/Cargo.toml"
            sed -i.tmp 's|neuro-divergent-registry = { version = "0.1.0", path = "../neuro-divergent-registry" }|neuro-divergent-registry = "0.1.0"|g' "$crate/Cargo.toml"
            
            # Clean up temp files
            rm -f "$crate/Cargo.toml.tmp"
        fi
    done
    
    echo -e "${GREEN}✅ Dependencies fixed for publishing${NC}"
}

# Function to restore dependencies after publishing
restore_dependencies() {
    echo -e "${BLUE}🔄 Restoring original dependencies...${NC}"
    
    for crate in neuro-divergent-core neuro-divergent-data neuro-divergent-training neuro-divergent-models neuro-divergent-registry .; do
        if [ -f "$crate/Cargo.toml.backup" ]; then
            echo "    Restoring $crate/Cargo.toml..."
            mv "$crate/Cargo.toml.backup" "$crate/Cargo.toml"
        fi
    done
    
    echo -e "${GREEN}✅ Original dependencies restored${NC}"
}

# Function to publish a crate
publish_crate() {
    local crate_name=$1
    local crate_path=$2
    local skip_path_fix=$3
    
    echo -e "${YELLOW}📦 Publishing $crate_name...${NC}"
    
    cd "$crate_path"
    
    # Fix path dependencies for this crate if not skipping
    if [ "$skip_path_fix" != "skip" ]; then
        echo "  🔧 Fixing path dependencies for publishing..."
        
        # Create backup
        cp Cargo.toml Cargo.toml.backup
        
        # For crates other than core, replace internal path deps with version deps
        if [ "$crate_name" != "neuro-divergent-core" ]; then
            # Replace path dependencies with published versions
            sed -i.tmp 's|neuro-divergent-core = { version = "0.1.0", path = "../neuro-divergent-core" }|neuro-divergent-core = "0.1.0"|g' Cargo.toml
            sed -i.tmp 's|neuro-divergent-data = { version = "0.1.0", path = "../neuro-divergent-data" }|neuro-divergent-data = "0.1.0"|g' Cargo.toml
            sed -i.tmp 's|neuro-divergent-training = { version = "0.1.0", path = "../neuro-divergent-training" }|neuro-divergent-training = "0.1.0"|g' Cargo.toml
            sed -i.tmp 's|neuro-divergent-models = { version = "0.1.0", path = "../neuro-divergent-models" }|neuro-divergent-models = "0.1.0"|g' Cargo.toml
            sed -i.tmp 's|neuro-divergent-registry = { version = "0.1.0", path = "../neuro-divergent-registry" }|neuro-divergent-registry = "0.1.0"|g' Cargo.toml
            
            # Clean up temp files
            rm -f Cargo.toml.tmp
        fi
    fi
    
    # Run dry-run first
    echo "  🔍 Running dry-run..."
    if ! cargo publish --dry-run; then
        echo -e "${RED}❌ Dry-run failed for $crate_name${NC}"
        
        # Restore backup
        if [ -f "Cargo.toml.backup" ]; then
            mv Cargo.toml.backup Cargo.toml
        fi
        
        cd - > /dev/null
        exit 1
    fi
    
    # Actual publish
    echo "  📤 Publishing to crates.io..."
    if ! cargo publish; then
        echo -e "${RED}❌ Publishing failed for $crate_name${NC}"
        
        # Restore backup
        if [ -f "Cargo.toml.backup" ]; then
            mv Cargo.toml.backup Cargo.toml
        fi
        
        cd - > /dev/null
        exit 1
    fi
    
    echo -e "${GREEN}✅ Successfully published $crate_name${NC}"
    
    # Restore original Cargo.toml
    if [ -f "Cargo.toml.backup" ]; then
        mv Cargo.toml.backup Cargo.toml
    fi
    
    # Wait for availability (except for the last crate)
    if [ "$crate_name" != "neuro-divergent" ]; then
        echo "  ⏳ Waiting for crate to be available on crates.io..."
        sleep 120  # Wait 2 minutes for crate to be available
    fi
    
    cd - > /dev/null
}

# Cleanup function for script interruption
cleanup() {
    echo -e "${YELLOW}🧹 Cleaning up...${NC}"
    restore_dependencies
    exit 1
}

# Set up trap for cleanup
trap cleanup INT TERM

# Verify we're in the right directory
if [ ! -f "Cargo.toml" ] || [ ! -d "neuro-divergent-core" ]; then
    echo -e "${RED}❌ Not in neuro-divergent root directory${NC}"
    echo "Please run this script from the neuro-divergent directory"
    exit 1
fi

# Check cargo authentication
check_cargo_auth

# Pre-publishing checks
echo -e "${BLUE}🔍 Pre-publishing verification...${NC}"

# Check basic compilation
echo "  🔧 Checking basic compilation..."
if ! cargo check --manifest-path neuro-divergent-core/Cargo.toml; then
    echo -e "${RED}❌ neuro-divergent-core compilation check failed${NC}"
    echo "Please fix compilation errors before publishing"
    exit 1
fi

echo -e "${GREEN}✅ Pre-publishing checks passed${NC}"
echo ""

# Publish in dependency order
echo "📦 Publishing crates in dependency order..."
echo ""

# 1. Core (foundation - no dependencies on other neuro-divergent crates)
publish_crate "neuro-divergent-core" "neuro-divergent-core" "skip"

# 2. Data (depends on core)
publish_crate "neuro-divergent-data" "neuro-divergent-data"

# 3. Training (depends on core)  
publish_crate "neuro-divergent-training" "neuro-divergent-training"

# 4. Models (depends on core, training, data)
publish_crate "neuro-divergent-models" "neuro-divergent-models"

# 5. Registry (depends on core, models)
publish_crate "neuro-divergent-registry" "neuro-divergent-registry"

# 6. Main crate (depends on all others)
publish_crate "neuro-divergent" "."

echo ""
echo -e "${GREEN}🎉 All neuro-divergent crates published successfully!${NC}"
echo ""
echo "📋 Published crates:"
echo "  • neuro-divergent-core v0.1.0"
echo "  • neuro-divergent-data v0.1.0"
echo "  • neuro-divergent-training v0.1.0"
echo "  • neuro-divergent-models v0.1.0"
echo "  • neuro-divergent-registry v0.1.0"
echo "  • neuro-divergent v0.1.0"
echo ""
echo "🔗 Available at: https://crates.io/search?q=neuro-divergent"
echo ""
echo "📚 Next steps:"
echo "  1. Update documentation with crates.io links"
echo "  2. Create GitHub release tags"
echo "  3. Announce to Rust community"
echo "  4. Update examples to use published crates"
echo "  5. Monitor downloads and feedback"