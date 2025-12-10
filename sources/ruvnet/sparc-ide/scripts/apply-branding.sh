#!/bin/bash
# SPARC IDE - Branding Modification Script
# This script applies SPARC IDE branding to the VSCodium build

set -e

# Configuration
VSCODIUM_DIR="vscodium"
BRANDING_DIR="branding"
PRODUCT_JSON="src/config/product.json"

# Print colored output
print_info() {
    echo -e "\e[1;34m[INFO]\e[0m $1"
}

print_success() {
    echo -e "\e[1;32m[SUCCESS]\e[0m $1"
}

print_error() {
    echo -e "\e[1;31m[ERROR]\e[0m $1"
}

# Check if VSCodium directory exists
check_vscodium() {
    print_info "Checking VSCodium directory..."
    
    if [ ! -d "$VSCODIUM_DIR" ]; then
        print_error "VSCodium directory not found. Please run setup-build-environment.sh first."
        exit 1
    fi
    
    print_success "VSCodium directory found."
}

# Apply product.json customizations
apply_product_json() {
    print_info "Applying product.json customizations..."
    
    if [ ! -f "$PRODUCT_JSON" ]; then
        print_error "Product JSON file not found at $PRODUCT_JSON"
        exit 1
    fi
    
    cp "$PRODUCT_JSON" "$VSCODIUM_DIR/product.json"
    
    print_success "Product JSON customizations applied."
}

# Apply icon customizations
apply_icons() {
    print_info "Applying icon customizations..."
    
    ICONS_DIR="$BRANDING_DIR/icons"
    TARGET_ICONS_DIR="$VSCODIUM_DIR/src/vs/workbench/browser/parts/editor/media"
    
    if [ ! -d "$ICONS_DIR" ]; then
        print_error "Icons directory not found at $ICONS_DIR"
        exit 1
    fi
    
    mkdir -p "$TARGET_ICONS_DIR"
    cp -r "$ICONS_DIR/"* "$TARGET_ICONS_DIR/"
    
    print_success "Icon customizations applied."
}

# Apply splash screen customizations
apply_splash() {
    print_info "Applying splash screen customizations..."
    
    SPLASH_DIR="$BRANDING_DIR/splash"
    TARGET_SPLASH_DIR="$VSCODIUM_DIR/src/vs/workbench/browser/parts/splash"
    
    if [ ! -d "$SPLASH_DIR" ]; then
        print_error "Splash screen directory not found at $SPLASH_DIR"
        exit 1
    fi
    
    mkdir -p "$TARGET_SPLASH_DIR"
    cp -r "$SPLASH_DIR/"* "$TARGET_SPLASH_DIR/"
    
    print_success "Splash screen customizations applied."
}

# Apply platform-specific branding
apply_platform_branding() {
    print_info "Applying platform-specific branding..."
    
    # Linux branding
    if [ -d "$BRANDING_DIR/linux" ]; then
        print_info "Applying Linux branding..."
        mkdir -p "$VSCODIUM_DIR/build/linux"
        cp -r "$BRANDING_DIR/linux/"* "$VSCODIUM_DIR/build/linux/"
    fi
    
    # Windows branding
    if [ -d "$BRANDING_DIR/windows" ]; then
        print_info "Applying Windows branding..."
        mkdir -p "$VSCODIUM_DIR/build/windows"
        cp -r "$BRANDING_DIR/windows/"* "$VSCODIUM_DIR/build/windows/"
    fi
    
    # macOS branding
    if [ -d "$BRANDING_DIR/macos" ]; then
        print_info "Applying macOS branding..."
        mkdir -p "$VSCODIUM_DIR/build/darwin"
        cp -r "$BRANDING_DIR/macos/"* "$VSCODIUM_DIR/build/darwin/"
    fi
    
    print_success "Platform-specific branding applied."
}

# Main function
main() {
    print_info "Applying SPARC IDE branding..."
    
    check_vscodium
    apply_product_json
    apply_icons
    apply_splash
    apply_platform_branding
    
    print_success "SPARC IDE branding applied successfully."
}

# Run main function
main