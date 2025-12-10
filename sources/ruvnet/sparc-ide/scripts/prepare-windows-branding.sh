#!/bin/bash
# SPARC IDE - Windows Branding Preparation Script
# This script downloads and prepares Windows branding assets

set -e

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

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BRANDING_DIR="$ROOT_DIR/branding/windows"
TEMP_DIR="$ROOT_DIR/temp/windows-branding"

# Create directories
mkdir -p "$BRANDING_DIR"
mkdir -p "$TEMP_DIR"

# Download sample icons and images
print_info "Downloading sample Windows branding assets..."

# Sample URLs for icons and images (replace with actual URLs in production)
SAMPLE_ICON_URL="https://github.com/VSCodium/vscodium/raw/master/src/stable/resources/win32/code.ico"
SAMPLE_INSTALLER_ICON_URL="https://github.com/VSCodium/vscodium/raw/master/src/stable/resources/win32/code.ico"
SAMPLE_BANNER_URL="https://github.com/VSCodium/vscodium/raw/master/src/stable/resources/win32/inno-big-100.bmp"
SAMPLE_DIALOG_URL="https://github.com/VSCodium/vscodium/raw/master/src/stable/resources/win32/inno-small-100.bmp"

# Download files
curl -L "$SAMPLE_ICON_URL" -o "$TEMP_DIR/sample-icon.ico"
curl -L "$SAMPLE_INSTALLER_ICON_URL" -o "$TEMP_DIR/sample-installer-icon.ico"
curl -L "$SAMPLE_BANNER_URL" -o "$TEMP_DIR/sample-banner.bmp"
curl -L "$SAMPLE_DIALOG_URL" -o "$TEMP_DIR/sample-dialog.bmp"

# Copy files to branding directory
cp "$TEMP_DIR/sample-icon.ico" "$BRANDING_DIR/sparc-ide.ico"
cp "$TEMP_DIR/sample-installer-icon.ico" "$BRANDING_DIR/sparc-ide-installer.ico"
cp "$TEMP_DIR/sample-banner.bmp" "$BRANDING_DIR/sparc-ide-installer-banner.bmp"
cp "$TEMP_DIR/sample-dialog.bmp" "$BRANDING_DIR/sparc-ide-installer-dialog.bmp"

# Create additional icons for file associations
cp "$TEMP_DIR/sample-icon.ico" "$BRANDING_DIR/markdown.ico"
cp "$TEMP_DIR/sample-icon.ico" "$BRANDING_DIR/json.ico"
cp "$TEMP_DIR/sample-icon.ico" "$BRANDING_DIR/sparc-ide-uninstaller.ico"
cp "$TEMP_DIR/sample-icon.ico" "$BRANDING_DIR/docs.ico"

# Clean up
rm -rf "$TEMP_DIR"

print_success "Windows branding assets prepared successfully."
print_info "The following files have been created:"
ls -la "$BRANDING_DIR"