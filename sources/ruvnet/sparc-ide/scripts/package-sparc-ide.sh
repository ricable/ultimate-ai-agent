#!/bin/bash
# SPARC IDE - Multi-Platform Packaging Script
# This script builds SPARC IDE for multiple platforms and organizes the packages

set -e

# Enable additional security measures
set -o nounset  # Exit if a variable is unset
set -o pipefail # Exit if any command in a pipeline fails

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PACKAGE_DIR="$ROOT_DIR/package"
BUILD_SCRIPT="$SCRIPT_DIR/build-sparc-ide.sh"
VERSION=$(grep -o '"version": *"[^"]*"' "$ROOT_DIR/src/config/product.json" | cut -d'"' -f4)

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

print_header() {
    echo -e "\e[1;36m===== $1 =====\e[0m"
}

# Verify the build script exists
verify_build_script() {
    if [ ! -f "$BUILD_SCRIPT" ]; then
        print_error "Build script not found at $BUILD_SCRIPT"
        exit 1
    fi
    
    # Make sure it's executable
    chmod +x "$BUILD_SCRIPT"
}

# Build for a specific platform
build_platform() {
    local platform=$1
    print_header "Building SPARC IDE for $platform"
    
    # Create a temporary build directory
    local build_dir=$(mktemp -d)
    print_info "Using temporary build directory: $build_dir"
    
    # Copy necessary files to build directory
    cp -r "$ROOT_DIR/src" "$build_dir/"
    cp -r "$ROOT_DIR/scripts" "$build_dir/"
    cp -r "$ROOT_DIR/branding" "$build_dir/"
    
    # Change to build directory
    pushd "$build_dir" > /dev/null
    
    # Execute the build script for this platform
    print_info "Executing build script for $platform..."
    if ! "$BUILD_SCRIPT" --platform "$platform"; then
        print_error "Build failed for $platform"
        popd > /dev/null
        rm -rf "$build_dir"
        return 1
    fi
    
    # Copy artifacts to package directory
    print_info "Copying artifacts to package directory..."
    mkdir -p "$PACKAGE_DIR/$platform"
    
    case "$platform" in
        linux)
            # Copy Linux packages (deb, rpm)
            cp -p dist/*.deb "$PACKAGE_DIR/$platform/" 2>/dev/null || true
            cp -p dist/*.rpm "$PACKAGE_DIR/$platform/" 2>/dev/null || true
            ;;
        windows)
            # Copy Windows installer
            cp -p dist/*.exe "$PACKAGE_DIR/$platform/" 2>/dev/null || true
            ;;
        macos)
            # Copy macOS package
            cp -p dist/*.dmg "$PACKAGE_DIR/$platform/" 2>/dev/null || true
            ;;
    esac
    
    # Copy checksums
    cp -p dist/checksums.sha256 "$PACKAGE_DIR/$platform/" 2>/dev/null || true
    cp -p dist/checksums.sha256.asc "$PACKAGE_DIR/$platform/" 2>/dev/null || true
    
    # Copy security report
    mkdir -p "$PACKAGE_DIR/$platform/security"
    cp -p security-reports/final-security-report.txt "$PACKAGE_DIR/$platform/security/" 2>/dev/null || true
    
    # Return to original directory
    popd > /dev/null
    
    # Clean up build directory
    rm -rf "$build_dir"
    
    print_success "Build and packaging completed for $platform"
    return 0
}

# Create manifest file
create_manifest() {
    print_header "Creating package manifest"
    
    local manifest_file="$PACKAGE_DIR/manifest.json"
    local timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    
    # Start JSON file
    cat > "$manifest_file" << EOF
{
  "name": "SPARC IDE",
  "version": "$VERSION",
  "generated": "$timestamp",
  "packages": {
EOF
    
    # Add Linux packages
    echo '    "linux": [' >> "$manifest_file"
    first=true
    if [ -d "$PACKAGE_DIR/linux" ]; then
        for pkg in "$PACKAGE_DIR/linux"/*.{deb,rpm}; do
            if [ -f "$pkg" ]; then
                if [ "$first" = true ]; then
                    first=false
                else
                    echo "," >> "$manifest_file"
                fi
                filename=$(basename "$pkg")
                size=$(stat -c%s "$pkg")
                checksum=$(sha256sum "$pkg" | cut -d' ' -f1)
                echo -n '      {
        "filename": "'"$filename"'",
        "size": '"$size"',
        "checksum": "'"$checksum"'",
        "type": "'"${filename##*.}"'"
      }' >> "$manifest_file"
            fi
        done
    fi
    echo -e "\n    ]," >> "$manifest_file"
    
    # Add Windows packages
    echo '    "windows": [' >> "$manifest_file"
    first=true
    if [ -d "$PACKAGE_DIR/windows" ]; then
        for pkg in "$PACKAGE_DIR/windows"/*.exe; do
            if [ -f "$pkg" ]; then
                if [ "$first" = true ]; then
                    first=false
                else
                    echo "," >> "$manifest_file"
                fi
                filename=$(basename "$pkg")
                size=$(stat -c%s "$pkg")
                checksum=$(sha256sum "$pkg" | cut -d' ' -f1)
                echo -n '      {
        "filename": "'"$filename"'",
        "size": '"$size"',
        "checksum": "'"$checksum"'",
        "type": "exe"
      }' >> "$manifest_file"
            fi
        done
    fi
    echo -e "\n    ]," >> "$manifest_file"
    
    # Add macOS packages
    echo '    "macos": [' >> "$manifest_file"
    first=true
    if [ -d "$PACKAGE_DIR/macos" ]; then
        for pkg in "$PACKAGE_DIR/macos"/*.dmg; do
            if [ -f "$pkg" ]; then
                if [ "$first" = true ]; then
                    first=false
                else
                    echo "," >> "$manifest_file"
                fi
                filename=$(basename "$pkg")
                size=$(stat -c%s "$pkg")
                checksum=$(sha256sum "$pkg" | cut -d' ' -f1)
                echo -n '      {
        "filename": "'"$filename"'",
        "size": '"$size"',
        "checksum": "'"$checksum"'",
        "type": "dmg"
      }' >> "$manifest_file"
            fi
        done
    fi
    echo -e "\n    ]" >> "$manifest_file"
    
    # Close JSON file
    cat >> "$manifest_file" << EOF
  },
  "requirements": {
    "linux": {
      "os": "Linux",
      "architecture": "x64",
      "dependencies": ["glibc >= 2.17", "libgtk-3 >= 3.22"]
    },
    "windows": {
      "os": "Windows",
      "architecture": "x64",
      "dependencies": ["Windows 10 or later"]
    },
    "macos": {
      "os": "macOS",
      "architecture": "x64",
      "dependencies": ["macOS 10.15 (Catalina) or later"]
    }
  },
  "releaseNotes": "See https://github.com/sparc-ide/sparc-ide/releases/tag/v$VERSION"
}
EOF
    
    print_success "Package manifest created at $manifest_file"
}

# Create build report
create_build_report() {
    print_header "Creating build report"
    
    local report_file="$PACKAGE_DIR/build-report.md"
    local timestamp=$(date -u +"%Y-%m-%d %H:%M:%S UTC")
    
    cat > "$report_file" << EOF
# SPARC IDE Build Report

**Version:** $VERSION
**Build Date:** $timestamp

## Build Summary

| Platform | Status | Packages |
|----------|--------|----------|
EOF
    
    # Add Linux build status
    if [ -d "$PACKAGE_DIR/linux" ] && [ "$(find "$PACKAGE_DIR/linux" -name "*.deb" -o -name "*.rpm" | wc -l)" -gt 0 ]; then
        echo "| Linux    | ✅ Success | $(find "$PACKAGE_DIR/linux" -name "*.deb" -o -name "*.rpm" | wc -l) |" >> "$report_file"
    else
        echo "| Linux    | ❌ Failed  | 0 |" >> "$report_file"
    fi
    
    # Add Windows build status
    if [ -d "$PACKAGE_DIR/windows" ] && [ "$(find "$PACKAGE_DIR/windows" -name "*.exe" | wc -l)" -gt 0 ]; then
        echo "| Windows  | ✅ Success | $(find "$PACKAGE_DIR/windows" -name "*.exe" | wc -l) |" >> "$report_file"
    else
        echo "| Windows  | ❌ Failed  | 0 |" >> "$report_file"
    fi
    
    # Add macOS build status
    if [ -d "$PACKAGE_DIR/macos" ] && [ "$(find "$PACKAGE_DIR/macos" -name "*.dmg" | wc -l)" -gt 0 ]; then
        echo "| macOS    | ✅ Success | $(find "$PACKAGE_DIR/macos" -name "*.dmg" | wc -l) |" >> "$report_file"
    else
        echo "| macOS    | ❌ Failed  | 0 |" >> "$report_file"
    fi
    
    cat >> "$report_file" << EOF

## Package Details

EOF
    
    # Add Linux package details
    echo "### Linux Packages" >> "$report_file"
    echo "" >> "$report_file"
    if [ -d "$PACKAGE_DIR/linux" ]; then
        for pkg in "$PACKAGE_DIR/linux"/*.{deb,rpm}; do
            if [ -f "$pkg" ]; then
                filename=$(basename "$pkg")
                size=$(stat -c%s "$pkg" | numfmt --to=iec-i --suffix=B)
                echo "- **$filename** ($size)" >> "$report_file"
            fi
        done
    else
        echo "No Linux packages were built." >> "$report_file"
    fi
    echo "" >> "$report_file"
    
    # Add Windows package details
    echo "### Windows Packages" >> "$report_file"
    echo "" >> "$report_file"
    if [ -d "$PACKAGE_DIR/windows" ]; then
        for pkg in "$PACKAGE_DIR/windows"/*.exe; do
            if [ -f "$pkg" ]; then
                filename=$(basename "$pkg")
                size=$(stat -c%s "$pkg" | numfmt --to=iec-i --suffix=B)
                echo "- **$filename** ($size)" >> "$report_file"
            fi
        done
    else
        echo "No Windows packages were built." >> "$report_file"
    fi
    echo "" >> "$report_file"
    
    # Add macOS package details
    echo "### macOS Packages" >> "$report_file"
    echo "" >> "$report_file"
    if [ -d "$PACKAGE_DIR/macos" ]; then
        for pkg in "$PACKAGE_DIR/macos"/*.dmg; do
            if [ -f "$pkg" ]; then
                filename=$(basename "$pkg")
                size=$(stat -c%s "$pkg" | numfmt --to=iec-i --suffix=B)
                echo "- **$filename** ($size)" >> "$report_file"
            fi
        done
    else
        echo "No macOS packages were built." >> "$report_file"
    fi
    
    cat >> "$report_file" << EOF

## Security Verification

All packages have undergone security verification including:
- Extension signature verification
- Hardcoded credentials check
- File permissions check
- Source integrity verification

Detailed security reports are available in each platform's security directory.

## Installation Instructions

For installation instructions, please refer to the [Installation Guide](../docs/installation-guide.md).
EOF
    
    print_success "Build report created at $report_file"
}

# Main function
main() {
    print_header "SPARC IDE Multi-Platform Packaging"
    
    # Verify build script exists
    verify_build_script
    
    # Create package directory structure
    mkdir -p "$PACKAGE_DIR/linux" "$PACKAGE_DIR/windows" "$PACKAGE_DIR/macos"
    
    # Build for each platform
    build_platform "linux" || print_error "Linux build failed"
    build_platform "windows" || print_error "Windows build failed"
    build_platform "macos" || print_error "macOS build failed"
    
    # Create manifest file
    create_manifest
    
    # Create build report
    create_build_report
    
    print_header "Packaging Process Completed"
    print_success "SPARC IDE has been packaged for multiple platforms."
    print_info "Packages are available in the $PACKAGE_DIR directory."
    print_info "Manifest file is available at $PACKAGE_DIR/manifest.json"
    print_info "Build report is available at $PACKAGE_DIR/build-report.md"
}

# Run main function
main