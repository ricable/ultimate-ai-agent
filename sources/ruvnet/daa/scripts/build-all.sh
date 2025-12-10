#!/usr/bin/env bash

# build-all.sh - Build NAPI bindings for all supported platforms locally
# This script helps test multi-platform builds before pushing to CI

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
NAPI_DIR="$PROJECT_ROOT/qudag/qudag-napi"
BUILD_DIR="$NAPI_DIR/build-artifacts"

# Build options
TARGETS=()
NODE_VERSIONS=("18" "20" "22")
RELEASE_MODE=false
CLEAN_BUILD=false
RUN_TESTS=false
PARALLEL_BUILDS=1

# Print colored output
print_info() {
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

# Print usage information
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Build NAPI bindings for multiple platforms locally.

OPTIONS:
    -t, --target TARGET       Specify target platform (can be used multiple times)
                             Available: linux-x64, linux-x64-musl, linux-arm64,
                                       macos-x64, macos-arm64, windows-x64, all
    -n, --node VERSION       Node.js version to test (default: 20)
    -r, --release            Build in release mode with optimizations
    -c, --clean              Clean build artifacts before building
    --test                   Run tests after building
    -j, --parallel N         Run N builds in parallel (default: 1)
    -h, --help               Show this help message

EXAMPLES:
    # Build for current platform in debug mode
    $0

    # Build for Linux x64 in release mode
    $0 --target linux-x64 --release

    # Build for all platforms and run tests
    $0 --target all --release --test

    # Build for macOS with specific Node.js version
    $0 --target macos-arm64 --node 20 --release

    # Clean build for multiple targets in parallel
    $0 -t linux-x64 -t macos-x64 --clean --parallel 2

EOF
    exit 1
}

# Detect current platform
detect_platform() {
    local os=$(uname -s)
    local arch=$(uname -m)

    case "$os" in
        Linux)
            case "$arch" in
                x86_64) echo "linux-x64" ;;
                aarch64|arm64) echo "linux-arm64" ;;
                *) echo "unknown" ;;
            esac
            ;;
        Darwin)
            case "$arch" in
                x86_64) echo "macos-x64" ;;
                arm64) echo "macos-arm64" ;;
                *) echo "unknown" ;;
            esac
            ;;
        MINGW*|MSYS*|CYGWIN*)
            echo "windows-x64"
            ;;
        *)
            echo "unknown"
            ;;
    esac
}

# Map platform name to Rust target triple
get_rust_target() {
    case "$1" in
        linux-x64) echo "x86_64-unknown-linux-gnu" ;;
        linux-x64-musl) echo "x86_64-unknown-linux-musl" ;;
        linux-arm64) echo "aarch64-unknown-linux-gnu" ;;
        linux-arm64-musl) echo "aarch64-unknown-linux-musl" ;;
        macos-x64) echo "x86_64-apple-darwin" ;;
        macos-arm64) echo "aarch64-apple-darwin" ;;
        windows-x64) echo "x86_64-pc-windows-msvc" ;;
        *) echo "" ;;
    esac
}

# Check if target is supported on current platform
is_target_supported() {
    local target=$1
    local current_platform=$(detect_platform)

    # Native builds are always supported
    if [[ "$target" == "$current_platform" ]]; then
        return 0
    fi

    # Cross-compilation support checks
    case "$current_platform" in
        linux-*)
            # Linux can cross-compile to musl and ARM with appropriate tools
            [[ "$target" =~ ^linux-.* ]] && return 0
            ;;
        macos-*)
            # macOS can cross-compile between x64 and arm64
            [[ "$target" =~ ^macos-.* ]] && return 0
            ;;
    esac

    return 1
}

# Install Rust target
install_rust_target() {
    local rust_target=$1

    print_info "Installing Rust target: $rust_target"

    if rustup target list | grep -q "$rust_target (installed)"; then
        print_success "Target $rust_target already installed"
    else
        rustup target add "$rust_target" || {
            print_error "Failed to install Rust target: $rust_target"
            return 1
        }
        print_success "Installed Rust target: $rust_target"
    fi
}

# Build for specific target
build_target() {
    local target=$1
    local rust_target=$(get_rust_target "$target")

    if [[ -z "$rust_target" ]]; then
        print_error "Unknown target: $target"
        return 1
    fi

    print_info "Building for target: $target ($rust_target)"

    # Check if target is supported
    if ! is_target_supported "$target"; then
        print_warning "Cross-compilation for $target may not be supported on this platform"
        print_warning "Attempting build anyway..."
    fi

    # Install Rust target
    install_rust_target "$rust_target" || return 1

    # Navigate to NAPI directory
    cd "$NAPI_DIR"

    # Build command
    local build_cmd="npm run build"
    if [[ "$RELEASE_MODE" == "true" ]]; then
        build_cmd="$build_cmd -- --release"
    fi
    build_cmd="$build_cmd -- --target $rust_target"

    print_info "Running: $build_cmd"

    # Execute build
    if eval "$build_cmd"; then
        print_success "Build completed for $target"

        # Create artifacts directory
        mkdir -p "$BUILD_DIR/$target"

        # Copy built artifacts
        if [[ -f "qudag_native.$rust_target.node" ]]; then
            cp "qudag_native.$rust_target.node" "$BUILD_DIR/$target/"
            print_success "Artifact saved to: $BUILD_DIR/$target/qudag_native.$rust_target.node"
        fi

        return 0
    else
        print_error "Build failed for $target"
        return 1
    fi
}

# Run tests for built artifacts
run_tests() {
    local target=$1
    local current_platform=$(detect_platform)

    # Only run tests on native platform
    if [[ "$target" != "$current_platform" ]]; then
        print_warning "Skipping tests for $target (cross-compiled)"
        return 0
    fi

    print_info "Running tests for $target"

    cd "$NAPI_DIR"

    if npm test; then
        print_success "Tests passed for $target"
        return 0
    else
        print_error "Tests failed for $target"
        return 1
    fi
}

# Clean build artifacts
clean_build() {
    print_info "Cleaning build artifacts..."

    cd "$NAPI_DIR"

    # Remove Rust build artifacts
    if [[ -d "target" ]]; then
        rm -rf target
        print_success "Removed Rust build artifacts"
    fi

    # Remove NAPI build artifacts
    rm -f *.node

    # Remove build artifacts directory
    if [[ -d "$BUILD_DIR" ]]; then
        rm -rf "$BUILD_DIR"
        print_success "Removed build artifacts directory"
    fi

    print_success "Clean completed"
}

# Main build process
main() {
    print_info "QuDAG NAPI Multi-Platform Build Script"
    print_info "========================================"

    # Check if we're in the right directory
    if [[ ! -f "$NAPI_DIR/Cargo.toml" ]]; then
        print_error "Cannot find NAPI package at: $NAPI_DIR"
        exit 1
    fi

    # Install dependencies
    print_info "Installing npm dependencies..."
    cd "$NAPI_DIR"
    npm ci || {
        print_error "Failed to install npm dependencies"
        exit 1
    }

    # Clean if requested
    if [[ "$CLEAN_BUILD" == "true" ]]; then
        clean_build
    fi

    # If no targets specified, use current platform
    if [[ ${#TARGETS[@]} -eq 0 ]]; then
        CURRENT_PLATFORM=$(detect_platform)
        if [[ "$CURRENT_PLATFORM" == "unknown" ]]; then
            print_error "Cannot detect current platform"
            exit 1
        fi
        TARGETS=("$CURRENT_PLATFORM")
        print_info "No targets specified, using current platform: $CURRENT_PLATFORM"
    fi

    # Build each target
    local failed_builds=()
    for target in "${TARGETS[@]}"; do
        print_info "----------------------------------------"
        if build_target "$target"; then
            if [[ "$RUN_TESTS" == "true" ]]; then
                run_tests "$target" || failed_builds+=("$target")
            fi
        else
            failed_builds+=("$target")
        fi
    done

    # Summary
    print_info "========================================"
    print_info "Build Summary"
    print_info "========================================"

    if [[ ${#failed_builds[@]} -eq 0 ]]; then
        print_success "All builds completed successfully!"
        print_info "Artifacts saved to: $BUILD_DIR"
        exit 0
    else
        print_error "Failed builds: ${failed_builds[*]}"
        exit 1
    fi
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--target)
            if [[ "$2" == "all" ]]; then
                TARGETS=("linux-x64" "linux-x64-musl" "linux-arm64" "macos-x64" "macos-arm64" "windows-x64")
            else
                TARGETS+=("$2")
            fi
            shift 2
            ;;
        -n|--node)
            NODE_VERSION="$2"
            shift 2
            ;;
        -r|--release)
            RELEASE_MODE=true
            shift
            ;;
        -c|--clean)
            CLEAN_BUILD=true
            shift
            ;;
        --test)
            RUN_TESTS=true
            shift
            ;;
        -j|--parallel)
            PARALLEL_BUILDS="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            print_error "Unknown option: $1"
            usage
            ;;
    esac
done

# Run main build process
main
