#!/bin/bash

# Docker Multi-Architecture Build Script for Synaptic Neural Mesh
# Builds and publishes containers for multiple architectures

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
IMAGE_NAME="${IMAGE_NAME:-synaptic-mesh}"
REGISTRY="${REGISTRY:-docker.io/ruvnet}"
VERSION="${VERSION:-$(node -p "require('$PROJECT_ROOT/package.json').version")}"
PLATFORMS="${PLATFORMS:-linux/amd64,linux/arm64,linux/arm/v7}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
    exit 1
}

info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

# Check dependencies
check_dependencies() {
    log "Checking dependencies..."
    
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed"
    fi
    
    if ! docker buildx version &> /dev/null; then
        error "Docker Buildx is not available"
    fi
    
    # Check if we can use multi-platform builds
    if ! docker buildx ls | grep -q "docker-container"; then
        log "Creating new buildx builder for multi-platform builds..."
        docker buildx create --name synaptic-builder --use --platform "$PLATFORMS"
    fi
    
    log "Dependencies check passed"
}

# Build single architecture image
build_single() {
    local platform="$1"
    local tag_suffix="$2"
    
    log "Building $platform image..."
    
    docker buildx build \
        --platform "$platform" \
        --tag "${REGISTRY}/${IMAGE_NAME}:${VERSION}${tag_suffix}" \
        --tag "${REGISTRY}/${IMAGE_NAME}:alpha${tag_suffix}" \
        --file "$PROJECT_ROOT/Dockerfile" \
        --load \
        "$PROJECT_ROOT"
        
    log "Built $platform image: ${REGISTRY}/${IMAGE_NAME}:${VERSION}${tag_suffix}"
}

# Build multi-architecture images
build_multi() {
    log "Building multi-architecture images for platforms: $PLATFORMS"
    
    # Build and push main image
    docker buildx build \
        --platform "$PLATFORMS" \
        --tag "${REGISTRY}/${IMAGE_NAME}:${VERSION}" \
        --tag "${REGISTRY}/${IMAGE_NAME}:alpha" \
        --tag "${REGISTRY}/${IMAGE_NAME}:latest" \
        --file "$PROJECT_ROOT/Dockerfile" \
        --push \
        "$PROJECT_ROOT"
    
    # Build Alpine variant
    if [ -f "$PROJECT_ROOT/Dockerfile.alpine" ]; then
        docker buildx build \
            --platform "$PLATFORMS" \
            --tag "${REGISTRY}/${IMAGE_NAME}:${VERSION}-alpine" \
            --tag "${REGISTRY}/${IMAGE_NAME}:alpha-alpine" \
            --file "$PROJECT_ROOT/Dockerfile.alpine" \
            --push \
            "$PROJECT_ROOT"
    fi
    
    # Build development variant
    if [ -f "$PROJECT_ROOT/Dockerfile.dev" ]; then
        docker buildx build \
            --platform "$PLATFORMS" \
            --tag "${REGISTRY}/${IMAGE_NAME}:${VERSION}-dev" \
            --tag "${REGISTRY}/${IMAGE_NAME}:alpha-dev" \
            --file "$PROJECT_ROOT/Dockerfile.dev" \
            --push \
            "$PROJECT_ROOT"
    fi
    
    log "Multi-architecture build completed"
}

# Test images
test_images() {
    log "Testing built images..."
    
    local images=(
        "${REGISTRY}/${IMAGE_NAME}:${VERSION}"
        "${REGISTRY}/${IMAGE_NAME}:alpha"
    )
    
    for image in "${images[@]}"; do
        info "Testing $image..."
        
        # Test basic functionality
        if docker run --rm "$image" --version; then
            log "✅ $image: version check passed"
        else
            warn "❌ $image: version check failed"
        fi
        
        # Test help command
        if docker run --rm "$image" --help > /dev/null; then
            log "✅ $image: help command passed"
        else
            warn "❌ $image: help command failed"
        fi
        
        # Test init command (dry run)
        if docker run --rm "$image" init --dry-run > /dev/null; then
            log "✅ $image: init dry-run passed"
        else
            warn "❌ $image: init dry-run failed"
        fi
    done
    
    log "Image testing completed"
}

# Generate manifest and security scan
generate_manifest() {
    log "Generating image manifest and security information..."
    
    local output_dir="$PROJECT_ROOT/docker-manifests"
    mkdir -p "$output_dir"
    
    # Generate manifest for main image
    docker manifest inspect "${REGISTRY}/${IMAGE_NAME}:${VERSION}" > "$output_dir/manifest-${VERSION}.json" || warn "Failed to generate manifest"
    
    # Generate SBOM if available
    if command -v syft &> /dev/null; then
        syft "${REGISTRY}/${IMAGE_NAME}:${VERSION}" -o json > "$output_dir/sbom-${VERSION}.json" || warn "Failed to generate SBOM"
    fi
    
    # Security scan if available
    if command -v grype &> /dev/null; then
        grype "${REGISTRY}/${IMAGE_NAME}:${VERSION}" -o json > "$output_dir/security-scan-${VERSION}.json" || warn "Failed to run security scan"
    fi
    
    log "Manifest generation completed"
}

# Main execution
main() {
    log "🧠 Synaptic Neural Mesh - Docker Multi-Architecture Build"
    log "========================================================="
    log "Image: ${REGISTRY}/${IMAGE_NAME}"
    log "Version: ${VERSION}"
    log "Platforms: ${PLATFORMS}"
    log ""
    
    cd "$PROJECT_ROOT"
    
    check_dependencies
    
    case "${1:-multi}" in
        "single")
            build_single "linux/amd64" ""
            ;;
        "multi")
            build_multi
            ;;
        "test")
            test_images
            ;;
        "manifest")
            generate_manifest
            ;;
        "all")
            build_multi
            test_images
            generate_manifest
            ;;
        *)
            error "Usage: $0 [single|multi|test|manifest|all]"
            ;;
    esac
    
    log "🎉 Build process completed successfully!"
}

# Execute main function with all arguments
main "$@"