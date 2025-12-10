#!/bin/bash

# Docker Hub Publishing Pipeline for Synaptic Neural Mesh
# Automates multi-architecture builds and publishing

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
REGISTRY="${REGISTRY:-docker.io/ruvnet}"
IMAGE_NAME="${IMAGE_NAME:-synaptic-mesh}"
VERSION="${VERSION:-$(node -p "require('$PROJECT_ROOT/package.json').version")}"
DRY_RUN="${DRY_RUN:-false}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() { echo -e "${GREEN}[$(date +'%H:%M:%S')] $1${NC}"; }
warn() { echo -e "${YELLOW}[$(date +'%H:%M:%S')] WARNING: $1${NC}"; }
error() { echo -e "${RED}[$(date +'%H:%M:%S')] ERROR: $1${NC}"; exit 1; }
info() { echo -e "${BLUE}[$(date +'%H:%M:%S')] $1${NC}"; }

# Validate environment
validate_environment() {
    log "Validating publishing environment..."
    
    if [ "$DRY_RUN" != "true" ]; then
        if [ -z "${DOCKER_HUB_TOKEN:-}" ]; then
            error "DOCKER_HUB_TOKEN environment variable is required for publishing"
        fi
        
        if [ -z "${DOCKER_HUB_USERNAME:-}" ]; then
            error "DOCKER_HUB_USERNAME environment variable is required for publishing"
        fi
        
        # Login to Docker Hub
        echo "$DOCKER_HUB_TOKEN" | docker login --username "$DOCKER_HUB_USERNAME" --password-stdin
        log "Successfully authenticated with Docker Hub"
    else
        log "Running in DRY_RUN mode - no actual publishing will occur"
    fi
    
    # Verify buildx
    if ! docker buildx version &> /dev/null; then
        error "Docker Buildx is required but not available"
    fi
    
    # Create/use builder
    if ! docker buildx ls | grep -q "synaptic-builder"; then
        docker buildx create --name synaptic-builder --use --platform linux/amd64,linux/arm64,linux/arm/v7
        log "Created multi-platform builder"
    else
        docker buildx use synaptic-builder
        log "Using existing multi-platform builder"
    fi
}

# Build and publish images
publish_images() {
    log "Building and publishing multi-architecture images..."
    
    local push_flag=""
    if [ "$DRY_RUN" != "true" ]; then
        push_flag="--push"
    else
        push_flag="--load"
        warn "DRY_RUN mode: Images will be built but not pushed"
    fi
    
    # Main production image
    info "Building main production image..."
    docker buildx build \
        --platform linux/amd64,linux/arm64,linux/arm/v7 \
        --tag "${REGISTRY}/${IMAGE_NAME}:${VERSION}" \
        --tag "${REGISTRY}/${IMAGE_NAME}:alpha" \
        --tag "${REGISTRY}/${IMAGE_NAME}:latest" \
        --file "$PROJECT_ROOT/Dockerfile" \
        $push_flag \
        "$PROJECT_ROOT" || error "Failed to build main image"
    
    # Alpine variant
    if [ -f "$PROJECT_ROOT/Dockerfile.alpine" ]; then
        info "Building Alpine variant..."
        docker buildx build \
            --platform linux/amd64,linux/arm64,linux/arm/v7 \
            --tag "${REGISTRY}/${IMAGE_NAME}:${VERSION}-alpine" \
            --tag "${REGISTRY}/${IMAGE_NAME}:alpha-alpine" \
            --file "$PROJECT_ROOT/Dockerfile.alpine" \
            $push_flag \
            "$PROJECT_ROOT" || warn "Failed to build Alpine variant"
    fi
    
    # Development variant
    if [ -f "$PROJECT_ROOT/Dockerfile.dev" ]; then
        info "Building development variant..."
        docker buildx build \
            --platform linux/amd64,linux/arm64 \
            --tag "${REGISTRY}/${IMAGE_NAME}:${VERSION}-dev" \
            --tag "${REGISTRY}/${IMAGE_NAME}:alpha-dev" \
            --file "$PROJECT_ROOT/Dockerfile.dev" \
            $push_flag \
            "$PROJECT_ROOT" || warn "Failed to build development variant"
    fi
    
    log "Image building completed successfully"
}

# Generate and upload metadata
upload_metadata() {
    if [ "$DRY_RUN" = "true" ]; then
        warn "DRY_RUN mode: Skipping metadata upload"
        return
    fi
    
    log "Generating and uploading image metadata..."
    
    local metadata_dir="$PROJECT_ROOT/docker-metadata"
    mkdir -p "$metadata_dir"
    
    # Generate image manifest
    docker manifest inspect "${REGISTRY}/${IMAGE_NAME}:${VERSION}" > "$metadata_dir/manifest.json" 2>/dev/null || warn "Failed to generate manifest"
    
    # Generate SBOM if syft is available
    if command -v syft &> /dev/null; then
        syft "${REGISTRY}/${IMAGE_NAME}:${VERSION}" -o json > "$metadata_dir/sbom.json" 2>/dev/null || warn "Failed to generate SBOM"
    fi
    
    # Security scan if grype is available
    if command -v grype &> /dev/null; then
        grype "${REGISTRY}/${IMAGE_NAME}:${VERSION}" -o json > "$metadata_dir/vulnerabilities.json" 2>/dev/null || warn "Failed to run security scan"
    fi
    
    # Create image description for Docker Hub
    cat > "$metadata_dir/description.md" << EOF
# Synaptic Neural Mesh

🧠 Self-evolving distributed neural fabric with quantum-resistant DAG networking

## Features

- **Distributed AI**: Self-organizing neural mesh architecture
- **Quantum-Resistant**: Advanced cryptographic security
- **P2P Networking**: Decentralized mesh topology
- **WASM Integration**: High-performance WebAssembly modules
- **Cross-Platform**: Supports Linux (x64, ARM64, ARMv7)

## Quick Start

\`\`\`bash
# Run with Docker
docker run -p 8080:8080 ${REGISTRY}/${IMAGE_NAME}:${VERSION}

# Or use NPX
npx synaptic-mesh@alpha init
\`\`\`

## Variants

- \`latest\`: Latest stable release
- \`alpha\`: Alpha release with latest features
- \`alpine\`: Lightweight Alpine-based image
- \`dev\`: Development image with debug tools

## Documentation

Visit [GitHub Repository](https://github.com/ruvnet/Synaptic-Neural-Mesh) for full documentation.

## Support

Report issues at: https://github.com/ruvnet/Synaptic-Neural-Mesh/issues
EOF
    
    log "Metadata generation completed"
}

# Test published images
test_published_images() {
    log "Testing published images..."
    
    local test_images=(
        "${REGISTRY}/${IMAGE_NAME}:${VERSION}"
        "${REGISTRY}/${IMAGE_NAME}:alpha"
    )
    
    for image in "${test_images[@]}"; do
        info "Testing $image..."
        
        if [ "$DRY_RUN" = "true" ]; then
            warn "DRY_RUN mode: Skipping image tests"
            continue
        fi
        
        # Test version command
        if docker run --rm "$image" --version > /dev/null 2>&1; then
            log "✅ $image: version check passed"
        else
            warn "❌ $image: version check failed"
        fi
        
        # Test help command
        if docker run --rm "$image" --help > /dev/null 2>&1; then
            log "✅ $image: help command passed"
        else
            warn "❌ $image: help command failed"
        fi
        
        # Test init dry-run
        if timeout 30 docker run --rm "$image" init --dry-run > /dev/null 2>&1; then
            log "✅ $image: init dry-run passed"
        else
            warn "❌ $image: init dry-run failed"
        fi
    done
    
    log "Image testing completed"
}

# Update release notes
update_release_notes() {
    log "Updating release notes..."
    
    local release_notes="$PROJECT_ROOT/RELEASE_NOTES.md"
    
    cat > "$release_notes" << EOF
# Synaptic Neural Mesh v${VERSION} Release Notes

## Docker Images

All images are available for multiple architectures (linux/amd64, linux/arm64, linux/arm/v7):

- \`${REGISTRY}/${IMAGE_NAME}:${VERSION}\` - Production release
- \`${REGISTRY}/${IMAGE_NAME}:alpha\` - Alpha release
- \`${REGISTRY}/${IMAGE_NAME}:latest\` - Latest stable
- \`${REGISTRY}/${IMAGE_NAME}:${VERSION}-alpine\` - Alpine variant
- \`${REGISTRY}/${IMAGE_NAME}:${VERSION}-dev\` - Development variant

## NPM Package

\`\`\`bash
# Install globally
npm install -g synaptic-mesh@alpha

# Or use with NPX
npx synaptic-mesh@alpha init
\`\`\`

## Kubernetes Deployment

\`\`\`bash
# Deploy to Kubernetes
kubectl apply -f k8s/production/
\`\`\`

## Release Information

- **Version**: ${VERSION}
- **Build Date**: $(date -u +"%Y-%m-%dT%H:%M:%SZ")
- **Node.js**: $(node --version)
- **Platforms**: linux/amd64, linux/arm64, linux/arm/v7
- **Architecture**: Multi-platform distribution ready

## Verification

Verify installation:

\`\`\`bash
# Check version
synaptic-mesh --version

# Test functionality
synaptic-mesh init --dry-run
\`\`\`

## Support

- Repository: https://github.com/ruvnet/Synaptic-Neural-Mesh
- Issues: https://github.com/ruvnet/Synaptic-Neural-Mesh/issues
- Documentation: https://github.com/ruvnet/Synaptic-Neural-Mesh/docs
EOF
    
    log "Release notes updated: $release_notes"
}

# Main execution
main() {
    log "🧠 Synaptic Neural Mesh - Docker Publishing Pipeline"
    log "=================================================="
    log "Registry: ${REGISTRY}"
    log "Image: ${IMAGE_NAME}"
    log "Version: ${VERSION}"
    log "Dry Run: ${DRY_RUN}"
    log ""
    
    cd "$PROJECT_ROOT"
    
    validate_environment
    publish_images
    upload_metadata
    test_published_images
    update_release_notes
    
    if [ "$DRY_RUN" = "true" ]; then
        log "🎯 DRY_RUN completed successfully - no images were published"
    else
        log "🎉 Publishing completed successfully!"
        log "Images are now available at: ${REGISTRY}/${IMAGE_NAME}"
    fi
}

# Execute main function
main "$@"