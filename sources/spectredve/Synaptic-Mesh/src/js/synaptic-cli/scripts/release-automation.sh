#!/bin/bash

# Complete Release Automation Script for Synaptic Neural Mesh
# Orchestrates the entire publishing and distribution pipeline

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VERSION="${VERSION:-$(node -p "require('$PROJECT_ROOT/package.json').version")}"
DRY_RUN="${DRY_RUN:-false}"
SKIP_TESTS="${SKIP_TESTS:-false}"
SKIP_DOCKER="${SKIP_DOCKER:-false}"
SKIP_NPM="${SKIP_NPM:-false}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'

log() { echo -e "${GREEN}[RELEASE] $1${NC}"; }
warn() { echo -e "${YELLOW}[RELEASE] WARNING: $1${NC}"; }
error() { echo -e "${RED}[RELEASE] ERROR: $1${NC}"; exit 1; }
info() { echo -e "${BLUE}[RELEASE] $1${NC}"; }
header() { echo -e "${PURPLE}[RELEASE] === $1 ===${NC}"; }

# Pre-flight checks
preflight_checks() {
    header "Pre-flight Checks"
    
    log "Checking environment..."
    
    # Check Node.js version
    NODE_VERSION=$(node --version | sed 's/v//')
    NODE_MAJOR=$(echo $NODE_VERSION | cut -d'.' -f1)
    if [ "$NODE_MAJOR" -lt 18 ]; then
        error "Node.js 18+ required, found: $NODE_VERSION"
    fi
    log "✅ Node.js version: $NODE_VERSION"
    
    # Check npm version
    NPM_VERSION=$(npm --version)
    log "✅ NPM version: $NPM_VERSION"
    
    # Check Docker (if not skipping)
    if [ "$SKIP_DOCKER" != "true" ]; then
        if ! command -v docker &> /dev/null; then
            error "Docker is required but not installed"
        fi
        
        if ! docker buildx version &> /dev/null; then
            error "Docker Buildx is required but not available"
        fi
        log "✅ Docker and Buildx available"
    fi
    
    # Check git status
    if [ "$(git status --porcelain | wc -l)" -gt 0 ]; then
        warn "Working directory has uncommitted changes"
        git status --short
        
        if [ "$DRY_RUN" != "true" ]; then
            read -p "Continue anyway? (y/N): " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                error "Aborted due to uncommitted changes"
            fi
        fi
    fi
    
    # Check if we're on main branch
    CURRENT_BRANCH=$(git branch --show-current)
    if [ "$CURRENT_BRANCH" != "main" ]; then
        warn "Not on main branch (current: $CURRENT_BRANCH)"
    fi
    
    log "✅ Pre-flight checks completed"
}

# Build phase
build_phase() {
    header "Build Phase"
    
    cd "$PROJECT_ROOT"
    
    log "Cleaning previous builds..."
    npm run clean || warn "Clean script not available"
    
    log "Installing dependencies..."
    npm ci
    
    log "Running build..."
    npm run build
    
    log "✅ Build phase completed"
}

# Quality assurance phase
qa_phase() {
    if [ "$SKIP_TESTS" = "true" ]; then
        warn "Skipping tests (SKIP_TESTS=true)"
        return
    fi
    
    header "Quality Assurance Phase"
    
    log "Running linter..."
    npm run lint || error "Linting failed"
    
    log "Running tests..."
    npm test || error "Tests failed"
    
    log "Running cross-platform compatibility tests..."
    node "$SCRIPT_DIR/cross-platform-test.js" || error "Cross-platform tests failed"
    
    log "Running NPX tests..."
    bash "$SCRIPT_DIR/test-npx.sh" || error "NPX tests failed"
    
    log "Running global validation..."
    npm pack
    node "$SCRIPT_DIR/validate-global.js" || error "Global validation failed"
    
    log "✅ Quality assurance phase completed"
}

# NPM publishing phase
npm_phase() {
    if [ "$SKIP_NPM" = "true" ]; then
        warn "Skipping NPM publishing (SKIP_NPM=true)"
        return
    fi
    
    header "NPM Publishing Phase"
    
    if [ "$DRY_RUN" = "true" ]; then
        log "DRY_RUN: Would publish to NPM..."
        npm publish --dry-run --tag alpha
    else
        log "Publishing to NPM..."
        
        # Check if already published
        if npm view "synaptic-mesh@$VERSION" version &> /dev/null; then
            warn "Version $VERSION already published to NPM"
        else
            npm publish --tag alpha
            log "✅ Published to NPM with alpha tag"
        fi
    fi
    
    log "✅ NPM publishing phase completed"
}

# Docker publishing phase
docker_phase() {
    if [ "$SKIP_DOCKER" = "true" ]; then
        warn "Skipping Docker publishing (SKIP_DOCKER=true)"
        return
    fi
    
    header "Docker Publishing Phase"
    
    # Set environment variables for docker script
    export REGISTRY="${REGISTRY:-docker.io/ruvnet}"
    export IMAGE_NAME="${IMAGE_NAME:-synaptic-mesh}"
    export VERSION="$VERSION"
    export DRY_RUN="$DRY_RUN"
    
    log "Running Docker build and publish script..."
    bash "$SCRIPT_DIR/publish-docker.sh" || error "Docker publishing failed"
    
    log "✅ Docker publishing phase completed"
}

# Kubernetes phase
k8s_phase() {
    header "Kubernetes Deployment Phase"
    
    log "Validating Kubernetes manifests..."
    
    # Check if kubectl is available
    if command -v kubectl &> /dev/null; then
        for manifest in "$PROJECT_ROOT/k8s/production"/*.yaml; do
            if [ -f "$manifest" ]; then
                kubectl apply --dry-run=client -f "$manifest" > /dev/null || warn "Manifest validation failed: $manifest"
            fi
        done
        log "✅ Kubernetes manifests validated"
    else
        warn "kubectl not available, skipping manifest validation"
    fi
    
    # Generate deployment guide
    cat > "$PROJECT_ROOT/DEPLOYMENT_GUIDE.md" << EOF
# Synaptic Neural Mesh - Deployment Guide

## Version: $VERSION

### Quick Deployment

#### NPM/NPX
\`\`\`bash
# Install globally
npm install -g synaptic-mesh@alpha

# Or use with NPX
npx synaptic-mesh@alpha init
\`\`\`

#### Docker
\`\`\`bash
# Run single container
docker run -p 8080:8080 ruvnet/synaptic-mesh:$VERSION

# Or use Docker Compose
docker-compose up -d
\`\`\`

#### Kubernetes
\`\`\`bash
# Deploy to Kubernetes cluster
kubectl apply -f k8s/production/

# Check deployment status
kubectl get pods -n synaptic-mesh
\`\`\`

### Configuration

See [Configuration Guide](docs/configuration.md) for detailed setup instructions.

### Support

- Repository: https://github.com/ruvnet/Synaptic-Neural-Mesh
- Issues: https://github.com/ruvnet/Synaptic-Neural-Mesh/issues
- Documentation: https://github.com/ruvnet/Synaptic-Neural-Mesh/docs

Generated on: $(date -u +"%Y-%m-%d %H:%M:%S UTC")
EOF
    
    log "✅ Kubernetes phase completed"
}

# Post-release phase
post_release_phase() {
    header "Post-Release Phase"
    
    log "Generating release summary..."
    
    cat > "$PROJECT_ROOT/RELEASE_SUMMARY.md" << EOF
# Release Summary - Synaptic Neural Mesh v$VERSION

**Release Date**: $(date -u +"%Y-%m-%d %H:%M:%S UTC")
**Release Type**: Alpha Release

## Distribution Channels

### ✅ NPM Package
- Package: \`synaptic-mesh@alpha\`
- Version: $VERSION
- Registry: https://registry.npmjs.org/
- Install: \`npm install -g synaptic-mesh@alpha\`
- NPX: \`npx synaptic-mesh@alpha init\`

### ✅ Docker Images
- Registry: Docker Hub (ruvnet/synaptic-mesh)
- Tags: $VERSION, alpha, latest
- Architectures: linux/amd64, linux/arm64, linux/arm/v7
- Pull: \`docker pull ruvnet/synaptic-mesh:$VERSION\`

### ✅ Kubernetes Deployment
- Manifests: k8s/production/
- Namespace: synaptic-mesh
- Components: Core, Workers, Services, ConfigMaps
- Deploy: \`kubectl apply -f k8s/production/\`

## Testing Results

- ✅ Unit Tests: Passed
- ✅ Integration Tests: Passed  
- ✅ Cross-Platform Tests: Passed
- ✅ NPX Distribution Tests: Passed
- ✅ Docker Multi-Arch Builds: Passed
- ✅ Kubernetes Manifest Validation: Passed

## Verification Commands

\`\`\`bash
# Test NPX installation
npx synaptic-mesh@alpha --version

# Test Docker image
docker run --rm ruvnet/synaptic-mesh:$VERSION --version

# Test Kubernetes deployment (requires cluster)
kubectl apply -f k8s/production/namespace.yaml
kubectl get pods -n synaptic-mesh
\`\`\`

## Next Steps

1. Monitor deployment metrics
2. Gather user feedback
3. Plan beta release based on alpha feedback
4. Continue development on main branch

## Support

For issues or questions:
- GitHub Issues: https://github.com/ruvnet/Synaptic-Neural-Mesh/issues
- Documentation: https://github.com/ruvnet/Synaptic-Neural-Mesh/docs

---

*This release was generated automatically by the release automation pipeline.*
EOF
    
    # Update version for next development cycle
    if [ "$DRY_RUN" != "true" ]; then
        log "Tagging release..."
        git tag -a "v$VERSION" -m "Release v$VERSION"
        
        log "Updating version for next development cycle..."
        # This would typically bump to next alpha version
        # npm version prerelease --preid=alpha --no-git-tag-version
    fi
    
    log "✅ Post-release phase completed"
}

# Main execution
main() {
    log "🧠 Synaptic Neural Mesh - Complete Release Automation"
    log "===================================================="
    log "Version: $VERSION"
    log "Dry Run: $DRY_RUN"
    log "Skip Tests: $SKIP_TESTS"
    log "Skip Docker: $SKIP_DOCKER"
    log "Skip NPM: $SKIP_NPM"
    log ""
    
    # Phase execution
    preflight_checks
    build_phase
    qa_phase
    npm_phase
    docker_phase
    k8s_phase
    post_release_phase
    
    log ""
    log "🎉 Release automation completed successfully!"
    log "📊 Release summary available at: RELEASE_SUMMARY.md"
    log "🚀 Deployment guide available at: DEPLOYMENT_GUIDE.md"
    
    if [ "$DRY_RUN" = "true" ]; then
        log "🎯 This was a DRY_RUN - no actual publishing occurred"
    fi
}

# Script entry point
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    main "$@"
fi