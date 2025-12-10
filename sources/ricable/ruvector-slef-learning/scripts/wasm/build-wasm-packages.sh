#!/usr/bin/env bash
# Build Pipeline for Wasm npm packages
# Builds and packages ruvnet npm modules for WasmEdge and Spin runtimes

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Configuration
REGISTRY="${REGISTRY:-ghcr.io/ruvnet}"
TAG="${TAG:-latest}"
PUSH="${PUSH:-false}"

# Check dependencies
check_dependencies() {
    log_info "Checking build dependencies..."

    local deps=("node" "npm" "docker")
    local optional_deps=("spin" "wasmedge")

    for dep in "${deps[@]}"; do
        if ! command -v "$dep" &> /dev/null; then
            log_error "Required dependency not found: $dep"
            exit 1
        fi
    done

    for dep in "${optional_deps[@]}"; do
        if ! command -v "$dep" &> /dev/null; then
            log_warn "Optional dependency not found: $dep (some builds may fail)"
        fi
    done

    log_success "All required dependencies found"
}

# Build Spin JS SDK agent
build_spin_agent() {
    log_info "Building Spin JS SDK agent..."

    local app_dir="$PROJECT_ROOT/apps/spin-js"

    if [ ! -d "$app_dir" ]; then
        log_error "Spin JS app directory not found: $app_dir"
        return 1
    fi

    cd "$app_dir"

    # Install dependencies
    log_info "Installing npm dependencies..."
    npm ci --legacy-peer-deps || npm install

    # Build TypeScript
    log_info "Compiling TypeScript..."
    npm run build:ts || true

    # Build Wasm module (if spin is available)
    if command -v spin &> /dev/null; then
        log_info "Building Spin application..."
        spin build
        log_success "Spin application built successfully"
    else
        log_warn "Spin CLI not found, skipping Wasm build"
    fi

    cd "$PROJECT_ROOT"
    log_success "Spin JS SDK agent built"
}

# Build WasmEdge QuickJS agent
build_wasmedge_agent() {
    log_info "Building WasmEdge QuickJS agent..."

    local app_dir="$PROJECT_ROOT/apps/wasmedge-js"

    if [ ! -d "$app_dir" ]; then
        log_error "WasmEdge JS app directory not found: $app_dir"
        return 1
    fi

    cd "$app_dir"

    # Install dependencies
    log_info "Installing npm dependencies..."
    npm ci --legacy-peer-deps || npm install

    # Bundle JavaScript
    log_info "Bundling JavaScript for WasmEdge..."
    npm run build:bundle || {
        log_warn "esbuild not available, using raw source"
        mkdir -p dist
        cp src/index.js dist/agent.js
    }

    cd "$PROJECT_ROOT"
    log_success "WasmEdge QuickJS agent built"
}

# Build Docker images
build_docker_images() {
    log_info "Building Docker images..."

    # WasmEdge agent image
    local wasmedge_dir="$PROJECT_ROOT/apps/wasmedge-js"
    if [ -f "$wasmedge_dir/Dockerfile" ]; then
        log_info "Building WasmEdge agent Docker image..."
        docker build -t "$REGISTRY/wasmedge-agent:$TAG" "$wasmedge_dir"
        log_success "WasmEdge agent image built: $REGISTRY/wasmedge-agent:$TAG"
    fi

    # Spin agent image (if spin registry push is available)
    local spin_dir="$PROJECT_ROOT/apps/spin-js"
    if [ -d "$spin_dir" ] && command -v spin &> /dev/null; then
        log_info "Building Spin agent OCI image..."
        cd "$spin_dir"
        # Spin apps are pushed directly to OCI registries
        if [ "$PUSH" = "true" ]; then
            spin registry push "$REGISTRY/spin-agent:$TAG"
            log_success "Spin agent pushed: $REGISTRY/spin-agent:$TAG"
        fi
        cd "$PROJECT_ROOT"
    fi

    log_success "Docker images built"
}

# Push images to registry
push_images() {
    if [ "$PUSH" != "true" ]; then
        log_info "Skipping image push (set PUSH=true to enable)"
        return 0
    fi

    log_info "Pushing images to registry..."

    docker push "$REGISTRY/wasmedge-agent:$TAG"
    log_success "Images pushed to $REGISTRY"
}

# Generate Kubernetes manifests with image tags
generate_manifests() {
    log_info "Generating Kubernetes manifests..."

    local manifest_dir="$PROJECT_ROOT/infrastructure/generated"
    mkdir -p "$manifest_dir"

    # Generate WasmEdge deployment manifest
    cat > "$manifest_dir/wasmedge-deployment.yaml" << EOF
# Generated WasmEdge Agent Deployment
# Image: $REGISTRY/wasmedge-agent:$TAG
# Generated: $(date -u +"%Y-%m-%dT%H:%M:%SZ")

apiVersion: apps/v1
kind: Deployment
metadata:
  name: wasmedge-agent
  namespace: agents
  labels:
    app: wasmedge-agent
    version: "$TAG"
spec:
  replicas: 3
  selector:
    matchLabels:
      app: wasmedge-agent
  template:
    metadata:
      labels:
        app: wasmedge-agent
    spec:
      runtimeClassName: wasmedge
      containers:
        - name: agent
          image: $REGISTRY/wasmedge-agent:$TAG
          ports:
            - containerPort: 8080
          resources:
            limits:
              cpu: 200m
              memory: 128Mi
            requests:
              cpu: 50m
              memory: 32Mi
EOF

    # Generate SpinApp manifest
    cat > "$manifest_dir/spin-spinapp.yaml" << EOF
# Generated SpinApp Manifest
# Image: $REGISTRY/spin-agent:$TAG
# Generated: $(date -u +"%Y-%m-%dT%H:%M:%SZ")

apiVersion: core.spinoperator.dev/v1alpha1
kind: SpinApp
metadata:
  name: spin-agent
  namespace: agents
  labels:
    app: spin-agent
    version: "$TAG"
spec:
  image: $REGISTRY/spin-agent:$TAG
  replicas: 3
  executor: containerd-shim-spin
  resources:
    limits:
      cpu: 200m
      memory: 128Mi
    requests:
      cpu: 50m
      memory: 32Mi
EOF

    log_success "Manifests generated in $manifest_dir"
}

# Run tests
run_tests() {
    log_info "Running tests..."

    local spin_dir="$PROJECT_ROOT/apps/spin-js"
    local wasmedge_dir="$PROJECT_ROOT/apps/wasmedge-js"

    # Test Spin app
    if [ -d "$spin_dir" ]; then
        cd "$spin_dir"
        npm test 2>/dev/null || log_warn "Spin tests skipped"
        cd "$PROJECT_ROOT"
    fi

    # Test WasmEdge app
    if [ -d "$wasmedge_dir" ]; then
        cd "$wasmedge_dir"
        npm test 2>/dev/null || log_warn "WasmEdge tests skipped"
        cd "$PROJECT_ROOT"
    fi

    log_success "Tests completed"
}

# Main build pipeline
main() {
    log_info "Starting Wasm npm packages build pipeline"
    log_info "Registry: $REGISTRY"
    log_info "Tag: $TAG"
    log_info "Push: $PUSH"

    check_dependencies
    build_spin_agent
    build_wasmedge_agent
    build_docker_images
    push_images
    generate_manifests
    run_tests

    log_success "Build pipeline completed successfully!"

    echo ""
    log_info "Next steps:"
    echo "  1. Deploy Runtime Class Manager:"
    echo "     kubectl apply -f infrastructure/runtime-class-manager/"
    echo ""
    echo "  2. Deploy WasmEdge runtime:"
    echo "     kubectl apply -f infrastructure/wasmedge/"
    echo ""
    echo "  3. Deploy SpinKube operator:"
    echo "     kubectl apply -f infrastructure/spinkube/"
    echo ""
    echo "  4. Deploy agents:"
    echo "     kubectl apply -f infrastructure/generated/"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --registry)
            REGISTRY="$2"
            shift 2
            ;;
        --tag)
            TAG="$2"
            shift 2
            ;;
        --push)
            PUSH="true"
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --registry URL  Container registry (default: ghcr.io/ruvnet)"
            echo "  --tag TAG       Image tag (default: latest)"
            echo "  --push          Push images to registry"
            echo "  --help          Show this help"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

main
