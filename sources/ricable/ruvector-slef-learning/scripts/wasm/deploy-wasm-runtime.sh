#!/usr/bin/env bash
# Deploy WasmEdge and Spin runtimes on Kubernetes
# Sets up Runtime Class Manager, shims, and operators

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Check kubectl connectivity
check_cluster() {
    log_info "Checking Kubernetes cluster connectivity..."

    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi

    local nodes=$(kubectl get nodes --no-headers | wc -l)
    log_success "Connected to cluster with $nodes node(s)"
}

# Label nodes for Wasm runtime support
label_nodes() {
    log_info "Labeling nodes for Wasm runtime support..."

    # Get all Linux nodes
    local nodes=$(kubectl get nodes -o jsonpath='{.items[*].metadata.name}')

    for node in $nodes; do
        log_info "Labeling node: $node"

        # Label for WasmEdge
        kubectl label node "$node" wasm.runtime/wasmedge=true --overwrite || true

        # Label for Spin
        kubectl label node "$node" wasm.runtime/spin=true --overwrite || true

        # Label for multi-runtime
        kubectl label node "$node" wasm.runtime/multi=true --overwrite || true
    done

    log_success "Nodes labeled for Wasm runtimes"
}

# Deploy Runtime Class Manager
deploy_rcm() {
    log_info "Deploying Runtime Class Manager..."

    kubectl apply -f "$PROJECT_ROOT/infrastructure/runtime-class-manager/runtime-class-manager.yaml"

    # Wait for deployment
    log_info "Waiting for Runtime Class Manager to be ready..."
    kubectl -n rcm-system wait --for=condition=available deployment/runtime-class-manager --timeout=120s || {
        log_warn "RCM deployment timeout - checking status..."
        kubectl -n rcm-system get pods
    }

    log_success "Runtime Class Manager deployed"
}

# Deploy shim definitions
deploy_shims() {
    log_info "Deploying containerd shim definitions..."

    kubectl apply -f "$PROJECT_ROOT/infrastructure/runtime-class-manager/shims.yaml"

    # Check shim status
    log_info "Checking shim installation status..."
    sleep 5
    kubectl get shims -o wide || log_warn "Shims CRD not yet available"

    log_success "Shim definitions deployed"
}

# Deploy WasmEdge runtime configuration
deploy_wasmedge() {
    log_info "Deploying WasmEdge runtime configuration..."

    kubectl apply -f "$PROJECT_ROOT/infrastructure/wasmedge/wasmedge-runtime.yaml"

    # Verify RuntimeClass
    log_info "Verifying WasmEdge RuntimeClass..."
    kubectl get runtimeclass wasmedge -o wide || log_warn "WasmEdge RuntimeClass not found"

    log_success "WasmEdge runtime deployed"
}

# Deploy SpinKube operator
deploy_spinkube() {
    log_info "Deploying SpinKube operator..."

    kubectl apply -f "$PROJECT_ROOT/infrastructure/spinkube/spin-operator.yaml"

    # Wait for operator
    log_info "Waiting for Spin operator to be ready..."
    kubectl -n spinkube-system wait --for=condition=available deployment/spin-operator --timeout=120s || {
        log_warn "Spin operator deployment timeout - checking status..."
        kubectl -n spinkube-system get pods
    }

    # Verify RuntimeClass
    log_info "Verifying Spin RuntimeClass..."
    kubectl get runtimeclass wasmtime-spin -o wide || log_warn "Spin RuntimeClass not found"

    log_success "SpinKube operator deployed"
}

# Create agents namespace and secrets
setup_agents_namespace() {
    log_info "Setting up agents namespace..."

    # Create namespace
    kubectl create namespace agents --dry-run=client -o yaml | kubectl apply -f -

    # Create secrets from environment variables (if available)
    if [ -n "${ANTHROPIC_API_KEY:-}" ]; then
        kubectl -n agents create secret generic agent-secrets \
            --from-literal=ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY" \
            --from-literal=OPENAI_API_KEY="${OPENAI_API_KEY:-}" \
            --from-literal=LITELLM_MASTER_KEY="${LITELLM_MASTER_KEY:-}" \
            --dry-run=client -o yaml | kubectl apply -f -
        log_success "Agent secrets created"
    else
        log_warn "ANTHROPIC_API_KEY not set - create secrets manually:"
        echo "  kubectl -n agents create secret generic agent-secrets \\"
        echo "    --from-literal=ANTHROPIC_API_KEY=sk-xxx"
    fi

    log_success "Agents namespace configured"
}

# Deploy SpinApp examples
deploy_examples() {
    log_info "Deploying SpinApp examples..."

    kubectl apply -f "$PROJECT_ROOT/infrastructure/spinkube/spinapp-examples.yaml" || {
        log_warn "Some examples may have failed - check with: kubectl -n agents get spinapps"
    }

    log_success "SpinApp examples deployed"
}

# Verify installation
verify_installation() {
    log_info "Verifying installation..."

    echo ""
    echo "=== RuntimeClasses ==="
    kubectl get runtimeclass

    echo ""
    echo "=== Shims ==="
    kubectl get shims 2>/dev/null || echo "No shims CRD found"

    echo ""
    echo "=== Operator Pods ==="
    kubectl get pods -n rcm-system 2>/dev/null || echo "RCM not deployed"
    kubectl get pods -n spinkube-system 2>/dev/null || echo "SpinKube not deployed"
    kubectl get pods -n wasmedge-system 2>/dev/null || echo "WasmEdge ns not deployed"

    echo ""
    echo "=== SpinApps ==="
    kubectl get spinapps -n agents 2>/dev/null || echo "No SpinApps deployed"

    echo ""
    log_success "Installation verification complete"
}

# Cleanup function
cleanup() {
    log_warn "Cleaning up Wasm runtime resources..."

    kubectl delete -f "$PROJECT_ROOT/infrastructure/spinkube/spinapp-examples.yaml" --ignore-not-found
    kubectl delete -f "$PROJECT_ROOT/infrastructure/spinkube/spin-operator.yaml" --ignore-not-found
    kubectl delete -f "$PROJECT_ROOT/infrastructure/wasmedge/wasmedge-runtime.yaml" --ignore-not-found
    kubectl delete -f "$PROJECT_ROOT/infrastructure/runtime-class-manager/shims.yaml" --ignore-not-found
    kubectl delete -f "$PROJECT_ROOT/infrastructure/runtime-class-manager/runtime-class-manager.yaml" --ignore-not-found

    kubectl delete namespace agents --ignore-not-found

    log_success "Cleanup complete"
}

# Main
main() {
    local action="${1:-deploy}"

    case "$action" in
        deploy)
            log_info "Starting Wasm runtime deployment..."
            check_cluster
            label_nodes
            deploy_rcm
            deploy_shims
            deploy_wasmedge
            deploy_spinkube
            setup_agents_namespace
            verify_installation

            echo ""
            log_success "Wasm runtime deployment complete!"
            echo ""
            echo "Next steps:"
            echo "  1. Deploy SpinApp examples:"
            echo "     kubectl apply -f infrastructure/spinkube/spinapp-examples.yaml"
            echo ""
            echo "  2. Check agent status:"
            echo "     kubectl -n agents get spinapps"
            echo "     kubectl -n agents get pods"
            ;;

        examples)
            check_cluster
            deploy_examples
            ;;

        verify)
            check_cluster
            verify_installation
            ;;

        cleanup)
            check_cluster
            cleanup
            ;;

        *)
            echo "Usage: $0 {deploy|examples|verify|cleanup}"
            exit 1
            ;;
    esac
}

main "$@"
