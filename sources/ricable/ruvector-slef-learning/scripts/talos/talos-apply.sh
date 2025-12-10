#!/usr/bin/env bash
# Talos Cluster Deployment Script
# Usage: ./scripts/talos/talos-apply.sh [action] [options]
#
# Actions:
#   genconfig   - Generate Talos configuration
#   apply       - Apply configuration to node
#   bootstrap   - Bootstrap the cluster
#   kubeconfig  - Get kubeconfig
#   status      - Check cluster status
#   upgrade     - Upgrade Talos on nodes

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
CLUSTER_NAME="${CLUSTER_NAME:-ruvector-cluster}"
TALOS_VERSION="${TALOS_VERSION:-v1.8.0}"
K8S_VERSION="${K8S_VERSION:-v1.31.0}"
OUTPUT_DIR="${OUTPUT_DIR:-infrastructure/talos/generated}"
CONTROL_PLANE_ENDPOINT="${CONTROL_PLANE_ENDPOINT:-https://192.168.1.100:6443}"

log() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

check_deps() {
    local deps=("talosctl")
    for dep in "${deps[@]}"; do
        if ! command -v "$dep" &> /dev/null; then
            error "$dep is required but not installed"
        fi
    done
}

genconfig() {
    log "Generating Talos configuration for cluster: $CLUSTER_NAME"

    mkdir -p "$OUTPUT_DIR"

    talosctl gen config "$CLUSTER_NAME" "$CONTROL_PLANE_ENDPOINT" \
        --output-dir "$OUTPUT_DIR" \
        --with-docs=false \
        --with-examples=false \
        --kubernetes-version "$K8S_VERSION" \
        --talos-version "$TALOS_VERSION"

    log "Applying Wasm runtime patch..."
    if [ -f "infrastructure/talos/patches/wasm-runtime-patch.yaml" ]; then
        talosctl machineconfig patch "$OUTPUT_DIR/controlplane.yaml" \
            --patch @infrastructure/talos/patches/wasm-runtime-patch.yaml \
            --output "$OUTPUT_DIR/controlplane-wasm.yaml"

        talosctl machineconfig patch "$OUTPUT_DIR/worker.yaml" \
            --patch @infrastructure/talos/patches/wasm-runtime-patch.yaml \
            --output "$OUTPUT_DIR/worker-wasm.yaml"
    fi

    log "Configuration generated in $OUTPUT_DIR"
    echo ""
    echo "Files created:"
    ls -la "$OUTPUT_DIR"
}

apply_config() {
    local NODE=$1
    local CONFIG=${2:-"$OUTPUT_DIR/controlplane-wasm.yaml"}

    if [ -z "$NODE" ]; then
        error "Usage: $0 apply <node-ip> [config-file]"
    fi

    log "Applying Talos config to $NODE..."
    talosctl apply-config --insecure --nodes "$NODE" --file "$CONFIG"
    log "Configuration applied to $NODE"
}

bootstrap_cluster() {
    local NODE=$1

    if [ -z "$NODE" ]; then
        error "Usage: $0 bootstrap <control-plane-ip>"
    fi

    log "Bootstrapping Talos cluster on $NODE..."
    talosctl bootstrap --nodes "$NODE" --endpoints "$NODE"
    log "Cluster bootstrapped"
}

get_kubeconfig() {
    local NODE=$1

    if [ -z "$NODE" ]; then
        error "Usage: $0 kubeconfig <control-plane-ip>"
    fi

    log "Fetching kubeconfig from $NODE..."
    talosctl kubeconfig --nodes "$NODE" --endpoints "$NODE"
    log "Kubeconfig saved to ~/.kube/config"
}

cluster_status() {
    log "Talos Cluster Status"
    echo ""

    echo -e "${BLUE}Machine Status:${NC}"
    talosctl get machinestatus 2>/dev/null || warn "Configure talosctl endpoint first"

    echo ""
    echo -e "${BLUE}Cluster Members:${NC}"
    talosctl get members 2>/dev/null || true

    echo ""
    echo -e "${BLUE}Services:${NC}"
    talosctl services 2>/dev/null || true
}

upgrade_node() {
    local NODE=$1
    local IMAGE=${2:-"ghcr.io/siderolabs/installer:$TALOS_VERSION"}

    if [ -z "$NODE" ]; then
        error "Usage: $0 upgrade <node-ip> [installer-image]"
    fi

    log "Upgrading Talos on $NODE to $IMAGE..."
    talosctl upgrade --nodes "$NODE" --image "$IMAGE"
    log "Upgrade initiated on $NODE"
}

show_help() {
    echo "Talos Cluster Deployment Script"
    echo ""
    echo "Usage: $0 <action> [options]"
    echo ""
    echo "Actions:"
    echo "  genconfig               Generate Talos configuration"
    echo "  apply <ip> [config]     Apply configuration to node"
    echo "  bootstrap <ip>          Bootstrap the cluster"
    echo "  kubeconfig <ip>         Get kubeconfig"
    echo "  status                  Check cluster status"
    echo "  upgrade <ip> [image]    Upgrade Talos on node"
    echo ""
    echo "Environment variables:"
    echo "  CLUSTER_NAME            Cluster name (default: ruvector-cluster)"
    echo "  TALOS_VERSION           Talos version (default: v1.8.0)"
    echo "  K8S_VERSION             Kubernetes version (default: v1.31.0)"
    echo "  OUTPUT_DIR              Output directory (default: infrastructure/talos/generated)"
    echo "  CONTROL_PLANE_ENDPOINT  Control plane endpoint"
}

main() {
    check_deps

    case "${1:-help}" in
        genconfig)
            genconfig
            ;;
        apply)
            apply_config "${2:-}" "${3:-}"
            ;;
        bootstrap)
            bootstrap_cluster "${2:-}"
            ;;
        kubeconfig)
            get_kubeconfig "${2:-}"
            ;;
        status)
            cluster_status
            ;;
        upgrade)
            upgrade_node "${2:-}" "${3:-}"
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            error "Unknown action: $1. Use --help for usage."
            ;;
    esac
}

main "$@"
