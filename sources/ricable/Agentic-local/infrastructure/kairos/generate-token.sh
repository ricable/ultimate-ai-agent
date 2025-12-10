#!/bin/bash
# =============================================================================
# Kairos P2P Network Token Generator
# Generates secure network tokens for P2P mesh cluster formation
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/secrets"

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

# =============================================================================
# GENERATE NETWORK TOKEN
# =============================================================================

generate_network_token() {
    log_info "Generating Kairos P2P network token..."

    mkdir -p "$OUTPUT_DIR"
    chmod 700 "$OUTPUT_DIR"

    # Check if kairos CLI is available
    if command -v kairos &> /dev/null; then
        kairos generate-token > "${OUTPUT_DIR}/network-token.txt"
        log_success "Token generated using Kairos CLI"
    else
        # Fallback: Generate a base64 encoded random token
        # Format compatible with EdgeVPN
        log_warn "Kairos CLI not found, generating compatible token..."

        NETWORK_ID=$(openssl rand -hex 16)
        NETWORK_KEY=$(openssl rand -hex 32)

        cat > "${OUTPUT_DIR}/network-token.txt" << EOF
{
  "network_id": "${NETWORK_ID}",
  "network_key": "${NETWORK_KEY}",
  "created_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "cluster_name": "edge-ai-cluster",
  "version": "1.0"
}
EOF

        # Base64 encode for use in cloud-config
        base64 -w0 "${OUTPUT_DIR}/network-token.txt" > "${OUTPUT_DIR}/network-token.b64"
    fi

    log_success "Network token saved to ${OUTPUT_DIR}/network-token.txt"
}

# =============================================================================
# GENERATE K3S TOKEN
# =============================================================================

generate_k3s_token() {
    log_info "Generating K3s cluster token..."

    K3S_TOKEN=$(openssl rand -hex 32)
    echo "$K3S_TOKEN" > "${OUTPUT_DIR}/k3s-token.txt"
    chmod 600 "${OUTPUT_DIR}/k3s-token.txt"

    log_success "K3s token saved to ${OUTPUT_DIR}/k3s-token.txt"
}

# =============================================================================
# GENERATE ENV FILE
# =============================================================================

generate_env_file() {
    log_info "Generating environment configuration..."

    NETWORK_TOKEN=$(cat "${OUTPUT_DIR}/network-token.b64" 2>/dev/null || cat "${OUTPUT_DIR}/network-token.txt")
    K3S_TOKEN=$(cat "${OUTPUT_DIR}/k3s-token.txt")

    cat > "${OUTPUT_DIR}/cluster.env" << EOF
# =============================================================================
# Edge-Native AI Cluster Environment Configuration
# Generated: $(date -u +%Y-%m-%dT%H:%M:%SZ)
# =============================================================================

# P2P Mesh Network Configuration
KAIROS_NETWORK_TOKEN="${NETWORK_TOKEN}"

# K3s Cluster Configuration
K3S_TOKEN="${K3S_TOKEN}"
K3S_SERVER_URL="https://control-plane.edge-ai.local:6443"

# Cluster Identification
CLUSTER_NAME="edge-ai-cluster"
CLUSTER_DOMAIN="edge-ai.local"

# Node Configuration (set per-node)
# NODE_ROLE="control-plane|worker"
# NODE_ARCH="amd64|arm64"

# AI Gateway Configuration
LITELLM_MASTER_KEY="$(openssl rand -hex 24)"
LITELLM_DATABASE_URL="postgresql://litellm:$(openssl rand -hex 16)@postgres:5432/litellm"

# AgentDB Configuration
AGENTDB_ENCRYPTION_KEY="$(openssl rand -hex 32)"
REDIS_URL="redis://:$(openssl rand -hex 16)@redis:6379"

# E2B Configuration (optional cloud sandbox)
E2B_API_KEY=""
EOF

    chmod 600 "${OUTPUT_DIR}/cluster.env"
    log_success "Environment file saved to ${OUTPUT_DIR}/cluster.env"
}

# =============================================================================
# MAIN
# =============================================================================

main() {
    echo ""
    echo "=============================================="
    echo "  Edge-Native AI - Cluster Token Generator"
    echo "=============================================="
    echo ""

    generate_network_token
    generate_k3s_token
    generate_env_file

    echo ""
    log_success "All tokens generated successfully!"
    echo ""
    echo "Next steps:"
    echo "  1. Copy ${OUTPUT_DIR}/cluster.env to each node"
    echo "  2. Source the env file: source ${OUTPUT_DIR}/cluster.env"
    echo "  3. Flash Kairos ISO with cloud-config.yaml"
    echo "  4. Boot nodes - they will auto-join the mesh"
    echo ""
    log_warn "Keep these tokens secure - they provide cluster access!"
}

main "$@"
