#!/usr/bin/env bash
# Setup mise shims for non-interactive environments
# Use for Docker, CI/CD, and Kubernetes deployments
#
# Usage:
#   ./scripts/mise/setup-shims.sh
#
# This script:
# - Installs mise if not present
# - Configures shims directory
# - Installs tools from mise.toml
# - Sets up PATH for shim usage

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

# =============================================================================
# INSTALL MISE
# =============================================================================
install_mise() {
    if command -v mise &> /dev/null; then
        log_info "mise already installed: $(mise --version)"
        return 0
    fi

    log_info "Installing mise..."

    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        if command -v brew &> /dev/null; then
            brew install mise
        else
            curl https://mise.run | sh
        fi
    else
        # Linux
        curl https://mise.run | sh
    fi

    # Add to PATH for current script
    export PATH="$HOME/.local/bin:$PATH"

    if command -v mise &> /dev/null; then
        log_success "mise installed: $(mise --version)"
    else
        log_error "Failed to install mise"
        exit 1
    fi
}

# =============================================================================
# SETUP SHIMS
# =============================================================================
setup_shims() {
    log_info "Setting up mise shims..."

    SHIMS_DIR="${MISE_SHIMS_DIR:-$HOME/.local/share/mise/shims}"

    # Create shims directory
    mkdir -p "$SHIMS_DIR"

    # Trust mise.toml files
    if [ -f "$PROJECT_ROOT/mise.toml" ]; then
        mise trust "$PROJECT_ROOT/mise.toml"
    fi

    # Trust agent-specific configs
    for config in "$PROJECT_ROOT"/agents/*/mise.toml; do
        if [ -f "$config" ]; then
            mise trust "$config"
        fi
    done

    # Install all tools
    log_info "Installing tools from mise.toml..."
    cd "$PROJECT_ROOT"
    mise install

    # Regenerate shims
    log_info "Regenerating shims..."
    mise reshim

    log_success "Shims directory: $SHIMS_DIR"
}

# =============================================================================
# CONFIGURE SHELL
# =============================================================================
configure_shell() {
    log_info "Configuring shell for shims..."

    SHIMS_DIR="${MISE_SHIMS_DIR:-$HOME/.local/share/mise/shims}"
    SHELL_CONFIG=""

    # Detect shell
    if [ -n "$ZSH_VERSION" ] || [ "$SHELL" = "/bin/zsh" ]; then
        SHELL_CONFIG="$HOME/.zshrc"
    elif [ -n "$BASH_VERSION" ] || [ "$SHELL" = "/bin/bash" ]; then
        SHELL_CONFIG="$HOME/.bashrc"
    fi

    if [ -z "$SHELL_CONFIG" ]; then
        log_warn "Could not detect shell config file"
        return 0
    fi

    # Add shims to PATH if not already present
    if ! grep -q "mise/shims" "$SHELL_CONFIG" 2>/dev/null; then
        log_info "Adding shims to $SHELL_CONFIG"
        cat >> "$SHELL_CONFIG" << EOF

# mise shims for non-interactive environments
export PATH="\$HOME/.local/share/mise/shims:\$PATH"
EOF
        log_success "Shell configured"
    else
        log_info "Shell already configured"
    fi
}

# =============================================================================
# VERIFY SHIMS
# =============================================================================
verify_shims() {
    log_info "Verifying shims..."

    SHIMS_DIR="${MISE_SHIMS_DIR:-$HOME/.local/share/mise/shims}"
    export PATH="$SHIMS_DIR:$PATH"

    echo ""
    echo "Tool Versions:"
    echo "=============="

    # Node.js
    if command -v node &> /dev/null; then
        echo "  Node.js: $(node --version)"
    else
        log_warn "node not found in shims"
    fi

    # Bun
    if command -v bun &> /dev/null; then
        echo "  Bun: $(bun --version)"
    else
        log_warn "bun not found in shims"
    fi

    # Python
    if command -v python &> /dev/null; then
        echo "  Python: $(python --version 2>&1)"
    else
        log_warn "python not found in shims"
    fi

    # Rust
    if command -v rustc &> /dev/null; then
        echo "  Rust: $(rustc --version)"
    else
        log_warn "rust not found in shims"
    fi

    # kubectl
    if command -v kubectl &> /dev/null; then
        echo "  kubectl: $(kubectl version --client --short 2>/dev/null || echo 'installed')"
    else
        log_warn "kubectl not found in shims"
    fi

    echo ""
    log_success "Shims verification complete"
}

# =============================================================================
# DOCKER SETUP
# =============================================================================
setup_docker_shims() {
    log_info "Generating Docker shims setup script..."

    cat > "$PROJECT_ROOT/config/docker/shims-entrypoint.sh" << 'EOF'
#!/bin/sh
# Docker entrypoint that activates mise shims
export PATH="$HOME/.local/share/mise/shims:$PATH"
exec "$@"
EOF

    chmod +x "$PROJECT_ROOT/config/docker/shims-entrypoint.sh"
    log_success "Docker shims entrypoint created"
}

# =============================================================================
# KUBERNETES SETUP
# =============================================================================
setup_k8s_shims() {
    log_info "Creating Kubernetes ConfigMap for mise shims..."

    mkdir -p "$PROJECT_ROOT/config/k8s"

    cat > "$PROJECT_ROOT/config/k8s/mise-shims-configmap.yaml" << 'EOF'
# ConfigMap containing mise shim setup for Kubernetes pods
apiVersion: v1
kind: ConfigMap
metadata:
  name: mise-shims-setup
  namespace: zgents
data:
  setup-shims.sh: |
    #!/bin/sh
    # Setup mise shims in Kubernetes pod

    # Install mise if not present
    if ! command -v mise &> /dev/null; then
      curl https://mise.run | sh
    fi

    # Add to PATH
    export PATH="$HOME/.local/bin:$HOME/.local/share/mise/shims:$PATH"

    # Install tools and generate shims
    mise install
    mise reshim

    echo "Mise shims configured"

  activate-shims.sh: |
    #!/bin/sh
    # Source this in container startup
    export PATH="$HOME/.local/share/mise/shims:$PATH"
EOF

    log_success "Kubernetes ConfigMap created"
}

# =============================================================================
# MAIN
# =============================================================================
main() {
    echo ""
    echo "╔═══════════════════════════════════════════╗"
    echo "║     mise Shims Setup for zgents           ║"
    echo "╚═══════════════════════════════════════════╝"
    echo ""

    install_mise
    setup_shims
    configure_shell
    setup_docker_shims
    setup_k8s_shims
    verify_shims

    echo ""
    log_success "mise shims setup complete!"
    echo ""
    echo "To activate shims in current session:"
    echo "  export PATH=\"\$HOME/.local/share/mise/shims:\$PATH\""
    echo ""
    echo "Or start a new shell session."
}

main "$@"
