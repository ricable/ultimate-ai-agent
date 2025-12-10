#!/usr/bin/env bash
# Install all tools defined in mise.toml configurations
# Use this for initial setup or CI/CD environments
#
# Usage:
#   ./scripts/mise/install-tools.sh [category]
#
# Categories: all, wasm, python, rust, infra

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

CATEGORY="${1:-all}"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

log() { echo -e "${BLUE}[mise]${NC} $1"; }
success() { echo -e "${GREEN}[✓]${NC} $1"; }

cd "$PROJECT_ROOT"

# Ensure mise is available
if ! command -v mise &> /dev/null; then
    log "Installing mise..."
    curl https://mise.run | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# Trust all configurations
log "Trusting mise configurations..."
mise trust mise.toml 2>/dev/null || true

case $CATEGORY in
    all)
        log "Installing ALL tools..."
        mise install

        # Install category-specific tools
        for config in agents/*/mise.toml; do
            if [ -f "$config" ]; then
                mise trust "$config" 2>/dev/null || true
            fi
        done
        mise install
        success "All tools installed"
        ;;

    wasm)
        log "Installing WASM tools..."
        mise trust agents/wasm/mise.toml 2>/dev/null || true
        mise use -g rust@stable
        mise use -g "cargo:wasm-pack"
        mise use -g "cargo:cargo-component"
        mise use -g "cargo:wasm-bindgen-cli"
        mise use -g node@22
        mise use -g bun@latest

        # Install WasmEdge if not present
        if ! command -v wasmedge &> /dev/null; then
            log "Installing WasmEdge..."
            curl -sSf https://raw.githubusercontent.com/WasmEdge/WasmEdge/master/utils/install.sh | bash
        fi

        # Install Spin if not present
        if ! command -v spin &> /dev/null; then
            log "Installing Spin..."
            curl -fsSL https://developer.fermyon.com/downloads/install.sh | bash
            sudo mv spin /usr/local/bin/ 2>/dev/null || mv spin ~/.local/bin/
        fi

        success "WASM tools installed"
        ;;

    python)
        log "Installing Python tools..."
        mise trust agents/python/mise.toml 2>/dev/null || true
        mise use -g python@3.12
        mise use -g uv@latest
        mise use -g "pipx:ruff"
        mise use -g "pipx:pytest"
        mise use -g "pipx:mypy"
        success "Python tools installed"
        ;;

    rust)
        log "Installing Rust tools..."
        mise trust agents/rust/mise.toml 2>/dev/null || true
        mise use -g rust@stable

        # Add WASM targets
        rustup target add wasm32-unknown-unknown
        rustup target add wasm32-wasi
        rustup target add wasm32-wasip1

        # Install cargo tools
        mise use -g "cargo:cargo-watch"
        mise use -g "cargo:cargo-edit"
        mise use -g "cargo:cargo-audit"
        mise use -g "cargo:wasm-pack"
        mise use -g "cargo:cargo-component"
        mise use -g "cargo:cross"
        success "Rust tools installed"
        ;;

    infra)
        log "Installing Infrastructure tools..."
        mise trust agents/infra/mise.toml 2>/dev/null || true
        mise use -g kubectl@latest
        mise use -g helm@latest
        mise use -g k9s@latest
        mise use -g kustomize@latest
        mise use -g terraform@latest
        success "Infrastructure tools installed"
        ;;

    *)
        echo "Unknown category: $CATEGORY"
        echo "Usage: $0 [all|wasm|python|rust|infra]"
        exit 1
        ;;
esac

# Regenerate shims
log "Regenerating shims..."
mise reshim

success "Tool installation complete!"
echo ""
echo "Installed tools:"
mise list
