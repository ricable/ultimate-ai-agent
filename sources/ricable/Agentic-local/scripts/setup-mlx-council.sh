#!/bin/bash
#
# MLX Deep Council Setup Script
#
# This script sets up the MLX Deep Council system on Mac machines.
# It installs required dependencies and configures the distributed environment.
#
# Usage:
#   ./setup-mlx-council.sh [options]
#
# Options:
#   --local         Setup for local development (single Mac)
#   --distributed   Setup for distributed cluster
#   --hosts <list>  Comma-separated list of hostnames for distributed setup
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running on macOS
check_macos() {
    if [[ "$(uname)" != "Darwin" ]]; then
        log_error "This script requires macOS with Apple Silicon"
        exit 1
    fi

    # Check for Apple Silicon
    if [[ "$(uname -m)" != "arm64" ]]; then
        log_error "Apple Silicon (M1/M2/M3/M4) is required"
        exit 1
    fi

    log_info "Running on macOS $(sw_vers -productVersion) with Apple Silicon"
}

# Install Python dependencies
install_python_deps() {
    log_info "Installing Python dependencies..."

    # Check for Python
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is required. Install with: brew install python3"
        exit 1
    fi

    # Install MLX and related packages
    pip3 install --upgrade pip
    pip3 install mlx mlx-lm fastapi uvicorn httpx

    log_info "Python dependencies installed"
}

# Download default models
download_models() {
    log_info "Downloading default MLX models..."

    # Create models directory
    mkdir -p ~/.mlx-council/models

    # Download recommended models
    python3 -c "
from mlx_lm import load
import sys

models = [
    'mlx-community/Llama-3.2-3B-Instruct-4bit',
    # Add more models as needed
]

for model in models:
    print(f'Downloading {model}...')
    try:
        load(model)
        print(f'  OK: {model}')
    except Exception as e:
        print(f'  FAILED: {e}', file=sys.stderr)
"

    log_info "Models downloaded"
}

# Setup SSH for distributed mode
setup_ssh() {
    local hosts="$1"

    log_info "Setting up SSH for distributed council..."

    # Generate SSH key if needed
    if [ ! -f ~/.ssh/id_ed25519 ]; then
        log_info "Generating SSH key..."
        ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -N ""
    fi

    # Copy key to remote hosts
    IFS=',' read -ra HOSTS <<< "$hosts"
    for host in "${HOSTS[@]}"; do
        log_info "Setting up SSH for $host..."
        ssh-copy-id -i ~/.ssh/id_ed25519.pub "$host" 2>/dev/null || \
            log_warn "Could not copy SSH key to $host (may already be configured)"
    done

    log_info "SSH configuration complete"
}

# Generate council configuration
generate_config() {
    local mode="$1"
    local hosts="$2"

    log_info "Generating council configuration..."

    if [ "$mode" == "local" ]; then
        cat > council.json << 'EOF'
{
  "name": "Local Development Council",
  "nodes": [
    {
      "hostname": "localhost",
      "ip": "127.0.0.1",
      "port": 8080,
      "model": "mlx-community/Llama-3.2-3B-Instruct-4bit",
      "gpuMemory": 32,
      "chip": "Apple Silicon"
    }
  ],
  "defaultModel": "mlx-community/Llama-3.2-3B-Instruct-4bit",
  "backend": "ring",
  "votingStrategy": "weighted"
}
EOF
    else
        # Generate distributed config
        python3 << EOF
import json
import subprocess
import socket

hosts = "$hosts".split(",")
nodes = []

for i, host in enumerate(hosts):
    host = host.strip()

    # Try to resolve IP
    try:
        ip = socket.gethostbyname(host)
    except:
        ip = host

    # Try to detect GPU memory via SSH
    gpu_memory = 32
    chip = "Apple Silicon"

    try:
        result = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=5", host, "sysctl -n hw.memsize"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            gpu_memory = int(result.stdout.strip()) // (1024**3)
    except:
        pass

    nodes.append({
        "hostname": host,
        "ip": ip,
        "port": 8080 + i,
        "model": "mlx-community/Llama-3.2-3B-Instruct-4bit",
        "gpuMemory": gpu_memory,
        "chip": chip
    })

config = {
    "name": "Distributed MLX Council",
    "nodes": nodes,
    "defaultModel": "mlx-community/Llama-3.2-3B-Instruct-4bit",
    "backend": "ring",
    "votingStrategy": "weighted"
}

with open("council.json", "w") as f:
    json.dump(config, f, indent=2)

print(f"Generated configuration for {len(nodes)} nodes")
EOF
    fi

    log_info "Configuration saved to council.json"
}

# Install council on remote nodes
install_remote() {
    local hosts="$1"

    log_info "Installing MLX council on remote nodes..."

    IFS=',' read -ra HOSTS <<< "$hosts"
    for host in "${HOSTS[@]}"; do
        log_info "Installing on $host..."

        ssh "$host" << 'REMOTE_SCRIPT'
# Install dependencies
pip3 install --upgrade pip
pip3 install mlx mlx-lm fastapi uvicorn httpx

# Create council directory
mkdir -p ~/.mlx-council

echo "Installation complete on $(hostname)"
REMOTE_SCRIPT

    done

    log_info "Remote installation complete"
}

# Main setup function
main() {
    local mode="local"
    local hosts=""

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --local)
                mode="local"
                shift
                ;;
            --distributed)
                mode="distributed"
                shift
                ;;
            --hosts)
                hosts="$2"
                shift 2
                ;;
            --help|-h)
                echo "Usage: $0 [--local|--distributed] [--hosts host1,host2,...]"
                echo ""
                echo "Options:"
                echo "  --local         Setup for single Mac development"
                echo "  --distributed   Setup for multi-Mac cluster"
                echo "  --hosts <list>  Hostnames for distributed setup"
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done

    echo ""
    echo "╔══════════════════════════════════════════════════════════════════╗"
    echo "║           MLX Deep Council Setup                                  ║"
    echo "║   Distributed Multi-Model Consensus for Apple Silicon            ║"
    echo "╚══════════════════════════════════════════════════════════════════╝"
    echo ""

    # Run setup steps
    check_macos
    install_python_deps

    if [ "$mode" == "distributed" ] && [ -n "$hosts" ]; then
        setup_ssh "$hosts"
        install_remote "$hosts"
    fi

    download_models
    generate_config "$mode" "$hosts"

    echo ""
    log_info "Setup complete!"
    echo ""
    echo "Next steps:"
    echo "  1. Review the generated council.json configuration"
    echo "  2. Start the council:"
    if [ "$mode" == "local" ]; then
        echo "     npx ts-node platform/council/council-launcher.ts launch"
    else
        echo "     npx ts-node platform/council/council-launcher.ts launch --config council.json"
    fi
    echo ""
    echo "  3. Or run a quick query:"
    echo "     npx ts-node platform/index.ts council \"Your question here\""
    echo ""
}

# Run main
main "$@"
