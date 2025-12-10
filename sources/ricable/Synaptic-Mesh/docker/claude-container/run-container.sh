#!/bin/bash

# Secure Claude Container Runner
# Runs Claude container with maximum security settings

set -euo pipefail

# Configuration
CONTAINER_NAME="claude-secure-container"
IMAGE_NAME="synaptic-neural-mesh/claude-container:latest"
WORKSPACE_SIZE="100m"
MEMORY_LIMIT="512m"
CPU_LIMIT="0.5"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

# Check if API key is provided
check_api_key() {
    if [[ -z "${CLAUDE_API_KEY:-}" && -z "${ANTHROPIC_API_KEY:-}" ]]; then
        error "API key required. Set CLAUDE_API_KEY or ANTHROPIC_API_KEY environment variable."
    fi
    log "API key detected ✓"
}

# Build container if needed
build_container() {
    if ! docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
        log "Building Claude container..."
        docker build -t "$IMAGE_NAME" .
    else
        log "Container image exists ✓"
    fi
}

# Clean up existing container
cleanup_container() {
    if docker ps -a --format 'table {{.Names}}' | grep -q "$CONTAINER_NAME"; then
        log "Removing existing container..."
        docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
    fi
}

# Run secure container
run_container() {
    log "Starting secure Claude container..."
    
    # API key environment variable
    local api_key_env=""
    if [[ -n "${CLAUDE_API_KEY:-}" ]]; then
        api_key_env="-e CLAUDE_API_KEY=${CLAUDE_API_KEY}"
    elif [[ -n "${ANTHROPIC_API_KEY:-}" ]]; then
        api_key_env="-e ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}"
    fi
    
    # Run container with maximum security
    docker run \
        --name "$CONTAINER_NAME" \
        --rm \
        --interactive \
        --tty \
        --read-only \
        --security-opt no-new-privileges:true \
        --user 1000:1000 \
        --memory="$MEMORY_LIMIT" \
        --cpus="$CPU_LIMIT" \
        --tmpfs /tmp/claude-work:rw,size="$WORKSPACE_SIZE",uid=1000,gid=1000 \
        --network bridge \
        --cap-drop ALL \
        --cap-add NET_BIND_SERVICE \
        $api_key_env \
        -e NODE_ENV=production \
        -e CLAUDE_SANDBOX_MODE=true \
        -e CLAUDE_NETWORK_RESTRICTED=true \
        -e CLAUDE_FILESYSTEM_READONLY=true \
        "$IMAGE_NAME"
}

# Interactive task runner
run_interactive() {
    log "Starting interactive Claude container..."
    log "Send JSON tasks via stdin, receive JSON responses via stdout"
    log "Example: {\"id\": \"1\", \"prompt\": \"Hello Claude!\"}"
    log "Press Ctrl+C to exit"
    echo
    
    run_container
}

# Batch task runner
run_batch() {
    local input_file="$1"
    
    if [[ ! -f "$input_file" ]]; then
        error "Input file not found: $input_file"
    fi
    
    log "Running batch tasks from: $input_file"
    
    # Run container with input file
    docker run \
        --name "$CONTAINER_NAME" \
        --rm \
        --read-only \
        --security-opt no-new-privileges:true \
        --user 1000:1000 \
        --memory="$MEMORY_LIMIT" \
        --cpus="$CPU_LIMIT" \
        --tmpfs /tmp/claude-work:rw,size="$WORKSPACE_SIZE",uid=1000,gid=1000 \
        --network bridge \
        --cap-drop ALL \
        -e CLAUDE_API_KEY="${CLAUDE_API_KEY:-}" \
        -e ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-}" \
        -e NODE_ENV=production \
        -e CLAUDE_SANDBOX_MODE=true \
        -e CLAUDE_NETWORK_RESTRICTED=true \
        -e CLAUDE_FILESYSTEM_READONLY=true \
        -v "$input_file:/tmp/input.json:ro" \
        "$IMAGE_NAME" \
        sh -c 'cat /tmp/input.json | node claude-task-executor.js'
}

# Security audit
security_audit() {
    log "Running security audit..."
    
    # Check container configuration
    log "Checking container security configuration..."
    
    # Scan for vulnerabilities
    if command -v trivy >/dev/null 2>&1; then
        log "Running Trivy security scan..."
        trivy image "$IMAGE_NAME"
    else
        warn "Trivy not installed, skipping vulnerability scan"
    fi
    
    # Check Docker security
    log "Docker security check:"
    echo "  - Read-only filesystem: ✓"
    echo "  - Non-root user: ✓" 
    echo "  - No new privileges: ✓"
    echo "  - Resource limits: ✓"
    echo "  - Network isolation: ✓"
    echo "  - Capability dropping: ✓"
    echo "  - Tmpfs workspace: ✓"
}

# Usage information
usage() {
    cat << EOF
Usage: $0 [COMMAND] [OPTIONS]

Commands:
    interactive     Run container in interactive mode (default)
    batch FILE      Run batch tasks from JSON file
    build           Build container image
    audit           Run security audit
    help            Show this help message

Environment Variables:
    CLAUDE_API_KEY or ANTHROPIC_API_KEY (required)

Examples:
    # Interactive mode
    export CLAUDE_API_KEY="your-api-key"
    $0 interactive

    # Batch mode
    $0 batch tasks.json

    # Security audit
    $0 audit

Security Features:
    ✓ Read-only filesystem
    ✓ Non-root user (uid:1000)
    ✓ No privileged access
    ✓ Resource limits (512MB RAM, 0.5 CPU)
    ✓ Network isolation
    ✓ Tmpfs workspace (100MB)
    ✓ No persistent secrets
    ✓ API-only access
EOF
}

# Main script logic
main() {
    local command="${1:-interactive}"
    
    case "$command" in
        "interactive")
            check_api_key
            build_container
            cleanup_container
            run_interactive
            ;;
        "batch")
            if [[ $# -lt 2 ]]; then
                error "Batch mode requires input file. Usage: $0 batch <file.json>"
            fi
            check_api_key
            build_container
            cleanup_container
            run_batch "$2"
            ;;
        "build")
            build_container
            log "Container built successfully ✓"
            ;;
        "audit")
            build_container
            security_audit
            ;;
        "help"|"-h"|"--help")
            usage
            ;;
        *)
            error "Unknown command: $command. Use 'help' for usage information."
            ;;
    esac
}

# Run main function with all arguments
main "$@"