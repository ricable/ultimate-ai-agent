#!/bin/bash

# Start API Gateway - Phase 3 Deployment Script
# Starts the enhanced distributed API server with full features

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG_DIR="$PROJECT_ROOT/config"
LOGS_DIR="$PROJECT_ROOT/logs"
PID_FILE="$LOGS_DIR/api_gateway.pid"

# Default values
NODE_ID="${NODE_ID:-api-gateway-1}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
CONFIG_FILE="${CONFIG_FILE:-$CONFIG_DIR/api_gateway.yaml}"
LOG_LEVEL="${LOG_LEVEL:-info}"
WORKERS="${WORKERS:-1}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_debug() {
    echo -e "${BLUE}[DEBUG]${NC} $1"
}

# Help function
show_help() {
    cat << EOF
Start API Gateway - Phase 3 Enhanced Distributed API Server

Usage: $0 [OPTIONS]

OPTIONS:
    -h, --help          Show this help message
    -n, --node-id       Node identifier (default: api-gateway-1)
    -H, --host          Host to bind to (default: 0.0.0.0)
    -p, --port          Port to bind to (default: 8000)
    -c, --config        Config file path (default: config/api_gateway.yaml)
    -l, --log-level     Log level (default: info)
    -w, --workers       Number of workers (default: 1)
    -d, --daemon        Run as daemon
    -s, --stop          Stop running server
    --status            Show server status
    --reload            Enable auto-reload for development

ENVIRONMENT VARIABLES:
    NODE_ID             Node identifier
    HOST                Host to bind to
    PORT                Port to bind to
    CONFIG_FILE         Configuration file path
    LOG_LEVEL           Logging level
    WORKERS             Number of worker processes

EXAMPLES:
    # Start with default settings
    $0

    # Start with custom node ID and port
    $0 --node-id api-gw-main --port 8080

    # Start as daemon
    $0 --daemon

    # Stop running server
    $0 --stop

    # Check status
    $0 --status

EOF
}

# Check dependencies
check_dependencies() {
    log_info "Checking dependencies..."
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is required but not installed"
        exit 1
    fi
    
    # Check virtual environment
    if [[ -z "${VIRTUAL_ENV}" ]]; then
        log_warn "No virtual environment activated"
        if [[ -f "$PROJECT_ROOT/venv/bin/activate" ]]; then
            log_info "Activating virtual environment..."
            source "$PROJECT_ROOT/venv/bin/activate"
        else
            log_warn "No virtual environment found. Consider creating one."
        fi
    fi
    
    # Check required Python packages
    local required_packages=("fastapi" "uvicorn" "pydantic")
    for package in "${required_packages[@]}"; do
        if ! python3 -c "import $package" &> /dev/null; then
            log_error "Required package '$package' not found"
            log_info "Install with: pip install $package"
            exit 1
        fi
    done
    
    log_info "✓ Dependencies check passed"
}

# Create necessary directories
setup_directories() {
    log_info "Setting up directories..."
    
    mkdir -p "$LOGS_DIR"
    mkdir -p "$CONFIG_DIR"
    mkdir -p "$PROJECT_ROOT/data"
    mkdir -p "$PROJECT_ROOT/models"
    
    log_info "✓ Directories created"
}

# Create default configuration if it doesn't exist
create_default_config() {
    if [[ ! -f "$CONFIG_FILE" ]]; then
        log_info "Creating default configuration..."
        
        cat > "$CONFIG_FILE" << 'EOF'
# API Gateway Configuration - Phase 3
# Enhanced distributed API server settings

server:
  host: "0.0.0.0"
  port: 8000
  workers: 1
  log_level: "info"
  reload: false
  trusted_hosts: ["*"]
  cors_origins: ["*"]

# Authentication settings
auth:
  enabled: true
  jwt_secret: null  # Will be auto-generated if null
  create_admin_key: true
  max_key_age_days: 365

# Rate limiting settings
rate_limiting:
  enabled: true
  adaptive: true
  default_tier: "standard"
  cleanup_interval: 3600

# Load balancing settings
load_balancing:
  enabled: true
  strategy: "resource_aware"  # round_robin, least_connections, weighted_round_robin, resource_aware, consistent_hashing
  health_check_interval: 30
  max_retries: 3

# Cluster nodes (worker nodes for load balancing)
cluster_nodes:
  - node_id: "worker-1"
    host: "10.0.1.10"
    port: 8001
    weight: 1.0
    max_connections: 100
    active_models: ["llama-7b", "mistral-7b"]
  - node_id: "worker-2"
    host: "10.0.1.11"
    port: 8001
    weight: 1.5
    max_connections: 150
    active_models: ["llama-7b", "llama-13b"]

# Default models to load
default_models:
  - name: "llama-7b"
    architecture: "llama"
    num_layers: 32
    hidden_size: 4096
    num_attention_heads: 32
    vocab_size: 32000
    max_sequence_length: 2048
  - name: "mistral-7b"
    architecture: "mistral"
    num_layers: 32
    hidden_size: 4096

# Monitoring settings
monitoring:
  metrics_enabled: true
  health_check_endpoint: true
  detailed_logging: true

# Security settings
security:
  request_timeout: 300
  max_request_size: "10MB"
  enable_ssl: false
  ssl_cert_path: null
  ssl_key_path: null
EOF
        
        log_info "✓ Default configuration created at $CONFIG_FILE"
    fi
}

# Parse command line arguments
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -n|--node-id)
                NODE_ID="$2"
                shift 2
                ;;
            -H|--host)
                HOST="$2"
                shift 2
                ;;
            -p|--port)
                PORT="$2"
                shift 2
                ;;
            -c|--config)
                CONFIG_FILE="$2"
                shift 2
                ;;
            -l|--log-level)
                LOG_LEVEL="$2"
                shift 2
                ;;
            -w|--workers)
                WORKERS="$2"
                shift 2
                ;;
            -d|--daemon)
                DAEMON=true
                shift
                ;;
            -s|--stop)
                STOP=true
                shift
                ;;
            --status)
                STATUS=true
                shift
                ;;
            --reload)
                RELOAD=true
                shift
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

# Check if server is running
is_running() {
    if [[ -f "$PID_FILE" ]]; then
        local pid=$(cat "$PID_FILE")
        if ps -p "$pid" > /dev/null 2>&1; then
            return 0
        else
            rm -f "$PID_FILE"
            return 1
        fi
    fi
    return 1
}

# Stop server
stop_server() {
    log_info "Stopping API Gateway..."
    
    if is_running; then
        local pid=$(cat "$PID_FILE")
        log_info "Sending SIGTERM to process $pid"
        kill "$pid"
        
        # Wait for graceful shutdown
        local count=0
        while ps -p "$pid" > /dev/null 2>&1 && [[ $count -lt 30 ]]; do
            sleep 1
            count=$((count + 1))
        done
        
        if ps -p "$pid" > /dev/null 2>&1; then
            log_warn "Process did not stop gracefully, sending SIGKILL"
            kill -9 "$pid"
        fi
        
        rm -f "$PID_FILE"
        log_info "✓ API Gateway stopped"
    else
        log_warn "API Gateway is not running"
    fi
}

# Show server status
show_status() {
    if is_running; then
        local pid=$(cat "$PID_FILE")
        log_info "API Gateway is running (PID: $pid)"
        
        # Try to get status from API
        if command -v curl &> /dev/null; then
            log_info "Checking API health..."
            if curl -s "http://$HOST:$PORT/health" > /dev/null; then
                log_info "✓ API Gateway is responding"
            else
                log_warn "API Gateway is not responding to health checks"
            fi
        fi
    else
        log_info "API Gateway is not running"
    fi
}

# Start server
start_server() {
    log_info "Starting API Gateway..."
    log_info "Node ID: $NODE_ID"
    log_info "Host: $HOST"
    log_info "Port: $PORT"
    log_info "Config: $CONFIG_FILE"
    log_info "Log Level: $LOG_LEVEL"
    
    # Check if already running
    if is_running; then
        log_error "API Gateway is already running (PID: $(cat "$PID_FILE"))"
        exit 1
    fi
    
    # Build command
    local cmd="python3 -m src.enhanced_api_server"
    local args="--node-id $NODE_ID --host $HOST --port $PORT --config $CONFIG_FILE --log-level $LOG_LEVEL"
    
    if [[ "$RELOAD" == "true" ]]; then
        args="$args --reload"
    fi
    
    # Set environment variables
    export NODE_ID="$NODE_ID"
    export HOST="$HOST"
    export PORT="$PORT"
    export CONFIG_FILE="$CONFIG_FILE"
    export LOG_LEVEL="$LOG_LEVEL"
    
    # Change to project directory
    cd "$PROJECT_ROOT"
    
    if [[ "$DAEMON" == "true" ]]; then
        # Start as daemon
        log_info "Starting as daemon..."
        nohup $cmd $args > "$LOGS_DIR/api_gateway.log" 2>&1 &
        local pid=$!
        echo "$pid" > "$PID_FILE"
        
        # Wait a moment and check if it started successfully
        sleep 2
        if ps -p "$pid" > /dev/null 2>&1; then
            log_info "✓ API Gateway started successfully (PID: $pid)"
            log_info "Logs: $LOGS_DIR/api_gateway.log"
        else
            log_error "Failed to start API Gateway"
            rm -f "$PID_FILE"
            exit 1
        fi
    else
        # Start in foreground
        log_info "Starting in foreground..."
        exec $cmd $args
    fi
}

# Main function
main() {
    # Parse arguments
    parse_arguments "$@"
    
    # Handle special commands
    if [[ "$STOP" == "true" ]]; then
        stop_server
        exit 0
    fi
    
    if [[ "$STATUS" == "true" ]]; then
        show_status
        exit 0
    fi
    
    # Setup
    check_dependencies
    setup_directories
    create_default_config
    
    # Start server
    start_server
}

# Trap signals for graceful shutdown
trap 'log_info "Received signal, shutting down..."; stop_server; exit 0' SIGINT SIGTERM

# Run main function
main "$@"