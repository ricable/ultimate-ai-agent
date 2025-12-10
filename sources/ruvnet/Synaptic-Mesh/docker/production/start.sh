#!/bin/bash
# Production startup script for Synaptic Neural Mesh
# Handles graceful startup, health checks, and process management

set -euo pipefail

# Environment variables with defaults
export NODE_ENV=${NODE_ENV:-production}
export NEURAL_MESH_MODE=${NEURAL_MESH_MODE:-coordinator}
export QUDAG_DATA_DIR=${QUDAG_DATA_DIR:-/app/data}
export QUDAG_CONFIG_DIR=${QUDAG_CONFIG_DIR:-/app/config}
export NEURAL_MESH_LOG_DIR=${NEURAL_MESH_LOG_DIR:-/app/logs}

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [STARTUP] $*" >&2
}

# Error handling
error_exit() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [ERROR] $*" >&2
    exit 1
}

# Signal handlers
cleanup() {
    log "Received shutdown signal, cleaning up..."
    if [[ -f /tmp/neural-mesh.pid ]]; then
        local pid=$(cat /tmp/neural-mesh.pid)
        if kill -0 "$pid" 2>/dev/null; then
            log "Gracefully stopping neural mesh (PID: $pid)..."
            kill -TERM "$pid"
            # Wait up to 30 seconds for graceful shutdown
            for i in {1..30}; do
                if ! kill -0 "$pid" 2>/dev/null; then
                    log "Neural mesh stopped gracefully"
                    break
                fi
                sleep 1
            done
            # Force kill if still running
            if kill -0 "$pid" 2>/dev/null; then
                log "Force killing neural mesh process..."
                kill -KILL "$pid"
            fi
        fi
        rm -f /tmp/neural-mesh.pid
    fi
    
    # Stop PM2 processes
    if command -v pm2 >/dev/null 2>&1; then
        pm2 delete all 2>/dev/null || true
        pm2 kill 2>/dev/null || true
    fi
    
    log "Cleanup completed"
    exit 0
}

# Set up signal handling
trap cleanup TERM INT QUIT

# Pre-startup validation
validate_environment() {
    log "Validating environment..."
    
    # Check required directories
    for dir in "$QUDAG_DATA_DIR" "$QUDAG_CONFIG_DIR" "$NEURAL_MESH_LOG_DIR"; do
        if [[ ! -d "$dir" ]]; then
            log "Creating directory: $dir"
            mkdir -p "$dir" || error_exit "Failed to create directory: $dir"
        fi
        if [[ ! -w "$dir" ]]; then
            error_exit "Directory not writable: $dir"
        fi
    done
    
    # Check required binaries
    for cmd in qudag node; do
        if ! command -v "$cmd" >/dev/null 2>&1; then
            error_exit "Required command not found: $cmd"
        fi
    done
    
    # Validate configuration
    if [[ ! -f "$QUDAG_CONFIG_DIR/node.toml" ]]; then
        log "Warning: No configuration file found at $QUDAG_CONFIG_DIR/node.toml"
        log "Creating default configuration..."
        /app/docker/create-default-config.sh
    fi
    
    log "Environment validation completed"
}

# Health check function
health_check() {
    local max_attempts=30
    local attempt=1
    
    while [[ $attempt -le $max_attempts ]]; do
        if curl -f -s "http://localhost:8080/health" >/dev/null 2>&1; then
            log "Health check passed (attempt $attempt/$max_attempts)"
            return 0
        fi
        log "Health check failed (attempt $attempt/$max_attempts), retrying in 2 seconds..."
        sleep 2
        ((attempt++))
    done
    
    error_exit "Health check failed after $max_attempts attempts"
}

# Start QuDAG node
start_qudag() {
    log "Starting QuDAG node..."
    
    local cmd="qudag start --config $QUDAG_CONFIG_DIR/node.toml"
    
    # Add bootstrap mode if required
    if [[ "${BOOTSTRAP_MODE:-false}" == "true" ]]; then
        cmd="$cmd --bootstrap"
    fi
    
    # Add bootstrap peers if specified
    if [[ -n "${BOOTSTRAP_PEERS:-}" ]]; then
        cmd="$cmd --bootstrap-peers $BOOTSTRAP_PEERS"
    fi
    
    log "Executing: $cmd"
    
    # Start QuDAG in background
    nohup $cmd > "$NEURAL_MESH_LOG_DIR/qudag.log" 2>&1 &
    local qudag_pid=$!
    echo "$qudag_pid" > /tmp/qudag.pid
    
    log "QuDAG started with PID: $qudag_pid"
}

# Start Neural Mesh services
start_neural_mesh() {
    log "Starting Neural Mesh services..."
    
    # Create PM2 ecosystem configuration
    cat > /tmp/neural-mesh-ecosystem.config.js << EOF
module.exports = {
  apps: [
    {
      name: 'neural-mesh-core',
      script: '/app/src/js/ruv-swarm/src/index.js',
      instances: 1,
      exec_mode: 'fork',
      env: {
        NODE_ENV: '${NODE_ENV}',
        NEURAL_MESH_MODE: '${NEURAL_MESH_MODE}',
        RUST_LOG: '${RUST_LOG:-info}',
        PORT: '8081'
      },
      log_file: '${NEURAL_MESH_LOG_DIR}/neural-mesh-core.log',
      error_file: '${NEURAL_MESH_LOG_DIR}/neural-mesh-core-error.log',
      out_file: '${NEURAL_MESH_LOG_DIR}/neural-mesh-core-out.log',
      merge_logs: true,
      time: true
    },
    {
      name: 'mcp-server',
      script: '/app/src/mcp/server/mcp-server.js',
      instances: 1,
      exec_mode: 'fork',
      env: {
        NODE_ENV: '${NODE_ENV}',
        MCP_PORT: '3000',
        NEURAL_MESH_NODES: 'localhost:8081'
      },
      log_file: '${NEURAL_MESH_LOG_DIR}/mcp-server.log',
      error_file: '${NEURAL_MESH_LOG_DIR}/mcp-server-error.log',
      out_file: '${NEURAL_MESH_LOG_DIR}/mcp-server-out.log',
      merge_logs: true,
      time: true
    }
  ]
};
EOF
    
    # Start PM2 processes
    pm2 start /tmp/neural-mesh-ecosystem.config.js --no-daemon &
    local pm2_pid=$!
    echo "$pm2_pid" > /tmp/neural-mesh.pid
    
    log "Neural Mesh services started with PM2 PID: $pm2_pid"
}

# Wait for services to be ready
wait_for_services() {
    log "Waiting for services to be ready..."
    
    # Wait for QuDAG
    local max_wait=60
    local wait_time=0
    while [[ $wait_time -lt $max_wait ]]; do
        if curl -f -s "http://localhost:8080/health" >/dev/null 2>&1; then
            log "QuDAG service is ready"
            break
        fi
        sleep 2
        ((wait_time+=2))
    done
    
    if [[ $wait_time -ge $max_wait ]]; then
        error_exit "QuDAG service failed to start within $max_wait seconds"
    fi
    
    # Wait for Neural Mesh API
    wait_time=0
    while [[ $wait_time -lt $max_wait ]]; do
        if curl -f -s "http://localhost:8081/health" >/dev/null 2>&1; then
            log "Neural Mesh API is ready"
            break
        fi
        sleep 2
        ((wait_time+=2))
    done
    
    if [[ $wait_time -ge $max_wait ]]; then
        error_exit "Neural Mesh API failed to start within $max_wait seconds"
    fi
    
    # Wait for MCP server
    wait_time=0
    while [[ $wait_time -lt $max_wait ]]; do
        if curl -f -s "http://localhost:3000/health" >/dev/null 2>&1; then
            log "MCP server is ready"
            break
        fi
        sleep 2
        ((wait_time+=2))
    done
    
    if [[ $wait_time -ge $max_wait ]]; then
        error_exit "MCP server failed to start within $max_wait seconds"
    fi
    
    log "All services are ready"
}

# Main startup sequence
main() {
    log "Starting Synaptic Neural Mesh production deployment..."
    log "Node ID: ${NODE_ID:-unknown}"
    log "Mesh Role: ${MESH_ROLE:-unknown}"
    log "Neural Mesh Mode: $NEURAL_MESH_MODE"
    
    validate_environment
    start_qudag
    start_neural_mesh
    wait_for_services
    
    log "Synaptic Neural Mesh started successfully"
    log "Services running:"
    log "  - QuDAG P2P: localhost:4001"
    log "  - QuDAG RPC: localhost:8080"
    log "  - Neural Mesh API: localhost:8081"
    log "  - MCP Server: localhost:3000"
    log "  - Metrics: localhost:9090"
    
    # Keep the container running and monitor processes
    while true; do
        # Check if QuDAG is still running
        if [[ -f /tmp/qudag.pid ]]; then
            local qudag_pid=$(cat /tmp/qudag.pid)
            if ! kill -0 "$qudag_pid" 2>/dev/null; then
                error_exit "QuDAG process died unexpectedly"
            fi
        fi
        
        # Check if PM2 is still running
        if [[ -f /tmp/neural-mesh.pid ]]; then
            local pm2_pid=$(cat /tmp/neural-mesh.pid)
            if ! kill -0 "$pm2_pid" 2>/dev/null; then
                error_exit "Neural Mesh processes died unexpectedly"
            fi
        fi
        
        sleep 30
    done
}

# Execute main function
main "$@"