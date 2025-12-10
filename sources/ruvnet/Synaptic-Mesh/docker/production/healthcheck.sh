#!/bin/bash
# Comprehensive health check for Synaptic Neural Mesh
# Validates all critical services and their interconnections

set -euo pipefail

# Configuration
HEALTH_CHECK_TIMEOUT=10
MAX_RESPONSE_TIME=5000  # milliseconds

# Logging
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [HEALTH] $*" >&2
}

# Check function with timeout
check_endpoint() {
    local url="$1"
    local description="$2"
    local expected_status="${3:-200}"
    
    log "Checking $description at $url..."
    
    local response
    local status_code
    local response_time
    
    # Use curl with timeout and capture timing
    if response=$(curl -s -w "HTTPSTATUS:%{http_code};TIME:%{time_total}" \
                       --max-time "$HEALTH_CHECK_TIMEOUT" \
                       --connect-timeout 5 \
                       "$url" 2>/dev/null); then
        
        status_code=$(echo "$response" | grep -o "HTTPSTATUS:[0-9]*" | cut -d: -f2)
        response_time=$(echo "$response" | grep -o "TIME:[0-9.]*" | cut -d: -f2)
        response_body=$(echo "$response" | sed -E 's/HTTPSTATUS:[0-9]*;TIME:[0-9.]*$//')
        
        # Convert response time to milliseconds
        response_time_ms=$(echo "$response_time * 1000" | bc -l | cut -d. -f1)
        
        if [[ "$status_code" == "$expected_status" ]]; then
            if [[ $response_time_ms -le $MAX_RESPONSE_TIME ]]; then
                log "✓ $description: OK (${response_time_ms}ms)"
                return 0
            else
                log "⚠ $description: SLOW (${response_time_ms}ms > ${MAX_RESPONSE_TIME}ms)"
                return 1
            fi
        else
            log "✗ $description: HTTP $status_code (expected $expected_status)"
            return 1
        fi
    else
        log "✗ $description: Connection failed"
        return 1
    fi
}

# Check process is running
check_process() {
    local process_name="$1"
    local pid_file="$2"
    
    if [[ -f "$pid_file" ]]; then
        local pid=$(cat "$pid_file")
        if kill -0 "$pid" 2>/dev/null; then
            log "✓ $process_name: Running (PID: $pid)"
            return 0
        else
            log "✗ $process_name: PID file exists but process is dead"
            return 1
        fi
    else
        log "✗ $process_name: PID file not found"
        return 1
    fi
}

# Check disk space
check_disk_space() {
    local path="$1"
    local min_free_percent="${2:-10}"
    
    if [[ -d "$path" ]]; then
        local usage=$(df "$path" | awk 'NR==2 {print $(NF-1)}' | sed 's/%//')
        local free_percent=$((100 - usage))
        
        if [[ $free_percent -ge $min_free_percent ]]; then
            log "✓ Disk space $path: ${free_percent}% free"
            return 0
        else
            log "⚠ Disk space $path: Only ${free_percent}% free (minimum: ${min_free_percent}%)"
            return 1
        fi
    else
        log "✗ Disk space: Path $path does not exist"
        return 1
    fi
}

# Check memory usage
check_memory() {
    local max_usage_percent="${1:-90}"
    
    if command -v free >/dev/null 2>&1; then
        local mem_usage=$(free | awk 'NR==2{printf "%.0f", $3*100/$2}')
        
        if [[ $mem_usage -le $max_usage_percent ]]; then
            log "✓ Memory usage: ${mem_usage}%"
            return 0
        else
            log "⚠ Memory usage: ${mem_usage}% (max: ${max_usage_percent}%)"
            return 1
        fi
    else
        log "⚠ Memory check: 'free' command not available"
        return 0  # Don't fail health check if we can't check memory
    fi
}

# Check QuDAG peer connectivity
check_qudag_peers() {
    local rpc_url="http://localhost:8080"
    
    log "Checking QuDAG peer connectivity..."
    
    # Try to get peer count via RPC
    local response
    if response=$(curl -s --max-time 5 \
                       -H "Content-Type: application/json" \
                       -d '{"method":"get_peer_count","params":[],"id":1}' \
                       "$rpc_url/rpc" 2>/dev/null); then
        
        if echo "$response" | grep -q '"result"'; then
            local peer_count=$(echo "$response" | jq -r '.result // 0' 2>/dev/null || echo "0")
            log "✓ QuDAG peers: $peer_count connected"
            return 0
        else
            log "⚠ QuDAG peers: RPC response invalid"
            return 1
        fi
    else
        log "⚠ QuDAG peers: RPC call failed"
        return 1
    fi
}

# Check neural mesh agent status
check_neural_agents() {
    local api_url="http://localhost:8081"
    
    log "Checking neural mesh agents..."
    
    if response=$(curl -s --max-time 5 "$api_url/agents/status" 2>/dev/null); then
        if echo "$response" | grep -q '"status"'; then
            local active_agents=$(echo "$response" | jq -r '.active_agents // 0' 2>/dev/null || echo "0")
            log "✓ Neural agents: $active_agents active"
            return 0
        else
            log "⚠ Neural agents: Invalid response"
            return 1
        fi
    else
        log "⚠ Neural agents: API call failed"
        return 1
    fi
}

# Main health check
main() {
    log "Starting comprehensive health check..."
    
    local health_status=0
    local warnings=0
    
    # Core service checks
    if ! check_endpoint "http://localhost:8080/health" "QuDAG RPC"; then
        health_status=1
    fi
    
    if ! check_endpoint "http://localhost:8081/health" "Neural Mesh API"; then
        health_status=1
    fi
    
    if ! check_endpoint "http://localhost:3000/health" "MCP Server"; then
        health_status=1
    fi
    
    if ! check_endpoint "http://localhost:9090/metrics" "Metrics endpoint"; then
        ((warnings++))
    fi
    
    # Process checks
    if ! check_process "QuDAG" "/tmp/qudag.pid"; then
        health_status=1
    fi
    
    if ! check_process "Neural Mesh" "/tmp/neural-mesh.pid"; then
        health_status=1
    fi
    
    # Resource checks
    if ! check_disk_space "${QUDAG_DATA_DIR:-/app/data}" 10; then
        ((warnings++))
    fi
    
    if ! check_disk_space "${NEURAL_MESH_LOG_DIR:-/app/logs}" 5; then
        ((warnings++))
    fi
    
    if ! check_memory 90; then
        ((warnings++))
    fi
    
    # Connectivity checks
    if ! check_qudag_peers; then
        ((warnings++))
    fi
    
    if ! check_neural_agents; then
        ((warnings++))
    fi
    
    # Summary
    if [[ $health_status -eq 0 ]]; then
        if [[ $warnings -eq 0 ]]; then
            log "✓ Health check PASSED - All systems operational"
            exit 0
        else
            log "⚠ Health check PASSED with $warnings warning(s)"
            exit 0
        fi
    else
        log "✗ Health check FAILED - Critical issues detected"
        exit 1
    fi
}

# Install bc if needed for calculations
if ! command -v bc >/dev/null 2>&1; then
    log "Installing bc for calculations..."
    if command -v apk >/dev/null 2>&1; then
        apk add --no-cache bc >/dev/null 2>&1 || true
    elif command -v apt-get >/dev/null 2>&1; then
        apt-get update >/dev/null 2>&1 && apt-get install -y bc >/dev/null 2>&1 || true
    fi
fi

# Install jq if needed for JSON parsing
if ! command -v jq >/dev/null 2>&1; then
    log "Installing jq for JSON parsing..."
    if command -v apk >/dev/null 2>&1; then
        apk add --no-cache jq >/dev/null 2>&1 || true
    elif command -v apt-get >/dev/null 2>&1; then
        apt-get update >/dev/null 2>&1 && apt-get install -y jq >/dev/null 2>&1 || true
    fi
fi

# Execute main function
main "$@"