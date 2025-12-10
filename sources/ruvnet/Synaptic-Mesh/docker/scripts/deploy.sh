#!/bin/bash
# Production deployment script for Synaptic Neural Mesh
# Handles zero-downtime deployments with rollback capabilities

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DEPLOY_ENV="${DEPLOY_ENV:-production}"
BACKUP_RETENTION_DAYS="${BACKUP_RETENTION_DAYS:-7}"
HEALTH_CHECK_TIMEOUT="${HEALTH_CHECK_TIMEOUT:-300}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')] [DEPLOY]${NC} $*"
}

warn() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] [WARN]${NC} $*" >&2
}

error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] [ERROR]${NC} $*" >&2
}

success() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] [SUCCESS]${NC} $*"
}

# Error handling
cleanup() {
    local exit_code=$?
    if [[ $exit_code -ne 0 ]]; then
        error "Deployment failed with exit code $exit_code"
        if [[ "${ROLLBACK_ON_FAILURE:-true}" == "true" ]]; then
            log "Initiating automatic rollback..."
            rollback_deployment
        fi
    fi
    exit $exit_code
}

trap cleanup EXIT

# Pre-deployment validation
validate_environment() {
    log "Validating deployment environment..."
    
    # Check required commands
    local required_commands=("docker" "docker-compose" "curl" "jq")
    for cmd in "${required_commands[@]}"; do
        if ! command -v "$cmd" >/dev/null 2>&1; then
            error "Required command not found: $cmd"
            exit 1
        fi
    done
    
    # Check Docker daemon
    if ! docker info >/dev/null 2>&1; then
        error "Docker daemon is not running or accessible"
        exit 1
    fi
    
    # Check available disk space (minimum 5GB)
    local available_space=$(df "$PROJECT_ROOT" | awk 'NR==2 {print $4}')
    local min_space=$((5 * 1024 * 1024))  # 5GB in KB
    
    if [[ $available_space -lt $min_space ]]; then
        error "Insufficient disk space. Available: $((available_space / 1024 / 1024))GB, Required: 5GB"
        exit 1
    fi
    
    # Check memory (minimum 4GB)
    local available_memory=$(free -m | awk 'NR==2{print $7}')
    if [[ $available_memory -lt 4096 ]]; then
        warn "Low available memory: ${available_memory}MB (recommended: 4096MB)"
    fi
    
    # Validate configuration files
    if [[ ! -f "$PROJECT_ROOT/docker-compose.yml" ]]; then
        error "Docker Compose file not found: $PROJECT_ROOT/docker-compose.yml"
        exit 1
    fi
    
    # Validate environment-specific configuration
    local env_file="$PROJECT_ROOT/.env.$DEPLOY_ENV"
    if [[ ! -f "$env_file" ]]; then
        warn "Environment file not found: $env_file"
    fi
    
    success "Environment validation completed"
}

# Backup current deployment
backup_deployment() {
    log "Creating deployment backup..."
    
    local backup_dir="$PROJECT_ROOT/backups/$(date '+%Y%m%d_%H%M%S')"
    mkdir -p "$backup_dir"
    
    # Backup volumes
    log "Backing up data volumes..."
    docker run --rm \
        -v neural-node-1-data:/source:ro \
        -v "$backup_dir:/backup" \
        alpine:latest \
        tar czf /backup/neural-node-1-data.tar.gz -C /source .
    
    # Backup configuration
    log "Backing up configuration..."
    if [[ -d "$PROJECT_ROOT/config" ]]; then
        tar czf "$backup_dir/config.tar.gz" -C "$PROJECT_ROOT" config/
    fi
    
    # Backup database (if running)
    if docker-compose ps postgres | grep -q "Up"; then
        log "Backing up PostgreSQL database..."
        docker-compose exec -T postgres pg_dump -U neural_mesh neural_mesh > "$backup_dir/database.sql"
    fi
    
    # Create metadata
    cat > "$backup_dir/metadata.json" << EOF
{
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)",
    "environment": "$DEPLOY_ENV",
    "git_commit": "$(git rev-parse HEAD 2>/dev/null || echo 'unknown')",
    "git_branch": "$(git branch --show-current 2>/dev/null || echo 'unknown')",
    "docker_images": $(docker images --format "{{json .}}" | jq -s '.')
}
EOF
    
    echo "$backup_dir" > /tmp/neural-mesh-backup-path
    success "Backup created: $backup_dir"
}

# Build and test images
build_images() {
    log "Building Docker images..."
    
    cd "$PROJECT_ROOT"
    
    # Build production image
    log "Building production image..."
    docker build -t neural-mesh/synaptic-mesh:latest \
                 -t "neural-mesh/synaptic-mesh:$(date +%Y%m%d-%H%M%S)" \
                 -f Dockerfile .
    
    # Build Alpine image for edge deployment
    log "Building Alpine image..."
    docker build -t neural-mesh/synaptic-mesh:alpine \
                 -f Dockerfile.alpine .
    
    # Security scan
    if command -v trivy >/dev/null 2>&1; then
        log "Running security scan..."
        trivy image --exit-code 1 --severity HIGH,CRITICAL neural-mesh/synaptic-mesh:latest
    else
        warn "Trivy not found, skipping security scan"
    fi
    
    success "Images built successfully"
}

# Rolling deployment
rolling_deployment() {
    log "Starting rolling deployment..."
    
    cd "$PROJECT_ROOT"
    
    # Load environment
    if [[ -f ".env.$DEPLOY_ENV" ]]; then
        set -a
        source ".env.$DEPLOY_ENV"
        set +a
    fi
    
    # Deploy services one by one for zero downtime
    local services=("neural-mesh-node-1" "neural-mesh-node-2" "neural-mesh-node-3" "mcp-server")
    
    for service in "${services[@]}"; do
        log "Deploying service: $service"
        
        # Scale up new instance
        docker-compose up -d --no-deps --scale "$service=2" "$service"
        
        # Wait for health check
        wait_for_service_health "$service"
        
        # Remove old instance
        local old_container=$(docker-compose ps -q "$service" | head -1)
        if [[ -n "$old_container" ]]; then
            docker stop "$old_container"
            docker rm "$old_container"
        fi
        
        # Scale back to 1
        docker-compose up -d --no-deps --scale "$service=1" "$service"
        
        log "Service $service deployed successfully"
        sleep 5
    done
    
    success "Rolling deployment completed"
}

# Wait for service health
wait_for_service_health() {
    local service="$1"
    local max_wait="$HEALTH_CHECK_TIMEOUT"
    local wait_time=0
    
    log "Waiting for $service to be healthy..."
    
    while [[ $wait_time -lt $max_wait ]]; do
        local container_id=$(docker-compose ps -q "$service" | tail -1)
        if [[ -n "$container_id" ]]; then
            local health_status=$(docker inspect --format='{{.State.Health.Status}}' "$container_id" 2>/dev/null || echo "none")
            
            case "$health_status" in
                "healthy")
                    success "$service is healthy"
                    return 0
                    ;;
                "unhealthy")
                    error "$service is unhealthy"
                    return 1
                    ;;
                "starting"|"none")
                    log "$service health check in progress... ($wait_time/${max_wait}s)"
                    ;;
            esac
        fi
        
        sleep 10
        ((wait_time+=10))
    done
    
    error "$service failed health check after ${max_wait}s"
    return 1
}

# Post-deployment validation
validate_deployment() {
    log "Validating deployment..."
    
    local endpoints=(
        "http://localhost:8080/health:QuDAG RPC"
        "http://localhost:8081/health:Neural Mesh API"
        "http://localhost:3000/health:MCP Server"
        "http://localhost:9090/metrics:Metrics"
    )
    
    for endpoint_info in "${endpoints[@]}"; do
        IFS=':' read -r url description <<< "$endpoint_info"
        
        log "Testing $description ($url)..."
        
        local max_attempts=30
        local attempt=1
        
        while [[ $attempt -le $max_attempts ]]; do
            if curl -f -s --max-time 10 "$url" >/dev/null 2>&1; then
                success "$description is responding"
                break
            fi
            
            if [[ $attempt -eq $max_attempts ]]; then
                error "$description failed validation"
                return 1
            fi
            
            log "Attempt $attempt/$max_attempts failed, retrying in 5s..."
            sleep 5
            ((attempt++))
        done
    done
    
    # Test mesh connectivity
    log "Testing mesh connectivity..."
    if ! test_mesh_connectivity; then
        error "Mesh connectivity test failed"
        return 1
    fi
    
    success "Deployment validation completed"
}

# Test mesh connectivity
test_mesh_connectivity() {
    local response
    if response=$(curl -s --max-time 10 "http://localhost:8080/rpc" \
                       -H "Content-Type: application/json" \
                       -d '{"method":"get_peer_count","params":[],"id":1}' 2>/dev/null); then
        
        local peer_count=$(echo "$response" | jq -r '.result // 0' 2>/dev/null || echo "0")
        if [[ "$peer_count" -gt 0 ]]; then
            log "Mesh connectivity OK: $peer_count peers connected"
            return 0
        else
            warn "No peers connected yet"
            return 1
        fi
    else
        warn "Failed to query peer count"
        return 1
    fi
}

# Rollback deployment
rollback_deployment() {
    log "Rolling back deployment..."
    
    local backup_path
    if [[ -f /tmp/neural-mesh-backup-path ]]; then
        backup_path=$(cat /tmp/neural-mesh-backup-path)
    else
        # Find latest backup
        backup_path=$(find "$PROJECT_ROOT/backups" -type d -name "*_*" | sort -r | head -1)
    fi
    
    if [[ -z "$backup_path" || ! -d "$backup_path" ]]; then
        error "No backup found for rollback"
        return 1
    fi
    
    log "Rolling back to backup: $backup_path"
    
    # Stop current services
    docker-compose down
    
    # Restore volumes
    if [[ -f "$backup_path/neural-node-1-data.tar.gz" ]]; then
        log "Restoring data volumes..."
        docker run --rm \
            -v neural-node-1-data:/target \
            -v "$backup_path:/backup:ro" \
            alpine:latest \
            tar xzf /backup/neural-node-1-data.tar.gz -C /target
    fi
    
    # Restore configuration
    if [[ -f "$backup_path/config.tar.gz" ]]; then
        log "Restoring configuration..."
        tar xzf "$backup_path/config.tar.gz" -C "$PROJECT_ROOT"
    fi
    
    # Restore database
    if [[ -f "$backup_path/database.sql" ]]; then
        log "Restoring database..."
        docker-compose up -d postgres
        sleep 10
        docker-compose exec -T postgres psql -U neural_mesh -d neural_mesh < "$backup_path/database.sql"
    fi
    
    # Start services
    docker-compose up -d
    
    # Wait for health
    sleep 30
    if validate_deployment; then
        success "Rollback completed successfully"
    else
        error "Rollback validation failed"
        return 1
    fi
}

# Cleanup old backups
cleanup_old_backups() {
    log "Cleaning up old backups..."
    
    local backup_base="$PROJECT_ROOT/backups"
    if [[ -d "$backup_base" ]]; then
        find "$backup_base" -type d -name "*_*" -mtime +$BACKUP_RETENTION_DAYS -exec rm -rf {} + 2>/dev/null || true
        
        local remaining_backups=$(find "$backup_base" -type d -name "*_*" | wc -l)
        log "Cleanup completed. Remaining backups: $remaining_backups"
    fi
}

# Send deployment notification
send_notification() {
    local status="$1"
    local message="$2"
    
    # Slack webhook (if configured)
    if [[ -n "${SLACK_WEBHOOK_URL:-}" ]]; then
        curl -X POST -H 'Content-type: application/json' \
             --data "{\"text\":\"🤖 Neural Mesh Deployment: $status - $message\"}" \
             "$SLACK_WEBHOOK_URL" >/dev/null 2>&1 || true
    fi
    
    # Discord webhook (if configured)
    if [[ -n "${DISCORD_WEBHOOK_URL:-}" ]]; then
        curl -X POST -H 'Content-type: application/json' \
             --data "{\"content\":\"🤖 Neural Mesh Deployment: $status - $message\"}" \
             "$DISCORD_WEBHOOK_URL" >/dev/null 2>&1 || true
    fi
}

# Main deployment function
main() {
    local start_time=$(date +%s)
    
    log "Starting Synaptic Neural Mesh deployment"
    log "Environment: $DEPLOY_ENV"
    log "Timestamp: $(date)"
    
    validate_environment
    backup_deployment
    build_images
    rolling_deployment
    validate_deployment
    cleanup_old_backups
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    success "Deployment completed successfully in ${duration}s"
    send_notification "SUCCESS" "Deployment completed in ${duration}s"
    
    # Display post-deployment information
    log "Deployment Summary:"
    log "  - QuDAG RPC: http://localhost:8080"
    log "  - Neural Mesh API: http://localhost:8081"
    log "  - MCP Server: http://localhost:3000"
    log "  - Metrics: http://localhost:9090"
    log "  - Grafana: http://localhost:3001"
    log "  - Load Balancer: http://localhost:80"
}

# Handle command line arguments
case "${1:-deploy}" in
    "deploy")
        main
        ;;
    "rollback")
        rollback_deployment
        ;;
    "validate")
        validate_deployment
        ;;
    "backup")
        backup_deployment
        ;;
    "cleanup")
        cleanup_old_backups
        ;;
    *)
        echo "Usage: $0 {deploy|rollback|validate|backup|cleanup}"
        echo ""
        echo "Commands:"
        echo "  deploy   - Perform full deployment (default)"
        echo "  rollback - Rollback to last backup"
        echo "  validate - Validate current deployment"
        echo "  backup   - Create backup only"
        echo "  cleanup  - Clean old backups"
        exit 1
        ;;
esac