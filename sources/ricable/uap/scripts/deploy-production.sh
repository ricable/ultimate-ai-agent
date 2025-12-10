#!/bin/bash
# scripts/deploy-production.sh
# Automated UAP Production Deployment Script
# Supports multi-cloud deployment with health checks and rollback

set -e

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_FILE="/tmp/uap-deploy-$(date +%Y%m%d-%H%M%S).log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    local level=$1
    shift
    local message="$@"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "${timestamp} [${level}] ${message}" | tee -a "$LOG_FILE"
}

info() { log "${BLUE}INFO${NC}" "$@"; }
warn() { log "${YELLOW}WARN${NC}" "$@"; }
error() { log "${RED}ERROR${NC}" "$@"; }
success() { log "${GREEN}SUCCESS${NC}" "$@"; }

# Help function
show_help() {
    cat << EOF
UAP Production Deployment Script

Usage: $0 [OPTIONS]

Options:
    -c, --cloud CLOUD       Target cloud provider (aws|gcp|azure|auto|cost-optimized)
    -e, --env ENV          Environment (production|staging) [default: production]
    -r, --region REGION    Target region (optional, auto-selected if not specified)
    -t, --test            Run deployment tests before actual deployment
    -d, --dry-run         Show what would be deployed without actually deploying
    -f, --force           Force deployment even if health checks fail
    -b, --backup          Create backup before deployment
    -m, --monitor         Enable monitoring setup
    -h, --help            Show this help message

Examples:
    $0 --cloud gcp --env production --test
    $0 --cloud cost-optimized --backup --monitor
    $0 --cloud auto --region us-west-2 --dry-run

EOF
}

# Default values
CLOUD="auto"
ENVIRONMENT="production"
REGION=""
RUN_TESTS=false
DRY_RUN=false
FORCE=false
CREATE_BACKUP=false
ENABLE_MONITORING=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--cloud)
            CLOUD="$2"
            shift 2
            ;;
        -e|--env)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -r|--region)
            REGION="$2"
            shift 2
            ;;
        -t|--test)
            RUN_TESTS=true
            shift
            ;;
        -d|--dry-run)
            DRY_RUN=true
            shift
            ;;
        -f|--force)
            FORCE=true
            shift
            ;;
        -b|--backup)
            CREATE_BACKUP=true
            shift
            ;;
        -m|--monitor)
            ENABLE_MONITORING=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Validate inputs
if [[ ! "$CLOUD" =~ ^(aws|gcp|azure|auto|cost-optimized)$ ]]; then
    error "Invalid cloud provider: $CLOUD"
    exit 1
fi

if [[ ! "$ENVIRONMENT" =~ ^(production|staging)$ ]]; then
    error "Invalid environment: $ENVIRONMENT"
    exit 1
fi

# Pre-deployment checks
check_dependencies() {
    info "Checking dependencies..."
    
    local deps=("sky" "teller" "docker" "curl" "jq")
    for dep in "${deps[@]}"; do
        if ! command -v "$dep" &> /dev/null; then
            error "Required dependency not found: $dep"
            exit 1
        fi
    done
    
    # Check Python and Node versions
    if ! python3 --version | grep -q "3.11"; then
        warn "Python 3.11 not found, using $(python3 --version)"
    fi
    
    if ! node --version | grep -q "v20"; then
        warn "Node.js 20 not found, using $(node --version)"
    fi
    
    success "All dependencies checked"
}

# Check secrets availability
check_secrets() {
    info "Checking secrets configuration..."
    
    if [[ ! -f "$PROJECT_ROOT/.teller.yml" ]]; then
        error "Teller configuration not found"
        exit 1
    fi
    
    # Test teller connection
    if ! teller run echo "Secrets test" &> /dev/null; then
        error "Failed to connect to secrets provider"
        exit 1
    fi
    
    success "Secrets configuration verified"
}

# Run tests if requested
run_tests() {
    if [[ "$RUN_TESTS" == "true" ]]; then
        info "Running pre-deployment tests..."
        
        cd "$PROJECT_ROOT"
        
        # Backend tests
        info "Running backend tests..."
        cd backend
        python -m pytest tests/ -v --tb=short || {
            error "Backend tests failed"
            exit 1
        }
        cd ..
        
        # Frontend tests
        info "Running frontend tests..."
        cd frontend
        npm test || {
            error "Frontend tests failed"
            exit 1
        }
        cd ..
        
        success "All tests passed"
    fi
}

# Create backup if requested
create_backup() {
    if [[ "$CREATE_BACKUP" == "true" ]]; then
        info "Creating backup..."
        
        local backup_dir="/tmp/uap-backup-$(date +%Y%m%d-%H%M%S)"
        mkdir -p "$backup_dir"
        
        # Backup configuration files
        cp -r "$PROJECT_ROOT"/{.env*,skypilot,scripts,docker-compose*.yml} "$backup_dir/" 2>/dev/null || true
        
        # Backup database if accessible
        if command -v pg_dump &> /dev/null && [[ -n "$DATABASE_URL" ]]; then
            pg_dump "$DATABASE_URL" > "$backup_dir/database.sql" || warn "Database backup failed"
        fi
        
        info "Backup created at: $backup_dir"
    fi
}

# Select optimal cloud configuration
select_cloud_config() {
    local config_file=""
    
    case "$CLOUD" in
        "aws")
            config_file="skypilot/uap-aws.yaml"
            ;;
        "gcp")
            config_file="skypilot/uap-gcp.yaml"
            ;;
        "azure")
            config_file="skypilot/uap-azure.yaml"
            ;;
        "cost-optimized")
            config_file="skypilot/uap-cost-optimized.yaml"
            ;;
        "auto")
            # Use the general production config that supports multi-cloud
            config_file="skypilot/uap-production.yaml"
            ;;
    esac
    
    if [[ ! -f "$PROJECT_ROOT/$config_file" ]]; then
        error "Configuration file not found: $config_file"
        exit 1
    fi
    
    echo "$config_file"
}

# Deploy to cloud
deploy_to_cloud() {
    local config_file=$(select_cloud_config)
    
    info "Deploying UAP to cloud using configuration: $config_file"
    
    cd "$PROJECT_ROOT"
    
    # Prepare deployment command
    local deploy_cmd="sky up -c $config_file"
    
    if [[ -n "$REGION" ]]; then
        deploy_cmd="$deploy_cmd --region $REGION"
    fi
    
    if [[ "$DRY_RUN" == "true" ]]; then
        info "DRY RUN: Would execute: $deploy_cmd"
        return 0
    fi
    
    # Execute deployment
    info "Executing deployment..."
    if teller run -- $deploy_cmd; then
        success "Deployment completed successfully"
    else
        error "Deployment failed"
        return 1
    fi
}

# Health check function
check_deployment_health() {
    info "Performing deployment health checks..."
    
    # Get cluster status
    local cluster_info=$(sky status --refresh 2>/dev/null | grep uap || echo "")
    
    if [[ -z "$cluster_info" ]]; then
        error "No UAP cluster found"
        return 1
    fi
    
    # Extract cluster details
    local cluster_ip=$(echo "$cluster_info" | awk '{print $3}')
    
    if [[ "$cluster_ip" == "-" ]] || [[ -z "$cluster_ip" ]]; then
        error "Cluster IP not available"
        return 1
    fi
    
    info "Cluster IP: $cluster_ip"
    
    # Wait for services to be ready
    info "Waiting for services to be ready..."
    local max_attempts=30
    local attempt=0
    
    while [[ $attempt -lt $max_attempts ]]; do
        if curl -f -m 10 "http://$cluster_ip:8000/health" &>/dev/null; then
            success "Health check passed"
            return 0
        fi
        
        attempt=$((attempt + 1))
        info "Health check attempt $attempt/$max_attempts failed, retrying in 10 seconds..."
        sleep 10
    done
    
    error "Health check failed after $max_attempts attempts"
    return 1
}

# Setup monitoring
setup_monitoring() {
    if [[ "$ENABLE_MONITORING" == "true" ]]; then
        info "Setting up monitoring..."
        
        # Deploy monitoring stack if not already deployed
        if [[ -f "$PROJECT_ROOT/docker-compose.monitoring.yml" ]]; then
            docker-compose -f "$PROJECT_ROOT/docker-compose.monitoring.yml" up -d
            success "Monitoring stack deployed"
        else
            warn "Monitoring configuration not found, skipping"
        fi
    fi
}

# Rollback function
rollback_deployment() {
    error "Deployment failed, attempting rollback..."
    
    # Stop current deployment
    sky down -y uap 2>/dev/null || true
    
    # If backup was created, suggest restore
    if [[ "$CREATE_BACKUP" == "true" ]]; then
        info "Backup is available for manual restore if needed"
    fi
    
    error "Rollback completed"
}

# Main deployment function
main() {
    info "Starting UAP deployment"
    info "Configuration:"
    info "  Cloud: $CLOUD"
    info "  Environment: $ENVIRONMENT"
    info "  Region: ${REGION:-auto}"
    info "  Tests: $RUN_TESTS"
    info "  Dry Run: $DRY_RUN"
    info "  Backup: $CREATE_BACKUP"
    info "  Monitoring: $ENABLE_MONITORING"
    
    # Pre-deployment steps
    check_dependencies
    check_secrets
    run_tests
    create_backup
    
    # Main deployment
    if deploy_to_cloud; then
        if [[ "$DRY_RUN" == "false" ]]; then
            if check_deployment_health || [[ "$FORCE" == "true" ]]; then
                setup_monitoring
                success "UAP deployment completed successfully!"
                
                # Show access information
                local cluster_info=$(sky status --refresh 2>/dev/null | grep uap || echo "")
                local cluster_ip=$(echo "$cluster_info" | awk '{print $3}')
                
                if [[ -n "$cluster_ip" ]] && [[ "$cluster_ip" != "-" ]]; then
                    info "Access your UAP deployment at:"
                    info "  API: http://$cluster_ip:8000"
                    info "  Docs: http://$cluster_ip:8000/docs"
                    info "  Health: http://$cluster_ip:8000/health"
                fi
            else
                rollback_deployment
                exit 1
            fi
        fi
    else
        rollback_deployment
        exit 1
    fi
}

# Trap errors and cleanup
trap 'error "Deployment interrupted"; exit 1' INT TERM

# Run main function
main

info "Deployment log saved to: $LOG_FILE"