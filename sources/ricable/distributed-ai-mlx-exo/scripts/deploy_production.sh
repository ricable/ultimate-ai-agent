#!/bin/bash

#
# Production Deployment Script for MLX-Exo Distributed Cluster
# Provides automated deployment, configuration management, and rolling updates
#

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG_DIR="$PROJECT_ROOT/config"
LOGS_DIR="$PROJECT_ROOT/logs"

# Default configuration
DEFAULT_CLUSTER_NAME="mlx-prod-cluster"
DEFAULT_ENVIRONMENT="production"
DEFAULT_NODES_FILE="$CONFIG_DIR/cluster_nodes.json"
DEFAULT_SSH_USER="mlx"
DEFAULT_SSH_KEY="$HOME/.ssh/mlx_cluster"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging
LOG_FILE="$LOGS_DIR/deployment_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$LOGS_DIR"

log() {
    local level=$1
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] [$level] $message" | tee -a "$LOG_FILE"
}

info() {
    echo -e "${BLUE}[INFO]${NC} $*" | tee -a "$LOG_FILE"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $*" | tee -a "$LOG_FILE"
}

warn() {
    echo -e "${YELLOW}[WARNING]${NC} $*" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[ERROR]${NC} $*" | tee -a "$LOG_FILE"
}

fatal() {
    error "$*"
    exit 1
}

# Help function
show_help() {
    cat << EOF
MLX-Exo Distributed Cluster Production Deployment Script

USAGE:
    $0 [OPTIONS] COMMAND

COMMANDS:
    deploy          Full deployment of the cluster
    update          Rolling update of cluster components
    rollback        Rollback to previous version
    validate        Validate cluster configuration
    status          Check deployment status
    backup          Backup cluster configuration and data
    restore         Restore from backup
    scale           Scale cluster (add/remove nodes)
    
OPTIONS:
    -e, --environment ENV       Environment (default: production)
    -n, --nodes FILE           Nodes configuration file
    -c, --cluster-name NAME    Cluster name (default: mlx-prod-cluster)
    -u, --ssh-user USER        SSH user (default: mlx)
    -k, --ssh-key PATH         SSH private key path
    -f, --force                Force deployment without confirmations
    -v, --verbose              Verbose output
    -h, --help                 Show this help

EXAMPLES:
    $0 deploy                                    # Deploy with defaults
    $0 -e staging deploy                         # Deploy to staging
    $0 -n custom_nodes.json deploy              # Deploy with custom nodes
    $0 update --component api-server            # Update only API server
    $0 rollback --version v1.2.3                # Rollback to specific version
    $0 scale --add-node 10.0.1.15               # Add a new node

ENVIRONMENT VARIABLES:
    MLX_CLUSTER_NAME        Override cluster name
    MLX_ENVIRONMENT         Override environment
    MLX_SSH_USER           Override SSH user
    MLX_SSH_KEY            Override SSH key path
    MLX_FORCE_DEPLOY       Set to 'true' to force deployment

EOF
}

# Parse command line arguments
parse_args() {
    CLUSTER_NAME="${MLX_CLUSTER_NAME:-$DEFAULT_CLUSTER_NAME}"
    ENVIRONMENT="${MLX_ENVIRONMENT:-$DEFAULT_ENVIRONMENT}"
    NODES_FILE="${MLX_NODES_FILE:-$DEFAULT_NODES_FILE}"
    SSH_USER="${MLX_SSH_USER:-$DEFAULT_SSH_USER}"
    SSH_KEY="${MLX_SSH_KEY:-$DEFAULT_SSH_KEY}"
    FORCE_DEPLOY="${MLX_FORCE_DEPLOY:-false}"
    VERBOSE=false
    COMMAND=""
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            -e|--environment)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -n|--nodes)
                NODES_FILE="$2"
                shift 2
                ;;
            -c|--cluster-name)
                CLUSTER_NAME="$2"
                shift 2
                ;;
            -u|--ssh-user)
                SSH_USER="$2"
                shift 2
                ;;
            -k|--ssh-key)
                SSH_KEY="$2"
                shift 2
                ;;
            -f|--force)
                FORCE_DEPLOY=true
                shift
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            deploy|update|rollback|validate|status|backup|restore|scale)
                if [[ -n "$COMMAND" ]]; then
                    fatal "Multiple commands specified. Use only one."
                fi
                COMMAND="$1"
                shift
                ;;
            --*)
                # Additional options for specific commands
                case $COMMAND in
                    update)
                        if [[ "$1" == "--component" ]]; then
                            UPDATE_COMPONENT="$2"
                            shift 2
                        else
                            fatal "Unknown option for update: $1"
                        fi
                        ;;
                    rollback)
                        if [[ "$1" == "--version" ]]; then
                            ROLLBACK_VERSION="$2"
                            shift 2
                        else
                            fatal "Unknown option for rollback: $1"
                        fi
                        ;;
                    scale)
                        if [[ "$1" == "--add-node" ]]; then
                            ADD_NODE="$2"
                            shift 2
                        elif [[ "$1" == "--remove-node" ]]; then
                            REMOVE_NODE="$2"
                            shift 2
                        else
                            fatal "Unknown option for scale: $1"
                        fi
                        ;;
                    *)
                        fatal "Unknown option: $1"
                        ;;
                esac
                ;;
            *)
                fatal "Unknown argument: $1"
                ;;
        esac
    done
    
    if [[ -z "$COMMAND" ]]; then
        fatal "No command specified. Use --help for usage information."
    fi
}

# Load and validate configuration
load_config() {
    info "Loading configuration..."
    
    if [[ ! -f "$NODES_FILE" ]]; then
        fatal "Nodes configuration file not found: $NODES_FILE"
    fi
    
    # Validate JSON format
    if ! jq empty "$NODES_FILE" 2>/dev/null; then
        fatal "Invalid JSON in nodes configuration file: $NODES_FILE"
    fi
    
    # Extract node information
    NODES=($(jq -r '.nodes[].ip' "$NODES_FILE"))
    NODE_IDS=($(jq -r '.nodes[].id' "$NODES_FILE"))
    
    if [[ ${#NODES[@]} -eq 0 ]]; then
        fatal "No nodes defined in configuration file"
    fi
    
    info "Loaded configuration for ${#NODES[@]} nodes"
    if [[ "$VERBOSE" == "true" ]]; then
        for i in "${!NODES[@]}"; do
            info "  Node ${NODE_IDS[$i]}: ${NODES[$i]}"
        done
    fi
}

# SSH connection test
test_ssh_connectivity() {
    info "Testing SSH connectivity to all nodes..."
    
    local failed_nodes=()
    
    for i in "${!NODES[@]}"; do
        local node="${NODES[$i]}"
        local node_id="${NODE_IDS[$i]}"
        
        if ssh -i "$SSH_KEY" -o ConnectTimeout=10 -o BatchMode=yes \
           "$SSH_USER@$node" "echo 'SSH test successful'" &>/dev/null; then
            success "SSH connection to $node_id ($node) successful"
        else
            error "SSH connection to $node_id ($node) failed"
            failed_nodes+=("$node_id")
        fi
    done
    
    if [[ ${#failed_nodes[@]} -gt 0 ]]; then
        fatal "SSH connectivity failed for nodes: ${failed_nodes[*]}"
    fi
}

# Check prerequisites on nodes
check_prerequisites() {
    info "Checking prerequisites on all nodes..."
    
    local check_script=$(cat << 'EOF'
#!/bin/bash
echo "=== System Information ==="
echo "OS: $(uname -s)"
echo "Architecture: $(uname -m)" 
echo "OS Version: $(sw_vers -productVersion 2>/dev/null || echo 'Unknown')"
echo "Available Memory: $(free -h 2>/dev/null | grep Mem || echo 'N/A')"
echo "Disk Space: $(df -h / | tail -1)"

echo -e "\n=== Required Software ==="
check_command() {
    if command -v "$1" &> /dev/null; then
        echo "$1: $(command -v "$1") - $($1 --version 2>/dev/null | head -1 || echo 'Installed')"
    else
        echo "$1: NOT FOUND"
        return 1
    fi
}

missing=0
check_command python3 || missing=$((missing + 1))
check_command git || missing=$((missing + 1))
check_command docker || missing=$((missing + 1))

if [ -d "$HOME/mlx-exo-env" ]; then
    echo "Python virtual environment: EXISTS"
else
    echo "Python virtual environment: NOT FOUND"
    missing=$((missing + 1))
fi

echo -e "\n=== MLX and Exo ==="
if [ -f "$HOME/mlx-exo-env/bin/activate" ]; then
    source "$HOME/mlx-exo-env/bin/activate"
    python -c "import mlx; print('MLX version:', mlx.__version__)" 2>/dev/null || echo "MLX: NOT INSTALLED"
    python -c "import exo; print('Exo: INSTALLED')" 2>/dev/null || echo "Exo: NOT INSTALLED"
    deactivate
fi

exit $missing
EOF
)
    
    local failed_nodes=()
    
    for i in "${!NODES[@]}"; do
        local node="${NODES[$i]}"
        local node_id="${NODE_IDS[$i]}"
        
        info "Checking prerequisites on $node_id..."
        
        if ssh -i "$SSH_KEY" "$SSH_USER@$node" "$check_script"; then
            success "Prerequisites check passed for $node_id"
        else
            error "Prerequisites check failed for $node_id"
            failed_nodes+=("$node_id")
        fi
    done
    
    if [[ ${#failed_nodes[@]} -gt 0 ]]; then
        if [[ "$FORCE_DEPLOY" == "true" ]]; then
            warn "Prerequisites check failed for nodes: ${failed_nodes[*]} (continuing due to --force)"
        else
            fatal "Prerequisites check failed for nodes: ${failed_nodes[*]}. Use --force to continue anyway."
        fi
    fi
}

# Deploy application code
deploy_code() {
    info "Deploying application code to all nodes..."
    
    local deployment_archive="/tmp/mlx-cluster-deployment-$(date +%s).tar.gz"
    
    # Create deployment archive
    info "Creating deployment archive..."
    tar -czf "$deployment_archive" \
        --exclude='.git' \
        --exclude='logs' \
        --exclude='__pycache__' \
        --exclude='*.pyc' \
        --exclude='.pytest_cache' \
        -C "$PROJECT_ROOT" .
    
    # Deploy to each node
    for i in "${!NODES[@]}"; do
        local node="${NODES[$i]}"
        local node_id="${NODE_IDS[$i]}"
        
        info "Deploying code to $node_id..."
        
        # Create deployment directory
        ssh -i "$SSH_KEY" "$SSH_USER@$node" "mkdir -p /opt/mlx-cluster"
        
        # Transfer and extract code
        scp -i "$SSH_KEY" "$deployment_archive" "$SSH_USER@$node:/tmp/"
        ssh -i "$SSH_KEY" "$SSH_USER@$node" \
            "cd /opt/mlx-cluster && tar -xzf /tmp/$(basename "$deployment_archive") && rm /tmp/$(basename "$deployment_archive")"
        
        success "Code deployed to $node_id"
    done
    
    # Cleanup
    rm -f "$deployment_archive"
}

# Install dependencies
install_dependencies() {
    info "Installing dependencies on all nodes..."
    
    local install_script=$(cat << 'EOF'
#!/bin/bash
set -e

# Activate virtual environment
source "$HOME/mlx-exo-env/bin/activate"

# Update pip
pip install --upgrade pip

# Install requirements
cd /opt/mlx-cluster
if [ -f requirements.txt ]; then
    pip install -r requirements.txt
fi

# Install MLX and Exo if not present
pip install "mlx>=0.22.1" "mlx-lm>=0.21.1" || echo "MLX installation failed"

# Install/update Exo
if [ ! -d "$HOME/exo" ]; then
    git clone https://github.com/exo-explore/exo.git "$HOME/exo"
fi
cd "$HOME/exo"
git pull
pip install -e .

echo "Dependencies installation completed"
EOF
)
    
    for i in "${!NODES[@]}"; do
        local node="${NODES[$i]}"
        local node_id="${NODE_IDS[$i]}"
        
        info "Installing dependencies on $node_id..."
        
        if ssh -i "$SSH_KEY" "$SSH_USER@$node" "$install_script"; then
            success "Dependencies installed on $node_id"
        else
            error "Failed to install dependencies on $node_id"
        fi
    done
}

# Configure services
configure_services() {
    info "Configuring services on all nodes..."
    
    # Generate systemd service files
    local api_server_service=$(cat << EOF
[Unit]
Description=MLX Cluster API Server
After=network.target
Requires=network.target

[Service]
Type=simple
User=$SSH_USER
WorkingDirectory=/opt/mlx-cluster
Environment=PATH=/home/$SSH_USER/mlx-exo-env/bin:/usr/local/bin:/usr/bin:/bin
ExecStart=/home/$SSH_USER/mlx-exo-env/bin/python src/enhanced_api_server.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF
)
    
    local health_monitor_service=$(cat << EOF
[Unit]
Description=MLX Cluster Health Monitor
After=network.target
Requires=network.target

[Service]
Type=simple
User=$SSH_USER
WorkingDirectory=/opt/mlx-cluster
Environment=PATH=/home/$SSH_USER/mlx-exo-env/bin:/usr/local/bin:/usr/bin:/bin
ExecStart=/home/$SSH_USER/mlx-exo-env/bin/python -m src.monitoring.health_monitor
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF
)
    
    for i in "${!NODES[@]}"; do
        local node="${NODES[$i]}"
        local node_id="${NODE_IDS[$i]}"
        
        info "Configuring services on $node_id..."
        
        # Install service files
        echo "$api_server_service" | ssh -i "$SSH_KEY" "$SSH_USER@$node" \
            "sudo tee /etc/systemd/system/mlx-api-server.service > /dev/null"
            
        echo "$health_monitor_service" | ssh -i "$SSH_KEY" "$SSH_USER@$node" \
            "sudo tee /etc/systemd/system/mlx-health-monitor.service > /dev/null"
        
        # Reload systemd and enable services
        ssh -i "$SSH_KEY" "$SSH_USER@$node" "
            sudo systemctl daemon-reload
            sudo systemctl enable mlx-api-server.service
            sudo systemctl enable mlx-health-monitor.service
        "
        
        success "Services configured on $node_id"
    done
}

# Start services
start_services() {
    info "Starting services on all nodes..."
    
    for i in "${!NODES[@]}"; do
        local node="${NODES[$i]}"
        local node_id="${NODE_IDS[$i]}"
        
        info "Starting services on $node_id..."
        
        ssh -i "$SSH_KEY" "$SSH_USER@$node" "
            sudo systemctl start mlx-health-monitor.service
            sleep 5
            sudo systemctl start mlx-api-server.service
            
            # Wait for services to start
            sleep 10
            
            # Check service status
            sudo systemctl is-active mlx-health-monitor.service
            sudo systemctl is-active mlx-api-server.service
        "
        
        success "Services started on $node_id"
    done
}

# Validate deployment
validate_deployment() {
    info "Validating deployment..."
    
    # Wait for cluster to stabilize
    info "Waiting for cluster to stabilize..."
    sleep 30
    
    # Test each node
    local failed_validations=()
    
    for i in "${!NODES[@]}"; do
        local node="${NODES[$i]}"
        local node_id="${NODE_IDS[$i]}"
        
        info "Validating node $node_id..."
        
        # Test health endpoint
        if curl -f -s "http://$node:52415/health" > /dev/null; then
            success "Health endpoint responding on $node_id"
        else
            error "Health endpoint not responding on $node_id"
            failed_validations+=("$node_id-health")
        fi
        
        # Test models endpoint
        if curl -f -s "http://$node:52415/v1/models" > /dev/null; then
            success "Models endpoint responding on $node_id"
        else
            error "Models endpoint not responding on $node_id"
            failed_validations+=("$node_id-models")
        fi
    done
    
    # Run integration tests
    info "Running integration tests..."
    if cd "$PROJECT_ROOT" && python -m pytest tests/test_system_integration.py -v; then
        success "Integration tests passed"
    else
        error "Integration tests failed"
        failed_validations+=("integration-tests")
    fi
    
    if [[ ${#failed_validations[@]} -gt 0 ]]; then
        error "Deployment validation failed: ${failed_validations[*]}"
        return 1
    else
        success "Deployment validation passed"
        return 0
    fi
}

# Main deployment function
deploy() {
    info "Starting deployment of $CLUSTER_NAME cluster ($ENVIRONMENT environment)"
    
    # Confirmation prompt
    if [[ "$FORCE_DEPLOY" != "true" ]]; then
        echo -e "\n${YELLOW}Deployment Configuration:${NC}"
        echo "  Cluster Name: $CLUSTER_NAME"
        echo "  Environment: $ENVIRONMENT"
        echo "  Nodes: ${#NODES[@]}"
        echo "  SSH User: $SSH_USER"
        echo -e "\n${YELLOW}This will deploy to the following nodes:${NC}"
        for i in "${!NODES[@]}"; do
            echo "  - ${NODE_IDS[$i]}: ${NODES[$i]}"
        done
        
        echo -e "\n${RED}WARNING: This will overwrite existing installations!${NC}"
        read -p "Continue with deployment? (yes/no): " -r
        if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
            info "Deployment cancelled by user"
            exit 0
        fi
    fi
    
    # Deployment steps
    test_ssh_connectivity
    check_prerequisites
    deploy_code
    install_dependencies
    configure_services
    start_services
    
    if validate_deployment; then
        success "Deployment completed successfully!"
        info "Cluster $CLUSTER_NAME is now running in $ENVIRONMENT environment"
        info "Access the API at: http://${NODES[0]}:52415"
        info "View logs with: journalctl -u mlx-api-server.service -f"
    else
        error "Deployment completed but validation failed!"
        exit 1
    fi
}

# Rolling update function
update() {
    info "Starting rolling update of $CLUSTER_NAME cluster"
    
    # Update one node at a time
    for i in "${!NODES[@]}"; do
        local node="${NODES[$i]}"
        local node_id="${NODE_IDS[$i]}"
        
        info "Updating node $node_id..."
        
        # Stop services
        ssh -i "$SSH_KEY" "$SSH_USER@$node" "
            sudo systemctl stop mlx-api-server.service
            sudo systemctl stop mlx-health-monitor.service
        "
        
        # Update code (reuse deploy_code for single node)
        NODES=("$node")
        NODE_IDS=("$node_id")
        deploy_code
        install_dependencies
        
        # Start services
        ssh -i "$SSH_KEY" "$SSH_USER@$node" "
            sudo systemctl start mlx-health-monitor.service
            sleep 5
            sudo systemctl start mlx-api-server.service
        "
        
        # Wait and validate
        sleep 15
        if curl -f -s "http://$node:52415/health" > /dev/null; then
            success "Node $node_id updated successfully"
        else
            error "Node $node_id update failed"
            # TODO: Implement rollback for failed node
        fi
        
        # Brief pause between nodes
        sleep 10
    done
    
    success "Rolling update completed"
}

# Status check function
status() {
    info "Checking status of $CLUSTER_NAME cluster"
    
    for i in "${!NODES[@]}"; do
        local node="${NODES[$i]}"
        local node_id="${NODE_IDS[$i]}"
        
        echo -e "\n${BLUE}Node: $node_id ($node)${NC}"
        
        # SSH connectivity
        if ssh -i "$SSH_KEY" -o ConnectTimeout=5 "$SSH_USER@$node" "echo 'SSH OK'" 2>/dev/null; then
            echo "  SSH: ✅ Connected"
        else
            echo "  SSH: ❌ Failed"
            continue
        fi
        
        # Service status
        local api_status=$(ssh -i "$SSH_KEY" "$SSH_USER@$node" "sudo systemctl is-active mlx-api-server.service" 2>/dev/null || echo "inactive")
        local health_status=$(ssh -i "$SSH_KEY" "$SSH_USER@$node" "sudo systemctl is-active mlx-health-monitor.service" 2>/dev/null || echo "inactive")
        
        echo "  API Server: $([ "$api_status" = "active" ] && echo "✅ Running" || echo "❌ $api_status")"
        echo "  Health Monitor: $([ "$health_status" = "active" ] && echo "✅ Running" || echo "❌ $health_status")"
        
        # API endpoint test
        if curl -f -s "http://$node:52415/health" > /dev/null 2>&1; then
            echo "  API Endpoint: ✅ Responding"
        else
            echo "  API Endpoint: ❌ Not responding"
        fi
        
        # System resources
        local cpu_usage=$(ssh -i "$SSH_KEY" "$SSH_USER@$node" "top -l 1 | grep 'CPU usage' | awk '{print \$3}'" 2>/dev/null || echo "N/A")
        local memory_usage=$(ssh -i "$SSH_KEY" "$SSH_USER@$node" "vm_stat | grep 'free\\|wired' | awk '{print \$3}' | tr -d '.' | awk 'NR==1{free=\$1} NR==2{wired=\$1} END{print int((wired/(free+wired))*100)}'" 2>/dev/null || echo "N/A")
        
        echo "  CPU Usage: $cpu_usage"
        echo "  Memory Usage: ${memory_usage}%"
    done
}

# Main execution
main() {
    # Parse command line arguments
    parse_args "$@"
    
    # Load configuration
    load_config
    
    # Execute command
    case $COMMAND in
        deploy)
            deploy
            ;;
        update)
            update
            ;;
        status)
            status
            ;;
        validate)
            validate_deployment
            ;;
        rollback)
            warn "Rollback functionality not yet implemented"
            ;;
        backup)
            warn "Backup functionality not yet implemented"
            ;;
        restore)
            warn "Restore functionality not yet implemented"
            ;;
        scale)
            warn "Scale functionality not yet implemented"
            ;;
        *)
            fatal "Unknown command: $COMMAND"
            ;;
    esac
}

# Run main function with all arguments
main "$@"