#!/bin/bash

# RAN Intelligent Automation System - Production Deployment Script
# Version: 1.0.0
# Description: Automated deployment with three-phase rollout strategy

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
K8S_DIR="${PROJECT_ROOT}/k8s"
LOG_FILE="${PROJECT_ROOT}/logs/deployment-$(date +%Y%m%d-%H%M%S).log"
BACKUP_DIR="${PROJECT_ROOT}/backups"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Global variables
NAMESPACE="ran-automation"
DRY_RUN=false
SKIP_VALIDATION=false
SKIP_BACKUP=false
ROLLBACK_MODE=false
DEPLOYMENT_PHASE="full"
ENVIRONMENT="production"

# Logging function
log() {
    local level=$1
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')

    case $level in
        "INFO")
            echo -e "${GREEN}[INFO]${NC} ${message}" | tee -a "$LOG_FILE"
            ;;
        "WARN")
            echo -e "${YELLOW}[WARN]${NC} ${message}" | tee -a "$LOG_FILE"
            ;;
        "ERROR")
            echo -e "${RED}[ERROR]${NC} ${message}" | tee -a "$LOG_FILE"
            ;;
        "DEBUG")
            echo -e "${BLUE}[DEBUG]${NC} ${message}" | tee -a "$LOG_FILE"
            ;;
    esac

    echo "[${timestamp}] [${level}] ${message}" >> "$LOG_FILE"
}

# Show usage
usage() {
    cat << EOF
RAN Intelligent Automation System - Production Deployment Script

Usage: $0 [OPTIONS]

OPTIONS:
    -e, --environment ENV     Set environment (default: production)
    -p, --phase PHASE        Deployment phase (canary|partial|full) (default: full)
    -n, --namespace NS       Set namespace (default: ran-automation)
    -d, --dry-run            Perform dry run without applying changes
    -s, --skip-validation    Skip pre-deployment validation
    -b, --skip-backup        Skip backup creation
    -r, --rollback           Rollback to previous deployment
    -h, --help              Show this help message

EXAMPLES:
    $0 --phase canary
    $0 --environment staging --phase partial
    $0 --dry-run --skip-validation
    $0 --rollback

EOF
    exit 1
}

# Parse command line arguments
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -e|--environment)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -p|--phase)
                DEPLOYMENT_PHASE="$2"
                if [[ ! "$DEPLOYMENT_PHASE" =~ ^(canary|partial|full)$ ]]; then
                    log "ERROR" "Invalid phase: $DEPLOYMENT_PHASE. Must be one of: canary, partial, full"
                    exit 1
                fi
                shift 2
                ;;
            -n|--namespace)
                NAMESPACE="$2"
                shift 2
                ;;
            -d|--dry-run)
                DRY_RUN=true
                shift
                ;;
            -s|--skip-validation)
                SKIP_VALIDATION=true
                shift
                ;;
            -b|--skip-backup)
                SKIP_BACKUP=true
                shift
                ;;
            -r|--rollback)
                ROLLBACK_MODE=true
                shift
                ;;
            -h|--help)
                usage
                ;;
            *)
                log "ERROR" "Unknown option: $1"
                usage
                ;;
        esac
    done
}

# Check prerequisites
check_prerequisites() {
    log "INFO" "Checking prerequisites..."

    # Check if kubectl is installed
    if ! command -v kubectl &> /dev/null; then
        log "ERROR" "kubectl is not installed"
        exit 1
    fi

    # Check if helm is installed
    if ! command -v helm &> /dev/null; then
        log "ERROR" "helm is not installed"
        exit 1
    fi

    # Check cluster connection
    if ! kubectl cluster-info &> /dev/null; then
        log "ERROR" "Cannot connect to Kubernetes cluster"
        exit 1
    fi

    # Check if cluster has sufficient resources
    local available_nodes=$(kubectl get nodes --no-headers | wc -l)
    if [[ $available_nodes -lt 3 ]]; then
        log "WARN" "Cluster has only $available_nodes nodes. Minimum 3 recommended for production"
    fi

    # Check if required storage classes exist
    if ! kubectl get storageclass fast-ssd &> /dev/null; then
        log "WARN" "fast-ssd storage class not found. Using default storage class"
    fi

    log "INFO" "Prerequisites check completed"
}

# Create backup of current deployment
create_backup() {
    if [[ "$SKIP_BACKUP" == true ]]; then
        log "INFO" "Skipping backup creation"
        return
    fi

    log "INFO" "Creating backup of current deployment..."

    local backup_name="backup-$(date +%Y%m%d-%H%M%S)"
    local backup_path="${BACKUP_DIR}/${backup_name}"

    mkdir -p "$backup_path"

    # Backup current resources
    kubectl get all -n "$NAMESPACE" -o yaml > "${backup_path}/current-deployment.yaml"
    kubectl get configmaps -n "$NAMESPACE" -o yaml > "${backup_path}/configmaps.yaml"
    kubectl get secrets -n "$NAMESPACE" -o yaml > "${backup_path}/secrets.yaml"
    kubectl get ingresses -n "$NAMESPACE" -o yaml > "${backup_path}/ingresses.yaml"

    # Backup ArgoCD applications if they exist
    if kubectl get applications -n argocd &> /dev/null; then
        kubectl get applications -n argocd -o yaml > "${backup_path}/argocd-apps.yaml"
    fi

    # Create backup metadata
    cat > "${backup_path}/metadata.yaml" << EOF
backup:
  name: $backup_name
  timestamp: $(date -Iseconds)
  environment: $ENVIRONMENT
  namespace: $NAMESPACE
  deployment_phase: $DEPLOYMENT_PHASE
  git_commit: $(git rev-parse HEAD 2>/dev/null || echo "unknown")
  kubernetes_context: $(kubectl config current-context)
EOF

    log "INFO" "Backup created at: $backup_path"
}

# Validate configuration files
validate_configuration() {
    if [[ "$SKIP_VALIDATION" == true ]]; then
        log "INFO" "Skipping configuration validation"
        return
    fi

    log "INFO" "Validating configuration files..."

    # Validate YAML syntax
    local yaml_files=(
        "${K8S_DIR}/namespaces/namespaces.yaml"
        "${K8S_DIR}/configmaps/configmaps.yaml"
        "${K8S_DIR}/deployments/ran-automation-services.yaml"
        "${K8S_DIR}/services/services.yaml"
        "${K8S_DIR}/ingress/ingress.yaml"
        "${K8S_DIR}/network-policies/network-policies.yaml"
        "${K8S_DIR}/resource-limits/resource-quotas.yaml"
    )

    for file in "${yaml_files[@]}"; do
        if [[ -f "$file" ]]; then
            if ! yq eval '.' "$file" > /dev/null 2>&1; then
                log "ERROR" "Invalid YAML syntax in $file"
                exit 1
            fi
        else
            log "WARN" "Configuration file not found: $file"
        fi
    done

    log "INFO" "Configuration validation completed"
}

# Pre-deployment health checks
health_checks() {
    log "INFO" "Running pre-deployment health checks..."

    # Check namespace exists or create it
    if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
        log "INFO" "Creating namespace: $NAMESPACE"
        if [[ "$DRY_RUN" == false ]]; then
            kubectl create namespace "$NAMESPACE"
        fi
    fi

    # Check if critical services are running
    local critical_services=("agentdb" "redis")
    for service in "${critical_services[@]}"; do
        if kubectl get deployment "$service-service" -n "$NAMESPACE" &> /dev/null; then
            local replicas=$(kubectl get deployment "$service-service" -n "$NAMESPACE" -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo "0")
            if [[ "$replicas" == "0" ]]; then
                log "WARN" "Critical service $service is not ready"
            fi
        fi
    done

    # Check resource availability
    local available_cpu=$(kubectl describe nodes | grep -i "cpu" | awk '{print $2}' | head -1 | sed 's/m//')
    local available_memory=$(kubectl describe nodes | grep -i "memory" | awk '{print $2}' | head -1 | sed 's/Ki//')

    log "INFO" "Available cluster resources: CPU: ${available_cpu}m, Memory: ${available_memory}Ki"

    log "INFO" "Health checks completed"
}

# Deploy based on phase
deploy_phase() {
    log "INFO" "Starting deployment phase: $DEPLOYMENT_PHASE"

    case "$DEPLOYMENT_PHASE" in
        "canary")
            deploy_canary
            ;;
        "partial")
            deploy_partial
            ;;
        "full")
            deploy_full
            ;;
    esac
}

# Canary deployment
deploy_canary() {
    log "INFO" "Starting canary deployment..."

    # Deploy canary configuration
    local canary_dir="${K8S_DIR}/canary"
    if [[ ! -d "$canary_dir" ]]; then
        mkdir -p "$canary_dir"
        # Create canary configuration from main config with reduced replicas
        yq eval '.spec.replicas = 1' "${K8S_DIR}/deployments/ran-automation-services.yaml" > "${canary_dir}/canary-deployment.yaml"
    fi

    # Apply canary deployment
    if [[ "$DRY_RUN" == false ]]; then
        kubectl apply -f "${canary_dir}" -n "$NAMESPACE"

        # Wait for canary to be ready
        log "INFO" "Waiting for canary deployment to be ready..."
        kubectl wait --for=condition=available --timeout=300s deployment/ran-automation-api -n "$NAMESPACE"

        # Monitor canary health
        monitor_canary_health
    else
        log "INFO" "DRY RUN: Would apply canary deployment"
    fi

    log "INFO" "Canary deployment completed"
}

# Partial deployment (25% of traffic)
deploy_partial() {
    log "INFO" "Starting partial deployment (25% traffic)..."

    # Update deployments to partial scale
    local partial_replicas=1

    if [[ "$DRY_RUN" == false ]]; then
        # Scale down current deployment
        kubectl scale deployment ran-automation-api --replicas=3 -n "$NAMESPACE"
        kubectl scale deployment cognitive-performance-service --replicas=1 -n "$NAMESPACE"
        kubectl scale deployment swarm-coordination-service --replicas=1 -n "$NAMESPACE"

        # Wait for scaling to complete
        kubectl wait --for=condition=available --timeout=300s deployment/ran-automation-api -n "$NAMESPACE"
        kubectl wait --for=condition=available --timeout=300s deployment/cognitive-performance-service -n "$NAMESPACE"
        kubectl wait --for=condition=available --timeout=300s deployment/swarm-coordination-service -n "$NAMESPACE"

        # Apply partial configuration
        kubectl apply -f "${K8S_DIR}/configmaps" -n "$NAMESPACE"
        kubectl apply -f "${K8S_DIR}/secrets" -n "$NAMESPACE"

        log "INFO" "Partial deployment completed"
    else
        log "INFO" "DRY RUN: Would apply partial deployment"
    fi
}

# Full deployment
deploy_full() {
    log "INFO" "Starting full deployment..."

    if [[ "$DRY_RUN" == false ]]; then
        # Apply all configurations in order
        local deployment_order=(
            "namespaces"
            "configmaps"
            "secrets"
            "resource-limits"
            "network-policies"
            "deployments"
            "services"
            "ingress"
        )

        for component in "${deployment_order[@]}"; do
            log "INFO" "Deploying $component..."
            kubectl apply -f "${K8S_DIR}/$component" -n "$NAMESPACE" || {
                log "ERROR" "Failed to deploy $component"
                exit 1
            }
        done

        # Wait for critical deployments to be ready
        log "INFO" "Waiting for deployments to be ready..."
        kubectl wait --for=condition=available --timeout=600s deployment/ran-automation-api -n "$NAMESPACE"
        kubectl wait --for=condition=available --timeout=600s deployment/agentdb-service -n "$NAMESPACE"
        kubectl wait --for=condition=available --timeout=600s deployment/cognitive-performance-service -n "$NAMESPACE"
        kubectl wait --for=condition=available --timeout=600s deployment/swarm-coordination-service -n "$NAMESPACE"

        log "INFO" "Full deployment completed"
    else
        log "INFO" "DRY RUN: Would apply full deployment"
    fi
}

# Monitor canary health
monitor_canary_health() {
    log "INFO" "Monitoring canary deployment health..."

    local canary_pod=$(kubectl get pods -n "$NAMESPACE" -l app=ran-automation,component=api,version=canary -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)

    if [[ -z "$canary_pod" ]]; then
        log "ERROR" "Canary pod not found"
        return 1
    fi

    # Check pod health
    local phase=$(kubectl get pod "$canary_pod" -n "$NAMESPACE" -o jsonpath='{.status.phase}')
    if [[ "$phase" != "Running" ]]; then
        log "ERROR" "Canary pod is not running: $phase"
        return 1
    fi

    # Check readiness
    local ready=$(kubectl get pod "$canary_pod" -n "$NAMESPACE" -o jsonpath='{.status.conditions[?(@.type=="Ready")].status}')
    if [[ "$ready" != "True" ]]; then
        log "ERROR" "Canary pod is not ready"
        return 1
    fi

    # Check application health endpoint
    local pod_ip=$(kubectl get pod "$canary_pod" -n "$NAMESPACE" -o jsonpath='{.status.podIP}')
    if curl -f --connect-timeout 10 --max-time 30 "http://${pod_ip}:8080/health/live" &> /dev/null; then
        log "INFO" "Canary health check passed"
        return 0
    else
        log "ERROR" "Canary health check failed"
        return 1
    fi
}

# Rollback to previous deployment
rollback_deployment() {
    log "INFO" "Starting rollback process..."

    # Find latest backup
    local latest_backup=$(ls -1t "${BACKUP_DIR}" | head -1)
    if [[ -z "$latest_backup" ]]; then
        log "ERROR" "No backup found for rollback"
        exit 1
    fi

    local backup_path="${BACKUP_DIR}/${latest_backup}"
    log "INFO" "Rolling back to backup: $latest_backup"

    if [[ "$DRY_RUN" == false ]]; then
        # Restore from backup
        kubectl apply -f "${backup_path}/current-deployment.yaml" -n "$NAMESPACE"
        kubectl apply -f "${backup_path}/configmaps.yaml" -n "$NAMESPACE"
        kubectl apply -f "${backup_path}/secrets.yaml" -n "$NAMESPACE"
        kubectl apply -f "${backup_path}/ingresses.yaml" -n "$NAMESPACE"

        log "INFO" "Rollback completed"
    else
        log "INFO" "DRY RUN: Would rollback to backup: $latest_backup"
    fi
}

# Post-deployment validation
post_deployment_validation() {
    log "INFO" "Running post-deployment validation..."

    # Check all deployments are ready
    local deployments=(
        "ran-automation-api"
        "agentdb-service"
        "cognitive-performance-service"
        "swarm-coordination-service"
        "redis-service"
    )

    for deployment in "${deployments[@]}"; do
        local replicas=$(kubectl get deployment "$deployment" -n "$NAMESPACE" -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo "0")
        local desired=$(kubectl get deployment "$deployment" -n "$NAMESPACE" -o jsonpath='{.spec.replicas}' 2>/dev/null || echo "0")

        if [[ "$replicas" != "$desired" ]]; then
            log "WARN" "Deployment $deployment not fully ready: $replicas/$desired replicas ready"
        fi
    done

    # Check services are accessible
    local services=(
        "ran-automation-api-service"
        "agentdb-service"
        "cognitive-performance-service"
        "swarm-coordination-service"
    )

    for service in "${services[@]}"; do
        if kubectl get service "$service" -n "$NAMESPACE" &> /dev/null; then
            log "INFO" "Service $service is accessible"
        else
            log "ERROR" "Service $service is not accessible"
        fi
    done

    # Check ingress if configured
    if kubectl get ingress ran-automation-ingress -n "$NAMESPACE" &> /dev/null; then
        local ingress_ip=$(kubectl get ingress ran-automation-ingress -n "$NAMESPACE" -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "pending")
        log "INFO" "Ingress IP: $ingress_ip"
    fi

    log "INFO" "Post-deployment validation completed"
}

# Generate deployment report
generate_report() {
    local report_file="${PROJECT_ROOT}/reports/deployment-report-$(date +%Y%m%d-%H%M%S).md"

    cat > "$report_file" << EOF
# RAN Intelligent Automation System - Deployment Report

## Deployment Summary
- **Timestamp**: $(date -Iseconds)
- **Environment**: $ENVIRONMENT
- **Namespace**: $NAMESPACE
- **Phase**: $DEPLOYMENT_PHASE
- **Mode**: $([ "$DRY_RUN" == true ] && echo "DRY RUN" || echo "PRODUCTION")
- **Git Commit**: $(git rev-parse HEAD 2>/dev/null || echo "unknown")

## Cluster Information
- **Kubernetes Context**: $(kubectl config current-context)
- **Kubernetes Version**: $(kubectl version --client -o json | jq -r '.clientVersion.gitVersion')
- **Nodes**: $(kubectl get nodes --no-headers | wc -l)

## Deployment Status
EOF

    # Add deployment status
    if [[ "$DRY_RUN" == false ]]; then
        kubectl get deployments -n "$NAMESPACE" -o custom-columns=NAME:.metadata.name,REPLICAS:.status.readyReplicas,DESIRED:.spec.replicas,READY:.status.conditions[?(@.type=="Available")].status >> "$report_file"
    else
        echo "DRY RUN - No actual deployment performed" >> "$report_file"
    fi

    cat >> "$report_file" << EOF

## Services Status
EOF

    kubectl get services -n "$NAMESPACE" >> "$report_file" 2>/dev/null || echo "No services found" >> "$report_file"

    cat >> "$report_file" << EOF

## Resource Usage
EOF

    kubectl top pods -n "$NAMESPACE" 2>/dev/null >> "$report_file" || echo "Resource metrics not available" >> "$report_file"

    cat >> "$report_file" << EOF

## Notes
- Deployment log: $LOG_FILE
- Backup location: $([ "$SKIP_BACKUP" == false ] && echo "${BACKUP_DIR}/backup-$(date +%Y%m%d-*)/*" || echo "No backup created")

---
*Generated by RAN Intelligent Automation System Deployment Script*
EOF

    log "INFO" "Deployment report generated: $report_file"
}

# Main deployment function
main() {
    log "INFO" "Starting RAN Intelligent Automation System deployment..."
    log "INFO" "Deployment parameters: Environment=$ENVIRONMENT, Phase=$DEPLOYMENT_PHASE, Namespace=$NAMESPACE"

    # Create necessary directories
    mkdir -p "$(dirname "$LOG_FILE")"
    mkdir -p "$BACKUP_DIR"
    mkdir -p "${PROJECT_ROOT}/reports"

    # Check prerequisites
    check_prerequisites

    # Parse arguments
    parse_arguments "$@"

    # Create backup unless in dry run mode or skipped
    if [[ "$DRY_RUN" == false && "$ROLLBACK_MODE" == false ]]; then
        create_backup
    fi

    # Validate configuration
    validate_configuration

    # Run health checks
    health_checks

    # Execute deployment or rollback
    if [[ "$ROLLBACK_MODE" == true ]]; then
        rollback_deployment
    else
        deploy_phase
    fi

    # Post-deployment validation (only for non-dry runs)
    if [[ "$DRY_RUN" == false ]]; then
        post_deployment_validation
    fi

    # Generate report
    generate_report

    log "INFO" "Deployment process completed successfully"
}

# Script entry point
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi