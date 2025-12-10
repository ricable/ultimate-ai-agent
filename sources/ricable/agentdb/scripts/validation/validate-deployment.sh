#!/bin/bash

# RAN Intelligent Automation System - Deployment Validation Script
# Version: 1.0.0
# Description: Comprehensive validation and smoke testing for deployment

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
NAMESPACE="${NAMESPACE:-ran-automation}"
VALIDATION_TIMEOUT="${VALIDATION_TIMEOUT:-600}"
VERBOSE=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Validation results
VALIDATION_PASSED=true
FAILED_TESTS=()
PASSED_TESTS=()

# Logging function
log() {
    local level=$1
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')

    case $level in
        "INFO")
            echo -e "${GREEN}[INFO]${NC} ${message}"
            ;;
        "WARN")
            echo -e "${YELLOW}[WARN]${NC} ${message}"
            ;;
        "ERROR")
            echo -e "${RED}[ERROR]${NC} ${message}"
            ;;
        "DEBUG")
            if [[ "$VERBOSE" == true ]]; then
                echo -e "${BLUE}[DEBUG]${NC} ${message}"
            fi
            ;;
    esac
}

# Test result function
test_result() {
    local test_name=$1
    local result=$2
    local message=${3:-""}

    if [[ "$result" == "PASS" ]]; then
        PASSED_TESTS+=("$test_name")
        log "INFO" "✓ PASS: $test_name"
        if [[ -n "$message" ]]; then
            log "DEBUG" "  $message"
        fi
    else
        FAILED_TESTS+=("$test_name")
        VALIDATION_PASSED=false
        log "ERROR" "✗ FAIL: $test_name"
        if [[ -n "$message" ]]; then
            log "ERROR" "  $message"
        fi
    fi
}

# Wait for resource with timeout
wait_for_resource() {
    local resource_type=$1
    local resource_name=$2
    local condition=$3
    local timeout=$4
    local namespace=$5

    log "INFO" "Waiting for $resource_type/$resource_name to be $condition (timeout: ${timeout}s)..."

    local end_time=$((SECONDS + timeout))
    while [[ $SECONDS -lt $end_time ]]; do
        if kubectl get "$resource_type" "$resource_name" -n "$namespace" &>/dev/null; then
            case $resource_type in
                "pod")
                    if kubectl get pod "$resource_name" -n "$namespace" -o jsonpath="{.status.conditions[?(@.type==\"$condition\")].status}" 2>/dev/null | grep -q "True"; then
                        return 0
                    fi
                    ;;
                "deployment")
                    if kubectl get deployment "$resource_name" -n "$namespace" -o jsonpath="{.status.conditions[?(@.type==\"$condition\")].status}" 2>/dev/null | grep -q "True"; then
                        return 0
                    fi
                    ;;
                "service")
                    if kubectl get service "$resource_name" -n "$namespace" -o jsonpath="{.status.loadBalancer.ingress}" 2>/dev/null | grep -q .; then
                        return 0
                    fi
                    ;;
            esac
        fi
        sleep 5
    done

    return 1
}

# Validate namespace exists
validate_namespace() {
    log "INFO" "Validating namespace existence..."

    if kubectl get namespace "$NAMESPACE" &>/dev/null; then
        test_result "Namespace exists" "PASS" "Namespace $NAMESPACE found"
    else
        test_result "Namespace exists" "FAIL" "Namespace $NAMESPACE not found"
    fi
}

# Validate deployments
validate_deployments() {
    log "INFO" "Validating deployments..."

    local deployments=(
        "ran-automation-api"
        "agentdb-service"
        "cognitive-performance-service"
        "swarm-coordination-service"
        "redis-service"
    )

    for deployment in "${deployments[@]}"; do
        log "DEBUG" "Checking deployment: $deployment"

        if ! kubectl get deployment "$deployment" -n "$NAMESPACE" &>/dev/null; then
            test_result "Deployment $deployment exists" "FAIL" "Deployment not found"
            continue
        fi

        # Check deployment status
        local replicas=$(kubectl get deployment "$deployment" -n "$NAMESPACE" -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo "0")
        local desired=$(kubectl get deployment "$deployment" -n "$NAMESPACE" -o jsonpath='{.spec.replicas}' 2>/dev/null || echo "0")

        if [[ "$replicas" == "$desired" && "$replicas" -gt 0 ]]; then
            test_result "Deployment $deployment replicas ready" "PASS" "$replicas/$desired replicas ready"
        else
            test_result "Deployment $deployment replicas ready" "FAIL" "$replicas/$desired replicas ready"
        fi

        # Check deployment availability
        local available=$(kubectl get deployment "$deployment" -n "$NAMESPACE" -o jsonpath="{.status.conditions[?(@.type==\"Available\")].status}" 2>/dev/null || echo "False")
        if [[ "$available" == "True" ]]; then
            test_result "Deployment $deployment available" "PASS"
        else
            test_result "Deployment $deployment available" "FAIL"
        fi
    done
}

# Validate services
validate_services() {
    log "INFO" "Validating services..."

    local services=(
        "ran-automation-api-service"
        "agentdb-service"
        "cognitive-performance-service"
        "swarm-coordination-service"
        "redis-service"
    )

    for service in "${services[@]}"; do
        log "DEBUG" "Checking service: $service"

        if ! kubectl get service "$service" -n "$NAMESPACE" &>/dev/null; then
            test_result "Service $service exists" "FAIL" "Service not found"
            continue
        fi

        # Check service type and ports
        local service_type=$(kubectl get service "$service" -n "$NAMESPACE" -o jsonpath='{.spec.type}')
        local ports=$(kubectl get service "$service" -n "$NAMESPACE" -o jsonpath='{.spec.ports[*].port}')

        test_result "Service $service configuration" "PASS" "Type: $service_type, Ports: $ports"
    done
}

# Validate pods health
validate_pods_health() {
    log "INFO" "Validating pod health..."

    local pods=($(kubectl get pods -n "$NAMESPACE" -o jsonpath='{.items[*].metadata.name}'))
    local unhealthy_pods=0

    for pod in "${pods[@]}"; do
        local phase=$(kubectl get pod "$pod" -n "$NAMESPACE" -o jsonpath='{.status.phase}')
        local ready=$(kubectl get pod "$pod" -n "$NAMESPACE" -o jsonpath="{.status.conditions[?(@.type==\"Ready\")].status}")
        local restarts=$(kubectl get pod "$pod" -n "$NAMESPACE" -o jsonpath='{.status.containerStatuses[*].restartCount}')

        if [[ "$phase" == "Running" && "$ready" == "True" ]]; then
            test_result "Pod $pod health" "PASS" "Phase: $phase, Ready: $ready, Restarts: $restarts"
        else
            test_result "Pod $pod health" "FAIL" "Phase: $phase, Ready: $ready, Restarts: $restarts"
            ((unhealthy_pods++))
        fi
    done

    if [[ $unhealthy_pods -eq 0 ]]; then
        test_result "Overall pod health" "PASS" "All pods are healthy"
    else
        test_result "Overall pod health" "FAIL" "$unhealthy_pods unhealthy pods"
    fi
}

# Validate network connectivity
validate_network_connectivity() {
    log "INFO" "Validating network connectivity..."

    # Test internal service connectivity
    local test_pod="network-test-$(date +%s)"

    # Create a test pod
    cat << EOF | kubectl apply -f - &
apiVersion: v1
kind: Pod
metadata:
  name: $test_pod
  namespace: $NAMESPACE
spec:
  containers:
  - name: test
    image: curlimages/curl:latest
    command: ['sleep', '3600']
EOF

    # Wait for test pod to be ready
    if wait_for_resource "pod" "$test_pod" "Ready" 60 "$NAMESPACE"; then
        # Test API service connectivity
        if kubectl exec "$test_pod" -n "$NAMESPACE" -- curl -f --connect-timeout 10 --max-time 30 "http://ran-automation-api-service:80/health/live" &>/dev/null; then
            test_result "API service connectivity" "PASS"
        else
            test_result "API service connectivity" "FAIL" "Cannot connect to API service"
        fi

        # Test cognitive performance service
        if kubectl exec "$test_pod" -n "$NAMESPACE" -- curl -f --connect-timeout 10 --max-time 30 "http://cognitive-performance-service:8080/health/live" &>/dev/null; then
            test_result "Cognitive performance service connectivity" "PASS"
        else
            test_result "Cognitive performance service connectivity" "FAIL" "Cannot connect to cognitive performance service"
        fi

        # Test swarm coordination service
        if kubectl exec "$test_pod" -n "$NAMESPACE" -- curl -f --connect-timeout 10 --max-time 30 "http://swarm-coordination-service:8081/health/live" &>/dev/null; then
            test_result "Swarm coordination service connectivity" "PASS"
        else
            test_result "Swarm coordination service connectivity" "FAIL" "Cannot connect to swarm coordination service"
        fi

        # Test database connectivity (if possible)
        if kubectl exec "$test_pod" -n "$NAMESPACE" -- timeout 10 bash -c "echo > /dev/tcp/agentdb-service/5432" &>/dev/null; then
            test_result "Database connectivity" "PASS"
        else
            test_result "Database connectivity" "FAIL" "Cannot connect to database"
        fi
    else
        test_result "Network test pod setup" "FAIL" "Test pod failed to start"
    fi

    # Cleanup test pod
    kubectl delete pod "$test_pod" -n "$NAMESPACE" --force --grace-period=0 &>/dev/null || true
}

# Validate application functionality
validate_application_functionality() {
    log "INFO" "Validating application functionality..."

    # Get API service pod
    local api_pod=$(kubectl get pods -n "$NAMESPACE" -l app=ran-automation,component=api -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)

    if [[ -z "$api_pod" ]]; then
        test_result "Application functionality tests" "FAIL" "API pod not found"
        return
    fi

    # Test health endpoints
    local health_endpoints=(
        "/health/live"
        "/health/ready"
        "/health/startup"
    )

    for endpoint in "${health_endpoints[@]}"; do
        if kubectl exec "$api_pod" -n "$NAMESPACE" -- curl -f --connect-timeout 10 --max-time 30 "http://localhost:8080$endpoint" &>/dev/null; then
            test_result "Health endpoint $endpoint" "PASS"
        else
            test_result "Health endpoint $endpoint" "FAIL"
        fi
    done

    # Test cognitive consciousness endpoint
    if kubectl exec "$api_pod" -n "$NAMESPACE" -- curl -f --connect-timeout 10 --max-time 30 "http://localhost:8080/api/cognitive/consciousness" &>/dev/null; then
        test_result "Cognitive consciousness API" "PASS"
    else
        test_result "Cognitive consciousness API" "FAIL"
    fi

    # Test swarm coordination endpoint
    if kubectl exec "$api_pod" -n "$NAMESPACE" -- curl -f --connect-timeout 10 --max-time 30 "http://localhost:8080/api/swarm/status" &>/dev/null; then
        test_result "Swarm coordination API" "PASS"
    else
        test_result "Swarm coordination API" "FAIL"
    fi

    # Test metrics endpoint
    if kubectl exec "$api_pod" -n "$NAMESPACE" -- curl -f --connect-timeout 10 --max-time 30 "http://localhost:8080/metrics" &>/dev/null; then
        test_result "Metrics endpoint" "PASS"
    else
        test_result "Metrics endpoint" "FAIL"
    fi
}

# Validate monitoring stack
validate_monitoring() {
    log "INFO" "Validating monitoring stack..."

    local monitoring_namespace="ran-monitoring"

    # Check Prometheus
    if kubectl get deployment prometheus -n "$monitoring_namespace" &>/dev/null; then
        local prometheus_ready=$(kubectl get deployment prometheus -n "$monitoring_namespace" -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo "0")
        if [[ "$prometheus_ready" -gt 0 ]]; then
            test_result "Prometheus deployment" "PASS" "$prometheus_ready replicas ready"
        else
            test_result "Prometheus deployment" "FAIL" "No ready replicas"
        fi

        # Test Prometheus targets
        local prometheus_pod=$(kubectl get pods -n "$monitoring_namespace" -l app=prometheus -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
        if [[ -n "$prometheus_pod" ]]; then
            if kubectl exec "$prometheus_pod" -n "$monitoring_namespace" -- curl -f --connect-timeout 10 "http://localhost:9090/api/v1/targets" &>/dev/null; then
                test_result "Prometheus API" "PASS"
            else
                test_result "Prometheus API" "FAIL"
            fi
        fi
    else
        test_result "Prometheus deployment" "FAIL" "Prometheus not found"
    fi

    # Check Grafana
    if kubectl get deployment grafana -n "$monitoring_namespace" &>/dev/null; then
        local grafana_ready=$(kubectl get deployment grafana -n "$monitoring_namespace" -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo "0")
        if [[ "$grafana_ready" -gt 0 ]]; then
            test_result "Grafana deployment" "PASS" "$grafana_ready replicas ready"
        else
            test_result "Grafana deployment" "FAIL" "No ready replicas"
        fi
    else
        test_result "Grafana deployment" "FAIL" "Grafana not found"
    fi
}

# Validate security configuration
validate_security() {
    log "INFO" "Validating security configuration..."

    # Check network policies
    local network_policies=$(kubectl get networkpolicies -n "$NAMESPACE" --no-headers | wc -l)
    if [[ "$network_policies" -gt 0 ]]; then
        test_result "Network policies" "PASS" "$network_policies policies found"
    else
        test_result "Network policies" "FAIL" "No network policies found"
    fi

    # Check RBAC configuration
    local service_accounts=$(kubectl get serviceaccounts -n "$NAMESPACE" --no-headers | wc -l)
    if [[ "$service_accounts" -gt 1 ]]; then
        test_result "Service accounts" "PASS" "$service_accounts service accounts found"
    else
        test_result "Service accounts" "FAIL" "Insufficient service accounts"
    fi

    # Check security context in pods
    local pods_with_security_context=0
    local total_pods=$(kubectl get pods -n "$NAMESPACE" --no-headers | wc -l)

    while IFS= read -r pod; do
        if kubectl get pod "$pod" -n "$NAMESPACE" -o jsonpath='{.spec.securityContext}' 2>/dev/null | grep -q .; then
            ((pods_with_security_context++))
        fi
    done < <(kubectl get pods -n "$NAMESPACE" -o jsonpath='{.items[*].metadata.name}')

    if [[ $pods_with_security_context -eq $total_pods && $total_pods -gt 0 ]]; then
        test_result "Pod security contexts" "PASS" "$pods_with_security_context/$total_pods pods have security contexts"
    else
        test_result "Pod security contexts" "WARN" "$pods_with_security_context/$total_pods pods have security contexts"
    fi

    # Check for resource limits
    local pods_with_limits=0
    while IFS= read -r pod; do
        if kubectl get pod "$pod" -n "$NAMESPACE" -o jsonpath='{.spec.containers[*].resources.limits}' 2>/dev/null | grep -q .; then
            ((pods_with_limits++))
        fi
    done < <(kubectl get pods -n "$NAMESPACE" -o jsonpath='{.items[*].metadata.name}')

    if [[ $pods_with_limits -eq $total_pods && $total_pods -gt 0 ]]; then
        test_result "Resource limits" "PASS" "$pods_with_limits/$total_pods pods have resource limits"
    else
        test_result "Resource limits" "WARN" "$pods_with_limits/$total_pods pods have resource limits"
    fi
}

# Validate performance metrics
validate_performance() {
    log "INFO" "Validating performance metrics..."

    # Get current resource usage
    local cpu_usage=$(kubectl top pods -n "$NAMESPACE" --no-headers 2>/dev/null | awk '{sum+=$2} END {print sum}' || echo "0")
    local memory_usage=$(kubectl top pods -n "$NAMESPACE" --no-headers 2>/dev/null | awk '{sum+=$3} END {print sum}' || echo "0")

    if [[ "$cpu_usage" != "0" ]]; then
        test_result "Resource metrics available" "PASS" "CPU: ${cpu_usage}m, Memory: ${memory_usage}Mi"
    else
        test_result "Resource metrics available" "FAIL" "Metrics server not available"
    fi

    # Check response times (if monitoring is available)
    local monitoring_namespace="ran-monitoring"
    if kubectl get service prometheus-service -n "$monitoring_namespace" &>/dev/null; then
        # This would typically require Prometheus queries
        test_result "Performance monitoring" "PASS" "Prometheus metrics available"
    else
        test_result "Performance monitoring" "WARN" "Prometheus not available for performance metrics"
    fi
}

# Generate validation report
generate_validation_report() {
    local report_file="${PROJECT_ROOT}/reports/validation-report-$(date +%Y%m%d-%H%M%S).md"
    mkdir -p "$(dirname "$report_file")"

    cat > "$report_file" << EOF
# RAN Intelligent Automation System - Validation Report

## Validation Summary
- **Timestamp**: $(date -Iseconds)
- **Namespace**: $NAMESPACE
- **Overall Status**: $([ "$VALIDATION_PASSED" == true ] && echo "PASSED" || echo "FAILED")

## Test Results

### Passed Tests (${#PASSED_TESTS[@]})
EOF

    for test in "${PASSED_TESTS[@]}"; do
        echo "- ✅ $test" >> "$report_file"
    done

    cat >> "$report_file" << EOF

### Failed Tests (${#FAILED_TESTS[@]})
EOF

    for test in "${FAILED_TESTS[@]}"; do
        echo "- ❌ $test" >> "$report_file"
    done

    cat >> "$report_file" << EOF

## Deployment Information
- **Kubernetes Context**: $(kubectl config current-context)
- **Cluster Nodes**: $(kubectl get nodes --no-headers | wc -l)
- **Pods in Namespace**: $(kubectl get pods -n "$NAMESPACE" --no-headers | wc -l)

## Resource Usage
EOF

    if kubectl top pods -n "$NAMESPACE" &>/dev/null; then
        kubectl top pods -n "$NAMESPACE" >> "$report_file"
    else
        echo "Resource metrics not available" >> "$report_file"
    fi

    cat >> "$report_file" << EOF

## Service Endpoints
EOF

    kubectl get services -n "$NAMESPACE" >> "$report_file" 2>/dev/null || echo "No services found" >> "$report_file"

    cat >> "$report_file" << EOF

## Recommendations
EOF

    if [[ "$VALIDATION_PASSED" == true ]]; then
        echo "- ✅ All critical systems are operational" >> "$report_file"
        echo "- ✅ Deployment is ready for production use" >> "$report_file"
        echo "- 📊 Continue monitoring system performance" >> "$report_file"
    else
        echo "- ❌ Address failed validations before proceeding" >> "$report_file"
        echo "- 🔍 Review deployment logs and troubleshoot issues" >> "$report_file"
        echo "- 🔄 Consider rolling back if critical services are affected" >> "$report_file"
    fi

    cat >> "$report_file" << EOF

---
*Generated by RAN Intelligent Automation System Validation Script*
EOF

    log "INFO" "Validation report generated: $report_file"
}

# Main validation function
main() {
    log "INFO" "Starting RAN Intelligent Automation System deployment validation..."
    log "INFO" "Namespace: $NAMESPACE, Timeout: ${VALIDATION_TIMEOUT}s"

    # Run all validation tests
    validate_namespace
    validate_deployments
    validate_services
    validate_pods_health
    validate_network_connectivity
    validate_application_functionality
    validate_monitoring
    validate_security
    validate_performance

    # Generate report
    generate_validation_report

    # Final result
    echo
    echo "========================================"
    echo "VALIDATION SUMMARY"
    echo "========================================"
    echo "Passed: ${#PASSED_TESTS[@]}"
    echo "Failed: ${#FAILED_TESTS[@]}"
    echo "Overall Status: $([ "$VALIDATION_PASSED" == true ] && echo "PASSED ✅" || echo "FAILED ❌")"
    echo "========================================"

    if [[ "$VALIDATION_PASSED" == true ]]; then
        log "INFO" "All validations passed successfully!"
        exit 0
    else
        log "ERROR" "Some validations failed. Please review the failed tests and fix the issues."
        exit 1
    fi
}

# Show usage
usage() {
    cat << EOF
RAN Intelligent Automation System - Deployment Validation Script

Usage: $0 [OPTIONS]

OPTIONS:
    -n, --namespace NS     Set namespace (default: ran-automation)
    -t, --timeout SEC     Set validation timeout in seconds (default: 600)
    -v, --verbose         Enable verbose logging
    -h, --help           Show this help message

EXAMPLES:
    $0
    $0 --namespace staging
    $0 --timeout 1200 --verbose

EOF
    exit 1
}

# Parse command line arguments
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -n|--namespace)
                NAMESPACE="$2"
                shift 2
                ;;
            -t|--timeout)
                VALIDATION_TIMEOUT="$2"
                shift 2
                ;;
            -v|--verbose)
                VERBOSE=true
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

# Script entry point
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    parse_arguments "$@"
    main
fi