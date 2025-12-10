#!/bin/bash

# RAN Intelligent Automation System - Smoke Tests
# Version: 1.0.0
# Description: Critical functionality smoke tests for deployment validation

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
NAMESPACE="${NAMESPACE:-ran-automation}"
API_URL="${API_URL:-https://api.ran-automation.example.com}"
COGNITIVE_URL="${COGNITIVE_URL:-https://cognitive.ran-automation.example.com}"
SWARM_URL="${SWARM_URL:-https://swarm.ran-automation.example.com}"
TIMEOUT="${TIMEOUT:-30}"
VERBOSE=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Test results
SMOKE_TESTS_PASSED=true
FAILED_SMOKE_TESTS=()
PASSED_SMOKE_TESTS=()

# Logging function
log() {
    local level=$1
    shift
    local message="$*"

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
smoke_test_result() {
    local test_name=$1
    local result=$2
    local message=${3:-""}

    if [[ "$result" == "PASS" ]]; then
        PASSED_SMOKE_TESTS+=("$test_name")
        log "INFO" "✓ PASS: $test_name"
        if [[ -n "$message" ]]; then
            log "DEBUG" "  $message"
        fi
    else
        FAILED_SMOKE_TESTS+=("$test_name")
        SMOKE_TESTS_PASSED=false
        log "ERROR" "✗ FAIL: $test_name"
        if [[ -n "$message" ]]; then
            log "ERROR" "  $message"
        fi
    fi
}

# HTTP request with timeout
make_request() {
    local url=$1
    local method=${2:-GET}
    local data=${3:-""}
    local headers=${4:-""}

    if [[ -n "$data" ]]; then
        curl -s -w "%{http_code}" -X "$method" -H "Content-Type: application/json" -d "$data" "$url" --connect-timeout "$TIMEOUT" --max-time "$TIMEOUT" 2>/dev/null | tr -d '\n'
    else
        curl -s -w "%{http_code}" -X "$method" -H "Content-Type: application/json" "$url" --connect-timeout "$TIMEOUT" --max-time "$TIMEOUT" 2>/dev/null | tr -d '\n'
    fi
}

# Test API connectivity
test_api_connectivity() {
    log "INFO" "Testing API connectivity..."

    local response=$(make_request "${API_URL}/health/live")
    local http_code="${response: -3}"
    local body="${response%???}"

    if [[ "$http_code" == "200" ]]; then
        smoke_test_result "API connectivity" "PASS" "HTTP $http_code"
    else
        smoke_test_result "API connectivity" "FAIL" "HTTP $http_code"
    fi
}

# Test API readiness
test_api_readiness() {
    log "INFO" "Testing API readiness..."

    local response=$(make_request "${API_URL}/health/ready")
    local http_code="${response: -3}"
    local body="${response%???}"

    if [[ "$http_code" == "200" ]]; then
        smoke_test_result "API readiness" "PASS" "HTTP $http_code"
    else
        smoke_test_result "API readiness" "FAIL" "HTTP $http_code"
    fi
}

# Test cognitive performance service
test_cognitive_performance() {
    log "INFO" "Testing cognitive performance service..."

    local response=$(make_request "${COGNITIVE_URL}/health/live")
    local http_code="${response: -3}"
    local body="${response%???}"

    if [[ "$http_code" == "200" ]]; then
        smoke_test_result "Cognitive performance service" "PASS" "HTTP $http_code"
    else
        smoke_test_result "Cognitive performance service" "FAIL" "HTTP $http_code"
    fi
}

# Test swarm coordination service
test_swarm_coordination() {
    log "INFO" "Testing swarm coordination service..."

    local response=$(make_request "${SWARM_URL}/health/live")
    local http_code="${response: -3}"
    local body="${response%???}"

    if [[ "$http_code" == "200" ]]; then
        smoke_test_result "Swarm coordination service" "PASS" "HTTP $http_code"
    else
        smoke_test_result "Swarm coordination service" "FAIL" "HTTP $http_code"
    fi
}

# Test cognitive consciousness endpoint
test_cognitive_consciousness() {
    log "INFO" "Testing cognitive consciousness endpoint..."

    local response=$(make_request "${API_URL}/api/cognitive/consciousness")
    local http_code="${response: -3}"
    local body="${response%???}"

    if [[ "$http_code" == "200" ]]; then
        # Check if response contains consciousness level
        if echo "$body" | grep -q "consciousness_level\|level\|consciousness"; then
            smoke_test_result "Cognitive consciousness API" "PASS" "Consciousness data returned"
        else
            smoke_test_result "Cognitive consciousness API" "FAIL" "No consciousness data in response"
        fi
    else
        smoke_test_result "Cognitive consciousness API" "FAIL" "HTTP $http_code"
    fi
}

# Test swarm status endpoint
test_swarm_status() {
    log "INFO" "Testing swarm status endpoint..."

    local response=$(make_request "${API_URL}/api/swarm/status")
    local http_code="${response: -3}"
    local body="${response%???}"

    if [[ "$http_code" == "200" ]]; then
        # Check if response contains swarm information
        if echo "$body" | grep -q "agents\|swarm\|status\|active"; then
            smoke_test_result "Swarm status API" "PASS" "Swarm status data returned"
        else
            smoke_test_result "Swarm status API" "FAIL" "No swarm status data in response"
        fi
    else
        smoke_test_result "Swarm status API" "FAIL" "HTTP $http_code"
    fi
}

# Test metrics endpoint
test_metrics_endpoint() {
    log "INFO" "Testing metrics endpoint..."

    local response=$(make_request "${API_URL}/metrics")
    local http_code="${response: -3}"
    local body="${response%???}"

    if [[ "$http_code" == "200" ]]; then
        # Check if response contains Prometheus metrics format
        if echo "$body" | grep -q "^# HELP\|^# TYPE\|[a-zA-Z_][a-zA-Z0-9_]*{"; then
            smoke_test_result "Metrics endpoint" "PASS" "Prometheus metrics format detected"
        else
            smoke_test_result "Metrics endpoint" "FAIL" "Invalid metrics format"
        fi
    else
        smoke_test_result "Metrics endpoint" "FAIL" "HTTP $http_code"
    fi
}

# Test temporal analysis endpoint
test_temporal_analysis() {
    log "INFO" "Testing temporal analysis endpoint..."

    local response=$(make_request "${API_URL}/api/cognitive/temporal")
    local http_code="${response: -3}"
    local body="${response%???}"

    if [[ "$http_code" == "200" ]]; then
        # Check if response contains temporal analysis data
        if echo "$body" | grep -q "temporal\|analysis\|expansion\|factor"; then
            smoke_test_result "Temporal analysis API" "PASS" "Temporal analysis data returned"
        else
            smoke_test_result "Temporal analysis API" "FAIL" "No temporal analysis data in response"
        fi
    else
        smoke_test_result "Temporal analysis API" "FAIL" "HTTP $http_code"
    fi
}

# Test AgentDB connectivity
test_agentdb_connectivity() {
    log "INFO" "Testing AgentDB connectivity..."

    # Get a pod to test from
    local test_pod=$(kubectl get pods -n "$NAMESPACE" -l app=ran-automation,component=api -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)

    if [[ -z "$test_pod" ]]; then
        smoke_test_result "AgentDB connectivity" "FAIL" "No API pod found for testing"
        return
    fi

    # Test TCP connection to database
    if kubectl exec "$test_pod" -n "$NAMESPACE" -- timeout 10 bash -c "echo > /dev/tcp/agentdb-service/5432" 2>/dev/null; then
        smoke_test_result "AgentDB connectivity" "PASS" "Database connection successful"
    else
        smoke_test_result "AgentDB connectivity" "FAIL" "Cannot connect to database"
    fi
}

# Test Redis connectivity
test_redis_connectivity() {
    log "INFO" "Testing Redis connectivity..."

    # Get a pod to test from
    local test_pod=$(kubectl get pods -n "$NAMESPACE" -l app=ran-automation,component=api -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)

    if [[ -z "$test_pod" ]]; then
        smoke_test_result "Redis connectivity" "FAIL" "No API pod found for testing"
        return
    fi

    # Test TCP connection to Redis
    if kubectl exec "$test_pod" -n "$NAMESPACE" -- timeout 10 bash -c "echo > /dev/tcp/redis-service/6379" 2>/dev/null; then
        smoke_test_result "Redis connectivity" "PASS" "Redis connection successful"
    else
        smoke_test_result "Redis connectivity" "FAIL" "Cannot connect to Redis"
    fi
}

# Test API authentication
test_api_authentication() {
    log "INFO" "Testing API authentication..."

    # Test protected endpoint without authentication (should fail)
    local response=$(make_request "${API_URL}/api/protected")
    local http_code="${response: -3}"

    if [[ "$http_code" == "401" || "$http_code" == "403" ]]; then
        smoke_test_result "API authentication" "PASS" "Protected endpoint properly secured"
    else
        smoke_test_result "API authentication" "FAIL" "Authentication not properly configured (HTTP $http_code)"
    fi
}

# Test rate limiting
test_rate_limiting() {
    log "INFO" "Testing rate limiting..."

    local requests_passed=0
    local rate_limit_hits=0

    # Make multiple rapid requests
    for i in {1..20}; do
        local response=$(make_request "${API_URL}/health/live")
        local http_code="${response: -3}"

        if [[ "$http_code" == "200" ]]; then
            ((requests_passed++))
        elif [[ "$http_code" == "429" ]]; then
            ((rate_limit_hits++))
        fi

        # Small delay between requests
        sleep 0.1
    done

    if [[ "$requests_passed" -gt 0 && "$rate_limit_hits" -gt 0 ]]; then
        smoke_test_result "Rate limiting" "PASS" "$requests_passed requests passed, $rate_limit_hits rate limited"
    elif [[ "$requests_passed" -gt 0 ]]; then
        smoke_test_result "Rate limiting" "WARN" "$requests_passed requests passed, no rate limiting detected"
    else
        smoke_test_result "Rate limiting" "FAIL" "All requests failed"
    fi
}

# Test service discovery
test_service_discovery() {
    log "INFO" "Testing service discovery..."

    # Test internal service resolution
    local test_pod="service-discovery-test-$(date +%s)"

    # Create a test pod
    cat << EOF | kubectl apply -f - &>/dev/null
apiVersion: v1
kind: Pod
metadata:
  name: $test_pod
  namespace: $NAMESPACE
spec:
  containers:
  - name: test
    image: busybox:1.35
    command: ['sleep', '3600']
EOF

    # Wait for pod to be ready
    local retry_count=0
    while [[ $retry_count -lt 12 ]]; do
        if kubectl get pod "$test_pod" -n "$NAMESPACE" -o jsonpath='{.status.phase}' 2>/dev/null | grep -q "Running"; then
            break
        fi
        sleep 5
        ((retry_count++))
    done

    # Test DNS resolution
    if kubectl exec "$test_pod" -n "$NAMESPACE" -- nslookup ran-automation-api-service.ran-automation.svc.cluster.local &>/dev/null; then
        smoke_test_result "Service discovery" "PASS" "DNS resolution successful"
    else
        smoke_test_result "Service discovery" "FAIL" "DNS resolution failed"
    fi

    # Cleanup test pod
    kubectl delete pod "$test_pod" -n "$NAMESPACE" --force --grace-period=0 &>/dev/null || true
}

# Test configuration loading
test_configuration_loading() {
    log "INFO" "Testing configuration loading..."

    # Get a pod to test from
    local test_pod=$(kubectl get pods -n "$NAMESPACE" -l app=ran-automation,component=api -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)

    if [[ -z "$test_pod" ]]; then
        smoke_test_result "Configuration loading" "FAIL" "No API pod found for testing"
        return
    fi

    # Check if environment variables are loaded
    if kubectl exec "$test_pod" -n "$NAMESPACE" -- env | grep -q "NODE_ENV=production" 2>/dev/null; then
        smoke_test_result "Configuration loading" "PASS" "Environment variables loaded correctly"
    else
        smoke_test_result "Configuration loading" "FAIL" "Environment variables not loaded correctly"
    fi
}

# Test resource limits
test_resource_limits() {
    log "INFO" "Testing resource limits..."

    local pods_with_limits=0
    local total_pods=$(kubectl get pods -n "$NAMESPACE" --no-headers | wc -l)

    while IFS= read -r pod; do
        if kubectl get pod "$pod" -n "$NAMESPACE" -o jsonpath='{.spec.containers[*].resources.limits}' 2>/dev/null | grep -q .; then
            ((pods_with_limits++))
        fi
    done < <(kubectl get pods -n "$NAMESPACE" -o jsonpath='{.items[*].metadata.name}')

    if [[ $pods_with_limits -eq $total_pods && $total_pods -gt 0 ]]; then
        smoke_test_result "Resource limits" "PASS" "$pods_with_limits/$total_pods pods have resource limits"
    else
        smoke_test_result "Resource limits" "WARN" "$pods_with_limits/$total_pods pods have resource limits"
    fi
}

# Test health check intervals
test_health_check_intervals() {
    log "INFO" "Testing health check intervals..."

    local response_times=()
    local start_time=$(date +%s%3N)

    # Make 5 consecutive health checks
    for i in {1..5}; do
        local request_start=$(date +%s%3N)
        local response=$(make_request "${API_URL}/health/live")
        local request_end=$(date +%s%3N)
        local response_time=$((request_end - request_start))
        response_times+=($response_time)
        sleep 1
    done

    local total_time=0
    for time in "${response_times[@]}"; do
        total_time=$((total_time + time))
    done
    local average_time=$((total_time / ${#response_times[@]}))

    if [[ $average_time -lt 1000 ]]; then  # Less than 1 second average
        smoke_test_result "Health check intervals" "PASS" "Average response time: ${average_time}ms"
    else
        smoke_test_result "Health check intervals" "WARN" "Average response time: ${average_time}ms (slow)"
    fi
}

# Generate smoke test report
generate_smoke_test_report() {
    local report_file="${PROJECT_ROOT}/reports/smoke-test-report-$(date +%Y%m%d-%H%M%S).md"
    mkdir -p "$(dirname "$report_file")"

    cat > "$report_file" << EOF
# RAN Intelligent Automation System - Smoke Test Report

## Test Summary
- **Timestamp**: $(date -Iseconds)
- **Namespace**: $NAMESPACE
- **API URL**: $API_URL
- **Overall Status**: $([ "$SMOKE_TESTS_PASSED" == true ] && echo "PASSED" || echo "FAILED")

## Test Results

### Passed Tests (${#PASSED_SMOKE_TESTS[@]})
EOF

    for test in "${PASSED_SMOKE_TESTS[@]}"; do
        echo "- ✅ $test" >> "$report_file"
    done

    cat >> "$report_file" << EOF

### Failed Tests (${#FAILED_SMOKE_TESTS[@]})
EOF

    for test in "${FAILED_SMOKE_TESTS[@]}"; do
        echo "- ❌ $test" >> "$report_file"
    done

    cat >> "$report_file" << EOF

## System Information
- **Kubernetes Context**: $(kubectl config current-context)
- **Pods in Namespace**: $(kubectl get pods -n "$NAMESPACE" --no-headers | wc -l)
- **Services**: $(kubectl get services -n "$NAMESPACE" --no-headers | wc -l)

## Test Environment
- **API URL**: $API_URL
- **Cognitive Performance URL**: $COGNITIVE_URL
- **Swarm Coordination URL**: $SWARM_URL
- **Request Timeout**: ${TIMEOUT}s

## Recommendations
EOF

    if [[ "$SMOKE_TESTS_PASSED" == true ]]; then
        echo "- ✅ All critical smoke tests passed" >> "$report_file"
        echo "- ✅ System is ready for production use" >> "$report_file"
        echo "- 📊 Continue monitoring system performance" >> "$report_file"
    else
        echo "- ❌ Some smoke tests failed" >> "$report_file"
        echo "- 🔍 Investigate failed tests and fix issues" >> "$report_file"
        echo "- 🔄 Re-run smoke tests after fixes" >> "$report_file"
    fi

    cat >> "$report_file" << EOF

---
*Generated by RAN Intelligent Automation System Smoke Test Script*
EOF

    log "INFO" "Smoke test report generated: $report_file"
}

# Main smoke test function
main() {
    log "INFO" "Starting RAN Intelligent Automation System smoke tests..."
    log "INFO" "API URL: $API_URL, Namespace: $NAMESPACE, Timeout: ${TIMEOUT}s"

    # Wait a moment for services to be ready
    log "INFO" "Waiting for services to stabilize..."
    sleep 10

    # Run all smoke tests
    test_api_connectivity
    test_api_readiness
    test_cognitive_performance
    test_swarm_coordination
    test_cognitive_consciousness
    test_swarm_status
    test_metrics_endpoint
    test_temporal_analysis
    test_agentdb_connectivity
    test_redis_connectivity
    test_api_authentication
    test_rate_limiting
    test_service_discovery
    test_configuration_loading
    test_resource_limits
    test_health_check_intervals

    # Generate report
    generate_smoke_test_report

    # Final result
    echo
    echo "========================================"
    echo "SMOKE TEST SUMMARY"
    echo "========================================"
    echo "Passed: ${#PASSED_SMOKE_TESTS[@]}"
    echo "Failed: ${#FAILED_SMOKE_TESTS[@]}"
    echo "Overall Status: $([ "$SMOKE_TESTS_PASSED" == true ] && echo "PASSED ✅" || echo "FAILED ❌")"
    echo "========================================"

    if [[ "$SMOKE_TESTS_PASSED" == true ]]; then
        log "INFO" "All smoke tests passed successfully!"
        exit 0
    else
        log "ERROR" "Some smoke tests failed. Please review the failed tests and fix the issues."
        exit 1
    fi
}

# Show usage
usage() {
    cat << EOF
RAN Intelligent Automation System - Smoke Test Script

Usage: $0 [OPTIONS]

OPTIONS:
    -n, --namespace NS     Set namespace (default: ran-automation)
    -a, --api-url URL     Set API URL (default: https://api.ran-automation.example.com)
    -c, --cognitive-url URL Set cognitive performance URL
    -s, --swarm-url URL   Set swarm coordination URL
    -t, --timeout SEC     Set request timeout in seconds (default: 30)
    -v, --verbose         Enable verbose logging
    -h, --help           Show this help message

EXAMPLES:
    $0
    $0 --namespace staging --api-url https://staging-api.example.com
    $0 --timeout 60 --verbose

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
            -a|--api-url)
                API_URL="$2"
                shift 2
                ;;
            -c|--cognitive-url)
                COGNITIVE_URL="$2"
                shift 2
                ;;
            -s|--swarm-url)
                SWARM_URL="$2"
                shift 2
                ;;
            -t|--timeout)
                TIMEOUT="$2"
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