#!/usr/bin/env bash
# Integration tests for WasmEdge and Spin agents on Kubernetes
# Validates deployment, health checks, and basic functionality

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[PASS]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[FAIL]${NC} $1"; }

TESTS_PASSED=0
TESTS_FAILED=0

# Test helper
run_test() {
    local name="$1"
    local cmd="$2"

    log_info "Running test: $name"
    if eval "$cmd"; then
        log_success "$name"
        TESTS_PASSED=$((TESTS_PASSED + 1))
    else
        log_error "$name"
        TESTS_FAILED=$((TESTS_FAILED + 1))
    fi
}

# Test: Kubernetes connectivity
test_k8s_connection() {
    kubectl cluster-info &> /dev/null
}

# Test: RuntimeClass exists
test_runtimeclass_wasmedge() {
    kubectl get runtimeclass wasmedge &> /dev/null
}

test_runtimeclass_spin() {
    kubectl get runtimeclass wasmtime-spin &> /dev/null
}

# Test: SpinKube operator running
test_spinkube_operator() {
    local pods=$(kubectl -n spinkube-system get pods -l app=spin-operator --no-headers 2>/dev/null | wc -l)
    [ "$pods" -gt 0 ]
}

# Test: Runtime Class Manager running
test_rcm_running() {
    local pods=$(kubectl -n rcm-system get pods -l app=runtime-class-manager --no-headers 2>/dev/null | wc -l)
    [ "$pods" -gt 0 ]
}

# Test: Shims CRD exists
test_shims_crd() {
    kubectl get crd shims.runtime.kwasm.sh &> /dev/null
}

# Test: SpinApp CRD exists
test_spinapp_crd() {
    kubectl get crd spinapps.core.spinoperator.dev &> /dev/null
}

# Test: Agents namespace exists
test_agents_namespace() {
    kubectl get namespace agents &> /dev/null
}

# Test: Deploy test SpinApp
test_deploy_spinapp() {
    local test_spinapp='
apiVersion: core.spinoperator.dev/v1alpha1
kind: SpinApp
metadata:
  name: test-spin-agent
  namespace: agents
spec:
  image: ghcr.io/ruvnet/spin-agent:test
  replicas: 1
  executor: containerd-shim-spin
  resources:
    limits:
      cpu: 100m
      memory: 64Mi
'
    echo "$test_spinapp" | kubectl apply -f - &> /dev/null
    sleep 5
    kubectl -n agents get spinapp test-spin-agent &> /dev/null
}

# Test: Deploy test WasmEdge pod
test_deploy_wasmedge_pod() {
    local test_pod='
apiVersion: v1
kind: Pod
metadata:
  name: test-wasmedge-pod
  namespace: agents
spec:
  runtimeClassName: wasmedge
  containers:
    - name: test
      image: ghcr.io/ruvnet/wasmedge-agent:test
      command: ["sleep", "30"]
  restartPolicy: Never
'
    echo "$test_pod" | kubectl apply -f - &> /dev/null 2>&1 || true
    sleep 3
    # Even if pod fails to run (due to missing image), check if it was created
    kubectl -n agents get pod test-wasmedge-pod &> /dev/null
}

# Test: Agent orchestrator Python syntax
test_agent_orchestrator_syntax() {
    python3 -m py_compile "$PROJECT_ROOT/apps/fastapi/agents.py" 2>/dev/null
}

# Test: Spin JS package.json valid
test_spin_package_json() {
    node -e "JSON.parse(require('fs').readFileSync('$PROJECT_ROOT/apps/spin-js/package.json'))"
}

# Test: WasmEdge JS package.json valid
test_wasmedge_package_json() {
    node -e "JSON.parse(require('fs').readFileSync('$PROJECT_ROOT/apps/wasmedge-js/package.json'))"
}

# Test: YAML manifests valid
test_yaml_syntax() {
    local files=(
        "$PROJECT_ROOT/infrastructure/wasmedge/wasmedge-runtime.yaml"
        "$PROJECT_ROOT/infrastructure/runtime-class-manager/runtime-class-manager.yaml"
        "$PROJECT_ROOT/infrastructure/spinkube/spin-operator.yaml"
        "$PROJECT_ROOT/infrastructure/spinkube/spinapp-examples.yaml"
    )

    for file in "${files[@]}"; do
        if [ -f "$file" ]; then
            kubectl apply --dry-run=client -f "$file" &> /dev/null || return 1
        fi
    done
    return 0
}

# Cleanup test resources
cleanup() {
    log_info "Cleaning up test resources..."
    kubectl -n agents delete spinapp test-spin-agent --ignore-not-found &> /dev/null || true
    kubectl -n agents delete pod test-wasmedge-pod --ignore-not-found &> /dev/null || true
}

# Main test runner
main() {
    log_info "Starting Wasm Agent Integration Tests"
    echo ""

    # Basic tests (don't require running cluster)
    run_test "Agent orchestrator Python syntax" "test_agent_orchestrator_syntax"
    run_test "Spin JS package.json valid" "test_spin_package_json"
    run_test "WasmEdge JS package.json valid" "test_wasmedge_package_json"

    # Kubernetes tests (require cluster)
    if kubectl cluster-info &> /dev/null; then
        run_test "Kubernetes connectivity" "test_k8s_connection"
        run_test "YAML manifests syntax" "test_yaml_syntax"
        run_test "Agents namespace exists" "test_agents_namespace"
        run_test "SpinApp CRD exists" "test_spinapp_crd"
        run_test "Shims CRD exists" "test_shims_crd"
        run_test "RuntimeClass wasmedge exists" "test_runtimeclass_wasmedge"
        run_test "RuntimeClass wasmtime-spin exists" "test_runtimeclass_spin"
        run_test "SpinKube operator running" "test_spinkube_operator"
        run_test "Runtime Class Manager running" "test_rcm_running"

        # Deployment tests (may fail if images don't exist)
        run_test "Deploy test SpinApp" "test_deploy_spinapp"
        run_test "Deploy test WasmEdge pod" "test_deploy_wasmedge_pod"

        cleanup
    else
        log_warn "Kubernetes cluster not available - skipping cluster tests"
    fi

    echo ""
    echo "=========================================="
    echo "Test Results"
    echo "=========================================="
    echo -e "Passed: ${GREEN}$TESTS_PASSED${NC}"
    echo -e "Failed: ${RED}$TESTS_FAILED${NC}"
    echo ""

    if [ "$TESTS_FAILED" -gt 0 ]; then
        log_error "Some tests failed!"
        exit 1
    else
        log_success "All tests passed!"
        exit 0
    fi
}

# Handle cleanup on exit
trap cleanup EXIT

main "$@"
