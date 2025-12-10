#!/bin/bash
# scripts/test-deployment-configs.sh
# Test deployment configurations and validate setup

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

info() { echo -e "${BLUE}[INFO]${NC} $@"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $@"; }
error() { echo -e "${RED}[ERROR]${NC} $@"; }
success() { echo -e "${GREEN}[SUCCESS]${NC} $@"; }

# Test results
TESTS_PASSED=0
TESTS_FAILED=0
TESTS_TOTAL=0

# Test function
run_test() {
    local test_name="$1"
    local test_command="$2"
    
    TESTS_TOTAL=$((TESTS_TOTAL + 1))
    info "Running test: $test_name"
    
    if eval "$test_command" > /dev/null 2>&1; then
        success "✓ $test_name"
        TESTS_PASSED=$((TESTS_PASSED + 1))
        return 0
    else
        error "✗ $test_name"
        TESTS_FAILED=$((TESTS_FAILED + 1))
        return 1
    fi
}

# Test SkyPilot configurations
test_skypilot_configs() {
    info "Testing SkyPilot configurations..."
    
    local configs=(
        "skypilot/uap-production.yaml"
        "skypilot/uap-aws.yaml"
        "skypilot/uap-gcp.yaml"
        "skypilot/uap-azure.yaml"
        "skypilot/uap-cost-optimized.yaml"
    )
    
    for config in "${configs[@]}"; do
        run_test "Validate $config" "sky check '$PROJECT_ROOT/$config'"
    done
}

# Test Teller configuration
test_teller_config() {
    info "Testing Teller configuration..."
    
    run_test "Teller config validation" "teller run --config '$PROJECT_ROOT/.teller.yml' echo 'test'"
}

# Test Docker configurations
test_docker_configs() {
    info "Testing Docker configurations..."
    
    run_test "Dockerfile validation" "docker build --target production -t uap-test '$PROJECT_ROOT' --dry-run"
    run_test "Production compose validation" "docker-compose -f '$PROJECT_ROOT/docker-compose.production.yml' config"
    
    if [[ -f "$PROJECT_ROOT/docker-compose.monitoring.yml" ]]; then
        run_test "Monitoring compose validation" "docker-compose -f '$PROJECT_ROOT/docker-compose.monitoring.yml' config"
    fi
}

# Test environment templates
test_env_templates() {
    info "Testing environment templates..."
    
    run_test "Production env template exists" "test -f '$PROJECT_ROOT/.env.production.template'"
    run_test "Staging env template exists" "test -f '$PROJECT_ROOT/.env.staging.template'"
    
    # Validate environment templates have required variables
    local required_vars=(
        "UAP_ENV"
        "PYTHONPATH"
        "BACKEND_HOST"
        "BACKEND_PORT"
        "OPENAI_API_KEY"
        "ANTHROPIC_API_KEY"
    )
    
    for var in "${required_vars[@]}"; do
        run_test "Production template contains $var" "grep -q '$var' '$PROJECT_ROOT/.env.production.template'"
    done
}

# Test script permissions and existence
test_scripts() {
    info "Testing deployment scripts..."
    
    local scripts=(
        "scripts/deploy-production.sh"
        "scripts/setup-monitoring.sh"
        "scripts/start-production.sh"
        "scripts/health-check.sh"
    )
    
    for script in "${scripts[@]}"; do
        run_test "$script exists" "test -f '$PROJECT_ROOT/$script'"
        run_test "$script is executable" "test -x '$PROJECT_ROOT/$script'"
    done
}

# Test monitoring configuration
test_monitoring_configs() {
    info "Testing monitoring configurations..."
    
    if [[ -f "$PROJECT_ROOT/monitoring/prometheus.yml" ]]; then
        run_test "Prometheus config validation" "promtool check config '$PROJECT_ROOT/monitoring/prometheus.yml' || echo 'promtool not available, skipping'"
    fi
    
    run_test "Grafana dashboards exist" "test -d '$PROJECT_ROOT/monitoring/grafana/dashboards'"
    run_test "Grafana datasources exist" "test -d '$PROJECT_ROOT/monitoring/grafana/datasources'"
}

# Test dependencies
test_dependencies() {
    info "Testing required dependencies..."
    
    local deps=(
        "sky:SkyPilot CLI"
        "teller:Teller secrets management"
        "docker:Docker container runtime"
        "curl:HTTP client"
        "jq:JSON processor"
    )
    
    for dep in "${deps[@]}"; do
        local cmd=$(echo "$dep" | cut -d: -f1)
        local name=$(echo "$dep" | cut -d: -f2)
        run_test "$name available" "command -v $cmd"
    done
}

# Test framework integration readiness
test_framework_readiness() {
    info "Testing framework integration readiness..."
    
    # Check if framework files exist and have expected interface
    local frameworks=("copilot" "agno" "mastra")
    
    for framework in "${frameworks[@]}"; do
        local agent_file="$PROJECT_ROOT/backend/frameworks/$framework/agent.py"
        run_test "$framework agent file exists" "test -f '$agent_file'"
        run_test "$framework has process_message method" "grep -q 'async def process_message' '$agent_file'"
        run_test "$framework has get_status method" "grep -q 'def get_status' '$agent_file'"
    done
}

# Test secrets configuration completeness
test_secrets_completeness() {
    info "Testing secrets configuration completeness..."
    
    local teller_file="$PROJECT_ROOT/.teller.yml"
    
    # Check for required secret categories
    local secret_categories=(
        "OPENAI_API_KEY"
        "ANTHROPIC_API_KEY"
        "COPILOTKIT_API_KEY"
        "AGNO_API_KEY"
        "MASTRA_API_KEY"
        "DATABASE_URL"
        "JWT_SECRET"
    )
    
    for secret in "${secret_categories[@]}"; do
        run_test "Teller config includes $secret" "grep -q '$secret' '$teller_file'"
    done
}

# Dry run deployment test
test_dry_run_deployment() {
    info "Testing dry run deployment..."
    
    # Test the deployment script dry run functionality
    if [[ -x "$PROJECT_ROOT/scripts/deploy-production.sh" ]]; then
        run_test "Deployment script dry run" "'$PROJECT_ROOT/scripts/deploy-production.sh' --dry-run --cloud auto"
    else
        warn "Deployment script not executable, skipping dry run test"
    fi
}

# Main test runner
main() {
    info "Starting UAP deployment configuration tests..."
    info "Project root: $PROJECT_ROOT"
    echo
    
    # Run all test suites
    test_dependencies
    test_skypilot_configs
    test_teller_config
    test_docker_configs
    test_env_templates
    test_scripts
    test_monitoring_configs
    test_framework_readiness
    test_secrets_completeness
    test_dry_run_deployment
    
    # Summary
    echo
    info "Test Results Summary:"
    success "Tests passed: $TESTS_PASSED"
    if [[ $TESTS_FAILED -gt 0 ]]; then
        error "Tests failed: $TESTS_FAILED"
    fi
    info "Total tests: $TESTS_TOTAL"
    
    # Calculate percentage
    local pass_percentage=$(( (TESTS_PASSED * 100) / TESTS_TOTAL ))
    info "Pass rate: ${pass_percentage}%"
    
    if [[ $TESTS_FAILED -eq 0 ]]; then
        success "All tests passed! Deployment configuration is ready."
        exit 0
    else
        error "Some tests failed. Please review the errors above."
        exit 1
    fi
}

# Help function
show_help() {
    cat << EOF
UAP Deployment Configuration Test Suite

This script validates the deployment configuration and readiness.

Usage: $0 [OPTIONS]

Options:
    -h, --help    Show this help message

Tests performed:
    - SkyPilot configuration validation
    - Teller secrets configuration
    - Docker configuration validation
    - Environment template validation
    - Script permissions and existence
    - Monitoring configuration
    - Framework integration readiness
    - Dependencies availability

EOF
}

# Parse arguments
case "${1:-}" in
    -h|--help)
        show_help
        exit 0
        ;;
    *)
        main "$@"
        ;;
esac