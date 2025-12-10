#!/bin/bash
# Enhanced DevPod provisioning for comprehensive functional testing
# Supports all 10 environments: 5 standard + 5 agentic variants

set -e

echo "🧪 Enhanced DevPod Functional Testing Environment Manager"
echo "========================================================"
echo "🎯 Target: 64+ MCP tools × 10 environments = 640+ test combinations"
echo ""

# Configuration
readonly STANDARD_ENVS=("python" "typescript" "rust" "go" "nushell")
readonly AGENTIC_ENVS=("agentic-python" "agentic-typescript" "agentic-rust" "agentic-go" "agentic-nushell")
readonly ALL_ENVS=("${STANDARD_ENVS[@]}" "${AGENTIC_ENVS[@]}")
readonly MAX_CONCURRENT=5
readonly TEST_MODE="${TEST_MODE:-false}"
readonly WORKSPACE_PREFIX="polyglot-test"

# Validation timeouts (seconds)
readonly PROVISION_TIMEOUT=300
readonly VALIDATION_TIMEOUT=120
readonly CLEANUP_TIMEOUT=60

show_usage() {
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "COMMANDS:"
    echo "  test-swarm         Provision all environments for functional testing"
    echo "  provision-matrix   Provision specific environment matrix"
    echo "  validate-all       Validate all provisioned environments"
    echo "  status-detailed    Show detailed status with health checks"
    echo "  benchmark          Run performance benchmarks"
    echo "  cleanup-test       Clean up test environments"
    echo "  run-mcp-tests     Execute full MCP tool test suite"
    echo ""
    echo "OPTIONS:"
    echo "  --parallel N       Number of parallel provisions (default: 5)"
    echo "  --include-agentic  Include agentic variants in provisioning"
    echo "  --test-mode        Enable functional testing mode"
    echo "  --validate-only    Only run validation without provisioning"
    echo "  --benchmark-only   Only run benchmarks on existing environments"
    echo ""
    echo "EXAMPLES:"
    echo "  ./enhanced-provision-all.sh test-swarm --include-agentic"
    echo "  ./enhanced-provision-all.sh validate-all --test-mode"
    echo "  ./enhanced-provision-all.sh run-mcp-tests --parallel 3"
}

log() {
    local level=$1
    shift
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] [$level] $*"
}

log_info() { log "INFO" "$@"; }
log_warn() { log "WARN" "$@"; }
log_error() { log "ERROR" "$@"; }
log_success() { log "SUCCESS" "$@"; }

check_prerequisites() {
    log_info "🔍 Checking prerequisites..."
    
    # Check required tools
    local tools=("devpod" "jq" "timeout" "docker" "nu")
    for tool in "${tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            log_error "❌ Required tool not found: $tool"
            exit 1
        fi
    done
    
    # Check DevPod status
    if ! devpod version &> /dev/null; then
        log_error "❌ DevPod not properly configured"
        exit 1
    fi
    
    # Check centralized DevPod management script
    local devpod_script="../../host-tooling/devpod-management/manage-devpod.nu"
    if [[ ! -f "$devpod_script" ]]; then
        log_error "❌ Centralized DevPod management script not found: $devpod_script"
        exit 1
    fi
    
    log_success "✅ All prerequisites satisfied"
}

provision_environment() {
    local env_name=$1
    local workspace_name="${WORKSPACE_PREFIX}-${env_name}-$(date +%s)"
    
    log_info "🚀 Provisioning $env_name environment: $workspace_name"
    
    # Use centralized DevPod management for provisioning
    if timeout "$PROVISION_TIMEOUT" nu ../../host-tooling/devpod-management/manage-devpod.nu provision "$env_name"; then
        log_success "✅ Successfully provisioned $env_name"
        echo "$workspace_name" >> "/tmp/provisioned_workspaces.txt"
        return 0
    else
        log_error "❌ Failed to provision $env_name"
        return 1
    fi
}

validate_environment() {
    local env_name=$1
    local workspace_name=$2
    
    log_info "🔍 Validating $env_name environment..."
    
    # Basic connectivity test
    if ! devpod ssh "$workspace_name" -- echo "Connection test" &> /dev/null; then
        log_error "❌ Cannot connect to workspace: $workspace_name"
        return 1
    fi
    
    # Environment-specific validation
    case "$env_name" in
        "python"|"agentic-python")
            validate_python_environment "$workspace_name"
            ;;
        "typescript"|"agentic-typescript")
            validate_typescript_environment "$workspace_name"
            ;;
        "rust"|"agentic-rust")
            validate_rust_environment "$workspace_name"
            ;;
        "go"|"agentic-go")
            validate_go_environment "$workspace_name"
            ;;
        "nushell"|"agentic-nushell")
            validate_nushell_environment "$workspace_name"
            ;;
        *)
            log_warn "⚠️ Unknown environment type: $env_name"
            return 1
            ;;
    esac
}

validate_python_environment() {
    local workspace_name=$1
    log_info "🐍 Validating Python environment..."
    
    # Check Python and uv
    devpod ssh "$workspace_name" -- "python --version && uv --version" || return 1
    
    # Check development tools
    devpod ssh "$workspace_name" -- "ruff --version && mypy --version && pytest --version" || return 1
    
    # Check Claude-Flow integration
    devpod ssh "$workspace_name" -- "npx --yes claude-flow@alpha --version" || return 1
    
    # Test basic functionality
    devpod ssh "$workspace_name" -- "echo 'print(\"Hello from Python DevPod\")' | python" || return 1
    
    log_success "✅ Python environment validation passed"
}

validate_typescript_environment() {
    local workspace_name=$1
    log_info "📘 Validating TypeScript environment..."
    
    # Check Node.js and npm
    devpod ssh "$workspace_name" -- "node --version && npm --version" || return 1
    
    # Check TypeScript tools
    devpod ssh "$workspace_name" -- "tsc --version && npx eslint --version" || return 1
    
    # Check Claude-Flow integration
    devpod ssh "$workspace_name" -- "npx --yes claude-flow@alpha --version" || return 1
    
    # Test basic functionality
    devpod ssh "$workspace_name" -- "echo 'console.log(\"Hello from TypeScript DevPod\");' | node" || return 1
    
    log_success "✅ TypeScript environment validation passed"
}

validate_rust_environment() {
    local workspace_name=$1
    log_info "🦀 Validating Rust environment..."
    
    # Check Rust toolchain
    devpod ssh "$workspace_name" -- "rustc --version && cargo --version" || return 1
    
    # Check development tools
    devpod ssh "$workspace_name" -- "rustfmt --version && cargo clippy --version" || return 1
    
    # Check Claude-Flow integration
    devpod ssh "$workspace_name" -- "npx --yes claude-flow@alpha --version" || return 1
    
    # Test basic functionality
    devpod ssh "$workspace_name" -- "echo 'fn main() { println!(\"Hello from Rust DevPod\"); }' > /tmp/test.rs && rustc /tmp/test.rs && /tmp/test" || return 1
    
    log_success "✅ Rust environment validation passed"
}

validate_go_environment() {
    local workspace_name=$1
    log_info "🐹 Validating Go environment..."
    
    # Check Go toolchain
    devpod ssh "$workspace_name" -- "go version" || return 1
    
    # Check development tools
    devpod ssh "$workspace_name" -- "golangci-lint --version" || return 1
    
    # Check Claude-Flow integration
    devpod ssh "$workspace_name" -- "npx --yes claude-flow@alpha --version" || return 1
    
    # Test basic functionality
    devpod ssh "$workspace_name" -- "echo 'package main; import \"fmt\"; func main() { fmt.Println(\"Hello from Go DevPod\") }' > /tmp/test.go && cd /tmp && go run test.go" || return 1
    
    log_success "✅ Go environment validation passed"
}

validate_nushell_environment() {
    local workspace_name=$1
    log_info "🐚 Validating Nushell environment..."
    
    # Check Nushell
    devpod ssh "$workspace_name" -- "nu --version" || return 1
    
    # Check Claude-Flow integration
    devpod ssh "$workspace_name" -- "npx --yes claude-flow@alpha --version" || return 1
    
    # Test basic functionality
    devpod ssh "$workspace_name" -- "echo 'print \"Hello from Nushell DevPod\"' | nu" || return 1
    
    log_success "✅ Nushell environment validation passed"
}

provision_test_swarm() {
    local include_agentic=${1:-false}
    local parallel=${2:-$MAX_CONCURRENT}
    
    log_info "🧪 Starting test swarm provisioning..."
    
    # Clean up previous test workspaces list
    rm -f "/tmp/provisioned_workspaces.txt"
    touch "/tmp/provisioned_workspaces.txt"
    
    # Determine environments to provision
    local envs_to_provision=("${STANDARD_ENVS[@]}")
    if [[ "$include_agentic" == "true" ]]; then
        envs_to_provision+=("${AGENTIC_ENVS[@]}")
    fi
    
    log_info "📋 Environments to provision: ${envs_to_provision[*]}"
    log_info "⚡ Parallel provisioning: $parallel"
    
    # Provision environments in parallel batches
    local batch_size=$parallel
    local total_envs=${#envs_to_provision[@]}
    local success_count=0
    local failure_count=0
    
    for ((i = 0; i < total_envs; i += batch_size)); do
        local batch=("${envs_to_provision[@]:i:batch_size}")
        log_info "🔄 Processing batch: ${batch[*]}"
        
        # Start parallel provisioning for current batch
        local pids=()
        for env in "${batch[@]}"; do
            (provision_environment "$env") &
            pids+=($!)
        done
        
        # Wait for all processes in batch to complete
        for pid in "${pids[@]}"; do
            if wait "$pid"; then
                ((success_count++))
            else
                ((failure_count++))
            fi
        done
        
        log_info "📊 Batch complete. Success: $success_count, Failures: $failure_count"
    done
    
    log_info "✅ Swarm provisioning complete!"
    log_info "📈 Final stats - Success: $success_count, Failures: $failure_count"
    
    if [[ $failure_count -gt 0 ]]; then
        log_warn "⚠️ Some environments failed to provision"
        return 1
    fi
    
    return 0
}

validate_all_environments() {
    log_info "🔍 Starting comprehensive environment validation..."
    
    # Get list of provisioned workspaces
    local workspaces
    mapfile -t workspaces < <(devpod list --output json 2>/dev/null | jq -r '.[].name' | grep "^${WORKSPACE_PREFIX}-" || echo "")
    
    if [[ ${#workspaces[@]} -eq 0 ]]; then
        log_warn "⚠️ No test workspaces found for validation"
        return 1
    fi
    
    log_info "📋 Found ${#workspaces[@]} test workspaces to validate"
    
    local success_count=0
    local failure_count=0
    
    for workspace in "${workspaces[@]}"; do
        local env_name
        env_name=$(echo "$workspace" | sed "s/^${WORKSPACE_PREFIX}-//" | sed 's/-[0-9]*$//')
        
        if validate_environment "$env_name" "$workspace"; then
            ((success_count++))
        else
            ((failure_count++))
        fi
    done
    
    log_info "✅ Validation complete!"
    log_info "📈 Final stats - Success: $success_count, Failures: $failure_count"
    
    return $failure_count
}

run_performance_benchmarks() {
    log_info "⚡ Running performance benchmarks..."
    
    # Benchmark DevPod provisioning time
    local start_time end_time duration
    start_time=$(date +%s)
    
    # Test single environment provisioning
    log_info "🧪 Benchmarking single environment provisioning..."
    provision_environment "python"
    
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    
    log_info "📊 Single environment provisioning: ${duration}s"
    
    # Benchmark MCP tool execution
    log_info "🧪 Benchmarking MCP tool execution..."
    # This would integrate with the MCP test suite
    
    log_success "✅ Performance benchmarks complete"
}

run_mcp_test_suite() {
    log_info "🧪 Starting comprehensive MCP tool test suite..."
    
    # Change to MCP directory and run tests
    cd ../../mcp || {
        log_error "❌ Could not change to MCP directory"
        return 1
    }
    
    # Build and test MCP server
    npm run build || {
        log_error "❌ MCP server build failed"
        return 1
    }
    
    # Run comprehensive test suite
    npm run test:functional || {
        log_error "❌ Functional tests failed"
        return 1
    }
    
    npm run test:modular || {
        log_error "❌ Modular tests failed"
        return 1
    }
    
    log_success "✅ MCP test suite completed successfully"
}

cleanup_test_environments() {
    log_info "🗑️ Cleaning up test environments..."
    
    local workspaces
    mapfile -t workspaces < <(devpod list --output json 2>/dev/null | jq -r '.[].name' | grep "^${WORKSPACE_PREFIX}-" || echo "")
    
    if [[ ${#workspaces[@]} -eq 0 ]]; then
        log_info "📝 No test workspaces to clean up"
        return 0
    fi
    
    log_info "🗑️ Found ${#workspaces[@]} test workspaces to clean up"
    
    for workspace in "${workspaces[@]}"; do
        log_info "🗑️ Deleting workspace: $workspace"
        if timeout "$CLEANUP_TIMEOUT" devpod delete "$workspace" --force; then
            log_success "✅ Deleted: $workspace"
        else
            log_error "❌ Failed to delete: $workspace"
        fi
    done
    
    # Clean up temporary files
    rm -f "/tmp/provisioned_workspaces.txt"
    
    log_success "✅ Cleanup complete"
}

show_detailed_status() {
    log_info "📊 Generating detailed status report..."
    
    echo ""
    echo "🐳 DevPod Workspaces Status"
    echo "=========================="
    devpod list || log_warn "Could not list workspaces"
    
    echo ""
    echo "🧪 Test Environment Health Check"
    echo "================================"
    
    local workspaces
    mapfile -t workspaces < <(devpod list --output json 2>/dev/null | jq -r '.[].name' | grep "^${WORKSPACE_PREFIX}-" || echo "")
    
    if [[ ${#workspaces[@]} -eq 0 ]]; then
        echo "📝 No test environments found"
    else
        for workspace in "${workspaces[@]}"; do
            local env_name
            env_name=$(echo "$workspace" | sed "s/^${WORKSPACE_PREFIX}-//" | sed 's/-[0-9]*$//')
            
            echo "🔍 $workspace ($env_name):"
            if devpod ssh "$workspace" -- echo "  ✅ Connection: OK" 2>/dev/null; then
                echo "  ✅ Connection: OK"
                
                # Basic health checks
                devpod ssh "$workspace" -- "df -h /" 2>/dev/null | tail -n 1 | awk '{print "  💾 Disk: " $5 " used"}'
                devpod ssh "$workspace" -- "free -h" 2>/dev/null | grep Mem | awk '{print "  🧠 Memory: " $3 "/" $2 " used"}'
            else
                echo "  ❌ Connection: Failed"
            fi
            echo ""
        done
    fi
    
    echo "🔧 System Resources"
    echo "=================="
    echo "💾 Host disk space:"
    df -h . | tail -n 1
    echo "🧠 Host memory:"
    free -h | grep Mem
    echo ""
}

# Parse command line arguments
INCLUDE_AGENTIC=false
PARALLEL=$MAX_CONCURRENT
VALIDATE_ONLY=false
BENCHMARK_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --include-agentic)
            INCLUDE_AGENTIC=true
            shift
            ;;
        --parallel)
            PARALLEL="$2"
            shift 2
            ;;
        --test-mode)
            export TEST_MODE=true
            shift
            ;;
        --validate-only)
            VALIDATE_ONLY=true
            shift
            ;;
        --benchmark-only)
            BENCHMARK_ONLY=true
            shift
            ;;
        *)
            break
            ;;
    esac
done

# Main command handling
main() {
    check_prerequisites
    
    case "${1:-status-detailed}" in
        "test-swarm")
            provision_test_swarm "$INCLUDE_AGENTIC" "$PARALLEL"
            if [[ $? -eq 0 && "$VALIDATE_ONLY" != "true" ]]; then
                validate_all_environments
            fi
            ;;
        "provision-matrix")
            provision_test_swarm "$INCLUDE_AGENTIC" "$PARALLEL"
            ;;
        "validate-all")
            validate_all_environments
            ;;
        "status-detailed")
            show_detailed_status
            ;;
        "benchmark")
            if [[ "$BENCHMARK_ONLY" == "true" ]]; then
                run_performance_benchmarks
            else
                provision_test_swarm false 1  # Single environment for benchmarking
                run_performance_benchmarks
            fi
            ;;
        "cleanup-test")
            cleanup_test_environments
            ;;
        "run-mcp-tests")
            if [[ "$TEST_MODE" == "true" ]]; then
                provision_test_swarm "$INCLUDE_AGENTIC" "$PARALLEL"
                validate_all_environments
            fi
            run_mcp_test_suite
            ;;
        "help"|"-h"|"--help")
            show_usage
            ;;
        *)
            log_error "❌ Unknown command: $1"
            echo ""
            show_usage
            exit 1
            ;;
    esac
}

# Execute main function with all arguments
main "$@"