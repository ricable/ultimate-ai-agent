#!/bin/bash

# Cross-Language Quality Gates for Polyglot Development Environment
# Traditional bash implementation (for compatibility)
# Usage: ./scripts/validate-all.sh [environment]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Main validation function
validate_environment() {
    local env_name=$1
    local env_dir=$2
    local commands=$3
    
    log_info "$env_name..."
    
    if [ ! -d "$env_dir" ]; then
        log_warn "$env_name directory not found: $env_dir"
        return 1
    fi
    
    cd "$env_dir"
    
    if [ ! -f "devbox.json" ]; then
        log_warn "No devbox.json found in $env_dir"
        cd ..
        return 1
    fi
    
    # Run commands
    IFS=',' read -ra ADDR <<< "$commands"
    for cmd in "${ADDR[@]}"; do
        log_info "  Running: devbox run $cmd"
        if ! devbox run "$cmd"; then
            log_error "  Command failed: $cmd"
            cd ..
            return 1
        fi
    done
    
    cd ..
    return 0
}

# Main execution
main() {
    local target_env=${1:-"all"}
    
    log_info "🚀 Starting cross-language quality gates validation..."
    log_info "Target environment: $target_env"
    
    local environments=(
        "🐍 Python,python-env,lint,test"
        "📘 TypeScript,typescript-env,lint,test"
        "🦀 Rust,rust-env,lint,test"
        "🐹 Go,go-env,lint,test"
        "🐚 Nushell,nushell-env,check,test"
    )
    
    local success_count=0
    local total_count=0
    
    for env_info in "${environments[@]}"; do
        IFS=',' read -ra ENV_PARTS <<< "$env_info"
        local env_name="${ENV_PARTS[0]}"
        local env_dir="${ENV_PARTS[1]}"
        local env_commands="${ENV_PARTS[2]},${ENV_PARTS[3]}"
        
        # Filter by target environment
        if [ "$target_env" != "all" ] && [[ ! "$env_dir" =~ "$target_env" ]]; then
            continue
        fi
        
        ((total_count++))
        
        if validate_environment "$env_name" "$env_dir" "$env_commands"; then
            log_success "✅ $env_name validation passed"
            ((success_count++))
        else
            log_error "❌ $env_name validation failed"
        fi
        
        echo ""
    done
    
    echo "=========================="
    log_info "Validation Results: $success_count/$total_count environments passed"
    
    if [ $success_count -eq $total_count ]; then
        log_success "🎉 All validations passed!"
        exit 0
    else
        log_error "💥 Some validations failed!"
        exit 1
    fi
}

# Handle help
if [ "$1" = "help" ] || [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo "Cross-Language Quality Gates for Polyglot Development Environment"
    echo ""
    echo "Usage: $0 [environment]"
    echo ""
    echo "Arguments:"
    echo "  environment    Target specific environment (default: all)"
    echo "                 Options: all, python, typescript, rust, go, nushell"
    echo ""
    echo "Examples:"
    echo "  $0              # Validate all environments"
    echo "  $0 python       # Validate Python environment only"
    echo "  $0 typescript   # Validate TypeScript environment only"
    echo ""
    echo "Note: For more advanced options, use the Nushell version:"
    echo "  nu scripts/validate-all.nu --help"
    exit 0
fi

# Run main function
main "$@"