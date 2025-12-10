#!/bin/bash
# Kimi-K2 Integration Crate Publishing Script
# Publishes Kimi-K2 crates for Synaptic Neural Mesh integration

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if Rust is installed
    if ! command -v cargo &> /dev/null; then
        log_error "Cargo not found. Please install Rust."
        exit 1
    fi
    
    # Check if logged into crates.io
    if ! cargo login --help &> /dev/null; then
        log_warning "Make sure you're logged into crates.io with 'cargo login'"
    fi
    
    log_success "Prerequisites check completed"
}

# Verify crate compilation
verify_compilation() {
    local crate_path=$1
    local crate_name=$2
    
    log_info "Verifying compilation for $crate_name..."
    
    cd "$crate_path"
    
    # Check compilation
    if cargo check --all-features; then
        log_success "$crate_name compiles successfully"
    else
        log_error "$crate_name failed to compile"
        return 1
    fi
    
    # Run tests
    if cargo test --all-features; then
        log_success "$crate_name tests pass"
    else
        log_error "$crate_name tests failed"
        return 1
    fi
    
    # Check for warnings
    if cargo clippy --all-features -- -W clippy::all; then
        log_success "$crate_name passed clippy checks"
    else
        log_warning "$crate_name has clippy warnings (non-blocking)"
    fi
    
    cd - > /dev/null
}

# Publish a single crate
publish_crate() {
    local crate_path=$1
    local crate_name=$2
    local is_dry_run=$3
    
    log_info "Publishing $crate_name..."
    
    cd "$crate_path"
    
    # Dry run first
    log_info "Running dry-run for $crate_name..."
    if cargo publish --dry-run --allow-dirty; then
        log_success "$crate_name dry-run successful"
    else
        log_error "$crate_name dry-run failed"
        cd - > /dev/null
        return 1
    fi
    
    # Actual publish (if not dry run mode)
    if [ "$is_dry_run" != "true" ]; then
        log_info "Publishing $crate_name to crates.io..."
        if cargo publish --allow-dirty; then
            log_success "$crate_name published successfully!"
        else
            log_error "$crate_name publishing failed"
            cd - > /dev/null
            return 1
        fi
        
        # Wait before next publish to avoid rate limiting
        log_info "Waiting 10 seconds before next publish..."
        sleep 10
    else
        log_info "Dry-run mode: Skipping actual publish of $crate_name"
    fi
    
    cd - > /dev/null
}

# Main publishing function
main() {
    local dry_run_mode="false"
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --dry-run)
                dry_run_mode="true"
                shift
                ;;
            -h|--help)
                echo "Usage: $0 [--dry-run] [--help]"
                echo "  --dry-run: Run in dry-run mode (no actual publishing)"
                echo "  --help: Show this help message"
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    echo "🧠 Kimi-K2 Integration Crate Publishing Script"
    echo "============================================="
    
    if [ "$dry_run_mode" = "true" ]; then
        log_warning "Running in DRY-RUN mode - no actual publishing will occur"
    fi
    
    # Check prerequisites
    check_prerequisites
    
    # Source Rust environment
    if [ -f "$HOME/.cargo/env" ]; then
        source "$HOME/.cargo/env"
    fi
    
    # Define base directory
    BASE_DIR="/workspaces/Synaptic-Neural-Mesh/standalone-crates"
    
    # Check if base directory exists
    if [ ! -d "$BASE_DIR" ]; then
        log_error "Base directory not found: $BASE_DIR"
        exit 1
    fi
    
    cd "$BASE_DIR"
    log_success "Changed to base directory: $BASE_DIR"
    
    # Define crates to publish in dependency order
    declare -a CRATES=(
        "synaptic-mesh-cli/crates/kimi-expert-analyzer:kimi-expert-analyzer"
        "kimi-fann-core:kimi-fann-core"
    )
    
    # Verify all crates exist and compile
    log_info "Verifying all crates before publishing..."
    for crate_info in "${CRATES[@]}"; do
        crate_path="${crate_info%%:*}"
        crate_name="${crate_info##*:}"
        
        if [ ! -d "$crate_path" ]; then
            log_error "Crate directory not found: $crate_path"
            exit 1
        fi
        
        if ! verify_compilation "$crate_path" "$crate_name"; then
            log_error "Verification failed for $crate_name"
            exit 1
        fi
    done
    
    log_success "All crates verified successfully!"
    
    # Check if dependencies are already published
    log_info "Checking dependency availability..."
    
    # Check if synaptic-neural-wasm is available (required by kimi-fann-core)
    if cargo search synaptic-neural-wasm | grep -q "synaptic-neural-wasm"; then
        log_success "Dependency synaptic-neural-wasm is available on crates.io"
    else
        log_error "Required dependency synaptic-neural-wasm not found on crates.io"
        log_error "Please publish synaptic-neural-wasm first"
        exit 1
    fi
    
    # Publish crates in order
    log_info "Starting publishing process..."
    
    published_count=0
    for crate_info in "${CRATES[@]}"; do
        crate_path="${crate_info%%:*}"
        crate_name="${crate_info##*:}"
        
        log_info "Publishing crate $((published_count + 1))/${#CRATES[@]}: $crate_name"
        
        if publish_crate "$crate_path" "$crate_name" "$dry_run_mode"; then
            ((published_count++))
            log_success "Successfully processed $crate_name"
        else
            log_error "Failed to publish $crate_name"
            exit 1
        fi
    done
    
    # Final summary
    echo ""
    echo "🎉 Publishing Summary"
    echo "==================="
    
    if [ "$dry_run_mode" = "true" ]; then
        log_success "Dry-run completed successfully for $published_count crates"
        log_info "All crates are ready for publishing. Run without --dry-run to publish."
    else
        log_success "Successfully published $published_count Kimi-K2 integration crates!"
        echo ""
        echo "📦 Published Crates:"
        echo "   🔬 kimi-expert-analyzer - Expert analysis tool for Kimi-K2 conversion"
        echo "   🧠 kimi-fann-core - Kimi-K2 micro-expert implementation with WASM support"
        echo ""
        echo "🔗 Next Steps:"
        echo "   - Update documentation with crates.io badges"
        echo "   - Test integration with existing Synaptic ecosystem"
        echo "   - Announce to Rust and AI communities"
        echo "   - Create usage examples and tutorials"
    fi
    
    log_success "Kimi-K2 crate publishing process completed!"
}

# Check if script is being sourced or executed
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi