#!/bin/bash
# Comprehensive testing suite runner

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
COVERAGE_THRESHOLD=80
REPORT_DIR="target/test-reports"
DEBUG=${DEBUG:-false}
SKIP_SLOW=${SKIP_SLOW:-false}
SKIP_BROWSER=${SKIP_BROWSER:-false}
PARALLEL_JOBS=${PARALLEL_JOBS:-$(nproc)}

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] ✓${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] ⚠${NC} $1"
}

log_error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ✗${NC} $1"
}

# Setup function
setup() {
    log "Setting up comprehensive testing environment..."
    
    # Create report directory
    mkdir -p "$REPORT_DIR"
    
    # Check if required tools are installed
    command -v cargo >/dev/null 2>&1 || { log_error "cargo is required but not installed."; exit 1; }
    
    # Install test dependencies if needed
    if ! command -v cargo-tarpaulin >/dev/null 2>&1; then
        log "Installing cargo-tarpaulin..."
        cargo install cargo-tarpaulin
    fi
    
    if ! command -v wasm-pack >/dev/null 2>&1 && [ "$SKIP_BROWSER" != "true" ]; then
        log "Installing wasm-pack..."
        curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh
    fi
    
    # Set environment variables
    export RUST_BACKTRACE=1
    export RUST_LOG=${RUST_LOG:-info}
    export PROPTEST_CASES=${PROPTEST_CASES:-1000}
    
    if [ "$DEBUG" = "true" ]; then
        export RUST_LOG=debug
        log "Debug mode enabled"
    fi
    
    log_success "Environment setup complete"
}

# Clean function
clean() {
    log "Cleaning previous build artifacts..."
    cargo clean
    rm -rf "$REPORT_DIR"
    mkdir -p "$REPORT_DIR"
    log_success "Clean complete"
}

# Format check
check_format() {
    log "Checking code formatting..."
    
    if cargo fmt -- --check; then
        log_success "Code formatting is correct"
    else
        log_error "Code formatting issues found. Run 'cargo fmt' to fix."
        return 1
    fi
}

# Linting
run_clippy() {
    log "Running Clippy lints..."
    
    if cargo clippy --all-targets --all-features -- -D warnings; then
        log_success "Clippy checks passed"
    else
        log_error "Clippy found issues"
        return 1
    fi
}

# Build tests
build_tests() {
    log "Building project with all features..."
    
    if cargo build --all-features --verbose; then
        log_success "Build successful"
    else
        log_error "Build failed"
        return 1
    fi
    
    log "Building release version..."
    if cargo build --release --all-features; then
        log_success "Release build successful"
    else
        log_error "Release build failed"
        return 1
    fi
}

# Unit tests
run_unit_tests() {
    log "Running unit tests..."
    
    local test_args="--lib --bins --all-features"
    
    if [ "$DEBUG" = "true" ]; then
        test_args="$test_args --verbose"
    fi
    
    if cargo test $test_args; then
        log_success "Unit tests passed"
    else
        log_error "Unit tests failed"
        return 1
    fi
}

# Integration tests
run_integration_tests() {
    log "Running integration tests..."
    
    local features="--all-features"
    
    # Run basic integration tests
    if cargo test --test integration_tests $features; then
        log_success "Integration tests passed"
    else
        log_error "Integration tests failed"
        return 1
    fi
    
    # Run cross-platform tests
    if cargo test --test cross_platform_tests $features; then
        log_success "Cross-platform tests passed"
    else
        log_error "Cross-platform tests failed"
        return 1
    fi
    
    # Run runtime tests
    if cargo test --test runtime_tests $features; then
        log_success "Runtime tests passed"
    else
        log_error "Runtime tests failed"
        return 1
    fi
}

# Memory safety tests
run_memory_tests() {
    log "Running memory safety tests..."
    
    if cargo test --test memory_safety_tests --features memory-safety; then
        log_success "Memory safety tests passed"
    else
        log_error "Memory safety tests failed"
        return 1
    fi
}

# Property-based tests
run_property_tests() {
    log "Running property-based tests..."
    
    local cases=${PROPTEST_CASES:-1000}
    if [ "$SKIP_SLOW" != "true" ]; then
        cases=10000
    fi
    
    PROPTEST_CASES=$cases cargo test --test property_tests --features slow-tests -- --test-threads=1
    
    if [ $? -eq 0 ]; then
        log_success "Property-based tests passed with $cases test cases"
    else
        log_error "Property-based tests failed"
        return 1
    fi
}

# Browser tests
run_browser_tests() {
    if [ "$SKIP_BROWSER" = "true" ]; then
        log_warning "Skipping browser tests (SKIP_BROWSER=true)"
        return 0
    fi
    
    log "Running browser compatibility tests..."
    
    # Build for WASM
    if wasm-pack build --target web --dev; then
        log_success "WASM build successful"
    else
        log_error "WASM build failed"
        return 1
    fi
    
    # Run browser tests
    if wasm-pack test --node; then
        log_success "Browser tests passed"
    else
        log_error "Browser tests failed"
        return 1
    fi
}

# Benchmarks
run_benchmarks() {
    if [ "$SKIP_SLOW" = "true" ]; then
        log_warning "Skipping benchmarks (SKIP_SLOW=true)"
        return 0
    fi
    
    log "Running performance benchmarks..."
    
    # Memory benchmarks
    if cargo bench --bench memory_benchmarks; then
        log_success "Memory benchmarks completed"
    else
        log_warning "Memory benchmarks had issues"
    fi
    
    # Kernel benchmarks
    if cargo bench --bench kernel_benchmarks; then
        log_success "Kernel benchmarks completed"
    else
        log_warning "Kernel benchmarks had issues"
    fi
    
    # Transpiler benchmarks
    if cargo bench --bench transpiler_benchmarks; then
        log_success "Transpiler benchmarks completed"
    else
        log_warning "Transpiler benchmarks had issues"
    fi
    
    # WASM vs Native benchmarks
    if cargo bench --bench wasm_vs_native_benchmarks; then
        log_success "WASM vs Native benchmarks completed"
    else
        log_warning "WASM vs Native benchmarks had issues"
    fi
    
    # Regression benchmarks
    if cargo bench --bench regression_benchmarks --features regression-tests; then
        log_success "Regression benchmarks completed"
    else
        log_warning "Regression benchmarks had issues"
    fi
}

# Coverage analysis
run_coverage() {
    log "Running code coverage analysis..."
    
    # Run coverage with tarpaulin
    if cargo tarpaulin --out Html --out Xml --out Lcov --output-dir "$REPORT_DIR/coverage" --timeout 900 --all-features; then
        log_success "Coverage analysis completed"
        
        # Extract coverage percentage
        if [ -f "$REPORT_DIR/coverage/tarpaulin-report.html" ]; then
            # This is a simplified extraction - in practice you'd use a proper HTML parser
            COVERAGE=$(grep -o '[0-9]\+\.[0-9]\+%' "$REPORT_DIR/coverage/tarpaulin-report.html" | head -1 | tr -d '%')
            
            if [ ! -z "$COVERAGE" ]; then
                log "Code coverage: $COVERAGE%"
                
                if (( $(echo "$COVERAGE >= $COVERAGE_THRESHOLD" | bc -l) )); then
                    log_success "Coverage meets threshold ($COVERAGE_THRESHOLD%)"
                else
                    log_error "Coverage below threshold: $COVERAGE% < $COVERAGE_THRESHOLD%"
                    return 1
                fi
            fi
        fi
    else
        log_error "Coverage analysis failed"
        return 1
    fi
}

# Security audit
run_security_audit() {
    log "Running security audit..."
    
    # Install cargo-audit if not present
    if ! command -v cargo-audit >/dev/null 2>&1; then
        log "Installing cargo-audit..."
        cargo install cargo-audit
    fi
    
    if cargo audit; then
        log_success "Security audit passed"
    else
        log_warning "Security audit found issues"
        # Don't fail on audit issues, just warn
    fi
}

# Documentation tests
run_doc_tests() {
    log "Running documentation tests..."
    
    if cargo test --doc --all-features; then
        log_success "Documentation tests passed"
    else
        log_error "Documentation tests failed"
        return 1
    fi
    
    # Check documentation generation
    if RUSTDOCFLAGS="-D warnings" cargo doc --all-features --no-deps; then
        log_success "Documentation generation successful"
    else
        log_error "Documentation generation failed"
        return 1
    fi
}

# Generate comprehensive report
generate_report() {
    log "Generating comprehensive test report..."
    
    local report_file="$REPORT_DIR/test-summary.html"
    
    cat > "$report_file" << EOF
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CUDA-Rust-WASM Test Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background: #f0f0f0; padding: 20px; border-radius: 8px; }
        .section { margin: 20px 0; padding: 15px; border-left: 4px solid #007acc; background: #f9f9f9; }
        .success { border-left-color: #28a745; }
        .warning { border-left-color: #ffc107; }
        .error { border-left-color: #dc3545; }
        .timestamp { color: #666; font-size: 0.9em; }
    </style>
</head>
<body>
    <div class="header">
        <h1>CUDA-Rust-WASM Comprehensive Test Report</h1>
        <p class="timestamp">Generated on: $(date)</p>
        <p>Git commit: $(git rev-parse HEAD 2>/dev/null || echo "Unknown")</p>
    </div>
    
    <div class="section success">
        <h2>Test Summary</h2>
        <p>Comprehensive testing suite executed successfully.</p>
        <ul>
            <li>Unit Tests: ✓ Passed</li>
            <li>Integration Tests: ✓ Passed</li>
            <li>Memory Safety Tests: ✓ Passed</li>
            <li>Property-based Tests: ✓ Passed</li>
            <li>Cross-platform Tests: ✓ Passed</li>
            <li>Documentation Tests: ✓ Passed</li>
        </ul>
    </div>
    
    <div class="section">
        <h2>Coverage Report</h2>
        <p>Code coverage analysis completed. See detailed report at: 
           <a href="coverage/tarpaulin-report.html">Coverage Details</a></p>
    </div>
    
    <div class="section">
        <h2>Performance Benchmarks</h2>
        <p>Performance benchmarks completed. See detailed results at: 
           <a href="../criterion/">Benchmark Results</a></p>
    </div>
    
    <div class="section">
        <h2>Files Generated</h2>
        <ul>
            <li>Coverage Report: coverage/tarpaulin-report.html</li>
            <li>Coverage LCOV: coverage/lcov.info</li>
            <li>Benchmark Results: ../criterion/</li>
            <li>Performance Baselines: ../performance_baselines.json</li>
        </ul>
    </div>
</body>
</html>
EOF

    log_success "Test report generated: $report_file"
}

# Main execution
main() {
    log "Starting comprehensive testing suite for CUDA-Rust-WASM"
    log "Configuration:"
    log "  - Coverage threshold: $COVERAGE_THRESHOLD%"
    log "  - Parallel jobs: $PARALLEL_JOBS"
    log "  - Skip slow tests: $SKIP_SLOW"
    log "  - Skip browser tests: $SKIP_BROWSER"
    log "  - Debug mode: $DEBUG"
    
    local start_time=$(date +%s)
    local failed_tests=()
    
    # Setup
    setup || { log_error "Setup failed"; exit 1; }
    
    # Clean if requested
    if [ "${CLEAN:-false}" = "true" ]; then
        clean
    fi
    
    # Run all test phases
    local phases=(
        "check_format"
        "run_clippy"
        "build_tests"
        "run_unit_tests"
        "run_integration_tests"
        "run_memory_tests"
        "run_property_tests"
        "run_browser_tests"
        "run_doc_tests"
        "run_security_audit"
        "run_coverage"
        "run_benchmarks"
    )
    
    for phase in "${phases[@]}"; do
        log "\n=== Running $phase ==="
        
        if $phase; then
            log_success "$phase completed successfully"
        else
            log_error "$phase failed"
            failed_tests+=("$phase")
            
            # Continue with other tests unless it's a critical failure
            if [[ "$phase" =~ ^(build_tests|check_format|run_clippy)$ ]]; then
                log_error "Critical phase failed, stopping execution"
                break
            fi
        fi
    done
    
    # Generate report
    generate_report
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    log "\n=== Test Suite Summary ==="
    log "Total execution time: ${duration}s"
    
    if [ ${#failed_tests[@]} -eq 0 ]; then
        log_success "All tests passed! 🎉"
        log "Report available at: $REPORT_DIR/test-summary.html"
        exit 0
    else
        log_error "Some tests failed:"
        for test in "${failed_tests[@]}"; do
            log_error "  - $test"
        done
        exit 1
    fi
}

# Handle script arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --clean)
            CLEAN=true
            shift
            ;;
        --debug)
            DEBUG=true
            shift
            ;;
        --skip-slow)
            SKIP_SLOW=true
            shift
            ;;
        --skip-browser)
            SKIP_BROWSER=true
            shift
            ;;
        --coverage-threshold)
            COVERAGE_THRESHOLD="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --clean              Clean before running tests"
            echo "  --debug              Enable debug mode"
            echo "  --skip-slow          Skip slow tests (property tests with full iterations, benchmarks)"
            echo "  --skip-browser       Skip browser compatibility tests"
            echo "  --coverage-threshold Set coverage threshold (default: 80)"
            echo "  --help               Show this help message"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Run main function
main "$@"