#!/bin/bash
# SPARC IDE Test Runner
# This script runs the test suite for SPARC IDE

set -e

# Configuration
TEST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPORT_DIR="$TEST_DIR/../test-reports"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$REPORT_DIR/test_run_$TIMESTAMP.log"

# Print colored output
print_info() {
    echo -e "\e[1;34m[INFO]\e[0m $1" | tee -a "$LOG_FILE"
}

print_success() {
    echo -e "\e[1;32m[SUCCESS]\e[0m $1" | tee -a "$LOG_FILE"
}

print_error() {
    echo -e "\e[1;31m[ERROR]\e[0m $1" | tee -a "$LOG_FILE"
}

print_header() {
    echo -e "\e[1;36m===== $1 =====\e[0m" | tee -a "$LOG_FILE"
}

# Initialize test environment
init_test_env() {
    print_info "Initializing test environment..."
    
    # Create report directory if it doesn't exist
    mkdir -p "$REPORT_DIR"
    
    # Initialize log file
    echo "SPARC IDE Test Run - $(date)" > "$LOG_FILE"
    echo "===============================" >> "$LOG_FILE"
    
    print_success "Test environment initialized."
}

# Run build script tests
run_build_script_tests() {
    print_header "Running Build Script Tests"
    
    if [ -d "$TEST_DIR/build-scripts" ]; then
        for test_file in "$TEST_DIR/build-scripts"/*_test.sh; do
            if [ -f "$test_file" ]; then
                print_info "Running test: $(basename "$test_file")"
                bash "$test_file" 2>&1 | tee -a "$LOG_FILE"
                
                # Check if test passed (last command in the pipe)
                if [ ${PIPESTATUS[0]} -eq 0 ]; then
                    print_success "Test passed: $(basename "$test_file")"
                else
                    print_error "Test failed: $(basename "$test_file")"
                    FAILED_TESTS+=("$(basename "$test_file")")
                fi
            fi
        done
    else
        print_info "No build script tests found."
    fi
}

# Run Roo Code integration tests
run_roo_code_tests() {
    print_header "Running Roo Code Integration Tests"
    
    if [ -d "$TEST_DIR/roo-code" ]; then
        for test_file in "$TEST_DIR/roo-code"/*_test.sh; do
            if [ -f "$test_file" ]; then
                print_info "Running test: $(basename "$test_file")"
                bash "$test_file" 2>&1 | tee -a "$LOG_FILE"
                
                # Check if test passed (last command in the pipe)
                if [ ${PIPESTATUS[0]} -eq 0 ]; then
                    print_success "Test passed: $(basename "$test_file")"
                else
                    print_error "Test failed: $(basename "$test_file")"
                    FAILED_TESTS+=("$(basename "$test_file")")
                fi
            fi
        done
    else
        print_info "No Roo Code integration tests found."
    fi
}

# Run UI configuration tests
run_ui_config_tests() {
    print_header "Running UI Configuration Tests"
    
    if [ -d "$TEST_DIR/ui-config" ]; then
        for test_file in "$TEST_DIR/ui-config"/*_test.sh; do
            if [ -f "$test_file" ]; then
                print_info "Running test: $(basename "$test_file")"
                bash "$test_file" 2>&1 | tee -a "$LOG_FILE"
                
                # Check if test passed (last command in the pipe)
                if [ ${PIPESTATUS[0]} -eq 0 ]; then
                    print_success "Test passed: $(basename "$test_file")"
                else
                    print_error "Test failed: $(basename "$test_file")"
                    FAILED_TESTS+=("$(basename "$test_file")")
                fi
            fi
        done
    else
        print_info "No UI configuration tests found."
    fi
}

# Run branding tests
run_branding_tests() {
    print_header "Running Branding Tests"
    
    if [ -d "$TEST_DIR/branding" ]; then
        for test_file in "$TEST_DIR/branding"/*_test.sh; do
            if [ -f "$test_file" ]; then
                print_info "Running test: $(basename "$test_file")"
                bash "$test_file" 2>&1 | tee -a "$LOG_FILE"
                
                # Check if test passed (last command in the pipe)
                if [ ${PIPESTATUS[0]} -eq 0 ]; then
                    print_success "Test passed: $(basename "$test_file")"
                else
                    print_error "Test failed: $(basename "$test_file")"
                    FAILED_TESTS+=("$(basename "$test_file")")
                fi
            fi
        done
    else
        print_info "No branding tests found."
    fi
}

# Generate test report
generate_report() {
    print_header "Generating Test Report"
    
    REPORT_FILE="$REPORT_DIR/test_report_$TIMESTAMP.html"
    
    # Create HTML report
    cat > "$REPORT_FILE" << EOL
<!DOCTYPE html>
<html>
<head>
    <title>SPARC IDE Test Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        h1 { color: #333; }
        .summary { margin: 20px 0; padding: 10px; background-color: #f5f5f5; border-radius: 5px; }
        .success { color: green; }
        .failure { color: red; }
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        tr:nth-child(even) { background-color: #f9f9f9; }
    </style>
</head>
<body>
    <h1>SPARC IDE Test Report</h1>
    <div class="summary">
        <p><strong>Date:</strong> $(date)</p>
        <p><strong>Total Tests:</strong> ${#ALL_TESTS[@]}</p>
        <p><strong>Passed:</strong> <span class="success">${#PASSED_TESTS[@]}</span></p>
        <p><strong>Failed:</strong> <span class="failure">${#FAILED_TESTS[@]}</span></p>
    </div>
    
    <h2>Test Results</h2>
    <table>
        <tr>
            <th>Test</th>
            <th>Result</th>
        </tr>
EOL
    
    # Add test results to the report
    for test in "${ALL_TESTS[@]}"; do
        if [[ " ${FAILED_TESTS[*]} " == *" $test "* ]]; then
            echo "<tr><td>$test</td><td class=\"failure\">Failed</td></tr>" >> "$REPORT_FILE"
        else
            echo "<tr><td>$test</td><td class=\"success\">Passed</td></tr>" >> "$REPORT_FILE"
        fi
    done
    
    # Close HTML tags
    cat >> "$REPORT_FILE" << EOL
    </table>
    
    <h2>Log</h2>
    <pre>$(cat "$LOG_FILE")</pre>
</body>
</html>
EOL
    
    print_success "Test report generated: $REPORT_FILE"
}

# Main function
main() {
    # Initialize arrays to track test results
    FAILED_TESTS=()
    PASSED_TESTS=()
    ALL_TESTS=()
    
    print_header "SPARC IDE Test Suite"
    
    init_test_env
    
    # Check if a specific test category was requested
    if [ $# -gt 0 ]; then
        case "$1" in
            build-scripts)
                run_build_script_tests
                ;;
            roo-code)
                run_roo_code_tests
                ;;
            ui-config)
                run_ui_config_tests
                ;;
            branding)
                run_branding_tests
                ;;
            *)
                print_error "Unknown test category: $1"
                echo "Available categories: build-scripts, roo-code, ui-config, branding"
                exit 1
                ;;
        esac
    else
        # Run all test categories
        run_build_script_tests
        run_roo_code_tests
        run_ui_config_tests
        run_branding_tests
    fi
    
    # Calculate passed tests
    for test in "${ALL_TESTS[@]}"; do
        if [[ ! " ${FAILED_TESTS[*]} " == *" $test "* ]]; then
            PASSED_TESTS+=("$test")
        fi
    done
    
    generate_report
    
    # Print summary
    print_header "Test Summary"
    print_info "Total tests: ${#ALL_TESTS[@]}"
    print_success "Passed: ${#PASSED_TESTS[@]}"
    
    if [ ${#FAILED_TESTS[@]} -gt 0 ]; then
        print_error "Failed: ${#FAILED_TESTS[@]}"
        print_error "Failed tests: ${FAILED_TESTS[*]}"
        exit 1
    else
        print_success "All tests passed!"
    fi
}

# Run main function with all arguments
main "$@"