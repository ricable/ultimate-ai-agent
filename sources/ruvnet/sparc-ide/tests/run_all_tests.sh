#!/bin/bash
# SPARC IDE - Main Test Runner
# This script runs all tests for the SPARC IDE implementation

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
REPORT_DIR="$SCRIPT_DIR/../test-reports"
SUMMARY_FILE="$REPORT_DIR/test_summary_$TIMESTAMP.txt"
HTML_REPORT="$REPORT_DIR/test_report_$TIMESTAMP.html"

# Print colored output
print_info() {
    echo -e "\e[1;34m[INFO]\e[0m $1"
}

print_success() {
    echo -e "\e[1;32m[SUCCESS]\e[0m $1"
}

print_error() {
    echo -e "\e[1;31m[ERROR]\e[0m $1"
}

print_header() {
    echo -e "\e[1;36m===== $1 =====\e[0m"
}

# Initialize test environment
init_test_env() {
    print_info "Initializing test environment..."
    
    # Create report directory if it doesn't exist
    mkdir -p "$REPORT_DIR"
    
    # Initialize summary file
    echo "SPARC IDE Test Summary - $(date)" > "$SUMMARY_FILE"
    echo "===============================" >> "$SUMMARY_FILE"
    
    print_success "Test environment initialized."
}

# Run build script tests
run_build_script_tests() {
    print_header "Running Build Script Tests"
    
    # Run setup-build-environment.sh tests
    print_info "Running setup-build-environment.sh tests..."
    if "$SCRIPT_DIR/build-scripts/setup_build_environment_test.sh" > "$REPORT_DIR/setup_build_environment_test_$TIMESTAMP.log" 2>&1; then
        print_success "setup-build-environment.sh tests passed."
        echo "✅ setup-build-environment.sh tests: PASSED" >> "$SUMMARY_FILE"
        PASSED_TESTS=$((PASSED_TESTS + 1))
    else
        print_error "setup-build-environment.sh tests failed."
        echo "❌ setup-build-environment.sh tests: FAILED" >> "$SUMMARY_FILE"
        FAILED_TESTS=$((FAILED_TESTS + 1))
    fi
    
    # Run build-sparc-ide.sh tests
    print_info "Running build-sparc-ide.sh tests..."
    if "$SCRIPT_DIR/build-scripts/build_sparc_ide_test.sh" > "$REPORT_DIR/build_sparc_ide_test_$TIMESTAMP.log" 2>&1; then
        print_success "build-sparc-ide.sh tests passed."
        echo "✅ build-sparc-ide.sh tests: PASSED" >> "$SUMMARY_FILE"
        PASSED_TESTS=$((PASSED_TESTS + 1))
    else
        print_error "build-sparc-ide.sh tests failed."
        echo "❌ build-sparc-ide.sh tests: FAILED" >> "$SUMMARY_FILE"
        FAILED_TESTS=$((FAILED_TESTS + 1))
    fi
}

# Run Roo Code integration tests
run_roo_code_tests() {
    print_header "Running Roo Code Integration Tests"
    
    # Run download-roo-code.sh tests
    print_info "Running download-roo-code.sh tests..."
    if "$SCRIPT_DIR/roo-code/download_roo_code_test.sh" > "$REPORT_DIR/download_roo_code_test_$TIMESTAMP.log" 2>&1; then
        print_success "download-roo-code.sh tests passed."
        echo "✅ download-roo-code.sh tests: PASSED" >> "$SUMMARY_FILE"
        PASSED_TESTS=$((PASSED_TESTS + 1))
    else
        print_error "download-roo-code.sh tests failed."
        echo "❌ download-roo-code.sh tests: FAILED" >> "$SUMMARY_FILE"
        FAILED_TESTS=$((FAILED_TESTS + 1))
    fi
}

# Run UI configuration tests
run_ui_config_tests() {
    print_header "Running UI Configuration Tests"
    
    # Run configure-ui.sh tests
    print_info "Running configure-ui.sh tests..."
    if "$SCRIPT_DIR/ui-config/configure_ui_test.sh" > "$REPORT_DIR/configure_ui_test_$TIMESTAMP.log" 2>&1; then
        print_success "configure-ui.sh tests passed."
        echo "✅ configure-ui.sh tests: PASSED" >> "$SUMMARY_FILE"
        PASSED_TESTS=$((PASSED_TESTS + 1))
    else
        print_error "configure-ui.sh tests failed."
        echo "❌ configure-ui.sh tests: FAILED" >> "$SUMMARY_FILE"
        FAILED_TESTS=$((FAILED_TESTS + 1))
    fi
}

# Run branding tests
run_branding_tests() {
    print_header "Running Branding Tests"
    
    # Run apply-branding.sh tests
    print_info "Running apply-branding.sh tests..."
    if "$SCRIPT_DIR/branding/apply_branding_test.sh" > "$REPORT_DIR/apply_branding_test_$TIMESTAMP.log" 2>&1; then
        print_success "apply-branding.sh tests passed."
        echo "✅ apply-branding.sh tests: PASSED" >> "$SUMMARY_FILE"
        PASSED_TESTS=$((PASSED_TESTS + 1))
    else
        print_error "apply-branding.sh tests failed."
        echo "❌ apply-branding.sh tests: FAILED" >> "$SUMMARY_FILE"
        FAILED_TESTS=$((FAILED_TESTS + 1))
    fi
}

# Run packaging tests
run_packaging_tests() {
    print_header "Running Packaging Tests"
    
    # Run package-sparc-ide.sh tests
    print_info "Running package-sparc-ide.sh tests..."
    if "$SCRIPT_DIR/packaging/test_package_sparc_ide.sh" > "$REPORT_DIR/test_package_sparc_ide_$TIMESTAMP.log" 2>&1; then
        print_success "package-sparc-ide.sh tests passed."
        echo "✅ package-sparc-ide.sh tests: PASSED" >> "$SUMMARY_FILE"
        PASSED_TESTS=$((PASSED_TESTS + 1))
    else
        print_error "package-sparc-ide.sh tests failed."
        echo "❌ package-sparc-ide.sh tests: FAILED" >> "$SUMMARY_FILE"
        FAILED_TESTS=$((FAILED_TESTS + 1))
    fi
}

# Generate HTML report
generate_html_report() {
    print_header "Generating HTML Report"
    
    # Create HTML report
    cat > "$HTML_REPORT" << EOL
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
        <p><strong>Total Tests:</strong> $((PASSED_TESTS + FAILED_TESTS))</p>
        <p><strong>Passed:</strong> <span class="success">$PASSED_TESTS</span></p>
        <p><strong>Failed:</strong> <span class="failure">$FAILED_TESTS</span></p>
    </div>
    
    <h2>Test Results</h2>
    <table>
        <tr>
            <th>Test</th>
            <th>Result</th>
            <th>Log</th>
        </tr>
EOL
    
    # Add build script test results
    echo "<tr><td>setup-build-environment.sh</td><td class=\"$(grep -q "setup-build-environment.sh tests: PASSED" "$SUMMARY_FILE" && echo "success" || echo "failure")\">$(grep -q "setup-build-environment.sh tests: PASSED" "$SUMMARY_FILE" && echo "Passed" || echo "Failed")</td><td><a href=\"setup_build_environment_test_$TIMESTAMP.log\">View Log</a></td></tr>" >> "$HTML_REPORT"
    echo "<tr><td>build-sparc-ide.sh</td><td class=\"$(grep -q "build-sparc-ide.sh tests: PASSED" "$SUMMARY_FILE" && echo "success" || echo "failure")\">$(grep -q "build-sparc-ide.sh tests: PASSED" "$SUMMARY_FILE" && echo "Passed" || echo "Failed")</td><td><a href=\"build_sparc_ide_test_$TIMESTAMP.log\">View Log</a></td></tr>" >> "$HTML_REPORT"
    
    # Add Roo Code integration test results
    echo "<tr><td>download-roo-code.sh</td><td class=\"$(grep -q "download-roo-code.sh tests: PASSED" "$SUMMARY_FILE" && echo "success" || echo "failure")\">$(grep -q "download-roo-code.sh tests: PASSED" "$SUMMARY_FILE" && echo "Passed" || echo "Failed")</td><td><a href=\"download_roo_code_test_$TIMESTAMP.log\">View Log</a></td></tr>" >> "$HTML_REPORT"
    
    # Add UI configuration test results
    echo "<tr><td>configure-ui.sh</td><td class=\"$(grep -q "configure-ui.sh tests: PASSED" "$SUMMARY_FILE" && echo "success" || echo "failure")\">$(grep -q "configure-ui.sh tests: PASSED" "$SUMMARY_FILE" && echo "Passed" || echo "Failed")</td><td><a href=\"configure_ui_test_$TIMESTAMP.log\">View Log</a></td></tr>" >> "$HTML_REPORT"
    
    # Add branding test results
    echo "<tr><td>apply-branding.sh</td><td class=\"$(grep -q "apply-branding.sh tests: PASSED" "$SUMMARY_FILE" && echo "success" || echo "failure")\">$(grep -q "apply-branding.sh tests: PASSED" "$SUMMARY_FILE" && echo "Passed" || echo "Failed")</td><td><a href=\"apply_branding_test_$TIMESTAMP.log\">View Log</a></td></tr>" >> "$HTML_REPORT"
    
    # Add packaging test results
    echo "<tr><td>package-sparc-ide.sh</td><td class=\"$(grep -q "package-sparc-ide.sh tests: PASSED" "$SUMMARY_FILE" && echo "success" || echo "failure")\">$(grep -q "package-sparc-ide.sh tests: PASSED" "$SUMMARY_FILE" && echo "Passed" || echo "Failed")</td><td><a href=\"test_package_sparc_ide_$TIMESTAMP.log\">View Log</a></td></tr>" >> "$HTML_REPORT"
    
    # Close HTML tags
    cat >> "$HTML_REPORT" << EOL
    </table>
    
    <h2>Summary</h2>
    <pre>$(cat "$SUMMARY_FILE")</pre>
</body>
</html>
EOL
    
    print_success "HTML report generated: $HTML_REPORT"
}

# Main function
main() {
    print_header "SPARC IDE Test Suite"
    
    # Initialize counters
    PASSED_TESTS=0
    FAILED_TESTS=0
    
    # Initialize test environment
    init_test_env
    
    # Run all test categories
    run_build_script_tests
    run_roo_code_tests
    run_ui_config_tests
    run_branding_tests
    run_packaging_tests
    
    # Generate HTML report
    generate_html_report
    
    # Print summary
    print_header "Test Summary"
    cat "$SUMMARY_FILE"
    
    print_info "Total tests: $((PASSED_TESTS + FAILED_TESTS))"
    print_success "Passed: $PASSED_TESTS"
    
    if [ $FAILED_TESTS -gt 0 ]; then
        print_error "Failed: $FAILED_TESTS"
        exit 1
    else
        print_success "All tests passed!"
    fi
}

# Run main function
main