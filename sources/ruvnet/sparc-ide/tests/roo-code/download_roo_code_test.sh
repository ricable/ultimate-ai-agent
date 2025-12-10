#!/bin/bash
# Test for download-roo-code.sh script

# Source test utilities
source "$(dirname "$0")/../helpers/test_utils.sh"

# Test functions
test_check_prerequisites() {
    print_test "Testing check_prerequisites function"
    
    # Create a test copy of the script
    local test_script=$(create_test_script_copy "$SPARC_IDE_ROOT/scripts/download-roo-code.sh")
    
    # Create a test directory structure
    local test_dir=$(create_test_directory_structure)
    cd "$test_dir"
    
    # Extract the check_prerequisites function from the script
    local check_prerequisites_function=$(grep -A 20 "check_prerequisites()" "$test_script" | sed -n '/check_prerequisites()/,/}/p')
    
    # Create a test script with just the check_prerequisites function
    local test_function_script="$TEMP_DIR/check_prerequisites.sh"
    cat > "$test_function_script" << EOL
#!/bin/bash
set -e

# Configuration
EXTENSIONS_DIR="extensions"

# Print colored output
print_info() {
    echo -e "\e[1;34m[INFO]\e[0m \$1"
}

print_success() {
    echo -e "\e[1;32m[SUCCESS]\e[0m \$1"
}

print_error() {
    echo -e "\e[1;31m[ERROR]\e[0m \$1"
}

# Mock command function
command() {
    if [[ "\$1" == "-v" && "\$2" == "curl" ]]; then
        return \$CURL_INSTALLED
    fi
}

$check_prerequisites_function

# Run the function
check_prerequisites
EOL
    
    # Make the test script executable
    chmod +x "$test_function_script"
    
    # Test with curl installed
    export CURL_INSTALLED=0
    local output=$("$test_function_script" 2>&1)
    
    # Assert that the function succeeds
    assert_command_succeeds "$test_function_script" "check_prerequisites function should succeed with curl installed"
    
    # Assert that the output contains success message
    assert_contains "$output" "All prerequisites are met" "Output should indicate that all prerequisites are met"
    
    # Test with curl not installed
    export CURL_INSTALLED=1
    output=$("$test_function_script" 2>&1 || true)
    assert_contains "$output" "curl is not installed" "Output should indicate that curl is not installed"
    
    # Test extensions directory creation
    rm -rf "extensions"
    export CURL_INSTALLED=0
    output=$("$test_function_script" 2>&1)
    assert_directory_exists "extensions" "Extensions directory should be created if it doesn't exist"
    
    return 0
}

test_download_roo_code() {
    print_test "Testing download_roo_code function"
    
    # Create a test copy of the script
    local test_script=$(create_test_script_copy "$SPARC_IDE_ROOT/scripts/download-roo-code.sh")
    
    # Create a test directory structure
    local test_dir=$(create_test_directory_structure)
    cd "$test_dir"
    
    # Create extensions directory
    mkdir -p "extensions"
    
    # Extract the download_roo_code function from the script
    local download_roo_code_function=$(grep -A 30 "download_roo_code()" "$test_script" | sed -n '/download_roo_code()/,/}/p')
    
    # Create a test script with just the download_roo_code function
    local test_function_script="$TEMP_DIR/download_roo_code.sh"
    cat > "$test_function_script" << EOL
#!/bin/bash
set -e

# Configuration
EXTENSIONS_DIR="extensions"
ROO_CODE_PUBLISHER="RooVeterinaryInc"
ROO_CODE_EXTENSION="roo-cline"
ROO_CODE_FILENAME="roo-code.vsix"
MARKETPLACE_URL="https://marketplace.visualstudio.com/_apis/public/gallery/publishers"

# Print colored output
print_info() {
    echo -e "\e[1;34m[INFO]\e[0m \$1"
}

print_success() {
    echo -e "\e[1;32m[SUCCESS]\e[0m \$1"
}

print_error() {
    echo -e "\e[1;31m[ERROR]\e[0m \$1"
}

# Mock curl command
curl() {
    if [[ "\$1" == "-L" && "\$2" == "-o" ]]; then
        # Create a dummy VSIX file
        echo "Mock VSIX content" > "\$3"
        return \$CURL_EXIT_CODE
    fi
}

$download_roo_code_function

# Run the function
download_roo_code
EOL
    
    # Make the test script executable
    chmod +x "$test_function_script"
    
    # Test successful download
    export CURL_EXIT_CODE=0
    local output=$("$test_function_script" 2>&1)
    
    # Assert that the function succeeds
    assert_command_succeeds "$test_function_script" "download_roo_code function should succeed with successful download"
    
    # Assert that the output contains success message
    assert_contains "$output" "Roo Code extension downloaded successfully" "Output should indicate that Roo Code extension was downloaded successfully"
    
    # Assert that the VSIX file was created
    assert_file_exists "extensions/roo-code.vsix" "Roo Code VSIX file should be created"
    
    # Test failed download
    export CURL_EXIT_CODE=1
    rm -f "extensions/roo-code.vsix"
    output=$("$test_function_script" 2>&1 || true)
    assert_contains "$output" "Failed to download Roo Code extension" "Output should indicate that download failed"
    
    return 0
}

test_verify_extension() {
    print_test "Testing verify_extension function"
    
    # Create a test copy of the script
    local test_script=$(create_test_script_copy "$SPARC_IDE_ROOT/scripts/download-roo-code.sh")
    
    # Create a test directory structure
    local test_dir=$(create_test_directory_structure)
    cd "$test_dir"
    
    # Create extensions directory
    mkdir -p "extensions"
    
    # Extract the verify_extension function from the script
    local verify_extension_function=$(grep -A 30 "verify_extension()" "$test_script" | sed -n '/verify_extension()/,/}/p')
    
    # Create a test script with just the verify_extension function
    local test_function_script="$TEMP_DIR/verify_extension.sh"
    cat > "$test_function_script" << EOL
#!/bin/bash
set -e

# Configuration
EXTENSIONS_DIR="extensions"
ROO_CODE_FILENAME="roo-code.vsix"

# Print colored output
print_info() {
    echo -e "\e[1;34m[INFO]\e[0m \$1"
}

print_success() {
    echo -e "\e[1;32m[SUCCESS]\e[0m \$1"
}

print_error() {
    echo -e "\e[1;31m[ERROR]\e[0m \$1"
}

# Mock file command
file() {
    if [[ "\$1" == "extensions/roo-code.vsix" ]]; then
        if [[ "\$VALID_VSIX" == "1" ]]; then
            echo "extensions/roo-code.vsix: Zip archive data, at least v2.0 to extract"
        else
            echo "extensions/roo-code.vsix: ASCII text"
        fi
    fi
}

$verify_extension_function

# Run the function
verify_extension
EOL
    
    # Make the test script executable
    chmod +x "$test_function_script"
    
    # Create a valid VSIX file
    echo "Mock VSIX content" > "extensions/roo-code.vsix"
    
    # Test with valid VSIX file
    export VALID_VSIX=1
    local output=$("$test_function_script" 2>&1)
    
    # Assert that the function succeeds
    assert_command_succeeds "$test_function_script" "verify_extension function should succeed with valid VSIX file"
    
    # Assert that the output contains success message
    assert_contains "$output" "Roo Code extension verified successfully" "Output should indicate that Roo Code extension was verified successfully"
    
    # Test with invalid VSIX file
    export VALID_VSIX=0
    output=$("$test_function_script" 2>&1 || true)
    assert_contains "$output" "Downloaded file is not a valid VSIX package" "Output should indicate that VSIX file is invalid"
    
    # Test with missing VSIX file
    rm -f "extensions/roo-code.vsix"
    output=$("$test_function_script" 2>&1 || true)
    assert_contains "$output" "Downloaded file is empty or does not exist" "Output should indicate that VSIX file is missing"
    
    return 0
}

test_configure_roo_code() {
    print_test "Testing configure_roo_code function"
    
    # Create a test copy of the script
    local test_script=$(create_test_script_copy "$SPARC_IDE_ROOT/scripts/download-roo-code.sh")
    
    # Create a test directory structure
    local test_dir=$(create_test_directory_structure)
    cd "$test_dir"
    
    # Extract the configure_roo_code function from the script
    local configure_roo_code_function=$(grep -A 30 "configure_roo_code()" "$test_script" | sed -n '/configure_roo_code()/,/}/p')
    
    # Create a test script with just the configure_roo_code function
    local test_function_script="$TEMP_DIR/configure_roo_code.sh"
    cat > "$test_function_script" << EOL
#!/bin/bash
set -e

# Print colored output
print_info() {
    echo -e "\e[1;34m[INFO]\e[0m \$1"
}

print_success() {
    echo -e "\e[1;32m[SUCCESS]\e[0m \$1"
}

print_error() {
    echo -e "\e[1;31m[ERROR]\e[0m \$1"
}

$configure_roo_code_function

# Run the function
configure_roo_code
EOL
    
    # Make the test script executable
    chmod +x "$test_function_script"
    
    # Create config directory and settings.json
    mkdir -p "src/config"
    echo "{}" > "src/config/settings.json"
    
    # Test with settings.json present
    local output=$("$test_function_script" 2>&1)
    
    # Assert that the function succeeds
    assert_command_succeeds "$test_function_script" "configure_roo_code function should succeed with settings.json present"
    
    # Assert that the output contains success message
    assert_contains "$output" "Roo Code integration configured successfully" "Output should indicate that Roo Code integration was configured successfully"
    
    # Test with missing settings.json
    rm -f "src/config/settings.json"
    output=$("$test_function_script" 2>&1 || true)
    assert_contains "$output" "settings.json not found" "Output should indicate that settings.json is missing"
    
    # Test with missing config directory
    rm -rf "src/config"
    mkdir -p "src"
    output=$("$test_function_script" 2>&1 || true)
    assert_contains "$output" "Creating config directory" "Output should indicate that config directory is being created"
    assert_directory_exists "src/config" "Config directory should be created if it doesn't exist"
    
    return 0
}

test_main_function() {
    print_test "Testing main function"
    
    # Create a test copy of the script
    local test_script=$(create_test_script_copy "$SPARC_IDE_ROOT/scripts/download-roo-code.sh")
    
    # Create a test directory structure
    local test_dir=$(create_test_directory_structure)
    cd "$test_dir"
    
    # Create config directory and settings.json
    mkdir -p "src/config"
    echo "{}" > "src/config/settings.json"
    
    # Mock functions
    cat > "$TEMP_DIR/mock_functions.sh" << EOL
#!/bin/bash

# Mock check_prerequisites
check_prerequisites() {
    echo "Checking prerequisites..."
    echo "All prerequisites are met."
}

# Mock download_roo_code
download_roo_code() {
    echo "Downloading Roo Code extension..."
    mkdir -p "extensions"
    echo "Mock VSIX content" > "extensions/roo-code.vsix"
    echo "Roo Code extension downloaded successfully."
}

# Mock verify_extension
verify_extension() {
    echo "Verifying Roo Code extension..."
    echo "Roo Code extension verified successfully."
}

# Mock configure_roo_code
configure_roo_code() {
    echo "Configuring Roo Code integration..."
    echo "Roo Code integration configured successfully."
}
EOL
    
    # Create a modified test script that sources the mock functions
    local modified_test_script="$TEMP_DIR/modified_download_roo_code.sh"
    cat "$test_script" | sed '/^main$/i source "$TEMP_DIR/mock_functions.sh"' > "$modified_test_script"
    chmod +x "$modified_test_script"
    
    # Run the test script
    local output=$("$modified_test_script" 2>&1)
    
    # Assert that the script succeeds
    assert_command_succeeds "$modified_test_script" "download-roo-code.sh script should succeed"
    
    # Assert that the output contains success message
    assert_contains "$output" "Roo Code integration set up successfully" "Output should indicate that Roo Code integration was set up successfully"
    
    # Assert that all functions were called
    assert_contains "$output" "Checking prerequisites" "check_prerequisites function should be called"
    assert_contains "$output" "Downloading Roo Code extension" "download_roo_code function should be called"
    assert_contains "$output" "Verifying Roo Code extension" "verify_extension function should be called"
    assert_contains "$output" "Configuring Roo Code integration" "configure_roo_code function should be called"
    
    return 0
}

# Run tests
run_test "Check Prerequisites" test_check_prerequisites
run_test "Download Roo Code" test_download_roo_code
run_test "Verify Extension" test_verify_extension
run_test "Configure Roo Code" test_configure_roo_code
run_test "Main Function" test_main_function

# Exit with success
exit 0