#!/bin/bash
# Test for setup-build-environment.sh script

# Source test utilities
source "$(dirname "$0")/../helpers/test_utils.sh"

# Test functions
test_prerequisites_check() {
    print_test "Testing prerequisites check function"
    
    # Create a test copy of the script
    local test_script=$(create_test_script_copy "$SPARC_IDE_ROOT/scripts/setup-build-environment.sh")
    
    # Mock commands
    mock_command "git" "git version 2.30.0" 0
    mock_command "node" "v16.14.0" 0
    mock_command "yarn" "1.22.19" 0
    
    # Extract the check_prerequisites function from the script
    local check_prerequisites_function=$(grep -A 50 "check_prerequisites()" "$test_script" | sed -n '/check_prerequisites()/,/}/p')
    
    # Create a test script with just the check_prerequisites function
    local test_function_script="$TEMP_DIR/check_prerequisites.sh"
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

# Node version
NODE_VERSION="16"

$check_prerequisites_function

# Run the function
check_prerequisites
EOL
    
    # Make the test script executable
    chmod +x "$test_function_script"
    
    # Run the test script
    local output=$("$test_function_script" 2>&1)
    
    # Assert that the function succeeds
    assert_command_succeeds "$test_function_script" "check_prerequisites function should succeed with all prerequisites met"
    
    # Assert that the output contains success message
    assert_contains "$output" "All prerequisites are met" "Output should indicate that all prerequisites are met"
    
    # Test with missing git
    mock_command "git" "" 1
    output=$("$test_function_script" 2>&1 || true)
    assert_contains "$output" "Git is not installed" "Output should indicate that git is not installed"
    
    # Restore git mock
    mock_command "git" "git version 2.30.0" 0
    
    # Test with older Node.js version
    mock_command "node" "v14.17.0" 0
    output=$("$test_function_script" 2>&1 || true)
    assert_contains "$output" "Node.js version" "Output should indicate that Node.js version is too old"
    
    return 0
}

test_clone_vscodium() {
    print_test "Testing clone_vscodium function"
    
    # Create a test copy of the script
    local test_script=$(create_test_script_copy "$SPARC_IDE_ROOT/scripts/setup-build-environment.sh")
    
    # Create a test directory structure
    local test_dir=$(create_test_directory_structure)
    cd "$test_dir"
    
    # Extract the clone_vscodium function from the script
    local clone_vscodium_function=$(grep -A 20 "clone_vscodium()" "$test_script" | sed -n '/clone_vscodium()/,/}/p')
    
    # Create a test script with just the clone_vscodium function
    local test_function_script="$TEMP_DIR/clone_vscodium.sh"
    cat > "$test_function_script" << EOL
#!/bin/bash
set -e

# Configuration
VSCODIUM_REPO="https://github.com/VSCodium/vscodium.git"
VSCODIUM_DIR="vscodium"

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

# Mock git clone
git() {
    if [[ "\$1" == "clone" ]]; then
        mkdir -p "\$3"
        echo "Cloning into '\$3'..."
    elif [[ "\$1" == "pull" ]]; then
        echo "Already up to date."
    fi
}

$clone_vscodium_function

# Run the function
clone_vscodium
EOL
    
    # Make the test script executable
    chmod +x "$test_function_script"
    
    # Run the test script
    local output=$("$test_function_script" 2>&1)
    
    # Assert that the function succeeds
    assert_command_succeeds "$test_function_script" "clone_vscodium function should succeed"
    
    # Assert that the output contains success message
    assert_contains "$output" "VSCodium repository cloned/updated successfully" "Output should indicate that VSCodium repository was cloned successfully"
    
    # Assert that the vscodium directory was created
    assert_directory_exists "$test_dir/vscodium" "VSCodium directory should be created"
    
    # Test with existing vscodium directory
    output=$("$test_function_script" 2>&1)
    assert_contains "$output" "VSCodium directory already exists" "Output should indicate that VSCodium directory already exists"
    
    return 0
}

test_setup_customizations() {
    print_test "Testing setup_customizations function"
    
    # Create a test copy of the script
    local test_script=$(create_test_script_copy "$SPARC_IDE_ROOT/scripts/setup-build-environment.sh")
    
    # Create a test directory structure
    local test_dir=$(create_test_directory_structure)
    cd "$test_dir"
    
    # Create vscodium directory
    mkdir -p "vscodium/src/vs/workbench/browser/parts/editor/media"
    mkdir -p "vscodium/src/vs/workbench/browser/parts/splash"
    
    # Create product.json
    mkdir -p "src/config"
    echo '{"name": "SPARC IDE"}' > "src/config/product.json"
    
    # Create branding assets
    mkdir -p "branding/icons"
    mkdir -p "branding/splash"
    echo "icon content" > "branding/icons/icon.svg"
    echo "splash content" > "branding/splash/splash.svg"
    
    # Extract the setup_customizations function from the script
    local setup_customizations_function=$(grep -A 40 "setup_customizations()" "$test_script" | sed -n '/setup_customizations()/,/}/p')
    
    # Create a test script with just the setup_customizations function
    local test_function_script="$TEMP_DIR/setup_customizations.sh"
    cat > "$test_function_script" << EOL
#!/bin/bash
set -e

# Configuration
VSCODIUM_DIR="vscodium"

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

$setup_customizations_function

# Run the function
setup_customizations
EOL
    
    # Make the test script executable
    chmod +x "$test_function_script"
    
    # Run the test script
    local output=$("$test_function_script" 2>&1)
    
    # Assert that the function succeeds
    assert_command_succeeds "$test_function_script" "setup_customizations function should succeed"
    
    # Assert that the output contains success message
    assert_contains "$output" "SPARC IDE customizations set up successfully" "Output should indicate that customizations were set up successfully"
    
    # Assert that product.json was copied
    assert_file_exists "$test_dir/vscodium/product.json" "product.json should be copied to vscodium directory"
    
    # Assert that branding assets were copied
    assert_file_exists "$test_dir/vscodium/src/vs/workbench/browser/parts/editor/media/icon.svg" "Icon should be copied to vscodium directory"
    assert_file_exists "$test_dir/vscodium/src/vs/workbench/browser/parts/splash/splash.svg" "Splash screen should be copied to vscodium directory"
    
    return 0
}

test_main_function() {
    print_test "Testing main function"
    
    # Create a test copy of the script
    local test_script=$(create_test_script_copy "$SPARC_IDE_ROOT/scripts/setup-build-environment.sh")
    
    # Create a test directory structure
    local test_dir=$(create_test_directory_structure)
    cd "$test_dir"
    
    # Mock functions
    cat > "$TEMP_DIR/mock_functions.sh" << EOL
#!/bin/bash

# Mock check_prerequisites
check_prerequisites() {
    echo "Checking prerequisites..."
    echo "All prerequisites are met."
}

# Mock clone_vscodium
clone_vscodium() {
    echo "Cloning VSCodium repository..."
    mkdir -p "vscodium"
    echo "VSCodium repository cloned/updated successfully."
}

# Mock install_dependencies
install_dependencies() {
    echo "Installing dependencies..."
    echo "Dependencies installed successfully."
}

# Mock setup_customizations
setup_customizations() {
    echo "Setting up SPARC IDE customizations..."
    echo "SPARC IDE customizations set up successfully."
}
EOL
    
    # Create a modified test script that sources the mock functions
    local modified_test_script="$TEMP_DIR/modified_setup_build_environment.sh"
    cat "$test_script" | sed '/^main$/i source "$TEMP_DIR/mock_functions.sh"' > "$modified_test_script"
    chmod +x "$modified_test_script"
    
    # Run the test script
    local output=$("$modified_test_script" 2>&1)
    
    # Assert that the script succeeds
    assert_command_succeeds "$modified_test_script" "setup-build-environment.sh script should succeed"
    
    # Assert that the output contains success message
    assert_contains "$output" "SPARC IDE build environment set up successfully" "Output should indicate that build environment was set up successfully"
    
    # Assert that all functions were called
    assert_contains "$output" "Checking prerequisites" "check_prerequisites function should be called"
    assert_contains "$output" "Cloning VSCodium repository" "clone_vscodium function should be called"
    assert_contains "$output" "Installing dependencies" "install_dependencies function should be called"
    assert_contains "$output" "Setting up SPARC IDE customizations" "setup_customizations function should be called"
    
    return 0
}

# Run tests
run_test "Prerequisites Check" test_prerequisites_check
run_test "Clone VSCodium" test_clone_vscodium
run_test "Setup Customizations" test_setup_customizations
run_test "Main Function" test_main_function

# Exit with success
exit 0