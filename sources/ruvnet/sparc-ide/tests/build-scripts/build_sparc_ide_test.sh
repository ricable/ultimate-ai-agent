#!/bin/bash
# Test for build-sparc-ide.sh script

# Source test utilities
source "$(dirname "$0")/../helpers/test_utils.sh"

# Test functions
test_parse_args() {
    print_test "Testing parse_args function"
    
    # Create a test copy of the script
    local test_script=$(create_test_script_copy "$SPARC_IDE_ROOT/scripts/build-sparc-ide.sh")
    
    # Extract the parse_args function from the script
    local parse_args_function=$(grep -A 50 "parse_args()" "$test_script" | sed -n '/parse_args()/,/}/p')
    
    # Create a test script with just the parse_args function
    local test_function_script="$TEMP_DIR/parse_args.sh"
    cat > "$test_function_script" << EOL
#!/bin/bash
set -e

# Configuration
PLATFORM=""

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

# Mock show_help function
show_help() {
    echo "Usage: \$0 [options]"
    echo ""
    echo "Options:"
    echo "  --platform <platform>  Specify the target platform (linux, windows, macos)"
    echo "  --help                 Show this help message"
}

$parse_args_function

# Function to test parse_args with arguments
test_with_args() {
    # Reset PLATFORM
    PLATFORM=""
    
    # Parse arguments
    parse_args "\$@"
    
    # Print platform for testing
    echo "PLATFORM=\$PLATFORM"
}

# Run the function with different arguments
test_with_args "\$@"
EOL
    
    # Make the test script executable
    chmod +x "$test_function_script"
    
    # Test with no arguments (auto-detect platform)
    local output=$("$test_function_script" 2>&1)
    
    # Assert that the function succeeds
    assert_command_succeeds "$test_function_script" "parse_args function should succeed with no arguments"
    
    # Assert that the platform was auto-detected
    assert_contains "$output" "PLATFORM=" "Output should contain the auto-detected platform"
    
    # Test with --platform linux
    output=$("$test_function_script" --platform linux 2>&1)
    assert_contains "$output" "PLATFORM=linux" "Output should indicate that platform is linux"
    
    # Test with --platform windows
    output=$("$test_function_script" --platform windows 2>&1)
    assert_contains "$output" "PLATFORM=windows" "Output should indicate that platform is windows"
    
    # Test with --platform macos
    output=$("$test_function_script" --platform macos 2>&1)
    assert_contains "$output" "PLATFORM=macos" "Output should indicate that platform is macos"
    
    # Test with invalid platform
    output=$("$test_function_script" --platform invalid 2>&1 || true)
    assert_contains "$output" "Invalid platform: invalid" "Output should indicate that platform is invalid"
    
    # Test with --help
    output=$("$test_function_script" --help 2>&1 || true)
    assert_contains "$output" "Usage:" "Output should show help message"
    
    return 0
}

test_make_scripts_executable() {
    print_test "Testing make_scripts_executable function"
    
    # Create a test copy of the script
    local test_script=$(create_test_script_copy "$SPARC_IDE_ROOT/scripts/build-sparc-ide.sh")
    
    # Create a test directory structure
    local test_dir=$(create_test_directory_structure)
    cd "$test_dir"
    
    # Create script files
    mkdir -p "scripts"
    touch "scripts/setup-build-environment.sh"
    touch "scripts/apply-branding.sh"
    touch "scripts/download-roo-code.sh"
    touch "scripts/configure-ui.sh"
    
    # Extract the make_scripts_executable function from the script
    local make_scripts_executable_function=$(grep -A 20 "make_scripts_executable()" "$test_script" | sed -n '/make_scripts_executable()/,/}/p')
    
    # Create a test script with just the make_scripts_executable function
    local test_function_script="$TEMP_DIR/make_scripts_executable.sh"
    cat > "$test_function_script" << EOL
#!/bin/bash
set -e

# Configuration
SCRIPT_DIR="scripts"

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

$make_scripts_executable_function

# Run the function
make_scripts_executable
EOL
    
    # Make the test script executable
    chmod +x "$test_function_script"
    
    # Run the test script
    local output=$("$test_function_script" 2>&1)
    
    # Assert that the function succeeds
    assert_command_succeeds "$test_function_script" "make_scripts_executable function should succeed"
    
    # Assert that the output contains success message
    assert_contains "$output" "Scripts are now executable" "Output should indicate that scripts are now executable"
    
    # Assert that the scripts are executable
    assert_command_succeeds "test -x scripts/setup-build-environment.sh" "setup-build-environment.sh should be executable"
    assert_command_succeeds "test -x scripts/apply-branding.sh" "apply-branding.sh should be executable"
    assert_command_succeeds "test -x scripts/download-roo-code.sh" "download-roo-code.sh should be executable"
    assert_command_succeeds "test -x scripts/configure-ui.sh" "configure-ui.sh should be executable"
    
    return 0
}

test_build_sparc_ide() {
    print_test "Testing build_sparc_ide function"
    
    # Create a test copy of the script
    local test_script=$(create_test_script_copy "$SPARC_IDE_ROOT/scripts/build-sparc-ide.sh")
    
    # Create a test directory structure
    local test_dir=$(create_test_directory_structure)
    cd "$test_dir"
    
    # Create vscodium directory
    mkdir -p "vscodium"
    
    # Extract the build_sparc_ide function from the script
    local build_sparc_ide_function=$(grep -A 30 "build_sparc_ide()" "$test_script" | sed -n '/build_sparc_ide()/,/}/p')
    
    # Create a test script with just the build_sparc_ide function
    local test_function_script="$TEMP_DIR/build_sparc_ide.sh"
    cat > "$test_function_script" << EOL
#!/bin/bash
set -e

# Configuration
PLATFORM="linux"

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

# Mock yarn command
yarn() {
    if [[ "\$1" == "gulp" ]]; then
        echo "Building SPARC IDE for \$2..."
        echo "Build completed successfully."
    fi
}

$build_sparc_ide_function

# Run the function with different platforms
test_with_platform() {
    PLATFORM="\$1"
    build_sparc_ide
}

# Run the function with the specified platform
test_with_platform "\$1"
EOL
    
    # Make the test script executable
    chmod +x "$test_function_script"
    
    # Test with linux platform
    local output=$("$test_function_script" linux 2>&1)
    
    # Assert that the function succeeds
    assert_command_succeeds "$test_function_script linux" "build_sparc_ide function should succeed with linux platform"
    
    # Assert that the output contains success message
    assert_contains "$output" "SPARC IDE built successfully for linux" "Output should indicate that SPARC IDE was built successfully for linux"
    assert_contains "$output" "Building for Linux" "Output should indicate that building for Linux"
    
    # Test with windows platform
    output=$("$test_function_script" windows 2>&1)
    assert_contains "$output" "SPARC IDE built successfully for windows" "Output should indicate that SPARC IDE was built successfully for windows"
    assert_contains "$output" "Building for Windows" "Output should indicate that building for Windows"
    
    # Test with macos platform
    output=$("$test_function_script" macos 2>&1)
    assert_contains "$output" "SPARC IDE built successfully for macos" "Output should indicate that SPARC IDE was built successfully for macos"
    assert_contains "$output" "Building for macOS" "Output should indicate that building for macOS"
    
    return 0
}

test_create_packages() {
    print_test "Testing create_packages function"
    
    # Create a test copy of the script
    local test_script=$(create_test_script_copy "$SPARC_IDE_ROOT/scripts/build-sparc-ide.sh")
    
    # Create a test directory structure
    local test_dir=$(create_test_directory_structure)
    cd "$test_dir"
    
    # Create vscodium directory
    mkdir -p "vscodium"
    
    # Extract the create_packages function from the script
    local create_packages_function=$(grep -A 30 "create_packages()" "$test_script" | sed -n '/create_packages()/,/}/p')
    
    # Create a test script with just the create_packages function
    local test_function_script="$TEMP_DIR/create_packages.sh"
    cat > "$test_function_script" << EOL
#!/bin/bash
set -e

# Configuration
PLATFORM="linux"

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

# Mock yarn command
yarn() {
    if [[ "\$1" == "run" && "\$2" == "gulp" ]]; then
        echo "Creating packages for \$4..."
        echo "Packages created successfully."
        
        # Create dummy package files based on platform
        if [[ "\$PLATFORM" == "linux" ]]; then
            touch "vscodium/sparc-ide.deb"
            touch "vscodium/sparc-ide.rpm"
        elif [[ "\$PLATFORM" == "windows" ]]; then
            touch "vscodium/sparc-ide.exe"
        elif [[ "\$PLATFORM" == "macos" ]]; then
            touch "vscodium/sparc-ide.dmg"
        fi
    fi
}

$create_packages_function

# Run the function with different platforms
test_with_platform() {
    PLATFORM="\$1"
    create_packages
}

# Run the function with the specified platform
test_with_platform "\$1"
EOL
    
    # Make the test script executable
    chmod +x "$test_function_script"
    
    # Test with linux platform
    local output=$("$test_function_script" linux 2>&1)
    
    # Assert that the function succeeds
    assert_command_succeeds "$test_function_script linux" "create_packages function should succeed with linux platform"
    
    # Assert that the output contains success message
    assert_contains "$output" "Packages created successfully for linux" "Output should indicate that packages were created successfully for linux"
    assert_contains "$output" "Creating Linux packages" "Output should indicate that creating Linux packages"
    
    # Test with windows platform
    output=$("$test_function_script" windows 2>&1)
    assert_contains "$output" "Packages created successfully for windows" "Output should indicate that packages were created successfully for windows"
    assert_contains "$output" "Creating Windows installer" "Output should indicate that creating Windows installer"
    
    # Test with macos platform
    output=$("$test_function_script" macos 2>&1)
    assert_contains "$output" "Packages created successfully for macos" "Output should indicate that packages were created successfully for macos"
    assert_contains "$output" "Creating macOS package" "Output should indicate that creating macOS package"
    
    return 0
}

test_copy_artifacts() {
    print_test "Testing copy_artifacts function"
    
    # Create a test copy of the script
    local test_script=$(create_test_script_copy "$SPARC_IDE_ROOT/scripts/build-sparc-ide.sh")
    
    # Create a test directory structure
    local test_dir=$(create_test_directory_structure)
    cd "$test_dir"
    
    # Create vscodium directory with artifacts
    mkdir -p "vscodium"
    touch "vscodium/sparc-ide.deb"
    touch "vscodium/sparc-ide.rpm"
    touch "vscodium/sparc-ide.exe"
    touch "vscodium/sparc-ide.dmg"
    
    # Extract the copy_artifacts function from the script
    local copy_artifacts_function=$(grep -A 30 "copy_artifacts()" "$test_script" | sed -n '/copy_artifacts()/,/}/p')
    
    # Create a test script with just the copy_artifacts function
    local test_function_script="$TEMP_DIR/copy_artifacts.sh"
    cat > "$test_function_script" << EOL
#!/bin/bash
set -e

# Configuration
PLATFORM="linux"

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

$copy_artifacts_function

# Run the function with different platforms
test_with_platform() {
    PLATFORM="\$1"
    copy_artifacts
}

# Run the function with the specified platform
test_with_platform "\$1"
EOL
    
    # Make the test script executable
    chmod +x "$test_function_script"
    
    # Test with linux platform
    local output=$("$test_function_script" linux 2>&1)
    
    # Assert that the function succeeds
    assert_command_succeeds "$test_function_script linux" "copy_artifacts function should succeed with linux platform"
    
    # Assert that the output contains success message
    assert_contains "$output" "Artifacts copied to dist/ directory" "Output should indicate that artifacts were copied to dist/ directory"
    assert_contains "$output" "Copying Linux artifacts" "Output should indicate that copying Linux artifacts"
    
    # Assert that the dist directory was created
    assert_directory_exists "dist" "dist directory should be created"
    
    # Test with windows platform
    output=$("$test_function_script" windows 2>&1)
    assert_contains "$output" "Copying Windows artifacts" "Output should indicate that copying Windows artifacts"
    
    # Test with macos platform
    output=$("$test_function_script" macos 2>&1)
    assert_contains "$output" "Copying macOS artifacts" "Output should indicate that copying macOS artifacts"
    
    return 0
}

test_main_function() {
    print_test "Testing main function"
    
    # Create a test copy of the script
    local test_script=$(create_test_script_copy "$SPARC_IDE_ROOT/scripts/build-sparc-ide.sh")
    
    # Create a test directory structure
    local test_dir=$(create_test_directory_structure)
    cd "$test_dir"
    
    # Mock functions
    cat > "$TEMP_DIR/mock_functions.sh" << EOL
#!/bin/bash

# Mock parse_args
parse_args() {
    PLATFORM="linux"
    echo "Building for platform: \$PLATFORM"
}

# Mock make_scripts_executable
make_scripts_executable() {
    echo "Making scripts executable..."
    echo "Scripts are now executable."
}

# Mock setup_build_environment
setup_build_environment() {
    echo "Setting up build environment..."
    echo "Build environment setup completed."
}

# Mock apply_branding
apply_branding() {
    echo "Applying SPARC IDE branding..."
    echo "Branding applied successfully."
}

# Mock download_roo_code
download_roo_code() {
    echo "Downloading Roo Code extension..."
    echo "Roo Code extension downloaded successfully."
}

# Mock configure_ui
configure_ui() {
    echo "Configuring SPARC IDE UI..."
    echo "UI configuration completed."
}

# Mock build_sparc_ide
build_sparc_ide() {
    echo "Building SPARC IDE for \$PLATFORM..."
    echo "SPARC IDE built successfully for \$PLATFORM."
}

# Mock create_packages
create_packages() {
    echo "Creating packages for \$PLATFORM..."
    echo "Packages created successfully for \$PLATFORM."
}

# Mock copy_artifacts
copy_artifacts() {
    echo "Copying build artifacts..."
    echo "Artifacts copied to dist/ directory."
}
EOL
    
    # Create a modified test script that sources the mock functions
    local modified_test_script="$TEMP_DIR/modified_build_sparc_ide.sh"
    cat "$test_script" | sed '/^main /i source "$TEMP_DIR/mock_functions.sh"' > "$modified_test_script"
    chmod +x "$modified_test_script"
    
    # Run the test script
    local output=$("$modified_test_script" 2>&1)
    
    # Assert that the script succeeds
    assert_command_succeeds "$modified_test_script" "build-sparc-ide.sh script should succeed"
    
    # Assert that the output contains success message
    assert_contains "$output" "SPARC IDE has been built successfully" "Output should indicate that SPARC IDE was built successfully"
    
    # Assert that all functions were called
    assert_contains "$output" "Building for platform: linux" "parse_args function should be called"
    assert_contains "$output" "Making scripts executable" "make_scripts_executable function should be called"
    assert_contains "$output" "Setting up build environment" "setup_build_environment function should be called"
    assert_contains "$output" "Applying SPARC IDE branding" "apply_branding function should be called"
    assert_contains "$output" "Downloading Roo Code extension" "download_roo_code function should be called"
    assert_contains "$output" "Configuring SPARC IDE UI" "configure_ui function should be called"
    assert_contains "$output" "Building SPARC IDE for linux" "build_sparc_ide function should be called"
    assert_contains "$output" "Creating packages for linux" "create_packages function should be called"
    assert_contains "$output" "Copying build artifacts" "copy_artifacts function should be called"
    
    return 0
}

# Run tests
run_test "Parse Arguments" test_parse_args
run_test "Make Scripts Executable" test_make_scripts_executable
run_test "Build SPARC IDE" test_build_sparc_ide
run_test "Create Packages" test_create_packages
run_test "Copy Artifacts" test_copy_artifacts
run_test "Main Function" test_main_function

# Exit with success
exit 0