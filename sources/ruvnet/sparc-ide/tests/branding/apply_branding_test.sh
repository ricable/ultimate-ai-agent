#!/bin/bash
# Test for apply-branding.sh script

# Source test utilities
source "$(dirname "$0")/../helpers/test_utils.sh"

# Test functions
test_check_vscodium() {
    print_test "Testing check_vscodium function"
    
    # Create a test copy of the script
    local test_script=$(create_test_script_copy "$SPARC_IDE_ROOT/scripts/apply-branding.sh")
    
    # Create a test directory structure
    local test_dir=$(create_test_directory_structure)
    cd "$test_dir"
    
    # Extract the check_vscodium function from the script
    local check_vscodium_function=$(grep -A 20 "check_vscodium()" "$test_script" | sed -n '/check_vscodium()/,/}/p')
    
    # Create a test script with just the check_vscodium function
    local test_function_script="$TEMP_DIR/check_vscodium.sh"
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

$check_vscodium_function

# Run the function
check_vscodium
EOL
    
    # Make the test script executable
    chmod +x "$test_function_script"
    
    # Test with vscodium directory present
    mkdir -p "vscodium"
    local output=$("$test_function_script" 2>&1)
    
    # Assert that the function succeeds
    assert_command_succeeds "$test_function_script" "check_vscodium function should succeed with vscodium directory present"
    
    # Assert that the output contains success message
    assert_contains "$output" "VSCodium directory found" "Output should indicate that VSCodium directory was found"
    
    # Test with missing vscodium directory
    rm -rf "vscodium"
    output=$("$test_function_script" 2>&1 || true)
    assert_contains "$output" "VSCodium directory not found" "Output should indicate that VSCodium directory was not found"
    
    return 0
}

test_apply_product_json() {
    print_test "Testing apply_product_json function"
    
    # Create a test copy of the script
    local test_script=$(create_test_script_copy "$SPARC_IDE_ROOT/scripts/apply-branding.sh")
    
    # Create a test directory structure
    local test_dir=$(create_test_directory_structure)
    cd "$test_dir"
    
    # Create vscodium directory
    mkdir -p "vscodium"
    
    # Create product.json
    mkdir -p "src/config"
    echo '{"name": "SPARC IDE"}' > "src/config/product.json"
    
    # Extract the apply_product_json function from the script
    local apply_product_json_function=$(grep -A 20 "apply_product_json()" "$test_script" | sed -n '/apply_product_json()/,/}/p')
    
    # Create a test script with just the apply_product_json function
    local test_function_script="$TEMP_DIR/apply_product_json.sh"
    cat > "$test_function_script" << EOL
#!/bin/bash
set -e

# Configuration
VSCODIUM_DIR="vscodium"
PRODUCT_JSON="src/config/product.json"

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

$apply_product_json_function

# Run the function
apply_product_json
EOL
    
    # Make the test script executable
    chmod +x "$test_function_script"
    
    # Test with product.json present
    local output=$("$test_function_script" 2>&1)
    
    # Assert that the function succeeds
    assert_command_succeeds "$test_function_script" "apply_product_json function should succeed with product.json present"
    
    # Assert that the output contains success message
    assert_contains "$output" "Product JSON customizations applied" "Output should indicate that product.json customizations were applied"
    
    # Assert that product.json was copied to vscodium directory
    assert_file_exists "vscodium/product.json" "product.json should be copied to vscodium directory"
    
    # Test with missing product.json
    rm -f "src/config/product.json"
    output=$("$test_function_script" 2>&1 || true)
    assert_contains "$output" "Product JSON file not found" "Output should indicate that product.json file was not found"
    
    return 0
}

test_apply_icons() {
    print_test "Testing apply_icons function"
    
    # Create a test copy of the script
    local test_script=$(create_test_script_copy "$SPARC_IDE_ROOT/scripts/apply-branding.sh")
    
    # Create a test directory structure
    local test_dir=$(create_test_directory_structure)
    cd "$test_dir"
    
    # Create vscodium directory
    mkdir -p "vscodium"
    
    # Create icons directory with sample icons
    mkdir -p "branding/icons"
    echo "icon content" > "branding/icons/icon.svg"
    
    # Extract the apply_icons function from the script
    local apply_icons_function=$(grep -A 20 "apply_icons()" "$test_script" | sed -n '/apply_icons()/,/}/p')
    
    # Create a test script with just the apply_icons function
    local test_function_script="$TEMP_DIR/apply_icons.sh"
    cat > "$test_function_script" << EOL
#!/bin/bash
set -e

# Configuration
VSCODIUM_DIR="vscodium"
BRANDING_DIR="branding"

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

$apply_icons_function

# Run the function
apply_icons
EOL
    
    # Make the test script executable
    chmod +x "$test_function_script"
    
    # Test with icons directory present
    local output=$("$test_function_script" 2>&1)
    
    # Assert that the function succeeds
    assert_command_succeeds "$test_function_script" "apply_icons function should succeed with icons directory present"
    
    # Assert that the output contains success message
    assert_contains "$output" "Icon customizations applied" "Output should indicate that icon customizations were applied"
    
    # Assert that icons were copied to vscodium directory
    assert_directory_exists "vscodium/src/vs/workbench/browser/parts/editor/media" "Icons target directory should be created"
    assert_file_exists "vscodium/src/vs/workbench/browser/parts/editor/media/icon.svg" "Icon should be copied to vscodium directory"
    
    # Test with missing icons directory
    rm -rf "branding/icons"
    output=$("$test_function_script" 2>&1 || true)
    assert_contains "$output" "Icons directory not found" "Output should indicate that icons directory was not found"
    
    return 0
}

test_apply_splash() {
    print_test "Testing apply_splash function"
    
    # Create a test copy of the script
    local test_script=$(create_test_script_copy "$SPARC_IDE_ROOT/scripts/apply-branding.sh")
    
    # Create a test directory structure
    local test_dir=$(create_test_directory_structure)
    cd "$test_dir"
    
    # Create vscodium directory
    mkdir -p "vscodium"
    
    # Create splash directory with sample splash screen
    mkdir -p "branding/splash"
    echo "splash content" > "branding/splash/splash.svg"
    
    # Extract the apply_splash function from the script
    local apply_splash_function=$(grep -A 20 "apply_splash()" "$test_script" | sed -n '/apply_splash()/,/}/p')
    
    # Create a test script with just the apply_splash function
    local test_function_script="$TEMP_DIR/apply_splash.sh"
    cat > "$test_function_script" << EOL
#!/bin/bash
set -e

# Configuration
VSCODIUM_DIR="vscodium"
BRANDING_DIR="branding"

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

$apply_splash_function

# Run the function
apply_splash
EOL
    
    # Make the test script executable
    chmod +x "$test_function_script"
    
    # Test with splash directory present
    local output=$("$test_function_script" 2>&1)
    
    # Assert that the function succeeds
    assert_command_succeeds "$test_function_script" "apply_splash function should succeed with splash directory present"
    
    # Assert that the output contains success message
    assert_contains "$output" "Splash screen customizations applied" "Output should indicate that splash screen customizations were applied"
    
    # Assert that splash screens were copied to vscodium directory
    assert_directory_exists "vscodium/src/vs/workbench/browser/parts/splash" "Splash target directory should be created"
    assert_file_exists "vscodium/src/vs/workbench/browser/parts/splash/splash.svg" "Splash screen should be copied to vscodium directory"
    
    # Test with missing splash directory
    rm -rf "branding/splash"
    output=$("$test_function_script" 2>&1 || true)
    assert_contains "$output" "Splash screen directory not found" "Output should indicate that splash screen directory was not found"
    
    return 0
}

test_apply_platform_branding() {
    print_test "Testing apply_platform_branding function"
    
    # Create a test copy of the script
    local test_script=$(create_test_script_copy "$SPARC_IDE_ROOT/scripts/apply-branding.sh")
    
    # Create a test directory structure
    local test_dir=$(create_test_directory_structure)
    cd "$test_dir"
    
    # Create vscodium directory
    mkdir -p "vscodium"
    
    # Create platform-specific branding directories with sample files
    mkdir -p "branding/linux"
    echo "linux branding" > "branding/linux/icon.png"
    
    mkdir -p "branding/windows"
    echo "windows branding" > "branding/windows/icon.ico"
    
    mkdir -p "branding/macos"
    echo "macos branding" > "branding/macos/icon.icns"
    
    # Extract the apply_platform_branding function from the script
    local apply_platform_branding_function=$(grep -A 30 "apply_platform_branding()" "$test_script" | sed -n '/apply_platform_branding()/,/}/p')
    
    # Create a test script with just the apply_platform_branding function
    local test_function_script="$TEMP_DIR/apply_platform_branding.sh"
    cat > "$test_function_script" << EOL
#!/bin/bash
set -e

# Configuration
VSCODIUM_DIR="vscodium"
BRANDING_DIR="branding"

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

$apply_platform_branding_function

# Run the function
apply_platform_branding
EOL
    
    # Make the test script executable
    chmod +x "$test_function_script"
    
    # Test with platform-specific branding directories present
    local output=$("$test_function_script" 2>&1)
    
    # Assert that the function succeeds
    assert_command_succeeds "$test_function_script" "apply_platform_branding function should succeed with platform-specific branding directories present"
    
    # Assert that the output contains success message
    assert_contains "$output" "Platform-specific branding applied" "Output should indicate that platform-specific branding was applied"
    
    # Assert that platform-specific branding was copied to vscodium directory
    assert_directory_exists "vscodium/build/linux" "Linux build directory should be created"
    assert_file_exists "vscodium/build/linux/icon.png" "Linux branding should be copied to vscodium directory"
    
    assert_directory_exists "vscodium/build/windows" "Windows build directory should be created"
    assert_file_exists "vscodium/build/windows/icon.ico" "Windows branding should be copied to vscodium directory"
    
    assert_directory_exists "vscodium/build/darwin" "macOS build directory should be created"
    assert_file_exists "vscodium/build/darwin/icon.icns" "macOS branding should be copied to vscodium directory"
    
    return 0
}

test_main_function() {
    print_test "Testing main function"
    
    # Create a test copy of the script
    local test_script=$(create_test_script_copy "$SPARC_IDE_ROOT/scripts/apply-branding.sh")
    
    # Create a test directory structure
    local test_dir=$(create_test_directory_structure)
    cd "$test_dir"
    
    # Create vscodium directory
    mkdir -p "vscodium"
    
    # Mock functions
    cat > "$TEMP_DIR/mock_functions.sh" << EOL
#!/bin/bash

# Mock check_vscodium
check_vscodium() {
    echo "Checking VSCodium directory..."
    echo "VSCodium directory found."
}

# Mock apply_product_json
apply_product_json() {
    echo "Applying product.json customizations..."
    echo "Product JSON customizations applied."
}

# Mock apply_icons
apply_icons() {
    echo "Applying icon customizations..."
    echo "Icon customizations applied."
}

# Mock apply_splash
apply_splash() {
    echo "Applying splash screen customizations..."
    echo "Splash screen customizations applied."
}

# Mock apply_platform_branding
apply_platform_branding() {
    echo "Applying platform-specific branding..."
    echo "Platform-specific branding applied."
}
EOL
    
    # Create a modified test script that sources the mock functions
    local modified_test_script="$TEMP_DIR/modified_apply_branding.sh"
    cat "$test_script" | sed '/^main$/i source "$TEMP_DIR/mock_functions.sh"' > "$modified_test_script"
    chmod +x "$modified_test_script"
    
    # Run the test script
    local output=$("$modified_test_script" 2>&1)
    
    # Assert that the script succeeds
    assert_command_succeeds "$modified_test_script" "apply-branding.sh script should succeed"
    
    # Assert that the output contains success message
    assert_contains "$output" "SPARC IDE branding applied successfully" "Output should indicate that SPARC IDE branding was applied successfully"
    
    # Assert that all functions were called
    assert_contains "$output" "Checking VSCodium directory" "check_vscodium function should be called"
    assert_contains "$output" "Applying product.json customizations" "apply_product_json function should be called"
    assert_contains "$output" "Applying icon customizations" "apply_icons function should be called"
    assert_contains "$output" "Applying splash screen customizations" "apply_splash function should be called"
    assert_contains "$output" "Applying platform-specific branding" "apply_platform_branding function should be called"
    
    return 0
}

test_branding_assets() {
    print_test "Testing branding assets"
    
    # Check if branding directories exist
    assert_directory_exists "$SPARC_IDE_ROOT/branding/icons" "Icons directory should exist"
    assert_directory_exists "$SPARC_IDE_ROOT/branding/splash" "Splash directory should exist"
    assert_directory_exists "$SPARC_IDE_ROOT/branding/linux" "Linux branding directory should exist"
    assert_directory_exists "$SPARC_IDE_ROOT/branding/windows" "Windows branding directory should exist"
    assert_directory_exists "$SPARC_IDE_ROOT/branding/macos" "macOS branding directory should exist"
    
    # Check if README files exist in branding directories
    assert_file_exists "$SPARC_IDE_ROOT/branding/icons/README.md" "Icons README should exist"
    assert_file_exists "$SPARC_IDE_ROOT/branding/linux/README.md" "Linux branding README should exist"
    assert_file_exists "$SPARC_IDE_ROOT/branding/windows/README.md" "Windows branding README should exist"
    assert_file_exists "$SPARC_IDE_ROOT/branding/macos/README.md" "macOS branding README should exist"
    assert_file_exists "$SPARC_IDE_ROOT/branding/splash/README.md" "Splash README should exist"
    
    return 0
}

# Run tests
run_test "Check VSCodium" test_check_vscodium
run_test "Apply Product JSON" test_apply_product_json
run_test "Apply Icons" test_apply_icons
run_test "Apply Splash" test_apply_splash
run_test "Apply Platform Branding" test_apply_platform_branding
run_test "Main Function" test_main_function
run_test "Branding Assets" test_branding_assets

# Exit with success
exit 0