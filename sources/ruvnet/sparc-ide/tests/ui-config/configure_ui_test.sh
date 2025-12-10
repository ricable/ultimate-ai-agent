#!/bin/bash
# Test for configure-ui.sh script

# Source test utilities
source "$(dirname "$0")/../helpers/test_utils.sh"

# Test functions
test_check_vscodium() {
    print_test "Testing check_vscodium function"
    
    # Create a test copy of the script
    local test_script=$(create_test_script_copy "$SPARC_IDE_ROOT/scripts/configure-ui.sh")
    
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

test_configure_layout() {
    print_test "Testing configure_layout function"
    
    # Create a test copy of the script
    local test_script=$(create_test_script_copy "$SPARC_IDE_ROOT/scripts/configure-ui.sh")
    
    # Create a test directory structure
    local test_dir=$(create_test_directory_structure)
    cd "$test_dir"
    
    # Create vscodium directory with layout.js
    mkdir -p "vscodium/src/vs/workbench/browser/parts"
    echo "// Original layout.js content" > "vscodium/src/vs/workbench/browser/parts/layout.js"
    
    # Extract the configure_layout function from the script
    local configure_layout_function=$(grep -A 50 "configure_layout()" "$test_script" | sed -n '/configure_layout()/,/}/p')
    
    # Create a test script with just the configure_layout function
    local test_function_script="$TEMP_DIR/configure_layout.sh"
    cat > "$test_function_script" << EOL
#!/bin/bash
set -e

# Configuration
VSCODIUM_DIR="vscodium"
LAYOUT_CONFIG_PATH="src/vs/workbench/browser/parts/layout.js"

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

# Mock patch command
patch() {
    if [[ "\$1" == "-N" ]]; then
        # Simulate patching by writing to the file
        echo "// Patched layout.js content" > "\$2"
        return 0
    fi
}

$configure_layout_function

# Run the function
configure_layout
EOL
    
    # Make the test script executable
    chmod +x "$test_function_script"
    
    # Run the test script
    local output=$("$test_function_script" 2>&1)
    
    # Assert that the function succeeds
    assert_command_succeeds "$test_function_script" "configure_layout function should succeed"
    
    # Assert that the output contains success message
    assert_contains "$output" "AI-centric layout configured" "Output should indicate that AI-centric layout was configured"
    
    # Assert that the layout patch file was created
    assert_file_exists "build/patches/ai-centric-layout.patch" "Layout patch file should be created"
    
    # Test with missing layout.js
    rm -f "vscodium/src/vs/workbench/browser/parts/layout.js"
    output=$("$test_function_script" 2>&1)
    assert_contains "$output" "Layout configuration will be applied during build" "Output should indicate that layout configuration will be applied during build"
    
    return 0
}

test_configure_themes() {
    print_test "Testing configure_themes function"
    
    # Create a test copy of the script
    local test_script=$(create_test_script_copy "$SPARC_IDE_ROOT/scripts/configure-ui.sh")
    
    # Create a test directory structure
    local test_dir=$(create_test_directory_structure)
    cd "$test_dir"
    
    # Extract the configure_themes function from the script
    local configure_themes_function=$(grep -A 300 "configure_themes()" "$test_script" | sed -n '/configure_themes()/,/}/p')
    
    # Create a test script with just the configure_themes function
    local test_function_script="$TEMP_DIR/configure_themes.sh"
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

$configure_themes_function

# Run the function
configure_themes
EOL
    
    # Make the test script executable
    chmod +x "$test_function_script"
    
    # Run the test script
    local output=$("$test_function_script" 2>&1)
    
    # Assert that the function succeeds
    assert_command_succeeds "$test_function_script" "configure_themes function should succeed"
    
    # Assert that the output contains success message
    assert_contains "$output" "Custom themes configured" "Output should indicate that custom themes were configured"
    
    # Assert that the themes directory was created
    assert_directory_exists "src/themes" "Themes directory should be created"
    
    # Assert that the theme files were created
    assert_file_exists "src/themes/dracula-pro.json" "Dracula Pro theme file should be created"
    assert_file_exists "src/themes/material-theme.json" "Material Theme file should be created"
    
    return 0
}

test_configure_sparc_workflow() {
    print_test "Testing configure_sparc_workflow function"
    
    # Create a test copy of the script
    local test_script=$(create_test_script_copy "$SPARC_IDE_ROOT/scripts/configure-ui.sh")
    
    # Create a test directory structure
    local test_dir=$(create_test_directory_structure)
    cd "$test_dir"
    
    # Extract the configure_sparc_workflow function from the script
    local configure_sparc_workflow_function=$(grep -A 100 "configure_sparc_workflow()" "$test_script" | sed -n '/configure_sparc_workflow()/,/}/p')
    
    # Create a test script with just the configure_sparc_workflow function
    local test_function_script="$TEMP_DIR/configure_sparc_workflow.sh"
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

$configure_sparc_workflow_function

# Run the function
configure_sparc_workflow
EOL
    
    # Make the test script executable
    chmod +x "$test_function_script"
    
    # Run the test script
    local output=$("$test_function_script" 2>&1)
    
    # Assert that the function succeeds
    assert_command_succeeds "$test_function_script" "configure_sparc_workflow function should succeed"
    
    # Assert that the output contains success message
    assert_contains "$output" "SPARC workflow UI configured" "Output should indicate that SPARC workflow UI was configured"
    
    # Assert that the sparc-workflow directory was created
    assert_directory_exists "src/sparc-workflow" "SPARC workflow directory should be created"
    
    # Assert that the config file was created
    assert_file_exists "src/sparc-workflow/config.json" "SPARC workflow config file should be created"
    
    return 0
}

test_main_function() {
    print_test "Testing main function"
    
    # Create a test copy of the script
    local test_script=$(create_test_script_copy "$SPARC_IDE_ROOT/scripts/configure-ui.sh")
    
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

# Mock configure_layout
configure_layout() {
    echo "Configuring AI-centric layout..."
    echo "AI-centric layout configured."
}

# Mock configure_themes
configure_themes() {
    echo "Configuring custom themes..."
    echo "Custom themes configured."
}

# Mock configure_sparc_workflow
configure_sparc_workflow() {
    echo "Configuring SPARC workflow UI..."
    echo "SPARC workflow UI configured."
}
EOL
    
    # Create a modified test script that sources the mock functions
    local modified_test_script="$TEMP_DIR/modified_configure_ui.sh"
    cat "$test_script" | sed '/^main$/i source "$TEMP_DIR/mock_functions.sh"' > "$modified_test_script"
    chmod +x "$modified_test_script"
    
    # Run the test script
    local output=$("$modified_test_script" 2>&1)
    
    # Assert that the script succeeds
    assert_command_succeeds "$modified_test_script" "configure-ui.sh script should succeed"
    
    # Assert that the output contains success message
    assert_contains "$output" "SPARC IDE UI configured successfully" "Output should indicate that SPARC IDE UI was configured successfully"
    
    # Assert that all functions were called
    assert_contains "$output" "Checking VSCodium directory" "check_vscodium function should be called"
    assert_contains "$output" "Configuring AI-centric layout" "configure_layout function should be called"
    assert_contains "$output" "Configuring custom themes" "configure_themes function should be called"
    assert_contains "$output" "Configuring SPARC workflow UI" "configure_sparc_workflow function should be called"
    
    return 0
}

test_ui_configuration_files() {
    print_test "Testing UI configuration files"
    
    # Create a test directory structure
    local test_dir=$(create_test_directory_structure)
    cd "$test_dir"
    
    # Copy configuration files from the source
    mkdir -p "src/config"
    cp "$SPARC_IDE_ROOT/src/config/product.json" "src/config/"
    cp "$SPARC_IDE_ROOT/src/config/settings.json" "src/config/"
    cp "$SPARC_IDE_ROOT/src/config/keybindings.json" "src/config/"
    
    # Test product.json
    assert_file_exists "src/config/product.json" "product.json should exist"
    local product_content=$(cat "src/config/product.json")
    assert_contains "$product_content" "\"nameShort\": \"SPARC IDE\"" "product.json should contain SPARC IDE name"
    assert_contains "$product_content" "\"aiConfig\":" "product.json should contain AI configuration"
    assert_contains "$product_content" "\"sparcConfig\":" "product.json should contain SPARC configuration"
    
    # Test settings.json
    assert_file_exists "src/config/settings.json" "settings.json should exist"
    local settings_content=$(cat "src/config/settings.json")
    assert_contains "$settings_content" "\"workbench.colorTheme\": \"Dracula Pro\"" "settings.json should contain theme configuration"
    assert_contains "$settings_content" "\"roo-code.defaultModel\":" "settings.json should contain Roo Code configuration"
    assert_contains "$settings_content" "\"sparc-workflow.enabled\":" "settings.json should contain SPARC workflow configuration"
    
    # Test keybindings.json
    assert_file_exists "src/config/keybindings.json" "keybindings.json should exist"
    local keybindings_content=$(cat "src/config/keybindings.json")
    assert_contains "$keybindings_content" "\"command\": \"roo-code.chat\"" "keybindings.json should contain Roo Code keybindings"
    assert_contains "$keybindings_content" "\"command\": \"sparc-workflow.switchPhase" "keybindings.json should contain SPARC workflow keybindings"
    
    return 0
}

# Run tests
run_test "Check VSCodium" test_check_vscodium
run_test "Configure Layout" test_configure_layout
run_test "Configure Themes" test_configure_themes
run_test "Configure SPARC Workflow" test_configure_sparc_workflow
run_test "Main Function" test_main_function
run_test "UI Configuration Files" test_ui_configuration_files

# Exit with success
exit 0