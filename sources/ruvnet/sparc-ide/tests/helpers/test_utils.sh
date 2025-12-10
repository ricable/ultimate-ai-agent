#!/bin/bash
# SPARC IDE Test Utilities
# This script provides common functions for all tests

# Configuration
SPARC_IDE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TEMP_DIR="$SPARC_IDE_ROOT/tests/temp"

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

print_test() {
    echo -e "\e[1;35m[TEST]\e[0m $1"
}

# Assert functions
assert_equals() {
    local expected="$1"
    local actual="$2"
    local message="$3"
    
    if [ "$expected" = "$actual" ]; then
        print_success "PASS: $message"
        return 0
    else
        print_error "FAIL: $message"
        echo "  Expected: $expected"
        echo "  Actual:   $actual"
        return 1
    fi
}

assert_not_equals() {
    local expected="$1"
    local actual="$2"
    local message="$3"
    
    if [ "$expected" != "$actual" ]; then
        print_success "PASS: $message"
        return 0
    else
        print_error "FAIL: $message"
        echo "  Expected not to equal: $expected"
        echo "  Actual:               $actual"
        return 1
    fi
}

assert_contains() {
    local haystack="$1"
    local needle="$2"
    local message="$3"
    
    if [[ "$haystack" == *"$needle"* ]]; then
        print_success "PASS: $message"
        return 0
    else
        print_error "FAIL: $message"
        echo "  Expected to contain: $needle"
        echo "  Actual:             $haystack"
        return 1
    fi
}

assert_not_contains() {
    local haystack="$1"
    local needle="$2"
    local message="$3"
    
    if [[ "$haystack" != *"$needle"* ]]; then
        print_success "PASS: $message"
        return 0
    else
        print_error "FAIL: $message"
        echo "  Expected not to contain: $needle"
        echo "  Actual:                 $haystack"
        return 1
    fi
}

assert_file_exists() {
    local file="$1"
    local message="$2"
    
    if [ -f "$file" ]; then
        print_success "PASS: $message"
        return 0
    else
        print_error "FAIL: $message"
        echo "  File does not exist: $file"
        return 1
    fi
}

assert_directory_exists() {
    local directory="$1"
    local message="$2"
    
    if [ -d "$directory" ]; then
        print_success "PASS: $message"
        return 0
    else
        print_error "FAIL: $message"
        echo "  Directory does not exist: $directory"
        return 1
    fi
}

assert_command_succeeds() {
    local command="$1"
    local message="$2"
    
    if eval "$command" > /dev/null 2>&1; then
        print_success "PASS: $message"
        return 0
    else
        print_error "FAIL: $message"
        echo "  Command failed: $command"
        return 1
    fi
}

assert_command_fails() {
    local command="$1"
    local message="$2"
    
    if ! eval "$command" > /dev/null 2>&1; then
        print_success "PASS: $message"
        return 0
    else
        print_error "FAIL: $message"
        echo "  Command succeeded but should have failed: $command"
        return 1
    fi
}

# Setup and teardown functions
setup_test_environment() {
    print_info "Setting up test environment..."
    
    # Create temp directory
    mkdir -p "$TEMP_DIR"
    
    print_info "Test environment set up."
}

teardown_test_environment() {
    print_info "Tearing down test environment..."
    
    # Remove temp directory
    if [ -d "$TEMP_DIR" ]; then
        rm -rf "$TEMP_DIR"
    fi
    
    print_info "Test environment torn down."
}

# Mock functions
mock_command() {
    local command="$1"
    local output="$2"
    local exit_code="${3:-0}"
    
    # Create mock directory if it doesn't exist
    mkdir -p "$TEMP_DIR/mock/bin"
    
    # Create mock command
    cat > "$TEMP_DIR/mock/bin/$command" << EOL
#!/bin/bash
echo "$output"
exit $exit_code
EOL
    
    # Make mock command executable
    chmod +x "$TEMP_DIR/mock/bin/$command"
    
    # Add mock directory to PATH
    export PATH="$TEMP_DIR/mock/bin:$PATH"
    
    print_info "Mocked command: $command"
}

# Run a test with setup and teardown
run_test() {
    local test_name="$1"
    local test_function="$2"
    
    print_test "Running test: $test_name"
    
    # Setup
    setup_test_environment
    
    # Run test
    if $test_function; then
        test_result=0
    else
        test_result=1
    fi
    
    # Teardown
    teardown_test_environment
    
    return $test_result
}

# Create a temporary copy of a script for testing
create_test_script_copy() {
    local original_script="$1"
    local test_script="$TEMP_DIR/$(basename "$original_script")"
    
    # Create temp directory
    mkdir -p "$TEMP_DIR"
    
    # Copy script
    cp "$original_script" "$test_script"
    
    # Make script executable
    chmod +x "$test_script"
    
    echo "$test_script"
}

# Create a temporary directory structure for testing
create_test_directory_structure() {
    local base_dir="$TEMP_DIR/test_structure"
    
    # Create base directory
    mkdir -p "$base_dir"
    
    # Create common directories
    mkdir -p "$base_dir/scripts"
    mkdir -p "$base_dir/src/config"
    mkdir -p "$base_dir/branding/icons"
    mkdir -p "$base_dir/branding/splash"
    mkdir -p "$base_dir/branding/linux"
    mkdir -p "$base_dir/branding/windows"
    mkdir -p "$base_dir/branding/macos"
    mkdir -p "$base_dir/extensions"
    
    echo "$base_dir"
}