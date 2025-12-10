#!/bin/bash
# SPARC IDE - Generate Admin Password Hash
# This script generates a secure password hash for the MCP server admin user

set -e
set -o nounset  # Exit if a variable is unset
set -o pipefail # Exit if any command in a pipeline fails

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

# Check prerequisites
check_prerequisites() {
    print_info "Checking prerequisites..."
    
    # Check if node is installed
    if ! command -v node &> /dev/null; then
        print_error "Node.js is not installed. Please install Node.js and try again."
        exit 1
    fi
    
    print_success "All prerequisites are met."
}

# Generate password hash
generate_password_hash() {
    print_info "Generating secure password hash..."
    
    # Prompt for password (without echoing to terminal)
    echo -n "Enter admin password: "
    read -s PASSWORD
    echo
    
    # Confirm password
    echo -n "Confirm admin password: "
    read -s PASSWORD_CONFIRM
    echo
    
    # Check if passwords match
    if [ "$PASSWORD" != "$PASSWORD_CONFIRM" ]; then
        print_error "Passwords do not match. Please try again."
        exit 1
    fi
    
    # Check password strength
    if [ ${#PASSWORD} -lt 12 ]; then
        print_error "Password is too short. Please use at least 12 characters."
        exit 1
    fi
    
    if ! echo "$PASSWORD" | grep -q "[A-Z]"; then
        print_error "Password must contain at least one uppercase letter."
        exit 1
    fi
    
    if ! echo "$PASSWORD" | grep -q "[a-z]"; then
        print_error "Password must contain at least one lowercase letter."
        exit 1
    fi
    
    if ! echo "$PASSWORD" | grep -q "[0-9]"; then
        print_error "Password must contain at least one number."
        exit 1
    fi
    
    if ! echo "$PASSWORD" | grep -q "[^A-Za-z0-9]"; then
        print_error "Password must contain at least one special character."
        exit 1
    fi
    
    # Generate hash using bcrypt with Node.js
    HASH=$(node -e "
        const bcrypt = require('bcrypt');
        const saltRounds = 12;
        bcrypt.hash('$PASSWORD', saltRounds, (err, hash) => {
            if (err) {
                console.error('Error generating hash:', err);
                process.exit(1);
            }
            console.log(hash);
        });
    ")
    
    # Check if hash was generated successfully
    if [ -z "$HASH" ]; then
        print_error "Failed to generate password hash."
        exit 1
    fi
    
    print_success "Password hash generated successfully."
    
    # Display instructions
    echo
    echo "Add the following to your .env file:"
    echo
    echo "MCP_ADMIN_USERNAME=admin"
    echo "MCP_ADMIN_PASSWORD_HASH=$HASH"
    echo
    print_info "Keep this information secure and do not share it."
}

# Main function
main() {
    print_info "SPARC IDE - Admin Password Hash Generator"
    
    check_prerequisites
    generate_password_hash
    
    print_success "Password hash generation completed."
}

# Run main function
main