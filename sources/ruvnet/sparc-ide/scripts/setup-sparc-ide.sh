#!/bin/bash
# SPARC IDE - Main Setup Script
# This script runs all the setup scripts in the correct order to set up the SPARC IDE environment

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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

# Make scripts executable
make_scripts_executable() {
    print_info "Making scripts executable..."
    
    chmod +x "$SCRIPT_DIR/setup-build-environment.sh"
    chmod +x "$SCRIPT_DIR/apply-branding.sh"
    chmod +x "$SCRIPT_DIR/download-roo-code.sh"
    chmod +x "$SCRIPT_DIR/configure-ui.sh"
    chmod +x "$SCRIPT_DIR/setup-mcp-server.sh"
    chmod +x "$SCRIPT_DIR/build-sparc-ide.sh"
    
    print_success "Scripts are now executable."
}

# Setup build environment
setup_build_environment() {
    print_header "Setting up build environment"
    
    "$SCRIPT_DIR/setup-build-environment.sh"
    
    print_success "Build environment setup completed."
}

# Apply branding
apply_branding() {
    print_header "Applying SPARC IDE branding"
    
    "$SCRIPT_DIR/apply-branding.sh"
    
    print_success "Branding applied successfully."
}

# Download Roo Code
download_roo_code() {
    print_header "Downloading Roo Code extension"
    
    "$SCRIPT_DIR/download-roo-code.sh"
    
    print_success "Roo Code extension downloaded successfully."
}

# Configure UI
configure_ui() {
    print_header "Configuring SPARC IDE UI"
    
    "$SCRIPT_DIR/configure-ui.sh"
    
    print_success "UI configuration completed."
}

# Setup MCP server
setup_mcp_server() {
    print_header "Setting up MCP server"
    
    "$SCRIPT_DIR/setup-mcp-server.sh"
    
    print_success "MCP server setup completed."
}

# Create SPARC workflow directories
create_sparc_workflow_directories() {
    print_header "Creating SPARC workflow directories"
    
    # Create .sparc directory in the user's home directory
    mkdir -p "$HOME/.sparc/templates"
    mkdir -p "$HOME/.sparc/artifacts"
    
    # Copy templates to the .sparc directory
    if [ -d "src/sparc-workflow/templates" ]; then
        cp -r "src/sparc-workflow/templates/"* "$HOME/.sparc/templates/"
        print_info "Copied SPARC templates to $HOME/.sparc/templates/"
    fi
    
    print_success "SPARC workflow directories created."
}

# Show next steps
show_next_steps() {
    print_header "Next Steps"
    
    echo "To build SPARC IDE, run the following command:"
    echo "  $SCRIPT_DIR/build-sparc-ide.sh"
    echo ""
    echo "To start the MCP server, run the following command:"
    echo "  src/mcp/start-mcp-server.sh"
    echo ""
    echo "After building SPARC IDE, you can find the installation packages in the dist/ directory."
}

# Main function
main() {
    print_header "SPARC IDE Setup"
    
    make_scripts_executable
    setup_build_environment
    apply_branding
    download_roo_code
    configure_ui
    setup_mcp_server
    create_sparc_workflow_directories
    
    print_header "Setup Completed"
    print_success "SPARC IDE environment has been set up successfully."
    
    show_next_steps
}

# Run main function
main