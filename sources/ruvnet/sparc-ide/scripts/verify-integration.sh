#!/bin/bash
# SPARC IDE - Integration Verification Script
# This script performs a comprehensive check of all components to ensure they are properly integrated

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
REPORT_FILE="$PROJECT_ROOT/package/integration-verification-report_$TIMESTAMP.md"

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

# Initialize report
init_report() {
    print_info "Initializing integration verification report..."
    
    cat > "$REPORT_FILE" << EOL
# SPARC IDE Integration Verification Report

**Date:** $(date)
**Version:** 1.0.0

## Verification Summary

EOL
    
    print_success "Report initialized: $REPORT_FILE"
}

# Verify build scripts
verify_build_scripts() {
    print_header "Verifying Build Scripts"
    
    local status="✅ Passed"
    local details=""
    
    # Check if build scripts exist
    print_info "Checking build scripts existence..."
    for script in setup-build-environment.sh build-sparc-ide.sh package-sparc-ide.sh; do
        if [ -f "$PROJECT_ROOT/scripts/$script" ]; then
            details+="- $script: Found\n"
        else
            details+="- $script: Not found\n"
            status="❌ Failed"
        fi
    done
    
    # Check if build scripts are executable
    print_info "Checking build scripts permissions..."
    for script in setup-build-environment.sh build-sparc-ide.sh package-sparc-ide.sh; do
        if [ -x "$PROJECT_ROOT/scripts/$script" ]; then
            details+="- $script: Executable\n"
        else
            details+="- $script: Not executable\n"
            status="❌ Failed"
        fi
    done
    
    # Add to report
    cat >> "$REPORT_FILE" << EOL
### Build Scripts
**Status:** $status

\`\`\`
${details}
\`\`\`

EOL
    
    if [ "$status" == "✅ Passed" ]; then
        print_success "Build scripts verification passed."
    else
        print_error "Build scripts verification failed."
    fi
}

# Verify Roo Code integration
verify_roo_code_integration() {
    print_header "Verifying Roo Code Integration"
    
    local status="✅ Passed"
    local details=""
    
    # Check if download script exists
    print_info "Checking download-roo-code.sh existence..."
    if [ -f "$PROJECT_ROOT/scripts/download-roo-code.sh" ]; then
        details+="- download-roo-code.sh: Found\n"
    else
        details+="- download-roo-code.sh: Not found\n"
        status="❌ Failed"
    fi
    
    # Check if download script is executable
    print_info "Checking download-roo-code.sh permissions..."
    if [ -x "$PROJECT_ROOT/scripts/download-roo-code.sh" ]; then
        details+="- download-roo-code.sh: Executable\n"
    else
        details+="- download-roo-code.sh: Not executable\n"
        status="❌ Failed"
    fi
    
    # Add to report
    cat >> "$REPORT_FILE" << EOL
### Roo Code Integration
**Status:** $status

\`\`\`
${details}
\`\`\`

EOL
    
    if [ "$status" == "✅ Passed" ]; then
        print_success "Roo Code integration verification passed."
    else
        print_error "Roo Code integration verification failed."
    fi
}

# Verify UI configuration
verify_ui_configuration() {
    print_header "Verifying UI Configuration"
    
    local status="✅ Passed"
    local details=""
    
    # Check if UI configuration files exist
    print_info "Checking UI configuration files existence..."
    for file in settings.json keybindings.json product.json; do
        if [ -f "$PROJECT_ROOT/src/config/$file" ]; then
            details+="- $file: Found\n"
        else
            details+="- $file: Not found\n"
            status="❌ Failed"
        fi
    done
    
    # Check if UI configuration files are valid JSON
    print_info "Checking UI configuration files validity..."
    # Check settings.json using the clean version if available
    if [ -f "$PROJECT_ROOT/src/config/settings.clean.json" ]; then
        if jq empty "$PROJECT_ROOT/src/config/settings.clean.json" 2>/dev/null; then
            details+="- settings.json: Valid JSON (using clean version)\n"
        else
            details+="- settings.json: Invalid JSON\n"
            status="❌ Failed"
        fi
    elif [ -f "$PROJECT_ROOT/src/config/settings.json" ]; then
        if jq empty "$PROJECT_ROOT/src/config/settings.json" 2>/dev/null; then
            details+="- settings.json: Valid JSON\n"
        else
            details+="- settings.json: Invalid JSON\n"
            status="❌ Failed"
        fi
    fi
    
    # Check keybindings.json using the clean version if available
    if [ -f "$PROJECT_ROOT/src/config/keybindings.clean.json" ]; then
        if jq empty "$PROJECT_ROOT/src/config/keybindings.clean.json" 2>/dev/null; then
            details+="- keybindings.json: Valid JSON (using clean version)\n"
        else
            details+="- keybindings.json: Invalid JSON\n"
            status="❌ Failed"
        fi
    elif [ -f "$PROJECT_ROOT/src/config/keybindings.json" ]; then
        if jq empty "$PROJECT_ROOT/src/config/keybindings.json" 2>/dev/null; then
            details+="- keybindings.json: Valid JSON\n"
        else
            details+="- keybindings.json: Invalid JSON\n"
            status="❌ Failed"
        fi
    fi
    
    # Check product.json
    if [ -f "$PROJECT_ROOT/src/config/product.json" ]; then
        if jq empty "$PROJECT_ROOT/src/config/product.json" 2>/dev/null; then
            details+="- product.json: Valid JSON\n"
        else
            details+="- product.json: Invalid JSON\n"
            status="❌ Failed"
        fi
    fi
    
    # Add to report
    cat >> "$REPORT_FILE" << EOL
### UI Configuration
**Status:** $status

\`\`\`
${details}
\`\`\`

EOL
    
    if [ "$status" == "✅ Passed" ]; then
        print_success "UI configuration verification passed."
    else
        print_error "UI configuration verification failed."
    fi
}

# Verify branding customization
verify_branding_customization() {
    print_header "Verifying Branding Customization"
    
    local status="✅ Passed"
    local details=""
    
    # Check if branding script exists
    print_info "Checking apply-branding.sh existence..."
    if [ -f "$PROJECT_ROOT/scripts/apply-branding.sh" ]; then
        details+="- apply-branding.sh: Found\n"
    else
        details+="- apply-branding.sh: Not found\n"
        status="❌ Failed"
    fi
    
    # Check if branding script is executable
    print_info "Checking apply-branding.sh permissions..."
    if [ -x "$PROJECT_ROOT/scripts/apply-branding.sh" ]; then
        details+="- apply-branding.sh: Executable\n"
    else
        details+="- apply-branding.sh: Not executable\n"
        status="❌ Failed"
    fi
    
    # Check if branding assets exist
    print_info "Checking branding assets existence..."
    for platform in linux windows macos; do
        if [ -d "$PROJECT_ROOT/branding/$platform" ]; then
            details+="- $platform branding: Found\n"
        else
            details+="- $platform branding: Not found\n"
            status="❌ Failed"
        fi
    done
    
    # Add to report
    cat >> "$REPORT_FILE" << EOL
### Branding Customization
**Status:** $status

\`\`\`
${details}
\`\`\`

EOL
    
    if [ "$status" == "✅ Passed" ]; then
        print_success "Branding customization verification passed."
    else
        print_error "Branding customization verification failed."
    fi
}

# Verify security enhancements
verify_security_enhancements() {
    print_header "Verifying Security Enhancements"
    
    local status="✅ Passed"
    local details=""
    
    # Check if security reports exist
    print_info "Checking security reports existence..."
    for platform in linux windows macos; do
        if [ -f "$PROJECT_ROOT/package/$platform/security/final-security-report.txt" ]; then
            details+="- $platform security report: Found\n"
        else
            details+="- $platform security report: Not found\n"
            status="❌ Failed"
        fi
    done
    
    # Check if security reports indicate success
    print_info "Checking security reports status..."
    for platform in linux windows macos; do
        if [ -f "$PROJECT_ROOT/package/$platform/security/final-security-report.txt" ]; then
            if grep -q "passed all security checks" "$PROJECT_ROOT/package/$platform/security/final-security-report.txt"; then
                details+="- $platform security status: Passed\n"
            else
                details+="- $platform security status: Failed\n"
                status="❌ Failed"
            fi
        fi
    done
    
    # Add to report
    cat >> "$REPORT_FILE" << EOL
### Security Enhancements
**Status:** $status

\`\`\`
${details}
\`\`\`

EOL
    
    if [ "$status" == "✅ Passed" ]; then
        print_success "Security enhancements verification passed."
    else
        print_error "Security enhancements verification failed."
    fi
}

# Verify package information
verify_package_information() {
    print_header "Verifying Package Information"
    
    local status="✅ Passed"
    local details=""
    
    # Check if manifest.json exists
    print_info "Checking manifest.json existence..."
    if [ -f "$PROJECT_ROOT/package/manifest.json" ]; then
        details+="- manifest.json: Found\n"
    else
        details+="- manifest.json: Not found\n"
        status="❌ Failed"
    fi
    
    # Check if manifest.json is valid JSON
    print_info "Checking manifest.json validity..."
    if [ -f "$PROJECT_ROOT/package/manifest.json" ]; then
        if jq empty "$PROJECT_ROOT/package/manifest.json" 2>/dev/null; then
            details+="- manifest.json: Valid JSON\n"
        else
            details+="- manifest.json: Invalid JSON\n"
            status="❌ Failed"
        fi
    fi
    
    # Check if checksums exist
    print_info "Checking checksums existence..."
    for platform in linux windows macos; do
        if [ -f "$PROJECT_ROOT/package/$platform/checksums.sha256" ]; then
            details+="- $platform checksums: Found\n"
        else
            details+="- $platform checksums: Not found\n"
            status="❌ Failed"
        fi
    done
    
    # Add to report
    cat >> "$REPORT_FILE" << EOL
### Package Information
**Status:** $status

\`\`\`
${details}
\`\`\`

EOL
    
    if [ "$status" == "✅ Passed" ]; then
        print_success "Package information verification passed."
    else
        print_error "Package information verification failed."
    fi
}

# Verify documentation
verify_documentation() {
    print_header "Verifying Documentation"
    
    local status="✅ Passed"
    local details=""
    
    # Check if documentation files exist
    print_info "Checking documentation files existence..."
    for doc in installation-guide.md user-guide.md developer-guide.md troubleshooting-guide.md security-enhancements.md; do
        if [ -f "$PROJECT_ROOT/docs/$doc" ]; then
            details+="- $doc: Found\n"
        else
            details+="- $doc: Not found\n"
            status="❌ Failed"
        fi
    done
    
    # Add to report
    cat >> "$REPORT_FILE" << EOL
### Documentation
**Status:** $status

\`\`\`
${details}
\`\`\`

EOL
    
    if [ "$status" == "✅ Passed" ]; then
        print_success "Documentation verification passed."
    else
        print_error "Documentation verification failed."
    fi
}

# Finalize report
finalize_report() {
    print_header "Finalizing Report"
    
    # Add overall status
    if grep -q "❌ Failed" "$REPORT_FILE"; then
        cat >> "$REPORT_FILE" << EOL
## Overall Status
**Status:** ❌ Failed

Some components failed verification. Please review the report for details.
EOL
        print_error "Integration verification failed. Please review the report for details: $REPORT_FILE"
    else
        cat >> "$REPORT_FILE" << EOL
## Overall Status
**Status:** ✅ Passed

All components passed verification. The SPARC IDE is ready for release.
EOL
        print_success "Integration verification passed. The SPARC IDE is ready for release."
    fi
    
    # Add timestamp
    cat >> "$REPORT_FILE" << EOL

---

*Generated on: $(date)*
EOL
    
    print_success "Report finalized: $REPORT_FILE"
}

# Main function
main() {
    print_header "SPARC IDE Integration Verification"
    
    # Initialize report
    init_report
    
    # Verify all components
    verify_build_scripts
    verify_roo_code_integration
    verify_ui_configuration
    verify_branding_customization
    verify_security_enhancements
    verify_package_information
    verify_documentation
    
    # Finalize report
    finalize_report
    
    print_header "Integration Verification Complete"
}

# Run main function
main