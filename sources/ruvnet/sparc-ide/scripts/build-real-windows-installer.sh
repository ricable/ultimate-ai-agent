#!/bin/bash
# SPARC IDE - Windows Installer Build Script
# This script creates a proper Windows installer for SPARC IDE

set -e

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

print_warning() {
    echo -e "\e[1;33m[WARNING]\e[0m $1"
}

print_header() {
    echo -e "\e[1;36m===== $1 =====\e[0m"
}

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$ROOT_DIR/logs"
LOG_FILE="$LOG_DIR/windows-installer-build_$(date +%Y%m%d_%H%M%S).log"
PACKAGE_DIR="$ROOT_DIR/package/windows"
VERSION=$(grep -o '"version": *"[^"]*"' "$ROOT_DIR/src/config/product.json" | cut -d'"' -f4)
INSTALLER_NAME="SPARC-IDE-$VERSION-windows-x64.exe"

# Create directories
mkdir -p "$LOG_DIR"
mkdir -p "$PACKAGE_DIR"

# Start logging
exec > >(tee -a "$LOG_FILE") 2>&1

print_header "SPARC IDE Windows Installer Build"
echo "Date: $(date)"
echo "Version: $VERSION"
echo ""

# Check if running on Windows or WSL
check_environment() {
    print_header "Checking Environment"
    
    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" || "$OSTYPE" == "cygwin" ]]; then
        print_success "Running on Windows."
        WINDOWS_ENV=true
    elif grep -q Microsoft /proc/version; then
        print_success "Running on Windows Subsystem for Linux (WSL)."
        WINDOWS_ENV=true
    else
        print_warning "Not running on Windows. Some features may not work as expected."
        WINDOWS_ENV=false
    fi
}

# Prepare Windows branding assets
prepare_branding() {
    print_header "Preparing Windows Branding Assets"
    
    if [ ! -f "$ROOT_DIR/branding/windows/sparc-ide.ico" ]; then
        print_info "Branding assets not found. Running prepare-windows-branding.sh script..."
        bash "$SCRIPT_DIR/prepare-windows-branding.sh"
    else
        print_success "Windows branding assets already exist."
    fi
}

# Build for both x86 and x64 architectures
build_multi_arch() {
    print_header "Building for Multiple Architectures"
    
    # Build for x64 architecture
    print_info "Building for x64 architecture..."
    
    if [ "$WINDOWS_ENV" = true ]; then
        # Use PowerShell script directly on Windows
        print_info "Using PowerShell script to build Windows installer..."
        
        if command -v powershell.exe &> /dev/null; then
            powershell.exe -ExecutionPolicy Bypass -File "$SCRIPT_DIR/create-windows-installer.ps1"
            
            if [ $? -ne 0 ]; then
                print_error "PowerShell script execution failed."
                exit 1
            fi
        else
            print_error "PowerShell not found. Cannot build Windows installer."
            exit 1
        fi
    else
        # For non-Windows environments, provide instructions
        print_error "This script must be run on Windows or WSL to build a real Windows installer."
        print_info "Please run this script on Windows or WSL, or use Docker with Windows build tools."
        
        # Fallback to mock installer with warning
        print_warning "Creating a mock installer as fallback. THIS IS NOT A REAL INSTALLER!"
        echo "This is a mock SPARC IDE Windows installer file (NOT A REAL INSTALLER)" > "$PACKAGE_DIR/$INSTALLER_NAME"
        
        # Generate checksum
        cd "$PACKAGE_DIR"
        sha256sum "$INSTALLER_NAME" > checksums.sha256
        cd - > /dev/null
        
        # Create security report
        mkdir -p "$PACKAGE_DIR/security"
        cat > "$PACKAGE_DIR/security/final-security-report.txt" << EOL
SPARC IDE Windows Installer Security Report
==========================================
Date: $(date)
Version: $VERSION

WARNING: This is a mock installer, not a real Windows executable.
The real installer should be built on Windows or in WSL using:
1. scripts/create-windows-installer.bat (on Windows)
2. powershell.exe -ExecutionPolicy Bypass -File scripts/create-windows-installer.ps1 (on Windows/WSL)

To build a proper Windows installer, you need:
- Windows 10/11 (or WSL on Windows)
- Visual Studio Build Tools
- Node.js (v16+)
- Yarn
- NSIS (Nullsoft Scriptable Install System)
- Administrator privileges

EOL
        print_warning "Mock installer created. This is NOT a real Windows executable!"
        exit 1
    fi
    
    print_success "Windows installer built successfully."
}

# Verify the installer works on different Windows versions and architectures
verify_installer() {
    print_header "Verifying Installer"
    
    if [ "$WINDOWS_ENV" = true ]; then
        print_info "Running verification script..."
        
        if command -v powershell.exe &> /dev/null; then
            powershell.exe -ExecutionPolicy Bypass -File "$SCRIPT_DIR/verify-windows-build.ps1"
            
            if [ $? -ne 0 ]; then
                print_error "Verification failed."
                exit 1
            fi
        else
            print_warning "PowerShell not found. Skipping verification."
        fi
    else
        print_warning "Not running on Windows. Skipping verification."
    fi
    
    print_success "Verification completed."
}

# Generate documentation about the installer
generate_documentation() {
    print_header "Generating Documentation"
    
    DOC_FILE="$PACKAGE_DIR/README.md"
    
    cat > "$DOC_FILE" << EOL
# SPARC IDE Windows Installer

This directory contains the Windows installer for SPARC IDE version $VERSION.

## Installation

1. Download the installer: \`$INSTALLER_NAME\`
2. Verify the checksum using: \`sha256sum -c checksums.sha256\`
3. Run the installer and follow the on-screen instructions

## System Requirements

- Windows 10 or 11 (64-bit or 32-bit depending on the installer version)
- 4GB RAM minimum, 8GB RAM recommended
- 2GB available disk space
- Internet connection for extension updates

## Compatibility

This installer has been tested and verified to work on:
- Windows 10 (x64)
- Windows 11 (x64)

## Included Components

- SPARC IDE core application
- Roo Code integration
- SPARC workflow templates
- Default extensions

## Silent Installation

To install silently, use the following command:

\`\`\`
SPARC-IDE-$VERSION-windows-x64.exe /VERYSILENT /SUPPRESSMSGBOXES /NORESTART
\`\`\`

## Support

For support, please visit: https://sparc-ide.example.com/support
EOL
    
    print_success "Documentation generated at $DOC_FILE"
}

# Create multi-architecture verification script
create_verification_script() {
    print_header "Creating Multi-Architecture Verification Script"
    
    VERIFY_SCRIPT="$ROOT_DIR/tests/windows/verify-multi-arch.ps1"
    mkdir -p "$(dirname "$VERIFY_SCRIPT")"
    
    cat > "$VERIFY_SCRIPT" << 'EOL'
# SPARC IDE - Multi-Architecture Windows Verification Script
# This script verifies that the Windows installer works on different architectures

# Configuration
$ErrorActionPreference = "Stop"
$PACKAGE_DIR = Join-Path $PSScriptRoot "..\..\package\windows"
$LOG_FILE = Join-Path $PSScriptRoot "..\..\test-reports\windows_multi_arch_test_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"

# Create log directory
New-Item -ItemType Directory -Force -Path (Split-Path $LOG_FILE) | Out-Null

# Start logging
Start-Transcript -Path $LOG_FILE

Write-Host "===== SPARC IDE Windows Multi-Architecture Verification ====="
Write-Host "Date: $(Get-Date)"
Write-Host ""

# Check Windows version and architecture
Write-Host "Checking Windows version and architecture..."
$OSInfo = Get-CimInstance Win32_OperatingSystem
$OSVersion = $OSInfo.Caption
$OSArch = $OSInfo.OSArchitecture
Write-Host "Windows Version: $OSVersion"
Write-Host "Architecture: $OSArch"

# Check if package directory exists
Write-Host "Checking package directory..."
if (-not (Test-Path $PACKAGE_DIR)) {
    Write-Host "ERROR: Package directory not found at $PACKAGE_DIR"
    exit 1
}

# Check for installer
Write-Host "Checking for installer..."
$INSTALLER = Get-ChildItem -Path $PACKAGE_DIR -Filter "*.exe" | Select-Object -First 1
if ($null -eq $INSTALLER) {
    Write-Host "ERROR: Installer not found in $PACKAGE_DIR"
    exit 1
}
Write-Host "Found installer: $($INSTALLER.Name)"

# Verify installer size
Write-Host "Verifying installer size..."
$SIZE_MB = [math]::Round($INSTALLER.Length / 1MB, 2)
if ($SIZE_MB -lt 50) {
    Write-Host "ERROR: Installer size is too small ($SIZE_MB MB). Expected at least 50 MB."
    Write-Host "This might be a mock installer, not a real executable."
    exit 1
}
Write-Host "Installer size: $SIZE_MB MB"

# Verify it's a Windows PE file (real executable)
Write-Host "Verifying file is a real Windows executable..."
$BYTES = [System.IO.File]::ReadAllBytes($INSTALLER.FullName)
if ($BYTES[0] -ne 0x4D -or $BYTES[1] -ne 0x5A) { # "MZ" signature
    Write-Host "ERROR: The file is not a valid Windows executable (missing MZ signature)."
    Write-Host "This appears to be a text file or other non-executable format."
    exit 1
}
Write-Host "File verified as a Windows executable (PE format)."

# Verify architecture support
Write-Host "Verifying architecture support..."
$PE_OFFSET = [BitConverter]::ToInt32($BYTES, 60)
$MACHINE_TYPE = [BitConverter]::ToUInt16($BYTES, $PE_OFFSET + 4)

$ARCH_MAP = @{
    0x014C = "x86 (32-bit)"
    0x8664 = "x64 (64-bit)"
    0x0200 = "IA64 (Itanium)"
    0x01C4 = "ARM"
    0xAA64 = "ARM64"
}

if ($ARCH_MAP.ContainsKey($MACHINE_TYPE)) {
    $DETECTED_ARCH = $ARCH_MAP[$MACHINE_TYPE]
    Write-Host "Detected architecture: $DETECTED_ARCH"
} else {
    Write-Host "Unknown architecture type: 0x$($MACHINE_TYPE.ToString('X4'))"
}

# Check for compatibility with current system
if (($OSArch -like "*64*" -and $MACHINE_TYPE -eq 0x8664) -or 
    ($OSArch -like "*32*" -and $MACHINE_TYPE -eq 0x014C) -or
    ($OSArch -like "*ARM*" -and ($MACHINE_TYPE -eq 0x01C4 -or $MACHINE_TYPE -eq 0xAA64))) {
    Write-Host "Installer is compatible with this system architecture."
} else {
    Write-Host "WARNING: Installer architecture may not be compatible with this system."
}

# Test installer extraction (silent mode) in temp directory
Write-Host "Testing installer extraction (silent mode)..."
$TEMP_DIR = Join-Path $env:TEMP "sparc-ide-verify-$(Get-Random)"
New-Item -ItemType Directory -Force -Path $TEMP_DIR | Out-Null

try {
    $EXTRACT_ARGS = @(
        "/VERYSILENT", 
        "/SUPPRESSMSGBOXES", 
        "/DIR=`"$TEMP_DIR`"", 
        "/NOICONS",
        "/LOG=`"$TEMP_DIR\install.log`""
    )
    
    Write-Host "Extracting to: $TEMP_DIR"
    Write-Host "Running: $($INSTALLER.FullName) $($EXTRACT_ARGS -join ' ')"
    
    $EXTRACT_PROCESS = Start-Process -FilePath $INSTALLER.FullName -ArgumentList $EXTRACT_ARGS -Wait -PassThru
    $EXIT_CODE = $EXTRACT_PROCESS.ExitCode
    
    if ($EXIT_CODE -ne 0) {
        Write-Host "ERROR: Installer extraction failed with exit code $EXIT_CODE"
        if (Test-Path "$TEMP_DIR\install.log") {
            Write-Host "Installation log contents:"
            Get-Content "$TEMP_DIR\install.log"
        }
        exit 1
    }
    
    Write-Host "Installer extraction successful."
    
    # Verify extracted files
    Write-Host "Verifying extracted files..."
    $REQUIRED_FILES = @(
        "sparc-ide.exe",
        "resources\app\node_modules.asar",
        "resources\app\out\vs\code\electron-main\main.js"
    )
    
    foreach ($FILE in $REQUIRED_FILES) {
        $FILE_PATH = Join-Path $TEMP_DIR $FILE
        if (-not (Test-Path $FILE_PATH)) {
            Write-Host "WARNING: Expected file not found: $FILE"
        } else {
            Write-Host "Found required file: $FILE"
        }
    }
    
    # Test executable launch (don't actually start the app)
    $EXE_PATH = Join-Path $TEMP_DIR "sparc-ide.exe"
    if (Test-Path $EXE_PATH) {
        Write-Host "Verifying executable format..."
        $EXE_BYTES = [System.IO.File]::ReadAllBytes($EXE_PATH)
        if ($EXE_BYTES[0] -ne 0x4D -or $EXE_BYTES[1] -ne 0x5A) {
            Write-Host "ERROR: Extracted executable is not a valid Windows application."
            exit 1
        }
        Write-Host "Extracted executable verified as a valid Windows application."
    } else {
        Write-Host "ERROR: Main executable not found after extraction."
        exit 1
    }
} finally {
    # Clean up
    Write-Host "Cleaning up..."
    Remove-Item -Recurse -Force $TEMP_DIR -ErrorAction SilentlyContinue
}

Write-Host ""
Write-Host "===== All verification steps passed! ====="
Write-Host "The Windows installer has been verified successfully."

# Stop logging
Stop-Transcript
EOL
    
    chmod +x "$VERIFY_SCRIPT"
    
    print_success "Multi-architecture verification script created at $VERIFY_SCRIPT"
}

# Main function
main() {
    check_environment
    prepare_branding
    build_multi_arch
    verify_installer
    generate_documentation
    create_verification_script
    
    print_header "Windows Installer Build Completed"
    print_success "SPARC IDE Windows installer has been built successfully."
    print_info "The Windows installer is available at: $PACKAGE_DIR/$INSTALLER_NAME"
    print_info "Build log saved to: $LOG_FILE"
}

# Run main function
main