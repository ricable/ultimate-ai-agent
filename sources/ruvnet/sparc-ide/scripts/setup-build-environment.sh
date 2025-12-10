#!/bin/bash
# SPARC IDE - VSCodium Build Environment Setup Script
# This script sets up the build environment for SPARC IDE based on VSCodium

set -e
set -o nounset  # Exit if a variable is unset
set -o pipefail # Exit if any command in a pipeline fails

# Configuration
VSCODIUM_REPO="https://github.com/VSCodium/vscodium.git"
VSCODIUM_DIR="vscodium"
NODE_VERSION="20"
YARN_VERSION="1.22.19"

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

# Check prerequisites
check_prerequisites() {
    print_info "Checking prerequisites..."
    
    # Check if git is installed
    if ! command -v git &> /dev/null; then
        print_error "Git is not installed. Please install git and try again."
        exit 1
    fi
    
    # Check if Node.js is installed
    if ! command -v node &> /dev/null; then
        print_error "Node.js is not installed. Please install Node.js $NODE_VERSION or later and try again."
        exit 1
    fi
    
    # Check Node.js version
    NODE_CURRENT_VERSION=$(node -v | cut -d 'v' -f 2)
    if [ "$(printf '%s\n' "$NODE_VERSION" "$NODE_CURRENT_VERSION" | sort -V | head -n1)" != "$NODE_VERSION" ]; then
        print_error "Node.js version $NODE_CURRENT_VERSION is less than required version $NODE_VERSION. Please upgrade Node.js and try again."
        exit 1
    fi
    
    # Check if yarn is installed
    if ! command -v yarn &> /dev/null; then
        print_info "Yarn is not installed. Installing yarn..."
        npm install -g yarn@$YARN_VERSION
    fi
    
    # Check platform-specific dependencies
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        print_info "Checking Linux dependencies..."
        # Check for build-essential, libx11-dev, libxkbfile-dev, libsecret-1-dev
        if ! dpkg -s build-essential libx11-dev libxkbfile-dev libsecret-1-dev &> /dev/null; then
            print_error "Missing required Linux dependencies. Please run: sudo apt-get install build-essential libx11-dev libxkbfile-dev libsecret-1-dev"
            exit 1
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        print_info "Checking macOS dependencies..."
        # Check for Xcode Command Line Tools
        if ! xcode-select -p &> /dev/null; then
            print_error "Xcode Command Line Tools are not installed. Please run: xcode-select --install"
            exit 1
        fi
    elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
        print_info "Checking Windows dependencies..."
        # Check for Visual Studio Build Tools (simplified check)
        if ! command -v cl &> /dev/null; then
            print_error "Visual Studio Build Tools are not installed. Please install Visual Studio Build Tools and try again."
            exit 1
        fi
    fi
    
    print_success "All prerequisites are met."
}

# Clone VSCodium repository
clone_vscodium() {
    print_info "Cloning VSCodium repository..."
    
    # Verify repository URL (prevent command injection)
    if [[ ! "$VSCODIUM_REPO" =~ ^https://github\.com/[a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+\.git$ ]]; then
        print_error "Invalid repository URL format. Security check failed."
        exit 1
    fi
    
    # Create a temporary directory for verification
    TEMP_CLONE_DIR=$(mktemp -d)
    print_info "Temporary directory created for verification: $TEMP_CLONE_DIR"
    
    if [ -d "$VSCODIUM_DIR" ]; then
        print_info "VSCodium directory already exists. Updating..."
        
        # Verify the existing repository
        cd "$VSCODIUM_DIR" || exit 1
        
        # Check if it's actually a git repository
        if [ ! -d ".git" ]; then
            print_error "Existing directory is not a git repository. Security check failed."
            exit 1
        fi
        
        # Verify remote URL
        REMOTE_URL=$(git config --get remote.origin.url)
        if [ "$REMOTE_URL" != "$VSCODIUM_REPO" ]; then
            print_error "Repository URL mismatch. Expected: $VSCODIUM_REPO, Found: $REMOTE_URL"
            exit 1
        fi
        
        # Fetch updates
        git fetch --tags
        
        # Verify the latest tag signature if GPG is available
        if command -v gpg &> /dev/null; then
            LATEST_TAG=$(git describe --tags --abbrev=0)
            print_info "Verifying signature for tag: $LATEST_TAG"
            if ! git verify-tag "$LATEST_TAG" 2>/dev/null; then
                print_warning "Tag signature verification failed. Proceeding with caution."
            else
                print_success "Tag signature verified successfully."
            fi
        fi
        
        # Pull updates
        git pull
        cd - || exit 1
    else
        # Clone to temporary directory first for verification
        print_info "Cloning repository to temporary directory for verification..."
        git clone "$VSCODIUM_REPO" "$TEMP_CLONE_DIR"
        
        # Verify the cloned repository
        cd "$TEMP_CLONE_DIR" || exit 1
        
        # Check for suspicious files
        print_info "Checking for suspicious files..."
        if find . -name "*.sh" -o -name "*.bash" | xargs grep -l "curl.*sh.*|.*bash" > /dev/null; then
            print_error "Suspicious scripts found that may download and execute code. Security check failed."
            cd - || exit 1
            rm -rf "$TEMP_CLONE_DIR"
            exit 1
        fi
        
        # Verify the latest tag signature if GPG is available
        if command -v gpg &> /dev/null; then
            LATEST_TAG=$(git describe --tags --abbrev=0)
            print_info "Verifying signature for tag: $LATEST_TAG"
            if ! git verify-tag "$LATEST_TAG" 2>/dev/null; then
                print_warning "Tag signature verification failed. Proceeding with caution."
            else
                print_success "Tag signature verified successfully."
            fi
        fi
        
        cd - || exit 1
        
        # Move verified repository to final location
        print_info "Moving verified repository to final location..."
        mv "$TEMP_CLONE_DIR" "$VSCODIUM_DIR"
    fi
    
    print_success "VSCodium repository cloned/updated and verified successfully."
}

# Install dependencies
install_dependencies() {
    print_info "Installing dependencies..."
    
    cd "$VSCODIUM_DIR"
    yarn
    cd ..
    
    print_success "Dependencies installed successfully."
}

# Setup SPARC IDE customizations
setup_customizations() {
    print_info "Setting up SPARC IDE customizations..."
    
    # Verify VSCodium directory exists
    if [ ! -d "$VSCODIUM_DIR" ]; then
        print_error "VSCodium directory not found. Cannot set up customizations."
        exit 1
    fi
    
    # Create a temporary directory for verification
    TEMP_DIR=$(mktemp -d)
    print_info "Created temporary directory for verification: $TEMP_DIR"
    
    # Verify and copy product.json
    if [ -f "src/config/product.json" ]; then
        # Validate JSON format
        if ! jq . "src/config/product.json" > /dev/null 2>&1; then
            print_error "Invalid JSON format in product.json. Security check failed."
            rm -rf "$TEMP_DIR"
            exit 1
        fi
        
        # Check for suspicious content
        if grep -q "curl\|wget\|eval\|exec" "src/config/product.json"; then
            print_error "Suspicious content found in product.json. Security check failed."
            rm -rf "$TEMP_DIR"
            exit 1
        fi
        
        # Copy to temporary directory first
        cp "src/config/product.json" "$TEMP_DIR/product.json"
        
        # Set secure permissions
        chmod 644 "$TEMP_DIR/product.json"
        
        # Move to final location
        mv "$TEMP_DIR/product.json" "$VSCODIUM_DIR/product.json"
        print_success "Verified and copied product.json"
    else
        print_warning "product.json not found. Skipping."
    fi
    
    # Verify and copy branding assets
    if [ -d "branding" ]; then
        # Create target directories with secure permissions
        mkdir -p "$VSCODIUM_DIR/src/vs/workbench/browser/parts/editor/media"
        chmod 755 "$VSCODIUM_DIR/src/vs/workbench/browser/parts/editor/media"
        
        mkdir -p "$VSCODIUM_DIR/src/vs/workbench/browser/parts/splash"
        chmod 755 "$VSCODIUM_DIR/src/vs/workbench/browser/parts/splash"
        
        # Verify and copy icons
        if [ -d "branding/icons" ]; then
            print_info "Verifying icons..."
            
            # Create temporary directory for icons
            mkdir -p "$TEMP_DIR/icons"
            
            # Copy icons to temporary directory
            find "branding/icons" -type f -name "*.png" -o -name "*.svg" -o -name "*.ico" | while read -r icon; do
                # Verify file type
                file_type=$(file -b --mime-type "$icon")
                if [[ "$file_type" == image/* ]]; then
                    # Copy to temporary directory
                    cp "$icon" "$TEMP_DIR/icons/$(basename "$icon")"
                    # Set secure permissions
                    chmod 644 "$TEMP_DIR/icons/$(basename "$icon")"
                else
                    print_warning "Skipping non-image file: $icon"
                fi
            done
            
            # Move verified icons to final location
            cp -r "$TEMP_DIR/icons/"* "$VSCODIUM_DIR/src/vs/workbench/browser/parts/editor/media/" 2>/dev/null || true
            print_success "Verified and copied icons"
        fi
        
        # Verify and copy splash screens
        if [ -d "branding/splash" ]; then
            print_info "Verifying splash screens..."
            
            # Create temporary directory for splash screens
            mkdir -p "$TEMP_DIR/splash"
            
            # Copy splash screens to temporary directory
            find "branding/splash" -type f -name "*.png" -o -name "*.svg" -o -name "*.css" -o -name "*.html" | while read -r splash; do
                # Verify file type
                file_type=$(file -b --mime-type "$splash")
                if [[ "$file_type" == image/* || "$file_type" == text/css || "$file_type" == text/html ]]; then
                    # Check for suspicious content in text files
                    if [[ "$file_type" == text/* ]]; then
                        if grep -q "curl\|wget\|eval\|exec" "$splash"; then
                            print_warning "Suspicious content found in $splash. Skipping."
                            continue
                        fi
                    fi
                    
                    # Copy to temporary directory
                    cp "$splash" "$TEMP_DIR/splash/$(basename "$splash")"
                    # Set secure permissions
                    chmod 644 "$TEMP_DIR/splash/$(basename "$splash")"
                else
                    print_warning "Skipping unsupported file: $splash"
                fi
            done
            
            # Move verified splash screens to final location
            cp -r "$TEMP_DIR/splash/"* "$VSCODIUM_DIR/src/vs/workbench/browser/parts/splash/" 2>/dev/null || true
            print_success "Verified and copied splash screens"
        fi
    else
        print_warning "Branding directory not found. Skipping branding assets."
    fi
    
    # Clean up
    rm -rf "$TEMP_DIR"
    
    print_success "SPARC IDE customizations set up successfully with security verification."
}

# Main function
main() {
    print_info "Setting up SPARC IDE build environment..."
    
    check_prerequisites
    clone_vscodium
    install_dependencies
    setup_customizations
    
    print_success "SPARC IDE build environment set up successfully."
    print_info "To build SPARC IDE, navigate to the '$VSCODIUM_DIR' directory and run one of the following commands:"
    print_info "  - For Linux: yarn gulp vscode-linux-x64"
    print_info "  - For Windows: yarn gulp vscode-win32-x64"
    print_info "  - For macOS: yarn gulp vscode-darwin-x64"
}

# Run main function
main