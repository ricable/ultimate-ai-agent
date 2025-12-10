#!/bin/bash
# Setup script for Apple Silicon MLX + Exo environment

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

# Check macOS version (13.5+ required)
check_macos_version() {
    log "Checking macOS version..."
    version=$(sw_vers -productVersion)
    if [[ ! "$version" > "13.5" ]]; then
        error "macOS 13.5+ required, found $version"
    fi
    log "macOS version $version is compatible"
}

# Install system dependencies
install_system_deps() {
    log "Installing system dependencies..."
    
    # Install Homebrew if not present
    if ! command -v brew &> /dev/null; then
        log "Installing Homebrew..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        
        # Add Homebrew to PATH for Apple Silicon Macs
        if [[ -f "/opt/homebrew/bin/brew" ]]; then
            eval "$(/opt/homebrew/bin/brew shellenv)"
        fi
    else
        log "Homebrew already installed"
    fi
    
    # Install required packages
    log "Installing required packages..."
    brew install python@3.12 git cmake openmpi
    brew install --cask docker
    
    log "System dependencies installed successfully"
}

# Setup Python environment with proper MLX version
setup_python_env() {
    log "Setting up Python environment..."
    
    # Create virtual environment
    python3.12 -m venv ~/mlx-exo-env
    source ~/mlx-exo-env/bin/activate
    
    # Install compatible MLX version (avoid 0.22.0 PyPI issue)
    log "Installing MLX framework..."
    pip install --upgrade pip
    pip install "mlx>=0.22.1"
    pip install "mlx-lm>=0.21.1"
    
    # Install Exo from source (recommended)
    log "Installing Exo framework from source..."
    if [ -d ~/exo ]; then
        warn "Exo directory already exists, removing..."
        rm -rf ~/exo
    fi
    
    git clone https://github.com/exo-explore/exo.git ~/exo
    cd ~/exo && pip install -e .
    
    # Additional dependencies for distributed computing
    log "Installing additional dependencies..."
    pip install ray[default] fastapi uvicorn prometheus-client
    pip install grpcio grpcio-tools
    pip install numpy torch transformers huggingface-hub
    pip install mpi4py
    
    log "Python environment setup complete"
}

# Configure MLX optimizations for Apple Silicon
configure_mlx_optimizations() {
    log "Configuring MLX optimizations..."
    
    cd ~/exo
    
    # Create configure_mlx.sh if it doesn't exist
    if [ ! -f configure_mlx.sh ]; then
        log "Creating MLX configuration script..."
        cat > configure_mlx.sh << 'EOF'
#!/bin/bash
# MLX optimization configuration for Apple Silicon

# Set MLX environment variables
export MLX_METAL_CAPTURE=0
export MLX_USE_METAL=1
export MLX_DISABLE_WARNING=1

# Optimize for Apple Silicon
export MACOSX_DEPLOYMENT_TARGET=13.0

# Configure Metal Performance Shaders
export MLX_GPU_MEMORY_FRACTION=0.8

echo "MLX optimizations configured for Apple Silicon"
EOF
        chmod +x configure_mlx.sh
    fi
    
    ./configure_mlx.sh
    log "MLX optimizations configured successfully"
}

# Create environment activation script
create_activation_script() {
    log "Creating environment activation script..."
    
    cat > ~/activate_mlx_env.sh << 'EOF'
#!/bin/bash
# Activate MLX-Exo environment

source ~/mlx-exo-env/bin/activate
cd ~/exo

# Set MLX environment variables
export MLX_METAL_CAPTURE=0
export MLX_USE_METAL=1
export MLX_DISABLE_WARNING=1
export MACOSX_DEPLOYMENT_TARGET=13.0
export MLX_GPU_MEMORY_FRACTION=0.8

# Add current directory to Python path
export PYTHONPATH=$PWD:$PYTHONPATH

echo "MLX-Exo environment activated"
echo "Python: $(which python)"
echo "MLX version: $(python -c 'import mlx; print(mlx.__version__)')"
EOF
    
    chmod +x ~/activate_mlx_env.sh
    log "Activation script created at ~/activate_mlx_env.sh"
}

# Verify installation
verify_installation() {
    log "Verifying installation..."
    
    source ~/mlx-exo-env/bin/activate
    
    # Test MLX import
    python -c "import mlx.core as mx; print(f'MLX version: {mx.__version__}')" || error "MLX import failed"
    
    # Test Exo import
    cd ~/exo
    python -c "import exo; print('Exo import successful')" || error "Exo import failed"
    
    # Test additional dependencies
    python -c "import ray; print(f'Ray version: {ray.__version__}')" || error "Ray import failed"
    python -c "import fastapi; print('FastAPI import successful')" || error "FastAPI import failed"
    
    log "All components verified successfully"
}

# Main execution
main() {
    log "Starting MLX-Exo environment setup..."
    
    check_macos_version
    install_system_deps
    setup_python_env
    configure_mlx_optimizations
    create_activation_script
    verify_installation
    
    log "Environment setup complete!"
    log "To activate the environment, run: source ~/activate_mlx_env.sh"
}

main "$@"