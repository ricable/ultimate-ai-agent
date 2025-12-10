# Installation Guide

## System Requirements

### Minimum Requirements
- **OS**: Linux (Ubuntu 20.04+), macOS (10.15+), Windows 10/11
- **CPU**: x86_64 with AVX2 support
- **RAM**: 8GB RAM (16GB+ recommended)
- **Storage**: 5GB free space
- **Rust**: 1.70.0 or later

### Recommended Requirements
- **CPU**: Intel i7/AMD Ryzen 7 or higher
- **RAM**: 32GB RAM for production workloads
- **GPU**: NVIDIA GPU with CUDA 11.8+ (optional but recommended)
- **Storage**: SSD with 20GB+ free space
- **Network**: High-speed internet for data ingestion

### GPU Requirements (Optional)
- **NVIDIA GPU**: Compute Capability 6.0+ (Pascal architecture or newer)
- **CUDA Toolkit**: 11.8 or later
- **GPU Memory**: 4GB+ VRAM (8GB+ recommended)
- **Driver**: NVIDIA Driver 520.0+ for CUDA 11.8

## Prerequisites

### Rust Installation

Install Rust using rustup:

```bash
# Install rustup
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Add Rust to PATH
source ~/.cargo/env

# Verify installation
rustc --version
cargo --version
```

### CUDA Installation (Optional)

For GPU acceleration, install CUDA Toolkit:

#### Ubuntu/Debian
```bash
# Add NVIDIA package repository
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update

# Install CUDA Toolkit
sudo apt-get install cuda-toolkit-11-8

# Add CUDA to PATH
echo 'export PATH=/usr/local/cuda-11.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

#### macOS
```bash
# Download and install CUDA from NVIDIA website
# https://developer.nvidia.com/cuda-downloads

# Verify installation
nvcc --version
```

#### Windows
1. Download CUDA Toolkit from [NVIDIA website](https://developer.nvidia.com/cuda-downloads)
2. Run the installer and follow instructions
3. Add CUDA to system PATH

### System Dependencies

#### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    pkg-config \
    libssl-dev \
    libclang-dev \
    cmake \
    git
```

#### macOS
```bash
# Install Xcode command line tools
xcode-select --install

# Install Homebrew if not installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install cmake pkg-config openssl
```

#### Windows
1. Install [Visual Studio Build Tools](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022)
2. Install [Git for Windows](https://git-scm.com/download/win)
3. Install [CMake](https://cmake.org/download/)

## Installation Methods

### Method 1: From Source (Recommended)

Clone and build from the GitHub repository:

```bash
# Clone the repository
git clone https://github.com/ran-opt/ran-opt.git
cd ran-opt

# Build with CPU-only support
cargo build --release

# Build with GPU support (requires CUDA)
cargo build --release --features gpu

# Build with all features
cargo build --release --all-features
```

### Method 2: Pre-built Binaries

Download pre-built binaries from the [releases page](https://github.com/ran-opt/ran-opt/releases):

```bash
# Download and extract (replace with latest version)
wget https://github.com/ran-opt/ran-opt/releases/download/v1.0.0/ran-opt-linux-x86_64.tar.gz
tar -xzf ran-opt-linux-x86_64.tar.gz
cd ran-opt

# Make executable
chmod +x ran-opt
```

### Method 3: Docker Installation

Use Docker for containerized deployment:

```bash
# Pull the latest image
docker pull ranopt/ran-opt:latest

# Run with CPU support
docker run -it --rm ranopt/ran-opt:latest

# Run with GPU support (requires nvidia-docker)
docker run -it --rm --gpus all ranopt/ran-opt:gpu
```

## Configuration

### Basic Configuration

Create a configuration file:

```bash
# Copy example configuration
cp config/config.example.toml config.toml

# Edit configuration
nano config.toml
```

Example `config.toml`:

```toml
[platform]
# Enable GPU acceleration (requires CUDA)
gpu_enabled = true
# Number of worker threads (0 = auto-detect)
worker_threads = 0
# Maximum batch size for processing
max_batch_size = 1024
# Enable SIMD optimizations
simd_enabled = true

[logging]
# Log level: trace, debug, info, warn, error
level = "info"
# Log format: json, pretty
format = "pretty"
# Enable file logging
file_enabled = true
file_path = "logs/ran-opt.log"

[monitoring]
# Prometheus metrics port
metrics_port = 9090
# Enable distributed tracing
tracing_enabled = true
# Tracing endpoint
tracing_endpoint = "http://localhost:14268/api/traces"

[networking]
# gRPC server port
grpc_port = 50051
# HTTP API port
http_port = 8080
# Enable TLS
tls_enabled = false

[storage]
# Data directory
data_dir = "data"
# Cache directory
cache_dir = "cache"
# Maximum cache size (MB)
max_cache_size = 2048

[agents]
# Enable specific agents
pfs_core = true
pfs_data = true
pfs_twin = true
pfs_genai = false  # Requires API keys
pfs_logs = true
dtm_traffic = true
dtm_power = true
dtm_mobility = true
afm_detect = true
afm_correlate = true
afm_rca = true
aos_heal = true
ric_tsa = false  # Requires RIC connection
ric_conflict = false
```

### GPU Configuration

For optimal GPU performance, configure CUDA settings:

```toml
[cuda]
# CUDA device ID (0 for first GPU)
device_id = 0
# Memory pool size (bytes, 0 = auto)
memory_pool_size = 1073741824  # 1GB
# Number of CUDA streams
num_streams = 4
# Enable memory optimization
optimize_memory = true
# Enable kernel fusion
enable_fusion = true
```

### Environment Variables

Set environment variables for additional configuration:

```bash
# Required for GPU support
export CUDA_VISIBLE_DEVICES=0

# Optional: Custom library paths
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Optional: Rust-specific settings
export RUST_LOG=ran_opt=info
export RUST_BACKTRACE=1

# Optional: Performance tuning
export RAYON_NUM_THREADS=16
export OMP_NUM_THREADS=16
```

## Verification

### Basic Verification

Test the installation:

```bash
# Check version
./ran-opt --version

# Run health check
./ran-opt health-check

# Test configuration
./ran-opt --config config.toml --dry-run
```

### GPU Verification

Test GPU acceleration:

```bash
# Check CUDA availability
./ran-opt gpu-info

# Run GPU benchmark
./ran-opt benchmark --gpu

# Test neural network inference
./ran-opt test-inference --gpu
```

### Performance Verification

Run performance benchmarks:

```bash
# Quick benchmark
cargo bench --bench quick

# Full benchmark suite
cargo bench

# Memory usage test
cargo bench --bench memory_usage

# Stress test
./ran-opt stress-test --duration 60s
```

## Common Issues

### Build Errors

**Issue**: Linker errors during build
**Solution**: Install build dependencies
```bash
# Ubuntu/Debian
sudo apt-get install build-essential pkg-config libssl-dev

# macOS
xcode-select --install
```

**Issue**: CUDA not found
**Solution**: Set CUDA environment variables
```bash
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

### Runtime Errors

**Issue**: "Permission denied" errors
**Solution**: Check file permissions
```bash
chmod +x ran-opt
sudo chown -R $USER:$USER data/ logs/
```

**Issue**: Out of memory errors
**Solution**: Reduce batch size or enable memory optimization
```toml
[platform]
max_batch_size = 512

[cuda]
optimize_memory = true
```

**Issue**: GPU out of memory
**Solution**: Reduce GPU memory usage
```toml
[cuda]
memory_pool_size = 536870912  # 512MB
```

### Network Issues

**Issue**: Port already in use
**Solution**: Change ports in configuration
```toml
[networking]
grpc_port = 50052
http_port = 8081
```

**Issue**: Connection refused
**Solution**: Check firewall settings
```bash
# Ubuntu/Debian
sudo ufw allow 50051
sudo ufw allow 8080

# Check if ports are open
netstat -tuln | grep -E ':(50051|8080)'
```

## Next Steps

After successful installation:

1. **Read the [Quick Start Guide](quick-start.md)** for basic usage
2. **Check the [Examples](examples.md)** for practical use cases
3. **Review [Configuration Guide](../deployment/configuration.md)** for advanced settings
4. **Set up [Monitoring](../deployment/performance-tuning.md)** for production use

## Getting Help

If you encounter issues:

1. **Check the [Troubleshooting Guide](../reference/troubleshooting.md)**
2. **Search [GitHub Issues](https://github.com/ran-opt/ran-opt/issues)**
3. **Join [GitHub Discussions](https://github.com/ran-opt/ran-opt/discussions)**
4. **Read the [Documentation](../README.md)**

## License

This software is licensed under the MIT License. See [LICENSE](../../LICENSE) for details.