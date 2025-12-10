# Installation Guide

This guide covers installing Neuro-Divergent in different environments and configurations.

## System Requirements

### Minimum Requirements
- **Rust**: 1.75.0 or later
- **Memory**: 4GB RAM (8GB+ recommended for large datasets)
- **Storage**: 2GB free space
- **OS**: Linux, macOS, or Windows

### Recommended Requirements
- **Rust**: Latest stable version
- **Memory**: 16GB+ RAM for production workloads
- **CPU**: Multi-core processor (4+ cores recommended)
- **Storage**: SSD with 10GB+ free space

## Installation Methods

### Method 1: Cargo (Recommended)

Add Neuro-Divergent to your `Cargo.toml`:

```toml
[dependencies]
neuro-divergent = "0.1.0"

# Optional features
neuro-divergent = { version = "0.1.0", features = ["gpu", "async", "serialization"] }
```

Then run:
```bash
cargo build
```

### Method 2: Git Dependency

For the latest development version:

```toml
[dependencies]
neuro-divergent = { git = "https://github.com/your-org/neuro-divergent", branch = "main" }
```

### Method 3: Local Development

Clone and build locally:

```bash
git clone https://github.com/your-org/neuro-divergent.git
cd neuro-divergent
cargo build --release
```

## Feature Flags

Neuro-Divergent uses feature flags to enable optional functionality:

```toml
[dependencies]
neuro-divergent = { 
    version = "0.1.0", 
    features = [
        "gpu",           # GPU acceleration (CUDA/OpenCL)
        "async",         # Async training and prediction
        "serialization", # Model save/load with serde
        "plotting",      # Built-in visualization
        "compression",   # Model compression utilities
        "distributed",   # Distributed training support
    ]
}
```

### Feature Details

| Feature | Description | Dependencies |
|---------|-------------|--------------|
| `gpu` | GPU acceleration support | CUDA 11.0+ or OpenCL 2.0+ |
| `async` | Async/await support | tokio runtime |
| `serialization` | Save/load models | serde, bincode |
| `plotting` | Visualization utilities | plotters |
| `compression` | Model compression | lz4, zstd |
| `distributed` | Multi-node training | MPI |

## Platform-Specific Instructions

### Linux (Ubuntu/Debian)

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# Install dependencies
sudo apt update
sudo apt install build-essential pkg-config libssl-dev

# For GPU support (optional)
sudo apt install nvidia-cuda-toolkit  # CUDA
# OR
sudo apt install ocl-icd-opencl-dev   # OpenCL

# Add to Cargo.toml and build
cargo add neuro-divergent
cargo build
```

### macOS

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# Install Xcode command line tools
xcode-select --install

# For GPU support (M1/M2 Macs)
# Metal support is included by default

# Add to Cargo.toml and build
cargo add neuro-divergent --features="gpu"
cargo build
```

### Windows

```powershell
# Install Rust
# Download and run rustup-init.exe from https://rustup.rs/

# Install Visual Studio Build Tools
# Download from Microsoft website

# For GPU support
# Install CUDA Toolkit 11.0+ from NVIDIA

# Add to Cargo.toml and build
cargo add neuro-divergent --features="gpu,async"
cargo build
```

## GPU Setup

### NVIDIA CUDA

```bash
# Verify CUDA installation
nvcc --version
nvidia-smi

# Set environment variables
export CUDA_ROOT=/usr/local/cuda
export PATH=$PATH:$CUDA_ROOT/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_ROOT/lib64
```

Enable in Cargo.toml:
```toml
[dependencies]
neuro-divergent = { version = "0.1.0", features = ["gpu"] }
```

### Apple Metal (M1/M2)

Metal support is enabled automatically on Apple Silicon:

```toml
[dependencies]
neuro-divergent = { version = "0.1.0", features = ["gpu"] }
```

### OpenCL

```bash
# Install OpenCL drivers
# Intel: intel-opencl-runtime
# AMD: rocm-opencl-runtime
# NVIDIA: nvidia-opencl-dev

# Verify installation
clinfo
```

## Verification

### Basic Installation Test

Create `test_installation.rs`:

```rust
use neuro_divergent::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Test basic functionality
    println!("Neuro-Divergent version: {}", neuro_divergent::info::version());
    
    // Test model creation
    let config = MLPConfig::new(10, 5);
    let model = MLP::<f64>::new(config)?;
    
    println!("✅ Installation successful!");
    println!("GPU support: {}", neuro_divergent::info::has_gpu_support());
    println!("Async support: {}", neuro_divergent::info::has_async_support());
    
    Ok(())
}
```

Run the test:
```bash
cargo run --bin test_installation
```

### Performance Benchmark

Run built-in benchmarks to verify performance:

```bash
# Basic performance test
cargo run --example benchmark --release

# GPU performance test (if enabled)
cargo run --example gpu_benchmark --release --features="gpu"
```

## Common Installation Issues

### Issue: "Could not find CUDA"

**Solution:**
```bash
# Set CUDA paths explicitly
export CUDA_ROOT=/usr/local/cuda-11.8  # Adjust version
export PATH=$PATH:$CUDA_ROOT/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_ROOT/lib64

# Verify
nvcc --version
```

### Issue: "Linker errors on Windows"

**Solution:**
- Install Visual Studio Build Tools 2019 or later
- Ensure Windows SDK is installed
- Use the x64 Native Tools Command Prompt

### Issue: "Missing OpenSSL on Linux"

**Solution:**
```bash
sudo apt install libssl-dev pkg-config
# OR
sudo yum install openssl-devel pkgconfig
```

### Issue: "Compilation takes too long"

**Solution:**
```bash
# Use more parallel jobs
export CARGO_BUILD_JOBS=8

# Enable incremental compilation
export CARGO_INCREMENTAL=1

# Use faster linker (Linux)
sudo apt install mold
export RUSTFLAGS="-C link-arg=-fuse-ld=mold"
```

## Development Setup

For contributing to Neuro-Divergent:

```bash
# Clone repository
git clone https://github.com/your-org/neuro-divergent.git
cd neuro-divergent

# Install development dependencies
cargo install cargo-watch cargo-expand cargo-criterion

# Run tests
cargo test --all-features

# Run benchmarks
cargo bench

# Watch for changes during development
cargo watch -x "test --all-features"
```

## Docker Setup

Use the official Docker image:

```dockerfile
FROM rust:1.75

# Install Neuro-Divergent
RUN cargo install neuro-divergent

# Copy your application
COPY . /app
WORKDIR /app

# Build
RUN cargo build --release

CMD ["./target/release/your-app"]
```

## Next Steps

Once installation is complete:

1. **Verify Installation**: Run the test above to ensure everything works
2. **Quick Start**: Follow the [Quick Start Guide](quick-start.md) for your first forecast
3. **Learn Concepts**: Read [Basic Concepts](basic-concepts.md) to understand the fundamentals
4. **Choose a Model**: Explore [Model Overview](models/index.md) to find the right model

## Getting Help

If you encounter installation issues:

1. **Check Prerequisites**: Ensure all system requirements are met
2. **Review Common Issues**: See the troubleshooting section above
3. **Search Issues**: Check GitHub issues for similar problems
4. **Ask for Help**: Create a new issue with:
   - Your OS and version
   - Rust version (`rustup --version`)
   - Complete error message
   - Steps to reproduce

Happy forecasting! 🚀