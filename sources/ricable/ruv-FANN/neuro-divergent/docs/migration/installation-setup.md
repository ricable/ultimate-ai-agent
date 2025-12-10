# Installation & Setup Guide

This guide covers the complete setup process for migrating from Python NeuralForecast to Rust neuro-divergent, including environment preparation, dependency management, and validation setup.

## Prerequisites

### System Requirements

**Minimum Requirements**:
- **OS**: Linux (Ubuntu 20.04+), macOS (10.15+), Windows 10+
- **RAM**: 8GB (16GB recommended for large datasets)
- **Storage**: 10GB free space
- **CPU**: Multi-core processor (4+ cores recommended)

**Optional GPU Support**:
- NVIDIA GPU with CUDA 11.8+ support
- AMD GPU with ROCm support
- Apple Silicon with Metal support

### Software Dependencies

**Core Dependencies**:
- Rust 1.70+ with Cargo
- Python 3.8+ (for validation and migration tools)
- Git 2.20+

**Optional Dependencies**:
- Docker (for containerized deployment)
- CUDA toolkit (for GPU acceleration)
- MLflow (for experiment tracking)

## Rust Installation

### Install Rust Toolchain

**Linux/macOS**:
```bash
# Install rustup (Rust installer)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Reload shell environment
source ~/.cargo/env

# Verify installation
rustc --version
cargo --version
```

**Windows**:
```powershell
# Download and run rustup-init.exe from https://rustup.rs/
# Or use winget
winget install Rust.Rustup

# Verify installation
rustc --version
cargo --version
```

### Configure Rust Environment

**Update to Latest Stable**:
```bash
rustup update stable
rustup default stable
```

**Add Useful Components**:
```bash
# Add components for better development experience
rustup component add clippy rust-analyzer rustfmt

# Add targets for cross-compilation (optional)
rustup target add x86_64-unknown-linux-musl
rustup target add x86_64-pc-windows-gnu
```

**Configure Cargo**:
```bash
# Create ~/.cargo/config.toml for global configuration
mkdir -p ~/.cargo
cat > ~/.cargo/config.toml << 'EOF'
[build]
# Use multiple CPU cores for compilation
jobs = 4

[target.x86_64-unknown-linux-gnu]
# Use lld linker for faster builds (install with: apt install lld)
linker = "clang"
rustflags = ["-C", "link-arg=-fuse-ld=lld"]

[registries.crates-io]
protocol = "sparse"
EOF
```

## neuro-divergent Installation

### Method 1: From Crates.io (Recommended)

```bash
# Add to your Cargo.toml
[dependencies]
neuro-divergent = "0.1.0"
polars = { version = "0.33", features = ["lazy", "csv", "parquet", "json"] }
tokio = { version = "1.0", features = ["full"] }
anyhow = "1.0"
```

### Method 2: From Source

```bash
# Clone the repository
git clone https://github.com/your-org/neuro-divergent.git
cd neuro-divergent

# Build the project
cargo build --release

# Run tests to verify installation
cargo test

# Install the binary (optional)
cargo install --path .
```

### Method 3: Using Cargo Install

```bash
# Install directly from git
cargo install --git https://github.com/your-org/neuro-divergent.git

# Or from crates.io when published
cargo install neuro-divergent
```

## Python Environment Setup

### Maintain Python Environment

Keep your existing Python environment for validation and migration utilities:

```bash
# Create dedicated migration environment
python -m venv neuro-migration-env
source neuro-migration-env/bin/activate  # Linux/macOS
# or
neuro-migration-env\Scripts\activate.bat  # Windows

# Install required packages
pip install neuralforecast pandas polars pyarrow numpy matplotlib seaborn
pip install jupyter notebook ipykernel
```

### Install Migration Utilities

```bash
# Install additional tools for migration
pip install pandas-profiling ydata-profiling
pip install plotly dash  # For visualization
pip install mlflow wandb  # For experiment tracking
```

## Project Structure Setup

### Create Migration Project

```bash
# Create new Rust project
cargo new neuro-forecast-migration --bin
cd neuro-forecast-migration

# Set up directory structure
mkdir -p {src/models,src/data,src/utils,examples,tests,data,notebooks,scripts}
```

### Cargo.toml Configuration

```toml
[package]
name = "neuro-forecast-migration"
version = "0.1.0"
edition = "2021"

[dependencies]
neuro-divergent = "0.1.0"
polars = { version = "0.33", features = [
    "lazy", "csv", "parquet", "json", "temporal", "strings", "dtype-datetime"
] }
tokio = { version = "1.0", features = ["full"] }
anyhow = "1.0"
serde = { version = "1.0", features = ["derive"] }
clap = { version = "4.0", features = ["derive"] }
tracing = "0.1"
tracing-subscriber = "0.3"

[dev-dependencies]
approx = "0.5"
criterion = "0.5"
tempfile = "3.0"

[[bin]]
name = "migrate"
path = "src/main.rs"

[[bench]]
name = "comparison"
harness = false
```

### Environment Configuration

**Create .env file**:
```bash
# Environment variables
NEURALFORECAST_DATA_PATH=./data
RUST_LOG=info
RUST_BACKTRACE=1

# GPU configuration (if available)
CUDA_VISIBLE_DEVICES=0
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Parallel processing
RAYON_NUM_THREADS=4
POLARS_MAX_THREADS=4
```

**Create config.toml**:
```toml
[migration]
python_env = "neuro-migration-env"
data_path = "./data"
output_path = "./output"
log_level = "info"

[validation]
tolerance = 1e-6
max_samples = 10000
enable_plotting = true

[performance]
benchmark_iterations = 100
warmup_iterations = 10
timeout_seconds = 300
```

## IDE and Development Setup

### VS Code Configuration

**Install Extensions**:
- rust-analyzer
- CodeLLDB (for debugging)
- Better TOML
- Polars (for data frame support)

**VS Code Settings** (`.vscode/settings.json`):
```json
{
    "rust-analyzer.check.command": "clippy",
    "rust-analyzer.cargo.features": "all",
    "rust-analyzer.procMacro.enable": true,
    "files.associations": {
        "*.rs": "rust"
    },
    "editor.formatOnSave": true,
    "[rust]": {
        "editor.defaultFormatter": "rust-lang.rust-analyzer"
    }
}
```

**Launch Configuration** (`.vscode/launch.json`):
```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "type": "lldb",
            "request": "launch",
            "name": "Debug Migration Tool",
            "cargo": {
                "args": ["build", "--bin=migrate"],
                "filter": {
                    "name": "migrate",
                    "kind": "bin"
                }
            },
            "args": ["--help"],
            "cwd": "${workspaceFolder}"
        }
    ]
}
```

### Alternative IDEs

**CLion**:
- Install Rust plugin
- Configure Rust toolchain
- Set up run configurations

**Vim/Neovim**:
- Install rust.vim plugin
- Configure rust-analyzer with coc.nvim or nvim-lsp

## Validation Setup

### Data Validation Environment

```bash
# Create validation scripts directory
mkdir -p validation/{python,rust,comparison}

# Set up Python validation environment
cd validation/python
python -m venv venv
source venv/bin/activate
pip install neuralforecast pandas numpy matplotlib
```

### Validation Data Preparation

```python
# validation/prepare_data.py
import pandas as pd
import numpy as np
from neuralforecast.utils import AirPassengersDF

def prepare_validation_data():
    # Use standard datasets for validation
    datasets = {
        'air_passengers': AirPassengersDF,
        'synthetic_trend': generate_trend_data(),
        'synthetic_seasonal': generate_seasonal_data(),
        'synthetic_noise': generate_noisy_data()
    }
    
    for name, data in datasets.items():
        data.to_csv(f'../data/{name}.csv', index=False)
        data.to_parquet(f'../data/{name}.parquet', index=False)

def generate_trend_data():
    dates = pd.date_range('2020-01-01', periods=1000, freq='D')
    trend = np.linspace(100, 200, 1000)
    noise = np.random.normal(0, 10, 1000)
    return pd.DataFrame({
        'ds': dates,
        'unique_id': 'trend_series',
        'y': trend + noise
    })

if __name__ == '__main__':
    prepare_validation_data()
```

### Performance Benchmarking Setup

```rust
// benches/comparison.rs
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use neuro_divergent::models::LSTM;
use polars::prelude::*;

fn benchmark_lstm_training(c: &mut Criterion) {
    let data = LazyFrame::scan_csv("data/air_passengers.csv", Default::default())
        .unwrap()
        .collect()
        .unwrap();
    
    c.bench_function("lstm_training", |b| {
        b.iter(|| {
            let mut model = LSTM::builder()
                .horizon(12)
                .input_size(24)
                .build()
                .unwrap();
            model.fit(black_box(&data)).unwrap();
        })
    });
}

criterion_group!(benches, benchmark_lstm_training);
criterion_main!(benches);
```

## GPU Setup (Optional)

### CUDA Setup

**Linux**:
```bash
# Install CUDA toolkit
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2004-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda

# Add to ~/.bashrc
export PATH=/usr/local/cuda-11.8/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

**Update Cargo.toml for GPU**:
```toml
[dependencies.neuro-divergent]
version = "0.1.0"
features = ["cuda", "gpu"]
```

## Verification

### Installation Verification

```bash
# Verify Rust installation
rustc --version
cargo --version

# Verify neuro-divergent
cargo check

# Run basic tests
cargo test --lib

# Benchmark performance
cargo bench

# Check formatting
cargo fmt --check

# Run linting
cargo clippy
```

### Python Validation

```python
# validation/verify_python.py
import neuralforecast
import pandas as pd
import numpy as np

def verify_python_setup():
    print(f"NeuralForecast version: {neuralforecast.__version__}")
    print(f"Pandas version: {pd.__version__}")
    print(f"NumPy version: {np.__version__}")
    
    # Test basic functionality
    from neuralforecast.models import LSTM
    model = LSTM(h=12, input_size=24)
    print("✅ Python NeuralForecast setup verified")

if __name__ == '__main__':
    verify_python_setup()
```

### Rust Validation

```rust
// src/bin/verify.rs
use neuro_divergent::models::LSTM;
use polars::prelude::*;

fn main() -> anyhow::Result<()> {
    println!("Verifying neuro-divergent setup...");
    
    // Test basic model creation
    let model = LSTM::builder()
        .horizon(12)
        .input_size(24)
        .build()?;
    
    println!("✅ Model created successfully");
    
    // Test data loading
    let df = df! [
        "ds" => ["2023-01-01", "2023-01-02", "2023-01-03"],
        "unique_id" => ["series1", "series1", "series1"],
        "y" => [1.0, 2.0, 3.0]
    ]?;
    
    println!("✅ Data frame created successfully");
    println!("✅ neuro-divergent setup verified");
    
    Ok(())
}
```

## Next Steps

1. **Verify Installation**: Run all verification scripts
2. **Data Preparation**: Convert your existing data to polars format
3. **API Exploration**: Review [API Mapping](api-mapping.md) guide
4. **First Migration**: Start with a simple model migration
5. **Validation**: Set up continuous validation pipeline

## Troubleshooting

### Common Issues

**Rust Compilation Errors**:
```bash
# Clear cache and rebuild
cargo clean
cargo build

# Update dependencies
cargo update
```

**Linker Errors on Linux**:
```bash
# Install build essentials
sudo apt-get install build-essential pkg-config libssl-dev

# For Ubuntu/Debian
sudo apt-get install clang lld
```

**GPU Issues**:
```bash
# Check CUDA installation
nvidia-smi
nvcc --version

# Verify GPU features
cargo test --features gpu
```

**Performance Issues**:
```bash
# Build in release mode
cargo build --release

# Use optimized profile
export CARGO_PROFILE_RELEASE_OPT_LEVEL=3
export CARGO_PROFILE_RELEASE_LTO=true
```

---

**Next**: Continue to [API Mapping](api-mapping.md) for detailed Python to Rust API equivalence documentation.