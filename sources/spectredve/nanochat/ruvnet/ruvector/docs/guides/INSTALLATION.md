# Installation Guide

This guide covers installation of Ruvector for all supported platforms: Rust, Node.js, WASM/Browser, and CLI.

## Prerequisites

### Rust
- **Rust 1.77+** (latest stable recommended)
- **Cargo** (included with Rust)

Install Rust from [rustup.rs](https://rustup.rs/):
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

### Node.js
- **Node.js 16+** (v18 or v20 recommended)
- **npm** or **yarn**

Download from [nodejs.org](https://nodejs.org/)

### Browser (WASM)
- Modern browser with WebAssembly support
- Chrome 91+, Firefox 89+, Safari 15+, Edge 91+

## Installation

### 1. Rust Library

#### Add to Cargo.toml
```toml
[dependencies]
ruvector-core = "0.1.0"
```

#### Build with optimizations
```bash
# Standard build
cargo build --release

# With SIMD optimizations (recommended)
RUSTFLAGS="-C target-cpu=native" cargo build --release

# For specific CPU features
RUSTFLAGS="-C target-feature=+avx2,+fma" cargo build --release
```

#### Optional features
```toml
[dependencies]
ruvector-core = { version = "0.1.0", features = ["agenticdb", "advanced"] }
```

Available features:
- `agenticdb`: AgenticDB API compatibility (enabled by default)
- `advanced`: Advanced features (product quantization, hybrid search)
- `simd`: SIMD intrinsics (enabled by default on x86_64)

### 2. Node.js Package

#### NPM
```bash
npm install ruvector
```

#### Yarn
```bash
yarn add ruvector
```

#### pnpm
```bash
pnpm add ruvector
```

#### Verify installation
```javascript
const { VectorDB } = require('ruvector');
console.log('Ruvector loaded successfully!');
```

#### Platform-specific binaries

Ruvector uses NAPI-RS for native bindings. Pre-built binaries are available for:
- **Linux**: x64, arm64 (glibc 2.17+)
- **macOS**: x64 (10.13+), arm64 (11.0+)
- **Windows**: x64, arm64

If no pre-built binary is available, it will compile from source (requires Rust).

### 3. Browser (WASM)

#### NPM package
```bash
npm install ruvector-wasm
```

#### Basic usage
```html
<!DOCTYPE html>
<html>
<head>
    <title>Ruvector WASM Demo</title>
</head>
<body>
    <script type="module">
        import init, { VectorDB } from './node_modules/ruvector-wasm/ruvector_wasm.js';

        async function main() {
            await init();

            const db = new VectorDB(128); // 128 dimensions
            const id = db.insert(new Float32Array(128).fill(0.1), null);
            console.log('Inserted:', id);

            const results = db.search(new Float32Array(128).fill(0.1), 10);
            console.log('Results:', results);
        }

        main();
    </script>
</body>
</html>
```

#### SIMD detection
```javascript
import { simd } from 'wasm-feature-detect';

const module = await simd()
  ? import('ruvector-wasm/ruvector_simd.wasm')
  : import('ruvector-wasm/ruvector.wasm');
```

#### Web Workers for parallelism
```javascript
// main.js
const workers = [];
const numWorkers = navigator.hardwareConcurrency || 4;

for (let i = 0; i < numWorkers; i++) {
    workers.push(new Worker('worker.js'));
}

// worker.js
importScripts('./ruvector_wasm.js');

self.onmessage = async (e) => {
    const { action, data } = e.data;
    const db = new VectorDB(128);

    if (action === 'search') {
        const results = db.search(data.query, data.k);
        self.postMessage({ results });
    }
};
```

### 4. CLI Tool

#### Install from crates.io
```bash
cargo install ruvector-cli
```

#### Build from source
```bash
git clone https://github.com/ruvnet/ruvector.git
cd ruvector
cargo install --path crates/ruvector-cli
```

#### Verify installation
```bash
ruvector --version
# Output: ruvector 0.1.0
```

#### Shell completion
```bash
# Bash
ruvector completions bash > /etc/bash_completion.d/ruvector

# Zsh
ruvector completions zsh > /usr/local/share/zsh/site-functions/_ruvector

# Fish
ruvector completions fish > ~/.config/fish/completions/ruvector.fish
```

## Platform-Specific Notes

### Linux

#### Dependencies
```bash
# Debian/Ubuntu
sudo apt-get install build-essential

# RHEL/CentOS/Fedora
sudo yum groupinstall "Development Tools"

# Arch
sudo pacman -S base-devel
```

#### Permissions
Ensure write access to database directory:
```bash
chmod 755 ./data
```

### macOS

#### Xcode Command Line Tools
```bash
xcode-select --install
```

#### Apple Silicon (M1/M2/M3)
NAPI-RS provides native arm64 binaries. For Rust, ensure you're using the correct toolchain:
```bash
rustup target add aarch64-apple-darwin
```

### Windows

#### Visual Studio Build Tools
Download from [visualstudio.microsoft.com](https://visualstudio.microsoft.com/downloads/)

Install "Desktop development with C++"

#### Windows Subsystem for Linux (WSL)
Alternatively, use WSL2:
```bash
wsl --install
```

Then follow Linux instructions.

## Docker

### Pre-built image
```bash
docker pull ruvector/ruvector:latest
docker run -p 8080:8080 ruvector/ruvector:latest
```

### Build from source
```dockerfile
FROM rust:1.77 as builder
WORKDIR /app
COPY . .
RUN cargo build --release

FROM debian:bookworm-slim
COPY --from=builder /app/target/release/ruvector-cli /usr/local/bin/
CMD ["ruvector-cli", "serve", "--host", "0.0.0.0"]
```

```bash
docker build -t ruvector .
docker run -v $(pwd)/data:/data -p 8080:8080 ruvector
```

## Verification

### Rust
```rust
use ruvector_core::VectorDB;

fn main() {
    println!("Ruvector version: {}", env!("CARGO_PKG_VERSION"));
}
```

### Node.js
```javascript
const { VectorDB } = require('ruvector');
const db = new VectorDB({ dimensions: 128 });
console.log('VectorDB created successfully!');
```

### CLI
```bash
ruvector --version
ruvector --help
```

## Troubleshooting

### Compilation Errors

**Error**: `error: linking with cc failed`
```bash
# Install build tools (see Platform-Specific Notes above)
```

**Error**: `error: failed to run custom build command for napi`
```bash
# Install Node.js and ensure it's in PATH
which node
npm --version
```

### Runtime Errors

**Error**: `cannot load native addon`
```bash
# Rebuild from source
npm rebuild ruvector
```

**Error**: `SIGSEGV` or segmentation fault
```bash
# Disable SIMD optimizations
export RUVECTOR_DISABLE_SIMD=1
```

### Performance Issues

**Slow queries**
```bash
# Enable SIMD optimizations
export RUSTFLAGS="-C target-cpu=native"
cargo build --release
```

**High memory usage**
```bash
# Enable quantization (see Advanced Features guide)
```

## Next Steps

- [Getting Started Guide](GETTING_STARTED.md) - Quick start tutorial
- [Basic Tutorial](BASIC_TUTORIAL.md) - Step-by-step examples
- [Performance Tuning](PERFORMANCE_TUNING.md) - Optimization guide
- [API Reference](../api/) - Complete API documentation

## Support

For installation issues:
1. Check [GitHub Issues](https://github.com/ruvnet/ruvector/issues)
2. Search [Stack Overflow](https://stackoverflow.com/questions/tagged/ruvector)
3. Open a new issue with:
   - OS and version
   - Rust/Node.js version
   - Error messages and logs
   - Steps to reproduce
