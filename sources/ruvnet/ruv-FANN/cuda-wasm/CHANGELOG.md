# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2024-12-12

### Added
- Initial release of @cuda-wasm/core
- CUDA to WebAssembly transpilation
- WebGPU compute shader support
- Rust-based transpiler with memory safety
- Performance analysis and benchmarking tools
- Cross-platform support (Linux, macOS, Windows)
- Browser and Node.js compatibility
- TypeScript definitions
- Comprehensive test suite
- CLI tool for kernel transpilation
- Example projects and documentation

### Features
- 🔄 CUDA to WebAssembly transpilation
- ⚡ WebGPU native browser GPU acceleration
- 🦀 Memory-safe GPU programming with Rust
- 📊 Built-in profiling and optimization
- 🔧 Simple CLI interface
- 🌐 Cross-platform compatibility
- 📝 Full TypeScript support

### Supported Targets
- WebAssembly (WASM)
- WebGPU compute shaders
- Native GPU backends (CUDA, OpenCL, Vulkan)

### CLI Commands
- `cuda-wasm transpile` - Convert CUDA kernels to WASM/WebGPU
- `cuda-wasm analyze` - Analyze kernel performance characteristics
- `cuda-wasm benchmark` - Run performance benchmarks

### API Features
- `transpileCuda()` - Main transpilation function
- `analyzeKernel()` - Kernel analysis and optimization suggestions
- `benchmark()` - Performance benchmarking
- `createWebGPUKernel()` - WebGPU kernel creation and execution

### Installation Methods
- NPX: `npx @cuda-wasm/core`
- Global: `npm install -g @cuda-wasm/core`
- Project: `npm install @cuda-wasm/core`

[1.0.0]: https://github.com/ruvnet/ruv-FANN/releases/tag/cuda-wasm-v1.0.0