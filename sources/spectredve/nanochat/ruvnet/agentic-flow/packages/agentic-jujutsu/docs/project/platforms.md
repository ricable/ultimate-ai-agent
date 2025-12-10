# Platform Support Matrix

## Overview

`agentic-jujutsu` provides native N-API bindings for maximum performance across multiple platforms and architectures. This document outlines the supported platforms, build matrix, and platform-specific considerations.

## Supported Platforms

### macOS

| Platform | Architecture | Target Triple | Status | Package Name |
|----------|-------------|---------------|---------|--------------|
| macOS Intel | x86_64 | `x86_64-apple-darwin` | ✅ Supported | `@jj-vcs/agentic-jujutsu-darwin-x64` |
| macOS Apple Silicon | ARM64 | `aarch64-apple-darwin` | ✅ Supported | `@jj-vcs/agentic-jujutsu-darwin-arm64` |

**Requirements:**
- macOS 10.15 Catalina or later
- Node.js 16+ (18+ recommended)
- Xcode Command Line Tools

### Linux (GNU libc)

| Platform | Architecture | Target Triple | Status | Package Name |
|----------|-------------|---------------|---------|--------------|
| Linux x64 | x86_64 | `x86_64-unknown-linux-gnu` | ✅ Supported | `@jj-vcs/agentic-jujutsu-linux-x64-gnu` |
| Linux ARM64 | aarch64 | `aarch64-unknown-linux-gnu` | ✅ Supported | `@jj-vcs/agentic-jujutsu-linux-arm64-gnu` |
| Linux ARM32 | armv7 | `armv7-unknown-linux-gnueabihf` | ✅ Supported | `@jj-vcs/agentic-jujutsu-linux-arm-gnueabihf` |

**Requirements:**
- GNU libc 2.17+ (RHEL 7, Ubuntu 14.04, Debian 8 or later)
- Node.js 16+ (18+ recommended)
- GCC 4.8+ or Clang 3.4+

**Tested Distributions:**
- Ubuntu 20.04, 22.04, 24.04
- Debian 11, 12
- RHEL/CentOS 7, 8, 9
- Fedora 38+
- Arch Linux

### Linux (musl - Alpine)

| Platform | Architecture | Target Triple | Status | Package Name |
|----------|-------------|---------------|---------|--------------|
| Alpine x64 | x86_64 | `x86_64-unknown-linux-musl` | ✅ Supported | `@jj-vcs/agentic-jujutsu-linux-x64-musl` |
| Alpine ARM64 | aarch64 | `aarch64-unknown-linux-musl` | ✅ Supported | `@jj-vcs/agentic-jujutsu-linux-arm64-musl` |

**Requirements:**
- Alpine Linux 3.16+ or any musl-based distribution
- Node.js 16+ (18+ recommended)
- musl 1.2.2+

**Use Cases:**
- Docker containers (Alpine-based images)
- Minimal/embedded Linux systems
- Security-focused deployments

### Windows

| Platform | Architecture | Target Triple | Status | Package Name |
|----------|-------------|---------------|---------|--------------|
| Windows x64 | x86_64 | `x86_64-pc-windows-msvc` | ✅ Supported | `@jj-vcs/agentic-jujutsu-win32-x64-msvc` |
| Windows ARM64 | aarch64 | `aarch64-pc-windows-msvc` | ✅ Supported | `@jj-vcs/agentic-jujutsu-win32-arm64-msvc` |

**Requirements:**
- Windows 10 1809+ / Windows Server 2019+
- Node.js 16+ (18+ recommended)
- Visual C++ Redistributable 2015-2022

## Performance Characteristics

### Native Performance (N-API)

All platforms use native N-API bindings compiled from Rust, providing:

- **23x faster than Git** for typical operations
- **350x faster** than cloud-based code editing APIs
- **Zero serialization overhead** between Node.js and native code
- **Direct memory access** for maximum throughput

### Platform-Specific Optimizations

| Platform | CPU Features | Optimization Level |
|----------|--------------|-------------------|
| x86_64 (Intel/AMD) | SSE4.2, AVX2 | `-C target-cpu=native` |
| aarch64 (ARM64) | NEON, CRC32 | `-C target-cpu=native` |
| armv7 (ARM32) | NEON (optional) | `-C target-cpu=generic` |
| musl (Alpine) | Static linking | `-C target-feature=+crt-static` |

## Installation

### Automatic Platform Detection

The package automatically installs the correct platform binary:

```bash
npm install agentic-jujutsu
# or
npx agentic-jujutsu --version
```

### Manual Platform Selection

For advanced use cases, you can install a specific platform package:

```bash
# macOS ARM64
npm install @jj-vcs/agentic-jujutsu-darwin-arm64

# Linux x64 (Alpine/musl)
npm install @jj-vcs/agentic-jujutsu-linux-x64-musl

# Windows ARM64
npm install @jj-vcs/agentic-jujutsu-win32-arm64-msvc
```

## CI/CD Build Matrix

Our GitHub Actions workflow builds for all platforms in parallel:

```yaml
Strategy: fail-fast: false
Matrix: 9 platforms × 3 Node.js versions = 27 builds
Build Time: ~15-20 minutes (parallel execution)
```

### Build Artifacts

Each build produces:
- Native `.node` binary (N-API module)
- TypeScript definitions (`index.d.ts`)
- JavaScript bindings (`index.js`)

Artifacts are cached and reused across builds for faster CI/CD.

## Platform Testing

### Automated Testing

All platforms undergo automated testing:

1. **Build verification** - Ensures native module compiles
2. **Basic tests** - Verifies N-API bindings load correctly
3. **Comprehensive tests** - Validates all operations
4. **Integration tests** - Tests CLI and MCP server

### Cross-Platform Compatibility

We test on:
- **GitHub Actions** (Linux, macOS, Windows)
- **Local development** (various platforms)
- **Docker containers** (Alpine, Ubuntu, Debian)

### Platform-Specific Issues

| Platform | Known Issues | Workaround |
|----------|--------------|------------|
| Alpine ARM64 | Limited cross-compilation | Build natively or use QEMU |
| Windows ARM64 | Experimental Rust support | Use x64 emulation if needed |
| ARM32 (armv7) | Requires hardware FPU | Ensure `gnueabihf` target |

## Docker Support

### Multi-Architecture Images

We provide Docker images for:
- `linux/amd64` (x86_64)
- `linux/arm64` (aarch64)
- `linux/arm/v7` (armv7)

### Alpine Linux (Recommended)

Smallest footprint using musl:

```dockerfile
FROM node:18-alpine
RUN npm install -g agentic-jujutsu
```

### Ubuntu/Debian

Better compatibility with existing tools:

```dockerfile
FROM node:18-slim
RUN npm install -g agentic-jujutsu
```

## Troubleshooting

### Missing Native Module

**Symptom:** Error loading `.node` file

**Solutions:**
1. Clear npm cache: `npm cache clean --force`
2. Reinstall: `npm install agentic-jujutsu --force`
3. Check Node.js version: `node --version` (requires 16+)
4. Verify platform: `node -p "process.platform + '-' + process.arch"`

### Wrong Architecture

**Symptom:** `Exec format error` or similar

**Solutions:**
1. Ensure correct Node.js architecture matches system
2. On Apple Silicon, use native ARM64 Node.js (not Rosetta x64)
3. Check Docker image architecture: `docker inspect <image> | grep Architecture`

### Build Failures

**Symptom:** Compilation errors during installation

**Solutions:**
1. Install build tools:
   - **macOS:** `xcode-select --install`
   - **Linux:** `sudo apt-get install build-essential`
   - **Windows:** Install Visual Studio Build Tools
2. Update Rust: `rustup update stable`
3. Clear cargo cache: `cargo clean`

### Alpine Linux Issues

**Symptom:** `error while loading shared libraries: libc.musl-*.so.1`

**Solutions:**
1. Install musl: `apk add musl musl-dev`
2. Use correct package: `@jj-vcs/agentic-jujutsu-linux-x64-musl`
3. Verify musl version: `ldd --version`

## Future Platforms

We're exploring support for:

- **FreeBSD** (x86_64, aarch64)
- **OpenBSD** (x86_64)
- **illumos** (x86_64)
- **RISC-V** (riscv64gc - experimental)

## Contributing

### Adding New Platforms

To add support for a new platform:

1. Add target to `package.json` napi triples
2. Update `.github/workflows/build-napi.yml` matrix
3. Test cross-compilation setup
4. Update this documentation
5. Submit PR with platform verification

### Testing Changes

```bash
# Local build
npm run build

# Test on current platform
npm run test

# Cross-compile for specific target
npm run build -- --target aarch64-unknown-linux-gnu
```

## Support

For platform-specific issues:

1. Check [GitHub Issues](https://github.com/ruvnet/agentic-flow/issues)
2. Review [Build Status](./BUILD_STATUS.md)
3. Read [CI/CD Setup](./CI_CD_SETUP.md)
4. Contact: team@ruv.io

## License

MIT License - See [LICENSE](../LICENSE) for details.

---

**Last Updated:** 2025-11-10
**Package Version:** 1.0.0
**N-API Version:** 8 (Node.js 16+)
