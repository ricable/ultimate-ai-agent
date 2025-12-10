# NAPI CI/CD Pipeline Guide

This guide covers the comprehensive CI/CD pipeline for building and publishing QuDAG NAPI bindings.

## Overview

The NAPI CI/CD pipeline consists of three main workflows:

1. **NAPI Build** (`napi-build.yml`) - Multi-platform matrix builds for development
2. **NAPI Test** (`napi-test.yml`) - Comprehensive testing on pull requests
3. **NAPI Publish** (`napi-publish.yml`) - Automated publishing to npm

## Workflows

### 1. NAPI Build Workflow

**Trigger**: Push to `main`/`develop`, PRs, or manual dispatch

**Purpose**: Build NAPI bindings for all supported platforms with multiple Node.js versions.

#### Matrix Configuration

| Platform | Architecture | Node.js Versions | Build Type |
|----------|-------------|------------------|------------|
| Linux | x86_64 (glibc) | 18, 20, 22 | Native |
| Linux | x86_64 (musl) | 18, 20, 22 | Cross-compile |
| Linux | ARM64 (glibc) | 18, 20, 22 | Cross-compile |
| Linux | ARM64 (musl) | 18, 20, 22 | Cross-compile |
| macOS | x86_64 | 18, 20, 22 | Native |
| macOS | ARM64 | 18, 20, 22 | Cross-compile |
| Windows | x86_64 (MSVC) | 18, 20, 22 | Native |

#### Features

- **Parallel Builds**: All platforms build simultaneously
- **Caching**: Rust compilation artifacts cached per target
- **Testing**: Native platforms run full test suite
- **Artifacts**: Compiled `.node` files uploaded for 7 days
- **Optimization**: Release builds use LTO, strip symbols

#### Platform-Specific Optimizations

**Linux (MUSL)**:
- Static linking for maximum portability
- No runtime dependencies on system libraries
- Ideal for Alpine Linux and minimal containers

**macOS (Cross-compilation)**:
- Universal binary support
- Minimum deployment target: macOS 10.13
- ARM64 builds on Intel runners (and vice versa)

**Windows (MSVC)**:
- Static CRT linking
- Optimized for size and performance

### 2. NAPI Test Workflow

**Trigger**: Pull requests and pushes to main branches

**Purpose**: Comprehensive testing including linting, unit tests, coverage, security audits, and performance benchmarks.

#### Jobs

##### Lint and Format
- Rust formatting check (`cargo fmt`)
- Clippy linting with strict warnings
- TypeScript type checking

##### Test Matrix
- Tests on Linux, macOS, Windows
- Node.js versions: 18, 20, 22
- Debug builds for faster iteration
- Smoke tests for benchmarks

##### Code Coverage
- Rust coverage via `cargo-llvm-cov`
- LCOV format for Codecov integration
- Upload to Codecov with flags

##### Security Audit
- `cargo audit` for Rust dependencies
- `npm audit` for Node.js dependencies
- Moderate severity threshold

##### Performance Benchmarks
- Compare base branch vs. PR branch
- Detect performance regressions
- Results posted to PR summary

##### Integration Tests
- Full NAPI binding tests
- QuDAG core compatibility checks
- End-to-end workflow validation

### 3. NAPI Publish Workflow

**Trigger**: Version tags (`qudag-napi-v*.*.*`) or manual dispatch

**Purpose**: Build release binaries for all platforms and publish to npm.

#### Process

1. **Build Release Artifacts**
   - Build for all platforms with `--release` flag
   - Run full test suite on native platforms
   - Strip debug symbols
   - Upload artifacts for 30 days

2. **Publish to npm**
   - Download all platform artifacts
   - Run `npm run artifacts` to prepare packages
   - Set version from git tag
   - Publish with public access

3. **Create GitHub Release**
   - Auto-generate release notes
   - Attach compiled `.node` files
   - Link to changelog
   - List supported platforms and Node.js versions

## Local Development

### Building Locally

```bash
# Build for current platform (debug)
cd qudag/qudag-napi
npm run build:debug

# Build for current platform (release)
npm run build

# Build for specific target
npm run build -- --target x86_64-unknown-linux-musl
```

### Multi-Platform Local Testing

Use the provided build script:

```bash
# Build for current platform
./scripts/build-all.sh

# Build for specific target with release optimizations
./scripts/build-all.sh --target linux-x64 --release

# Build for all supported platforms
./scripts/build-all.sh --target all --release --test

# Clean build
./scripts/build-all.sh --clean --release

# Parallel builds
./scripts/build-all.sh --target linux-x64 --target macos-x64 --parallel 2
```

#### Supported Targets

- `linux-x64` - Linux x86_64 with glibc
- `linux-x64-musl` - Linux x86_64 with musl (static)
- `linux-arm64` - Linux ARM64 with glibc
- `linux-arm64-musl` - Linux ARM64 with musl (static)
- `macos-x64` - macOS Intel
- `macos-arm64` - macOS Apple Silicon
- `windows-x64` - Windows x64 MSVC
- `all` - Build for all platforms

### Testing

```bash
# Run unit tests
npm test

# Run benchmarks
npm run benchmark

# Run with specific Node.js version (via nvm)
nvm use 20
npm test
```

## Publishing a New Version

### 1. Prepare Release

```bash
# Update version in package.json and Cargo.toml
npm version 0.2.0 --no-git-tag-version
cd ../..
git add qudag/qudag-napi/package.json qudag/qudag-napi/Cargo.toml

# Update CHANGELOG.md
# Add release notes, breaking changes, new features
```

### 2. Create Tag

```bash
# Commit changes
git commit -m "chore(napi): bump version to 0.2.0"

# Create and push tag
git tag qudag-napi-v0.2.0
git push origin main
git push origin qudag-napi-v0.2.0
```

### 3. Monitor Release

1. Go to **Actions** tab on GitHub
2. Watch `NAPI Publish` workflow
3. Verify all platform builds succeed
4. Check npm package: https://www.npmjs.com/package/@daa/qudag-native
5. Verify GitHub release created

### 4. Verify Published Package

```bash
# Install from npm
npm install @daa/qudag-native@0.2.0

# Test installation
node -e "const qudag = require('@daa/qudag-native'); console.log('✓ Package works');"
```

## Troubleshooting

### Build Failures

**ARM64 Cross-Compilation**:
```bash
# Install cross-compilation tools
sudo apt-get install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu

# Set linker environment variable
export CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc
```

**MUSL Static Linking**:
```bash
# Install musl tools
sudo apt-get install musl-tools

# Add Rust target
rustup target add x86_64-unknown-linux-musl
```

**macOS Cross-Compilation**:
```bash
# Ensure Xcode Command Line Tools installed
xcode-select --install

# Set minimum deployment target
export MACOSX_DEPLOYMENT_TARGET=10.13
```

### Test Failures

**Module Not Found**:
```bash
# Rebuild native module
npm run build:debug

# Clear node_modules and rebuild
rm -rf node_modules package-lock.json
npm install
```

**Segmentation Fault**:
```bash
# Enable debug symbols
npm run build:debug

# Run with backtrace
RUST_BACKTRACE=1 npm test
```

### Publish Failures

**Authentication Error**:
```bash
# Ensure NPM_TOKEN secret is set in GitHub
# Settings > Secrets > Actions > NPM_TOKEN
```

**Version Conflict**:
```bash
# Check existing versions
npm view @daa/qudag-native versions

# Ensure version in package.json is unique
```

**Missing Artifacts**:
```bash
# Verify all build jobs succeeded
# Check artifact upload step in workflow logs
```

## Performance Optimization

### Cargo.toml Optimizations

```toml
[profile.release]
lto = true              # Link-time optimization
codegen-units = 1       # Single codegen unit for better optimization
opt-level = 3           # Maximum optimization level
strip = true            # Strip debug symbols
```

### Build Flags

```bash
# Enable CPU-specific optimizations
RUSTFLAGS="-C target-cpu=native" npm run build

# Profile-guided optimization (advanced)
RUSTFLAGS="-C profile-generate=/tmp/pgo-data" npm run build
# Run benchmarks to generate profile data
RUSTFLAGS="-C profile-use=/tmp/pgo-data" npm run build
```

### Binary Size Optimization

```bash
# Use musl for smaller binaries
npm run build -- --target x86_64-unknown-linux-musl

# Strip symbols (done automatically in release)
strip qudag_native.linux-x64.node

# Check size
ls -lh *.node
```

## Security Best Practices

### Dependencies

- Regularly run `cargo audit` and `npm audit`
- Keep NAPI-rs and Rust toolchain updated
- Review dependency changes in PRs

### Secrets Management

- Never commit private keys or tokens
- Use GitHub Secrets for sensitive data
- Rotate NPM tokens periodically

### Code Review

- Require approval for NAPI binding changes
- Test on all platforms before merging
- Review benchmark results for regressions

## Monitoring and Metrics

### GitHub Actions

- Monitor workflow run times
- Track artifact sizes over time
- Review test coverage trends

### npm Package

- Monitor download statistics
- Track supported Node.js versions
- Review issue reports and feedback

### Performance

- Benchmark against previous versions
- Track regression reports
- Monitor memory usage

## Resources

- **NAPI-rs Documentation**: https://napi.rs/
- **Rust Cross-Compilation**: https://rust-lang.github.io/rustup/cross-compilation.html
- **npm Publishing**: https://docs.npmjs.com/cli/v8/commands/npm-publish
- **GitHub Actions**: https://docs.github.com/en/actions

## Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines on contributing to the NAPI bindings.

## Support

- **Issues**: https://github.com/ruvnet/daa/issues
- **Discussions**: https://github.com/ruvnet/daa/discussions
- **Email**: support@ruv.io

---

**Last Updated**: 2025-11-11
