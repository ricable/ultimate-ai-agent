# CI/CD Multi-Platform Build Setup

## Overview

This document describes the GitHub Actions CI/CD pipeline for building and publishing `agentic-jujutsu` across multiple platforms with both WASM and native N-API bindings.

## Build Workflows

### 1. Build Native Addons (`build-native.yml`)

**Triggers:**
- Push to `main`, `napi`, or `develop` branches
- Pull requests to `main` or `develop`
- Changes to package code or workflow files

**Platforms Supported:**

| Platform | Target Triple | Runner |
|----------|--------------|--------|
| macOS Intel | `x86_64-apple-darwin` | `macos-latest` |
| macOS Apple Silicon | `aarch64-apple-darwin` | `macos-latest` |
| Linux x64 | `x86_64-unknown-linux-gnu` | `ubuntu-latest` |
| Linux ARM64 | `aarch64-unknown-linux-gnu` | `ubuntu-latest` |
| Windows x64 | `x86_64-pc-windows-msvc` | `windows-latest` |
| Windows x86 | `i686-pc-windows-msvc` | `windows-latest` |

**Build Process:**
1. Checkout code
2. Setup Node.js 20 with npm cache
3. Setup Rust toolchain with target triple
4. Cache Cargo registry, index, and build artifacts
5. Install npm dependencies
6. Build native addon for target platform
7. Upload build artifacts

**Jobs:**
- `build`: Parallel builds for all 6 platforms
- `test-bindings`: Test bindings on ubuntu, macos, windows
- `verify-build`: Verify all platforms built successfully

### 2. NPM Publishing (`publish.yml`)

**Triggers:**
- Push tags matching `v*.*.*` or `agentic-jujutsu-v*.*.*`
- Manual workflow dispatch with version input

**Build Process:**
1. Build native binaries for all 6 platforms (parallel)
2. Build WASM packages for web/node/bundler/deno
3. Download all artifacts
4. Verify package integrity
5. Publish to NPM with NPM_TOKEN
6. Create GitHub Release with artifacts
7. Verify NPM package availability

**Required Secrets:**
- `NPM_TOKEN`: NPM access token with publish permissions
- `GITHUB_TOKEN`: Automatically provided by GitHub Actions

## Package Configuration

### napi Configuration

```json
{
  "napi": {
    "name": "agentic-jujutsu",
    "triples": {
      "defaults": true,
      "additional": [
        "x86_64-apple-darwin",
        "aarch64-apple-darwin",
        "x86_64-unknown-linux-gnu",
        "aarch64-unknown-linux-gnu",
        "x86_64-pc-windows-msvc",
        "i686-pc-windows-msvc"
      ]
    }
  }
}
```

### Build Scripts

- `npm run build`: Build WASM packages (release mode)
- `npm run build:dev`: Build WASM packages (dev mode)
- `npm run verify`: Verify build integrity
- `npm run test`: Run verification and WASM tests

## Local Testing

### Test Current Platform Build

```bash
cd /workspaces/agentic-flow/packages/agentic-jujutsu

# Build for current platform
cargo build --release

# List build artifacts
ls -lh target/release/*.node target/release/*.so target/release/*.dylib target/release/*.dll 2>/dev/null

# Check file type
file target/release/*.node 2>/dev/null
```

### Test WASM Build

```bash
# Build WASM packages
npm run build

# Verify packages
npm run verify

# Test WASM
npm run test:wasm
```

### Cross-Platform Build (requires cross)

```bash
# Install cross-compilation tool
cargo install cross

# Build for specific platform
cross build --release --target aarch64-unknown-linux-gnu
```

## Caching Strategy

### Cargo Caching
- **Registry cache**: `~/.cargo/registry`
- **Index cache**: `~/.cargo/git`
- **Build cache**: `packages/agentic-jujutsu/target`
- **Cache keys**: OS + target + Cargo.lock hash

### NPM Caching
- Node.js setup action automatically caches npm packages
- Cache key based on package-lock.json

## Deployment Process

### Publishing a New Version

1. **Update version in package.json**
   ```bash
   cd packages/agentic-jujutsu
   npm version patch|minor|major
   ```

2. **Create and push tag**
   ```bash
   git tag -a v1.0.1 -m "Release v1.0.1"
   git push origin v1.0.1
   ```

3. **GitHub Actions automatically:**
   - Builds all 6 native platforms
   - Builds WASM packages
   - Publishes to NPM
   - Creates GitHub Release

4. **Verify publication**
   ```bash
   npm view agentic-jujutsu
   npx agentic-jujutsu --version
   ```

### Manual Publishing

```bash
# Trigger workflow manually
gh workflow run publish.yml -f version=1.0.1
```

## Troubleshooting

### Build Failures

**Linux ARM64 cross-compilation:**
```bash
# Ensure cross-compiler is installed
sudo apt-get install gcc-aarch64-linux-gnu
```

**macOS universal binary:**
```bash
# Build both architectures
cargo build --release --target x86_64-apple-darwin
cargo build --release --target aarch64-apple-darwin

# Create universal binary (optional)
lipo -create \
  target/x86_64-apple-darwin/release/libagentic_jujutsu.dylib \
  target/aarch64-apple-darwin/release/libagentic_jujutsu.dylib \
  -output libagentic_jujutsu.dylib
```

**Windows MSVC toolchain:**
- Ensure Visual Studio Build Tools are installed
- Use `windows-latest` runner with MSVC toolchain

### Cache Issues

```bash
# Clear GitHub Actions cache
gh cache delete --all

# Local cargo clean
cargo clean
```

### NPM Publish Issues

**Token permissions:**
- Ensure NPM_TOKEN has publish access
- Token must be for a user with package permissions

**Package already exists:**
- Increment version number
- Check NPM registry: `npm view agentic-jujutsu`

## Performance Optimization

### Build Time Optimization
- Parallel matrix builds (6 platforms simultaneously)
- Cargo caching reduces build time by ~70%
- NPM caching speeds up dependency installation

### Artifact Size
- Native binaries: ~2-10 MB per platform
- WASM packages: ~500 KB - 2 MB
- Total package size: ~15-30 MB

## Security Best Practices

1. **Secret Management**
   - NPM_TOKEN stored in GitHub Secrets
   - Never commit tokens to repository
   - Use scoped tokens with minimal permissions

2. **Dependency Verification**
   - Package verification before publishing
   - Test installations after publishing
   - Automated security audits in test workflow

3. **Code Signing** (Future)
   - Sign native binaries
   - Verify signatures during installation

## Monitoring

### Build Status
- Check workflow runs: https://github.com/ruvnet/agentic-flow/actions
- Monitor build times and cache hit rates

### NPM Package Health
```bash
# Check package info
npm view agentic-jujutsu

# Check download stats
npm info agentic-jujutsu downloads

# Test installation
npx agentic-jujutsu --version
```

## Future Enhancements

- [ ] Add Android builds (aarch64-linux-android)
- [ ] Add iOS builds (aarch64-apple-ios)
- [ ] Add FreeBSD support
- [ ] Implement binary signing
- [ ] Add performance benchmarks in CI
- [ ] Automated changelog generation
- [ ] Automated version bumping
- [ ] Pre-release beta channel
