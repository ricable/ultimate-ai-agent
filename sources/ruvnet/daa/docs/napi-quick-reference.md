# NAPI CI/CD Quick Reference

## Workflows

### Build (`napi-build.yml`)
**Trigger**: Push, PR, manual  
**Purpose**: Build NAPI bindings for all platforms  
**Platforms**: Linux (x64, ARM64), macOS (x64, ARM64), Windows (x64)  
**Node**: 18, 20, 22

### Test (`napi-test.yml`)
**Trigger**: PR, push  
**Purpose**: Comprehensive testing  
**Jobs**: Lint, test, coverage, security, performance, integration

### Publish (`napi-publish.yml`)
**Trigger**: Tag `qudag-napi-v*`, manual  
**Purpose**: Build and publish to npm  
**Outputs**: npm package, GitHub release

## Local Commands

### Basic Build
```bash
cd qudag/qudag-napi
npm run build          # Release build for current platform
npm run build:debug    # Debug build
npm test               # Run tests
npm run benchmark      # Run benchmarks
```

### Multi-Platform Build
```bash
./scripts/build-all.sh                      # Current platform
./scripts/build-all.sh --target linux-x64   # Specific platform
./scripts/build-all.sh --target all         # All platforms
./scripts/build-all.sh --release --test     # Release + tests
./scripts/build-all.sh --clean --parallel 4 # Clean, parallel
```

### Available Targets
- `linux-x64`, `linux-x64-musl`
- `linux-arm64`, `linux-arm64-musl`
- `macos-x64`, `macos-arm64`
- `windows-x64`
- `all` (build all)

## Publishing

### Create Release
```bash
# 1. Update versions
npm version 0.2.0 --no-git-tag-version

# 2. Commit and tag
git add .
git commit -m "chore(napi): bump version to 0.2.0"
git tag qudag-napi-v0.2.0

# 3. Push
git push origin main
git push origin qudag-napi-v0.2.0

# 4. Monitor workflow
# https://github.com/ruvnet/daa/actions
```

### Manual Publish
```bash
# Trigger workflow manually
# Actions > NAPI Publish > Run workflow
# Enter version: 0.2.0
```

## Troubleshooting

### Build Errors
```bash
# Linux ARM64
sudo apt-get install gcc-aarch64-linux-gnu

# Linux MUSL
sudo apt-get install musl-tools

# macOS
xcode-select --install
export MACOSX_DEPLOYMENT_TARGET=10.13

# Rust target
rustup target add x86_64-unknown-linux-musl
```

### Test Failures
```bash
# Rebuild
npm run build:debug

# Clean
rm -rf node_modules target
npm ci

# Debug
RUST_BACKTRACE=1 npm test
```

### Cache Issues
```bash
# Clear npm cache
npm cache clean --force

# Clear Rust cache
cargo clean

# Rebuild
npm run build
```

## CI/CD Secrets

### Required Secrets
- `NPM_TOKEN` - npm publish authentication
- `CODECOV_TOKEN` - Code coverage upload (optional)
- `GITHUB_TOKEN` - Automatic (provided by GitHub)

### Setting Secrets
Settings > Secrets and variables > Actions > New repository secret

## File Locations

```
.github/workflows/
  ├── napi-build.yml       # Build workflow
  ├── napi-test.yml        # Test workflow
  └── napi-publish.yml     # Publish workflow

scripts/
  └── build-all.sh         # Local build script

qudag/qudag-napi/
  ├── src/                 # Rust source
  ├── Cargo.toml           # Rust config
  ├── package.json         # npm config
  ├── README.md            # Package docs
  └── *.node               # Built binaries

docs/
  ├── napi-ci-cd-guide.md  # Detailed guide
  └── napi-quick-reference.md  # This file
```

## Status Badges

```markdown
[![NAPI Build](https://github.com/ruvnet/daa/actions/workflows/napi-build.yml/badge.svg)](https://github.com/ruvnet/daa/actions/workflows/napi-build.yml)
[![NAPI Test](https://github.com/ruvnet/daa/actions/workflows/napi-test.yml/badge.svg)](https://github.com/ruvnet/daa/actions/workflows/napi-test.yml)
```

## Links

- **Workflows**: https://github.com/ruvnet/daa/actions
- **npm Package**: https://www.npmjs.com/package/@daa/qudag-native
- **NAPI-rs Docs**: https://napi.rs/
- **Issues**: https://github.com/ruvnet/daa/issues

---

**Quick Help**: `./scripts/build-all.sh --help`
