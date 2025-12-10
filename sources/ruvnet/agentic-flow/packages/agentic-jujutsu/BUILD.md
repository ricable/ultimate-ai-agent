# Building agentic-jujutsu Native Binaries

This document explains how to build and publish agentic-jujutsu with native binaries for all supported platforms.

## Supported Platforms

agentic-jujutsu supports 8 platforms through NAPI-RS:

1. **Linux x64 GNU** (`linux-x64-gnu`) - Standard Linux distributions
2. **Linux ARM64 GNU** (`linux-arm64-gnu`) - ARM64 Linux (Raspberry Pi 4+)
3. **Linux x64 MUSL** (`linux-x64-musl`) - Alpine Linux x64
4. **Linux ARM64 MUSL** (`linux-arm64-musl`) - Alpine Linux ARM64
5. **Linux ARMv7** (`linux-arm-gnueabihf`) - ARM32 devices
6. **macOS x64** (`darwin-x64`) - Intel Macs
7. **macOS ARM64** (`darwin-arm64`) - Apple Silicon Macs (M1/M2/M3)
8. **Windows x64** (`win32-x64-msvc`) - Windows 64-bit

## Automated Builds via GitHub Actions

The repository includes a GitHub Actions workflow that automatically builds binaries for all platforms.

### Workflow: `.github/workflows/build-agentic-jujutsu.yml`

**Triggers:**
- Push to `main` branch (paths: `packages/agentic-jujutsu/**`)
- Pull requests to `main`
- Tags matching `agentic-jujutsu-v*` (e.g., `agentic-jujutsu-v2.3.6`)
- Manual workflow dispatch

**Build Matrix:**
- Uses ubuntu-22.04 for all Linux builds
- Uses macos-13 for Intel Mac builds
- Uses macos-14 for Apple Silicon builds
- Uses windows-2022 for Windows builds

**Process:**
1. Checks out code
2. Sets up Node.js 20 and Rust toolchain
3. Installs cross-compilation tools (for ARM targets)
4. Builds native module with `napi build`
5. Tests on native platforms
6. Uploads artifacts

### Publishing Process

When a tag `agentic-jujutsu-v*` is pushed, the workflow automatically:

1. **Downloads all platform binaries** from build artifacts
2. **Creates platform-specific packages**:
   - `agentic-jujutsu-darwin-x64`
   - `agentic-jujutsu-darwin-arm64`
   - `agentic-jujutsu-linux-x64-gnu`
   - `agentic-jujutsu-linux-x64-musl`
   - `agentic-jujutsu-linux-arm64-gnu`
   - `agentic-jujutsu-linux-arm64-musl`
   - `agentic-jujutsu-linux-arm-gnueabihf`
   - `agentic-jujutsu-win32-x64-msvc`
3. **Publishes platform packages** to npm
4. **Publishes main package** with optionalDependencies

## Manual Build Instructions

### Prerequisites

- Node.js 16+ (recommended: 20)
- Rust toolchain (via rustup)
- NAPI-RS CLI: `npm install -g @napi-rs/cli`

### Building for Your Platform

```bash
cd packages/agentic-jujutsu
npm install
npm run build  # or: napi build --platform --release
```

This creates a `.node` file for your current platform:
- `agentic-jujutsu.linux-x64-gnu.node` (on Linux)
- `agentic-jujutsu.darwin-arm64.node` (on Apple Silicon Mac)
- etc.

### Cross-Compilation (Advanced)

#### Linux ARM64 from x64:

```bash
# Install cross-compilation tools
sudo apt-get install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu

# Add target
rustup target add aarch64-unknown-linux-gnu

# Build
CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc \
  napi build --platform --release --target aarch64-unknown-linux-gnu
```

#### MUSL targets:

```bash
# Install musl tools
sudo apt-get install musl-tools

# Add target
rustup target add x86_64-unknown-linux-musl

# Build
napi build --platform --release --target x86_64-unknown-linux-musl
```

## Releasing a New Version

### Step 1: Update Version

```bash
cd packages/agentic-jujutsu
npm version 2.3.6  # or patch/minor/major
```

### Step 2: Create and Push Tag

```bash
git add package.json
git commit -m "chore: bump agentic-jujutsu to v2.3.6"
git tag agentic-jujutsu-v2.3.6
git push origin main --tags
```

### Step 3: Monitor GitHub Actions

1. Go to: https://github.com/ruvnet/agentic-flow/actions
2. Watch the "Build Agentic-Jujutsu Native Modules" workflow
3. Verify all 8 platforms build successfully
4. Verify publish job completes

### Step 4: Verify npm Packages

```bash
# Check main package
npm view agentic-jujutsu@2.3.6

# Check platform packages
npm view agentic-jujutsu-darwin-arm64@2.3.6
npm view agentic-jujutsu-linux-x64-gnu@2.3.6
# ... etc for all 8 platforms
```

## Testing Installation

### Test on Your Platform

```bash
npm install agentic-jujutsu@2.3.6
node -e "const {JjWrapper} = require('agentic-jujutsu'); console.log('✓ Loaded')"
```

### Test Platform-Specific Package

```bash
npm install agentic-jujutsu-darwin-arm64@2.3.6
ls node_modules/agentic-jujutsu-darwin-arm64/*.node
```

## Troubleshooting

### Build Fails on ARM Targets

Ensure cross-compilation tools are installed:
```bash
sudo apt-get update
sudo apt-get install -y \
  gcc-aarch64-linux-gnu \
  g++-aarch64-linux-gnu \
  gcc-arm-linux-gnueabihf \
  g++-arm-linux-gnueabihf
```

### Wrong .node File Name

NAPI-RS creates files with platform suffix. The workflow handles mapping:
- Build output: `agentic-jujutsu.linux-x64-gnu.node`
- Artifact location: `npm/linux-x64-gnu/agentic-jujutsu.linux-x64-gnu.node`

### Platform Package Not Found

Platform packages must be published BEFORE the main package. The workflow does this automatically, but for manual publishing:

```bash
# 1. Build all platforms
# 2. Publish platform packages first
cd npm/darwin-arm64
npm publish --access public

# 3. Then publish main package
cd ../..
npm publish --access public
```

## Architecture

### NAPI-RS Structure

```
packages/agentic-jujutsu/
├── src/                       # Rust source code
│   ├── lib.rs                 # NAPI bindings
│   ├── operations.rs          # Jujutsu operations
│   └── ...
├── index.js                   # JavaScript entry point
├── index.d.ts                 # TypeScript definitions
├── *.node                     # Built native modules (gitignored)
└── npm/                       # Platform packages (created by CI)
    ├── darwin-arm64/
    │   ├── package.json
    │   └── agentic-jujutsu.darwin-arm64.node
    ├── linux-x64-gnu/
    │   ├── package.json
    │   └── agentic-jujutsu.linux-x64-gnu.node
    └── ...
```

### Loading Mechanism

The main `index.js` checks your platform and loads the appropriate `.node` file:
1. Tries local file (e.g., `./agentic-jujutsu.darwin-arm64.node`)
2. Falls back to platform package (e.g., `require('agentic-jujutsu-darwin-arm64')`)

This is why optionalDependencies are needed in `package.json`.

## CI/CD Environment Variables

The workflow requires the following GitHub secret:
- `NPM_TOKEN` - npm access token with publish permissions

**CRITICAL**: The automated workflow will fail without this secret configured. All platform builds will succeed, but the publish job will fail with `ENEEDAUTH` errors.

To set up:
1. Generate token at https://www.npmjs.com/settings/YOUR_USERNAME/tokens
   - Token type: "Automation" (for CI/CD)
   - Permissions: "Read and write" (for publishing)
2. Add as repository secret at https://github.com/ruvnet/agentic-flow/settings/secrets/actions
   - Name: `NPM_TOKEN`
   - Value: Your npm automation token
3. Verify in workflow runs that `NODE_AUTH_TOKEN` is set (masked in logs)

### Manual Publishing (if NPM_TOKEN not configured)

If the GitHub secret is not configured, you can publish manually after builds complete:

```bash
# Wait for all 8 platform builds to complete
gh run watch <run-id>

# Download artifacts
gh run download <run-id> -D /tmp/artifacts

# Build local binary
cd packages/agentic-jujutsu
npm install
npm run build

# Publish (requires npm login)
npm whoami  # Verify authentication
npm publish --access public
```

The main package includes the linux-x64-gnu binary by default, so it will work on Linux platforms immediately after manual publishing.

## Performance Notes

### Build Times (approximate)

- Linux x64 GNU: ~3-4 minutes
- Linux ARM64 (cross): ~4-5 minutes
- macOS x64: ~4-5 minutes
- macOS ARM64: ~3-4 minutes
- Windows x64: ~5-6 minutes
- MUSL targets: ~4-5 minutes

### Binary Sizes

- Linux GNU: ~27 MB
- Linux MUSL: ~28 MB
- macOS: ~4-5 MB (Apple's stripping)
- Windows: ~4-5 MB

## References

- NAPI-RS Documentation: https://napi.rs/
- GitHub Actions: https://docs.github.com/en/actions
- Cross-compilation: https://rust-lang.github.io/rustup/cross-compilation.html

