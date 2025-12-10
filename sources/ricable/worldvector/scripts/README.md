# RuVector Automation Scripts

This directory contains automation scripts to streamline development, deployment, and prevent common issues.

## 📜 Available Scripts

### 🚀 deploy.sh
Comprehensive deployment script for publishing to crates.io and npm.

Handles:
- Version management and synchronization
- Pre-deployment checks (tests, linting, formatting)
- WASM package builds
- Crate publishing to crates.io
- NPM package publishing
- GitHub Actions trigger for cross-platform builds

**Usage:**
```bash
# Full deployment
./scripts/deploy.sh

# Dry run (test without publishing)
./scripts/deploy.sh --dry-run

# See all options
./scripts/deploy.sh --help
```

**See:** [DEPLOYMENT.md](DEPLOYMENT.md) for complete documentation

### 🧪 test-deploy.sh
Tests the deployment script without publishing.

**Usage:** `./scripts/test-deploy.sh`

### 🔄 sync-lockfile.sh
Automatically syncs `package-lock.json` with `package.json` changes.

**Usage:** `./scripts/sync-lockfile.sh`

### 🪝 install-hooks.sh
Installs git hooks for automatic lock file management.

**Usage:** `./scripts/install-hooks.sh`

### 🤖 ci-sync-lockfile.sh
CI/CD script for automatic lock file fixing.

**Usage:** `./scripts/ci-sync-lockfile.sh`

### 📦 publish-crates.sh
Legacy script for publishing individual crates. Use `deploy.sh` instead.

### 🧭 validate-packages.sh
Validates package configurations and dependencies.

## 🚀 Quick Start

### For Development

1. **Install git hooks** (recommended):
   ```bash
   ./scripts/install-hooks.sh
   ```

2. **Test the hook**:
   ```bash
   cd npm/packages/ruvector
   npm install chalk
   git add package.json
   git commit -m "test: Add chalk dependency"
   # Hook automatically updates lock file
   ```

### For Deployment

1. **Test deployment script**:
   ```bash
   ./scripts/test-deploy.sh
   ```

2. **Set credentials** (required):
   ```bash
   export CRATES_API_KEY="your-crates-io-token"
   export NPM_TOKEN="your-npm-token"
   ```

3. **Run dry run** (recommended first):
   ```bash
   ./scripts/deploy.sh --dry-run
   ```

4. **Deploy**:
   ```bash
   ./scripts/deploy.sh
   ```

## 📖 Documentation

- **[DEPLOYMENT.md](DEPLOYMENT.md)** - Comprehensive deployment guide
- **[../docs/CONTRIBUTING.md](../docs/CONTRIBUTING.md)** - Development guide

## 🔐 Security

**Never commit credentials!** Always use environment variables or secure credential storage.

See [DEPLOYMENT.md#security-best-practices](DEPLOYMENT.md#security-best-practices) for details.
