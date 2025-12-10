# Installation Guide - agentic-jujutsu

## Overview

`agentic-jujutsu` is a **wrapper** around the jujutsu VCS with AI agent features. It consists of:

1. **The npm package** (WASM bindings + CLI wrapper) - Installed via npm
2. **The jj binary** (actual Jujutsu VCS) - Must be installed separately

## Quick Install (Recommended)

### Option 1: Automatic Installation with Cargo

If you have Cargo installed, enable automatic jj installation:

```bash
# Enable auto-install
export AGENTIC_JUJUTSU_AUTO_INSTALL=true

# Install the package (will auto-install jj)
npm install -g agentic-jujutsu
```

### Option 2: Manual Installation

```bash
# Step 1: Install jj binary
cargo install --git https://github.com/martinvonz/jj jj-cli

# Step 2: Install agentic-jujutsu
npm install -g agentic-jujutsu

# Step 3: Verify
jj --version
agentic-jujutsu version
```

## Installation Methods for jj Binary

### Method 1: Cargo (Recommended - All Platforms)

```bash
# Standard installation
cargo install --git https://github.com/martinvonz/jj jj-cli

# With locked dependencies (more stable)
cargo install --git https://github.com/martinvonz/jj jj-cli --locked
```

**Prerequisites**: Install Rust/Cargo first:
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

### Method 2: Homebrew (macOS/Linux)

```bash
brew install jj
```

### Method 3: Nix (Linux/macOS)

```bash
nix-env -iA nixpkgs.jujutsu
```

### Method 4: From Source

```bash
git clone https://github.com/martinvonz/jj
cd jj
cargo build --release
# Binary will be at: target/release/jj
```

## npm Package Installation

### Global Installation (Recommended for CLI Use)

```bash
npm install -g agentic-jujutsu

# Now available globally
agentic-jujutsu status
jj-ai help
```

### Project Installation (For Programmatic Use)

```bash
npm install agentic-jujutsu

# Use in code
const jj = require('agentic-jujutsu');
```

### npx (Zero Installation)

```bash
# No installation required - runs directly
npx agentic-jujutsu status
npx agentic-jujutsu help
```

## Environment Variables

Control the installation behavior with these environment variables:

### `AGENTIC_JUJUTSU_AUTO_INSTALL`

Enable automatic jj installation during `npm install`:

```bash
export AGENTIC_JUJUTSU_AUTO_INSTALL=true
npm install -g agentic-jujutsu
```

**Default**: `false` (manual installation required)

### `AGENTIC_JUJUTSU_SKIP_INSTALL`

Skip the postinstall jj check entirely:

```bash
export AGENTIC_JUJUTSU_SKIP_INSTALL=true
npm install -g agentic-jujutsu
```

**Use case**: CI environments, containers, when you know jj is already installed

### `SKIP_JJ_INSTALL`

Alternative to `AGENTIC_JUJUTSU_SKIP_INSTALL`:

```bash
export SKIP_JJ_INSTALL=true
npm install -g agentic-jujutsu
```

### `CI`

Automatically detected. Skips installation prompts in CI environments:

```bash
# Automatically set by GitHub Actions, GitLab CI, etc.
CI=true npm install
```

## Verification

After installation, verify everything works:

```bash
# Check jj binary
jj --version
# Expected: jj 0.x.x

# Check agentic-jujutsu
agentic-jujutsu version
# Expected: agentic-jujutsu v1.0.0

# Test functionality
agentic-jujutsu status
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   agentic-jujutsu                       │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  📦 npm package (THIS INSTALLS VIA npm)                │
│   ├─ WASM bindings (Rust → JavaScript)                 │
│   ├─ CLI wrapper (bin/cli.js)                          │
│   ├─ MCP server integration                            │
│   └─ AI agent features (AST, hooks)                    │
│                                                         │
│  🦀 jj binary (INSTALL SEPARATELY)                     │
│   ├─ Actual Jujutsu VCS                                │
│   ├─ Version control operations                        │
│   └─ Installed via cargo/brew/nix                      │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

## Execution Flow

### When you run `npx agentic-jujutsu status`:

1. npm downloads/runs the agentic-jujutsu package
2. Package's postinstall script checks for jj binary
3. If jj not found, shows installation instructions
4. CLI wrapper (`bin/cli.js`) executes
5. Wrapper calls the real `jj` binary: `jj status`
6. Output is enhanced with AI agent features

### What gets installed where:

```
npm install -g agentic-jujutsu
  → Installs to: ~/.npm/lib/node_modules/agentic-jujutsu
  → Creates bin: /usr/local/bin/agentic-jujutsu

cargo install jj-cli
  → Installs to: ~/.cargo/bin/jj
  → Available as: jj command
```

## Troubleshooting

### "jj: command not found"

The jj binary is not installed or not in PATH.

**Solution**:
```bash
# Install jj
cargo install --git https://github.com/martinvonz/jj jj-cli

# Verify PATH includes ~/.cargo/bin
echo $PATH | grep cargo

# If not, add to ~/.bashrc or ~/.zshrc:
export PATH="$HOME/.cargo/bin:$PATH"
```

### "Cannot find module '../pkg/node'"

The WASM bindings weren't built during package installation.

**Solution**:
```bash
# Rebuild from source
git clone https://github.com/ruvnet/agentic-flow
cd agentic-flow/packages/agentic-jujutsu
npm run build
npm install -g .
```

### "This is a WASM simulation"

Old version of the package. Update to latest:

```bash
npm install -g agentic-jujutsu@latest
```

### Automatic installation hangs

Cargo is compiling jj, which takes 5-10 minutes first time.

**Solution**: Be patient, or install jj manually first:
```bash
# Cancel npm install (Ctrl+C)
# Install jj separately
cargo install --git https://github.com/martinvonz/jj jj-cli
# Then install package
npm install -g agentic-jujutsu
```

## Platform-Specific Notes

### macOS

```bash
# Homebrew (easiest)
brew install jj

# Or Cargo
cargo install --git https://github.com/martinvonz/jj jj-cli
```

### Linux

```bash
# Cargo (works on all distros)
cargo install --git https://github.com/martinvonz/jj jj-cli

# Nix
nix-env -iA nixpkgs.jujutsu
```

### Windows

```bash
# Cargo (via Rust installer)
# 1. Install Rust: https://rustup.rs/
# 2. Then:
cargo install --git https://github.com/martinvonz/jj jj-cli
```

### Docker/Containers

```dockerfile
FROM node:20

# Install Rust and Cargo
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Install jj
RUN cargo install --git https://github.com/martinvonz/jj jj-cli

# Install agentic-jujutsu
RUN npm install -g agentic-jujutsu

# Verify
RUN jj --version && agentic-jujutsu version
```

## CI/CD Setup

### GitHub Actions

```yaml
name: Test agentic-jujutsu

on: [push]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Setup Rust
        uses: actions-rs/toolchain@v1
        with:
          toolchain: stable

      - name: Install jj
        run: cargo install --git https://github.com/martinvonz/jj jj-cli

      - name: Install agentic-jujutsu
        run: npm install -g agentic-jujutsu

      - name: Test
        run: agentic-jujutsu status
```

## Links

- **jj Installation**: https://github.com/martinvonz/jj#installation
- **Rust/Cargo**: https://rustup.rs/
- **npm Global Install**: https://docs.npmjs.com/downloading-and-installing-packages-globally
- **Package**: https://www.npmjs.com/package/agentic-jujutsu
- **Source**: https://github.com/ruvnet/agentic-flow/tree/main/packages/agentic-jujutsu
