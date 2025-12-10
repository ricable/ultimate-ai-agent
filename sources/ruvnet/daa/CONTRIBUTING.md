# Contributing to DAA with NAPI-rs

Thank you for your interest in contributing to the DAA (Distributed Agentic Architecture) project! This guide will help you get started with development, especially for the NAPI-rs native bindings.

---

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [Development Workflow](#development-workflow)
- [Code Style Guidelines](#code-style-guidelines)
- [Testing Requirements](#testing-requirements)
- [Pull Request Process](#pull-request-process)
- [Release Process](#release-process)
- [Community](#community)

---

## Code of Conduct

We are committed to providing a welcoming and inclusive environment for all contributors. Please:

- Be respectful and considerate
- Welcome newcomers and help them get started
- Focus on constructive feedback
- Report unacceptable behavior to the maintainers

---

## Getting Started

### Prerequisites

Before you begin, ensure you have:

- **Rust 1.75+** (`rustup install stable`)
- **Node.js 18+** (`nvm install 20`)
- **Git** for version control
- **VS Code** (recommended) or your preferred editor

### Platform-Specific Requirements

#### Linux
```bash
sudo apt-get update
sudo apt-get install build-essential pkg-config libssl-dev
```

#### macOS
```bash
xcode-select --install
brew install openssl
```

#### Windows
- Install [Visual Studio Build Tools](https://visualstudio.microsoft.com/downloads/)
- Select "Desktop development with C++"
- Install [Python 3](https://www.python.org/downloads/)

---

## Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub first
git clone https://github.com/YOUR-USERNAME/daa.git
cd daa

# Add upstream remote
git remote add upstream https://github.com/ruvnet/daa.git
```

### 2. Install Dependencies

```bash
# Install Node.js dependencies
npm install

# Install Rust toolchain
rustup install stable
rustup default stable

# Install NAPI-rs CLI
npm install -g @napi-rs/cli
```

### 3. Build the Project

```bash
# Build all Rust crates
cargo build --workspace

# Build NAPI bindings
cd qudag/qudag-napi
npm run build

# Build SDK
cd ../../packages/daa-sdk
npm run build
```

### 4. Verify Installation

```bash
# Run tests
cargo test --workspace
npm test

# Run examples
npm run example:basic-crypto
```

---

## Project Structure

```
daa/
├── qudag/                    # Quantum-resistant networking
│   ├── qudag-napi/          # NAPI-rs bindings
│   │   ├── src/
│   │   │   ├── lib.rs       # Main entry point
│   │   │   ├── crypto.rs    # Crypto operations
│   │   │   ├── vault.rs     # Password vault
│   │   │   └── exchange.rs  # Token exchange
│   │   ├── Cargo.toml
│   │   └── package.json
│   └── qudag-wasm/          # WASM bindings (browser)
│
├── packages/
│   └── daa-sdk/             # Unified SDK
│       ├── src/
│       │   ├── index.ts     # Main exports
│       │   ├── platform.ts  # Platform detection
│       │   └── cli/         # CLI tools
│       └── package.json
│
├── docs/                     # Documentation
│   ├── api-reference.md
│   ├── migration-guide.md
│   ├── troubleshooting.md
│   └── napi-rs-integration-plan.md
│
├── examples/                 # Example code
│   ├── basic-crypto.ts
│   ├── orchestrator.ts
│   ├── federated-learning.ts
│   ├── full-stack-agent.ts
│   └── performance-benchmark.ts
│
└── tests/                    # Integration tests
```

---

## Development Workflow

### Creating a New Feature

1. **Create a branch**:
```bash
git checkout -b feature/my-new-feature
```

2. **Make changes**:
```bash
# Edit files
code qudag/qudag-napi/src/crypto.rs

# Build
cd qudag/qudag-napi
npm run build

# Test
npm test
```

3. **Commit**:
```bash
git add .
git commit -m "feat: add ML-KEM-1024 support"
```

4. **Push and create PR**:
```bash
git push origin feature/my-new-feature
# Create PR on GitHub
```

### Working with NAPI-rs

#### Adding a New Binding

```rust
// src/new_module.rs
use napi::bindgen_prelude::*;
use napi_derive::napi;

#[napi]
pub struct NewFeature {}

#[napi]
impl NewFeature {
    #[napi(constructor)]
    pub fn new() -> Result<Self> {
        Ok(Self {})
    }

    #[napi]
    pub fn do_something(&self, input: Buffer) -> Result<Buffer> {
        // Implementation
        Ok(input)
    }
}
```

```rust
// src/lib.rs
mod new_module;
pub use new_module::*;
```

#### Testing Your Binding

```typescript
// test.ts
import { NewFeature } from '@daa/qudag-native';

const feature = new NewFeature();
const result = feature.doSomething(Buffer.from('test'));
console.log(result);
```

#### Generating TypeScript Definitions

```bash
npm run build
# TypeScript definitions are auto-generated in index.d.ts
```

---

## Code Style Guidelines

### Rust

#### Formatting
```bash
# Format code
cargo fmt --all

# Check formatting
cargo fmt --all -- --check

# Lint
cargo clippy --all-targets --all-features
```

#### Conventions
- Use `snake_case` for functions and variables
- Use `PascalCase` for types and structs
- Add doc comments for all public items:
```rust
/// Calculate ML-KEM-768 shared secret
///
/// # Arguments
/// * `ciphertext` - Encapsulated ciphertext (1088 bytes)
/// * `secret_key` - Recipient's secret key (2400 bytes)
///
/// # Returns
/// 32-byte shared secret
#[napi]
pub fn decapsulate(&self, ciphertext: Buffer, secret_key: Buffer) -> Result<Buffer> {
    // Implementation
}
```

### TypeScript

#### Formatting
```bash
# Format code
npm run format

# Lint
npm run lint
```

#### Conventions
- Use `camelCase` for functions and variables
- Use `PascalCase` for classes and interfaces
- Add JSDoc comments:
```typescript
/**
 * Process a task with the agent
 *
 * @param task - Task to process
 * @returns Promise resolving to task result
 */
async processTask(task: Task): Promise<TaskResult> {
    // Implementation
}
```

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add ML-KEM-1024 support
fix: correct buffer size validation
docs: update API reference
test: add ML-DSA verification tests
refactor: optimize BLAKE3 hashing
chore: update dependencies
```

---

## Testing Requirements

### Unit Tests (Rust)

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ml_kem_keygen() {
        let mlkem = MlKem768::new().unwrap();
        let keypair = mlkem.generate_keypair().unwrap();

        assert_eq!(keypair.public_key.len(), 1184);
        assert_eq!(keypair.secret_key.len(), 2400);
    }

    #[tokio::test]
    async fn test_async_operation() {
        let result = some_async_fn().await;
        assert!(result.is_ok());
    }
}
```

Run tests:
```bash
cargo test --workspace
cargo test --package qudag-napi
```

### Integration Tests (Node.js)

```typescript
import { test } from 'node:test';
import { strict as assert } from 'node:assert';
import { MlKem768 } from '@daa/qudag-native';

test('ML-KEM-768 round-trip', () => {
  const mlkem = new MlKem768();
  const { publicKey, secretKey } = mlkem.generateKeypair();

  const { ciphertext, sharedSecret } = mlkem.encapsulate(publicKey);
  const decrypted = mlkem.decapsulate(ciphertext, secretKey);

  assert.deepEqual(sharedSecret, decrypted);
});
```

Run tests:
```bash
npm test
npm run test:integration
```

### Coverage Requirements

- **Minimum**: 80% code coverage
- **Target**: 90% code coverage
- **Critical paths**: 100% coverage (crypto operations, security)

```bash
# Generate coverage report
cargo tarpaulin --out Html --output-dir coverage/

# View report
open coverage/index.html
```

---

## Pull Request Process

### Before Submitting

1. **Update your branch**:
```bash
git fetch upstream
git rebase upstream/main
```

2. **Run all checks**:
```bash
# Format
cargo fmt --all
npm run format

# Lint
cargo clippy --all
npm run lint

# Test
cargo test --workspace
npm test

# Build
cargo build --release
npm run build
```

3. **Update documentation**:
- Add API documentation
- Update README if needed
- Add examples for new features

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests pass locally
- [ ] No new warnings
```

### Review Process

1. **Automated checks**: CI must pass
2. **Code review**: At least 1 approval required
3. **Testing**: Verify changes work as expected
4. **Documentation**: Check docs are updated
5. **Merge**: Squash and merge to main

---

## Release Process

### Versioning

We follow [Semantic Versioning](https://semver.org/):

- **Major** (1.0.0): Breaking changes
- **Minor** (0.1.0): New features, backward compatible
- **Patch** (0.0.1): Bug fixes

### Release Checklist

1. **Update version**:
```bash
# Cargo.toml
version = "0.2.0"

# package.json
"version": "0.2.0"
```

2. **Update changelog**:
```markdown
## [0.2.0] - 2025-11-15

### Added
- ML-KEM-1024 support
- Async vault operations

### Fixed
- Buffer size validation
- Memory leak in encapsulation
```

3. **Build and test**:
```bash
cargo build --release --workspace
npm run build
cargo test --workspace
npm test
```

4. **Tag release**:
```bash
git tag -a v0.2.0 -m "Release v0.2.0"
git push upstream main --tags
```

5. **Publish**:
```bash
# Publish to crates.io
cargo publish --package qudag-napi

# Publish to npm
cd qudag/qudag-napi
npm publish --access public
```

---

## Development Tips

### Debugging NAPI-rs

```bash
# Enable debug logging
export NAPI_RS_DEBUG=1
export RUST_BACKTRACE=1

node your-script.js
```

### Performance Profiling

```bash
# Rust profiling
cargo flamegraph --bin your-binary

# Node.js profiling
node --prof your-script.js
node --prof-process isolate-*.log > profile.txt
```

### Memory Debugging

```bash
# Valgrind (Linux)
valgrind --leak-check=full node your-script.js

# Node.js heap snapshot
node --inspect --heap-prof your-script.js
```

---

## Community

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: Questions and community support
- **Discord**: Real-time chat (coming soon)

### Getting Help

1. Check [documentation](https://github.com/ruvnet/daa/tree/main/docs)
2. Search [existing issues](https://github.com/ruvnet/daa/issues)
3. Ask in [discussions](https://github.com/ruvnet/daa/discussions)
4. Create a new issue

### Reporting Security Issues

**Do not** open public issues for security vulnerabilities.

Email: security@daa.dev

---

## Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- Credited in documentation

Thank you for contributing to DAA! 🚀

---

**Last Updated**: 2025-11-11
**Maintainers**: @ruvnet
