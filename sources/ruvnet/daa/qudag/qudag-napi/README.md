# @daa/qudag-native

[![NAPI Build](https://github.com/ruvnet/daa/actions/workflows/napi-build.yml/badge.svg)](https://github.com/ruvnet/daa/actions/workflows/napi-build.yml)
[![NAPI Test](https://github.com/ruvnet/daa/actions/workflows/napi-test.yml/badge.svg)](https://github.com/ruvnet/daa/actions/workflows/napi-test.yml)
[![npm version](https://img.shields.io/npm/v/@daa/qudag-native.svg)](https://www.npmjs.com/package/@daa/qudag-native)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Node.js Version](https://img.shields.io/node/v/@daa/qudag-native.svg)](https://nodejs.org)

**Native Node.js bindings for QuDAG quantum-resistant cryptography**

High-performance, quantum-resistant cryptographic operations for Node.js applications using NAPI-rs bindings to the Rust-based QuDAG library.

## Features

- **🔐 Post-Quantum Cryptography**: ML-DSA (Dilithium) signatures, ML-KEM (Kyber) key exchange
- **⚡ Native Performance**: Rust implementation with zero-overhead NAPI bindings
- **🌍 Cross-Platform**: Linux (x64, ARM64), macOS (Intel, Apple Silicon), Windows (x64)
- **🛡️ Secure by Default**: Quantum-resistant algorithms approved by NIST
- **🔒 Static Linking**: MUSL-based builds for maximum portability
- **📦 Easy Integration**: Drop-in replacement for traditional crypto operations

## Installation

```bash
npm install @daa/qudag-native
```

### Requirements

- **Node.js**: 18.x, 20.x, or 22.x
- **Platform**: Linux (x64/ARM64), macOS (Intel/Apple Silicon), Windows (x64)

## Quick Start

```javascript
const qudag = require('@daa/qudag-native');

// Generate ML-DSA (Dilithium) keypair for quantum-resistant signatures
const { publicKey, privateKey } = qudag.generateKeypair();

// Sign a message
const message = Buffer.from('Hello, quantum-resistant world!');
const signature = qudag.sign(message, privateKey);

// Verify signature
const isValid = qudag.verify(message, signature, publicKey);
console.log('Signature valid:', isValid); // true

// ML-KEM (Kyber) key encapsulation
const kemKeys = qudag.generateKemKeys();
const { ciphertext, sharedSecret } = qudag.encapsulate(kemKeys.publicKey);
const decapsulatedSecret = qudag.decapsulate(ciphertext, kemKeys.privateKey);

// BLAKE3 hashing
const hash = qudag.blake3Hash(message);
console.log('BLAKE3 hash:', hash.toString('hex'));
```

## API Reference

### Digital Signatures (ML-DSA)

#### `generateKeypair()`

Generate a new ML-DSA-65 keypair for quantum-resistant digital signatures.

```javascript
const { publicKey, privateKey } = qudag.generateKeypair();
// Returns: { publicKey: Buffer, privateKey: Buffer }
```

#### `sign(message, privateKey)`

Sign a message using ML-DSA (Dilithium).

```javascript
const signature = qudag.sign(Buffer.from('message'), privateKey);
// Returns: Buffer (signature)
```

#### `verify(message, signature, publicKey)`

Verify an ML-DSA signature.

```javascript
const isValid = qudag.verify(message, signature, publicKey);
// Returns: boolean
```

### Key Encapsulation (ML-KEM)

#### `generateKemKeys()`

Generate ML-KEM-768 keypair for key encapsulation.

```javascript
const { publicKey, privateKey } = qudag.generateKemKeys();
// Returns: { publicKey: Buffer, privateKey: Buffer }
```

#### `encapsulate(publicKey)`

Encapsulate a shared secret using the recipient's public key.

```javascript
const { ciphertext, sharedSecret } = qudag.encapsulate(publicKey);
// Returns: { ciphertext: Buffer, sharedSecret: Buffer }
```

#### `decapsulate(ciphertext, privateKey)`

Decapsulate the shared secret using the private key.

```javascript
const sharedSecret = qudag.decapsulate(ciphertext, privateKey);
// Returns: Buffer (shared secret)
```

### Hashing (BLAKE3)

#### `blake3Hash(data)`

Compute BLAKE3 hash of data.

```javascript
const hash = qudag.blake3Hash(Buffer.from('data'));
// Returns: Buffer (32-byte hash)
```

#### `blake3Keyed(data, key)`

Compute keyed BLAKE3 hash (MAC).

```javascript
const mac = qudag.blake3Keyed(data, key); // key must be 32 bytes
// Returns: Buffer (32-byte MAC)
```

## Platform Support

| Platform | Architecture | Node.js Versions | Status |
|----------|-------------|------------------|--------|
| Linux | x86_64 (glibc) | 18, 20, 22 | ✅ Fully Supported |
| Linux | x86_64 (musl) | 18, 20, 22 | ✅ Fully Supported |
| Linux | ARM64 (glibc) | 18, 20, 22 | ✅ Fully Supported |
| Linux | ARM64 (musl) | 18, 20, 22 | ✅ Fully Supported |
| macOS | x86_64 (Intel) | 18, 20, 22 | ✅ Fully Supported |
| macOS | ARM64 (Apple Silicon) | 18, 20, 22 | ✅ Fully Supported |
| Windows | x86_64 | 18, 20, 22 | ✅ Fully Supported |

### Static Linking (MUSL)

MUSL-based Linux builds provide static linking for maximum portability:

- No dependency on system GLIBC version
- Works on Alpine Linux and minimal containers
- Ideal for Docker deployments

## Performance

QuDAG NAPI bindings provide native Rust performance:

```
Benchmark results (Apple M1 Max):
  ML-DSA Sign:           12,500 ops/sec
  ML-DSA Verify:          8,300 ops/sec
  ML-KEM Encapsulate:    15,000 ops/sec
  ML-KEM Decapsulate:    16,500 ops/sec
  BLAKE3 Hash (1KB):    950,000 ops/sec
```

Run benchmarks:

```bash
npm run benchmark
```

## Development

### Building from Source

```bash
# Install dependencies
npm install

# Build debug version
npm run build:debug

# Build release version
npm run build

# Run tests
npm test

# Run benchmarks
npm run benchmark
```

### Cross-Platform Builds

Use the provided build script for local multi-platform testing:

```bash
# Build for current platform
./scripts/build-all.sh

# Build for specific target
./scripts/build-all.sh --target linux-x64 --release

# Build for all platforms
./scripts/build-all.sh --target all --release --test

# Clean and rebuild
./scripts/build-all.sh --clean --release
```

### Requirements

- **Rust**: 1.75.0 or later
- **Node.js**: 18.x, 20.x, or 22.x
- **NAPI-rs CLI**: Installed via `npm install`

#### Platform-Specific Tools

**Linux Cross-Compilation:**
```bash
# ARM64 cross-compilation
sudo apt-get install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu

# MUSL static linking
sudo apt-get install musl-tools
```

**macOS Cross-Compilation:**
```bash
# Install Xcode Command Line Tools
xcode-select --install

# Both x86_64 and ARM64 supported natively on Apple Silicon
```

**Windows:**
```powershell
# Install Visual Studio Build Tools
# or Visual Studio with C++ development tools
```

## Security

### Quantum Resistance

QuDAG uses NIST-approved post-quantum cryptographic algorithms:

- **ML-DSA-65** (Dilithium): Quantum-resistant digital signatures
- **ML-KEM-768** (Kyber): Quantum-resistant key encapsulation
- **BLAKE3**: Fast, secure cryptographic hashing

These algorithms are designed to resist attacks by quantum computers and are suitable for long-term security.

### Security Auditing

```bash
# Run Rust security audit
cargo audit

# Run npm security audit
npm audit
```

### Reporting Security Issues

Please report security vulnerabilities to: security@ruv.io

## CI/CD

### GitHub Actions Workflows

- **NAPI Build** (`.github/workflows/napi-build.yml`): Multi-platform matrix builds
- **NAPI Test** (`.github/workflows/napi-test.yml`): Comprehensive testing on PRs
- **NAPI Publish** (`.github/workflows/napi-publish.yml`): Automated npm publishing

### Releasing

1. Update version in `package.json` and `Cargo.toml`
2. Create a git tag: `git tag qudag-napi-v0.1.0`
3. Push the tag: `git push origin qudag-napi-v0.1.0`
4. GitHub Actions will automatically build and publish to npm

## Examples

### Hybrid Encryption

Combine ML-KEM and symmetric encryption:

```javascript
const crypto = require('crypto');
const qudag = require('@daa/qudag-native');

// Sender side
const recipientKemKeys = qudag.generateKemKeys();
const { ciphertext, sharedSecret } = qudag.encapsulate(recipientKemKeys.publicKey);

// Use shared secret as symmetric key
const cipher = crypto.createCipheriv('aes-256-gcm', sharedSecret, nonce);
const encrypted = cipher.update(plaintext);
// Send: ciphertext + encrypted data

// Recipient side
const recoveredSecret = qudag.decapsulate(ciphertext, recipientKemKeys.privateKey);
const decipher = crypto.createDecipheriv('aes-256-gcm', recoveredSecret, nonce);
const decrypted = decipher.update(encrypted);
```

### Digital Certificate Chain

```javascript
const qudag = require('@daa/qudag-native');

// Root CA
const rootKeys = qudag.generateKeypair();

// Sign intermediate certificate
const intermediateCert = Buffer.from(JSON.stringify({
  subject: 'Intermediate CA',
  publicKey: intermediateKeys.publicKey.toString('hex'),
  issuer: 'Root CA'
}));
const intermediateSig = qudag.sign(intermediateCert, rootKeys.privateKey);

// Verify chain
const isValid = qudag.verify(intermediateCert, intermediateSig, rootKeys.publicKey);
```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](../../CONTRIBUTING.md) for guidelines.

### Development Workflow

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Run `npm test` and `cargo test`
5. Submit a pull request

## License

MIT License - see [LICENSE](../../LICENSE) for details.

## Acknowledgments

- Built with [NAPI-rs](https://napi.rs/) for seamless Rust-Node.js integration
- Post-quantum cryptography from NIST PQC standardization
- Part of the [QuDAG](https://github.com/ruvnet/daa/tree/main/qudag) project
- Developed by [rUv](https://github.com/ruvnet)

## Links

- **npm Package**: https://www.npmjs.com/package/@daa/qudag-native
- **Documentation**: https://docs.rs/qudag-napi
- **QuDAG Project**: https://github.com/ruvnet/daa/tree/main/qudag
- **Issue Tracker**: https://github.com/ruvnet/daa/issues
- **CI/CD Pipelines**: https://github.com/ruvnet/daa/actions

---

**Built with ❤️ for quantum-resistant security**
