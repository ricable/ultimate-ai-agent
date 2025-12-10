# NAPI-rs Troubleshooting Guide

**Version**: 1.0.0
**Date**: 2025-11-11
**For**: @daa/qudag-native v0.1.0+

---

## Table of Contents

- [Common Issues](#common-issues)
- [Platform-Specific Problems](#platform-specific-problems)
- [Build Errors](#build-errors)
- [Runtime Errors](#runtime-errors)
- [Performance Issues](#performance-issues)
- [Integration Problems](#integration-problems)
- [Debugging Tools](#debugging-tools)

---

## Common Issues

### Module Not Found

**Symptom**:
```
Error: Cannot find module '@daa/qudag-native'
```

**Causes & Solutions**:

#### 1. Package not installed
```bash
npm install @daa/qudag-native
```

#### 2. Wrong package manager
```bash
# If using yarn
yarn add @daa/qudag-native

# If using pnpm
pnpm add @daa/qudag-native
```

#### 3. Workspace resolution issues
```json
// package.json
{
  "workspaces": ["packages/*"],
  "dependencies": {
    "@daa/qudag-native": "workspace:*"
  }
}
```

### Native Binding Failed to Load

**Symptom**:
```
Error: The specified module could not be found.
\\?\C:\...\qudag-native.win32-x64-msvc.node
```

**Solutions**:

#### 1. Check Node.js version
```bash
node --version
# Must be >= 18.0.0
```

#### 2. Rebuild native bindings
```bash
npm rebuild @daa/qudag-native

# or from source
cd node_modules/@daa/qudag-native
npm run build
```

#### 3. Install platform-specific package
```bash
# Detect platform
node -p "process.platform-process.arch"

# Install correct package
npm install @daa/qudag-native-linux-x64
# or
npm install @daa/qudag-native-darwin-arm64
```

#### 4. Check for conflicting native modules
```bash
# Remove node_modules and reinstall
rm -rf node_modules package-lock.json
npm install
```

### TypeScript Type Errors

**Symptom**:
```typescript
Property 'generateKeypair' does not exist on type 'MlKem768'
```

**Solutions**:

#### 1. Update TypeScript version
```bash
npm install --save-dev typescript@^5.0.0
```

#### 2. Regenerate type definitions
```bash
cd qudag/qudag-napi
npm run build
```

#### 3. Check tsconfig.json
```json
{
  "compilerOptions": {
    "moduleResolution": "node",
    "esModuleInterop": true,
    "allowSyntheticDefaultImports": true,
    "resolveJsonModule": true
  }
}
```

#### 4. Clear TypeScript cache
```bash
rm -rf node_modules/.cache
```

### Buffer vs Uint8Array Confusion

**Symptom**:
```
Type 'Uint8Array' is not assignable to type 'Buffer'
```

**Solution**:

Native bindings use `Buffer`, not `Uint8Array`:

```typescript
// Wrong
const data: Uint8Array = new Uint8Array([1, 2, 3]);
const hash = blake3Hash(data);  // Type error

// Correct
const data: Buffer = Buffer.from([1, 2, 3]);
const hash = blake3Hash(data);  // OK

// Or convert
const data: Uint8Array = new Uint8Array([1, 2, 3]);
const hash = blake3Hash(Buffer.from(data));  // OK
```

---

## Platform-Specific Problems

### Linux

#### Issue: GLIBC version mismatch

**Symptom**:
```
Error: /lib/x86_64-linux-gnu/libc.so.6: version `GLIBC_2.29' not found
```

**Solutions**:

1. Use musl build:
```bash
npm install @daa/qudag-native-linux-x64-musl
```

2. Update system:
```bash
# Ubuntu/Debian
sudo apt-get update && sudo apt-get upgrade

# CentOS/RHEL
sudo yum update
```

3. Build from source:
```bash
cd qudag/qudag-napi
cargo build --release
npm run artifacts
```

#### Issue: Permission denied

**Symptom**:
```
Error: EACCES: permission denied, open '/usr/local/lib/node_modules/@daa/qudag-native'
```

**Solution**:
```bash
# Use prefix for npm global installs
npm config set prefix ~/.npm-global
echo 'export PATH=~/.npm-global/bin:$PATH' >> ~/.bashrc
source ~/.bashrc

# Or use sudo (not recommended)
sudo npm install -g @daa/qudag-native
```

### macOS

#### Issue: Code signature invalid

**Symptom**:
```
Error: Code signature invalid
```

**Solutions**:

1. Re-sign the binary:
```bash
codesign --force --sign - node_modules/@daa/qudag-native/*.node
```

2. Allow unsigned binaries (development only):
```bash
sudo spctl --master-disable
```

3. Rebuild from source:
```bash
cd qudag/qudag-napi
npm run build
```

#### Issue: Architecture mismatch

**Symptom**:
```
Error: dlopen(...): tried: '.../qudag-native.darwin-x64.node'
(mach-o file, but is an incompatible architecture)
```

**Solutions**:

1. Install correct architecture:
```bash
# Apple Silicon (M1/M2)
npm install @daa/qudag-native-darwin-arm64

# Intel
npm install @daa/qudag-native-darwin-x64
```

2. Use Rosetta (not recommended):
```bash
arch -x86_64 npm install
```

#### Issue: Xcode command line tools missing

**Symptom**:
```
Error: Cannot find module 'node-gyp'
```

**Solution**:
```bash
xcode-select --install
```

### Windows

#### Issue: Visual Studio Build Tools not found

**Symptom**:
```
Error: Could not find any Visual Studio installation to use
```

**Solution**:

1. Install Visual Studio Build Tools:
   - Download from: https://visualstudio.microsoft.com/downloads/
   - Select "Desktop development with C++"

2. Or use windows-build-tools:
```powershell
npm install --global windows-build-tools
```

#### Issue: Python not found

**Symptom**:
```
Error: Can't find Python executable "python"
```

**Solution**:
```powershell
# Install Python 3
choco install python

# Or specify Python path
npm config set python "C:\Python39\python.exe"
```

#### Issue: Long path issues

**Symptom**:
```
Error: ENAMETOOLONG: name too long
```

**Solution**:

1. Enable long paths:
```powershell
# Run as Administrator
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" `
  -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

2. Use shorter path:
```powershell
cd C:\dev
git clone <repo>
```

#### Issue: DLL not found

**Symptom**:
```
Error: The specified module could not be found.
```

**Solution**:

1. Install Visual C++ Redistributable:
   - Download from: https://aka.ms/vs/17/release/vc_redist.x64.exe

2. Check PATH:
```powershell
echo $env:PATH
# Should include Windows system32
```

---

## Build Errors

### Cargo Build Failed

**Symptom**:
```
error: could not compile `qudag-napi`
```

**Solutions**:

#### 1. Update Rust toolchain
```bash
rustup update stable
rustup default stable
```

#### 2. Check Rust version
```bash
rustc --version
# Should be 1.75.0 or newer
```

#### 3. Clean and rebuild
```bash
cargo clean
cargo build --release
```

#### 4. Check dependencies
```bash
cargo tree
# Look for version conflicts
```

### NAPI-rs CLI Errors

**Symptom**:
```
Error: Failed to parse napi options
```

**Solutions**:

#### 1. Update NAPI-rs CLI
```bash
npm install --save-dev @napi-rs/cli@latest
```

#### 2. Check package.json configuration
```json
{
  "napi": {
    "name": "qudag-native",
    "triples": {
      "defaults": true
    }
  }
}
```

#### 3. Regenerate artifacts
```bash
npm run artifacts
```

### Cross-Compilation Failures

**Symptom**:
```
Error: Failed to build for target aarch64-unknown-linux-gnu
```

**Solutions**:

#### 1. Install target
```bash
rustup target add aarch64-unknown-linux-gnu
```

#### 2. Install cross-compilation tools (Linux)
```bash
sudo apt-get install gcc-aarch64-linux-gnu
```

#### 3. Use GitHub Actions for cross-platform builds
```yaml
- uses: actions-rs/toolchain@v1
  with:
    toolchain: stable
    target: ${{ matrix.target }}
```

---

## Runtime Errors

### Invalid Buffer Length

**Symptom**:
```
Error: Invalid public key length: expected 1184 bytes, got 100
```

**Cause**: Using incorrect key sizes for ML-KEM-768

**Solution**:

```typescript
// Check sizes before passing to crypto functions
function validateMLKEMPublicKey(key: Buffer): void {
  if (key.length !== 1184) {
    throw new Error(`Invalid ML-KEM-768 public key: ${key.length} bytes`);
  }
}

function validateMLKEMSecretKey(key: Buffer): void {
  if (key.length !== 2400) {
    throw new Error(`Invalid ML-KEM-768 secret key: ${key.length} bytes`);
  }
}

// Use validation
validateMLKEMPublicKey(publicKey);
const result = mlkem.encapsulate(publicKey);
```

### Signature Verification Failed

**Symptom**:
```typescript
const isValid = mldsa.verify(message, signature, publicKey);
console.log(isValid);  // false
```

**Common causes**:

1. **Wrong public key**:
```typescript
// Make sure public key matches the secret key used for signing
const correctPublicKey = signerKeypair.publicKey;
```

2. **Message modified**:
```typescript
// Message must be EXACTLY the same
const message = Buffer.from('exact message', 'utf8');
const signature = mldsa.sign(message, secretKey);

// Later...
const sameMessage = Buffer.from('exact message', 'utf8');
const isValid = mldsa.verify(sameMessage, signature, publicKey);
```

3. **Encoding issues**:
```typescript
// Wrong - encoding mismatch
const message = 'Hello';
const signature = mldsa.sign(Buffer.from(message, 'utf8'), secretKey);
const isValid = mldsa.verify(Buffer.from(message, 'ascii'), signature, publicKey);

// Correct - same encoding
const message = Buffer.from('Hello', 'utf8');
const signature = mldsa.sign(message, secretKey);
const isValid = mldsa.verify(message, signature, publicKey);
```

### Memory Leaks

**Symptom**: Memory usage grows over time

**Solutions**:

#### 1. Don't hold references unnecessarily
```typescript
// Bad - holds all keypairs in memory
const keypairs = [];
for (let i = 0; i < 1000000; i++) {
  keypairs.push(mlkem.generateKeypair());
}

// Good - only store what you need
for (let i = 0; i < 1000000; i++) {
  const { publicKey } = mlkem.generateKeypair();
  await savePublicKey(i, publicKey);
  // keypair garbage collected after each iteration
}
```

#### 2. Use streams for large data
```typescript
import { createReadStream } from 'fs';
import { blake3Hash } from '@daa/qudag-native';

async function hashLargeFile(path: string) {
  const stream = createReadStream(path, { highWaterMark: 1024 * 1024 });
  const chunks: Buffer[] = [];

  for await (const chunk of stream) {
    chunks.push(blake3Hash(chunk));
  }

  return Buffer.concat(chunks);
}
```

#### 3. Monitor memory usage
```typescript
console.log('Memory usage:', process.memoryUsage());

setInterval(() => {
  if (global.gc) global.gc();
  console.log('After GC:', process.memoryUsage());
}, 10000);
```

---

## Performance Issues

### Not Meeting Expected Performance

**Symptom**: Native bindings not 2-5x faster than WASM

**Diagnosis**:

```typescript
import { performance } from 'perf_hooks';
import { MlKem768 } from '@daa/qudag-native';

const mlkem = new MlKem768();

// Warmup
for (let i = 0; i < 100; i++) {
  mlkem.generateKeypair();
}

// Benchmark
const start = performance.now();
for (let i = 0; i < 1000; i++) {
  mlkem.generateKeypair();
}
const end = performance.now();

const avgTime = (end - start) / 1000;
console.log(`Average time: ${avgTime.toFixed(2)}ms`);
// Should be ~1.8ms for ML-KEM-768 key generation
```

**Solutions**:

#### 1. Verify using native, not WASM
```typescript
import { version } from '@daa/qudag-native';
console.log(version());  // Should print version number
```

#### 2. Check build configuration
```bash
# Make sure release mode is used
cd qudag/qudag-napi
npm run build --release

# Verify optimization flags in Cargo.toml
grep -A5 "\[profile.release\]" Cargo.toml
```

#### 3. Remove unnecessary conversions
```typescript
// Slow - unnecessary conversions
const data = Buffer.from(new Uint8Array([1, 2, 3]));
const hash = blake3Hash(data);
const hex = Buffer.from(hash).toString('hex');

// Fast - direct operations
const data = Buffer.from([1, 2, 3]);
const hash = blake3Hash(data);
const hex = hash.toString('hex');
```

#### 4. Profile with Node.js profiler
```bash
node --prof app.js
node --prof-process isolate-*.log > profile.txt
```

### High CPU Usage

**Symptom**: 100% CPU usage during crypto operations

**Explanation**: This is normal for cryptographic operations. They are CPU-intensive by design.

**Solutions**:

#### 1. Use Worker threads for parallelization
```typescript
import { Worker, isMainThread, parentPort, workerData } from 'worker_threads';
import { MlKem768 } from '@daa/qudag-native';

if (isMainThread) {
  // Main thread - distribute work
  const numWorkers = require('os').cpus().length;
  const workers = Array.from({ length: numWorkers }, (_, i) =>
    new Worker(__filename, { workerData: { id: i } })
  );

  workers.forEach(worker => {
    worker.on('message', (result) => {
      console.log('Worker completed:', result);
    });
  });
} else {
  // Worker thread - perform crypto
  const mlkem = new MlKem768();
  const result = mlkem.generateKeypair();
  parentPort!.postMessage({ id: workerData.id, result });
}
```

#### 2. Implement rate limiting
```typescript
import pLimit from 'p-limit';

const limit = pLimit(4);  // Max 4 concurrent crypto operations

const tasks = publicKeys.map(pk =>
  limit(() => mlkem.encapsulate(pk))
);

const results = await Promise.all(tasks);
```

#### 3. Add delays for background tasks
```typescript
async function processQueuedItems() {
  for (const item of queue) {
    await processCryptoOperation(item);
    await new Promise(resolve => setImmediate(resolve));  // Yield to event loop
  }
}
```

---

## Integration Problems

### Webpack Issues

**Symptom**:
```
Module not found: Can't resolve '@daa/qudag-native'
```

**Solution**:

```javascript
// webpack.config.js
module.exports = {
  target: 'node',  // Important for native modules
  externals: {
    '@daa/qudag-native': 'commonjs @daa/qudag-native'
  }
};
```

### Electron Issues

**Symptom**: Native module crashes in Electron

**Solutions**:

#### 1. Rebuild for Electron
```bash
npm install --save-dev electron-rebuild
npx electron-rebuild
```

#### 2. Use correct Node version
```json
// package.json
{
  "electronRebuildConfig": {
    "onlyModules": ["@daa/qudag-native"]
  }
}
```

#### 3. Handle renderer/main process
```javascript
// main.js (Node.js context - OK to use native)
const { MlKem768 } = require('@daa/qudag-native');

// renderer.js (browser context - use WASM instead)
import { MlKem768 } from 'qudag-wasm';
```

### Docker Issues

**Symptom**: Native module doesn't work in Docker container

**Solutions**:

#### 1. Use correct base image
```dockerfile
# Dockerfile
FROM node:20-alpine

# Install build dependencies
RUN apk add --no-cache python3 make g++

# Install dependencies
COPY package*.json ./
RUN npm ci --only=production

# Copy app
COPY . .
```

#### 2. Multi-stage build for smaller images
```dockerfile
# Build stage
FROM node:20 AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

# Production stage
FROM node:20-alpine
WORKDIR /app
COPY --from=builder /app/dist ./dist
COPY --from=builder /app/node_modules ./node_modules
CMD ["node", "dist/index.js"]
```

#### 3. Use platform-specific image
```dockerfile
FROM node:20-bullseye  # Better compatibility than alpine
```

### TypeScript + ESM Issues

**Symptom**:
```
TypeError: MlKem768 is not a constructor
```

**Solution**:

```json
// tsconfig.json
{
  "compilerOptions": {
    "module": "ESNext",
    "moduleResolution": "node",
    "esModuleInterop": true,
    "allowSyntheticDefaultImports": true
  }
}

// package.json
{
  "type": "module"
}
```

```typescript
// Correct import
import { MlKem768 } from '@daa/qudag-native';

// NOT this
import * as qudag from '@daa/qudag-native';
```

---

## Debugging Tools

### Enable Debug Logging

```bash
# Enable NAPI-rs debug output
export NAPI_RS_DEBUG=1
node app.js

# Enable Rust backtrace
export RUST_BACKTRACE=1
node app.js

# Enable full backtrace
export RUST_BACKTRACE=full
node app.js
```

### Memory Profiling

```bash
# Node.js heap snapshot
node --inspect --heap-prof app.js

# Open Chrome DevTools
# chrome://inspect
```

### Performance Profiling

```bash
# Generate CPU profile
node --prof app.js

# Analyze profile
node --prof-process isolate-*.log > profile.txt
```

### Check Native Module Info

```typescript
import { getModuleInfo } from '@daa/qudag-native';

console.log(getModuleInfo());
// {
//   name: "qudag-native",
//   version: "0.1.0",
//   features: ["ML-KEM-768", "ML-DSA", "BLAKE3", ...]
// }
```

### Verify Installation

```bash
# Check if native module loaded correctly
node -e "console.log(require('@daa/qudag-native').version())"

# Check platform compatibility
node -p "process.platform + '-' + process.arch"

# List installed packages
npm ls @daa/qudag-native
```

---

## Getting Help

If you're still experiencing issues:

1. **Check existing issues**: [https://github.com/ruvnet/daa/issues](https://github.com/ruvnet/daa/issues)

2. **Create detailed bug report**:
   - Node.js version (`node --version`)
   - OS and architecture (`node -p "process.platform + '-' + process.arch"`)
   - Package versions (`npm ls`)
   - Error message and stack trace
   - Minimal reproduction code

3. **Join community discussions**: [https://github.com/ruvnet/daa/discussions](https://github.com/ruvnet/daa/discussions)

---

**Troubleshooting Guide Version**: 1.0.0
**Last Updated**: 2025-11-11
**Covers**: @daa/qudag-native v0.1.0+
