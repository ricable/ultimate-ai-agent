# Basic DAA Agent Template

A simple DAA agent demonstrating quantum-resistant cryptography operations.

## Features

- **ML-KEM-768**: Quantum-resistant key encapsulation mechanism
- **ML-DSA**: Post-quantum digital signature algorithm
- **BLAKE3**: High-performance cryptographic hashing
- **Quantum Fingerprinting**: Advanced data integrity verification
- **Password Vault**: Secure credential storage

## Quick Start

### Installation

```bash
npm install
```

### Build

```bash
npm run build
```

### Run

```bash
npm start
```

### Development

```bash
npm run dev
```

## Project Structure

```
basic/
├── src/
│   └── index.ts          # Main agent implementation
├── package.json          # Dependencies and scripts
├── tsconfig.json         # TypeScript configuration
└── README.md            # This file
```

## Usage Examples

### ML-KEM Key Encapsulation

```typescript
import { DAA } from 'daa-sdk';

const daa = new DAA();
await daa.init();

const mlkem = daa.crypto.mlkem();
const keypair = mlkem.generateKeypair();
const { ciphertext, sharedSecret } = mlkem.encapsulate(keypair.publicKey);
```

### Digital Signatures

```typescript
const mldsa = daa.crypto.mldsa();
const keypair = mldsa.generateKeypair();

const message = new TextEncoder().encode('Hello!');
const signature = mldsa.sign(keypair.secretKey, message);
const isValid = mldsa.verify(keypair.publicKey, message, signature);
```

### Password Vault

```typescript
const vault = daa.vault.create('master-password');
vault.store('service', 'username', 'password');

const credentials = vault.get('service');
console.log(credentials.username, credentials.password);
```

## Performance

- **Native Bindings** (Node.js): Optimal performance using NAPI-rs
- **WASM Fallback** (Browsers): ~40% of native speed, full compatibility

Check your runtime:

```bash
npx daa-sdk info
```

## Security Features

1. **Post-Quantum Cryptography**: NIST-standardized algorithms
2. **Memory Safety**: Rust-based implementation
3. **Constant-Time Operations**: Resistant to timing attacks
4. **Zero-Copy Optimization**: Minimal memory overhead

## Next Steps

1. Explore the [Full-Stack Template](../full-stack) for orchestration features
2. Check the [ML Training Template](../ml-training) for federated learning
3. Read the [DAA SDK Documentation](https://github.com/ruvnet/daa)

## License

MIT
