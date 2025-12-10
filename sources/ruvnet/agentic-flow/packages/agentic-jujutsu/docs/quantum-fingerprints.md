# Quantum Fingerprints for JJ Operations

## Overview

Agentic-Jujutsu integrates quantum-resistant fingerprints using [@qudag/napi-core](https://github.com/ruvnet/QuDAG) for fast integrity verification of operations. This provides post-quantum security for operation validation and tamper detection.

## Features

- **Quantum-Resistant**: Uses NIST-approved ML-DSA (CRYSTALS-Dilithium) signatures
- **Fast Verification**: BLAKE3 hashing for millisecond-level fingerprint generation
- **Tamper Detection**: Cryptographic proof of operation integrity
- **Zero-Copy**: Efficient N-API integration with minimal overhead

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    JJ Operation                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Operation Data (JSON)                                │   │
│  │  - id, command, user, timestamp, etc.               │   │
│  └──────────────────┬──────────────────────────────────┘   │
│                     │                                        │
│                     ▼                                        │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ @qudag/napi-core QuantumFingerprint                 │   │
│  │  ┌──────────────┐    ┌────────────────────┐       │   │
│  │  │ BLAKE3 Hash  │───▶│ ML-DSA-65 Signature│       │   │
│  │  │ (64 bytes)   │    │ (~3309 bytes)      │       │   │
│  │  └──────────────┘    └────────────────────┘       │   │
│  └─────────────────────────────────────────────────────┘   │
│                     │                                        │
│                     ▼                                        │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Stored Quantum Fingerprint (hex string)             │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Usage

### Basic Fingerprint Generation

```javascript
const { JjWrapper } = require('agentic-jujutsu');

const wrapper = new JjWrapper();

// Execute an operation
await wrapper.execute(['describe', '-m', 'Add new feature']);

// Get the operation
const operations = wrapper.getOperations(1);
const operation = operations[0];

// Generate quantum fingerprint
const fingerprint = await wrapper.generateOperationFingerprint(operation.id);
console.log('Quantum fingerprint:', fingerprint);

// Verify fingerprint
const isValid = await wrapper.verifyOperationFingerprint(operation.id, fingerprint);
console.log('Fingerprint valid:', isValid);
```

### Using @qudag/napi-core Directly

For maximum control, use @qudag/napi-core directly:

```javascript
const { JjWrapper } = require('agentic-jujutsu');
const qudag = require('@qudag/napi-core');

const wrapper = new JjWrapper();
const operations = wrapper.getOperations(1);
const operation = operations[0];

// Get operation data
const operationData = wrapper.getOperationData(operation.id);

// Generate quantum fingerprint with full control
const quantumFingerprint = qudag.QuantumFingerprint.generate(operationData);

// Access components
const fingerprintHex = quantumFingerprint.asHex();
const signature = quantumFingerprint.getSignature();
const publicKey = quantumFingerprint.getPublicKey();

console.log('Fingerprint (BLAKE3):', fingerprintHex);
console.log('Signature size:', signature.length, 'bytes');
console.log('Public key size:', publicKey.length, 'bytes');

// Verify
const isValid = quantumFingerprint.verify();
console.log('Signature valid:', isValid);

// Store in operation
wrapper.setOperationFingerprint(operation.id, fingerprintHex);
```

### Convenience Functions

```javascript
const qudag = require('@qudag/napi-core');

// Quick generation
const data = Buffer.from('Important operation data');
const fingerprintBytes = qudag.generateQuantumFingerprint(data);

// Quick verification
const isValid = qudag.verifyQuantumFingerprint(data, fingerprintBytes);
```

## API Reference

### JJWrapper Methods

#### `generateOperationFingerprint(operationId: string): Promise<string>`

Generates a quantum fingerprint for an operation.

**Parameters:**
- `operationId`: The unique ID of the operation

**Returns:**
- Quantum fingerprint as a hex string

**Example:**
```javascript
const fingerprint = await wrapper.generateOperationFingerprint(opId);
```

#### `verifyOperationFingerprint(operationId: string, fingerprint: string): Promise<boolean>`

Verifies an operation's quantum fingerprint.

**Parameters:**
- `operationId`: The operation to verify
- `fingerprint`: Expected fingerprint (hex string)

**Returns:**
- `true` if fingerprint is valid, `false` otherwise

**Example:**
```javascript
const isValid = await wrapper.verifyOperationFingerprint(opId, fingerprint);
```

#### `getOperationData(operationId: string): Buffer`

Gets the raw operation data for fingerprinting.

**Parameters:**
- `operationId`: The operation ID

**Returns:**
- Operation data as a Buffer

**Example:**
```javascript
const data = wrapper.getOperationData(opId);
const fp = qudag.generateQuantumFingerprint(data);
```

#### `setOperationFingerprint(operationId: string, fingerprint: string): void`

Stores a quantum fingerprint with an operation.

**Parameters:**
- `operationId`: The operation ID
- `fingerprint`: Fingerprint to store (hex string)

**Example:**
```javascript
wrapper.setOperationFingerprint(opId, fingerprintHex);
```

### @qudag/napi-core Integration

#### `QuantumFingerprint.generate(data: Buffer): QuantumFingerprint`

Generates a quantum-resistant fingerprint.

**Features:**
- BLAKE3 hashing (64 bytes)
- ML-DSA-65 signature (~3309 bytes)
- Security level 3 (AES-192 equivalent)

**Example:**
```javascript
const qf = qudag.QuantumFingerprint.generate(operationData);
```

#### `quantumFingerprint.asHex(): string`

Returns fingerprint as hex string (128 characters).

#### `quantumFingerprint.asBytes(): Uint8Array`

Returns raw fingerprint bytes (64 bytes).

#### `quantumFingerprint.getSignature(): Uint8Array`

Returns ML-DSA signature (~3309 bytes).

#### `quantumFingerprint.getPublicKey(): Uint8Array`

Returns ML-DSA public key (1952 bytes).

#### `quantumFingerprint.verify(): boolean`

Verifies the ML-DSA signature.

## Performance

Quantum fingerprints are designed for high performance:

| Operation | Time | Throughput |
|-----------|------|------------|
| Fingerprint Generation | <1ms | >1000 ops/sec |
| Signature Verification | <1ms | >1000 ops/sec |
| Data Hashing (BLAKE3) | <0.1ms | >10000 ops/sec |

Benchmarks on typical hardware:
- **Generation**: 1000 fingerprints in ~300ms (0.3ms each)
- **Verification**: 1000 verifications in ~200ms (0.2ms each)

## Security Properties

### Quantum Resistance

Uses **ML-DSA-65** (CRYSTALS-Dilithium):
- NIST-approved post-quantum signature scheme
- Security level 3 (192-bit classical security)
- Resistant to Shor's algorithm and Grover's algorithm
- Based on Module-LWE hardness

### Fingerprint Properties

- **Collision Resistance**: BLAKE3 provides 256-bit security
- **Pre-image Resistance**: Computationally infeasible to find data for a given fingerprint
- **Tamper Detection**: Any modification to operation data invalidates the fingerprint
- **Signature Uniqueness**: ML-DSA uses randomness, so each signature is unique

## Use Cases

### 1. Operation Integrity Verification

Verify that operations haven't been tampered with:

```javascript
// After loading operations from disk
for (const op of operations) {
  if (op.quantumFingerprint) {
    const isValid = await wrapper.verifyOperationFingerprint(
      op.id,
      op.quantumFingerprint
    );
    if (!isValid) {
      console.error('Operation integrity check failed!', op.id);
    }
  }
}
```

### 2. Audit Trail

Create an immutable audit trail:

```javascript
// Generate fingerprints for all operations
for (const op of operations) {
  const fp = await wrapper.generateOperationFingerprint(op.id);
  auditLog.record({
    operationId: op.id,
    fingerprint: fp,
    timestamp: Date.now()
  });
}
```

### 3. Multi-Agent Coordination

Verify operations between agents:

```javascript
// Agent A generates fingerprint
const fpA = await wrapperA.generateOperationFingerprint(opId);

// Send to Agent B
await sendToAgentB({ operationId: opId, fingerprint: fpA });

// Agent B verifies
const isValid = await wrapperB.verifyOperationFingerprint(opId, fpA);
if (!isValid) {
  console.error('Operation verification failed from Agent A');
}
```

### 4. Operation Signing

Sign operations with ML-DSA for non-repudiation:

```javascript
const operationData = wrapper.getOperationData(opId);
const qf = qudag.QuantumFingerprint.generate(operationData);

// Store signature with operation
const operation = operations.find(o => o.id === opId);
operation.signature = Buffer.from(qf.getSignature()).toString('hex');
operation.signaturePublicKey = Buffer.from(qf.getPublicKey()).toString('hex');

// Later, verify the signature
const publicKey = qudag.MlDsaPublicKey.fromHex(operation.signaturePublicKey);
const isValid = publicKey.verify(
  operationData,
  Buffer.from(operation.signature, 'hex')
);
```

## Best Practices

### 1. Always Verify Before Trust

```javascript
// ❌ Bad: Trust operation data without verification
const operation = loadOperationFromDisk(opId);
processOperation(operation);

// ✅ Good: Verify first
const operation = loadOperationFromDisk(opId);
if (operation.quantumFingerprint) {
  const isValid = await wrapper.verifyOperationFingerprint(
    operation.id,
    operation.quantumFingerprint
  );
  if (!isValid) {
    throw new Error('Operation integrity check failed');
  }
}
processOperation(operation);
```

### 2. Store Fingerprints Immediately

```javascript
// Generate and store fingerprint as soon as operation is created
const result = await wrapper.execute(['describe', '-m', 'message']);
const operations = wrapper.getOperations(1);
const operation = operations[0];

// Immediately generate and store fingerprint
const fingerprint = await wrapper.generateOperationFingerprint(operation.id);
wrapper.setOperationFingerprint(operation.id, fingerprint);

// Save to persistent storage
await saveOperation(operation);
```

### 3. Use Batch Processing for Performance

```javascript
// For multiple operations, batch the fingerprint generation
const operations = wrapper.getOperations(100);
const fingerprints = await Promise.all(
  operations.map(op => wrapper.generateOperationFingerprint(op.id))
);

// Store all fingerprints
operations.forEach((op, i) => {
  wrapper.setOperationFingerprint(op.id, fingerprints[i]);
});
```

### 4. Implement Tamper Detection

```javascript
// Set up periodic integrity checks
setInterval(async () => {
  const operations = wrapper.getOperations(100);
  for (const op of operations) {
    if (op.quantumFingerprint) {
      const isValid = await wrapper.verifyOperationFingerprint(
        op.id,
        op.quantumFingerprint
      );
      if (!isValid) {
        console.error('SECURITY ALERT: Operation tampered!', {
          operationId: op.id,
          command: op.command,
          timestamp: op.timestamp
        });
        // Take action: notify, rollback, etc.
      }
    }
  }
}, 60000); // Check every minute
```

## Comparison with Traditional Hashes

| Feature | SHA-256 | BLAKE3 | Quantum Fingerprint |
|---------|---------|--------|---------------------|
| Hash Speed | Fast | Very Fast | Very Fast |
| Quantum Resistant | ❌ No | ❌ No | ✅ Yes (ML-DSA) |
| Signature | ❌ No | ❌ No | ✅ Yes (~3309 bytes) |
| Security Level | 128-bit | 128-bit | 192-bit (NIST Level 3) |
| Future-Proof | Limited | Limited | ✅ Post-Quantum |

## Troubleshooting

### Fingerprint Verification Fails

**Cause**: Operation data has been modified.

**Solution**: Regenerate the fingerprint or restore from backup.

```javascript
const operationData = wrapper.getOperationData(opId);
const newFingerprint = qudag.generateQuantumFingerprint(operationData);
wrapper.setOperationFingerprint(opId, Buffer.from(newFingerprint).toString('hex'));
```

### Performance Issues

**Cause**: Generating too many fingerprints synchronously.

**Solution**: Use batch processing with `Promise.all`:

```javascript
// ❌ Slow: Sequential processing
for (const op of operations) {
  await wrapper.generateOperationFingerprint(op.id);
}

// ✅ Fast: Parallel processing
await Promise.all(
  operations.map(op => wrapper.generateOperationFingerprint(op.id))
);
```

### Memory Usage

**Cause**: Storing too many signatures in memory.

**Solution**: Store signatures externally and load on demand:

```javascript
// Store only fingerprint hex, not full signature
const qf = qudag.QuantumFingerprint.generate(operationData);
const fingerprint = qf.asHex(); // 128 chars
// Don't store: qf.getSignature() (3309 bytes)

// Save to disk/database
await database.save({
  operationId: op.id,
  fingerprint: fingerprint
});
```

## Further Reading

- [@qudag/napi-core Documentation](https://github.com/ruvnet/QuDAG)
- [NIST Post-Quantum Cryptography](https://csrc.nist.gov/projects/post-quantum-cryptography)
- [ML-DSA (FIPS 204)](https://csrc.nist.gov/pubs/fips/204/final)
- [BLAKE3 Specification](https://github.com/BLAKE3-team/BLAKE3-specs)

## License

MIT
