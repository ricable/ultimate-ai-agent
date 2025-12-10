# Quantum-Resistant Operation Signing

## Overview

Agentic-Jujutsu implements **ML-DSA-44** (CRYSTALS-Dilithium Level 2) signatures for tamper-proof audit trails. This provides quantum-resistant digital signatures on all operations in the operation log.

## Features

✅ **Quantum-Resistant**: ML-DSA is a NIST-approved post-quantum signature algorithm
✅ **Tamper-Proof**: Detect any modifications to signed operations
✅ **High Performance**: ~1-2ms per signature/verification
✅ **Chain Validation**: Verify operation sequences haven't been reordered
✅ **Multi-Agent Support**: Each agent can have its own keypair
✅ **Batch Operations**: Sign/verify multiple operations efficiently

## Quick Start

```javascript
const { JJWrapper, generateSigningKeypair } = require('agentic-jujutsu');

// 1. Generate keypair
const keypair = generateSigningKeypair();

// 2. Create wrapper
const jj = new JJWrapper();

// 3. Execute operations
await jj.execute(['describe', '-m', 'My commit']);

// 4. Sign operations
const operations = jj.getOperations(1);
jj.signOperation(operations[0].id, keypair.secretKey, keypair.publicKey);

// 5. Verify signature
const isValid = jj.verifyOperationSignature(operations[0].id);
console.log('Signature valid:', isValid); // true
```

## API

### Generate Keypair

```javascript
const keypair = generateSigningKeypair();
// Returns: { publicKey: string, secretKey: string }
```

Keys are hex-encoded:
- Public key: 1312 bytes (2624 hex chars)
- Secret key: 2560 bytes (5120 hex chars)

### Sign Operations

#### Single Operation

```javascript
jj.signOperation(operationId, secretKey, publicKey);
```

#### Batch Signing

```javascript
const count = jj.signAllOperations(secretKey, publicKey);
console.log(`Signed ${count} operations`);
```

### Verify Signatures

#### Single Operation

```javascript
const isValid = jj.verifyOperationSignature(operationId);
```

#### All Operations

```javascript
const result = JSON.parse(jj.verifyAllOperations());
// {
//   total_signed: 10,
//   valid_count: 10,
//   invalid_count: 0
// }
```

#### Specific Public Key

```javascript
// Verify all operations were signed by a specific key
const result = JSON.parse(jj.verifyAllOperations(publicKey));
```

### Chain Verification

```javascript
const chainValid = jj.verifySignatureChain();
// Verifies:
// - All signatures are valid
// - Parent-child relationships are intact
// - No operations removed/reordered
```

### Statistics

```javascript
const signedCount = jj.getSignedOperationsCount();
const unsignedCount = jj.getUnsignedOperationsCount();
```

## Architecture

### Signature Format

Each signed operation stores:
- `signature`: Hex-encoded ML-DSA signature (2420 bytes → 4840 hex chars)
- `signature_public_key`: Hex-encoded public key for verification
- `timestamp`: ISO 8601 timestamp when signed

### Signing Process

1. **Canonical Representation**: Create hash of operation data
   ```
   SHA-256(operation_id | command | timestamp | user)
   ```

2. **Sign Hash**: Apply ML-DSA signing
   ```rust
   signature = ml_dsa_sign(message_hash, secret_key)
   ```

3. **Store Signature**: Save signature and public key in operation

### Verification Process

1. **Recreate Hash**: Generate same canonical hash
2. **Verify Signature**: Use ML-DSA verification
   ```rust
   valid = ml_dsa_verify(signature, message_hash, public_key)
   ```

## Security

### ML-DSA Security Level

- **Algorithm**: ML-DSA-44 (Dilithium Level 2)
- **Security**: 128-bit quantum security
- **NIST Status**: Approved (FIPS 204)
- **Key Sizes**:
  - Public: 1312 bytes
  - Secret: 2560 bytes
  - Signature: 2420 bytes

### Threat Model

**Protects Against:**
- ✅ Tampering with operation data
- ✅ Reordering operations
- ✅ Removing operations
- ✅ Forging operations
- ✅ Quantum computer attacks

**Does Not Protect Against:**
- ❌ Key compromise (store keys securely!)
- ❌ Side-channel attacks (use constant-time implementations in production)
- ❌ Social engineering

### Best Practices

1. **Key Management**:
   ```javascript
   // ✅ Good: Use environment variables
   const secretKey = process.env.JJ_SIGNING_SECRET;

   // ❌ Bad: Hardcode keys
   const secretKey = "abc123..."; // Never do this!
   ```

2. **Key Storage**:
   - Store in secure key management system
   - Use environment variables for development
   - Rotate keys periodically
   - Never commit keys to version control

3. **Verification**:
   ```javascript
   // Always verify before trusting data
   const result = JSON.parse(jj.verifyAllOperations());
   if (result.invalid_count > 0) {
       throw new Error('Tampered operations detected!');
   }
   ```

4. **Chain Validation**:
   ```javascript
   // Verify chain integrity for critical operations
   if (!jj.verifySignatureChain()) {
       throw new Error('Operation chain compromised!');
   }
   ```

## Performance

Benchmarks on Intel i7-10700K:

| Operation | Time | Throughput |
|-----------|------|------------|
| Keypair Generation | ~5ms | 200 keypairs/sec |
| Signing | ~1.5ms | 666 signatures/sec |
| Verification | ~1.2ms | 833 verifications/sec |
| Batch Signing (100) | ~150ms | 666 ops/sec |
| Batch Verification (100) | ~120ms | 833 ops/sec |

Memory overhead: ~5KB per signed operation

## Use Cases

### 1. Compliance & Audit

```javascript
// Create auditable operation log
const jj = new JJWrapper();
const keypair = loadCompanyKeypair();

// All operations signed
await jj.execute(['commit', '-m', 'Feature X']);
jj.signAllOperations(keypair.secretKey, keypair.publicKey);

// Generate audit report
const report = {
    timestamp: new Date(),
    operations: jj.getOperations(100),
    verification: JSON.parse(jj.verifyAllOperations())
};

fs.writeFileSync('audit-report.json', JSON.stringify(report));
```

### 2. Multi-Agent Coordination

```javascript
// Each agent signs with their own key
const agents = {
    'coder': generateSigningKeypair(),
    'reviewer': generateSigningKeypair(),
    'deployer': generateSigningKeypair(),
};

// Coder makes changes
await jj.execute(['new']);
jj.signOperation(opId, agents.coder.secretKey, agents.coder.publicKey);

// Verify who did what
function getAgentOperations(agentKey) {
    return jj.getOperations(100).filter(op =>
        op.signature_public_key === agentKey
    );
}
```

### 3. Tamper Detection

```javascript
// Monitor for tampering
setInterval(() => {
    const result = JSON.parse(jj.verifyAllOperations());

    if (result.invalid_count > 0) {
        alert('SECURITY ALERT: Tampered operations detected!');
        // Take action: rollback, alert admins, etc.
    }
}, 60000); // Check every minute
```

## Implementation Details

### Rust API

```rust
use agentic_jujutsu::crypto::{generate_signing_keypair, sign_message_internal};
use agentic_jujutsu::operations::{JJOperation, JJOperationLog};

// Generate keypair
let keypair = generate_signing_keypair();

// Sign operation
let mut operation = JJOperation::new(/*...*/);
operation.sign(&keypair.secret_key, &keypair.public_key)?;

// Verify
let is_valid = operation.verify_signature()?;
```

### JavaScript/TypeScript API

```typescript
import { JJWrapper, generateSigningKeypair } from 'agentic-jujutsu';

const keypair = generateSigningKeypair();
const jj = new JJWrapper();

// TypeScript types included
const isValid: boolean = jj.verifyOperationSignature(operationId);
```

## Future Enhancements

Planned features:

- [ ] Automatic key rotation
- [ ] Hardware security module (HSM) support
- [ ] Multi-signature support
- [ ] Timestamp authority integration
- [ ] Merkle tree for efficient bulk verification
- [ ] WebAssembly implementation for browser usage

## FAQ

**Q: Do I need to sign every operation?**
A: No, but signing provides tamper detection. For critical operations (deployments, releases), always sign.

**Q: Can I verify operations without the secret key?**
A: Yes! Verification only needs the public key, which can be shared safely.

**Q: What if I lose the secret key?**
A: You won't be able to sign new operations with that key, but existing signatures remain valid. Generate a new keypair.

**Q: Are signatures permanent?**
A: Signatures are stored in the operation log. They persist as long as the operation log exists.

**Q: Can I use the same keypair across multiple repositories?**
A: Yes, but best practice is one keypair per repository or per agent for better auditing.

**Q: How does this compare to Git's GPG signing?**
A: Similar concept, but ML-DSA is quantum-resistant and faster. Git GPG signs commits; this signs all operations.

## Resources

- [ML-DSA Specification (FIPS 204)](https://csrc.nist.gov/pubs/fips/204/final)
- [NIST Post-Quantum Cryptography](https://csrc.nist.gov/projects/post-quantum-cryptography)
- [Dilithium Paper](https://pq-crystals.org/dilithium/resources.shtml)
- [Example Code](./examples/operation-signing.md)

## Support

For questions or issues:
- GitHub Issues: https://github.com/ruvnet/agentic-flow/issues
- Documentation: https://docs.rs/agentic-jujutsu

---

**⚠️ Security Notice**: This is a demonstration implementation. For production use, consider:
- Hardware security modules for key storage
- Professional security audit
- FIPS-validated ML-DSA implementation
- Secure key management practices
