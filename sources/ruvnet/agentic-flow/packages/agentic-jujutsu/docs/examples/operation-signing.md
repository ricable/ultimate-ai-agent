# Operation Log Signing with ML-DSA

This guide demonstrates how to use quantum-resistant digital signatures for tamper-proof audit trails in agentic-jujutsu.

## Overview

The operation log signing feature uses **ML-DSA-44** (formerly CRYSTALS-Dilithium), a NIST-approved post-quantum digital signature algorithm. This provides:

- **Quantum resistance**: Signatures remain secure even against quantum computers
- **Tamper detection**: Any modification to signed operations is immediately detectable
- **Audit trails**: Cryptographically verifiable history of all operations
- **Chain validation**: Verify that operation sequences haven't been reordered or removed

## Quick Start

### 1. Generate a Keypair

```javascript
const { generateSigningKeypair } = require('agentic-jujutsu');

// Generate a new keypair for signing
const keypair = generateSigningKeypair();

console.log('Public Key:', keypair.publicKey.substring(0, 32) + '...');
console.log('Secret Key:', keypair.secretKey.substring(0, 32) + '...');

// ⚠️ IMPORTANT: Store the secret key securely!
// Never commit it to version control or expose it publicly.
```

### 2. Sign Operations

```javascript
const { JJWrapper } = require('agentic-jujutsu');

const jj = new JJWrapper();

// Execute some operations
await jj.execute(['describe', '-m', 'Initial commit']);
await jj.execute(['new']);

// Get the operation IDs
const operations = jj.getOperations(2);

// Sign each operation
for (const op of operations) {
    jj.signOperation(op.id, keypair.secretKey, keypair.publicKey);
    console.log(`✓ Signed operation: ${op.id.substring(0, 8)}...`);
}
```

### 3. Verify Signatures

```javascript
// Verify a single operation
const opId = operations[0].id;
const isValid = jj.verifyOperationSignature(opId);
console.log(`Signature valid: ${isValid}`);

// Verify all operations
const result = JSON.parse(jj.verifyAllOperations());
console.log(`Verified ${result.valid_count}/${result.total_signed} operations`);
console.log(`Invalid signatures: ${result.invalid_count}`);
```

## Usage Patterns

### Pattern 1: Sign All Operations Automatically

```javascript
const jj = new JJWrapper();
const keypair = generateSigningKeypair();

// Store keypair for the session
process.env.JJ_SIGNING_SECRET = keypair.secretKey;
process.env.JJ_SIGNING_PUBLIC = keypair.publicKey;

// Hook into operation execution
const originalExecute = jj.execute.bind(jj);
jj.execute = async function(...args) {
    const result = await originalExecute(...args);

    // Sign all unsigned operations
    const unsigned = jj.getUnsignedOperationsCount();
    if (unsigned > 0) {
        jj.signAllOperations(
            process.env.JJ_SIGNING_SECRET,
            process.env.JJ_SIGNING_PUBLIC
        );
    }

    return result;
};

// Now all operations are automatically signed
await jj.execute(['describe', '-m', 'Auto-signed commit']);
```

### Pattern 2: Batch Signing for Performance

```javascript
const jj = new JJWrapper();
const keypair = generateSigningKeypair();

// Execute many operations
for (let i = 0; i < 100; i++) {
    await jj.execute(['new']);
}

// Sign all at once (more efficient)
const signedCount = jj.signAllOperations(
    keypair.secretKey,
    keypair.publicKey
);
console.log(`Signed ${signedCount} operations`);
```

### Pattern 3: Tamper Detection

```javascript
const jj = new JJWrapper();
const keypair = generateSigningKeypair();

// Sign operations
await jj.execute(['describe', '-m', 'Important change']);
const operations = jj.getOperations(1);
jj.signOperation(operations[0].id, keypair.secretKey, keypair.publicKey);

// Later: Verify integrity
const isValid = jj.verifyOperationSignature(operations[0].id);

if (!isValid) {
    console.error('⚠️  TAMPERING DETECTED! Operation has been modified.');
    // Take action: alert, rollback, etc.
} else {
    console.log('✓ Operation integrity verified');
}
```

### Pattern 4: Multi-Agent Signing

```javascript
const { JJWrapper, generateSigningKeypair } = require('agentic-jujutsu');

// Each agent has its own keypair
const agents = {
    'agent-1': generateSigningKeypair(),
    'agent-2': generateSigningKeypair(),
    'agent-3': generateSigningKeypair(),
};

const jj = new JJWrapper();

// Agent 1 performs operation
await jj.execute(['new']);
const op1 = jj.getOperations(1)[0];
jj.signOperation(op1.id, agents['agent-1'].secretKey, agents['agent-1'].publicKey);

// Agent 2 performs operation
await jj.execute(['describe', '-m', 'Agent 2 change']);
const op2 = jj.getOperations(1)[0];
jj.signOperation(op2.id, agents['agent-2'].secretKey, agents['agent-2'].publicKey);

// Verify operations by agent
function verifyByAgent(agentId, publicKey) {
    const result = JSON.parse(jj.verifyAllOperations(publicKey));
    console.log(`Agent ${agentId}: ${result.valid_count} valid operations`);
    return result.valid_count;
}

verifyByAgent('agent-1', agents['agent-1'].publicKey);
verifyByAgent('agent-2', agents['agent-2'].publicKey);
```

### Pattern 5: Signature Chain Verification

```javascript
const jj = new JJWrapper();
const keypair = generateSigningKeypair();

// Create a chain of operations
for (let i = 0; i < 5; i++) {
    await jj.execute(['new']);
}

// Sign all operations
jj.signAllOperations(keypair.secretKey, keypair.publicKey);

// Verify the entire chain
const chainValid = jj.verifySignatureChain();

if (chainValid) {
    console.log('✓ Signature chain is valid');
    console.log('  - No operations removed');
    console.log('  - No operations reordered');
    console.log('  - No operations modified');
} else {
    console.error('⚠️  CHAIN BROKEN! Audit trail compromised.');
}
```

## API Reference

### `generateSigningKeypair()`

Generates a new ML-DSA-44 keypair.

**Returns:** `{ publicKey: string, secretKey: string }`

### `jj.signOperation(operationId, secretKey, publicKey)`

Signs a single operation.

**Parameters:**
- `operationId` (string): The operation ID to sign
- `secretKey` (string): The secret key in hex format
- `publicKey` (string): The corresponding public key in hex format

### `jj.verifyOperationSignature(operationId)`

Verifies a single operation's signature.

**Parameters:**
- `operationId` (string): The operation ID to verify

**Returns:** `boolean` - `true` if valid, `false` if invalid

### `jj.verifyAllOperations(publicKey?)`

Verifies all signed operations in the log.

**Parameters:**
- `publicKey` (string, optional): Public key to verify against. If omitted, uses each operation's stored public key.

**Returns:** JSON string with format:
```json
{
    "total_signed": 10,
    "valid_count": 10,
    "invalid_count": 0
}
```

### `jj.signAllOperations(secretKey, publicKey)`

Signs all unsigned operations in batch.

**Parameters:**
- `secretKey` (string): The secret key in hex format
- `publicKey` (string): The corresponding public key in hex format

**Returns:** `number` - Count of operations signed

### `jj.getSignedOperationsCount()`

Returns the number of signed operations in the log.

**Returns:** `number`

### `jj.getUnsignedOperationsCount()`

Returns the number of unsigned operations in the log.

**Returns:** `number`

### `jj.verifySignatureChain()`

Verifies the integrity of the entire operation chain.

**Returns:** `boolean` - `true` if chain is valid

## Performance

ML-DSA-44 provides excellent performance:

- **Signing**: ~1-2ms per operation
- **Verification**: ~1-2ms per operation
- **Batch operations**: 100 operations in ~100-200ms
- **Memory**: Minimal overhead (~5KB per signature)

## Security Considerations

1. **Key Management**:
   - Store secret keys securely (e.g., environment variables, key management systems)
   - Never commit secret keys to version control
   - Rotate keys periodically

2. **Verification**:
   - Always verify signatures before trusting operation data
   - Implement signature chain verification for critical workflows
   - Alert on signature verification failures

3. **Quantum Resistance**:
   - ML-DSA is designed to resist quantum attacks
   - Provides security level 2 (128-bit security)
   - NIST-approved post-quantum algorithm

4. **Audit Trail**:
   - Signatures prove operation authenticity
   - Detect tampering, deletion, or reordering
   - Create cryptographically verifiable history

## Example: Complete Audit System

```javascript
const { JJWrapper, generateSigningKeypair } = require('agentic-jujutsu');
const fs = require('fs');

class AuditedJJWrapper extends JJWrapper {
    constructor() {
        super();
        this.keypair = this.loadOrGenerateKeypair();
        this.setupAutoSigning();
    }

    loadOrGenerateKeypair() {
        const keyPath = '.jj/signing-keypair.json';

        if (fs.existsSync(keyPath)) {
            return JSON.parse(fs.readFileSync(keyPath, 'utf8'));
        }

        const keypair = generateSigningKeypair();
        fs.writeFileSync(keyPath, JSON.stringify(keypair));
        return keypair;
    }

    setupAutoSigning() {
        const originalExecute = this.execute.bind(this);

        this.execute = async function(...args) {
            const result = await originalExecute(...args);

            // Auto-sign new operations
            const unsigned = this.getUnsignedOperationsCount();
            if (unsigned > 0) {
                this.signAllOperations(
                    this.keypair.secretKey,
                    this.keypair.publicKey
                );
            }

            return result;
        };
    }

    async verifyAuditTrail() {
        // Verify all signatures
        const result = JSON.parse(this.verifyAllOperations());

        if (result.invalid_count > 0) {
            throw new Error('Audit trail compromised! Invalid signatures detected.');
        }

        // Verify chain integrity
        const chainValid = this.verifySignatureChain();
        if (!chainValid) {
            throw new Error('Audit trail compromised! Chain integrity broken.');
        }

        return {
            verified: true,
            totalOperations: result.total_signed,
            message: 'Audit trail verified successfully'
        };
    }

    async generateAuditReport() {
        const operations = this.getOperations(100);
        const signedCount = this.getSignedOperationsCount();

        return {
            timestamp: new Date().toISOString(),
            totalOperations: operations.length,
            signedOperations: signedCount,
            publicKey: this.keypair.publicKey.substring(0, 16) + '...',
            operations: operations.map(op => ({
                id: op.id.substring(0, 8),
                type: op.operation_type,
                timestamp: op.timestamp,
                signed: !!op.signature
            }))
        };
    }
}

// Usage
const jj = new AuditedJJWrapper();

// All operations are automatically signed
await jj.execute(['describe', '-m', 'Audited commit']);

// Verify audit trail
const audit = await jj.verifyAuditTrail();
console.log('✓', audit.message);

// Generate report
const report = await jj.generateAuditReport();
console.log(JSON.stringify(report, null, 2));
```

## Conclusion

Operation log signing provides cryptographic proof of operation authenticity and integrity. Use it for:

- Compliance requirements
- Multi-agent coordination
- Tamper detection
- Audit trails
- Security-critical workflows

The ML-DSA algorithm ensures your audit trails remain secure even in the post-quantum era.
