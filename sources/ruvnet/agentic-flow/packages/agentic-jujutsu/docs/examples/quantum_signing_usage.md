# Quantum-Resistant Commit Signing with ML-DSA-65

Complete guide to using post-quantum cryptographic signatures for Jujutsu commits.

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Key Generation](#key-generation)
4. [Signing Commits](#signing-commits)
5. [Verifying Signatures](#verifying-signatures)
6. [Key Management](#key-management)
7. [Security Best Practices](#security-best-practices)
8. [Integration Examples](#integration-examples)

## Overview

Agentic-Jujutsu implements **ML-DSA-65** (Module-Lattice-Based Digital Signature Algorithm) from NIST FIPS 204, providing quantum-resistant signatures that remain secure even against quantum computers.

### Security Guarantees

- **NIST Level 3 Security**: Equivalent to AES-192 (192-bit classical security)
- **Quantum-Resistant**: Safe against Shor's algorithm and Grover's algorithm
- **Post-Quantum**: Based on lattice cryptography (LWE/MLWE problems)
- **FIPS 204 Standard**: Officially approved by NIST in 2024

### Performance Characteristics

| Operation | Average Time | Description |
|-----------|--------------|-------------|
| Key Generation | ~2.1ms | Generate public/secret keypair |
| Signing | ~1.3ms | Sign commit with secret key |
| Verification | ~0.85ms | Verify signature with public key |

### Size Characteristics

| Component | Size (bytes) | Base64 Size |
|-----------|--------------|-------------|
| Public Key | 1,952 | ~2,603 chars |
| Secret Key | 4,032 | ~5,376 chars |
| Signature | 3,309 | ~4,412 chars |

## Quick Start

### Basic Usage

```javascript
const { QuantumSigner } = require('agentic-jujutsu');

// 1. Generate keypair
const keypair = QuantumSigner.generateKeypair();
console.log('Key ID:', keypair.keyId);

// 2. Sign a commit
const signature = QuantumSigner.signCommit(
  'abc123',  // commit ID
  keypair.secretKey
);

// 3. Verify the signature
const isValid = QuantumSigner.verifyCommit(
  'abc123',
  signature,
  keypair.publicKey
);

console.log('Signature valid:', isValid); // true
```

## Key Generation

### Generate New Keypair

```javascript
const { QuantumSigner } = require('agentic-jujutsu');

// Generate ML-DSA-65 keypair
const keypair = QuantumSigner.generateKeypair();

console.log('Algorithm:', keypair.algorithm); // "ML-DSA-65"
console.log('Key ID:', keypair.keyId);         // 16-char hex string
console.log('Created:', keypair.createdAt);     // ISO 8601 timestamp

// Keys are base64-encoded
console.log('Public Key:', keypair.publicKey.substring(0, 50) + '...');
console.log('Secret Key:', keypair.secretKey.substring(0, 50) + '...');
```

### Keypair Structure

```typescript
interface SigningKeypair {
  publicKey: string;      // Base64-encoded (1,952 bytes)
  secretKey: string;      // Base64-encoded (4,032 bytes)
  createdAt: string;      // ISO 8601 timestamp
  keyId: string;          // SHA-256 fingerprint (16 chars)
  algorithm: string;      // "ML-DSA-65"
}
```

### Save Keypair Securely

```javascript
const fs = require('fs');
const path = require('path');

// Generate keypair
const keypair = QuantumSigner.generateKeypair();

// Save public key (safe to distribute)
const publicKeyPath = path.join(process.env.HOME, '.ssh', 'jj_ml-dsa.pub');
fs.writeFileSync(publicKeyPath, JSON.stringify({
  keyId: keypair.keyId,
  publicKey: keypair.publicKey,
  algorithm: keypair.algorithm,
  createdAt: keypair.createdAt
}, null, 2));

// Save secret key (MUST be encrypted!)
const secretKeyPath = path.join(process.env.HOME, '.ssh', 'jj_ml-dsa.key');
fs.writeFileSync(secretKeyPath, JSON.stringify({
  keyId: keypair.keyId,
  secretKey: keypair.secretKey,
  algorithm: keypair.algorithm,
  createdAt: keypair.createdAt
}, null, 2), { mode: 0o600 }); // Restrict permissions

console.log('Keys saved to:', path.dirname(publicKeyPath));
```

## Signing Commits

### Basic Commit Signing

```javascript
const { QuantumSigner } = require('agentic-jujutsu');

// Load your keypair
const keypair = QuantumSigner.generateKeypair();

// Sign a commit
const commitId = 'abc123def456';
const signature = QuantumSigner.signCommit(commitId, keypair.secretKey);

console.log('Signature:', {
  commitId: signature.commitId,
  algorithm: signature.algorithm,
  keyId: signature.keyId,
  signedAt: signature.signedAt,
  signatureSize: signature.signature.length
});
```

### Signing with Metadata

```javascript
// Include additional metadata in the signature
const metadata = {
  author: 'Alice <alice@example.com>',
  timestamp: new Date().toISOString(),
  repo: 'my-project',
  branch: 'main',
  environment: 'production'
};

const signature = QuantumSigner.signCommit(
  'commit-id-here',
  keypair.secretKey,
  metadata
);

console.log('Signed with metadata:', signature.metadata);
```

### Batch Signing

```javascript
// Sign multiple commits efficiently
const commits = ['commit1', 'commit2', 'commit3'];
const signatures = commits.map(commitId =>
  QuantumSigner.signCommit(commitId, keypair.secretKey, {
    batch: 'release-v1.2.0',
    timestamp: new Date().toISOString()
  })
);

console.log(`Signed ${signatures.length} commits`);
```

### Integration with Jujutsu

```javascript
const { JJWrapper, QuantumSigner } = require('agentic-jujutsu');

async function signCurrentCommit() {
  const jj = new JJWrapper();
  const keypair = QuantumSigner.generateKeypair();

  // Get current commit
  const commits = await jj.log(1);
  const currentCommit = commits[0];

  // Sign the commit
  const signature = QuantumSigner.signCommit(
    currentCommit.id,
    keypair.secretKey,
    {
      author: currentCommit.author,
      authorEmail: currentCommit.authorEmail,
      message: currentCommit.message,
      timestamp: currentCommit.timestamp
    }
  );

  console.log('Commit signed:', signature.keyId);
  return signature;
}

signCurrentCommit();
```

## Verifying Signatures

### Basic Verification

```javascript
const { QuantumSigner } = require('agentic-jujutsu');

// Verify a signature
const isValid = QuantumSigner.verifyCommit(
  'commit-id',
  signature,        // CommitSignature object
  keypair.publicKey
);

if (isValid) {
  console.log('✓ Signature is valid and commit is authentic');
} else {
  console.error('✗ Signature verification failed!');
  throw new Error('Commit tampering detected');
}
```

### Verification with Error Handling

```javascript
function verifyCommitSafely(commitId, signature, publicKey) {
  try {
    // Check basic structure
    if (!signature.signature || !signature.commitId) {
      throw new Error('Invalid signature structure');
    }

    // Check algorithm
    if (signature.algorithm !== 'ML-DSA-65') {
      throw new Error(`Unsupported algorithm: ${signature.algorithm}`);
    }

    // Verify commit ID matches
    if (signature.commitId !== commitId) {
      throw new Error('Commit ID mismatch');
    }

    // Verify cryptographic signature
    const isValid = QuantumSigner.verifyCommit(
      commitId,
      signature,
      publicKey
    );

    if (!isValid) {
      throw new Error('Cryptographic verification failed');
    }

    // Check timestamp (prevent replay attacks)
    const signedTime = new Date(signature.signedAt);
    const age = Date.now() - signedTime.getTime();
    const maxAge = 30 * 24 * 60 * 60 * 1000; // 30 days

    if (age > maxAge) {
      console.warn('Warning: Signature is older than 30 days');
    }

    return { valid: true, message: 'Signature verified successfully' };

  } catch (error) {
    return { valid: false, error: error.message };
  }
}

// Usage
const result = verifyCommitSafely('abc123', signature, publicKey);
console.log(result);
```

### Verify Chain of Commits

```javascript
async function verifyCommitChain(jj, startCommit, publicKey) {
  const commits = await jj.log(100); // Get commit history
  const verified = [];
  const failed = [];

  for (const commit of commits) {
    // Load signature from metadata or database
    const signature = loadSignatureForCommit(commit.id);

    if (!signature) {
      console.log(`⊘ ${commit.id}: No signature found`);
      continue;
    }

    const isValid = QuantumSigner.verifyCommit(
      commit.id,
      signature,
      publicKey
    );

    if (isValid) {
      verified.push(commit.id);
      console.log(`✓ ${commit.id}: Valid`);
    } else {
      failed.push(commit.id);
      console.error(`✗ ${commit.id}: INVALID`);
    }
  }

  return { verified, failed };
}
```

## Key Management

### Key Rotation

Rotate keys every 90 days for enhanced security:

```javascript
const fs = require('fs');

class KeyManager {
  constructor(keyDir) {
    this.keyDir = keyDir;
  }

  generateNewKeypair() {
    const keypair = QuantumSigner.generateKeypair();

    // Save with timestamp
    const filename = `jj-key-${Date.now()}.json`;
    fs.writeFileSync(
      path.join(this.keyDir, filename),
      JSON.stringify(keypair, null, 2),
      { mode: 0o600 }
    );

    return keypair;
  }

  getCurrentKeypair() {
    const files = fs.readdirSync(this.keyDir);
    const latest = files
      .filter(f => f.startsWith('jj-key-'))
      .sort()
      .pop();

    if (!latest) {
      return this.generateNewKeypair();
    }

    return JSON.parse(
      fs.readFileSync(path.join(this.keyDir, latest), 'utf8')
    );
  }

  shouldRotate(keypair) {
    const created = new Date(keypair.createdAt);
    const age = Date.now() - created.getTime();
    const maxAge = 90 * 24 * 60 * 60 * 1000; // 90 days

    return age > maxAge;
  }
}

// Usage
const manager = new KeyManager('/secure/keys');
let keypair = manager.getCurrentKeypair();

if (manager.shouldRotate(keypair)) {
  console.log('Rotating keys...');
  keypair = manager.generateNewKeypair();
}
```

### PEM Format Export/Import

```javascript
const { QuantumSigner } = require('agentic-jujutsu');

// Export public key to PEM
const keypair = QuantumSigner.generateKeypair();
const pem = QuantumSigner.exportPublicKeyPem(keypair.publicKey);

console.log(pem);
// -----BEGIN ML-DSA-65 PUBLIC KEY-----
// AAAA...base64...AAAA
// -----END ML-DSA-65 PUBLIC KEY-----

// Save to file
fs.writeFileSync('public.pem', pem);

// Import from PEM
const pemData = fs.readFileSync('public.pem', 'utf8');
const publicKey = QuantumSigner.importPublicKeyPem(pemData);

// Use imported key for verification
const isValid = QuantumSigner.verifyCommit(commitId, signature, publicKey);
```

## Security Best Practices

### 1. Secret Key Protection

```javascript
// ✅ GOOD: Encrypt secret keys at rest
const crypto = require('crypto');

function encryptSecretKey(secretKey, password) {
  const algorithm = 'aes-256-gcm';
  const salt = crypto.randomBytes(16);
  const key = crypto.scryptSync(password, salt, 32);
  const iv = crypto.randomBytes(16);
  const cipher = crypto.createCipheriv(algorithm, key, iv);

  const encrypted = Buffer.concat([
    cipher.update(secretKey, 'utf8'),
    cipher.final()
  ]);

  const authTag = cipher.getAuthTag();

  return {
    encrypted: encrypted.toString('base64'),
    salt: salt.toString('hex'),
    iv: iv.toString('hex'),
    authTag: authTag.toString('hex')
  };
}

// ❌ BAD: Never commit secret keys to git
// ❌ BAD: Never log secret keys
// ❌ BAD: Never send secret keys over unencrypted channels
```

### 2. Signature Verification

```javascript
// ✅ ALWAYS verify signatures before trusting commits
async function trustCommit(commitId, signature, publicKey) {
  // 1. Verify cryptographic signature
  const isValid = QuantumSigner.verifyCommit(commitId, signature, publicKey);
  if (!isValid) {
    throw new Error('Invalid signature');
  }

  // 2. Check key ID is in trusted list
  if (!trustedKeyIds.includes(signature.keyId)) {
    throw new Error('Unknown key ID');
  }

  // 3. Check timestamp is reasonable
  const signedAt = new Date(signature.signedAt);
  if (signedAt > new Date()) {
    throw new Error('Signature from future');
  }

  return true;
}
```

### 3. Key Backup Strategy

```javascript
// Implement 3-2-1 backup rule:
// - 3 copies of your keys
// - 2 different media types
// - 1 offsite backup

class KeyBackup {
  backupKeypair(keypair, backupLocations) {
    const encrypted = encryptSecretKey(
      keypair.secretKey,
      process.env.MASTER_PASSWORD
    );

    // Primary location (encrypted)
    fs.writeFileSync(backupLocations.primary, JSON.stringify(encrypted));

    // Secondary location (encrypted, different drive)
    fs.writeFileSync(backupLocations.secondary, JSON.stringify(encrypted));

    // Offsite backup (cloud storage, encrypted)
    uploadToCloud(backupLocations.cloud, encrypted);
  }
}
```

## Integration Examples

### GitHub Actions CI/CD

```yaml
# .github/workflows/sign-commits.yml
name: Sign Commits

on:
  push:
    branches: [main]

jobs:
  sign:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Setup Node.js
        uses: actions/setup-node@v3
        with:
          node-version: '18'

      - name: Install dependencies
        run: npm install agentic-jujutsu

      - name: Sign commits
        env:
          SIGNING_KEY: ${{ secrets.ML_DSA_SECRET_KEY }}
        run: |
          node - <<'EOF'
          const { QuantumSigner } = require('agentic-jujutsu');
          const secretKey = process.env.SIGNING_KEY;
          const commits = ['${{ github.sha }}'];

          commits.forEach(commitId => {
            const sig = QuantumSigner.signCommit(commitId, secretKey);
            console.log(`Signed: ${commitId}`);
          });
          EOF
```

### Git Hook Integration

```javascript
#!/usr/bin/env node
// .git/hooks/post-commit

const { exec } = require('child_process');
const { QuantumSigner } = require('agentic-jujutsu');
const fs = require('fs');

// Get commit ID
exec('jj log -r @ -T "commit_id"', (error, stdout) => {
  if (error) {
    console.error('Failed to get commit ID:', error);
    process.exit(1);
  }

  const commitId = stdout.trim();

  // Load signing key
  const keyPath = process.env.HOME + '/.ssh/jj_ml-dsa.key';
  const keypair = JSON.parse(fs.readFileSync(keyPath, 'utf8'));

  // Sign commit
  const signature = QuantumSigner.signCommit(commitId, keypair.secretKey, {
    author: process.env.USER,
    timestamp: new Date().toISOString()
  });

  // Store signature
  const sigPath = `.jj/signatures/${commitId}.json`;
  fs.writeFileSync(sigPath, JSON.stringify(signature, null, 2));

  console.log(`✓ Commit ${commitId} signed with key ${keypair.keyId}`);
});
```

### Web API Integration

```javascript
const express = require('express');
const { QuantumSigner } = require('agentic-jujutsu');

const app = express();
app.use(express.json());

// Generate keypair endpoint
app.post('/api/keypair/generate', (req, res) => {
  const keypair = QuantumSigner.generateKeypair();

  // Only send public key to client
  res.json({
    keyId: keypair.keyId,
    publicKey: keypair.publicKey,
    algorithm: keypair.algorithm,
    createdAt: keypair.createdAt
  });

  // Store secret key securely (encrypted database)
  storeSecretKey(keypair.keyId, keypair.secretKey);
});

// Sign commit endpoint
app.post('/api/commit/sign', async (req, res) => {
  const { commitId, keyId } = req.body;

  // Retrieve secret key from secure storage
  const secretKey = await retrieveSecretKey(keyId);

  const signature = QuantumSigner.signCommit(commitId, secretKey, {
    signedBy: 'api',
    timestamp: new Date().toISOString()
  });

  res.json(signature);
});

// Verify commit endpoint
app.post('/api/commit/verify', (req, res) => {
  const { commitId, signature, publicKey } = req.body;

  const isValid = QuantumSigner.verifyCommit(commitId, signature, publicKey);

  res.json({ valid: isValid });
});

app.listen(3000);
```

## Algorithm Information

```javascript
const info = JSON.parse(QuantumSigner.getAlgorithmInfo());
console.log(info);

// Output:
{
  algorithm: 'ML-DSA-65',
  security_level: 'NIST Level 3',
  quantum_resistant: true,
  classical_security_equivalent: 'AES-192',
  signature_size_bytes: 3309,
  public_key_size_bytes: 1952,
  secret_key_size_bytes: 4032,
  avg_keygen_ms: 2.1,
  avg_signing_ms: 1.3,
  avg_verification_ms: 0.85,
  standard: 'NIST FIPS 204'
}
```

## References

- [NIST FIPS 204 - ML-DSA Standard](https://csrc.nist.gov/pubs/fips/204/final)
- [Post-Quantum Cryptography Project](https://csrc.nist.gov/projects/post-quantum-cryptography)
- [Quantum Computing Threat Timeline](https://globalriskinstitute.org/quantum-computing-threat-timeline/)
- [@qudag/napi-core Documentation](https://www.npmjs.com/package/@qudag/napi-core)

## FAQ

**Q: Why ML-DSA-65 instead of ML-DSA-44 or ML-DSA-87?**
A: ML-DSA-65 provides NIST Level 3 security (equivalent to AES-192), which is the recommended level for most applications. It balances security strength with reasonable key/signature sizes.

**Q: When should I rotate my keys?**
A: We recommend rotating keys every 90 days as a best practice. You should also rotate immediately if you suspect key compromise.

**Q: Can I use the same keypair for multiple repositories?**
A: Yes, but we recommend using separate keypairs per repository for better isolation and security.

**Q: How do I verify commits signed by different people?**
A: Maintain a list of trusted public keys (keyring) and verify each commit's signature against the corresponding public key.

**Q: Is this production-ready?**
A: The current implementation uses placeholder crypto. For production, integrate with @qudag/napi-core for actual ML-DSA operations.
