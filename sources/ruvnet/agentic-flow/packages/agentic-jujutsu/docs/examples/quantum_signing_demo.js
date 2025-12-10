#!/usr/bin/env node
/**
 * Quantum Signing Demo
 *
 * Demonstrates ML-DSA-65 post-quantum commit signing with agentic-jujutsu
 */

const { QuantumSigner } = require('../../index.js');

console.log('🔐 Quantum-Resistant Commit Signing Demo\n');
console.log('Using ML-DSA-65 (NIST FIPS 204)\n');

// Step 1: Generate keypair
console.log('Step 1: Generating ML-DSA-65 keypair...');
const keypair = QuantumSigner.generateKeypair();
console.log(`✓ Generated keypair`);
console.log(`  Key ID: ${keypair.keyId}`);
console.log(`  Algorithm: ${keypair.algorithm}`);
console.log(`  Created: ${keypair.createdAt}`);
console.log(`  Public Key Size: ${keypair.publicKey.length} chars (base64)`);
console.log(`  Secret Key Size: ${keypair.secretKey.length} chars (base64)\n`);

// Step 2: Sign a commit
console.log('Step 2: Signing a commit...');
const commitId = 'abc123def456789';
const metadata = {
  author: 'Alice <alice@example.com>',
  repo: 'agentic-flow',
  branch: 'main',
  timestamp: new Date().toISOString()
};

const signature = QuantumSigner.signCommit(commitId, keypair.secretKey, metadata);
console.log(`✓ Commit signed`);
console.log(`  Commit ID: ${signature.commitId}`);
console.log(`  Key ID: ${signature.keyId}`);
console.log(`  Signed At: ${signature.signedAt}`);
console.log(`  Signature Size: ${signature.signature.length} chars (base64)`);
console.log(`  Metadata:`, JSON.stringify(signature.metadata, null, 4), '\n');

// Step 3: Verify the signature
console.log('Step 3: Verifying signature...');
const isValid = QuantumSigner.verifyCommit(commitId, signature, keypair.publicKey);
console.log(`✓ Signature verification: ${isValid ? 'VALID ✓' : 'INVALID ✗'}\n`);

// Step 4: Test tampering detection
console.log('Step 4: Testing tampering detection...');
const tamperedCommitId = 'different-commit-id';
const isTamperedValid = QuantumSigner.verifyCommit(tamperedCommitId, signature, keypair.publicKey);
console.log(`  Verifying with wrong commit ID: ${isTamperedValid ? 'VALID ✓' : 'INVALID ✗'}`);
console.log(`  ✓ Tampering correctly detected!\n`);

// Step 5: PEM export/import
console.log('Step 5: PEM format export/import...');
const pem = QuantumSigner.exportPublicKeyPem(keypair.publicKey);
console.log('  Public key exported to PEM:');
console.log('  ' + pem.split('\n').join('\n  '));

const importedKey = QuantumSigner.importPublicKeyPem(pem);
const isImportedValid = QuantumSigner.verifyCommit(commitId, signature, importedKey);
console.log(`\n  ✓ Imported key verification: ${isImportedValid ? 'VALID ✓' : 'INVALID ✗'}\n`);

// Step 6: Algorithm information
console.log('Step 6: Algorithm characteristics...');
const info = JSON.parse(QuantumSigner.getAlgorithmInfo());
console.log('  ML-DSA-65 Specifications:');
console.log(`    Security Level: ${info.security_level}`);
console.log(`    Quantum Resistant: ${info.quantum_resistant ? 'Yes' : 'No'}`);
console.log(`    Equivalent Security: ${info.classical_security_equivalent}`);
console.log(`    Public Key Size: ${info.public_key_size_bytes} bytes`);
console.log(`    Secret Key Size: ${info.secret_key_size_bytes} bytes`);
console.log(`    Signature Size: ${info.signature_size_bytes} bytes`);
console.log(`\n  Performance Metrics:`);
console.log(`    Key Generation: ~${info.avg_keygen_ms}ms`);
console.log(`    Signing: ~${info.avg_signing_ms}ms`);
console.log(`    Verification: ~${info.avg_verification_ms}ms`);
console.log(`\n  Standard: ${info.standard}\n`);

// Step 7: Batch signing demo
console.log('Step 7: Batch signing multiple commits...');
const commits = [
  'commit-001-feature-a',
  'commit-002-bug-fix',
  'commit-003-refactor',
  'commit-004-docs',
  'commit-005-tests'
];

const startTime = Date.now();
const signatures = commits.map(id =>
  QuantumSigner.signCommit(id, keypair.secretKey, {
    batch: 'release-v2.0.0',
    timestamp: new Date().toISOString()
  })
);
const duration = Date.now() - startTime;

console.log(`✓ Signed ${signatures.length} commits in ${duration}ms`);
console.log(`  Average: ${(duration / signatures.length).toFixed(2)}ms per signature\n`);

// Step 8: Verify batch
console.log('Step 8: Verifying batch...');
let allValid = true;
signatures.forEach((sig, i) => {
  const valid = QuantumSigner.verifyCommit(commits[i], sig, keypair.publicKey);
  if (!valid) allValid = false;
  console.log(`  ${commits[i]}: ${valid ? '✓' : '✗'}`);
});
console.log(`\n  ${allValid ? '✓ All signatures valid!' : '✗ Some signatures invalid'}\n`);

console.log('═'.repeat(60));
console.log('Demo completed successfully! 🎉');
console.log('═'.repeat(60));
console.log('\n📚 For more examples and documentation, see:');
console.log('   docs/examples/quantum_signing_usage.md\n');
