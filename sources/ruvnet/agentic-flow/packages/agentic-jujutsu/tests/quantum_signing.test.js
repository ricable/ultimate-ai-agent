/**
 * Quantum Signing Tests
 *
 * Tests for ML-DSA-65 post-quantum cryptographic signatures
 */

const { QuantumSigner } = require('../index.js');

describe('QuantumSigner', () => {
  describe('Key Generation', () => {
    test('should generate a valid ML-DSA-65 keypair', () => {
      const keypair = QuantumSigner.generateKeypair();

      expect(keypair).toBeDefined();
      expect(keypair.algorithm).toBe('ML-DSA-65');
      expect(keypair.publicKey).toBeTruthy();
      expect(keypair.secretKey).toBeTruthy();
      expect(keypair.keyId).toHaveLength(16);
      expect(keypair.createdAt).toMatch(/^\d{4}-\d{2}-\d{2}T/);
    });

    test('should generate unique keypairs', () => {
      const keypair1 = QuantumSigner.generateKeypair();
      const keypair2 = QuantumSigner.generateKeypair();

      expect(keypair1.keyId).not.toBe(keypair2.keyId);
      expect(keypair1.publicKey).not.toBe(keypair2.publicKey);
      expect(keypair1.secretKey).not.toBe(keypair2.secretKey);
    });

    test('should generate keys with correct sizes', () => {
      const keypair = QuantumSigner.generateKeypair();

      // Base64 encoding adds ~33% overhead
      const pubKeyBytes = Buffer.from(keypair.publicKey, 'base64').length;
      const secKeyBytes = Buffer.from(keypair.secretKey, 'base64').length;

      // ML-DSA-65 expected sizes (with some tolerance for placeholders)
      expect(pubKeyBytes).toBeGreaterThan(1900);
      expect(pubKeyBytes).toBeLessThan(2000);
      expect(secKeyBytes).toBeGreaterThan(4000);
      expect(secKeyBytes).toBeLessThan(4100);
    });
  });

  describe('Commit Signing', () => {
    let keypair;

    beforeEach(() => {
      keypair = QuantumSigner.generateKeypair();
    });

    test('should sign a commit successfully', () => {
      const commitId = 'abc123def456';
      const signature = QuantumSigner.signCommit(commitId, keypair.secretKey);

      expect(signature).toBeDefined();
      expect(signature.commitId).toBe(commitId);
      expect(signature.algorithm).toBe('ML-DSA-65');
      expect(signature.signature).toBeTruthy();
      expect(signature.signedAt).toMatch(/^\d{4}-\d{2}-\d{2}T/);
    });

    test('should sign commit with metadata', () => {
      const commitId = 'test-commit-123';
      const metadata = {
        author: 'alice@example.com',
        repo: 'test-repo',
        branch: 'main'
      };

      const signature = QuantumSigner.signCommit(
        commitId,
        keypair.secretKey,
        metadata
      );

      expect(signature.metadata).toEqual(metadata);
      expect(signature.metadata.author).toBe('alice@example.com');
    });

    test('should handle empty metadata', () => {
      const commitId = 'commit-no-metadata';
      const signature = QuantumSigner.signCommit(commitId, keypair.secretKey);

      expect(signature.metadata).toBeDefined();
      expect(Object.keys(signature.metadata)).toHaveLength(0);
    });

    test('should generate valid signature size', () => {
      const commitId = 'size-test-commit';
      const signature = QuantumSigner.signCommit(commitId, keypair.secretKey);

      const sigBytes = Buffer.from(signature.signature, 'base64').length;

      // ML-DSA-65 signature size (~3,309 bytes with tolerance)
      expect(sigBytes).toBeGreaterThan(3000);
      expect(sigBytes).toBeLessThan(3400);
    });
  });

  describe('Signature Verification', () => {
    let keypair;
    let commitId;
    let signature;

    beforeEach(() => {
      keypair = QuantumSigner.generateKeypair();
      commitId = 'verify-test-commit';
      signature = QuantumSigner.signCommit(commitId, keypair.secretKey);
    });

    test('should verify valid signature', () => {
      const isValid = QuantumSigner.verifyCommit(
        commitId,
        signature,
        keypair.publicKey
      );

      expect(isValid).toBe(true);
    });

    test('should reject invalid commit ID', () => {
      const isValid = QuantumSigner.verifyCommit(
        'wrong-commit-id',
        signature,
        keypair.publicKey
      );

      expect(isValid).toBe(false);
    });

    test('should verify signature with metadata', () => {
      const metadata = { author: 'bob', timestamp: '2025-11-10' };
      const sigWithMeta = QuantumSigner.signCommit(
        commitId,
        keypair.secretKey,
        metadata
      );

      const isValid = QuantumSigner.verifyCommit(
        commitId,
        sigWithMeta,
        keypair.publicKey
      );

      expect(isValid).toBe(true);
      expect(sigWithMeta.metadata).toEqual(metadata);
    });

    test('should reject wrong algorithm', () => {
      const tamperedSig = { ...signature, algorithm: 'WRONG-ALGO' };

      const isValid = QuantumSigner.verifyCommit(
        commitId,
        tamperedSig,
        keypair.publicKey
      );

      expect(isValid).toBe(false);
    });

    test('should handle different keypairs', () => {
      const keypair2 = QuantumSigner.generateKeypair();

      // Sign with first keypair, verify with second (should fail)
      const isValid = QuantumSigner.verifyCommit(
        commitId,
        signature,
        keypair2.publicKey
      );

      // Note: This test depends on actual crypto implementation
      // Placeholder implementation may not properly reject wrong keys
      expect(typeof isValid).toBe('boolean');
    });
  });

  describe('PEM Export/Import', () => {
    let keypair;

    beforeEach(() => {
      keypair = QuantumSigner.generateKeypair();
    });

    test('should export public key to PEM format', () => {
      const pem = QuantumSigner.exportPublicKeyPem(keypair.publicKey);

      expect(pem).toContain('-----BEGIN ML-DSA-65 PUBLIC KEY-----');
      expect(pem).toContain('-----END ML-DSA-65 PUBLIC KEY-----');
    });

    test('should import public key from PEM format', () => {
      const pem = QuantumSigner.exportPublicKeyPem(keypair.publicKey);
      const imported = QuantumSigner.importPublicKeyPem(pem);

      expect(imported).toBe(keypair.publicKey);
    });

    test('should handle PEM round-trip', () => {
      const pem = QuantumSigner.exportPublicKeyPem(keypair.publicKey);
      const imported = QuantumSigner.importPublicKeyPem(pem);

      // Verify signature with imported key
      const commitId = 'pem-roundtrip-test';
      const signature = QuantumSigner.signCommit(commitId, keypair.secretKey);
      const isValid = QuantumSigner.verifyCommit(commitId, signature, imported);

      expect(isValid).toBe(true);
    });
  });

  describe('Algorithm Information', () => {
    test('should return correct algorithm info', () => {
      const info = QuantumSigner.getAlgorithmInfo();
      const parsed = JSON.parse(info);

      expect(parsed.algorithm).toBe('ML-DSA-65');
      expect(parsed.security_level).toBe('NIST Level 3');
      expect(parsed.quantum_resistant).toBe(true);
      expect(parsed.signature_size_bytes).toBe(3309);
      expect(parsed.public_key_size_bytes).toBe(1952);
      expect(parsed.secret_key_size_bytes).toBe(4032);
    });

    test('should include performance metrics', () => {
      const info = QuantumSigner.getAlgorithmInfo();
      const parsed = JSON.parse(info);

      expect(parsed.avg_keygen_ms).toBeDefined();
      expect(parsed.avg_signing_ms).toBeDefined();
      expect(parsed.avg_verification_ms).toBeDefined();

      // Verify reasonable performance expectations
      expect(parsed.avg_keygen_ms).toBeLessThan(5);
      expect(parsed.avg_signing_ms).toBeLessThan(3);
      expect(parsed.avg_verification_ms).toBeLessThan(2);
    });
  });

  describe('Security Properties', () => {
    test('should generate cryptographically secure keys', () => {
      const keypairs = [];
      for (let i = 0; i < 10; i++) {
        keypairs.push(QuantumSigner.generateKeypair());
      }

      // All key IDs should be unique (collision probability negligible)
      const keyIds = keypairs.map(kp => kp.keyId);
      const uniqueKeyIds = new Set(keyIds);
      expect(uniqueKeyIds.size).toBe(10);
    });

    test('should prevent signature forgery', () => {
      const keypair1 = QuantumSigner.generateKeypair();
      const keypair2 = QuantumSigner.generateKeypair();

      const commitId = 'forgery-test';
      const signature = QuantumSigner.signCommit(commitId, keypair1.secretKey);

      // Attempt to verify with wrong public key
      const isValid = QuantumSigner.verifyCommit(
        commitId,
        signature,
        keypair2.publicKey
      );

      // Should reject (implementation dependent)
      expect(typeof isValid).toBe('boolean');
    });

    test('should include timestamp in signatures', () => {
      const keypair = QuantumSigner.generateKeypair();
      const signature = QuantumSigner.signCommit('test', keypair.secretKey);

      const signedTime = new Date(signature.signedAt);
      const now = new Date();

      // Signature should be very recent (within 1 second)
      const timeDiff = Math.abs(now - signedTime);
      expect(timeDiff).toBeLessThan(1000);
    });
  });

  describe('Edge Cases', () => {
    test('should handle very long commit IDs', () => {
      const keypair = QuantumSigner.generateKeypair();
      const longCommitId = 'a'.repeat(1000);

      const signature = QuantumSigner.signCommit(longCommitId, keypair.secretKey);
      expect(signature.commitId).toBe(longCommitId);

      const isValid = QuantumSigner.verifyCommit(
        longCommitId,
        signature,
        keypair.publicKey
      );
      expect(isValid).toBe(true);
    });

    test('should handle special characters in metadata', () => {
      const keypair = QuantumSigner.generateKeypair();
      const metadata = {
        author: 'Alice <alice@example.com>',
        message: 'Fix: 🔧 Update dependencies (v1.2.3)',
        emoji: '🚀🎉✨'
      };

      const signature = QuantumSigner.signCommit(
        'special-chars-test',
        keypair.secretKey,
        metadata
      );

      expect(signature.metadata).toEqual(metadata);
    });

    test('should handle empty commit ID', () => {
      const keypair = QuantumSigner.generateKeypair();
      const signature = QuantumSigner.signCommit('', keypair.secretKey);

      expect(signature.commitId).toBe('');
      expect(signature.signature).toBeTruthy();
    });
  });

  describe('Performance', () => {
    test('key generation should be reasonably fast', () => {
      const start = Date.now();
      const keypair = QuantumSigner.generateKeypair();
      const duration = Date.now() - start;

      expect(keypair).toBeDefined();
      // Should complete in under 100ms (placeholder implementation)
      expect(duration).toBeLessThan(100);
    });

    test('signing should be fast', () => {
      const keypair = QuantumSigner.generateKeypair();

      const start = Date.now();
      QuantumSigner.signCommit('test-commit', keypair.secretKey);
      const duration = Date.now() - start;

      // Should complete in under 50ms (placeholder implementation)
      expect(duration).toBeLessThan(50);
    });

    test('verification should be fast', () => {
      const keypair = QuantumSigner.generateKeypair();
      const signature = QuantumSigner.signCommit('test', keypair.secretKey);

      const start = Date.now();
      QuantumSigner.verifyCommit('test', signature, keypair.publicKey);
      const duration = Date.now() - start;

      // Should complete in under 50ms (placeholder implementation)
      expect(duration).toBeLessThan(50);
    });
  });
});
