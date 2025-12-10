/**
 * Quantum Fingerprint Tests
 *
 * Tests for quantum-resistant fingerprint generation and verification
 * using @qudag/napi-core integration with agentic-jujutsu.
 */

const { JjWrapper } = require('../index.js');
const qudag = require('@qudag/napi-core');
const assert = require('assert');
const { describe, it, before } = require('node:test');

describe('Quantum Fingerprint Integration', () => {
  let wrapper;
  let operationId;

  before(async () => {
    wrapper = new JjWrapper();

    // Create a test operation
    const result = await wrapper.execute(['status']);
    const operations = wrapper.getOperations(1);
    assert(operations.length > 0, 'Should have at least one operation');
    operationId = operations[0].id;
  });

  describe('Basic Fingerprint Operations', () => {
    it('should generate quantum fingerprint for an operation', async () => {
      const fingerprint = await wrapper.generateOperationFingerprint(operationId);

      assert(fingerprint, 'Fingerprint should be generated');
      assert(typeof fingerprint === 'string', 'Fingerprint should be a string');
      assert(fingerprint.length > 0, 'Fingerprint should not be empty');

      console.log(`Generated fingerprint (length: ${fingerprint.length}): ${fingerprint.substring(0, 32)}...`);
    });

    it('should retrieve operation data for fingerprinting', () => {
      const data = wrapper.getOperationData(operationId);

      assert(data instanceof Buffer, 'Should return a Buffer');
      assert(data.length > 0, 'Data should not be empty');

      console.log(`Operation data size: ${data.length} bytes`);
    });

    it('should verify a valid fingerprint', async () => {
      const fingerprint = await wrapper.generateOperationFingerprint(operationId);
      const isValid = await wrapper.verifyOperationFingerprint(operationId, fingerprint);

      assert.strictEqual(isValid, true, 'Fingerprint should be valid');
    });

    it('should reject an invalid fingerprint', async () => {
      const invalidFingerprint = 'invalid_fingerprint_12345';
      const isValid = await wrapper.verifyOperationFingerprint(operationId, invalidFingerprint);

      assert.strictEqual(isValid, false, 'Invalid fingerprint should be rejected');
    });
  });

  describe('Quantum Fingerprint with @qudag/napi-core', () => {
    it('should generate quantum fingerprint using QuantumFingerprint class', () => {
      const operationData = wrapper.getOperationData(operationId);

      // Generate quantum fingerprint using @qudag/napi-core
      const quantumFingerprint = qudag.QuantumFingerprint.generate(operationData);

      assert(quantumFingerprint, 'Quantum fingerprint should be generated');

      const fpHex = quantumFingerprint.asHex();
      assert(typeof fpHex === 'string', 'Fingerprint should be convertible to hex');
      assert(fpHex.length === 128, 'BLAKE3 hash should be 64 bytes (128 hex chars)');

      console.log(`Quantum fingerprint (BLAKE3): ${fpHex.substring(0, 32)}...`);
    });

    it('should verify quantum fingerprint', () => {
      const operationData = wrapper.getOperationData(operationId);

      // Generate and verify
      const quantumFingerprint = qudag.QuantumFingerprint.generate(operationData);
      const isValid = quantumFingerprint.verify();

      assert.strictEqual(isValid, true, 'Quantum fingerprint should be valid');
    });

    it('should use convenience functions for quantum fingerprinting', () => {
      const operationData = wrapper.getOperationData(operationId);

      // Generate using convenience function
      const fingerprintBytes = qudag.generateQuantumFingerprint(operationData);

      assert(fingerprintBytes instanceof Uint8Array, 'Should return Uint8Array');
      assert(fingerprintBytes.length === 64, 'BLAKE3 produces 64 bytes');

      // Verify using convenience function
      const isValid = qudag.verifyQuantumFingerprint(operationData, Buffer.from(fingerprintBytes));
      assert.strictEqual(isValid, true, 'Fingerprint should verify successfully');

      console.log(`Fingerprint bytes (first 16): ${Array.from(fingerprintBytes.slice(0, 16)).map(b => b.toString(16).padStart(2, '0')).join('')}`);
    });

    it('should detect data tampering', () => {
      const operationData = wrapper.getOperationData(operationId);

      // Generate fingerprint
      const fingerprintBytes = qudag.generateQuantumFingerprint(operationData);

      // Tamper with data
      const tamperedData = Buffer.from(operationData);
      tamperedData[0] = tamperedData[0] ^ 0xFF; // Flip bits

      // Verify should fail
      const isValid = qudag.verifyQuantumFingerprint(tamperedData, Buffer.from(fingerprintBytes));
      assert.strictEqual(isValid, false, 'Tampered data should fail verification');
    });
  });

  describe('Fingerprint Storage and Retrieval', () => {
    it('should store quantum fingerprint with operation', () => {
      const operationData = wrapper.getOperationData(operationId);
      const fingerprint = qudag.QuantumFingerprint.generate(operationData);
      const fpHex = fingerprint.asHex();

      // Store fingerprint
      wrapper.setOperationFingerprint(operationId, fpHex);

      // Retrieve and verify
      const operations = wrapper.getOperations(100);
      const op = operations.find(o => o.id === operationId);

      assert(op, 'Operation should be found');
      assert.strictEqual(op.quantumFingerprint, fpHex, 'Stored fingerprint should match');
    });

    it('should persist fingerprint across operation queries', async () => {
      // Generate and store
      const operationData = wrapper.getOperationData(operationId);
      const fingerprint = qudag.QuantumFingerprint.generate(operationData);
      const fpHex = fingerprint.asHex();

      wrapper.setOperationFingerprint(operationId, fpHex);

      // Verify persistence
      const isValid = await wrapper.verifyOperationFingerprint(operationId, fpHex);
      assert.strictEqual(isValid, true, 'Persisted fingerprint should verify');
    });
  });

  describe('Performance Benchmarks', () => {
    it('should benchmark quantum fingerprint generation', () => {
      const operationData = wrapper.getOperationData(operationId);
      const iterations = 1000;

      const start = Date.now();
      for (let i = 0; i < iterations; i++) {
        qudag.generateQuantumFingerprint(operationData);
      }
      const elapsed = Date.now() - start;

      const avgTime = elapsed / iterations;
      console.log(`\nPerformance: ${iterations} fingerprints in ${elapsed}ms (${avgTime.toFixed(3)}ms each)`);

      // Should be fast (< 1ms per fingerprint on most systems)
      assert(avgTime < 10, 'Fingerprint generation should be fast');
    });

    it('should benchmark quantum fingerprint verification', () => {
      const operationData = wrapper.getOperationData(operationId);
      const fingerprint = qudag.QuantumFingerprint.generate(operationData);
      const iterations = 1000;

      const start = Date.now();
      for (let i = 0; i < iterations; i++) {
        fingerprint.verify();
      }
      const elapsed = Date.now() - start;

      const avgTime = elapsed / iterations;
      console.log(`Performance: ${iterations} verifications in ${elapsed}ms (${avgTime.toFixed(3)}ms each)`);

      assert(avgTime < 10, 'Verification should be fast');
    });
  });

  describe('Integration with ML-DSA Signatures', () => {
    it('should access ML-DSA signature from quantum fingerprint', () => {
      const operationData = wrapper.getOperationData(operationId);
      const quantumFingerprint = qudag.QuantumFingerprint.generate(operationData);

      // Get signature and public key
      const signature = quantumFingerprint.getSignature();
      const publicKey = quantumFingerprint.getPublicKey();

      assert(signature instanceof Uint8Array, 'Signature should be Uint8Array');
      assert(publicKey instanceof Uint8Array, 'Public key should be Uint8Array');

      console.log(`\nML-DSA-65 Signature size: ${signature.length} bytes`);
      console.log(`ML-DSA-65 Public key size: ${publicKey.length} bytes`);

      // Verify signature is valid
      const isValid = quantumFingerprint.verify();
      assert.strictEqual(isValid, true, 'ML-DSA signature should be valid');
    });

    it('should verify ML-DSA signature independently', () => {
      const operationData = wrapper.getOperationData(operationId);
      const quantumFingerprint = qudag.QuantumFingerprint.generate(operationData);

      // Get components
      const fingerprintBytes = quantumFingerprint.asBytes();
      const signature = quantumFingerprint.getSignature();
      const publicKeyBytes = quantumFingerprint.getPublicKey();

      // Create public key object and verify
      const publicKey = qudag.MlDsaPublicKey.fromBytes(Buffer.from(publicKeyBytes));
      const isValid = publicKey.verify(Buffer.from(fingerprintBytes), Buffer.from(signature));

      assert.strictEqual(isValid, true, 'Independent ML-DSA verification should succeed');
    });
  });

  describe('Error Handling', () => {
    it('should handle non-existent operation', async () => {
      const invalidId = 'non-existent-operation-id';

      try {
        await wrapper.generateOperationFingerprint(invalidId);
        assert.fail('Should throw error for non-existent operation');
      } catch (error) {
        assert(error.message.includes('not found'), 'Error should mention operation not found');
      }
    });

    it('should handle invalid fingerprint format', async () => {
      const invalidFingerprint = 'not-a-valid-hex-string!@#$';
      const isValid = await wrapper.verifyOperationFingerprint(operationId, invalidFingerprint);

      assert.strictEqual(isValid, false, 'Invalid fingerprint format should fail verification');
    });
  });

  describe('Quantum Resistance Features', () => {
    it('should use NIST-approved ML-DSA algorithm', () => {
      const info = qudag.getMlDsaInfo();

      assert.strictEqual(info.algorithm, 'ML-DSA-65', 'Should use ML-DSA-65');
      assert.strictEqual(info.securityLevel, 3, 'Should provide security level 3 (AES-192 equivalent)');
      assert.strictEqual(info.publicKeySize, 1952, 'Public key should be 1952 bytes');
      assert.strictEqual(info.signatureSize, 3309, 'Signature should be ~3309 bytes');

      console.log('\nML-DSA Algorithm Info:');
      console.log(`  Algorithm: ${info.algorithm}`);
      console.log(`  Security Level: ${info.securityLevel} (NIST)`);
      console.log(`  Public Key: ${info.publicKeySize} bytes`);
      console.log(`  Signature: ${info.signatureSize} bytes`);
    });

    it('should demonstrate quantum-resistant properties', () => {
      const operationData = wrapper.getOperationData(operationId);

      // Generate quantum fingerprint
      const qf1 = qudag.QuantumFingerprint.generate(operationData);
      const qf2 = qudag.QuantumFingerprint.generate(operationData);

      // Fingerprints of same data should match (deterministic hash)
      assert.strictEqual(qf1.asHex(), qf2.asHex(), 'Fingerprints should be deterministic');

      // But signatures will differ (randomized signing)
      const sig1 = Buffer.from(qf1.getSignature()).toString('hex');
      const sig2 = Buffer.from(qf2.getSignature()).toString('hex');
      assert.notStrictEqual(sig1, sig2, 'ML-DSA signatures should use randomness');

      console.log('\nQuantum Resistance Demo:');
      console.log(`  Fingerprint 1: ${qf1.asHex().substring(0, 32)}...`);
      console.log(`  Fingerprint 2: ${qf2.asHex().substring(0, 32)}...`);
      console.log(`  Fingerprints match: ${qf1.asHex() === qf2.asHex()}`);
      console.log(`  Signatures differ: ${sig1 !== sig2} (expected for ML-DSA)`);
    });
  });
});
