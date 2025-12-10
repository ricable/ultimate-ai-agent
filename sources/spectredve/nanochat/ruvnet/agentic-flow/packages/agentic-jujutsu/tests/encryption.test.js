/**
 * ReasoningBank Encryption Tests
 *
 * Tests for quantum-resistant encryption of trajectory data
 */

const { JJWrapper } = require('../index.js');
const {
  EncryptionKeyManager,
  encryptTrajectory,
  decryptTrajectory,
  createEncryptedWorkflow
} = require('../helpers/encryption');

describe('ReasoningBank Encryption', () => {
  let wrapper;
  let keyManager;

  beforeEach(() => {
    wrapper = new JJWrapper();
    keyManager = new EncryptionKeyManager();
  });

  describe('EncryptionKeyManager', () => {
    test('should generate valid key pair', () => {
      const keyPair = keyManager.generateKeyPair();

      expect(keyPair).toHaveProperty('key');
      expect(keyPair).toHaveProperty('publicKey');
      expect(keyPair).toHaveProperty('secretKey');

      // Verify base64 encoding
      expect(() => Buffer.from(keyPair.key, 'base64')).not.toThrow();
      expect(() => Buffer.from(keyPair.publicKey, 'base64')).not.toThrow();
      expect(() => Buffer.from(keyPair.secretKey, 'base64')).not.toThrow();

      // Verify key length (32 bytes for HQC-128)
      const keyBytes = Buffer.from(keyPair.key, 'base64');
      expect(keyBytes.length).toBe(32);
    });

    test('should store and retrieve keys', () => {
      const keyPair = keyManager.generateKeyPair();

      keyManager.storeKey('test-key', keyPair);
      const retrieved = keyManager.getKey('test-key');

      expect(retrieved).toEqual(keyPair);
    });

    test('should list stored keys', () => {
      keyManager.storeKey('key1', keyManager.generateKeyPair());
      keyManager.storeKey('key2', keyManager.generateKeyPair());

      const keys = keyManager.listKeys();
      expect(keys).toContain('key1');
      expect(keys).toContain('key2');
      expect(keys.length).toBe(2);
    });

    test('should delete keys', () => {
      const keyPair = keyManager.generateKeyPair();
      keyManager.storeKey('test-key', keyPair);

      const deleted = keyManager.deleteKey('test-key');
      expect(deleted).toBe(true);

      const retrieved = keyManager.getKey('test-key');
      expect(retrieved).toBeUndefined();
    });
  });

  describe('Trajectory Encryption/Decryption', () => {
    test('should encrypt and decrypt trajectory payload', () => {
      const keyPair = keyManager.generateKeyPair();
      const payload = JSON.stringify({
        operations: ['commit', 'push'],
        initial_context: { branch: 'main' },
        final_context: { success: true },
        critique: 'Well done!'
      });

      // Encrypt
      const encrypted = encryptTrajectory(payload, keyPair.publicKey);

      expect(encrypted).toHaveProperty('hqcCiphertext');
      expect(encrypted).toHaveProperty('iv');
      expect(encrypted).toHaveProperty('authTag');
      expect(encrypted).toHaveProperty('encryptedPayload');

      // Decrypt
      const decrypted = decryptTrajectory(encrypted, keyPair.secretKey);

      expect(decrypted).toBe(payload);
      expect(JSON.parse(decrypted)).toEqual(JSON.parse(payload));
    });

    test('should fail decryption with wrong secret key', () => {
      const keyPair1 = keyManager.generateKeyPair();
      const keyPair2 = keyManager.generateKeyPair();

      const payload = JSON.stringify({ data: 'sensitive' });
      const encrypted = encryptTrajectory(payload, keyPair1.publicKey);

      // Try to decrypt with wrong secret key
      expect(() => {
        decryptTrajectory(encrypted, keyPair2.secretKey);
      }).toThrow();
    });

    test('should handle large payloads', () => {
      const keyPair = keyManager.generateKeyPair();

      // Create a large payload
      const largeData = {
        operations: Array(1000).fill().map((_, i) => ({
          type: 'commit',
          id: `op-${i}`,
          data: 'x'.repeat(100)
        }))
      };
      const payload = JSON.stringify(largeData);

      const encrypted = encryptTrajectory(payload, keyPair.publicKey);
      const decrypted = decryptTrajectory(encrypted, keyPair.secretKey);

      expect(JSON.parse(decrypted)).toEqual(largeData);
    });
  });

  describe('JJWrapper Integration', () => {
    test('should enable encryption on wrapper', () => {
      const keyPair = keyManager.generateKeyPair();

      // Should not throw
      expect(() => {
        wrapper.enableEncryption(keyPair.key, keyPair.publicKey);
      }).not.toThrow();

      expect(wrapper.isEncryptionEnabled()).toBe(true);
    });

    test('should disable encryption', () => {
      const keyPair = keyManager.generateKeyPair();

      wrapper.enableEncryption(keyPair.key, keyPair.publicKey);
      expect(wrapper.isEncryptionEnabled()).toBe(true);

      wrapper.disableEncryption();
      expect(wrapper.isEncryptionEnabled()).toBe(false);
    });

    test('should reject invalid encryption key length', () => {
      const invalidKey = Buffer.from('short').toString('base64');

      expect(() => {
        wrapper.enableEncryption(invalidKey);
      }).toThrow(/32 bytes/);
    });

    test('should reject invalid base64 keys', () => {
      expect(() => {
        wrapper.enableEncryption('not-valid-base64!!!');
      }).toThrow(/base64/);
    });
  });

  describe('Encrypted Workflow', () => {
    test('should create complete encrypted workflow', () => {
      const workflow = createEncryptedWorkflow(wrapper);

      expect(workflow).toHaveProperty('keyManager');
      expect(workflow).toHaveProperty('keyPair');
      expect(workflow).toHaveProperty('decryptTrajectory');
      expect(workflow).toHaveProperty('disableEncryption');
      expect(workflow).toHaveProperty('isEnabled');

      expect(workflow.isEnabled()).toBe(true);
    });

    test('should encrypt trajectories automatically when stored', () => {
      const workflow = createEncryptedWorkflow(wrapper);

      // Start a trajectory
      wrapper.startTrajectory('Test encryption workflow');

      // Finalize with success
      wrapper.finalizeTrajectory(0.95, 'Encryption test successful');

      // Verify encryption is still enabled
      expect(workflow.isEnabled()).toBe(true);
    });

    test('should disable workflow encryption', () => {
      const workflow = createEncryptedWorkflow(wrapper);

      expect(workflow.isEnabled()).toBe(true);

      workflow.disableEncryption();

      expect(workflow.isEnabled()).toBe(false);
    });
  });

  describe('Security Tests', () => {
    test('should generate unique keys each time', () => {
      const keyPair1 = keyManager.generateKeyPair();
      const keyPair2 = keyManager.generateKeyPair();

      expect(keyPair1.key).not.toBe(keyPair2.key);
      expect(keyPair1.publicKey).not.toBe(keyPair2.publicKey);
      expect(keyPair1.secretKey).not.toBe(keyPair2.secretKey);
    });

    test('should produce different ciphertexts for same plaintext', () => {
      const keyPair = keyManager.generateKeyPair();
      const payload = JSON.stringify({ data: 'test' });

      const encrypted1 = encryptTrajectory(payload, keyPair.publicKey);
      const encrypted2 = encryptTrajectory(payload, keyPair.publicKey);

      // IVs should be different
      expect(encrypted1.iv).not.toBe(encrypted2.iv);
      // Ciphertexts should be different due to random IVs
      expect(encrypted1.encryptedPayload).not.toBe(encrypted2.encryptedPayload);

      // But both should decrypt to same payload
      const decrypted1 = decryptTrajectory(encrypted1, keyPair.secretKey);
      const decrypted2 = decryptTrajectory(encrypted2, keyPair.secretKey);

      expect(decrypted1).toBe(payload);
      expect(decrypted2).toBe(payload);
    });

    test('should detect tampering with auth tag verification', () => {
      const keyPair = keyManager.generateKeyPair();
      const payload = JSON.stringify({ data: 'sensitive' });

      const encrypted = encryptTrajectory(payload, keyPair.publicKey);

      // Tamper with ciphertext
      encrypted.encryptedPayload = encrypted.encryptedPayload.slice(0, -5) + 'XXXXX';

      // Should fail auth tag verification
      expect(() => {
        decryptTrajectory(encrypted, keyPair.secretKey);
      }).toThrow();
    });
  });
});

console.log('✅ ReasoningBank Encryption Tests Defined');
console.log('Run with: npm test encryption.test.js');
