/**
 * ReasoningBank Encryption Helpers
 *
 * Provides quantum-resistant encryption utilities for ReasoningBank trajectories
 * using HQC (Hamming Quasi-Cyclic) encryption from @qudag/napi-core
 *
 * @module encryption
 */

const crypto = require('crypto');
const qudag = require('@qudag/napi-core');

/**
 * Encryption key manager for ReasoningBank
 */
class EncryptionKeyManager {
  constructor() {
    this.keys = new Map();
    this.hqc128 = new qudag.Hqc128Wrapper();
  }

  /**
   * Generate a new HQC-128 encryption key pair
   * @returns {Object} { key: base64 string (32 bytes), publicKey: base64 string, secretKey: base64 string }
   */
  generateKeyPair() {
    // Generate key pair for HQC-128
    const { publicKey, secretKey } = this.hqc128.keygen();

    // Generate symmetric key for trajectory encryption (32 bytes for HQC-128)
    const keyBytes = crypto.randomBytes(32);

    return {
      key: keyBytes.toString('base64'),
      publicKey: Buffer.from(publicKey).toString('base64'),
      secretKey: Buffer.from(secretKey).toString('base64')
    };
  }

  /**
   * Store an encryption key securely (in memory)
   * @param {string} id - Key identifier
   * @param {Object} keyPair - Key pair object from generateKeyPair()
   */
  storeKey(id, keyPair) {
    this.keys.set(id, keyPair);
  }

  /**
   * Retrieve an encryption key
   * @param {string} id - Key identifier
   * @returns {Object|undefined} Key pair or undefined if not found
   */
  getKey(id) {
    return this.keys.get(id);
  }

  /**
   * Delete an encryption key
   * @param {string} id - Key identifier
   * @returns {boolean} True if deleted, false if not found
   */
  deleteKey(id) {
    return this.keys.delete(id);
  }

  /**
   * List all stored key IDs
   * @returns {string[]} Array of key identifiers
   */
  listKeys() {
    return Array.from(this.keys.keys());
  }
}

/**
 * Encrypt a trajectory payload using HQC-128
 *
 * @param {string} payload - JSON payload to encrypt
 * @param {string} publicKey - Base64-encoded HQC public key
 * @returns {Object} { ciphertext: base64 string, sharedSecret: base64 string }
 */
function encryptTrajectory(payload, publicKey) {
  const hqc128 = new qudag.Hqc128Wrapper();

  // Convert base64 public key to buffer
  const publicKeyBytes = Buffer.from(publicKey, 'base64');

  // Perform HQC encapsulation to get shared secret
  const { ciphertext, sharedSecret } = hqc128.encapsulate(publicKeyBytes);

  // Use shared secret to encrypt the payload with AES-256-GCM
  const iv = crypto.randomBytes(16);
  const cipher = crypto.createCipheriv('aes-256-gcm',
    Buffer.from(sharedSecret).slice(0, 32), iv);

  let encrypted = cipher.update(payload, 'utf8', 'base64');
  encrypted += cipher.final('base64');
  const authTag = cipher.getAuthTag();

  // Combine: ciphertext + iv + authTag + encrypted payload
  const result = {
    hqcCiphertext: Buffer.from(ciphertext).toString('base64'),
    iv: iv.toString('base64'),
    authTag: authTag.toString('base64'),
    encryptedPayload: encrypted
  };

  return result;
}

/**
 * Decrypt a trajectory payload using HQC-128
 *
 * @param {Object} encryptedData - Encrypted data object from encryptTrajectory()
 * @param {string} secretKey - Base64-encoded HQC secret key
 * @returns {string} Decrypted JSON payload
 */
function decryptTrajectory(encryptedData, secretKey) {
  const hqc128 = new qudag.Hqc128Wrapper();

  // Convert base64 inputs to buffers
  const secretKeyBytes = Buffer.from(secretKey, 'base64');
  const hqcCiphertext = Buffer.from(encryptedData.hqcCiphertext, 'base64');

  // Perform HQC decapsulation to recover shared secret
  const sharedSecret = hqc128.decapsulate(secretKeyBytes, hqcCiphertext);

  // Decrypt the payload with AES-256-GCM
  const iv = Buffer.from(encryptedData.iv, 'base64');
  const authTag = Buffer.from(encryptedData.authTag, 'base64');

  const decipher = crypto.createDecipheriv('aes-256-gcm',
    Buffer.from(sharedSecret).slice(0, 32), iv);
  decipher.setAuthTag(authTag);

  let decrypted = decipher.update(encryptedData.encryptedPayload, 'base64', 'utf8');
  decrypted += decipher.final('utf8');

  return decrypted;
}

/**
 * Enable encryption for a JJWrapper instance
 *
 * @param {Object} wrapper - JJWrapper instance
 * @param {Object} keyPair - Key pair from generateKeyPair()
 * @returns {Promise<void>}
 */
async function enableWrapperEncryption(wrapper, keyPair) {
  wrapper.enableEncryption(keyPair.key, keyPair.publicKey);
}

/**
 * Decrypt a stored trajectory from JJWrapper
 *
 * @param {Object} wrapper - JJWrapper instance
 * @param {string} trajectoryId - UUID of the trajectory
 * @param {string} secretKey - Base64-encoded HQC secret key
 * @returns {Promise<Object>} Decrypted trajectory object
 */
async function decryptWrapperTrajectory(wrapper, trajectoryId, secretKey) {
  // Get encrypted payload from wrapper
  const encryptedPayload = wrapper.getTrajectoryPayload(trajectoryId);

  if (!encryptedPayload) {
    throw new Error(`Trajectory ${trajectoryId} not found or not encrypted`);
  }

  // Parse encrypted data
  const encryptedData = JSON.parse(encryptedPayload);

  // Decrypt using HQC
  const decryptedPayload = decryptTrajectory(encryptedData, secretKey);

  // Return decrypted trajectory to wrapper
  const trajectoryJson = wrapper.decryptTrajectory(trajectoryId, decryptedPayload);
  return JSON.parse(trajectoryJson);
}

/**
 * Create a complete encrypted trajectory workflow
 *
 * @param {Object} wrapper - JJWrapper instance
 * @returns {Object} Helper object with encryption methods
 */
function createEncryptedWorkflow(wrapper) {
  const keyManager = new EncryptionKeyManager();
  const keyPair = keyManager.generateKeyPair();

  // Store key for later use
  keyManager.storeKey('default', keyPair);

  // Enable encryption on wrapper
  wrapper.enableEncryption(keyPair.key, keyPair.publicKey);

  return {
    keyManager,
    keyPair,

    /**
     * Decrypt a trajectory
     * @param {string} trajectoryId - UUID of trajectory to decrypt
     * @returns {Promise<Object>} Decrypted trajectory
     */
    async decryptTrajectory(trajectoryId) {
      return decryptWrapperTrajectory(wrapper, trajectoryId, keyPair.secretKey);
    },

    /**
     * Disable encryption
     */
    disableEncryption() {
      wrapper.disableEncryption();
    },

    /**
     * Check if encryption is enabled
     * @returns {boolean}
     */
    isEnabled() {
      return wrapper.isEncryptionEnabled();
    }
  };
}

module.exports = {
  EncryptionKeyManager,
  encryptTrajectory,
  decryptTrajectory,
  enableWrapperEncryption,
  decryptWrapperTrajectory,
  createEncryptedWorkflow
};
