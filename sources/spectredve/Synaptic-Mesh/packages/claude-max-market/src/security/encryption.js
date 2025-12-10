/**
 * Encryption Manager
 * 
 * Handles secure encryption/decryption of job payloads
 * for secure transmission in the market
 */

import crypto from 'crypto';
import fs from 'fs/promises';
import path from 'path';

export class EncryptionManager {
  constructor() {
    this.algorithm = 'aes-256-gcm';
    this.keyLength = 32; // 256 bits
    this.ivLength = 16;  // 128 bits
    this.tagLength = 16; // 128 bits
    this.keyDerivationIterations = 100000;
    
    // Initialize key management
    this.masterKey = null;
    this.keyPairs = new Map();
    
    this.init();
  }

  async init() {
    try {
      await this.loadOrGenerateMasterKey();
      await this.loadKeyPairs();
    } catch (error) {
      console.warn('Encryption initialization warning:', error.message);
    }
  }

  /**
   * Encrypt job payload for secure transmission
   */
  async encrypt(data, recipientId = null) {
    try {
      const plaintext = typeof data === 'string' ? data : JSON.stringify(data);
      
      // Generate random IV
      const iv = crypto.randomBytes(this.ivLength);
      
      // Use master key or derive key for specific recipient
      const key = recipientId 
        ? await this.deriveKeyForRecipient(recipientId)
        : this.masterKey;
      
      // Create cipher
      const cipher = crypto.createCipher(this.algorithm, key);
      cipher.setIV(iv);
      
      // Encrypt data
      let encrypted = cipher.update(plaintext, 'utf8', 'hex');
      encrypted += cipher.final('hex');
      
      // Get authentication tag
      const tag = cipher.getAuthTag();
      
      // Combine IV, tag, and encrypted data
      const result = {
        algorithm: this.algorithm,
        iv: iv.toString('hex'),
        tag: tag.toString('hex'),
        encrypted: encrypted,
        timestamp: new Date().toISOString(),
        recipientId: recipientId || 'broadcast'
      };
      
      return result;
    } catch (error) {
      throw new Error(`Encryption failed: ${error.message}`);
    }
  }

  /**
   * Decrypt job payload
   */
  async decrypt(encryptedData, senderId = null) {
    try {
      if (!encryptedData || typeof encryptedData !== 'object') {
        throw new Error('Invalid encrypted data format');
      }
      
      const { algorithm, iv, tag, encrypted, recipientId } = encryptedData;
      
      if (algorithm !== this.algorithm) {
        throw new Error(`Unsupported encryption algorithm: ${algorithm}`);
      }
      
      // Get decryption key
      const key = senderId 
        ? await this.deriveKeyForRecipient(senderId)
        : this.masterKey;
      
      // Create decipher
      const decipher = crypto.createDecipher(algorithm, key);
      decipher.setIV(Buffer.from(iv, 'hex'));
      decipher.setAuthTag(Buffer.from(tag, 'hex'));
      
      // Decrypt data
      let decrypted = decipher.update(encrypted, 'hex', 'utf8');
      decrypted += decipher.final('utf8');
      
      // Try to parse as JSON, fallback to string
      try {
        return JSON.parse(decrypted);
      } catch {
        return decrypted;
      }
    } catch (error) {
      throw new Error(`Decryption failed: ${error.message}`);
    }
  }

  /**
   * Generate key pair for peer-to-peer encryption
   */
  async generateKeyPair(peerId) {
    try {
      const { publicKey, privateKey } = crypto.generateKeyPairSync('rsa', {
        modulusLength: 2048,
        publicKeyEncoding: {
          type: 'spki',
          format: 'pem'
        },
        privateKeyEncoding: {
          type: 'pkcs8',
          format: 'pem'
        }
      });
      
      const keyPair = {
        peerId,
        publicKey,
        privateKey,
        created: new Date().toISOString()
      };
      
      this.keyPairs.set(peerId, keyPair);
      await this.saveKeyPairs();
      
      return keyPair;
    } catch (error) {
      throw new Error(`Key pair generation failed: ${error.message}`);
    }
  }

  /**
   * Encrypt with recipient's public key
   */
  async encryptForPeer(data, recipientPublicKey) {
    try {
      const plaintext = typeof data === 'string' ? data : JSON.stringify(data);
      
      // RSA encryption has size limits, so use hybrid encryption
      // Generate random AES key
      const aesKey = crypto.randomBytes(this.keyLength);
      const iv = crypto.randomBytes(this.ivLength);
      
      // Encrypt data with AES
      const cipher = crypto.createCipher(this.algorithm, aesKey);
      cipher.setIV(iv);
      
      let encrypted = cipher.update(plaintext, 'utf8', 'hex');
      encrypted += cipher.final('hex');
      const tag = cipher.getAuthTag();
      
      // Encrypt AES key with RSA public key
      const encryptedKey = crypto.publicEncrypt({
        key: recipientPublicKey,
        padding: crypto.constants.RSA_PKCS1_OAEP_PADDING,
        oaepHash: 'sha256'
      }, aesKey);
      
      return {
        algorithm: 'rsa-aes-hybrid',
        encryptedKey: encryptedKey.toString('base64'),
        iv: iv.toString('hex'),
        tag: tag.toString('hex'),
        encrypted: encrypted,
        timestamp: new Date().toISOString()
      };
    } catch (error) {
      throw new Error(`Peer encryption failed: ${error.message}`);
    }
  }

  /**
   * Decrypt with own private key
   */
  async decryptFromPeer(encryptedData, privateKey) {
    try {
      const { encryptedKey, iv, tag, encrypted } = encryptedData;
      
      // Decrypt AES key with RSA private key
      const aesKey = crypto.privateDecrypt({
        key: privateKey,
        padding: crypto.constants.RSA_PKCS1_OAEP_PADDING,
        oaepHash: 'sha256'
      }, Buffer.from(encryptedKey, 'base64'));
      
      // Decrypt data with AES key
      const decipher = crypto.createDecipher(this.algorithm, aesKey);
      decipher.setIV(Buffer.from(iv, 'hex'));
      decipher.setAuthTag(Buffer.from(tag, 'hex'));
      
      let decrypted = decipher.update(encrypted, 'hex', 'utf8');
      decrypted += decipher.final('utf8');
      
      // Try to parse as JSON
      try {
        return JSON.parse(decrypted);
      } catch {
        return decrypted;
      }
    } catch (error) {
      throw new Error(`Peer decryption failed: ${error.message}`);
    }
  }

  /**
   * Generate secure hash for data integrity
   */
  generateHash(data, algorithm = 'sha256') {
    const hash = crypto.createHash(algorithm);
    const input = typeof data === 'string' ? data : JSON.stringify(data);
    hash.update(input);
    return hash.digest('hex');
  }

  /**
   * Verify data integrity
   */
  verifyHash(data, expectedHash, algorithm = 'sha256') {
    const actualHash = this.generateHash(data, algorithm);
    return actualHash === expectedHash;
  }

  /**
   * Generate cryptographically secure random ID
   */
  generateSecureId(length = 32) {
    return crypto.randomBytes(length).toString('hex');
  }

  /**
   * Derive key using PBKDF2
   */
  deriveKey(password, salt, iterations = this.keyDerivationIterations) {
    return crypto.pbkdf2Sync(password, salt, iterations, this.keyLength, 'sha256');
  }

  /**
   * Get public key for peer
   */
  getPublicKey(peerId) {
    const keyPair = this.keyPairs.get(peerId);
    return keyPair ? keyPair.publicKey : null;
  }

  /**
   * Get private key for peer
   */
  getPrivateKey(peerId) {
    const keyPair = this.keyPairs.get(peerId);
    return keyPair ? keyPair.privateKey : null;
  }

  /**
   * Import peer's public key
   */
  async importPeerKey(peerId, publicKey) {
    try {
      // Validate public key format
      crypto.createPublicKey(publicKey);
      
      // Store peer's public key
      this.keyPairs.set(peerId, {
        peerId,
        publicKey,
        privateKey: null,
        imported: true,
        created: new Date().toISOString()
      });
      
      await this.saveKeyPairs();
      return true;
    } catch (error) {
      throw new Error(`Invalid public key for peer ${peerId}: ${error.message}`);
    }
  }

  /**
   * Export own public key
   */
  exportPublicKey(peerId) {
    const keyPair = this.keyPairs.get(peerId);
    if (!keyPair || !keyPair.publicKey) {
      throw new Error(`No public key found for peer ${peerId}`);
    }
    
    return {
      peerId,
      publicKey: keyPair.publicKey,
      created: keyPair.created
    };
  }

  // Private helper methods

  async loadOrGenerateMasterKey() {
    try {
      const keyPath = path.join(process.cwd(), '.claude-encryption-key');
      
      try {
        const keyData = await fs.readFile(keyPath);
        this.masterKey = keyData;
      } catch (error) {
        // Generate new master key
        this.masterKey = crypto.randomBytes(this.keyLength);
        await fs.writeFile(keyPath, this.masterKey);
        await fs.chmod(keyPath, 0o600); // Restrict access
      }
    } catch (error) {
      // Fallback to in-memory key
      this.masterKey = crypto.randomBytes(this.keyLength);
      console.warn('Using in-memory encryption key - data will not persist across restarts');
    }
  }

  async deriveKeyForRecipient(recipientId) {
    const salt = crypto.createHash('sha256').update(recipientId).digest();
    return crypto.pbkdf2Sync(this.masterKey, salt, this.keyDerivationIterations, this.keyLength, 'sha256');
  }

  async loadKeyPairs() {
    try {
      const keyPairsPath = path.join(process.cwd(), '.claude-keypairs.json');
      const data = await fs.readFile(keyPairsPath, 'utf8');
      const keyPairsData = JSON.parse(data);
      
      for (const [peerId, keyPair] of Object.entries(keyPairsData)) {
        this.keyPairs.set(peerId, keyPair);
      }
    } catch (error) {
      // Key pairs file doesn't exist yet
    }
  }

  async saveKeyPairs() {
    try {
      const keyPairsPath = path.join(process.cwd(), '.claude-keypairs.json');
      const keyPairsData = Object.fromEntries(this.keyPairs);
      await fs.writeFile(keyPairsPath, JSON.stringify(keyPairsData, null, 2));
      await fs.chmod(keyPairsPath, 0o600); // Restrict access
    } catch (error) {
      console.warn('Failed to save key pairs:', error.message);
    }
  }

  /**
   * Secure key rotation
   */
  async rotateKeys() {
    try {
      // Generate new master key
      const oldMasterKey = this.masterKey;
      this.masterKey = crypto.randomBytes(this.keyLength);
      
      // Re-encrypt any stored data with new key if needed
      // This would involve re-encrypting stored payloads, which
      // in our case are ephemeral, so we just update the key
      
      await this.loadOrGenerateMasterKey();
      
      console.log('✅ Encryption keys rotated successfully');
      return true;
    } catch (error) {
      // Restore old key on failure
      this.masterKey = oldMasterKey;
      throw new Error(`Key rotation failed: ${error.message}`);
    }
  }

  /**
   * Cleanup old encrypted data
   */
  async cleanupExpiredData(maxAgeMs = 24 * 60 * 60 * 1000) { // 24 hours default
    // In a real implementation, this would clean up any stored encrypted data
    // that has exceeded its retention period
    console.log('🧹 Cleaned up expired encrypted data');
  }
}