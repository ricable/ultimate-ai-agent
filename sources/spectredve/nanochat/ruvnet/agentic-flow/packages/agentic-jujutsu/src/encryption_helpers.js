/**
 * ReasoningBank Encryption Helpers
 *
 * Provides quantum-resistant encryption utilities for ReasoningBank trajectories
 * using HQC (Hamming Quasi-Cyclic) encryption from @qudag/napi-core
 *
 * @module encryption_helpers
 */

const crypto = require('crypto');
const qudag = require('@qudag/napi-core');

/**
 * Encryption key manager for ReasoningBank
 */
class EncryptionKeyManager {
  constructor() {
    this.keys = new Map();
  }

  /**
   * Generate a new HQC-128 encryption key
   * @returns {Object} { key: base64 string (32 bytes), publicKey: base64 string }
   */
  generateKey() {
    // Generate 32 bytes for HQC-128
    const keyBytes = crypto.randomBytes(32);
    const key = keyBytes.toString('base64');

    // For HQC, we'll use the key as seed for key generation
    // In production, this shoul