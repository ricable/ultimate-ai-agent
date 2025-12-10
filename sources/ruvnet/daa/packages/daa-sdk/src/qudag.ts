/**
 * QuDAG Native NAPI bindings wrapper
 *
 * Provides TypeScript types and a consistent API wrapper around the
 * @daa/qudag-native NAPI-rs bindings.
 *
 * @module qudag
 */

/**
 * ML-KEM-768 Key Pair
 */
export interface KeyPair {
  /** Public key (1184 bytes for ML-KEM-768) */
  publicKey: Buffer;
  /** Secret key (2400 bytes for ML-KEM-768) */
  secretKey: Buffer;
}

/**
 * Encapsulated secret from ML-KEM
 */
export interface EncapsulatedSecret {
  /** Ciphertext to be sent to recipient */
  ciphertext: Buffer;
  /** Shared secret (32 bytes) */
  sharedSecret: Buffer;
}

/**
 * Module information
 */
export interface ModuleInfo {
  name: string;
  version: string;
  description: string;
  features: string[];
}

/**
 * Raw NAPI bindings interface (internal)
 */
interface QuDAGNative {
  init(): string;
  version(): string;
  getModuleInfo(): {
    name: string;
    version: string;
    description: string;
    features: string[];
  };

  // ML-KEM-768
  MlKem768: new () => {
    generateKeypair(): { public_key: Buffer; secret_key: Buffer };
    encapsulate(publicKey: Buffer): { ciphertext: Buffer; shared_secret: Buffer };
    decapsulate(ciphertext: Buffer, secretKey: Buffer): Buffer;
  };

  // ML-DSA
  MlDsa: new () => {
    sign(message: Buffer, secretKey: Buffer): Buffer;
    verify(message: Buffer, signature: Buffer, publicKey: Buffer): boolean;
  };

  // BLAKE3
  blake3Hash(data: Buffer): Buffer;
  blake3HashHex(data: Buffer): string;
  quantumFingerprint(data: Buffer): string;
}

let native: QuDAGNative | null = null;

/**
 * Load the native NAPI bindings
 */
export function loadNative(): QuDAGNative {
  if (native !== null) {
    return native;
  }

  try {
    native = require('@daa/qudag-native') as QuDAGNative;
    return native;
  } catch (error) {
    throw new Error(`Failed to load @daa/qudag-native: ${error}`);
  }
}

/**
 * Check if native bindings are available
 */
export function isNativeAvailable(): boolean {
  try {
    require.resolve('@daa/qudag-native');
    return true;
  } catch {
    return false;
  }
}

/**
 * ML-KEM-768 Key Encapsulation Mechanism
 *
 * Wrapper around the native NAPI bindings with a consistent TypeScript API.
 *
 * @example
 * ```typescript
 * const mlkem = new MlKem768();
 *
 * // Alice generates keypair
 * const { publicKey, secretKey } = mlkem.generateKeypair();
 *
 * // Bob encapsulates a secret
 * const { ciphertext, sharedSecret } = mlkem.encapsulate(publicKey);
 *
 * // Alice decapsulates
 * const aliceSecret = mlkem.decapsulate(ciphertext, secretKey);
 *
 * // Both parties now share the same secret
 * assert(sharedSecret.equals(aliceSecret));
 * ```
 */
export class MlKem768 {
  private instance: any;

  constructor() {
    const nativeModule = loadNative();
    this.instance = new nativeModule.MlKem768();
  }

  /**
   * Generate a new ML-KEM-768 keypair
   *
   * @returns KeyPair with public key (1184 bytes) and secret key (2400 bytes)
   */
  generateKeypair(): KeyPair {
    const result = this.instance.generateKeypair();
    return {
      publicKey: result.public_key,
      secretKey: result.secret_key,
    };
  }

  /**
   * Encapsulate a shared secret using a public key
   *
   * @param publicKey - Recipient's public key (1184 bytes)
   * @returns EncapsulatedSecret with ciphertext and shared secret
   */
  encapsulate(publicKey: Buffer): EncapsulatedSecret {
    const result = this.instance.encapsulate(publicKey);
    return {
      ciphertext: result.ciphertext,
      sharedSecret: result.shared_secret,
    };
  }

  /**
   * Decapsulate a shared secret using a secret key
   *
   * @param ciphertext - Encapsulated ciphertext (1088 bytes)
   * @param secretKey - Recipient's secret key (2400 bytes)
   * @returns Shared secret (32 bytes)
   */
  decapsulate(ciphertext: Buffer, secretKey: Buffer): Buffer {
    return this.instance.decapsulate(ciphertext, secretKey);
  }
}

/**
 * ML-DSA Digital Signature Algorithm
 *
 * Wrapper around the native NAPI bindings for quantum-resistant signatures.
 *
 * @example
 * ```typescript
 * const mldsa = new MlDsa();
 *
 * const message = Buffer.from('Hello, QuDAG!');
 * const secretKey = Buffer.alloc(4032); // ML-DSA-65 secret key
 *
 * // Sign the message
 * const signature = mldsa.sign(message, secretKey);
 *
 * // Verify the signature
 * const publicKey = Buffer.alloc(1952); // ML-DSA-65 public key
 * const isValid = mldsa.verify(message, signature, publicKey);
 * ```
 */
export class MlDsa {
  private instance: any;

  constructor() {
    const nativeModule = loadNative();
    this.instance = new nativeModule.MlDsa();
  }

  /**
   * Sign a message with a secret key
   *
   * @param message - Message to sign
   * @param secretKey - Signer's secret key
   * @returns Digital signature
   */
  sign(message: Buffer, secretKey: Buffer): Buffer {
    return this.instance.sign(message, secretKey);
  }

  /**
   * Verify a signature with a public key
   *
   * @param message - Original message
   * @param signature - Digital signature to verify
   * @param publicKey - Signer's public key
   * @returns true if signature is valid, false otherwise
   */
  verify(message: Buffer, signature: Buffer, publicKey: Buffer): boolean {
    return this.instance.verify(message, signature, publicKey);
  }
}

/**
 * BLAKE3 Cryptographic Hash Functions
 */
export class Blake3 {
  /**
   * Hash data using BLAKE3
   *
   * @param data - Data to hash
   * @returns 32-byte hash
   */
  static hash(data: Buffer): Buffer {
    const nativeModule = loadNative();
    return nativeModule.blake3Hash(data);
  }

  /**
   * Hash data using BLAKE3 and return as hex string
   *
   * @param data - Data to hash
   * @returns Hex-encoded hash
   */
  static hashHex(data: Buffer): string {
    const nativeModule = loadNative();
    return nativeModule.blake3HashHex(data);
  }

  /**
   * Generate quantum-resistant fingerprint
   *
   * @param data - Data to fingerprint
   * @returns Quantum fingerprint string (format: "qf:<hex>")
   */
  static quantumFingerprint(data: Buffer): string {
    const nativeModule = loadNative();
    return nativeModule.quantumFingerprint(data);
  }
}

/**
 * Initialize the QuDAG native module
 *
 * @returns Initialization message
 */
export function init(): string {
  const nativeModule = loadNative();
  return nativeModule.init();
}

/**
 * Get module version
 *
 * @returns Version string
 */
export function version(): string {
  const nativeModule = loadNative();
  return nativeModule.version();
}

/**
 * Get detailed module information
 *
 * @returns Module information object
 */
export function getModuleInfo(): ModuleInfo {
  const nativeModule = loadNative();
  return nativeModule.getModuleInfo();
}

/**
 * Convenience exports for direct access to crypto operations
 */
export const Crypto = {
  MlKem768,
  MlDsa,
  Blake3,
};

/**
 * Default export for convenience
 */
export default {
  init,
  version,
  getModuleInfo,
  isNativeAvailable,
  MlKem768,
  MlDsa,
  Blake3,
  Crypto,
};
