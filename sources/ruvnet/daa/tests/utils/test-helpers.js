/**
 * Test Utilities and Helpers
 *
 * Common utilities for testing NAPI bindings and SDK
 */

const crypto = require('crypto');

/**
 * Generate random buffer of specified length
 */
function randomBuffer(length) {
  return crypto.randomBytes(length);
}

/**
 * Create mock ML-KEM-768 keypair
 */
function createMockKeypair() {
  return {
    publicKey: Buffer.alloc(1184),
    secretKey: Buffer.alloc(2400)
  };
}

/**
 * Create mock ML-DSA signature
 */
function createMockSignature() {
  return Buffer.alloc(3309);
}

/**
 * Create mock transaction
 */
function createMockTransaction(from, to, amount) {
  return {
    from,
    to,
    amount,
    timestamp: Date.now()
  };
}

/**
 * Measure execution time of async function
 */
async function measureTime(fn) {
  const start = process.hrtime.bigint();
  const result = await fn();
  const end = process.hrtime.bigint();
  const timeMs = Number(end - start) / 1_000_000;

  return { result, timeMs };
}

/**
 * Measure execution time of sync function
 */
function measureTimeSync(fn) {
  const start = process.hrtime.bigint();
  const result = fn();
  const end = process.hrtime.bigint();
  const timeMs = Number(end - start) / 1_000_000;

  return { result, timeMs };
}

/**
 * Run function multiple times and collect statistics
 */
function benchmark(fn, iterations = 100) {
  const times = [];

  for (let i = 0; i < iterations; i++) {
    const { timeMs } = measureTimeSync(fn);
    times.push(timeMs);
  }

  times.sort((a, b) => a - b);

  return {
    iterations,
    min: times[0],
    max: times[times.length - 1],
    avg: times.reduce((a, b) => a + b, 0) / times.length,
    median: times[Math.floor(times.length / 2)],
    p95: times[Math.floor(times.length * 0.95)],
    p99: times[Math.floor(times.length * 0.99)]
  };
}

/**
 * Compare two buffers
 */
function buffersEqual(buf1, buf2) {
  if (buf1.length !== buf2.length) return false;
  return buf1.equals(buf2);
}

/**
 * Assert buffer length
 */
function assertBufferLength(buffer, expectedLength, message) {
  if (buffer.length !== expectedLength) {
    throw new Error(
      message || `Buffer length mismatch: expected ${expectedLength}, got ${buffer.length}`
    );
  }
}

/**
 * Create mock platform detection
 */
function createMockPlatform(platformType = 'native') {
  return {
    detectPlatform: () => platformType,
    isNodeJs: () => platformType === 'native',
    isBrowser: () => platformType === 'wasm'
  };
}

/**
 * Create mock QuDAG instance
 */
function createMockQuDAG() {
  return {
    MlKem768: class {
      generateKeypair() {
        return createMockKeypair();
      }
      encapsulate(publicKey) {
        return {
          ciphertext: Buffer.alloc(1088),
          sharedSecret: Buffer.alloc(32)
        };
      }
      decapsulate(ciphertext, secretKey) {
        return Buffer.alloc(32);
      }
    },
    MlDsa: class {
      sign(message, secretKey) {
        return createMockSignature();
      }
      verify(message, signature, publicKey) {
        return true;
      }
    },
    blake3Hash: (data) => Buffer.alloc(32),
    blake3HashHex: (data) => '0'.repeat(64),
    quantumFingerprint: (data) => 'qf:' + '0'.repeat(64)
  };
}

/**
 * Create mock PasswordVault
 */
function createMockVault(masterPassword = 'default') {
  return class PasswordVault {
    constructor(password) {
      this.storage = new Map();
      this.masterPassword = password || masterPassword;
    }
    unlock(password) {
      return password === this.masterPassword;
    }
    async store(key, value) {
      this.storage.set(key, value);
    }
    async retrieve(key) {
      return this.storage.get(key) || null;
    }
    async delete(key) {
      return this.storage.delete(key);
    }
    async list() {
      return Array.from(this.storage.keys());
    }
  };
}

/**
 * Create mock RuvToken
 */
function createMockExchange() {
  return class RuvToken {
    async createTransaction(from, to, amount) {
      return createMockTransaction(from, to, amount);
    }
    async signTransaction(transaction, privateKey) {
      return {
        transaction,
        signature: createMockSignature()
      };
    }
    verifyTransaction(signedTx) {
      return signedTx && signedTx.signature && signedTx.signature.length === 3309;
    }
    async submitTransaction(signedTx) {
      return 'tx_' + randomBuffer(16).toString('hex');
    }
  };
}

/**
 * Wait for specified milliseconds
 */
function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

/**
 * Retry function with exponential backoff
 */
async function retry(fn, maxAttempts = 3, delayMs = 100) {
  let lastError;

  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error;
      if (attempt < maxAttempts) {
        await sleep(delayMs * Math.pow(2, attempt - 1));
      }
    }
  }

  throw lastError;
}

/**
 * Create test data of specified size
 */
function createTestData(sizeBytes) {
  return Buffer.alloc(sizeBytes);
}

/**
 * Format bytes to human-readable string
 */
function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(2)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(2)} MB`;
}

/**
 * Format time in milliseconds
 */
function formatTime(ms) {
  if (ms < 1) return `${(ms * 1000).toFixed(2)} μs`;
  if (ms < 1000) return `${ms.toFixed(3)} ms`;
  return `${(ms / 1000).toFixed(2)} s`;
}

/**
 * Calculate throughput (bytes per second)
 */
function calculateThroughput(bytes, timeMs) {
  const bytesPerSecond = (bytes / timeMs) * 1000;
  return formatBytes(bytesPerSecond) + '/s';
}

module.exports = {
  randomBuffer,
  createMockKeypair,
  createMockSignature,
  createMockTransaction,
  measureTime,
  measureTimeSync,
  benchmark,
  buffersEqual,
  assertBufferLength,
  createMockPlatform,
  createMockQuDAG,
  createMockVault,
  createMockExchange,
  sleep,
  retry,
  createTestData,
  formatBytes,
  formatTime,
  calculateThroughput
};
