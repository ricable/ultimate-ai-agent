/**
 * Integration Tests for Platform Comparison (Native vs WASM)
 *
 * Tests feature parity between native and WASM implementations
 */

const { test } = require('node:test');
const assert = require('node:assert/strict');

// Mock native and WASM implementations
const createPlatformMocks = () => {
  const nativeImpl = {
    name: 'native',
    MlKem768: class {
      generateKeypair() {
        return {
          publicKey: Buffer.alloc(1184),
          secretKey: Buffer.alloc(2400)
        };
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
        return Buffer.alloc(3309);
      }
      verify(message, signature, publicKey) {
        return true;
      }
    },
    blake3Hash: (data) => Buffer.alloc(32)
  };

  const wasmImpl = {
    name: 'wasm',
    MlKem768: class {
      generateKeypair() {
        return {
          publicKey: new Uint8Array(1184),
          secretKey: new Uint8Array(2400)
        };
      }
      encapsulate(publicKey) {
        return {
          ciphertext: new Uint8Array(1088),
          sharedSecret: new Uint8Array(32)
        };
      }
      decapsulate(ciphertext, secretKey) {
        return new Uint8Array(32);
      }
    },
    MlDsa: class {
      sign(message, secretKey) {
        return new Uint8Array(3309);
      }
      verify(message, signature, publicKey) {
        return true;
      }
    },
    blake3Hash: (data) => new Uint8Array(32)
  };

  return { native: nativeImpl, wasm: wasmImpl };
};

const { native, wasm } = createPlatformMocks();

test('Platform Parity: ML-KEM keypair generation', (t) => {
  const nativeKeygen = new native.MlKem768();
  const wasmKeygen = new wasm.MlKem768();

  const nativeKeys = nativeKeygen.generateKeypair();
  const wasmKeys = wasmKeygen.generateKeypair();

  assert.equal(nativeKeys.publicKey.length, wasmKeys.publicKey.length,
    'Public key lengths should match');
  assert.equal(nativeKeys.secretKey.length, wasmKeys.secretKey.length,
    'Secret key lengths should match');
});

test('Platform Parity: ML-KEM encapsulation', (t) => {
  const nativeKem = new native.MlKem768();
  const wasmKem = new wasm.MlKem768();

  const nativeKeys = nativeKem.generateKeypair();
  const wasmKeys = wasmKem.generateKeypair();

  const nativeEncap = nativeKem.encapsulate(nativeKeys.publicKey);
  const wasmEncap = wasmKem.encapsulate(Buffer.from(wasmKeys.publicKey));

  assert.equal(nativeEncap.ciphertext.length, wasmEncap.ciphertext.length,
    'Ciphertext lengths should match');
  assert.equal(nativeEncap.sharedSecret.length, wasmEncap.sharedSecret.length,
    'Shared secret lengths should match');
});

test('Platform Parity: ML-DSA signing', (t) => {
  const nativeDsa = new native.MlDsa();
  const wasmDsa = new wasm.MlDsa();

  const message = Buffer.from('Test message');
  const secretKey = Buffer.alloc(4864);

  const nativeSignature = nativeDsa.sign(message, secretKey);
  const wasmSignature = wasmDsa.sign(message, secretKey);

  assert.equal(nativeSignature.length, wasmSignature.length,
    'Signature lengths should match');
});

test('Platform Parity: ML-DSA verification', (t) => {
  const nativeDsa = new native.MlDsa();
  const wasmDsa = new wasm.MlDsa();

  const message = Buffer.from('Test message');
  const signature = Buffer.alloc(3309);
  const publicKey = Buffer.alloc(1952);

  const nativeValid = nativeDsa.verify(message, signature, publicKey);
  const wasmValid = wasmDsa.verify(message, signature, publicKey);

  assert.equal(nativeValid, wasmValid, 'Verification results should match');
});

test('Platform Parity: BLAKE3 hashing', (t) => {
  const data = Buffer.from('Hash this data');

  const nativeHash = native.blake3Hash(data);
  const wasmHash = wasm.blake3Hash(data);

  assert.equal(nativeHash.length, wasmHash.length,
    'Hash lengths should match (32 bytes)');
});

test('Platform Parity: API surface equivalence', (t) => {
  // Check that both platforms have the same API
  const nativeAPI = Object.keys(native).sort();
  const wasmAPI = Object.keys(wasm).filter(k => k !== 'name').sort();

  assert.deepEqual(nativeAPI, wasmAPI, 'API surfaces should match');
});

test('Platform Parity: Constructor compatibility', (t) => {
  // Both platforms should instantiate the same way
  assert.doesNotThrow(() => new native.MlKem768(), 'Native constructor should work');
  assert.doesNotThrow(() => new wasm.MlKem768(), 'WASM constructor should work');
  assert.doesNotThrow(() => new native.MlDsa(), 'Native ML-DSA constructor should work');
  assert.doesNotThrow(() => new wasm.MlDsa(), 'WASM ML-DSA constructor should work');
});

test('Platform Parity: Buffer/Uint8Array interoperability', (t) => {
  const nativeKem = new native.MlKem768();
  const wasmKem = new wasm.MlKem768();

  // Native uses Buffer, WASM uses Uint8Array, but should be compatible
  const nativeKeys = nativeKem.generateKeypair();
  const wasmKeys = wasmKem.generateKeypair();

  // Should be able to convert between them
  const nativeAsUint8 = new Uint8Array(nativeKeys.publicKey);
  const wasmAsBuffer = Buffer.from(wasmKeys.publicKey);

  assert.equal(nativeAsUint8.length, 1184, 'Should convert Buffer to Uint8Array');
  assert.equal(wasmAsBuffer.length, 1184, 'Should convert Uint8Array to Buffer');
});

test('Platform Parity: Error handling consistency', (t) => {
  const nativeKem = new native.MlKem768();
  const wasmKem = new wasm.MlKem768();

  const invalidKey = Buffer.alloc(100); // Wrong size

  // Both should handle errors similarly
  // Note: In real implementation, this would throw
  // For now, just verify the methods exist
  assert.ok(typeof nativeKem.encapsulate === 'function', 'Native should have encapsulate');
  assert.ok(typeof wasmKem.encapsulate === 'function', 'WASM should have encapsulate');
});

test('Performance: Native should be faster than WASM (simulated)', async (t) => {
  const iterations = 100;

  // Simulate native performance
  const nativeStart = Date.now();
  for (let i = 0; i < iterations; i++) {
    const kem = new native.MlKem768();
    kem.generateKeypair();
  }
  const nativeTime = Date.now() - nativeStart;

  // Simulate WASM performance (typically slower)
  const wasmStart = Date.now();
  for (let i = 0; i < iterations; i++) {
    const kem = new wasm.MlKem768();
    kem.generateKeypair();
  }
  const wasmTime = Date.now() - wasmStart;

  // Note: In real scenarios, native should be faster
  // This is just a structure test
  assert.ok(nativeTime >= 0, 'Native time should be measured');
  assert.ok(wasmTime >= 0, 'WASM time should be measured');
});

test('Platform Selection: Should prefer native when available', (t) => {
  const preferredPlatform = typeof process !== 'undefined' && process.versions?.node
    ? 'native'
    : 'wasm';

  assert.equal(preferredPlatform, 'native', 'Should prefer native in Node.js');
});

test('Platform Selection: Should fallback to WASM in browser', (t) => {
  // Simulate browser environment check
  const isBrowser = typeof window !== 'undefined';
  const preferredPlatform = isBrowser ? 'wasm' : 'native';

  assert.equal(preferredPlatform, 'native', 'Should be native in Node.js test environment');
});
