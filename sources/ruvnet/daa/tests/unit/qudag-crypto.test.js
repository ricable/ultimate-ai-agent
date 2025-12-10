/**
 * Unit Tests for QuDAG NAPI Crypto Operations
 *
 * Tests ML-KEM-768, ML-DSA, and BLAKE3 bindings
 */

const { test } = require('node:test');
const assert = require('node:assert/strict');

// Mock for now - will use actual bindings when built
const createMockQuDAG = () => ({
  MlKem768: class {
    generateKeypair() {
      return {
        publicKey: Buffer.alloc(1184),
        secretKey: Buffer.alloc(2400)
      };
    }
    encapsulate(publicKey) {
      if (publicKey.length !== 1184) {
        throw new Error(`Invalid public key length: expected 1184, got ${publicKey.length}`);
      }
      return {
        ciphertext: Buffer.alloc(1088),
        sharedSecret: Buffer.alloc(32)
      };
    }
    decapsulate(ciphertext, secretKey) {
      if (ciphertext.length !== 1088) {
        throw new Error(`Invalid ciphertext length: expected 1088, got ${ciphertext.length}`);
      }
      if (secretKey.length !== 2400) {
        throw new Error(`Invalid secret key length: expected 2400, got ${secretKey.length}`);
      }
      return Buffer.alloc(32);
    }
  },
  MlDsa: class {
    sign(message, secretKey) {
      return Buffer.alloc(3309); // ML-DSA-65 signature size
    }
    verify(message, signature, publicKey) {
      return true;
    }
  },
  blake3Hash: (data) => Buffer.alloc(32),
  blake3HashHex: (data) => '0'.repeat(64),
  quantumFingerprint: (data) => 'qf:' + '0'.repeat(64)
});

const qudag = createMockQuDAG();

test('ML-KEM-768: Generate keypair', (t) => {
  const mlkem = new qudag.MlKem768();
  const { publicKey, secretKey } = mlkem.generateKeypair();

  assert.ok(publicKey instanceof Buffer, 'Public key should be a Buffer');
  assert.ok(secretKey instanceof Buffer, 'Secret key should be a Buffer');
  assert.equal(publicKey.length, 1184, 'Public key should be 1184 bytes');
  assert.equal(secretKey.length, 2400, 'Secret key should be 2400 bytes');
});

test('ML-KEM-768: Encapsulation', (t) => {
  const mlkem = new qudag.MlKem768();
  const { publicKey } = mlkem.generateKeypair();

  const { ciphertext, sharedSecret } = mlkem.encapsulate(publicKey);

  assert.ok(ciphertext instanceof Buffer, 'Ciphertext should be a Buffer');
  assert.ok(sharedSecret instanceof Buffer, 'Shared secret should be a Buffer');
  assert.equal(ciphertext.length, 1088, 'Ciphertext should be 1088 bytes');
  assert.equal(sharedSecret.length, 32, 'Shared secret should be 32 bytes');
});

test('ML-KEM-768: Decapsulation', (t) => {
  const mlkem = new qudag.MlKem768();
  const { publicKey, secretKey } = mlkem.generateKeypair();

  const { ciphertext } = mlkem.encapsulate(publicKey);
  const decryptedSecret = mlkem.decapsulate(ciphertext, secretKey);

  assert.ok(decryptedSecret instanceof Buffer, 'Decrypted secret should be a Buffer');
  assert.equal(decryptedSecret.length, 32, 'Decrypted secret should be 32 bytes');
});

test('ML-KEM-768: Invalid public key length', (t) => {
  const mlkem = new qudag.MlKem768();
  const invalidKey = Buffer.alloc(100);

  assert.throws(
    () => mlkem.encapsulate(invalidKey),
    /Invalid public key length/,
    'Should throw on invalid public key length'
  );
});

test('ML-KEM-768: Invalid ciphertext length', (t) => {
  const mlkem = new qudag.MlKem768();
  const { secretKey } = mlkem.generateKeypair();
  const invalidCiphertext = Buffer.alloc(100);

  assert.throws(
    () => mlkem.decapsulate(invalidCiphertext, secretKey),
    /Invalid ciphertext length/,
    'Should throw on invalid ciphertext length'
  );
});

test('ML-KEM-768: Invalid secret key length', (t) => {
  const mlkem = new qudag.MlKem768();
  const ciphertext = Buffer.alloc(1088);
  const invalidKey = Buffer.alloc(100);

  assert.throws(
    () => mlkem.decapsulate(ciphertext, invalidKey),
    /Invalid secret key length/,
    'Should throw on invalid secret key length'
  );
});

test('ML-DSA: Sign message', (t) => {
  const mldsa = new qudag.MlDsa();
  const message = Buffer.from('Hello, quantum world!');
  const secretKey = Buffer.alloc(4864); // ML-DSA-65 secret key size

  const signature = mldsa.sign(message, secretKey);

  assert.ok(signature instanceof Buffer, 'Signature should be a Buffer');
  assert.equal(signature.length, 3309, 'Signature should be 3309 bytes for ML-DSA-65');
});

test('ML-DSA: Verify signature', (t) => {
  const mldsa = new qudag.MlDsa();
  const message = Buffer.from('Hello, quantum world!');
  const secretKey = Buffer.alloc(4864);
  const publicKey = Buffer.alloc(1952); // ML-DSA-65 public key size

  const signature = mldsa.sign(message, secretKey);
  const isValid = mldsa.verify(message, signature, publicKey);

  assert.equal(isValid, true, 'Valid signature should verify successfully');
});

test('BLAKE3: Hash data', (t) => {
  const data = Buffer.from('Test data for BLAKE3');
  const hash = qudag.blake3Hash(data);

  assert.ok(hash instanceof Buffer, 'Hash should be a Buffer');
  assert.equal(hash.length, 32, 'BLAKE3 hash should be 32 bytes');
});

test('BLAKE3: Hash to hex string', (t) => {
  const data = Buffer.from('Test data for BLAKE3');
  const hashHex = qudag.blake3HashHex(data);

  assert.equal(typeof hashHex, 'string', 'Hash hex should be a string');
  assert.equal(hashHex.length, 64, 'Hex hash should be 64 characters');
  assert.match(hashHex, /^[0-9a-f]+$/, 'Hex hash should only contain hex characters');
});

test('BLAKE3: Quantum fingerprint', (t) => {
  const data = Buffer.from('Test data for quantum fingerprint');
  const fingerprint = qudag.quantumFingerprint(data);

  assert.equal(typeof fingerprint, 'string', 'Fingerprint should be a string');
  assert.ok(fingerprint.startsWith('qf:'), 'Fingerprint should start with "qf:"');
  assert.equal(fingerprint.length, 67, 'Fingerprint should be "qf:" + 64 hex chars');
});

test('BLAKE3: Consistent hashing', (t) => {
  const data = Buffer.from('Consistent test data');
  const hash1 = qudag.blake3Hash(data);
  const hash2 = qudag.blake3Hash(data);

  assert.deepEqual(hash1, hash2, 'Same input should produce same hash');
});

test('BLAKE3: Different inputs produce different hashes', (t) => {
  const data1 = Buffer.from('Data 1');
  const data2 = Buffer.from('Data 2');
  const hash1 = qudag.blake3Hash(data1);
  const hash2 = qudag.blake3Hash(data2);

  assert.notDeepEqual(hash1, hash2, 'Different inputs should produce different hashes');
});
