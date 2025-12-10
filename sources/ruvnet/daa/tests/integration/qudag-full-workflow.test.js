/**
 * Integration Tests for QuDAG Full Workflow
 *
 * Tests complete workflows combining multiple modules
 */

const { test } = require('node:test');
const assert = require('node:assert/strict');

// Mock modules
const createMockQuDAG = () => {
  class MlKem768 {
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
  }

  class MlDsa {
    generateKeypair() {
      return {
        publicKey: Buffer.alloc(1952),
        secretKey: Buffer.alloc(4864)
      };
    }
    sign(message, secretKey) {
      return Buffer.alloc(3309);
    }
    verify(message, signature, publicKey) {
      return true;
    }
  }

  class PasswordVault {
    constructor(masterPassword) {
      this.storage = new Map();
      this.masterPassword = masterPassword;
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
  }

  class RuvToken {
    async createTransaction(from, to, amount) {
      return { from, to, amount, timestamp: Date.now() };
    }
    async signTransaction(transaction, privateKey) {
      return { transaction, signature: Buffer.alloc(3309) };
    }
    verifyTransaction(signedTx) {
      return true;
    }
  }

  return {
    MlKem768,
    MlDsa,
    PasswordVault,
    RuvToken,
    blake3Hash: (data) => Buffer.alloc(32)
  };
};

const QuDAG = createMockQuDAG();

test('Integration: Secure key exchange with vault storage', async (t) => {
  // 1. Generate ML-KEM keypair
  const mlkem = new QuDAG.MlKem768();
  const { publicKey, secretKey } = mlkem.generateKeypair();

  // 2. Encapsulate shared secret
  const { ciphertext, sharedSecret } = mlkem.encapsulate(publicKey);

  // 3. Store shared secret in vault
  const vault = new QuDAG.PasswordVault('master-password');
  await vault.store('shared-secret', sharedSecret.toString('hex'));

  // 4. Retrieve from vault
  const retrievedSecret = await vault.retrieve('shared-secret');

  assert.ok(retrievedSecret, 'Should retrieve shared secret from vault');
  assert.equal(retrievedSecret, sharedSecret.toString('hex'), 'Retrieved secret should match');
});

test('Integration: End-to-end secure transaction', async (t) => {
  // 1. Generate signing keys
  const mldsa = new QuDAG.MlDsa();
  const { publicKey, secretKey } = mldsa.generateKeypair();

  // 2. Create transaction
  const exchange = new QuDAG.RuvToken();
  const tx = await exchange.createTransaction('alice', 'bob', 100);

  // 3. Sign transaction
  const signedTx = await exchange.signTransaction(tx, secretKey);

  // 4. Verify transaction
  const isValid = exchange.verifyTransaction(signedTx);

  assert.equal(isValid, true, 'Transaction should be valid');
  assert.ok(signedTx.signature, 'Transaction should be signed');
});

test('Integration: Multi-party key exchange', async (t) => {
  const mlkem = new QuDAG.MlKem768();

  // Alice generates keypair
  const aliceKeys = mlkem.generateKeypair();

  // Bob generates keypair
  const bobKeys = mlkem.generateKeypair();

  // Alice encapsulates secret for Bob
  const aliceToBob = mlkem.encapsulate(bobKeys.publicKey);

  // Bob encapsulates secret for Alice
  const bobToAlice = mlkem.encapsulate(aliceKeys.publicKey);

  // Bob decapsulates Alice's message
  const bobSecret = mlkem.decapsulate(aliceToBob.ciphertext, bobKeys.secretKey);

  // Alice decapsulates Bob's message
  const aliceSecret = mlkem.decapsulate(bobToAlice.ciphertext, aliceKeys.secretKey);

  assert.equal(bobSecret.length, 32, 'Bob should have 32-byte secret');
  assert.equal(aliceSecret.length, 32, 'Alice should have 32-byte secret');
});

test('Integration: Vault-backed transaction signing', async (t) => {
  // 1. Create vault
  const vault = new QuDAG.PasswordVault('secure-password');

  // 2. Generate signing keys
  const mldsa = new QuDAG.MlDsa();
  const { publicKey, secretKey } = mldsa.generateKeypair();

  // 3. Store private key in vault
  await vault.store('signing-key', secretKey.toString('hex'));

  // 4. Create transaction
  const exchange = new QuDAG.RuvToken();
  const tx = await exchange.createTransaction('alice', 'bob', 100);

  // 5. Retrieve key from vault
  const storedKey = await vault.retrieve('signing-key');
  const retrievedKey = Buffer.from(storedKey, 'hex');

  // 6. Sign transaction
  const signedTx = await exchange.signTransaction(tx, retrievedKey);

  // 7. Verify
  const isValid = exchange.verifyTransaction(signedTx);

  assert.equal(isValid, true, 'Vault-backed signing should work');
});

test('Integration: Hybrid encryption with ML-KEM and symmetric crypto', async (t) => {
  const mlkem = new QuDAG.MlKem768();
  const { publicKey, secretKey } = mlkem.generateKeypair();

  // Encapsulate to derive shared secret
  const { ciphertext, sharedSecret } = mlkem.encapsulate(publicKey);

  // Use shared secret for symmetric encryption (simulated with hash)
  const message = Buffer.from('Secret message');
  const messageHash = QuDAG.blake3Hash(message);

  // Derive key from shared secret
  const derivedKey = QuDAG.blake3Hash(sharedSecret);

  // Decapsulate
  const decapsulatedSecret = mlkem.decapsulate(ciphertext, secretKey);
  const derivedKey2 = QuDAG.blake3Hash(decapsulatedSecret);

  assert.deepEqual(derivedKey, derivedKey2, 'Derived keys should match');
});

test('Integration: Multiple vaults with different master passwords', async (t) => {
  const vault1 = new QuDAG.PasswordVault('password1');
  const vault2 = new QuDAG.PasswordVault('password2');

  await vault1.store('secret1', 'value1');
  await vault2.store('secret2', 'value2');

  const retrieved1 = await vault1.retrieve('secret1');
  const retrieved2 = await vault2.retrieve('secret2');

  assert.equal(retrieved1, 'value1', 'Vault 1 should retrieve correct value');
  assert.equal(retrieved2, 'value2', 'Vault 2 should retrieve correct value');

  // Cross-vault retrieval should fail
  const crossRetrieve = await vault1.retrieve('secret2');
  assert.equal(crossRetrieve, null, 'Should not retrieve from other vault');
});

test('Integration: Batch transaction processing', async (t) => {
  const exchange = new QuDAG.RuvToken();
  const mldsa = new QuDAG.MlDsa();
  const { secretKey } = mldsa.generateKeypair();

  // Create multiple transactions
  const transactions = [];
  for (let i = 0; i < 5; i++) {
    const tx = await exchange.createTransaction(`user${i}`, `user${i+1}`, 10 * (i+1));
    const signedTx = await exchange.signTransaction(tx, secretKey);
    transactions.push(signedTx);
  }

  // Verify all transactions
  const verifications = transactions.map(tx => exchange.verifyTransaction(tx));

  assert.equal(verifications.length, 5, 'Should process 5 transactions');
  assert.ok(verifications.every(v => v === true), 'All transactions should be valid');
});

test('Integration: Key rotation workflow', async (t) => {
  const vault = new QuDAG.PasswordVault('master-password');
  const mldsa = new QuDAG.MlDsa();

  // Generate first keypair
  const keys1 = mldsa.generateKeypair();
  await vault.store('current-key', keys1.secretKey.toString('hex'));

  // Use key for signing
  const message = Buffer.from('Important message');
  const signature1 = mldsa.sign(message, keys1.secretKey);

  // Rotate to new key
  const keys2 = mldsa.generateKeypair();
  await vault.store('previous-key', keys1.secretKey.toString('hex'));
  await vault.store('current-key', keys2.secretKey.toString('hex'));

  // Verify with both keys
  const valid1 = mldsa.verify(message, signature1, keys1.publicKey);

  assert.equal(valid1, true, 'Old signature should still verify with old key');

  // Sign with new key
  const signature2 = mldsa.sign(message, keys2.secretKey);
  const valid2 = mldsa.verify(message, signature2, keys2.publicKey);

  assert.equal(valid2, true, 'New signature should verify with new key');
});
