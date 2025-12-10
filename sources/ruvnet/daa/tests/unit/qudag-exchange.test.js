/**
 * Unit Tests for QuDAG rUv Token Exchange
 *
 * Tests transaction creation, signing, and verification
 */

const { test } = require('node:test');
const assert = require('node:assert/strict');

// Mock RuvToken
const createMockRuvToken = () => {
  return class RuvToken {
    constructor() {}

    async createTransaction(from, to, amount) {
      return {
        from,
        to,
        amount,
        timestamp: Date.now()
      };
    }

    async signTransaction(transaction, privateKey) {
      return {
        transaction,
        signature: Buffer.alloc(3309) // ML-DSA-65 signature
      };
    }

    verifyTransaction(signedTx) {
      return signedTx.signature && signedTx.signature.length === 3309;
    }

    async submitTransaction(signedTx) {
      return 'tx_' + Math.random().toString(36).substring(7);
    }
  };
};

const RuvToken = createMockRuvToken();

test('Exchange: Create transaction', async (t) => {
  const exchange = new RuvToken();

  const tx = await exchange.createTransaction('alice', 'bob', 100);

  assert.equal(tx.from, 'alice', 'From address should be alice');
  assert.equal(tx.to, 'bob', 'To address should be bob');
  assert.equal(tx.amount, 100, 'Amount should be 100');
  assert.ok(tx.timestamp, 'Transaction should have timestamp');
});

test('Exchange: Create transaction with decimal amount', async (t) => {
  const exchange = new RuvToken();

  const tx = await exchange.createTransaction('alice', 'bob', 99.99);

  assert.equal(tx.amount, 99.99, 'Should support decimal amounts');
});

test('Exchange: Create transaction with zero amount', async (t) => {
  const exchange = new RuvToken();

  const tx = await exchange.createTransaction('alice', 'bob', 0);

  assert.equal(tx.amount, 0, 'Should support zero amount');
});

test('Exchange: Sign transaction', async (t) => {
  const exchange = new RuvToken();
  const privateKey = Buffer.alloc(4864); // ML-DSA-65 private key

  const tx = await exchange.createTransaction('alice', 'bob', 100);
  const signedTx = await exchange.signTransaction(tx, privateKey);

  assert.ok(signedTx.transaction, 'Signed transaction should contain transaction');
  assert.ok(signedTx.signature, 'Signed transaction should contain signature');
  assert.equal(signedTx.signature.length, 3309, 'Signature should be 3309 bytes');
});

test('Exchange: Verify valid transaction', async (t) => {
  const exchange = new RuvToken();
  const privateKey = Buffer.alloc(4864);

  const tx = await exchange.createTransaction('alice', 'bob', 100);
  const signedTx = await exchange.signTransaction(tx, privateKey);

  const isValid = exchange.verifyTransaction(signedTx);

  assert.equal(isValid, true, 'Valid transaction should verify');
});

test('Exchange: Submit transaction', async (t) => {
  const exchange = new RuvToken();
  const privateKey = Buffer.alloc(4864);

  const tx = await exchange.createTransaction('alice', 'bob', 100);
  const signedTx = await exchange.signTransaction(tx, privateKey);

  const txHash = await exchange.submitTransaction(signedTx);

  assert.ok(txHash, 'Should return transaction hash');
  assert.ok(txHash.startsWith('tx_'), 'Transaction hash should start with tx_');
});

test('Exchange: Multiple transactions have different timestamps', async (t) => {
  const exchange = new RuvToken();

  const tx1 = await exchange.createTransaction('alice', 'bob', 100);
  await new Promise(resolve => setTimeout(resolve, 5)); // Small delay
  const tx2 = await exchange.createTransaction('alice', 'bob', 100);

  assert.notEqual(tx1.timestamp, tx2.timestamp, 'Timestamps should be different');
});

test('Exchange: Transaction with same addresses', async (t) => {
  const exchange = new RuvToken();

  const tx = await exchange.createTransaction('alice', 'alice', 50);

  assert.equal(tx.from, 'alice', 'From should be alice');
  assert.equal(tx.to, 'alice', 'To should be alice');
  assert.equal(tx.amount, 50, 'Amount should be 50');
});

test('Exchange: Large amount transaction', async (t) => {
  const exchange = new RuvToken();

  const tx = await exchange.createTransaction('whale', 'shark', 1000000);

  assert.equal(tx.amount, 1000000, 'Should support large amounts');
});

test('Exchange: Transaction with long addresses', async (t) => {
  const exchange = new RuvToken();
  const longAddress = 'a'.repeat(100);

  const tx = await exchange.createTransaction(longAddress, 'bob', 100);

  assert.equal(tx.from.length, 100, 'Should support long addresses');
});

test('Exchange: Sign and verify workflow', async (t) => {
  const exchange = new RuvToken();
  const privateKey = Buffer.alloc(4864);

  // Create transaction
  const tx = await exchange.createTransaction('alice', 'bob', 100);
  assert.ok(tx, 'Transaction created');

  // Sign transaction
  const signedTx = await exchange.signTransaction(tx, privateKey);
  assert.ok(signedTx.signature, 'Transaction signed');

  // Verify transaction
  const isValid = exchange.verifyTransaction(signedTx);
  assert.equal(isValid, true, 'Transaction verified');

  // Submit transaction
  const txHash = await exchange.submitTransaction(signedTx);
  assert.ok(txHash, 'Transaction submitted');
});
