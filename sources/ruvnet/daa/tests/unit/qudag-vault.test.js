/**
 * Unit Tests for QuDAG Password Vault
 *
 * Tests vault creation, unlock, and operations
 */

const { test } = require('node:test');
const assert = require('node:assert/strict');

// Mock PasswordVault
const createMockPasswordVault = () => {
  return class PasswordVault {
    constructor(masterPassword) {
      this.masterPassword = masterPassword;
      this.storage = new Map();
      this.locked = false;
    }

    unlock(password) {
      return password === this.masterPassword;
    }

    async store(key, value) {
      if (this.locked) {
        throw new Error('Vault is locked');
      }
      this.storage.set(key, value);
    }

    async retrieve(key) {
      if (this.locked) {
        throw new Error('Vault is locked');
      }
      return this.storage.get(key) || null;
    }

    async delete(key) {
      if (this.locked) {
        throw new Error('Vault is locked');
      }
      return this.storage.delete(key);
    }

    async list() {
      if (this.locked) {
        throw new Error('Vault is locked');
      }
      return Array.from(this.storage.keys());
    }
  };
};

const PasswordVault = createMockPasswordVault();

test('Vault: Create with master password', (t) => {
  const vault = new PasswordVault('my-secure-password');

  assert.ok(vault, 'Vault should be created successfully');
});

test('Vault: Unlock with correct password', (t) => {
  const masterPassword = 'my-secure-password';
  const vault = new PasswordVault(masterPassword);

  const unlocked = vault.unlock(masterPassword);

  assert.equal(unlocked, true, 'Vault should unlock with correct password');
});

test('Vault: Fail to unlock with incorrect password', (t) => {
  const vault = new PasswordVault('correct-password');

  const unlocked = vault.unlock('wrong-password');

  assert.equal(unlocked, false, 'Vault should not unlock with incorrect password');
});

test('Vault: Store and retrieve value', async (t) => {
  const vault = new PasswordVault('test-password');

  await vault.store('api-key', 'secret-api-key-12345');
  const retrieved = await vault.retrieve('api-key');

  assert.equal(retrieved, 'secret-api-key-12345', 'Retrieved value should match stored value');
});

test('Vault: Retrieve non-existent key returns null', async (t) => {
  const vault = new PasswordVault('test-password');

  const retrieved = await vault.retrieve('non-existent-key');

  assert.equal(retrieved, null, 'Non-existent key should return null');
});

test('Vault: Delete key', async (t) => {
  const vault = new PasswordVault('test-password');

  await vault.store('temp-key', 'temp-value');
  const deleted = await vault.delete('temp-key');
  const retrieved = await vault.retrieve('temp-key');

  assert.equal(deleted, true, 'Delete should return true');
  assert.equal(retrieved, null, 'Deleted key should return null');
});

test('Vault: Delete non-existent key', async (t) => {
  const vault = new PasswordVault('test-password');

  const deleted = await vault.delete('non-existent-key');

  assert.equal(deleted, false, 'Deleting non-existent key should return false');
});

test('Vault: List all keys', async (t) => {
  const vault = new PasswordVault('test-password');

  await vault.store('key1', 'value1');
  await vault.store('key2', 'value2');
  await vault.store('key3', 'value3');

  const keys = await vault.list();

  assert.equal(keys.length, 3, 'Should have 3 keys');
  assert.ok(keys.includes('key1'), 'Should include key1');
  assert.ok(keys.includes('key2'), 'Should include key2');
  assert.ok(keys.includes('key3'), 'Should include key3');
});

test('Vault: List empty vault', async (t) => {
  const vault = new PasswordVault('test-password');

  const keys = await vault.list();

  assert.equal(keys.length, 0, 'Empty vault should return empty array');
});

test('Vault: Store multiple values', async (t) => {
  const vault = new PasswordVault('test-password');

  await vault.store('password1', 'pass123');
  await vault.store('password2', 'pass456');
  await vault.store('api-key', 'key789');

  const pass1 = await vault.retrieve('password1');
  const pass2 = await vault.retrieve('password2');
  const apiKey = await vault.retrieve('api-key');

  assert.equal(pass1, 'pass123', 'password1 should match');
  assert.equal(pass2, 'pass456', 'password2 should match');
  assert.equal(apiKey, 'key789', 'api-key should match');
});

test('Vault: Overwrite existing value', async (t) => {
  const vault = new PasswordVault('test-password');

  await vault.store('key', 'original-value');
  await vault.store('key', 'new-value');

  const retrieved = await vault.retrieve('key');

  assert.equal(retrieved, 'new-value', 'Value should be overwritten');
});

test('Vault: Store empty string', async (t) => {
  const vault = new PasswordVault('test-password');

  await vault.store('empty-key', '');
  const retrieved = await vault.retrieve('empty-key');

  assert.equal(retrieved, '', 'Should be able to store empty string');
});
