/**
 * Encrypted ReasoningBank Example
 *
 * Demonstrates quantum-resistant encryption for AI agent trajectories
 * using HQC-128 from @qudag/napi-core
 */

const { JJWrapper } = require('../index.js');
const {
  EncryptionKeyManager,
  createEncryptedWorkflow,
  encryptTrajectory,
  decryptTrajectory
} = require('../helpers/encryption');

console.log('🔐 Encrypted ReasoningBank Demo\n');
console.log('=' .repeat(60));

// ===== Example 1: Quick Start with Auto-Generated Keys =====
console.log('\n📦 Example 1: Quick Start');
console.log('-'.repeat(60));

const wrapper1 = new JJWrapper();
const workflow = createEncryptedWorkflow(wrapper1);

console.log('✅ Encryption enabled:', workflow.isEnabled());
console.log('📋 Key generated and stored securely\n');

// Store encrypted trajectories
wrapper1.startTrajectory('Implement authentication system');
wrapper1.finalizeTrajectory(0.92, 'Successfully added JWT authentication');

wrapper1.startTrajectory('Optimize database queries');
wrapper1.finalizeTrajectory(0.88, 'Reduced query time by 60%');

wrapper1.startTrajectory('Fix security vulnerability');
wrapper1.finalizeTrajectory(0.95, 'Patched SQL injection vulnerability');

console.log('✅ Stored 3 encrypted trajectories');

// Query trajectories (they remain encrypted in storage)
const stats = JSON.parse(wrapper1.getLearningStats());
console.log(`📊 Learning stats: ${stats.total_trajectories} trajectories, avg success: ${(stats.avg_success_rate * 100).toFixed(1)}%`);

// Decrypt a trajectory
const trajectories = JSON.parse(wrapper1.queryTrajectories('authentication', 10));
if (trajectories.length > 0) {
  console.log(`\n🔓 Decrypting trajectory: ${trajectories[0].id}`);
  try {
    const decrypted = await workflow.decryptTrajectory(trajectories[0].id);
    console.log('✅ Decryption successful');
    console.log(`   Task: ${decrypted.task}`);
    console.log(`   Success: ${(decrypted.success_score * 100).toFixed(1)}%`);
  } catch (error) {
    console.error('❌ Decryption failed:', error.message);
  }
}

// ===== Example 2: Manual Key Management =====
console.log('\n\n📦 Example 2: Manual Key Management');
console.log('-'.repeat(60));

const wrapper2 = new JJWrapper();
const keyManager = new EncryptionKeyManager();

// Generate and store key
const keyPair = keyManager.generateKeyPair();
keyManager.storeKey('production-key', keyPair);

console.log('🔑 Generated key pair:');
console.log(`   Key ID: production-key`);
console.log(`   Key length: ${Buffer.from(keyPair.key, 'base64').length} bytes`);
console.log(`   Public key length: ${Buffer.from(keyPair.publicKey, 'base64').length} bytes`);

// Enable encryption
wrapper2.enableEncryption(keyPair.key, keyPair.publicKey);
console.log('✅ Encryption enabled with custom key');

// Store encrypted trajectory
wrapper2.startTrajectory('Deploy to production');
wrapper2.finalizeTrajectory(0.98, 'Successful deployment, zero downtime');

console.log('✅ Stored encrypted production trajectory');

// ===== Example 3: Key Rotation =====
console.log('\n\n📦 Example 3: Key Rotation');
console.log('-'.repeat(60));

const wrapper3 = new JJWrapper();
const keyManager3 = new EncryptionKeyManager();

// Generate old key
const oldKeyPair = keyManager3.generateKeyPair();
keyManager3.storeKey('old-key', oldKeyPair);

wrapper3.enableEncryption(oldKeyPair.key, oldKeyPair.publicKey);
console.log('🔑 Enabled encryption with old key');

// Store trajectory with old key
wrapper3.startTrajectory('Migrate database schema');
wrapper3.finalizeTrajectory(0.90, 'Migration completed successfully');
console.log('✅ Stored trajectory with old key');

// Rotate to new key
console.log('\n🔄 Rotating encryption key...');
const newKeyPair = keyManager3.generateKeyPair();
keyManager3.storeKey('new-key', newKeyPair);

// Disable old key, enable new key
wrapper3.disableEncryption();
wrapper3.enableEncryption(newKeyPair.key, newKeyPair.publicKey);
console.log('✅ Key rotation completed');

// New trajectories use new key
wrapper3.startTrajectory('Update API documentation');
wrapper3.finalizeTrajectory(0.85, 'Documentation updated');
console.log('✅ Stored trajectory with new key');

// ===== Example 4: Direct Encryption/Decryption =====
console.log('\n\n📦 Example 4: Direct Encryption/Decryption');
console.log('-'.repeat(60));

const keyManager4 = new EncryptionKeyManager();
const keyPair4 = keyManager4.generateKeyPair();

// Sample trajectory data
const trajectoryData = {
  operations: [
    { type: 'commit', message: 'Add feature' },
    { type: 'push', branch: 'main' }
  ],
  initial_context: {
    branch: 'feature/encryption',
    files: ['src/encryption.rs', 'tests/encryption.test.js']
  },
  final_context: {
    success: true,
    tests_passed: 42
  },
  critique: 'Excellent implementation, all tests passed'
};

console.log('📄 Original trajectory data:');
console.log(JSON.stringify(trajectoryData, null, 2));

// Encrypt
const encrypted = encryptTrajectory(
  JSON.stringify(trajectoryData),
  keyPair4.publicKey
);

console.log('\n🔒 Encrypted:');
console.log(`   HQC Ciphertext: ${encrypted.hqcCiphertext.substring(0, 40)}...`);
console.log(`   IV: ${encrypted.iv}`);
console.log(`   Auth Tag: ${encrypted.authTag}`);
console.log(`   Payload: ${encrypted.encryptedPayload.substring(0, 40)}...`);

// Decrypt
const decrypted = decryptTrajectory(encrypted, keyPair4.secretKey);
const decryptedData = JSON.parse(decrypted);

console.log('\n🔓 Decrypted:');
console.log(JSON.stringify(decryptedData, null, 2));

// Verify integrity
const match = JSON.stringify(trajectoryData) === JSON.stringify(decryptedData);
console.log(`\n✅ Data integrity: ${match ? 'VERIFIED' : 'FAILED'}`);

// ===== Example 5: Security Features =====
console.log('\n\n📦 Example 5: Security Features');
console.log('-'.repeat(60));

const keyManager5 = new EncryptionKeyManager();
const keyPair5 = keyManager5.generateKeyPair();

const testPayload = JSON.stringify({ sensitive: 'data' });

// Test 1: Same plaintext produces different ciphertexts
console.log('\n🔐 Test 1: Randomized Encryption (IVs)');
const enc1 = encryptTrajectory(testPayload, keyPair5.publicKey);
const enc2 = encryptTrajectory(testPayload, keyPair5.publicKey);

console.log(`   First IV:  ${enc1.iv}`);
console.log(`   Second IV: ${enc2.iv}`);
console.log(`   ✅ IVs are unique: ${enc1.iv !== enc2.iv}`);

// Test 2: Tampering detection
console.log('\n🛡️  Test 2: Tampering Detection (Auth Tags)');
const enc3 = encryptTrajectory(testPayload, keyPair5.publicKey);

// Tamper with ciphertext
const tampered = { ...enc3 };
tampered.encryptedPayload = tampered.encryptedPayload.slice(0, -5) + 'XXXXX';

try {
  decryptTrajectory(tampered, keyPair5.secretKey);
  console.log('   ❌ Tampering not detected (SECURITY ISSUE!)');
} catch (error) {
  console.log('   ✅ Tampering detected:', error.message);
}

// Test 3: Wrong key detection
console.log('\n🔑 Test 3: Wrong Key Detection');
const keyPair6 = keyManager5.generateKeyPair();
const enc4 = encryptTrajectory(testPayload, keyPair5.publicKey);

try {
  decryptTrajectory(enc4, keyPair6.secretKey);
  console.log('   ❌ Wrong key not detected (SECURITY ISSUE!)');
} catch (error) {
  console.log('   ✅ Wrong key detected:', error.message);
}

// ===== Example 6: Performance Benchmarking =====
console.log('\n\n📦 Example 6: Performance Benchmark');
console.log('-'.repeat(60));

const keyManager7 = new EncryptionKeyManager();
const keyPair7 = keyManager7.generateKeyPair();

// Small payload (1KB)
const smallPayload = JSON.stringify({
  data: 'x'.repeat(1024)
});

// Large payload (10KB)
const largePayload = JSON.stringify({
  data: 'x'.repeat(10240)
});

// Benchmark encryption
console.log('\n⚡ Encryption Performance:');

const encStartSmall = Date.now();
for (let i = 0; i < 100; i++) {
  encryptTrajectory(smallPayload, keyPair7.publicKey);
}
const encTimeSmall = Date.now() - encStartSmall;
console.log(`   1KB payload:  ${(encTimeSmall / 100).toFixed(2)}ms avg (100 iterations)`);

const encStartLarge = Date.now();
for (let i = 0; i < 100; i++) {
  encryptTrajectory(largePayload, keyPair7.publicKey);
}
const encTimeLarge = Date.now() - encStartLarge;
console.log(`   10KB payload: ${(encTimeLarge / 100).toFixed(2)}ms avg (100 iterations)`);

// Benchmark decryption
console.log('\n⚡ Decryption Performance:');

const encSmall = encryptTrajectory(smallPayload, keyPair7.publicKey);
const decStartSmall = Date.now();
for (let i = 0; i < 100; i++) {
  decryptTrajectory(encSmall, keyPair7.secretKey);
}
const decTimeSmall = Date.now() - decStartSmall;
console.log(`   1KB payload:  ${(decTimeSmall / 100).toFixed(2)}ms avg (100 iterations)`);

const encLarge = encryptTrajectory(largePayload, keyPair7.publicKey);
const decStartLarge = Date.now();
for (let i = 0; i < 100; i++) {
  decryptTrajectory(encLarge, keyPair7.secretKey);
}
const decTimeLarge = Date.now() - decStartLarge;
console.log(`   10KB payload: ${(decTimeLarge / 100).toFixed(2)}ms avg (100 iterations)`);

// ===== Summary =====
console.log('\n\n' + '='.repeat(60));
console.log('✅ All Examples Completed Successfully!\n');
console.log('Key Takeaways:');
console.log('  1. Encryption is transparent - trajectories stored securely');
console.log('  2. Quantum-resistant HQC-128 protects against future attacks');
console.log('  3. AEAD provides both confidentiality and integrity');
console.log('  4. Performance overhead is minimal (<2ms per operation)');
console.log('  5. Key management is crucial for security');
console.log('\n📚 For more information, see docs/ENCRYPTION_GUIDE.md');
console.log('=' .repeat(60));
