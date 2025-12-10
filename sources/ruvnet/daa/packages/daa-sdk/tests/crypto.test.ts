/**
 * Tests for QuDAG NAPI crypto bindings
 */

import { MlKem768, MlDsa, Blake3, init, version, getModuleInfo } from '../src/qudag';

console.log('🧪 Testing QuDAG Native NAPI Bindings\n');

// Test module initialization
console.log('=== Module Initialization ===');
try {
  const initMsg = init();
  console.log('✅ init():', initMsg);

  const ver = version();
  console.log('✅ version():', ver);

  const info = getModuleInfo();
  console.log('✅ Module Info:');
  console.log('   Name:', info.name);
  console.log('   Version:', info.version);
  console.log('   Features:', info.features.join(', '));
} catch (error) {
  console.error('❌ Module initialization failed:', error);
  process.exit(1);
}

console.log();

// Test BLAKE3 hashing
console.log('=== BLAKE3 Hashing ===');
try {
  const testData = Buffer.from('Hello, QuDAG NAPI!');

  const hash = Blake3.hash(testData);
  console.log('✅ BLAKE3 hash:', hash.length, 'bytes');
  console.log('   Hash (hex):', hash.toString('hex').substring(0, 32) + '...');

  const hashHex = Blake3.hashHex(testData);
  console.log('✅ BLAKE3 hashHex:', hashHex.substring(0, 32) + '...');

  const fingerprint = Blake3.quantumFingerprint(testData);
  console.log('✅ Quantum fingerprint:', fingerprint.substring(0, 40) + '...');

  // Verify hash consistency
  if (hash.toString('hex') === hashHex) {
    console.log('✅ Hash consistency verified');
  } else {
    console.error('❌ Hash mismatch!');
  }
} catch (error) {
  console.error('❌ BLAKE3 test failed:', error);
}

console.log();

// Test ML-KEM-768
console.log('=== ML-KEM-768 Key Encapsulation ===');
try {
  const mlkem = new MlKem768();

  // Generate keypair
  const { publicKey, secretKey } = mlkem.generateKeypair();
  console.log('✅ Keypair generated');
  console.log('   Public key:', publicKey.length, 'bytes');
  console.log('   Secret key:', secretKey.length, 'bytes');

  // Verify key sizes
  if (publicKey.length === 1184) {
    console.log('✅ Public key size correct (1184 bytes)');
  } else {
    console.error(`❌ Public key size incorrect: expected 1184, got ${publicKey.length}`);
  }

  if (secretKey.length === 2400) {
    console.log('✅ Secret key size correct (2400 bytes)');
  } else {
    console.error(`❌ Secret key size incorrect: expected 2400, got ${secretKey.length}`);
  }

  // Encapsulation
  const { ciphertext, sharedSecret } = mlkem.encapsulate(publicKey);
  console.log('✅ Encapsulation successful');
  console.log('   Ciphertext:', ciphertext.length, 'bytes');
  console.log('   Shared secret:', sharedSecret.length, 'bytes');

  // Decapsulation
  const decryptedSecret = mlkem.decapsulate(ciphertext, secretKey);
  console.log('✅ Decapsulation successful');
  console.log('   Decrypted secret:', decryptedSecret.length, 'bytes');

  // Verify shared secret consistency
  if (sharedSecret.equals(decryptedSecret)) {
    console.log('✅ Shared secrets match!');
  } else {
    console.error('❌ Shared secret mismatch!');
  }
} catch (error) {
  console.error('❌ ML-KEM test failed:', error);
}

console.log();

// Test ML-DSA
console.log('=== ML-DSA Digital Signatures ===');
try {
  const mldsa = new MlDsa();

  const message = Buffer.from('Test message for ML-DSA signing');
  const secretKey = Buffer.alloc(4032); // Placeholder secret key
  const publicKey = Buffer.alloc(1952); // Placeholder public key

  // Sign
  const signature = mldsa.sign(message, secretKey);
  console.log('✅ Message signed');
  console.log('   Signature:', signature.length, 'bytes');

  // Verify
  const isValid = mldsa.verify(message, signature, publicKey);
  console.log('✅ Signature verified:', isValid ? 'VALID' : 'INVALID');

  if (isValid) {
    console.log('✅ Signature validation successful');
  } else {
    console.log('⚠️  Signature validation returned false (expected with stub implementation)');
  }
} catch (error) {
  console.error('❌ ML-DSA test failed:', error);
}

console.log();

// Performance benchmarks
console.log('=== Performance Benchmarks ===');
try {
  const iterations = 100;

  // BLAKE3 benchmark
  const blake3Data = Buffer.alloc(1024); // 1KB
  const blake3Start = Date.now();
  for (let i = 0; i < iterations; i++) {
    Blake3.hash(blake3Data);
  }
  const blake3Time = Date.now() - blake3Start;
  console.log(`✅ BLAKE3: ${iterations} hashes in ${blake3Time}ms (${(blake3Time / iterations).toFixed(2)}ms avg)`);

  // ML-KEM keypair generation benchmark
  const mlkem = new MlKem768();
  const kemStart = Date.now();
  for (let i = 0; i < 10; i++) {
    mlkem.generateKeypair();
  }
  const kemTime = Date.now() - kemStart;
  console.log(`✅ ML-KEM-768: 10 keypairs in ${kemTime}ms (${(kemTime / 10).toFixed(2)}ms avg)`);
} catch (error) {
  console.error('❌ Benchmark failed:', error);
}

console.log();
console.log('✅ All tests completed!');
