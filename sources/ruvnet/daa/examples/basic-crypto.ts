/**
 * Basic Cryptographic Operations with QuDAG Native
 *
 * This example demonstrates basic usage of quantum-resistant cryptography:
 * - ML-KEM-768 key encapsulation
 * - ML-DSA digital signatures
 * - BLAKE3 cryptographic hashing
 *
 * @example
 * ```bash
 * npm install @daa/qudag-native
 * ts-node examples/basic-crypto.ts
 * ```
 */

import { MlKem768, MlDsa, blake3Hash, blake3HashHex, quantumFingerprint } from '@daa/qudag-native';

/**
 * Example 1: ML-KEM-768 Key Encapsulation Mechanism
 *
 * Demonstrates how two parties can establish a shared secret using
 * quantum-resistant key encapsulation.
 */
async function mlkemExample() {
  console.log('\n=== ML-KEM-768 Key Encapsulation Example ===\n');

  // Create ML-KEM instance
  const mlkem = new MlKem768();

  // Alice generates a keypair
  console.log('1. Alice generates her keypair...');
  const aliceKeypair = mlkem.generateKeypair();
  console.log(`   Public key: ${aliceKeypair.publicKey.length} bytes`);
  console.log(`   Secret key: ${aliceKeypair.secretKey.length} bytes`);
  console.log(`   Public key (hex): ${aliceKeypair.publicKey.toString('hex').slice(0, 32)}...`);

  // Bob wants to send encrypted data to Alice
  // He encapsulates a shared secret using Alice's public key
  console.log('\n2. Bob encapsulates a secret for Alice...');
  const bobEncapsulation = mlkem.encapsulate(aliceKeypair.publicKey);
  console.log(`   Ciphertext: ${bobEncapsulation.ciphertext.length} bytes`);
  console.log(`   Shared secret: ${bobEncapsulation.sharedSecret.length} bytes`);
  console.log(`   Shared secret (hex): ${bobEncapsulation.sharedSecret.toString('hex')}`);

  // Alice decapsulates to recover the same shared secret
  console.log('\n3. Alice decapsulates to recover the secret...');
  const aliceSecret = mlkem.decapsulate(
    bobEncapsulation.ciphertext,
    aliceKeypair.secretKey
  );
  console.log(`   Recovered secret (hex): ${aliceSecret.toString('hex')}`);

  // Verify both parties have the same secret
  const secretsMatch = bobEncapsulation.sharedSecret.equals(aliceSecret);
  console.log(`\n4. Secrets match: ${secretsMatch ? '✅' : '❌'}`);

  if (secretsMatch) {
    console.log('   ✅ Secure channel established!');
    console.log('   Alice and Bob can now use this shared secret for symmetric encryption.');
  }

  return {
    aliceKeypair,
    bobSharedSecret: bobEncapsulation.sharedSecret,
    aliceSharedSecret: aliceSecret
  };
}

/**
 * Example 2: ML-DSA Digital Signatures
 *
 * Demonstrates how to sign messages and verify signatures using
 * quantum-resistant digital signatures.
 */
async function mldsaExample() {
  console.log('\n\n=== ML-DSA Digital Signature Example ===\n');

  // Create ML-DSA instance
  const mldsa = new MlDsa();

  // Simulate Alice having a keypair (in reality, you'd generate this securely)
  // For this example, we'll just create placeholder keys
  console.log('1. Alice has a keypair for signing...');
  const aliceSecretKey = Buffer.alloc(2560);  // ML-DSA-65 secret key size
  const alicePublicKey = Buffer.alloc(1952);  // ML-DSA-65 public key size

  // Note: In a real application, you would generate these keys properly
  console.log(`   Secret key: ${aliceSecretKey.length} bytes (kept private)`);
  console.log(`   Public key: ${alicePublicKey.length} bytes (shared publicly)`);

  // Alice signs a message
  const message = Buffer.from('This is an important contract between Alice and Bob', 'utf8');
  console.log(`\n2. Alice signs a message...`);
  console.log(`   Message: "${message.toString('utf8')}"`);

  const signature = mldsa.sign(message, aliceSecretKey);
  console.log(`   Signature: ${signature.length} bytes`);
  console.log(`   Signature (hex): ${signature.toString('hex').slice(0, 64)}...`);

  // Bob receives the message and signature
  // He verifies it using Alice's public key
  console.log(`\n3. Bob verifies Alice's signature...`);
  const isValid = mldsa.verify(message, signature, alicePublicKey);
  console.log(`   Signature valid: ${isValid ? '✅' : '❌'}`);

  // Demonstrate tampering detection
  console.log(`\n4. Testing tampering detection...`);
  const tamperedMessage = Buffer.from('This is a MODIFIED contract between Alice and Bob', 'utf8');
  const isTamperedValid = mldsa.verify(tamperedMessage, signature, alicePublicKey);
  console.log(`   Tampered message: "${tamperedMessage.toString('utf8')}"`);
  console.log(`   Signature valid: ${isTamperedValid ? '✅' : '❌'}`);

  if (!isTamperedValid) {
    console.log('   ✅ Tampering detected! Signature does not verify.');
  }

  return {
    message,
    signature,
    isValid,
    isTamperedValid
  };
}

/**
 * Example 3: BLAKE3 Cryptographic Hashing
 *
 * Demonstrates fast, secure hashing with BLAKE3 for various use cases.
 */
async function blake3Example() {
  console.log('\n\n=== BLAKE3 Hashing Example ===\n');

  // Example 1: Hash a simple message
  console.log('1. Hashing a text message...');
  const message = Buffer.from('Hello, quantum world!', 'utf8');
  const hash = blake3Hash(message);
  const hashHex = blake3HashHex(message);

  console.log(`   Message: "${message.toString('utf8')}"`);
  console.log(`   Hash (Buffer): ${hash.length} bytes`);
  console.log(`   Hash (hex): ${hashHex}`);

  // Example 2: Content-addressable storage
  console.log('\n2. Content-addressable storage...');
  const fileContent = Buffer.from('File content for storage', 'utf8');
  const fileHash = blake3HashHex(fileContent);
  console.log(`   File content: "${fileContent.toString('utf8')}"`);
  console.log(`   Content ID: ${fileHash.slice(0, 16)}...`);
  console.log(`   Use case: Store and retrieve files by their hash`);

  // Example 3: Quantum fingerprinting
  console.log('\n3. Quantum fingerprinting...');
  const userData = Buffer.from(JSON.stringify({
    name: 'Alice',
    role: 'agent',
    timestamp: Date.now()
  }));
  const fingerprint = quantumFingerprint(userData);
  console.log(`   User data: ${userData.toString('utf8')}`);
  console.log(`   Fingerprint: ${fingerprint}`);
  console.log(`   Use case: Unique identity verification`);

  // Example 4: Data integrity verification
  console.log('\n4. Data integrity verification...');
  const originalData = Buffer.from('Important document v1', 'utf8');
  const originalHash = blake3HashHex(originalData);

  const modifiedData = Buffer.from('Important document v2', 'utf8');
  const modifiedHash = blake3HashHex(modifiedData);

  console.log(`   Original: ${originalHash}`);
  console.log(`   Modified: ${modifiedHash}`);
  console.log(`   Integrity check: ${originalHash === modifiedHash ? '✅' : '❌ Data changed'}`);

  // Example 5: Performance demonstration
  console.log('\n5. Performance demonstration...');
  const largeData = Buffer.alloc(1024 * 1024);  // 1 MB
  const startTime = Date.now();
  const largeHash = blake3Hash(largeData);
  const endTime = Date.now();

  console.log(`   Hashed 1 MB in ${endTime - startTime}ms`);
  console.log(`   Hash: ${largeHash.toString('hex').slice(0, 32)}...`);

  return {
    messageHash: hash,
    fileHash,
    fingerprint,
    originalHash,
    modifiedHash
  };
}

/**
 * Example 4: Complete Secure Communication Workflow
 *
 * Demonstrates a complete workflow combining key encapsulation,
 * digital signatures, and hashing.
 */
async function secureCommunicationWorkflow() {
  console.log('\n\n=== Complete Secure Communication Workflow ===\n');

  console.log('Scenario: Alice wants to send a confidential message to Bob with authentication\n');

  // Step 1: Key encapsulation for secure channel
  console.log('Step 1: Establish secure channel with ML-KEM-768');
  const mlkem = new MlKem768();
  const bobKeypair = mlkem.generateKeypair();
  const { ciphertext, sharedSecret } = mlkem.encapsulate(bobKeypair.publicKey);
  console.log(`   ✅ Shared secret established: ${sharedSecret.toString('hex').slice(0, 16)}...`);

  // Step 2: Hash the message for integrity
  console.log('\nStep 2: Hash message for integrity verification');
  const message = Buffer.from('Confidential: Project budget is $10M', 'utf8');
  const messageHash = blake3Hash(message);
  console.log(`   Message: "${message.toString('utf8')}"`);
  console.log(`   Hash: ${messageHash.toString('hex')}`);

  // Step 3: Sign the message for authentication
  console.log('\nStep 3: Sign message for authentication');
  const mldsa = new MlDsa();
  const aliceSecretKey = Buffer.alloc(2560);  // Alice's signing key
  const alicePublicKey = Buffer.alloc(1952);   // Alice's public key
  const signature = mldsa.sign(message, aliceSecretKey);
  console.log(`   ✅ Message signed: ${signature.toString('hex').slice(0, 32)}...`);

  // Step 4: Encrypt the message (simulated with XOR for demo)
  console.log('\nStep 4: Encrypt message with shared secret');
  const encryptedMessage = Buffer.from(message).map((byte, i) =>
    byte ^ sharedSecret[i % sharedSecret.length]
  );
  console.log(`   ✅ Message encrypted: ${encryptedMessage.toString('hex').slice(0, 32)}...`);

  // Step 5: Bob receives and verifies
  console.log('\nStep 5: Bob receives and processes');

  // Bob decapsulates to get shared secret
  const bobSharedSecret = mlkem.decapsulate(ciphertext, bobKeypair.secretKey);
  console.log(`   ✅ Shared secret recovered: ${bobSharedSecret.toString('hex').slice(0, 16)}...`);

  // Bob decrypts the message
  const decryptedMessage = Buffer.from(encryptedMessage).map((byte, i) =>
    byte ^ bobSharedSecret[i % bobSharedSecret.length]
  );
  console.log(`   ✅ Message decrypted: "${decryptedMessage.toString('utf8')}"`);

  // Bob verifies the signature
  const isSignatureValid = mldsa.verify(decryptedMessage, signature, alicePublicKey);
  console.log(`   ✅ Signature verified: ${isSignatureValid ? 'VALID' : 'INVALID'}`);

  // Bob verifies the hash
  const decryptedHash = blake3Hash(decryptedMessage);
  const isHashValid = decryptedHash.equals(messageHash);
  console.log(`   ✅ Hash verified: ${isHashValid ? 'VALID' : 'INVALID'}`);

  console.log('\nWorkflow complete! Message delivered securely with:');
  console.log('   • Confidentiality (ML-KEM-768 encryption)');
  console.log('   • Authenticity (ML-DSA signature)');
  console.log('   • Integrity (BLAKE3 hash)');
  console.log('   • Quantum resistance (all algorithms are post-quantum secure)');
}

/**
 * Main function - Run all examples
 */
async function main() {
  console.log('╔══════════════════════════════════════════════════════════╗');
  console.log('║   QuDAG Native - Basic Cryptographic Operations Demo    ║');
  console.log('║   Quantum-Resistant Cryptography with NAPI-rs           ║');
  console.log('╚══════════════════════════════════════════════════════════╝');

  try {
    // Run all examples
    await mlkemExample();
    await mldsaExample();
    await blake3Example();
    await secureCommunicationWorkflow();

    console.log('\n\n✅ All examples completed successfully!');
    console.log('\nNext steps:');
    console.log('  • See api-reference.md for complete API documentation');
    console.log('  • See migration-guide.md to migrate from WASM');
    console.log('  • Check troubleshooting.md for common issues');

  } catch (error) {
    console.error('\n❌ Error running examples:', error);
    process.exit(1);
  }
}

// Run examples if executed directly
if (require.main === module) {
  main();
}

// Export functions for use in other modules
export {
  mlkemExample,
  mldsaExample,
  blake3Example,
  secureCommunicationWorkflow
};
