/**
 * Basic DAA Agent Example
 *
 * Demonstrates core DAA functionality:
 * - Quantum-resistant cryptography (ML-KEM-768)
 * - Digital signatures (ML-DSA)
 * - Secure hashing (BLAKE3)
 * - Password vault operations
 */

import { DAA } from 'daa-sdk';

async function main() {
  console.log('🚀 Starting Basic DAA Agent\n');

  // Initialize DAA SDK
  const daa = new DAA({
    qudag: {
      enableCrypto: true,
      enableVault: true,
    },
  });

  await daa.init();

  console.log('Platform:', daa.getPlatform());
  console.log('Initialized:', daa.isInitialized());
  console.log();

  // Example 1: ML-KEM Key Encapsulation
  console.log('📦 Example 1: ML-KEM-768 Key Encapsulation');
  console.log('-------------------------------------------');

  const mlkem = daa.crypto.mlkem();

  // Generate keypair
  const keypair = mlkem.generateKeypair();
  console.log('✅ Generated keypair');
  console.log('  Public key length:', keypair.publicKey.length, 'bytes');
  console.log('  Secret key length:', keypair.secretKey.length, 'bytes');

  // Encapsulate (create shared secret)
  const encapsulated = mlkem.encapsulate(keypair.publicKey);
  console.log('✅ Encapsulated shared secret');
  console.log('  Ciphertext length:', encapsulated.ciphertext.length, 'bytes');
  console.log('  Shared secret length:', encapsulated.sharedSecret.length, 'bytes');

  // Decapsulate (recover shared secret)
  const decapsulated = mlkem.decapsulate(keypair.secretKey, encapsulated.ciphertext);
  console.log('✅ Decapsulated shared secret');

  // Verify shared secrets match
  const match = decapsulated.every((byte, i) => byte === encapsulated.sharedSecret[i]);
  console.log('  Secrets match:', match ? '✅ Yes' : '❌ No');
  console.log();

  // Example 2: ML-DSA Digital Signatures
  console.log('✍️  Example 2: ML-DSA Digital Signatures');
  console.log('----------------------------------------');

  const mldsa = daa.crypto.mldsa();

  // Generate signing keypair
  const signingKeypair = mldsa.generateKeypair();
  console.log('✅ Generated signing keypair');
  console.log('  Public key length:', signingKeypair.publicKey.length, 'bytes');
  console.log('  Secret key length:', signingKeypair.secretKey.length, 'bytes');

  // Sign a message
  const message = new TextEncoder().encode('Hello, quantum-resistant world!');
  const signature = mldsa.sign(signingKeypair.secretKey, message);
  console.log('✅ Signed message');
  console.log('  Message length:', message.length, 'bytes');
  console.log('  Signature length:', signature.length, 'bytes');

  // Verify signature
  const isValid = mldsa.verify(signingKeypair.publicKey, message, signature);
  console.log('✅ Verified signature:', isValid ? '✅ Valid' : '❌ Invalid');
  console.log();

  // Example 3: BLAKE3 Cryptographic Hashing
  console.log('🔒 Example 3: BLAKE3 Hashing');
  console.log('----------------------------');

  const data = new TextEncoder().encode('Data to hash');
  const hash = daa.crypto.blake3(data);
  console.log('✅ Hashed data');
  console.log('  Input length:', data.length, 'bytes');
  console.log('  Hash length:', hash.length, 'bytes');
  console.log('  Hash (hex):', Buffer.from(hash).toString('hex').substring(0, 32) + '...');
  console.log();

  // Example 4: Quantum Fingerprinting
  console.log('🔍 Example 4: Quantum Fingerprinting');
  console.log('-------------------------------------');

  const fingerprint = daa.crypto.quantumFingerprint(data);
  console.log('✅ Generated quantum fingerprint');
  console.log('  Fingerprint:', fingerprint);
  console.log();

  // Example 5: Password Vault
  console.log('🔐 Example 5: Password Vault');
  console.log('-----------------------------');

  const vault = daa.vault.create('master-password-123');

  // Store credentials
  vault.store('github', 'my-username', 'my-secure-password');
  vault.store('aws', 'access-key-id', 'secret-access-key');
  console.log('✅ Stored credentials in vault');

  // Retrieve credentials
  const githubCreds = vault.get('github');
  console.log('✅ Retrieved GitHub credentials');
  console.log('  Username:', githubCreds.username);
  console.log('  Password: [REDACTED]');

  // List all services
  const services = vault.list();
  console.log('✅ Vault contains', services.length, 'services:', services.join(', '));
  console.log();

  console.log('🎉 All examples completed successfully!');
}

// Run the agent
main().catch((error) => {
  console.error('❌ Error:', error);
  process.exit(1);
});
