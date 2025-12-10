/**
 * Simple Quantum Fingerprint Test
 */

const { JjWrapper } = require('../index.js');
const qudag = require('@qudag/napi-core');
const fs = require('fs');
const path = require('path');
const os = require('os');

async function test() {
  console.log('=== Quantum Fingerprint Test ===\n');

  // Create temporary directory for test repo
  const testDir = path.join(os.tmpdir(), `jj-quantum-test-${Date.now()}`);
  const originalCwd = process.cwd();

  try {
    // Setup test repo
    console.log('0. Setting up test repository...');
    fs.mkdirSync(testDir, { recursive: true });
    process.chdir(testDir);

    // Initialize jj repo
    const wrapper = new JjWrapper();
    await wrapper.execute(['git', 'init']);
    console.log(`   ✓ Initialized jj repo in ${testDir}\n`);

    // Create wrapper
    console.log('1. Creating JJWrapper...');
    console.log('   ✓ JJWrapper created\n');

    // Execute operation to create a test operation
    console.log('2. Executing describe command...');
    await wrapper.execute(['describe', '-m', 'Test quantum fingerprints']);
    console.log('   ✓ Describe command executed\n');

    // Get operation
    console.log('3. Getting operation...');
    const operations = wrapper.getOperations(1);
    if (operations.length === 0) {
      throw new Error('No operations found!');
    }
    const operation = operations[0];
    console.log(`   ✓ Found operation: ${operation.id}`);
    console.log(`   Command: ${operation.command}\n`);

    // Get operation data
    console.log('4. Getting operation data...');
    const operationData = wrapper.getOperationData(operation.id);
    console.log(`   ✓ Operation data: ${operationData.length} bytes\n`);

    // Generate quantum fingerprint using @qudag/napi-core
    console.log('5. Generating quantum fingerprint with @qudag/napi-core...');
    const quantumFingerprint = qudag.QuantumFingerprint.generate(Buffer.from(operationData));
    const fpHex = quantumFingerprint.asHex();
    console.log(`   ✓ Fingerprint: ${fpHex.substring(0, 32)}...`);
    console.log(`   Length: ${fpHex.length} chars (${fpHex.length / 2} bytes)\n`);

    // Verify quantum fingerprint
    console.log('6. Verifying quantum fingerprint...');
    const isValid = quantumFingerprint.verify();
    console.log(`   ✓ Fingerprint ${isValid ? 'VALID ✓' : 'INVALID ✗'}\n`);

    if (!isValid) {
      throw new Error('Quantum fingerprint verification failed!');
    }

    // Store fingerprint
    console.log('7. Storing fingerprint with operation...');
    wrapper.setOperationFingerprint(operation.id, fpHex);
    const updatedOps = wrapper.getOperations(1);
    const updatedOp = updatedOps[0];
    console.log(`   ✓ Stored: ${updatedOp.quantumFingerprint?.substring(0, 32)}...\n`);

    if (updatedOp.quantumFingerprint !== fpHex) {
      throw new Error('Stored fingerprint does not match!');
    }

    // Verify using wrapper
    console.log('8. Verifying with wrapper...');
    const wrapperValid = await wrapper.verifyOperationFingerprint(operation.id, fpHex);
    console.log(`   ✓ Wrapper verification: ${wrapperValid ? 'PASS ✓' : 'FAIL ✗'}\n`);

    if (!wrapperValid) {
      throw new Error('Wrapper verification failed!');
    }

    // Test convenience functions
    console.log('9. Testing convenience functions...');
    const fingerprintBytes = qudag.generateQuantumFingerprint(Buffer.from(operationData));
    console.log(`   ✓ Fingerprint bytes: ${fingerprintBytes.length} bytes`);

    const quickVerify = qudag.verifyQuantumFingerprint(Buffer.from(operationData), Buffer.from(fingerprintBytes));
    console.log(`   ✓ Quick verify: ${quickVerify ? 'PASS ✓' : 'FAIL ✗'}\n`);

    if (!quickVerify) {
      throw new Error('Quick verify failed!');
    }

    // ML-DSA Signature details
    console.log('10. ML-DSA Signature details...');
    const signature = quantumFingerprint.getSignature();
    const publicKey = quantumFingerprint.getPublicKey();
    console.log(`   ✓ Signature: ${signature.length} bytes (ML-DSA-65)`);
    console.log(`   ✓ Public key: ${publicKey.length} bytes\n`);

    // Verify signature independently
    console.log('11. Independent ML-DSA verification...');
    const fingerprintBytes2 = quantumFingerprint.asBytes();
    const mlDsaPublicKey = qudag.MlDsaPublicKey.fromBytes(Buffer.from(publicKey));
    const sigValid = mlDsaPublicKey.verify(Buffer.from(fingerprintBytes2), Buffer.from(signature));
    console.log(`   ✓ ML-DSA signature verification: ${sigValid ? 'PASS ✓' : 'FAIL ✗'}\n`);

    if (!sigValid) {
      throw new Error('ML-DSA signature verification failed!');
    }

    // Test tamper detection
    console.log('12. Testing tamper detection...');
    const tamperedData = Buffer.from(operationData);
    tamperedData[0] = tamperedData[0] ^ 0xFF; // Flip bits
    const tamperCheck = qudag.verifyQuantumFingerprint(Buffer.from(tamperedData), Buffer.from(fingerprintBytes));
    console.log(`   ✓ Tampered data detected: ${!tamperCheck ? 'YES ✓' : 'NO ✗'}\n`);

    if (tamperCheck) {
      throw new Error('Tamper detection failed - should have detected tampering!');
    }

    // Performance test
    console.log('13. Performance test (1000 iterations)...');
    const iterations = 1000;
    const operationDataBuffer = Buffer.from(operationData);
    const start = Date.now();
    for (let i = 0; i < iterations; i++) {
      qudag.generateQuantumFingerprint(operationDataBuffer);
    }
    const elapsed = Date.now() - start;
    const avgTime = elapsed / iterations;
    console.log(`   ✓ ${iterations} fingerprints in ${elapsed}ms`);
    console.log(`   Average: ${avgTime.toFixed(3)}ms per fingerprint\n`);

    // ML-DSA algorithm info
    console.log('14. ML-DSA Algorithm Information...');
    const mlDsaInfo = qudag.getMlDsaInfo();
    console.log(`   Algorithm: ${mlDsaInfo.algorithm}`);
    console.log(`   Security Level: ${mlDsaInfo.securityLevel} (NIST)`);
    console.log(`   Public Key: ${mlDsaInfo.publicKeySize} bytes`);
    console.log(`   Signature: ${mlDsaInfo.signatureSize} bytes\n`);

    console.log('=== ALL TESTS PASSED ✓ ===\n');
    return true;

  } catch (error) {
    console.error('\n❌ ERROR:', error.message);
    console.error(error.stack);
    return false;
  } finally {
    // Cleanup
    process.chdir(originalCwd);
    try {
      fs.rmSync(testDir, { recursive: true, force: true });
    } catch (e) {
      // Ignore cleanup errors
    }
  }
}

// Run test
test().then(success => {
  process.exit(success ? 0 : 1);
}).catch(error => {
  console.error('Fatal error:', error);
  process.exit(1);
});
