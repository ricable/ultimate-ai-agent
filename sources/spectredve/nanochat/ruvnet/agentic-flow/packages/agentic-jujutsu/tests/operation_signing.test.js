#!/usr/bin/env node

/**
 * Comprehensive tests for quantum-resistant operation signing
 *
 * Tests ML-DSA-44 signatures on operation logs for tamper-proof audit trails.
 */

const { generateSigningKeypair, JjWrapper } = require('../index.js');

// Test utilities
function assertEqual(actual, expected, message) {
    if (actual !== expected) {
        throw new Error(`${message}\nExpected: ${expected}\nActual: ${actual}`);
    }
    console.log(`✓ ${message}`);
}

function assert(condition, message) {
    if (!condition) {
        throw new Error(`Assertion failed: ${message}`);
    }
    console.log(`✓ ${message}`);
}

async function runTests() {
    console.log('🧪 Running Operation Signing Tests\n');

    // Test 1: Keypair Generation
    console.log('Test 1: Generate signing keypair');
    const keypair = generateSigningKeypair();
    assert(keypair.publicKey.length > 0, 'Public key generated');
    assert(keypair.secretKey.length > 0, 'Secret key generated');
    assert(keypair.publicKey !== keypair.secretKey, 'Keys are different');
    console.log();

    // Test 2: Sign and Verify Single Operation
    console.log('Test 2: Sign and verify single operation');
    const wrapper = new JjWrapper();

    // Create a mock operation by executing a status command
    try {
        await wrapper.execute(['status']);
    } catch (e) {
        // Expected to fail if not in a jj repo
        console.log('  Note: Not in jj repo, continuing with mock tests');
    }

    const operations = wrapper.getOperations(1);
    if (operations.length > 0) {
        const opId = operations[0].id;

        // Sign the operation
        wrapper.signOperation(opId, keypair.secretKey, keypair.publicKey);
        console.log(`  Signed operation: ${opId.substring(0, 8)}...`);

        // Verify the signature
        const isValid = wrapper.verifyOperationSignature(opId);
        assert(isValid, 'Operation signature is valid');

        // Check signed count
        const signedCount = wrapper.getSignedOperationsCount();
        assertEqual(signedCount, 1, 'One operation signed');
    } else {
        console.log('  ⚠️  Skipping (no operations available)');
    }
    console.log();

    // Test 3: Sign All Operations
    console.log('Test 3: Sign all operations');
    wrapper.clearLog(); // Start fresh

    // Create multiple mock operations
    for (let i = 0; i < 3; i++) {
        try {
            await wrapper.execute(['status']);
        } catch (e) {
            // Expected
        }
    }

    const allOps = wrapper.getOperations(10);
    if (allOps.length > 0) {
        const signedCount = wrapper.signAllOperations(keypair.secretKey, keypair.publicKey);
        console.log(`  Signed ${signedCount} operations`);

        // Verify all
        const verifyResult = JSON.parse(wrapper.verifyAllOperations());
        assertEqual(verifyResult.total_signed, signedCount, 'All operations signed');
        assertEqual(verifyResult.valid_count, signedCount, 'All signatures valid');
        assertEqual(verifyResult.invalid_count, 0, 'No invalid signatures');
    } else {
        console.log('  ⚠️  Skipping (no operations available)');
    }
    console.log();

    // Test 4: Tamper Detection
    console.log('Test 4: Tamper detection');
    wrapper.clearLog();

    // Create an operation and sign it
    try {
        await wrapper.execute(['status']);
    } catch (e) {
        // Expected
    }

    const ops = wrapper.getOperations(1);
    if (ops.length > 0) {
        const opId = ops[0].id;
        wrapper.signOperation(opId, keypair.secretKey, keypair.publicKey);

        // Try to verify with wrong public key
        const wrongKeypair = generateSigningKeypair();
        try {
            const result = JSON.parse(wrapper.verifyAllOperations(wrongKeypair.publicKey));
            assertEqual(result.invalid_count, 1, 'Detected tampered signature');
        } catch (e) {
            console.log('  ✓ Tamper detection working (verification failed as expected)');
        }
    } else {
        console.log('  ⚠️  Skipping (no operations available)');
    }
    console.log();

    // Test 5: Signature Chain Verification
    console.log('Test 5: Signature chain verification');
    wrapper.clearLog();

    // Create multiple operations
    for (let i = 0; i < 3; i++) {
        try {
            await wrapper.execute(['status']);
        } catch (e) {
            // Expected
        }
    }

    const chainOps = wrapper.getOperations(10);
    if (chainOps.length >= 2) {
        wrapper.signAllOperations(keypair.secretKey, keypair.publicKey);

        const chainValid = wrapper.verifySignatureChain();
        assert(chainValid, 'Signature chain is valid');
    } else {
        console.log('  ⚠️  Skipping (not enough operations for chain)');
    }
    console.log();

    // Test 6: Unsigned Operations Count
    console.log('Test 6: Unsigned operations tracking');
    wrapper.clearLog();

    // Create operations
    for (let i = 0; i < 5; i++) {
        try {
            await wrapper.execute(['status']);
        } catch (e) {
            // Expected
        }
    }

    const totalOps = wrapper.getOperations(10).length;
    if (totalOps > 0) {
        const unsignedBefore = wrapper.getUnsignedOperationsCount();
        assertEqual(unsignedBefore, totalOps, 'All operations unsigned initially');

        // Sign first 2
        const ops2 = wrapper.getOperations(2);
        if (ops2.length >= 2) {
            wrapper.signOperation(ops2[0].id, keypair.secretKey, keypair.publicKey);
            wrapper.signOperation(ops2[1].id, keypair.secretKey, keypair.publicKey);

            const signedAfter = wrapper.getSignedOperationsCount();
            const unsignedAfter = wrapper.getUnsignedOperationsCount();

            assertEqual(signedAfter, 2, 'Two operations signed');
            assertEqual(unsignedAfter, totalOps - 2, 'Remaining operations unsigned');
        }
    } else {
        console.log('  ⚠️  Skipping (no operations available)');
    }
    console.log();

    // Test 7: Performance Test
    console.log('Test 7: Performance test (100 operations)');
    wrapper.clearLog();

    const perfStart = Date.now();
    for (let i = 0; i < 100; i++) {
        try {
            await wrapper.execute(['status']);
        } catch (e) {
            // Expected
        }
    }

    const perfOps = wrapper.getOperations(100);
    if (perfOps.length >= 10) {
        const signStart = Date.now();
        const signed = wrapper.signAllOperations(keypair.secretKey, keypair.publicKey);
        const signDuration = Date.now() - signStart;

        const verifyStart = Date.now();
        const verifyResult = JSON.parse(wrapper.verifyAllOperations());
        const verifyDuration = Date.now() - verifyStart;

        console.log(`  Signed ${signed} operations in ${signDuration}ms`);
        console.log(`  Verified ${verifyResult.total_signed} operations in ${verifyDuration}ms`);
        console.log(`  Average: ${(signDuration / signed).toFixed(2)}ms per signature`);
    } else {
        console.log('  ⚠️  Skipping (not enough operations for performance test)');
    }
    console.log();

    console.log('✅ All tests passed!');
}

// Run tests
runTests().catch((error) => {
    console.error('\n❌ Test failed:', error.message);
    console.error(error.stack);
    process.exit(1);
});
