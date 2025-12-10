#!/usr/bin/env node
/**
 * QuDAG Integration Test
 *
 * Verifies that @qudag/napi-core is properly integrated and working
 */

const qudag = require('@qudag/napi-core');
const assert = require('assert');

console.log('🧪 Testing QuDAG Integration\n');

// Test 1: Package loads correctly
console.log('1. Package Loading...');
assert.ok(qudag, 'QuDAG package should load');
assert.ok(qudag.getVersion, 'getVersion should exist');
console.log('   ✅ Package loaded successfully\n');

// Test 2: Version info
console.log('2. Version Information...');
const version = qudag.getVersion();
assert.ok(version, 'Version should be returned');
console.log(`   Version: ${version}`);

const buildInfo = qudag.getBuildInfo();
assert.ok(buildInfo, 'Build info should be returned');
console.log(`   Target: ${buildInfo.target}`);
console.log(`   OS: ${buildInfo.os}`);
console.log('   ✅ Version info working\n');

// Test 3: ML-DSA (Digital Signatures)
console.log('3. ML-DSA Digital Signatures...');
const keypair = qudag.MlDsaKeyPair.generate();
assert.ok(keypair, 'Keypair should be generated');

const message = Buffer.from('Test message for agentic-jujutsu integration');
const signature = keypair.sign(message);
assert.ok(signature, 'Signature should be created');
assert.ok(signature.length > 0, 'Signature should have data');

const publicKey = keypair.toPublicKey();
const isValid = publicKey.verify(message, signature);
assert.strictEqual(isValid, true, 'Signature should be valid');
console.log('   ✅ ML-DSA signatures working\n');

// Test 4: Quantum Fingerprints
console.log('4. Quantum Fingerprints...');
const data = Buffer.from('Important data for agentic-jujutsu');
const fingerprint = qudag.QuantumFingerprint.generate(data);
assert.ok(fingerprint, 'Fingerprint should be generated');

const fpBytes = fingerprint.asBytes();
assert.strictEqual(fpBytes.length, 64, 'Fingerprint should be 64 bytes');

const fpValid = fingerprint.verify();
assert.strictEqual(fpValid, true, 'Fingerprint should be valid');
console.log(`   Fingerprint size: ${fpBytes.length} bytes`);
console.log('   ✅ Quantum fingerprints working\n');

// Test 5: QuantumDAG (async test)
console.log('5. QuantumDAG...');
(async () => {
    try {
        const dag = new qudag.QuantumDag();
        assert.ok(dag, 'DAG should be created');

        const id1 = await dag.addMessage(Buffer.from('Message 1'));
        assert.ok(id1, 'Should return vertex ID');
        assert.ok(typeof id1 === 'string', 'Vertex ID should be string');

        const id2 = await dag.addMessage(Buffer.from('Message 2'));
        assert.ok(id2, 'Should return second vertex ID');

        const count = await dag.vertexCount();
        assert.strictEqual(count, 2, 'Should have 2 vertices');

        const tips = await dag.getTips();
        assert.ok(Array.isArray(tips), 'Tips should be an array');

        console.log(`   Vertices: ${count}`);
        console.log(`   Tips: ${tips.length}`);
        console.log('   ✅ QuantumDAG working\n');

        console.log('═══════════════════════════════════════════════════');
        console.log('✅ ALL QUDAG INTEGRATION TESTS PASSED!');
        console.log('═══════════════════════════════════════════════════\n');

        console.log('Next Steps:');
        console.log('  1. ✅ @qudag/napi-core dependency added');
        console.log('  2. ✅ All features verified working');
        console.log('  3. ⬜ Implement AgentCoordination module');
        console.log('  4. ⬜ Add N-API bindings to JJWrapper');
        console.log('  5. ⬜ Release v2.2.0 with quantum features\n');

        process.exit(0);
    } catch (error) {
        console.error('❌ QuantumDAG test failed:', error.message);
        process.exit(1);
    }
})();
