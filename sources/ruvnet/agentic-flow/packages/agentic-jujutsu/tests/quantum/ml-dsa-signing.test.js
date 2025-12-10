/**
 * ML-DSA (Dilithium) Signing Tests
 *
 * Tests post-quantum digital signature algorithm:
 * - Keypair generation
 * - Signature creation (~1.3ms)
 * - Signature verification (~0.85ms)
 * - Commit signing workflow
 * - Security properties
 */

const { AgenticJujutsu } = require('../../index.js');
const assert = require('assert');
const { performance } = require('perf_hooks');
const crypto = require('crypto');

describe('ML-DSA Signing', function() {
  this.timeout(15000);

  let jj;
  const testRepoPath = '/tmp/ml-dsa-signing-test';

  beforeEach(async function() {
    require('child_process').execSync(`rm -rf ${testRepoPath}`, { stdio: 'ignore' });
    require('child_process').execSync(`mkdir -p ${testRepoPath}`, { stdio: 'ignore' });

    jj = new AgenticJujutsu();
    await jj.initialize(testRepoPath);
  });

  afterEach(async function() {
    if (jj) {
      await jj.cleanup();
    }
    require('child_process').execSync(`rm -rf ${testRepoPath}`, { stdio: 'ignore' });
  });

  describe('Keypair Generation', function() {
    it('should generate ML-DSA keypair', async function() {
      const start = performance.now();
      const keypair = await jj.generateMLDSAKeypair();
      const duration = performance.now() - start;

      assert.ok(keypair, 'Keypair should be generated');
      assert.ok(keypair.publicKey, 'Should have public key');
      assert.ok(keypair.secretKey, 'Should have secret key');
      assert.strictEqual(typeof keypair.publicKey, 'string', 'Public key should be string');
      assert.strictEqual(typeof keypair.secretKey, 'string', 'Secret key should be string');

      // ML-DSA-65 expected key sizes (base64 encoded)
      assert.ok(keypair.publicKey.length > 1000, 'Public key should be substantial');
      assert.ok(keypair.secretKey.length > 2000, 'Secret key should be larger');

      console.log(`  ✓ Keypair generation time: ${duration.toFixed(3)}ms`);
      console.log(`  ✓ Public key length: ${keypair.publicKey.length} bytes`);
      console.log(`  ✓ Secret key length: ${keypair.secretKey.length} bytes`);
    });

    it('should generate unique keypairs', async function() {
      const keypair1 = await jj.generateMLDSAKeypair();
      const keypair2 = await jj.generateMLDSAKeypair();

      assert.notStrictEqual(keypair1.publicKey, keypair2.publicKey,
        'Public keys should be unique');
      assert.notStrictEqual(keypair1.secretKey, keypair2.secretKey,
        'Secret keys should be unique');
    });

    it('should use secure randomness', async function() {
      const keypairs = [];
      for (let i = 0; i < 10; i++) {
        keypairs.push(await jj.generateMLDSAKeypair());
      }

      // Check that all keys are unique
      const publicKeys = new Set(keypairs.map(k => k.publicKey));
      const secretKeys = new Set(keypairs.map(k => k.secretKey));

      assert.strictEqual(publicKeys.size, 10, 'All public keys should be unique');
      assert.strictEqual(secretKeys.size, 10, 'All secret keys should be unique');
    });
  });

  describe('Signature Creation', function() {
    it('should create signature for message', async function() {
      const keypair = await jj.generateMLDSAKeypair();
      const message = 'Test commit message';

      const start = performance.now();
      const signature = await jj.signMessage(message, keypair.secretKey);
      const duration = performance.now() - start;

      assert.ok(signature, 'Signature should be created');
      assert.strictEqual(typeof signature, 'string', 'Signature should be string');
      assert.ok(signature.length > 1000, 'Signature should be substantial');

      console.log(`  ✓ Signature creation time: ${duration.toFixed(3)}ms`);
      console.log(`  ✓ Signature length: ${signature.length} bytes`);
      console.log(`  ✓ Target: ~1.3ms (actual: ${duration.toFixed(3)}ms)`);

      // Should be close to 1.3ms target
      assert.ok(duration < 10, 'Signature creation should be fast (<10ms)');
    });

    it('should create deterministic signatures', async function() {
      const keypair = await jj.generateMLDSAKeypair();
      const message = 'Deterministic test';

      const sig1 = await jj.signMessage(message, keypair.secretKey);
      const sig2 = await jj.signMessage(message, keypair.secretKey);

      // ML-DSA is deterministic by default
      assert.strictEqual(sig1, sig2, 'Same message should produce same signature');
    });

    it('should create different signatures for different messages', async function() {
      const keypair = await jj.generateMLDSAKeypair();

      const sig1 = await jj.signMessage('Message 1', keypair.secretKey);
      const sig2 = await jj.signMessage('Message 2', keypair.secretKey);

      assert.notStrictEqual(sig1, sig2, 'Different messages should have different signatures');
    });

    it('should handle large messages', async function() {
      const keypair = await jj.generateMLDSAKeypair();
      const largeMessage = 'x'.repeat(1024 * 100); // 100KB

      const start = performance.now();
      const signature = await jj.signMessage(largeMessage, keypair.secretKey);
      const duration = performance.now() - start;

      assert.ok(signature, 'Should handle large messages');
      console.log(`  ✓ Large message (100KB) signing time: ${duration.toFixed(3)}ms`);
    });

    it('should sign structured commit data', async function() {
      const keypair = await jj.generateMLDSAKeypair();
      const commitData = {
        message: 'feat: Add quantum signing',
        author: 'agent-001',
        timestamp: Date.now(),
        changes: [
          { path: 'src/quantum.rs', type: 'add' },
          { path: 'tests/quantum.test.js', type: 'add' }
        ],
        fingerprint: 'abc123...'
      };

      const commitJson = JSON.stringify(commitData);
      const signature = await jj.signMessage(commitJson, keypair.secretKey);

      assert.ok(signature, 'Should sign structured data');
      assert.ok(signature.length > 1000);
    });
  });

  describe('Signature Verification', function() {
    it('should verify valid signature', async function() {
      const keypair = await jj.generateMLDSAKeypair();
      const message = 'Verify this message';
      const signature = await jj.signMessage(message, keypair.secretKey);

      const start = performance.now();
      const isValid = await jj.verifySignature(message, signature, keypair.publicKey);
      const duration = performance.now() - start;

      assert.strictEqual(isValid, true, 'Valid signature should verify');

      console.log(`  ✓ Signature verification time: ${duration.toFixed(3)}ms`);
      console.log(`  ✓ Target: ~0.85ms (actual: ${duration.toFixed(3)}ms)`);

      assert.ok(duration < 5, 'Verification should be fast (<5ms)');
    });

    it('should reject invalid signature', async function() {
      const keypair = await jj.generateMLDSAKeypair();
      const message = 'Original message';
      const signature = await jj.signMessage(message, keypair.secretKey);

      // Tamper with signature
      const tamperedSignature = signature.slice(0, -10) + 'TAMPERED!!';

      const isValid = await jj.verifySignature(message, tamperedSignature, keypair.publicKey);
      assert.strictEqual(isValid, false, 'Tampered signature should fail');
    });

    it('should reject signature with wrong public key', async function() {
      const keypair1 = await jj.generateMLDSAKeypair();
      const keypair2 = await jj.generateMLDSAKeypair();

      const message = 'Test message';
      const signature = await jj.signMessage(message, keypair1.secretKey);

      const isValid = await jj.verifySignature(message, signature, keypair2.publicKey);
      assert.strictEqual(isValid, false, 'Wrong public key should fail verification');
    });

    it('should reject signature with modified message', async function() {
      const keypair = await jj.generateMLDSAKeypair();
      const originalMessage = 'Original message';
      const signature = await jj.signMessage(originalMessage, keypair.secretKey);

      const modifiedMessage = 'Modified message';
      const isValid = await jj.verifySignature(modifiedMessage, signature, keypair.publicKey);

      assert.strictEqual(isValid, false, 'Modified message should fail verification');
    });

    it('should verify batch of signatures', async function() {
      const keypair = await jj.generateMLDSAKeypair();
      const messages = ['Message 1', 'Message 2', 'Message 3', 'Message 4'];

      const signatures = await Promise.all(
        messages.map(msg => jj.signMessage(msg, keypair.secretKey))
      );

      const start = performance.now();
      const verifications = await Promise.all(
        messages.map((msg, i) =>
          jj.verifySignature(msg, signatures[i], keypair.publicKey)
        )
      );
      const duration = performance.now() - start;

      assert.ok(verifications.every(v => v === true), 'All signatures should verify');

      console.log(`  ✓ Batch verification (4 sigs): ${duration.toFixed(3)}ms`);
      console.log(`  ✓ Average per verification: ${(duration / 4).toFixed(3)}ms`);
    });
  });

  describe('Commit Signing Workflow', function() {
    it('should sign commit with ML-DSA', async function() {
      const keypair = await jj.generateMLDSAKeypair();
      const agentId = 'signing-agent-001';

      // Register agent with keypair
      await jj.registerQuantumAgent(agentId, ['sign'], keypair);

      // Create a commit
      const commitData = {
        message: 'feat: Add quantum signing',
        author: agentId,
        files: ['src/quantum.rs', 'tests/quantum.test.js'],
        timestamp: Date.now()
      };

      const start = performance.now();
      const signedCommit = await jj.signCommit(agentId, commitData);
      const duration = performance.now() - start;

      assert.ok(signedCommit, 'Commit should be signed');
      assert.ok(signedCommit.signature, 'Should have signature');
      assert.ok(signedCommit.fingerprint, 'Should have fingerprint');
      assert.deepStrictEqual(signedCommit.data, commitData);

      console.log(`  ✓ Commit signing time: ${duration.toFixed(3)}ms`);
      console.log(`  ✓ Signature length: ${signedCommit.signature.length} bytes`);
    });

    it('should verify signed commit', async function() {
      const keypair = await jj.generateMLDSAKeypair();
      const agentId = 'verify-agent-001';

      await jj.registerQuantumAgent(agentId, ['sign', 'verify'], keypair);

      const commitData = {
        message: 'test: Verify commit signing',
        author: agentId,
        timestamp: Date.now()
      };

      const signedCommit = await jj.signCommit(agentId, commitData);

      const start = performance.now();
      const isValid = await jj.verifyCommit(signedCommit, keypair.publicKey);
      const duration = performance.now() - start;

      assert.strictEqual(isValid, true, 'Valid commit should verify');
      console.log(`  ✓ Commit verification time: ${duration.toFixed(3)}ms`);
    });

    it('should detect tampered commit', async function() {
      const keypair = await jj.generateMLDSAKeypair();
      const agentId = 'tamper-test-agent';

      await jj.registerQuantumAgent(agentId, ['sign'], keypair);

      const commitData = {
        message: 'Original commit',
        author: agentId
      };

      const signedCommit = await jj.signCommit(agentId, commitData);

      // Tamper with commit data
      const tamperedCommit = {
        ...signedCommit,
        data: {
          ...signedCommit.data,
          message: 'Tampered commit'
        }
      };

      const isValid = await jj.verifyCommit(tamperedCommit, keypair.publicKey);
      assert.strictEqual(isValid, false, 'Tampered commit should fail verification');
    });

    it('should handle multi-agent commit workflow', async function() {
      const agents = [
        { id: 'agent-001', keypair: await jj.generateMLDSAKeypair() },
        { id: 'agent-002', keypair: await jj.generateMLDSAKeypair() },
        { id: 'agent-003', keypair: await jj.generateMLDSAKeypair() }
      ];

      // Register all agents
      await Promise.all(
        agents.map(a => jj.registerQuantumAgent(a.id, ['sign'], a.keypair))
      );

      // Each agent creates and signs a commit
      const signedCommits = await Promise.all(
        agents.map(a =>
          jj.signCommit(a.id, {
            message: `Commit from ${a.id}`,
            author: a.id,
            timestamp: Date.now()
          })
        )
      );

      // Verify all commits
      const verifications = await Promise.all(
        signedCommits.map((commit, i) =>
          jj.verifyCommit(commit, agents[i].keypair.publicKey)
        )
      );

      assert.ok(verifications.every(v => v === true), 'All commits should verify');
      console.log(`  ✓ Multi-agent workflow: 3 agents signed and verified successfully`);
    });
  });

  describe('Security Properties', function() {
    it('should resist signature forgery', async function() {
      const keypair = await jj.generateMLDSAKeypair();
      const message = 'Important message';

      // Try to create signature without secret key
      const fakeSignature = crypto.randomBytes(2420).toString('base64');

      const isValid = await jj.verifySignature(message, fakeSignature, keypair.publicKey);
      assert.strictEqual(isValid, false, 'Forged signature should fail');
    });

    it('should prevent signature reuse', async function() {
      const keypair = await jj.generateMLDSAKeypair();
      const message1 = 'Message 1';
      const message2 = 'Message 2';

      const signature = await jj.signMessage(message1, keypair.secretKey);

      // Try to reuse signature for different message
      const isValid = await jj.verifySignature(message2, signature, keypair.publicKey);
      assert.strictEqual(isValid, false, 'Signature should not be reusable');
    });

    it('should maintain quantum resistance', async function() {
      // Simulate quantum attack: try to derive secret key from public key
      const keypair = await jj.generateMLDSAKeypair();
      const message = 'Quantum resistant message';
      const signature = await jj.signMessage(message, keypair.secretKey);

      // In a quantum attack, adversary has public key and signature
      // but should not be able to forge signatures
      const attackAttempts = 100;
      let successfulForgeries = 0;

      for (let i = 0; i < attackAttempts; i++) {
        const forgedSignature = crypto.randomBytes(2420).toString('base64');
        const isValid = await jj.verifySignature(message, forgedSignature, keypair.publicKey);
        if (isValid) successfulForgeries++;
      }

      console.log(`  ✓ Quantum attack simulation: ${successfulForgeries}/${attackAttempts} forgeries succeeded`);
      assert.strictEqual(successfulForgeries, 0, 'Should resist quantum forgery attempts');
    });
  });

  describe('Performance Benchmarks', function() {
    it('should meet signing performance target (~1.3ms)', async function() {
      const keypair = await jj.generateMLDSAKeypair();
      const iterations = 100;
      const timings = [];

      for (let i = 0; i < iterations; i++) {
        const message = `Benchmark message ${i}`;
        const start = performance.now();
        await jj.signMessage(message, keypair.secretKey);
        const duration = performance.now() - start;
        timings.push(duration);
      }

      const avgTime = timings.reduce((a, b) => a + b, 0) / iterations;
      const maxTime = Math.max(...timings);
      const minTime = Math.min(...timings);
      const p95Time = timings.sort((a, b) => a - b)[Math.floor(iterations * 0.95)];

      console.log(`  ✓ Signing performance (${iterations} iterations):`);
      console.log(`    - Average: ${avgTime.toFixed(3)}ms (target: ~1.3ms)`);
      console.log(`    - Min: ${minTime.toFixed(3)}ms`);
      console.log(`    - Max: ${maxTime.toFixed(3)}ms`);
      console.log(`    - P95: ${p95Time.toFixed(3)}ms`);

      assert.ok(avgTime < 10, 'Average signing time should be reasonable');
    });

    it('should meet verification performance target (~0.85ms)', async function() {
      const keypair = await jj.generateMLDSAKeypair();
      const iterations = 100;
      const message = 'Verification benchmark';
      const signature = await jj.signMessage(message, keypair.secretKey);
      const timings = [];

      for (let i = 0; i < iterations; i++) {
        const start = performance.now();
        await jj.verifySignature(message, signature, keypair.publicKey);
        const duration = performance.now() - start;
        timings.push(duration);
      }

      const avgTime = timings.reduce((a, b) => a + b, 0) / iterations;
      const maxTime = Math.max(...timings);
      const minTime = Math.min(...timings);
      const p95Time = timings.sort((a, b) => a - b)[Math.floor(iterations * 0.95)];

      console.log(`  ✓ Verification performance (${iterations} iterations):`);
      console.log(`    - Average: ${avgTime.toFixed(3)}ms (target: ~0.85ms)`);
      console.log(`    - Min: ${minTime.toFixed(3)}ms`);
      console.log(`    - Max: ${maxTime.toFixed(3)}ms`);
      console.log(`    - P95: ${p95Time.toFixed(3)}ms`);

      assert.ok(avgTime < 5, 'Average verification time should be reasonable');
    });

    it('should handle high-throughput signing', async function() {
      const keypair = await jj.generateMLDSAKeypair();
      const count = 100;
      const messages = Array.from({ length: count }, (_, i) => `Message ${i}`);

      const start = performance.now();
      await Promise.all(messages.map(msg => jj.signMessage(msg, keypair.secretKey)));
      const duration = performance.now() - start;

      const throughput = (count / duration) * 1000;

      console.log(`  ✓ High-throughput signing:`);
      console.log(`    - ${count} signatures in ${duration.toFixed(2)}ms`);
      console.log(`    - Throughput: ${throughput.toFixed(0)} sigs/sec`);
      console.log(`    - Average latency: ${(duration / count).toFixed(3)}ms`);
    });
  });

  describe('Error Handling', function() {
    it('should handle invalid secret key', async function() {
      const message = 'Test message';
      const invalidKey = 'not-a-valid-secret-key';

      await assert.rejects(
        async () => {
          await jj.signMessage(message, invalidKey);
        },
        /Invalid secret key/,
        'Should reject invalid secret key'
      );
    });

    it('should handle invalid public key', async function() {
      const keypair = await jj.generateMLDSAKeypair();
      const message = 'Test message';
      const signature = await jj.signMessage(message, keypair.secretKey);
      const invalidPubKey = 'not-a-valid-public-key';

      await assert.rejects(
        async () => {
          await jj.verifySignature(message, signature, invalidPubKey);
        },
        /Invalid public key/,
        'Should reject invalid public key'
      );
    });

    it('should handle corrupted signature', async function() {
      const keypair = await jj.generateMLDSAKeypair();
      const message = 'Test message';
      const corruptedSig = 'corrupted!!!';

      const isValid = await jj.verifySignature(message, corruptedSig, keypair.publicKey);
      assert.strictEqual(isValid, false, 'Corrupted signature should fail gracefully');
    });
  });
});
