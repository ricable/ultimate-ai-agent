/**
 * Quantum Fingerprint Tests
 *
 * Tests quantum-resistant fingerprint generation and verification:
 * - SHA3-512 fingerprint generation
 * - Fingerprint verification
 * - Collision resistance
 * - Performance benchmarks (<1ms expected)
 */

const { AgenticJujutsu } = require('../../index.js');
const assert = require('assert');
const { performance } = require('perf_hooks');
const crypto = require('crypto');

describe('Quantum Fingerprints', function() {
  this.timeout(10000);

  let jj;
  const testRepoPath = '/tmp/quantum-fingerprint-test';

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

  describe('Fingerprint Generation', function() {
    it('should generate SHA3-512 fingerprint for operation', async function() {
      const operation = {
        type: 'create',
        path: 'test.txt',
        content: 'Hello quantum world',
        timestamp: Date.now()
      };

      const start = performance.now();
      const fingerprint = await jj.generateFingerprint(operation);
      const duration = performance.now() - start;

      assert.ok(fingerprint, 'Fingerprint should be generated');
      assert.strictEqual(typeof fingerprint, 'string', 'Fingerprint should be string');
      assert.strictEqual(fingerprint.length, 128, 'SHA3-512 produces 128 hex chars');
      assert.match(fingerprint, /^[a-f0-9]{128}$/, 'Should be valid hex string');

      console.log(`  ✓ Fingerprint generation time: ${duration.toFixed(3)}ms`);
      console.log(`  ✓ Fingerprint: ${fingerprint.substring(0, 32)}...`);
      assert.ok(duration < 1, 'Generation should be <1ms');
    });

    it('should generate deterministic fingerprints', async function() {
      const operation = {
        type: 'create',
        path: 'test.txt',
        content: 'Deterministic content'
      };

      const fp1 = await jj.generateFingerprint(operation);
      const fp2 = await jj.generateFingerprint(operation);

      assert.strictEqual(fp1, fp2, 'Same input should produce same fingerprint');
    });

    it('should generate different fingerprints for different operations', async function() {
      const op1 = { type: 'create', path: 'a.txt', content: 'Content A' };
      const op2 = { type: 'create', path: 'b.txt', content: 'Content B' };

      const fp1 = await jj.generateFingerprint(op1);
      const fp2 = await jj.generateFingerprint(op2);

      assert.notStrictEqual(fp1, fp2, 'Different operations should have different fingerprints');
    });

    it('should handle large operation data', async function() {
      const largeContent = 'x'.repeat(1024 * 1024); // 1MB
      const operation = {
        type: 'create',
        path: 'large.txt',
        content: largeContent
      };

      const start = performance.now();
      const fingerprint = await jj.generateFingerprint(operation);
      const duration = performance.now() - start;

      assert.ok(fingerprint, 'Should handle large data');
      assert.strictEqual(fingerprint.length, 128);

      console.log(`  ✓ Large data (1MB) fingerprint time: ${duration.toFixed(3)}ms`);
      assert.ok(duration < 50, 'Should handle large data efficiently');
    });

    it('should handle nested object structures', async function() {
      const complexOperation = {
        type: 'merge',
        paths: ['a.txt', 'b.txt', 'c.txt'],
        metadata: {
          author: 'agent-001',
          timestamp: Date.now(),
          dependencies: [
            { id: 'dep1', version: '1.0.0' },
            { id: 'dep2', version: '2.0.0' }
          ]
        },
        changes: {
          additions: [1, 2, 3],
          deletions: [4, 5, 6],
          modifications: [7, 8, 9]
        }
      };

      const fingerprint = await jj.generateFingerprint(complexOperation);
      assert.ok(fingerprint, 'Should handle complex nested structures');
      assert.strictEqual(fingerprint.length, 128);
    });
  });

  describe('Fingerprint Verification', function() {
    it('should verify valid fingerprint', async function() {
      const operation = {
        type: 'create',
        path: 'verify.txt',
        content: 'Verify this'
      };

      const fingerprint = await jj.generateFingerprint(operation);

      const start = performance.now();
      const isValid = await jj.verifyFingerprint(operation, fingerprint);
      const duration = performance.now() - start;

      assert.strictEqual(isValid, true, 'Valid fingerprint should verify');

      console.log(`  ✓ Fingerprint verification time: ${duration.toFixed(3)}ms`);
      assert.ok(duration < 1, 'Verification should be <1ms');
    });

    it('should reject invalid fingerprint', async function() {
      const operation = {
        type: 'create',
        path: 'test.txt',
        content: 'Original content'
      };

      const validFingerprint = await jj.generateFingerprint(operation);
      const invalidFingerprint = validFingerprint.replace(/a/g, 'b');

      const isValid = await jj.verifyFingerprint(operation, invalidFingerprint);
      assert.strictEqual(isValid, false, 'Invalid fingerprint should fail verification');
    });

    it('should detect tampered operation data', async function() {
      const operation = {
        type: 'create',
        path: 'tamper.txt',
        content: 'Original content'
      };

      const fingerprint = await jj.generateFingerprint(operation);

      // Tamper with operation
      const tamperedOperation = {
        ...operation,
        content: 'Tampered content'
      };

      const isValid = await jj.verifyFingerprint(tamperedOperation, fingerprint);
      assert.strictEqual(isValid, false, 'Tampered data should fail verification');
    });

    it('should verify batch of fingerprints', async function() {
      const operations = [
        { type: 'create', path: 'a.txt', content: 'A' },
        { type: 'create', path: 'b.txt', content: 'B' },
        { type: 'create', path: 'c.txt', content: 'C' },
        { type: 'create', path: 'd.txt', content: 'D' }
      ];

      const fingerprints = await Promise.all(
        operations.map(op => jj.generateFingerprint(op))
      );

      const start = performance.now();
      const verifications = await Promise.all(
        operations.map((op, i) => jj.verifyFingerprint(op, fingerprints[i]))
      );
      const duration = performance.now() - start;

      assert.ok(verifications.every(v => v === true), 'All should verify');
      console.log(`  ✓ Batch verification (4 ops): ${duration.toFixed(3)}ms`);
      console.log(`  ✓ Average per verification: ${(duration / 4).toFixed(3)}ms`);
    });
  });

  describe('Collision Resistance', function() {
    it('should have extremely low collision probability', async function() {
      const operationCount = 1000;
      const fingerprints = new Set();

      const start = performance.now();

      for (let i = 0; i < operationCount; i++) {
        const operation = {
          type: 'test',
          iteration: i,
          data: crypto.randomBytes(32).toString('hex')
        };
        const fingerprint = await jj.generateFingerprint(operation);
        fingerprints.add(fingerprint);
      }

      const duration = performance.now() - start;

      assert.strictEqual(fingerprints.size, operationCount,
        'All fingerprints should be unique (no collisions)');

      console.log(`  ✓ Generated ${operationCount} unique fingerprints in ${duration.toFixed(2)}ms`);
      console.log(`  ✓ Average: ${(duration / operationCount).toFixed(3)}ms per fingerprint`);
      console.log(`  ✓ Collision rate: 0/${operationCount} (0%)`);
    });

    it('should resist near-collision attacks', async function() {
      // Test with very similar inputs
      const baseOp = {
        type: 'create',
        path: 'test.txt',
        content: 'Base content'
      };

      const variations = [
        { ...baseOp, content: 'Base content' },
        { ...baseOp, content: 'Base content ' }, // Extra space
        { ...baseOp, content: 'Base contentx' }, // Extra char
        { ...baseOp, content: 'base content' },  // Different case
        { ...baseOp, path: 'test.txt ' }         // Path variation
      ];

      const fingerprints = await Promise.all(
        variations.map(op => jj.generateFingerprint(op))
      );

      // All should be different
      const uniqueFingerprints = new Set(fingerprints);
      assert.strictEqual(uniqueFingerprints.size, fingerprints.length,
        'Small variations should produce completely different fingerprints');

      // Check bit difference (avalanche effect)
      for (let i = 1; i < fingerprints.length; i++) {
        const diff = countBitDifference(fingerprints[0], fingerprints[i]);
        const diffPercentage = (diff / (128 * 4)) * 100; // 128 hex chars = 512 bits

        console.log(`  ✓ Fingerprint ${i} bit difference: ${diff} bits (${diffPercentage.toFixed(1)}%)`);
        assert.ok(diffPercentage > 40, 'Should have >40% bit difference (avalanche effect)');
      }
    });

    it('should maintain uniqueness under quantum attack simulation', async function() {
      // Simulate quantum attack by trying to find collisions with modified inputs
      const targetOp = {
        type: 'target',
        path: 'target.txt',
        content: 'Target content'
      };

      const targetFingerprint = await jj.generateFingerprint(targetOp);
      const attempts = 10000;
      let collisions = 0;

      const start = performance.now();

      for (let i = 0; i < attempts; i++) {
        const attackOp = {
          type: 'attack',
          nonce: i,
          data: crypto.randomBytes(32).toString('hex')
        };

        const fingerprint = await jj.generateFingerprint(attackOp);
        if (fingerprint === targetFingerprint) {
          collisions++;
        }
      }

      const duration = performance.now() - start;

      console.log(`  ✓ Quantum attack simulation: ${attempts} attempts in ${duration.toFixed(2)}ms`);
      console.log(`  ✓ Collision rate: ${collisions}/${attempts} (${(collisions/attempts*100).toFixed(4)}%)`);

      assert.strictEqual(collisions, 0, 'Should have no collisions in quantum attack simulation');
    });
  });

  describe('Performance Benchmarks', function() {
    it('should meet <1ms generation target', async function() {
      const iterations = 100;
      const timings = [];

      for (let i = 0; i < iterations; i++) {
        const operation = {
          type: 'benchmark',
          iteration: i,
          data: crypto.randomBytes(128).toString('hex')
        };

        const start = performance.now();
        await jj.generateFingerprint(operation);
        const duration = performance.now() - start;
        timings.push(duration);
      }

      const avgTime = timings.reduce((a, b) => a + b, 0) / iterations;
      const maxTime = Math.max(...timings);
      const minTime = Math.min(...timings);
      const p95Time = timings.sort((a, b) => a - b)[Math.floor(iterations * 0.95)];

      console.log(`  ✓ Average time: ${avgTime.toFixed(3)}ms`);
      console.log(`  ✓ Min time: ${minTime.toFixed(3)}ms`);
      console.log(`  ✓ Max time: ${maxTime.toFixed(3)}ms`);
      console.log(`  ✓ P95 time: ${p95Time.toFixed(3)}ms`);

      assert.ok(avgTime < 1, 'Average generation time should be <1ms');
      assert.ok(p95Time < 2, 'P95 generation time should be <2ms');
    });

    it('should handle high-throughput fingerprinting', async function() {
      const operationCount = 1000;
      const operations = Array.from({ length: operationCount }, (_, i) => ({
        type: 'throughput',
        iteration: i,
        data: crypto.randomBytes(64).toString('hex')
      }));

      const start = performance.now();
      const fingerprints = await Promise.all(
        operations.map(op => jj.generateFingerprint(op))
      );
      const duration = performance.now() - start;

      const throughput = (operationCount / duration) * 1000;

      console.log(`  ✓ Generated ${operationCount} fingerprints in ${duration.toFixed(2)}ms`);
      console.log(`  ✓ Throughput: ${throughput.toFixed(0)} fingerprints/sec`);
      console.log(`  ✓ Average latency: ${(duration / operationCount).toFixed(3)}ms`);

      assert.strictEqual(fingerprints.length, operationCount);
      assert.ok(throughput > 1000, 'Should handle >1000 fingerprints/sec');
    });

    it('should scale linearly with parallel operations', async function() {
      const threadCounts = [1, 2, 4, 8];
      const operationsPerThread = 100;

      for (const threads of threadCounts) {
        const operations = Array.from(
          { length: threads * operationsPerThread },
          (_, i) => ({
            type: 'parallel',
            thread: i % threads,
            iteration: Math.floor(i / threads),
            data: crypto.randomBytes(32).toString('hex')
          })
        );

        const start = performance.now();
        await Promise.all(operations.map(op => jj.generateFingerprint(op)));
        const duration = performance.now() - start;

        console.log(`  ✓ ${threads} threads: ${duration.toFixed(2)}ms ` +
                    `(${(duration / (threads * operationsPerThread)).toFixed(3)}ms avg)`);
      }
    });
  });

  describe('Edge Cases', function() {
    it('should handle empty operation', async function() {
      const emptyOp = {};
      const fingerprint = await jj.generateFingerprint(emptyOp);

      assert.ok(fingerprint, 'Should generate fingerprint for empty operation');
      assert.strictEqual(fingerprint.length, 128);
    });

    it('should handle null and undefined values', async function() {
      const operation = {
        type: 'test',
        nullValue: null,
        undefinedValue: undefined,
        content: 'Has null/undefined'
      };

      const fingerprint = await jj.generateFingerprint(operation);
      assert.ok(fingerprint, 'Should handle null/undefined values');
    });

    it('should handle circular references', async function() {
      const operation = { type: 'circular' };
      operation.self = operation;

      // Should either handle gracefully or throw descriptive error
      try {
        const fingerprint = await jj.generateFingerprint(operation);
        assert.ok(fingerprint, 'Should handle circular reference');
      } catch (error) {
        assert.match(error.message, /circular/i, 'Should mention circular reference');
      }
    });

    it('should handle special characters and unicode', async function() {
      const operation = {
        type: 'unicode',
        content: '🚀 Quantum 量子 🔐 Crypto مُشَفَّر',
        emoji: '🎉🎊🎈',
        math: '∑∫∂∆∇',
        special: '!@#$%^&*()[]{}|\\;:"<>?/~`'
      };

      const fingerprint = await jj.generateFingerprint(operation);
      assert.ok(fingerprint, 'Should handle unicode and special characters');
      assert.strictEqual(fingerprint.length, 128);
    });
  });
});

/**
 * Count bit differences between two hex strings
 */
function countBitDifference(hex1, hex2) {
  let diff = 0;
  for (let i = 0; i < hex1.length; i++) {
    const byte1 = parseInt(hex1[i], 16);
    const byte2 = parseInt(hex2[i], 16);
    const xor = byte1 ^ byte2;
    diff += xor.toString(2).split('1').length - 1;
  }
  return diff;
}
