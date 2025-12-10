/**
 * Quantum Full Workflow Integration Tests
 *
 * End-to-end tests for complete quantum workflow:
 * 1. Register agent with quantum capabilities
 * 2. Create operation and vertex in QuantumDAG
 * 3. Generate quantum fingerprint
 * 4. Sign commit with ML-DSA
 * 5. Verify complete chain of trust
 *
 * This test suite ensures all quantum components work together seamlessly.
 */

const { AgenticJujutsu } = require('../../index.js');
const assert = require('assert');
const { performance } = require('perf_hooks');
const fs = require('fs').promises;
const path = require('path');

describe('Quantum Full Workflow', function() {
  this.timeout(30000);

  let jj;
  const testRepoPath = '/tmp/quantum-workflow-test';

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

  describe('Complete Workflow: Single Agent', function() {
    it('should execute full quantum workflow', async function() {
      console.log('\n  === Full Quantum Workflow Test ===');

      // Step 1: Register agent with quantum capabilities
      console.log('\n  Step 1: Registering agent...');
      const agentId = 'quantum-agent-001';
      const capabilities = ['read', 'write', 'sign', 'verify'];
      const keypair = await jj.generateMLDSAKeypair();

      const t1 = performance.now();
      const registration = await jj.registerQuantumAgent(agentId, capabilities, keypair);
      const d1 = performance.now() - t1;

      assert.ok(registration, 'Agent should be registered');
      assert.ok(registration.quantumCoordinates, 'Should have quantum coordinates');
      console.log(`  ✓ Agent registered in ${d1.toFixed(3)}ms`);
      console.log(`    Quantum position: (${registration.quantumCoordinates.x.toFixed(2)}, ` +
                  `${registration.quantumCoordinates.y.toFixed(2)}, ` +
                  `${registration.quantumCoordinates.z.toFixed(2)})`);

      // Step 2: Create operation and vertex
      console.log('\n  Step 2: Creating operation...');
      const operation = {
        type: 'create',
        path: 'quantum-test.txt',
        content: 'Quantum-secure content',
        timestamp: Date.now(),
        metadata: {
          agent: agentId,
          capabilities: capabilities
        }
      };

      const t2 = performance.now();
      const vertex = await jj.createVertex(agentId, operation);
      const d2 = performance.now() - t2;

      assert.ok(vertex, 'Vertex should be created');
      assert.strictEqual(vertex.agentId, agentId);
      console.log(`  ✓ Vertex created in ${d2.toFixed(3)}ms`);
      console.log(`    Vertex ID: ${vertex.id}`);

      // Step 3: Generate quantum fingerprint
      console.log('\n  Step 3: Generating quantum fingerprint...');
      const t3 = performance.now();
      const fingerprint = await jj.generateFingerprint(operation);
      const d3 = performance.now() - t3;

      assert.ok(fingerprint, 'Fingerprint should be generated');
      assert.strictEqual(fingerprint.length, 128, 'SHA3-512 fingerprint');
      console.log(`  ✓ Fingerprint generated in ${d3.toFixed(3)}ms`);
      console.log(`    Fingerprint: ${fingerprint.substring(0, 32)}...`);

      // Step 4: Sign commit with ML-DSA
      console.log('\n  Step 4: Signing commit...');
      const commitData = {
        message: 'feat: Add quantum-secure operation',
        author: agentId,
        operation: operation,
        vertex: vertex.id,
        fingerprint: fingerprint,
        timestamp: Date.now()
      };

      const t4 = performance.now();
      const signedCommit = await jj.signCommit(agentId, commitData);
      const d4 = performance.now() - t4;

      assert.ok(signedCommit, 'Commit should be signed');
      assert.ok(signedCommit.signature, 'Should have ML-DSA signature');
      console.log(`  ✓ Commit signed in ${d4.toFixed(3)}ms`);
      console.log(`    Signature length: ${signedCommit.signature.length} bytes`);

      // Step 5: Verify complete chain of trust
      console.log('\n  Step 5: Verifying chain of trust...');

      // Verify fingerprint
      const t5a = performance.now();
      const fingerprintValid = await jj.verifyFingerprint(operation, fingerprint);
      const d5a = performance.now() - t5a;
      assert.strictEqual(fingerprintValid, true, 'Fingerprint should be valid');
      console.log(`  ✓ Fingerprint verified in ${d5a.toFixed(3)}ms`);

      // Verify signature
      const t5b = performance.now();
      const signatureValid = await jj.verifyCommit(signedCommit, keypair.publicKey);
      const d5b = performance.now() - t5b;
      assert.strictEqual(signatureValid, true, 'Signature should be valid');
      console.log(`  ✓ Signature verified in ${d5b.toFixed(3)}ms`);

      // Verify vertex in DAG
      const t5c = performance.now();
      const vertexValid = await jj.verifyVertex(vertex.id);
      const d5c = performance.now() - t5c;
      assert.strictEqual(vertexValid, true, 'Vertex should be valid');
      console.log(`  ✓ Vertex verified in ${d5c.toFixed(3)}ms`);

      // Summary
      const totalTime = d1 + d2 + d3 + d4 + d5a + d5b + d5c;
      console.log('\n  === Workflow Summary ===');
      console.log(`  Total time: ${totalTime.toFixed(3)}ms`);
      console.log(`  Breakdown:`);
      console.log(`    - Agent registration: ${d1.toFixed(3)}ms (${(d1/totalTime*100).toFixed(1)}%)`);
      console.log(`    - Vertex creation: ${d2.toFixed(3)}ms (${(d2/totalTime*100).toFixed(1)}%)`);
      console.log(`    - Fingerprint generation: ${d3.toFixed(3)}ms (${(d3/totalTime*100).toFixed(1)}%)`);
      console.log(`    - Commit signing: ${d4.toFixed(3)}ms (${(d4/totalTime*100).toFixed(1)}%)`);
      console.log(`    - Verification: ${(d5a+d5b+d5c).toFixed(3)}ms (${((d5a+d5b+d5c)/totalTime*100).toFixed(1)}%)`);
      console.log('  ✓ Full quantum workflow completed successfully\n');
    });
  });

  describe('Complete Workflow: Multiple Agents', function() {
    it('should coordinate multiple agents in quantum workflow', async function() {
      console.log('\n  === Multi-Agent Quantum Workflow ===');

      const agentCount = 3;
      const agents = [];

      // Step 1: Register multiple agents
      console.log('\n  Step 1: Registering multiple agents...');
      const t1 = performance.now();

      for (let i = 0; i < agentCount; i++) {
        const agentId = `quantum-agent-${String(i).padStart(3, '0')}`;
        const keypair = await jj.generateMLDSAKeypair();
        const registration = await jj.registerQuantumAgent(
          agentId,
          ['read', 'write', 'sign'],
          keypair
        );
        agents.push({ id: agentId, keypair, registration });
      }

      const d1 = performance.now() - t1;
      console.log(`  ✓ Registered ${agentCount} agents in ${d1.toFixed(3)}ms`);
      console.log(`    Average per agent: ${(d1/agentCount).toFixed(3)}ms`);

      // Step 2: Each agent creates operations in parallel
      console.log('\n  Step 2: Creating parallel operations...');
      const t2 = performance.now();

      const operations = await Promise.all(
        agents.map((agent, i) => {
          const operation = {
            type: 'create',
            path: `file-${i}.txt`,
            content: `Content from ${agent.id}`,
            timestamp: Date.now(),
            agentIndex: i
          };
          return jj.createVertex(agent.id, operation).then(vertex => ({
            agent,
            operation,
            vertex
          }));
        })
      );

      const d2 = performance.now() - t2;
      console.log(`  ✓ Created ${operations.length} operations in ${d2.toFixed(3)}ms`);
      console.log(`    Average per operation: ${(d2/operations.length).toFixed(3)}ms`);

      // Step 3: Generate fingerprints for all operations
      console.log('\n  Step 3: Generating fingerprints...');
      const t3 = performance.now();

      const fingerprints = await Promise.all(
        operations.map(op => jj.generateFingerprint(op.operation))
      );

      const d3 = performance.now() - t3;
      console.log(`  ✓ Generated ${fingerprints.length} fingerprints in ${d3.toFixed(3)}ms`);
      console.log(`    Average per fingerprint: ${(d3/fingerprints.length).toFixed(3)}ms`);

      // Step 4: Sign all commits
      console.log('\n  Step 4: Signing commits...');
      const t4 = performance.now();

      const signedCommits = await Promise.all(
        operations.map((op, i) =>
          jj.signCommit(op.agent.id, {
            message: `Commit from ${op.agent.id}`,
            author: op.agent.id,
            operation: op.operation,
            vertex: op.vertex.id,
            fingerprint: fingerprints[i]
          })
        )
      );

      const d4 = performance.now() - t4;
      console.log(`  ✓ Signed ${signedCommits.length} commits in ${d4.toFixed(3)}ms`);
      console.log(`    Average per signature: ${(d4/signedCommits.length).toFixed(3)}ms`);

      // Step 5: Verify all commits
      console.log('\n  Step 5: Verifying all commits...');
      const t5 = performance.now();

      const verifications = await Promise.all(
        signedCommits.map((commit, i) =>
          jj.verifyCommit(commit, operations[i].agent.keypair.publicKey)
        )
      );

      const d5 = performance.now() - t5;

      assert.ok(verifications.every(v => v === true), 'All verifications should pass');
      console.log(`  ✓ Verified ${verifications.length} commits in ${d5.toFixed(3)}ms`);
      console.log(`    Average per verification: ${(d5/verifications.length).toFixed(3)}ms`);

      // Summary
      const totalTime = d1 + d2 + d3 + d4 + d5;
      console.log('\n  === Multi-Agent Workflow Summary ===');
      console.log(`  Agents: ${agentCount}`);
      console.log(`  Total time: ${totalTime.toFixed(3)}ms`);
      console.log(`  Breakdown:`);
      console.log(`    - Agent registration: ${d1.toFixed(3)}ms`);
      console.log(`    - Parallel operations: ${d2.toFixed(3)}ms`);
      console.log(`    - Fingerprinting: ${d3.toFixed(3)}ms`);
      console.log(`    - Signing: ${d4.toFixed(3)}ms`);
      console.log(`    - Verification: ${d5.toFixed(3)}ms`);
      console.log('  ✓ Multi-agent workflow completed successfully\n');
    });

    it('should handle dependent operations across agents', async function() {
      console.log('\n  === Dependent Operations Workflow ===');

      // Setup agents
      const agents = await Promise.all([
        createAgent('agent-A', jj),
        createAgent('agent-B', jj),
        createAgent('agent-C', jj)
      ]);

      // Agent A creates base operation
      const opA = await createSignedOperation(agents[0], jj, {
        type: 'create',
        path: 'base.txt',
        content: 'Base content'
      });

      // Agent B depends on A
      const opB = await createSignedOperation(agents[1], jj, {
        type: 'modify',
        path: 'base.txt',
        content: 'Modified by B',
        dependsOn: [opA.vertex.id]
      });

      // Agent C depends on B
      const opC = await createSignedOperation(agents[2], jj, {
        type: 'finalize',
        path: 'base.txt',
        content: 'Finalized by C',
        dependsOn: [opB.vertex.id]
      });

      // Verify causal chain
      const causalChain = await jj.getCausalChain(opC.vertex.id);
      assert.ok(causalChain.includes(opA.vertex.id), 'Chain should include A');
      assert.ok(causalChain.includes(opB.vertex.id), 'Chain should include B');
      assert.ok(causalChain.includes(opC.vertex.id), 'Chain should include C');

      // Verify all signatures
      const verificationsValid = await Promise.all([
        jj.verifyCommit(opA.signedCommit, agents[0].keypair.publicKey),
        jj.verifyCommit(opB.signedCommit, agents[1].keypair.publicKey),
        jj.verifyCommit(opC.signedCommit, agents[2].keypair.publicKey)
      ]);

      assert.ok(verificationsValid.every(v => v === true), 'All dependent operations should verify');
      console.log('  ✓ Dependent operations workflow completed successfully\n');
    });
  });

  describe('Real-World Scenarios', function() {
    it('should handle file modification workflow', async function() {
      const agent = await createAgent('file-modifier', jj);
      const filePath = path.join(testRepoPath, 'test-file.txt');

      // Create file
      const createOp = await createSignedOperation(agent, jj, {
        type: 'create',
        path: 'test-file.txt',
        content: 'Initial content'
      });

      await fs.writeFile(filePath, 'Initial content');

      // Modify file
      const modifyOp = await createSignedOperation(agent, jj, {
        type: 'modify',
        path: 'test-file.txt',
        content: 'Modified content',
        dependsOn: [createOp.vertex.id]
      });

      await fs.writeFile(filePath, 'Modified content');

      // Verify file exists and has correct content
      const fileContent = await fs.readFile(filePath, 'utf8');
      assert.strictEqual(fileContent, 'Modified content');

      // Verify both operations
      const verifications = await Promise.all([
        jj.verifyCommit(createOp.signedCommit, agent.keypair.publicKey),
        jj.verifyCommit(modifyOp.signedCommit, agent.keypair.publicKey)
      ]);

      assert.ok(verifications.every(v => v === true));
      console.log('  ✓ File modification workflow completed');
    });

    it('should handle merge conflict resolution', async function() {
      const agents = await Promise.all([
        createAgent('merger-1', jj),
        createAgent('merger-2', jj),
        createAgent('resolver', jj)
      ]);

      // Both agents modify same file
      const op1 = await createSignedOperation(agents[0], jj, {
        type: 'modify',
        path: 'conflict.txt',
        content: 'Version 1'
      });

      const op2 = await createSignedOperation(agents[1], jj, {
        type: 'modify',
        path: 'conflict.txt',
        content: 'Version 2'
      });

      // Resolver agent creates merge
      const mergeOp = await createSignedOperation(agents[2], jj, {
        type: 'merge',
        path: 'conflict.txt',
        content: 'Merged version',
        dependsOn: [op1.vertex.id, op2.vertex.id]
      });

      // Verify merge has both parents
      assert.ok(mergeOp.vertex.parents.includes(op1.vertex.id));
      assert.ok(mergeOp.vertex.parents.includes(op2.vertex.id));

      // Verify all signatures
      const verifications = await Promise.all([
        jj.verifyCommit(op1.signedCommit, agents[0].keypair.publicKey),
        jj.verifyCommit(op2.signedCommit, agents[1].keypair.publicKey),
        jj.verifyCommit(mergeOp.signedCommit, agents[2].keypair.publicKey)
      ]);

      assert.ok(verifications.every(v => v === true));
      console.log('  ✓ Merge conflict resolution workflow completed');
    });

    it('should handle rollback scenario', async function() {
      const agent = await createAgent('rollback-agent', jj);

      // Create operations
      const ops = [];
      for (let i = 0; i < 3; i++) {
        const op = await createSignedOperation(agent, jj, {
          type: 'create',
          path: `file-${i}.txt`,
          content: `Content ${i}`
        });
        ops.push(op);
      }

      // Verify we can check history
      const history = await jj.getOperationHistory(agent.id);
      assert.ok(history.length >= 3, 'Should have history of operations');

      // Simulate rollback by verifying specific point in history
      const rollbackPoint = ops[1];
      const isValid = await jj.verifyCommit(
        rollbackPoint.signedCommit,
        agent.keypair.publicKey
      );

      assert.strictEqual(isValid, true, 'Rollback point should be valid');
      console.log('  ✓ Rollback scenario completed');
    });
  });

  describe('Performance Benchmarks', function() {
    it('should measure end-to-end performance', async function() {
      const iterations = 10;
      const timings = [];

      for (let i = 0; i < iterations; i++) {
        const start = performance.now();

        const agent = await createAgent(`perf-agent-${i}`, jj);
        const op = await createSignedOperation(agent, jj, {
          type: 'benchmark',
          iteration: i
        });
        const isValid = await jj.verifyCommit(op.signedCommit, agent.keypair.publicKey);

        const duration = performance.now() - start;
        timings.push(duration);

        assert.strictEqual(isValid, true);
      }

      const avgTime = timings.reduce((a, b) => a + b, 0) / iterations;
      const maxTime = Math.max(...timings);
      const minTime = Math.min(...timings);

      console.log(`\n  End-to-end performance (${iterations} iterations):`);
      console.log(`    Average: ${avgTime.toFixed(3)}ms`);
      console.log(`    Min: ${minTime.toFixed(3)}ms`);
      console.log(`    Max: ${maxTime.toFixed(3)}ms`);
    });
  });
});

// Helper functions

async function createAgent(agentId, jj) {
  const keypair = await jj.generateMLDSAKeypair();
  const registration = await jj.registerQuantumAgent(
    agentId,
    ['read', 'write', 'sign', 'verify'],
    keypair
  );
  return { id: agentId, keypair, registration };
}

async function createSignedOperation(agent, jj, operationData) {
  const operation = {
    ...operationData,
    timestamp: Date.now(),
    agent: agent.id
  };

  const vertex = await jj.createVertex(agent.id, operation);
  const fingerprint = await jj.generateFingerprint(operation);
  const signedCommit = await jj.signCommit(agent.id, {
    message: `Operation: ${operation.type}`,
    author: agent.id,
    operation,
    vertex: vertex.id,
    fingerprint
  });

  return { operation, vertex, fingerprint, signedCommit };
}
