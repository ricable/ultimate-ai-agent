/**
 * QuantumDAG Integration Tests
 *
 * Tests QuantumDAG coordination features including:
 * - Vertex creation and management
 * - DAG tips tracking
 * - Agent registration in quantum space
 * - Parallel operation coordination
 */

const { AgenticJujutsu } = require('../../index.js');
const assert = require('assert');
const { performance } = require('perf_hooks');

describe('QuantumDAG Integration', function() {
  this.timeout(10000);

  let jj;
  const testRepoPath = '/tmp/quantum-dag-test';

  beforeEach(async function() {
    // Clean up test repo
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

  describe('Vertex Creation', function() {
    it('should create a vertex for new operation', async function() {
      const agentId = 'test-agent-001';
      const operation = {
        type: 'create',
        path: 'test.txt',
        content: 'Hello quantum world'
      };

      const start = performance.now();
      const vertex = await jj.createVertex(agentId, operation);
      const duration = performance.now() - start;

      assert.ok(vertex, 'Vertex should be created');
      assert.strictEqual(typeof vertex.id, 'string', 'Vertex should have ID');
      assert.strictEqual(vertex.agentId, agentId, 'Vertex should track agent');
      assert.deepStrictEqual(vertex.operation, operation, 'Vertex should store operation');
      assert.ok(Array.isArray(vertex.parents), 'Vertex should have parents array');
      assert.ok(vertex.timestamp, 'Vertex should have timestamp');

      console.log(`  ✓ Vertex creation time: ${duration.toFixed(3)}ms`);
      assert.ok(duration < 100, 'Vertex creation should be fast (<100ms)');
    });

    it('should link vertices with parent relationships', async function() {
      const agent1 = 'agent-001';
      const agent2 = 'agent-002';

      // Create first vertex
      const vertex1 = await jj.createVertex(agent1, {
        type: 'create',
        path: 'file1.txt'
      });

      // Create second vertex that depends on first
      const vertex2 = await jj.createVertex(agent2, {
        type: 'modify',
        path: 'file1.txt',
        dependsOn: [vertex1.id]
      });

      assert.ok(vertex2.parents.includes(vertex1.id),
        'Second vertex should reference first as parent');
    });

    it('should handle multiple parallel vertices', async function() {
      const agents = ['agent-001', 'agent-002', 'agent-003', 'agent-004'];
      const operations = agents.map((agent, i) => ({
        agentId: agent,
        operation: {
          type: 'create',
          path: `file${i}.txt`,
          content: `Content ${i}`
        }
      }));

      const start = performance.now();
      const vertices = await Promise.all(
        operations.map(op => jj.createVertex(op.agentId, op.operation))
      );
      const duration = performance.now() - start;

      assert.strictEqual(vertices.length, 4, 'Should create 4 vertices');
      vertices.forEach((vertex, i) => {
        assert.strictEqual(vertex.agentId, agents[i], 'Vertex should have correct agent');
      });

      console.log(`  ✓ Parallel vertex creation (4 agents): ${duration.toFixed(3)}ms`);
      assert.ok(duration < 500, 'Parallel creation should be efficient');
    });
  });

  describe('DAG Tips Tracking', function() {
    it('should track DAG tips correctly', async function() {
      const tips1 = await jj.getDAGTips();
      assert.ok(Array.isArray(tips1), 'Tips should be array');

      // Create some vertices
      await jj.createVertex('agent-001', { type: 'create', path: 'a.txt' });
      await jj.createVertex('agent-002', { type: 'create', path: 'b.txt' });

      const tips2 = await jj.getDAGTips();
      assert.ok(tips2.length > tips1.length, 'Tips should increase after operations');
    });

    it('should update tips on vertex creation', async function() {
      const vertex1 = await jj.createVertex('agent-001', {
        type: 'create',
        path: 'test.txt'
      });

      const tips = await jj.getDAGTips();
      assert.ok(tips.some(tip => tip.id === vertex1.id),
        'New vertex should become a tip');
    });

    it('should handle tip convergence', async function() {
      // Create two parallel branches
      const v1 = await jj.createVertex('agent-001', { type: 'create', path: 'a.txt' });
      const v2 = await jj.createVertex('agent-002', { type: 'create', path: 'b.txt' });

      const tipsBefore = await jj.getDAGTips();
      assert.ok(tipsBefore.length >= 2, 'Should have at least 2 tips');

      // Create convergence point
      const v3 = await jj.createVertex('agent-003', {
        type: 'merge',
        dependsOn: [v1.id, v2.id]
      });

      const tipsAfter = await jj.getDAGTips();
      assert.ok(tipsAfter.some(tip => tip.id === v3.id),
        'Merge vertex should become new tip');
    });
  });

  describe('Agent Registration in Quantum Space', function() {
    it('should register agent in quantum coordinate system', async function() {
      const agentId = 'quantum-agent-001';
      const capabilities = ['read', 'write', 'execute'];

      const start = performance.now();
      const registration = await jj.registerQuantumAgent(agentId, capabilities);
      const duration = performance.now() - start;

      assert.ok(registration, 'Registration should succeed');
      assert.strictEqual(registration.agentId, agentId);
      assert.deepStrictEqual(registration.capabilities, capabilities);
      assert.ok(registration.quantumCoordinates, 'Should have quantum coordinates');
      assert.strictEqual(typeof registration.quantumCoordinates.x, 'number');
      assert.strictEqual(typeof registration.quantumCoordinates.y, 'number');
      assert.strictEqual(typeof registration.quantumCoordinates.z, 'number');

      console.log(`  ✓ Agent registration time: ${duration.toFixed(3)}ms`);
      console.log(`  ✓ Quantum coordinates: (${registration.quantumCoordinates.x.toFixed(2)}, ` +
                  `${registration.quantumCoordinates.y.toFixed(2)}, ` +
                  `${registration.quantumCoordinates.z.toFixed(2)})`);
    });

    it('should track multiple agents in quantum space', async function() {
      const agents = [
        { id: 'agent-001', capabilities: ['read'] },
        { id: 'agent-002', capabilities: ['write'] },
        { id: 'agent-003', capabilities: ['execute'] },
        { id: 'agent-004', capabilities: ['read', 'write'] }
      ];

      const registrations = await Promise.all(
        agents.map(a => jj.registerQuantumAgent(a.id, a.capabilities))
      );

      assert.strictEqual(registrations.length, 4, 'Should register all agents');

      // Verify all have unique coordinates
      const coordinates = registrations.map(r => r.quantumCoordinates);
      const uniqueCoords = new Set(coordinates.map(c => `${c.x},${c.y},${c.z}`));
      assert.strictEqual(uniqueCoords.size, 4, 'Each agent should have unique quantum position');
    });

    it('should retrieve agent quantum state', async function() {
      const agentId = 'query-agent-001';
      await jj.registerQuantumAgent(agentId, ['read', 'write']);

      const state = await jj.getAgentQuantumState(agentId);
      assert.ok(state, 'Should retrieve quantum state');
      assert.strictEqual(state.agentId, agentId);
      assert.ok(state.quantumCoordinates);
      assert.ok(Array.isArray(state.activeOperations));
    });
  });

  describe('Parallel Operation Coordination', function() {
    it('should coordinate parallel agent operations', async function() {
      const agents = ['agent-001', 'agent-002', 'agent-003'];

      // Register all agents
      await Promise.all(
        agents.map((id, i) =>
          jj.registerQuantumAgent(id, [`capability-${i}`])
        )
      );

      // Execute parallel operations
      const start = performance.now();
      const operations = await Promise.all(
        agents.map((id, i) =>
          jj.createVertex(id, {
            type: 'parallel-op',
            path: `file${i}.txt`,
            content: `Parallel content ${i}`
          })
        )
      );
      const duration = performance.now() - start;

      assert.strictEqual(operations.length, 3, 'Should complete all operations');
      operations.forEach((op, i) => {
        assert.strictEqual(op.agentId, agents[i]);
      });

      console.log(`  ✓ Parallel coordination (3 agents): ${duration.toFixed(3)}ms`);
      console.log(`  ✓ Average per agent: ${(duration / 3).toFixed(3)}ms`);
    });

    it('should detect and resolve conflicts', async function() {
      const agent1 = 'agent-001';
      const agent2 = 'agent-002';

      await jj.registerQuantumAgent(agent1, ['write']);
      await jj.registerQuantumAgent(agent2, ['write']);

      // Create conflicting operations on same file
      const v1 = await jj.createVertex(agent1, {
        type: 'modify',
        path: 'shared.txt',
        content: 'Version 1'
      });

      const v2 = await jj.createVertex(agent2, {
        type: 'modify',
        path: 'shared.txt',
        content: 'Version 2'
      });

      // Detect conflict
      const conflicts = await jj.detectConflicts([v1.id, v2.id]);
      assert.ok(conflicts.length > 0, 'Should detect conflict');
      assert.strictEqual(conflicts[0].path, 'shared.txt');
      assert.ok(conflicts[0].vertices.includes(v1.id));
      assert.ok(conflicts[0].vertices.includes(v2.id));
    });

    it('should maintain causality in distributed operations', async function() {
      // Create causal chain: A -> B -> C
      const v1 = await jj.createVertex('agent-001', {
        type: 'create',
        path: 'base.txt'
      });

      const v2 = await jj.createVertex('agent-002', {
        type: 'modify',
        path: 'base.txt',
        dependsOn: [v1.id]
      });

      const v3 = await jj.createVertex('agent-003', {
        type: 'finalize',
        path: 'base.txt',
        dependsOn: [v2.id]
      });

      // Verify causal ordering
      const causalChain = await jj.getCausalChain(v3.id);
      assert.ok(causalChain.includes(v1.id), 'Chain should include first vertex');
      assert.ok(causalChain.includes(v2.id), 'Chain should include second vertex');
      assert.ok(causalChain.includes(v3.id), 'Chain should include final vertex');

      // Verify ordering
      const idx1 = causalChain.indexOf(v1.id);
      const idx2 = causalChain.indexOf(v2.id);
      const idx3 = causalChain.indexOf(v3.id);
      assert.ok(idx1 < idx2 && idx2 < idx3, 'Causal order should be preserved');
    });
  });

  describe('Performance Benchmarks', function() {
    it('should handle high-throughput vertex creation', async function() {
      const operationCount = 100;
      const agents = Array.from({ length: 10 }, (_, i) => `bench-agent-${i}`);

      const start = performance.now();
      const vertices = [];

      for (let i = 0; i < operationCount; i++) {
        const agentId = agents[i % agents.length];
        const vertex = await jj.createVertex(agentId, {
          type: 'benchmark',
          iteration: i
        });
        vertices.push(vertex);
      }

      const duration = performance.now() - start;
      const opsPerSecond = (operationCount / duration) * 1000;

      console.log(`  ✓ Created ${operationCount} vertices in ${duration.toFixed(2)}ms`);
      console.log(`  ✓ Throughput: ${opsPerSecond.toFixed(0)} ops/sec`);
      console.log(`  ✓ Average latency: ${(duration / operationCount).toFixed(3)}ms`);

      assert.strictEqual(vertices.length, operationCount);
      assert.ok(opsPerSecond > 100, 'Should handle >100 ops/sec');
    });

    it('should scale with concurrent agents', async function() {
      const agentCounts = [1, 5, 10, 20];
      const results = [];

      for (const count of agentCounts) {
        const agents = Array.from({ length: count }, (_, i) => `scale-agent-${i}`);

        const start = performance.now();
        await Promise.all(
          agents.map(id => jj.createVertex(id, { type: 'scale-test' }))
        );
        const duration = performance.now() - start;

        results.push({ agents: count, time: duration });
        console.log(`  ✓ ${count} agents: ${duration.toFixed(2)}ms`);
      }

      // Verify sub-linear scaling (good parallelization)
      const ratio = results[3].time / results[0].time;
      assert.ok(ratio < 20, 'Should scale sub-linearly (20x agents should take <20x time)');
    });
  });

  describe('Error Handling', function() {
    it('should handle invalid agent registration', async function() {
      await assert.rejects(
        async () => {
          await jj.registerQuantumAgent(null, ['read']);
        },
        /Invalid agent ID/,
        'Should reject null agent ID'
      );

      await assert.rejects(
        async () => {
          await jj.registerQuantumAgent('agent-001', null);
        },
        /Invalid capabilities/,
        'Should reject null capabilities'
      );
    });

    it('should handle missing parent vertices', async function() {
      await assert.rejects(
        async () => {
          await jj.createVertex('agent-001', {
            type: 'modify',
            dependsOn: ['non-existent-vertex-id']
          });
        },
        /Parent vertex not found/,
        'Should reject invalid parent reference'
      );
    });

    it('should handle concurrent modifications gracefully', async function() {
      const agentId = 'concurrent-agent';
      await jj.registerQuantumAgent(agentId, ['write']);

      // Simulate concurrent operations
      const operations = Array.from({ length: 10 }, (_, i) =>
        jj.createVertex(agentId, {
          type: 'concurrent',
          iteration: i
        })
      );

      const results = await Promise.allSettled(operations);
      const succeeded = results.filter(r => r.status === 'fulfilled');

      assert.ok(succeeded.length > 0, 'At least some operations should succeed');
      console.log(`  ✓ ${succeeded.length}/10 concurrent operations succeeded`);
    });
  });
});
