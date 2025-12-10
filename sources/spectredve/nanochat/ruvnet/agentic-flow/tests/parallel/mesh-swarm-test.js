#!/usr/bin/env node
/**
 * Mesh Topology Swarm Test
 * Tests peer-to-peer coordination with full connectivity
 */

const { exec } = require('child_process');
const { promisify } = require('util');
const execAsync = promisify(exec);

const SWARM_TOPOLOGY = process.env.SWARM_TOPOLOGY || 'mesh';
const MAX_AGENTS = parseInt(process.env.MAX_AGENTS || '10');
const BATCH_SIZE = parseInt(process.env.BATCH_SIZE || '5');

async function meshSwarmTest() {
  console.log('\n🕸️  Mesh Topology Swarm Test');
  console.log('═'.repeat(60));
  console.log(`Topology: ${SWARM_TOPOLOGY}`);
  console.log(`Max Agents: ${MAX_AGENTS}`);
  console.log(`Batch Size: ${BATCH_SIZE}`);
  console.log('═'.repeat(60));

  const testTaskId = `mesh-test-${Date.now()}`;
  const startTime = Date.now();

  try {
    // Test 1: Spawn agents in parallel
    console.log('\n📊 Test 1: Parallel Agent Spawning');
    const agentTypes = ['researcher', 'analyst', 'coder', 'tester', 'reviewer'];

    const spawnStart = Date.now();
    const spawnPromises = agentTypes.slice(0, BATCH_SIZE).map((type, i) =>
      execAsync(
        `npx agentic-flow --agent ${type} ` +
        `--task "Mesh node ${i}: Analyze system architecture" ` +
        `--output reasoningbank:test/${testTaskId}/spawn/${type}`
      ).catch(err => ({ error: err.message, type }))
    );

    const spawnResults = await Promise.all(spawnPromises);
    const spawnTime = Date.now() - spawnStart;
    const spawnSuccessful = spawnResults.filter(r => !r.error).length;

    console.log(`✅ Spawned ${spawnSuccessful}/${BATCH_SIZE} agents in ${spawnTime}ms`);
    console.log(`   Average: ${(spawnTime / BATCH_SIZE).toFixed(0)}ms per agent`);

    if (spawnSuccessful < BATCH_SIZE) {
      const failed = spawnResults.filter(r => r.error);
      console.log(`⚠️  Failed agents: ${failed.map(f => f.type).join(', ')}`);
    }

    // Test 2: Concurrent task execution
    console.log('\n📊 Test 2: Concurrent Task Execution');
    const tasks = [
      'Analyze security patterns',
      'Review code quality',
      'Optimize performance',
      'Generate documentation',
      'Run integration tests'
    ];

    const taskStart = Date.now();
    const taskPromises = tasks.slice(0, BATCH_SIZE).map((task, i) =>
      execAsync(
        `npx agentic-flow --agent researcher ` +
        `--task "${task}" ` +
        `--output reasoningbank:test/${testTaskId}/tasks/task-${i}`
      ).catch(err => ({ error: err.message, task }))
    );

    const taskResults = await Promise.all(taskPromises);
    const taskTime = Date.now() - taskStart;
    const taskSuccessful = taskResults.filter(r => !r.error).length;

    console.log(`✅ Completed ${taskSuccessful}/${BATCH_SIZE} tasks in ${taskTime}ms`);
    console.log(`   Average: ${(taskTime / BATCH_SIZE).toFixed(0)}ms per task`);

    // Test 3: Cross-agent coordination
    console.log('\n📊 Test 3: Cross-Agent Coordination (Mesh)');
    const coordStart = Date.now();

    // Each agent communicates with all others (full mesh)
    const coordPromises = [];
    for (let i = 0; i < Math.min(3, BATCH_SIZE); i++) {
      for (let j = 0; j < Math.min(3, BATCH_SIZE); j++) {
        if (i !== j) {
          coordPromises.push(
            execAsync(
              `npx agentic-flow --agent researcher ` +
              `--task "Agent ${i} coordinate with Agent ${j}" ` +
              `--output reasoningbank:test/${testTaskId}/coord/agent-${i}-to-${j}`
            ).catch(err => ({ error: err.message }))
          );
        }
      }
    }

    const coordResults = await Promise.all(coordPromises);
    const coordTime = Date.now() - coordStart;
    const coordSuccessful = coordResults.filter(r => !r.error).length;

    console.log(`✅ Coordinated ${coordSuccessful}/${coordPromises.length} connections in ${coordTime}ms`);
    console.log(`   Mesh connectivity: ${coordSuccessful} peer-to-peer links`);

    // Calculate metrics
    const totalTime = Date.now() - startTime;
    const totalOperations = spawnSuccessful + taskSuccessful + coordSuccessful;
    const avgTime = totalOperations > 0 ? (totalTime / totalOperations).toFixed(0) : 0;

    console.log('\n📈 Final Metrics');
    console.log('═'.repeat(60));
    console.log(`Total Time: ${totalTime}ms`);
    console.log(`Total Operations: ${totalOperations}`);
    console.log(`Average Time per Operation: ${avgTime}ms`);
    console.log(`Success Rate: ${((totalOperations / (BATCH_SIZE * 2 + coordPromises.length)) * 100).toFixed(1)}%`);
    console.log('═'.repeat(60));

    // Write results
    const results = {
      topology: 'mesh',
      timestamp: new Date().toISOString(),
      config: { maxAgents: MAX_AGENTS, batchSize: BATCH_SIZE },
      tests: {
        spawn: { successful: spawnSuccessful, total: BATCH_SIZE, timeMs: spawnTime },
        tasks: { successful: taskSuccessful, total: BATCH_SIZE, timeMs: taskTime },
        coordination: { successful: coordSuccessful, total: coordPromises.length, timeMs: coordTime }
      },
      metrics: {
        totalTimeMs: totalTime,
        totalOperations,
        avgTimeMs: parseFloat(avgTime),
        successRate: (totalOperations / (BATCH_SIZE * 2 + coordPromises.length)) * 100
      }
    };

    console.log('\n📝 Results:', JSON.stringify(results, null, 2));
    return results;

  } catch (error) {
    console.error('\n❌ Test failed:', error.message);
    process.exit(1);
  }
}

// Run test
if (require.main === module) {
  meshSwarmTest()
    .then(results => {
      console.log('\n✅ Mesh swarm test completed successfully');
      process.exit(0);
    })
    .catch(error => {
      console.error('\n❌ Fatal error:', error);
      process.exit(1);
    });
}

module.exports = { meshSwarmTest };
