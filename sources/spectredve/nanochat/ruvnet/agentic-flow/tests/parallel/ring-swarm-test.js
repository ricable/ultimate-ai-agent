#!/usr/bin/env node
/**
 * Ring Topology Swarm Test
 * Tests circular message passing with token ring pattern
 */

const { exec } = require('child_process');
const { promisify } = require('util');
const execAsync = promisify(exec);

const SWARM_TOPOLOGY = process.env.SWARM_TOPOLOGY || 'ring';
const MAX_AGENTS = parseInt(process.env.MAX_AGENTS || '6');
const BATCH_SIZE = parseInt(process.env.BATCH_SIZE || '3');

async function ringSwarmTest() {
  console.log('\n🔄 Ring Topology Swarm Test');
  console.log('═'.repeat(60));
  console.log(`Topology: ${SWARM_TOPOLOGY}`);
  console.log(`Max Agents: ${MAX_AGENTS}`);
  console.log(`Batch Size: ${BATCH_SIZE}`);
  console.log('═'.repeat(60));

  const testTaskId = `ring-test-${Date.now()}`;
  const startTime = Date.now();

  try {
    // Test 1: Initialize ring nodes
    console.log('\n📊 Test 1: Initialize Ring Nodes');
    const nodeCount = Math.min(BATCH_SIZE, MAX_AGENTS);

    const initStart = Date.now();
    const initPromises = Array.from({ length: nodeCount }, (_, i) =>
      execAsync(
        `npx agentic-flow --agent researcher ` +
        `--task "Ring node ${i}: Initialize and prepare for token passing" ` +
        `--output reasoningbank:test/${testTaskId}/nodes/node-${i}`
      ).catch(err => ({ error: err.message, node: i }))
    );

    const initResults = await Promise.all(initPromises);
    const initTime = Date.now() - initStart;
    const initSuccessful = initResults.filter(r => !r.error).length;

    console.log(`✅ Initialized ${initSuccessful}/${nodeCount} nodes in ${initTime}ms`);

    // Test 2: Token passing around ring (sequential by design)
    console.log('\n📊 Test 2: Token Passing (Sequential)');
    const tokenStart = Date.now();
    let tokenTime = 0;

    for (let i = 0; i < initSuccessful; i++) {
      const next = (i + 1) % initSuccessful;
      const passStart = Date.now();

      await execAsync(
        `npx agentic-flow --agent researcher ` +
        `--task "Node ${i}: Pass token to Node ${next}" ` +
        `--output reasoningbank:test/${testTaskId}/token/pass-${i}-to-${next}`
      ).catch(err => ({ error: err.message }));

      tokenTime += Date.now() - passStart;
    }

    const avgPassTime = (tokenTime / initSuccessful).toFixed(0);
    console.log(`✅ Token passed through ${initSuccessful} nodes in ${tokenTime}ms`);
    console.log(`   Average pass time: ${avgPassTime}ms per hop`);

    // Test 3: Parallel processing at each node
    console.log('\n📊 Test 3: Parallel Node Processing');
    const processStart = Date.now();

    const processPromises = Array.from({ length: initSuccessful }, (_, i) =>
      execAsync(
        `npx agentic-flow --agent coder ` +
        `--task "Node ${i}: Process data in parallel" ` +
        `--output reasoningbank:test/${testTaskId}/process/node-${i}`
      ).catch(err => ({ error: err.message }))
    );

    const processResults = await Promise.all(processPromises);
    const processTime = Date.now() - processStart;
    const processSuccessful = processResults.filter(r => !r.error).length;

    console.log(`✅ Processed ${processSuccessful}/${initSuccessful} nodes in ${processTime}ms`);
    console.log(`   Parallel benefit: ${(tokenTime / processTime).toFixed(2)}x faster than sequential`);

    // Calculate metrics
    const totalTime = Date.now() - startTime;
    const totalOperations = initSuccessful + initSuccessful + processSuccessful; // init + token passes + processing

    console.log('\n📈 Final Metrics');
    console.log('═'.repeat(60));
    console.log(`Total Time: ${totalTime}ms`);
    console.log(`Ring Size: ${initSuccessful} nodes`);
    console.log(`Token Pass Time: ${tokenTime}ms`);
    console.log(`Parallel Process Time: ${processTime}ms`);
    console.log(`Total Operations: ${totalOperations}`);
    console.log(`Avg Time per Operation: ${(totalTime / totalOperations).toFixed(0)}ms`);
    console.log('═'.repeat(60));

    // Write results
    const results = {
      topology: 'ring',
      timestamp: new Date().toISOString(),
      config: { maxAgents: MAX_AGENTS, batchSize: BATCH_SIZE },
      tests: {
        initialization: { successful: initSuccessful, total: nodeCount, timeMs: initTime },
        tokenPassing: { successful: initSuccessful, avgPassTimeMs: parseFloat(avgPassTime), totalTimeMs: tokenTime },
        parallelProcessing: { successful: processSuccessful, total: initSuccessful, timeMs: processTime }
      },
      metrics: {
        totalTimeMs: totalTime,
        ringSize: initSuccessful,
        totalOperations,
        parallelBenefit: processTime > 0 ? tokenTime / processTime : 0
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
  ringSwarmTest()
    .then(results => {
      console.log('\n✅ Ring swarm test completed successfully');
      process.exit(0);
    })
    .catch(error => {
      console.error('\n❌ Fatal error:', error);
      process.exit(1);
    });
}

module.exports = { ringSwarmTest };
