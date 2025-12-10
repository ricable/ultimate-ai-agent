#!/usr/bin/env node
/**
 * Hierarchical Topology Swarm Test
 * Tests coordinator-worker pattern with delegation
 */

const { exec } = require('child_process');
const { promisify } = require('util');
const execAsync = promisify(exec);

const SWARM_TOPOLOGY = process.env.SWARM_TOPOLOGY || 'hierarchical';
const MAX_AGENTS = parseInt(process.env.MAX_AGENTS || '8');
const BATCH_SIZE = parseInt(process.env.BATCH_SIZE || '4');

async function hierarchicalSwarmTest() {
  console.log('\n🏛️  Hierarchical Topology Swarm Test');
  console.log('═'.repeat(60));
  console.log(`Topology: ${SWARM_TOPOLOGY}`);
  console.log(`Max Agents: ${MAX_AGENTS}`);
  console.log(`Batch Size: ${BATCH_SIZE}`);
  console.log('═'.repeat(60));

  const testTaskId = `hierarchical-test-${Date.now()}`;
  const startTime = Date.now();

  try {
    // Level 1: Coordinator decomposes task
    console.log('\n📊 Level 1: Coordinator Task Decomposition');
    const coordStart = Date.now();

    const coordResult = await execAsync(
      `npx agentic-flow --agent task-orchestrator ` +
      `--task "Decompose: Build a full-stack web application" ` +
      `--output reasoningbank:test/${testTaskId}/coordinator`
    ).catch(err => ({ error: err.message }));

    const coordTime = Date.now() - coordStart;
    console.log(`✅ Coordinator completed in ${coordTime}ms`);

    // Level 2: Spawn specialized workers in parallel
    console.log('\n📊 Level 2: Parallel Worker Spawning');
    const workerTypes = ['researcher', 'coder', 'tester', 'reviewer'];
    const workerTasks = [
      'Research best practices',
      'Implement backend API',
      'Create test suite',
      'Review code quality'
    ];

    const workerStart = Date.now();
    const workerPromises = workerTypes.slice(0, BATCH_SIZE).map((type, i) =>
      execAsync(
        `npx agentic-flow --agent ${type} ` +
        `--task "${workerTasks[i]}" ` +
        `--output reasoningbank:test/${testTaskId}/workers/worker-${i}`
      ).catch(err => ({ error: err.message, type }))
    );

    const workerResults = await Promise.all(workerPromises);
    const workerTime = Date.now() - workerStart;
    const workerSuccessful = workerResults.filter(r => !r.error).length;

    console.log(`✅ Spawned ${workerSuccessful}/${BATCH_SIZE} workers in ${workerTime}ms`);
    console.log(`   Parallel speedup: ${(coordTime + workerTime < coordTime * (BATCH_SIZE + 1)) ? 'YES' : 'NO'}`);

    // Level 3: Parallel review of worker outputs
    console.log('\n📊 Level 3: Parallel Output Review');
    const reviewStart = Date.now();

    const reviewPromises = workerTypes.slice(0, workerSuccessful).map((_, i) =>
      execAsync(
        `npx agentic-flow --agent reviewer ` +
        `--task "Review worker-${i} output" ` +
        `--output reasoningbank:test/${testTaskId}/reviews/worker-${i}`
      ).catch(err => ({ error: err.message }))
    );

    const reviewResults = await Promise.all(reviewPromises);
    const reviewTime = Date.now() - reviewStart;
    const reviewSuccessful = reviewResults.filter(r => !r.error).length;

    console.log(`✅ Reviewed ${reviewSuccessful}/${workerSuccessful} outputs in ${reviewTime}ms`);

    // Level 4: Coordinator synthesis
    console.log('\n📊 Level 4: Coordinator Synthesis');
    const synthesisStart = Date.now();

    const synthesisResult = await execAsync(
      `npx agentic-flow --agent task-orchestrator ` +
      `--task "Synthesize all worker outputs and reviews" ` +
      `--output reasoningbank:test/${testTaskId}/synthesis`
    ).catch(err => ({ error: err.message }));

    const synthesisTime = Date.now() - synthesisStart;
    console.log(`✅ Synthesis completed in ${synthesisTime}ms`);

    // Calculate metrics
    const totalTime = Date.now() - startTime;
    const totalOperations = 1 + workerSuccessful + reviewSuccessful + 1; // coord + workers + reviews + synthesis
    const avgTime = (totalTime / totalOperations).toFixed(0);

    // Estimate sequential time (if done one after another)
    const estimatedSequentialTime = coordTime + (workerTime * BATCH_SIZE) + (reviewTime * workerSuccessful) + synthesisTime;
    const speedup = (estimatedSequentialTime / totalTime).toFixed(2);

    console.log('\n📈 Final Metrics');
    console.log('═'.repeat(60));
    console.log(`Total Time: ${totalTime}ms`);
    console.log(`Estimated Sequential Time: ${estimatedSequentialTime}ms`);
    console.log(`Parallel Speedup: ${speedup}x`);
    console.log(`Total Operations: ${totalOperations}`);
    console.log(`Average Time per Operation: ${avgTime}ms`);
    console.log('═'.repeat(60));

    // Write results
    const results = {
      topology: 'hierarchical',
      timestamp: new Date().toISOString(),
      config: { maxAgents: MAX_AGENTS, batchSize: BATCH_SIZE },
      levels: {
        coordinator: { timeMs: coordTime, success: !coordResult.error },
        workers: { successful: workerSuccessful, total: BATCH_SIZE, timeMs: workerTime },
        reviews: { successful: reviewSuccessful, total: workerSuccessful, timeMs: reviewTime },
        synthesis: { timeMs: synthesisTime, success: !synthesisResult.error }
      },
      metrics: {
        totalTimeMs: totalTime,
        estimatedSequentialTimeMs: estimatedSequentialTime,
        speedup: parseFloat(speedup),
        totalOperations,
        avgTimeMs: parseFloat(avgTime)
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
  hierarchicalSwarmTest()
    .then(results => {
      console.log('\n✅ Hierarchical swarm test completed successfully');
      process.exit(0);
    })
    .catch(error => {
      console.error('\n❌ Fatal error:', error);
      process.exit(1);
    });
}

module.exports = { hierarchicalSwarmTest };
