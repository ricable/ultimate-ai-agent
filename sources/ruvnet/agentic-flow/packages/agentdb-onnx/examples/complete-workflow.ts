#!/usr/bin/env tsx
/**
 * Complete Workflow Example: AgentDB + ONNX
 *
 * Demonstrates:
 * - Pattern storage and retrieval
 * - Episodic memory with self-critique
 * - Batch operations
 * - Performance optimization
 */

import { createONNXAgentDB } from '../src/index.js';
import { unlink } from 'fs/promises';

async function main() {
  console.log('🚀 AgentDB + ONNX Complete Workflow\n');

  // Step 1: Initialize
  console.log('1️⃣  Initializing AgentDB with ONNX embeddings...');
  const agentdb = await createONNXAgentDB({
    dbPath: './example-workflow.db',
    modelName: 'Xenova/all-MiniLM-L6-v2',
    useGPU: false, // Set to true for GPU acceleration
    batchSize: 32,
    cacheSize: 10000
  });
  console.log('✅ Initialized\n');

  // Step 2: Store reasoning patterns
  console.log('2️⃣  Storing reasoning patterns...');

  const patterns = [
    {
      taskType: 'debugging',
      approach: 'Start with logs, reproduce the issue, then binary search for the root cause',
      successRate: 0.92,
      tags: ['systematic', 'efficient'],
      domain: 'software-engineering'
    },
    {
      taskType: 'debugging',
      approach: 'Use debugger breakpoints to step through execution',
      successRate: 0.85,
      tags: ['interactive', 'detailed'],
      domain: 'software-engineering'
    },
    {
      taskType: 'optimization',
      approach: 'Profile first, identify bottlenecks, then optimize hot paths',
      successRate: 0.95,
      tags: ['data-driven', 'methodical'],
      domain: 'performance'
    },
    {
      taskType: 'api-design',
      approach: 'RESTful principles with versioning and clear documentation',
      successRate: 0.88,
      tags: ['standards', 'maintainable'],
      domain: 'architecture'
    },
    {
      taskType: 'testing',
      approach: 'Write unit tests first (TDD), then integration tests',
      successRate: 0.90,
      tags: ['test-driven', 'reliable'],
      domain: 'quality-assurance'
    }
  ];

  // Batch store for efficiency
  const patternIds = await agentdb.reasoningBank.storePatternsBatch(patterns);
  console.log(`✅ Stored ${patternIds.length} patterns\n`);

  // Step 3: Search for patterns
  console.log('3️⃣  Searching for debugging strategies...');

  const debugPatterns = await agentdb.reasoningBank.searchPatterns(
    'how to debug a memory leak in production',
    {
      k: 3,
      threshold: 0.5,
      filters: {
        taskType: 'debugging',
        minSuccessRate: 0.8
      }
    }
  );

  console.log(`Found ${debugPatterns.length} relevant patterns:`);
  debugPatterns.forEach((p, i) => {
    console.log(`\n   ${i + 1}. ${p.approach}`);
    console.log(`      Success rate: ${(p.successRate * 100).toFixed(1)}%`);
    console.log(`      Similarity:   ${(p.similarity * 100).toFixed(1)}%`);
    console.log(`      Tags:         ${p.tags?.join(', ')}`);
  });
  console.log();

  // Step 4: Store episodes with self-critique
  console.log('4️⃣  Storing reflexion episodes...');

  const episodes = [
    {
      sessionId: 'debug-session-1',
      task: 'Fix memory leak in Node.js application',
      reward: 0.95,
      success: true,
      input: 'Application consuming 2GB+ memory',
      output: 'Found event listener leak, fixed with proper cleanup',
      critique: 'Should have checked event listeners earlier. Profiling was key.',
      latencyMs: 1800,
      tokensUsed: 450
    },
    {
      sessionId: 'debug-session-1',
      task: 'Reproduce memory leak locally',
      reward: 0.88,
      success: true,
      input: 'Production memory leak',
      output: 'Reproduced with stress test script',
      critique: 'Reproduction helped a lot. Could have added heap snapshots.',
      latencyMs: 600,
      tokensUsed: 200
    },
    {
      sessionId: 'optimize-session-1',
      task: 'Optimize API response time',
      reward: 0.92,
      success: true,
      input: 'API taking 2-3 seconds per request',
      output: 'Added database indexes, reduced to 200ms',
      critique: 'Profiling showed N+1 queries. Database indexes solved it.',
      latencyMs: 1200,
      tokensUsed: 350
    },
    {
      sessionId: 'test-session-1',
      task: 'Write tests for authentication',
      reward: 0.85,
      success: true,
      input: 'JWT authentication module',
      output: 'Unit tests with 95% coverage',
      critique: 'TDD approach worked well. Could add more edge cases.',
      latencyMs: 900,
      tokensUsed: 300
    }
  ];

  const episodeIds = await agentdb.reflexionMemory.storeEpisodesBatch(episodes);
  console.log(`✅ Stored ${episodeIds.length} episodes\n`);

  // Step 5: Learn from past experiences
  console.log('5️⃣  Learning from past experiences...');

  const similarExperiences = await agentdb.reflexionMemory.retrieveRelevant(
    'how to fix performance issues in production',
    {
      k: 3,
      onlySuccesses: true,
      minReward: 0.85
    }
  );

  console.log(`Found ${similarExperiences.length} relevant experiences:`);
  similarExperiences.forEach((e, i) => {
    console.log(`\n   ${i + 1}. ${e.task}`);
    console.log(`      Reward:      ${(e.reward * 100).toFixed(1)}%`);
    console.log(`      Similarity:  ${(e.similarity * 100).toFixed(1)}%`);
    console.log(`      Critique:    ${e.critique?.substring(0, 80)}...`);
  });
  console.log();

  // Step 6: Get critique summary
  console.log('6️⃣  Getting critique summary for debugging tasks...');

  const critiques = await agentdb.reflexionMemory.getCritiqueSummary(
    'debugging memory issues',
    5
  );

  console.log(`Learned ${critiques.length} insights from past debugging:`);
  critiques.forEach((c, i) => {
    console.log(`   ${i + 1}. ${c}`);
  });
  console.log();

  // Step 7: Performance statistics
  console.log('7️⃣  Performance statistics:');

  const stats = agentdb.getStats();

  console.log(`\n   📊 Embeddings:`);
  console.log(`      Total:       ${stats.embedder.totalEmbeddings}`);
  console.log(`      Avg latency: ${stats.embedder.avgLatency.toFixed(2)}ms`);
  console.log(`      Cache hits:  ${(stats.embedder.cache.hitRate * 100).toFixed(1)}%`);
  console.log(`      Cache size:  ${stats.embedder.cache.size}/${stats.embedder.cache.maxSize}`);

  console.log(`\n   💾 Database:`);
  console.log(`      Backend:     ${stats.database?.backend || 'Unknown'}`);

  // Step 8: Real-world scenario simulation
  console.log('\n8️⃣  Simulating real-world agent workflow...\n');

  // Scenario: Agent needs to solve a new task
  const newTask = 'Optimize database query performance in production API';

  console.log(`   📝 New task: "${newTask}"\n`);

  // Step 8a: Retrieve relevant patterns
  console.log('   🔍 Searching for relevant patterns...');
  const relevantPatterns = await agentdb.reasoningBank.searchPatterns(
    newTask,
    { k: 2, threshold: 0.6 }
  );

  console.log(`   Found ${relevantPatterns.length} patterns:`);
  relevantPatterns.forEach((p, i) => {
    console.log(`      ${i + 1}. ${p.approach} (${(p.similarity * 100).toFixed(0)}% match)`);
  });

  // Step 8b: Retrieve similar episodes
  console.log('\n   📚 Searching for similar past experiences...');
  const relevantEpisodes = await agentdb.reflexionMemory.retrieveRelevant(
    newTask,
    { k: 2, onlySuccesses: true }
  );

  console.log(`   Found ${relevantEpisodes.length} experiences:`);
  relevantEpisodes.forEach((e, i) => {
    console.log(`      ${i + 1}. ${e.task} (${(e.similarity * 100).toFixed(0)}% match)`);
    console.log(`         Critique: ${e.critique?.substring(0, 60)}...`);
  });

  // Step 8c: Agent executes task with learned approach
  console.log('\n   ⚡ Agent executes with learned approach...');
  console.log('   ✅ Task completed successfully!\n');

  // Step 8d: Store new episode with self-critique
  console.log('   💭 Agent reflects on execution...');
  const newEpisodeId = await agentdb.reflexionMemory.storeEpisode({
    sessionId: 'new-session',
    task: newTask,
    reward: 0.94,
    success: true,
    input: 'Slow database queries in production',
    output: 'Added composite indexes, query time reduced by 10x',
    critique: 'Profiling showed table scans. Learned from past episode about indexing. Very effective approach.',
    latencyMs: 1500,
    tokensUsed: 400
  });

  console.log(`   ✅ Stored new episode (ID: ${newEpisodeId})\n`);

  // Step 9: Show improvement over time
  console.log('9️⃣  Self-improvement demonstration:\n');

  const taskStats = await agentdb.reflexionMemory.getTaskStats();
  console.log(`   Total episodes:  ${taskStats.totalEpisodes}`);
  console.log(`   Success rate:    ${(taskStats.successRate * 100).toFixed(1)}%`);
  console.log(`   Avg reward:      ${(taskStats.avgReward * 100).toFixed(1)}%`);
  console.log(`   Avg latency:     ${taskStats.avgLatency.toFixed(0)}ms`);

  // Cleanup
  console.log('\n🧹 Cleaning up...');
  await agentdb.close();

  try {
    await unlink('./example-workflow.db');
  } catch {}

  console.log('✅ Complete!\n');
  console.log('═══════════════════════════════════════════════════');
  console.log('Key Takeaways:');
  console.log('  • ONNX embeddings provide fast, local semantic search');
  console.log('  • Batch operations are 3-4x faster than sequential');
  console.log('  • ReasoningBank stores successful patterns');
  console.log('  • ReflexionMemory enables self-improvement');
  console.log('  • Cache provides significant speedup for common queries');
  console.log('═══════════════════════════════════════════════════\n');
}

main().catch(console.error);
