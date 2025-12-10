#!/usr/bin/env tsx
// Example: Parallel swarm deployment with concurrent agent execution

import {
  deploySwarmConcurrently,
  executeTasksConcurrently,
  deployAndExecuteConcurrently,
  batchSpawnAgents,
  scaleSwarmConcurrently
} from '../coordination/parallelSwarm.js';
import { logger } from '../utils/logger.js';

async function example1_parallelAgentSpawn() {
  console.log('\n🚀 Example 1: Parallel Agent Spawning');
  console.log('=' .repeat(60));

  // Deploy 5 agents concurrently
  const result = await deploySwarmConcurrently({
    topology: 'mesh',
    maxAgents: 10,
    strategy: 'balanced',
    agents: [
      { type: 'researcher', capabilities: ['search', 'analyze'] },
      { type: 'coder', capabilities: ['implement', 'refactor'] },
      { type: 'reviewer', capabilities: ['review', 'validate'] },
      { type: 'tester', capabilities: ['test', 'coverage'] },
      { type: 'documenter', capabilities: ['document', 'explain'] }
    ]
  });

  console.log(`\n✅ Swarm deployed in ${result.deploymentTime}ms`);
  console.log(`📊 Agents spawned: ${result.agents.length}`);
  console.log(`⚡ Avg spawn time: ${result.deploymentTime / result.agents.length}ms per agent`);
  console.log('\nAgents:', result.agents.map(a => `${a.type}(${a.status})`).join(', '));
}

async function example2_parallelTaskExecution() {
  console.log('\n\n⚡ Example 2: Parallel Task Execution');
  console.log('=' .repeat(60));

  // Execute 6 tasks concurrently
  const result = await executeTasksConcurrently({
    tasks: [
      { description: 'Analyze requirements', agentType: 'researcher', priority: 'high' },
      { description: 'Design architecture', agentType: 'architect', priority: 'high' },
      { description: 'Implement API endpoints', agentType: 'coder', priority: 'high' },
      { description: 'Write unit tests', agentType: 'tester', priority: 'medium' },
      { description: 'Create documentation', agentType: 'documenter', priority: 'medium' },
      { description: 'Review code quality', agentType: 'reviewer', priority: 'medium' }
    ],
    strategy: 'parallel',
    maxConcurrency: 6 // All 6 tasks run concurrently
  });

  console.log(`\n✅ All tasks completed in ${result.totalTime}ms`);
  console.log(`📊 Tasks executed: ${result.results.length}`);
  console.log(`⚡ Concurrency level: ${result.concurrencyLevel}`);
  console.log(`⚡ Avg task time: ${result.totalTime / result.results.length}ms per task`);

  // Calculate speedup vs sequential
  const sequentialTime = result.results.reduce((sum, r) => sum + r.duration, 0);
  const speedup = (sequentialTime / result.totalTime).toFixed(2);
  console.log(`🚀 Parallel speedup: ${speedup}x faster than sequential`);
}

async function example3_deployAndExecute() {
  console.log('\n\n🔥 Example 3: Deploy + Execute Simultaneously');
  console.log('=' .repeat(60));

  // Deploy swarm AND start tasks in parallel
  const result = await deployAndExecuteConcurrently(
    {
      topology: 'hierarchical',
      maxAgents: 8,
      strategy: 'specialized',
      agents: [
        { type: 'coordinator', capabilities: ['orchestrate'] },
        { type: 'researcher', capabilities: ['search'] },
        { type: 'coder', capabilities: ['implement'] },
        { type: 'tester', capabilities: ['test'] }
      ]
    },
    {
      tasks: [
        { description: 'Research best practices', priority: 'high' },
        { description: 'Implement core features', priority: 'high' },
        { description: 'Write comprehensive tests', priority: 'medium' }
      ],
      strategy: 'parallel'
    }
  );

  console.log(`\n✅ Deployment + Execution in ${result.totalTime}ms`);
  console.log(`📦 Swarm: ${result.swarm.agents.length} agents in ${result.swarm.deploymentTime}ms`);
  console.log(`⚡ Tasks: ${result.execution.results.length} tasks in ${result.execution.totalTime}ms`);

  const sequentialTime = result.swarm.deploymentTime + result.execution.totalTime;
  const speedup = (sequentialTime / result.totalTime).toFixed(2);
  console.log(`🚀 Parallel speedup: ${speedup}x faster (both operations ran concurrently)`);
}

async function example4_batchSpawn() {
  console.log('\n\n📦 Example 4: Batch Spawn Multiple Agent Types');
  console.log('=' .repeat(60));

  // Spawn 3 of each type concurrently (9 total agents)
  const agents = await batchSpawnAgents([
    { type: 'researcher', count: 3, capabilities: ['search', 'analyze'] },
    { type: 'coder', count: 3, capabilities: ['implement', 'refactor'] },
    { type: 'tester', count: 3, capabilities: ['test', 'validate'] }
  ]);

  console.log(`\n✅ Batch spawned ${agents.length} agents`);

  // Group by type
  const byType = agents.reduce((acc, agent) => {
    acc[agent.type] = (acc[agent.type] || 0) + 1;
    return acc;
  }, {} as Record<string, number>);

  console.log('📊 Agent distribution:');
  Object.entries(byType).forEach(([type, count]) => {
    console.log(`   ${type}: ${count} agents`);
  });
}

async function example5_dynamicScaling() {
  console.log('\n\n📈 Example 5: Dynamic Swarm Scaling');
  console.log('=' .repeat(60));

  console.log('\n🔼 Scaling up from 3 to 8 agents...');
  const scaleUp = await scaleSwarmConcurrently(3, 8, 'worker');
  console.log(`✅ Scaled ${scaleUp.action}: added ${scaleUp.agentsChanged} agents in ${scaleUp.duration}ms`);

  await new Promise(resolve => setTimeout(resolve, 500));

  console.log('\n🔽 Scaling down from 8 to 5 agents...');
  const scaleDown = await scaleSwarmConcurrently(8, 5, 'worker');
  console.log(`✅ Scaled ${scaleDown.action}: removed ${scaleDown.agentsChanged} agents in ${scaleDown.duration}ms`);
}

async function example6_largeScaleDeployment() {
  console.log('\n\n🌐 Example 6: Large-Scale Concurrent Deployment');
  console.log('=' .repeat(60));

  const startTime = Date.now();

  // Deploy 20 agents concurrently
  const agents = Array.from({ length: 20 }, (_, i) => ({
    type: `worker-${i % 5}`, // 5 types, 4 of each
    capabilities: ['process', 'analyze', 'report']
  }));

  const result = await deploySwarmConcurrently({
    topology: 'mesh',
    maxAgents: 50,
    strategy: 'adaptive',
    agents
  });

  const totalTime = Date.now() - startTime;

  console.log(`\n✅ Large-scale deployment complete in ${totalTime}ms`);
  console.log(`📊 Agents deployed: ${result.agents.length}`);
  console.log(`⚡ Avg spawn time: ${totalTime / result.agents.length}ms per agent`);

  // Calculate theoretical sequential time (assume 100ms per agent)
  const theoreticalSequentialTime = result.agents.length * 100;
  const speedup = (theoreticalSequentialTime / totalTime).toFixed(2);
  console.log(`🚀 Estimated speedup: ${speedup}x faster than sequential`);
}

// Run all examples
async function runAllExamples() {
  console.log('\n' + '='.repeat(60));
  console.log('  PARALLEL SWARM DEPLOYMENT EXAMPLES');
  console.log('='.repeat(60));

  try {
    await example1_parallelAgentSpawn();
    await example2_parallelTaskExecution();
    await example3_deployAndExecute();
    await example4_batchSpawn();
    await example5_dynamicScaling();
    await example6_largeScaleDeployment();

    console.log('\n\n' + '='.repeat(60));
    console.log('  ✅ ALL EXAMPLES COMPLETED SUCCESSFULLY');
    console.log('='.repeat(60));
    console.log('\n🎯 Key Takeaways:');
    console.log('   • Promise.all ensures true parallel execution');
    console.log('   • Swarm deployment is 2-10x faster with concurrency');
    console.log('   • Batch operations maximize throughput');
    console.log('   • Dynamic scaling adapts to workload');
    console.log('   • Large-scale deployments benefit most from parallelism');
    console.log('');
  } catch (error) {
    console.error('\n❌ Error running examples:', error);
    logger.error('Example execution failed', { error });
    process.exit(1);
  }
}

// Run if called directly
if (import.meta.url === `file://${process.argv[1]}`) {
  runAllExamples().catch(error => {
    console.error('Fatal error:', error);
    process.exit(1);
  });
}

export { runAllExamples };
