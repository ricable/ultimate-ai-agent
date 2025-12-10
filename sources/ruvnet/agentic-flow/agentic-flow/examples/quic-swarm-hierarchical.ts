// Example: QUIC-enabled Hierarchical Swarm for Task Distribution
// Demonstrates coordinator-worker architecture with QUIC transport

import { initSwarm } from '../src/swarm/index.js';
import { logger } from '../src/utils/logger.js';

async function main() {
  logger.info('Initializing QUIC-enabled hierarchical swarm...');

  // Initialize swarm with QUIC transport
  const swarm = await initSwarm({
    swarmId: 'task-distribution',
    topology: 'hierarchical',
    transport: 'quic',
    maxAgents: 20,
    quicPort: 4433,
    enableFallback: true
  });

  logger.info('Swarm initialized', await swarm.getStats());

  // Register coordinator agents
  logger.info('Registering coordinators...');
  for (let i = 1; i <= 3; i++) {
    await swarm.registerAgent({
      id: `coordinator-${i}`,
      role: 'coordinator',
      host: 'localhost',
      port: 4433 + i,
      capabilities: ['orchestrate', 'aggregate', 'monitor']
    });
    logger.info(`Registered coordinator: coordinator-${i}`);
  }

  // Register worker agents
  logger.info('Registering workers...');
  for (let i = 1; i <= 15; i++) {
    await swarm.registerAgent({
      id: `worker-${i}`,
      role: 'worker',
      host: 'localhost',
      port: 4440 + i,
      capabilities: ['compute', 'process']
    });
    logger.info(`Registered worker: worker-${i}`);
  }

  // Wait for state synchronization
  await new Promise(resolve => setTimeout(resolve, 3000));

  // Get swarm statistics
  const stats = await swarm.getStats();
  logger.info('Swarm statistics:', {
    swarmId: stats.swarmId,
    topology: stats.topology,
    transport: stats.transport,
    totalAgents: stats.coordinatorStats?.totalAgents,
    activeAgents: stats.coordinatorStats?.activeAgents,
    quicEnabled: stats.quicAvailable,
    quicStats: stats.transportStats
  });

  // Simulate task distribution
  logger.info('Distributing tasks to workers via coordinators...');

  // In a real scenario, coordinators would distribute tasks:
  // - Coordinator receives tasks
  // - Distributes to workers based on load balancing
  // - Aggregates results from workers
  // - Reports back to caller

  // Wait for task processing
  await new Promise(resolve => setTimeout(resolve, 10000));

  // Get final statistics
  const finalStats = await swarm.getStats();
  logger.info('Final statistics:', {
    totalMessages: finalStats.coordinatorStats?.totalMessages,
    messagesPerSecond: finalStats.coordinatorStats?.messagesPerSecond,
    averageLatency: finalStats.coordinatorStats?.averageLatency
  });

  // Cleanup
  logger.info('Shutting down swarm...');
  await swarm.shutdown();
  logger.info('Swarm shutdown complete');
}

main().catch(error => {
  logger.error('Error in hierarchical swarm example', { error });
  process.exit(1);
});
