// Example: QUIC-enabled Mesh Swarm for Distributed Computation
// Demonstrates peer-to-peer agent communication with QUIC transport

import { initSwarm } from '../src/swarm/index.js';
import { logger } from '../src/utils/logger.js';

async function main() {
  logger.info('Initializing QUIC-enabled mesh swarm...');

  // Initialize swarm with QUIC transport
  const swarm = await initSwarm({
    swarmId: 'compute-mesh',
    topology: 'mesh',
    transport: 'quic',
    maxAgents: 5,
    quicPort: 4433,
    enableFallback: true
  });

  logger.info('Swarm initialized', swarm.getStats());

  // Register compute agents
  const agents = [];
  for (let i = 1; i <= 5; i++) {
    const agent = {
      id: `compute-${i}`,
      role: 'worker' as const,
      host: 'localhost',
      port: 4433 + i,
      capabilities: ['compute', 'analyze', 'aggregate']
    };

    await swarm.registerAgent(agent);
    agents.push(agent);
    logger.info(`Registered agent: ${agent.id}`);
  }

  // Wait for state synchronization
  await new Promise(resolve => setTimeout(resolve, 2000));

  // Get swarm statistics
  const stats = await swarm.getStats();
  logger.info('Swarm statistics:', {
    swarmId: stats.swarmId,
    topology: stats.topology,
    transport: stats.transport,
    totalAgents: stats.coordinatorStats?.totalAgents,
    quicEnabled: stats.quicAvailable
  });

  // Simulate distributed computation
  logger.info('Starting distributed computation...');

  // In a real scenario, you would send tasks to agents here
  // For example:
  // await coordinator.broadcast({
  //   id: 'task-1',
  //   from: 'controller',
  //   type: 'task',
  //   payload: { operation: 'compute', data: [...] },
  //   timestamp: Date.now()
  // });

  // Wait for computation to complete
  await new Promise(resolve => setTimeout(resolve, 5000));

  // Get final statistics
  const finalStats = await swarm.getStats();
  logger.info('Final statistics:', finalStats);

  // Cleanup
  logger.info('Shutting down swarm...');
  await swarm.shutdown();
  logger.info('Swarm shutdown complete');
}

main().catch(error => {
  logger.error('Error in mesh swarm example', { error });
  process.exit(1);
});
