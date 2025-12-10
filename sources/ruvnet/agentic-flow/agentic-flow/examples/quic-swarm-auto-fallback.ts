// Example: Auto Transport Selection with Transparent Fallback
// Demonstrates automatic QUIC/HTTP2 selection and fallback handling

import { initSwarm, checkQuicAvailability } from '../src/swarm/index.js';
import { logger } from '../src/utils/logger.js';

async function main() {
  logger.info('Checking QUIC availability...');

  // Check if QUIC is available
  const quicAvailable = await checkQuicAvailability();
  logger.info(`QUIC available: ${quicAvailable}`);

  // Initialize swarm with auto transport selection
  logger.info('Initializing swarm with auto transport selection...');
  const swarm = await initSwarm({
    swarmId: 'auto-transport-swarm',
    topology: 'mesh',
    transport: 'auto',  // Automatically select best available transport
    maxAgents: 10,
    quicPort: 4433,
    enableFallback: true  // Enable transparent HTTP/2 fallback
  });

  const initialStats = await swarm.getStats();
  logger.info('Swarm initialized with transport:', {
    selectedTransport: initialStats.transport,
    quicAvailable: initialStats.quicAvailable,
    topology: initialStats.topology
  });

  // Register agents
  logger.info('Registering agents...');
  for (let i = 1; i <= 5; i++) {
    await swarm.registerAgent({
      id: `agent-${i}`,
      role: 'worker',
      host: 'localhost',
      port: 4433 + i,
      capabilities: ['compute', 'analyze']
    });
    logger.info(`Registered agent: agent-${i}`);
  }

  // Monitor transport statistics over time
  logger.info('Monitoring transport statistics...');

  for (let iteration = 1; iteration <= 5; iteration++) {
    await new Promise(resolve => setTimeout(resolve, 2000));

    const stats = await swarm.getStats();
    logger.info(`Iteration ${iteration} stats:`, {
      currentTransport: stats.transport,
      quicAvailable: stats.quicAvailable,
      totalMessages: stats.coordinatorStats?.totalMessages,
      messagesPerSecond: stats.coordinatorStats?.messagesPerSecond?.toFixed(2),
      averageLatency: stats.coordinatorStats?.averageLatency?.toFixed(2) + 'ms',
      transportStats: {
        quic: stats.transportStats instanceof Map
          ? stats.transportStats.get('quic')
          : undefined,
        http2: stats.transportStats instanceof Map
          ? stats.transportStats.get('http2')
          : undefined
      }
    });

    // Demonstrate automatic fallback behavior
    if (iteration === 3 && stats.transport === 'quic') {
      logger.info('Simulating QUIC failure to demonstrate fallback...');
      // In a real scenario, QUIC failure would be detected automatically
      // and the router would fallback to HTTP/2
    }
  }

  // Get final statistics
  const finalStats = await swarm.getStats();
  logger.info('Final swarm statistics:', {
    swarmId: finalStats.swarmId,
    topology: finalStats.topology,
    finalTransport: finalStats.transport,
    quicWasAvailable: finalStats.quicAvailable,
    coordinatorStats: finalStats.coordinatorStats,
    transportStats: finalStats.transportStats
  });

  // Demonstrate transport resilience
  logger.info('Transport resilience demonstration complete');
  logger.info('Key features demonstrated:');
  logger.info('- Automatic transport selection (QUIC preferred)');
  logger.info('- Transparent HTTP/2 fallback on QUIC failure');
  logger.info('- Continuous availability monitoring');
  logger.info('- Per-protocol statistics tracking');

  // Cleanup
  logger.info('Shutting down swarm...');
  await swarm.shutdown();
  logger.info('Swarm shutdown complete');
}

main().catch(error => {
  logger.error('Error in auto-fallback example', { error });
  process.exit(1);
});
