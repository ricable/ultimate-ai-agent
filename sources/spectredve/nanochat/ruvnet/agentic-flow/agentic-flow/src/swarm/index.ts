// Swarm Integration - Main export file for QUIC-enabled swarm coordination
// Provides unified interface for multi-agent swarm initialization and management

export { QuicCoordinator, type SwarmAgent, type SwarmMessage, type SwarmState, type SwarmStats, type QuicCoordinatorConfig } from './quic-coordinator.js';
export { TransportRouter, type TransportProtocol, type TransportConfig, type TransportStats, type RouteResult } from './transport-router.js';

import { QuicClient, QuicConnectionPool } from '../transport/quic.js';
import { QuicCoordinator, SwarmAgent } from './quic-coordinator.js';
import { TransportRouter, TransportConfig } from './transport-router.js';
import { logger } from '../utils/logger.js';

export interface SwarmInitOptions {
  swarmId: string;
  topology: 'mesh' | 'hierarchical' | 'ring' | 'star';
  transport?: 'quic' | 'http2' | 'auto';
  maxAgents?: number;
  quicPort?: number;
  quicHost?: string;
  enableFallback?: boolean;
}

export interface SwarmInstance {
  swarmId: string;
  topology: 'mesh' | 'hierarchical' | 'ring' | 'star';
  transport: 'quic' | 'http2';
  coordinator?: QuicCoordinator;
  router: TransportRouter;
  registerAgent: (agent: SwarmAgent) => Promise<void>;
  unregisterAgent: (agentId: string) => Promise<void>;
  getStats: () => any;
  shutdown: () => Promise<void>;
}

/**
 * Initialize a multi-agent swarm with QUIC transport
 *
 * @example
 * ```typescript
 * const swarm = await initSwarm({
 *   swarmId: 'my-swarm',
 *   topology: 'mesh',
 *   transport: 'quic',
 *   quicPort: 4433
 * });
 *
 * await swarm.registerAgent({
 *   id: 'agent-1',
 *   role: 'worker',
 *   host: 'localhost',
 *   port: 4434,
 *   capabilities: ['compute', 'analyze']
 * });
 * ```
 */
export async function initSwarm(options: SwarmInitOptions): Promise<SwarmInstance> {
  const {
    swarmId,
    topology,
    transport = 'auto',
    maxAgents = 10,
    quicPort = 4433,
    quicHost = 'localhost',
    enableFallback = true
  } = options;

  logger.info('Initializing swarm', {
    swarmId,
    topology,
    transport,
    maxAgents,
    quicPort
  });

  // Create transport router configuration
  const transportConfig: TransportConfig = {
    protocol: transport,
    enableFallback,
    quicConfig: {
      host: quicHost,
      port: quicPort,
      maxConnections: maxAgents * 2 // Allow some overhead
    },
    http2Config: {
      host: quicHost,
      port: quicPort + 1000, // HTTP/2 fallback port
      maxConnections: maxAgents * 2,
      secure: true
    }
  };

  // Initialize transport router
  const router = new TransportRouter(transportConfig);
  await router.initialize();

  const actualProtocol = router.getCurrentProtocol();
  logger.info('Transport initialized', {
    requestedProtocol: transport,
    actualProtocol,
    quicAvailable: router.isQuicAvailable()
  });

  // Initialize swarm coordinator if using QUIC
  let coordinator: QuicCoordinator | undefined;
  if (actualProtocol === 'quic') {
    coordinator = await router.initializeSwarm(swarmId, topology, maxAgents);
  }

  // Create swarm instance
  const swarm: SwarmInstance = {
    swarmId,
    topology,
    transport: actualProtocol,
    coordinator,
    router,

    async registerAgent(agent: SwarmAgent): Promise<void> {
      if (coordinator) {
        await coordinator.registerAgent(agent);
      } else {
        logger.warn('QUIC coordinator not available, agent registration skipped', {
          agentId: agent.id
        });
      }
    },

    async unregisterAgent(agentId: string): Promise<void> {
      if (coordinator) {
        await coordinator.unregisterAgent(agentId);
      } else {
        logger.warn('QUIC coordinator not available, agent unregistration skipped', {
          agentId
        });
      }
    },

    async getStats() {
      return {
        swarmId,
        topology,
        transport: actualProtocol,
        coordinatorStats: coordinator ? (await coordinator.getState()).stats : undefined,
        transportStats: router.getStats(),
        quicAvailable: router.isQuicAvailable()
      };
    },

    async shutdown(): Promise<void> {
      logger.info('Shutting down swarm', { swarmId });
      await router.shutdown();
      logger.info('Swarm shutdown complete', { swarmId });
    }
  };

  logger.info('Swarm initialized successfully', {
    swarmId,
    topology,
    transport: actualProtocol,
    quicEnabled: coordinator !== undefined
  });

  return swarm;
}

/**
 * Check if QUIC transport is available
 */
export async function checkQuicAvailability(): Promise<boolean> {
  try {
    const client = new QuicClient();
    await client.initialize();
    await client.shutdown();
    return true;
  } catch (error) {
    logger.debug('QUIC not available', { error });
    return false;
  }
}
