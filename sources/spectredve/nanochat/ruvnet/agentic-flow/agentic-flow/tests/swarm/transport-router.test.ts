// Transport Router Test Suite
// Tests for protocol selection, fallback, and routing

import { describe, it, expect, beforeEach, afterEach } from '@jest/globals';
import { TransportRouter, TransportConfig } from '../../src/swarm/transport-router.js';
import { SwarmAgent, SwarmMessage } from '../../src/swarm/quic-coordinator.js';

describe('TransportRouter', () => {
  let router: TransportRouter;

  afterEach(async () => {
    if (router) {
      await router.shutdown();
    }
  });

  describe('Protocol Selection', () => {
    it('should initialize with QUIC protocol', async () => {
      const config: TransportConfig = {
        protocol: 'quic',
        enableFallback: true,
        quicConfig: {
          host: 'localhost',
          port: 4433,
          maxConnections: 10
        }
      };

      router = new TransportRouter(config);
      await router.initialize();

      expect(router.isQuicAvailable()).toBe(true);
      expect(router.getCurrentProtocol()).toBe('quic');
    });

    it('should initialize with HTTP/2 protocol', async () => {
      const config: TransportConfig = {
        protocol: 'http2',
        enableFallback: false,
        http2Config: {
          host: 'localhost',
          port: 8443,
          maxConnections: 10,
          secure: true
        }
      };

      router = new TransportRouter(config);
      await router.initialize();

      expect(router.getCurrentProtocol()).toBe('http2');
    });

    it('should auto-select protocol based on availability', async () => {
      const config: TransportConfig = {
        protocol: 'auto',
        enableFallback: true,
        quicConfig: {
          host: 'localhost',
          port: 4433,
          maxConnections: 10
        },
        http2Config: {
          host: 'localhost',
          port: 8443,
          maxConnections: 10,
          secure: true
        }
      };

      router = new TransportRouter(config);
      await router.initialize();

      const protocol = router.getCurrentProtocol();
      expect(['quic', 'http2']).toContain(protocol);
    });
  });

  describe('Transparent Fallback', () => {
    it('should fallback to HTTP/2 when QUIC fails', async () => {
      const config: TransportConfig = {
        protocol: 'quic',
        enableFallback: true,
        quicConfig: {
          host: 'invalid-host',
          port: 9999,
          maxConnections: 10
        },
        http2Config: {
          host: 'localhost',
          port: 8443,
          maxConnections: 10,
          secure: true
        }
      };

      router = new TransportRouter(config);
      await router.initialize();

      // Should fallback to HTTP/2
      expect(router.getCurrentProtocol()).toBe('http2');
    });

    it('should throw error when fallback disabled and QUIC fails', async () => {
      const config: TransportConfig = {
        protocol: 'quic',
        enableFallback: false,
        quicConfig: {
          host: 'invalid-host',
          port: 9999,
          maxConnections: 10
        }
      };

      router = new TransportRouter(config);

      await expect(router.initialize()).rejects.toThrow();
    });
  });

  describe('Message Routing', () => {
    it('should route message via QUIC', async () => {
      const config: TransportConfig = {
        protocol: 'quic',
        enableFallback: true,
        quicConfig: {
          host: 'localhost',
          port: 4433,
          maxConnections: 10
        }
      };

      router = new TransportRouter(config);
      await router.initialize();

      const message: SwarmMessage = {
        id: 'msg-1',
        from: 'agent-1',
        to: 'agent-2',
        type: 'task',
        payload: { data: 'test' },
        timestamp: Date.now()
      };

      const target: SwarmAgent = {
        id: 'agent-2',
        role: 'worker',
        host: 'localhost',
        port: 4434,
        capabilities: ['compute']
      };

      const result = await router.route(message, target);
      expect(result.success).toBe(true);
      expect(result.protocol).toBe('quic');
      expect(result.latency).toBeGreaterThanOrEqual(0);
    });

    it('should route message via HTTP/2', async () => {
      const config: TransportConfig = {
        protocol: 'http2',
        enableFallback: false,
        http2Config: {
          host: 'localhost',
          port: 8443,
          maxConnections: 10,
          secure: true
        }
      };

      router = new TransportRouter(config);
      await router.initialize();

      const message: SwarmMessage = {
        id: 'msg-1',
        from: 'agent-1',
        to: 'agent-2',
        type: 'task',
        payload: { data: 'test' },
        timestamp: Date.now()
      };

      const target: SwarmAgent = {
        id: 'agent-2',
        role: 'worker',
        host: 'localhost',
        port: 8444,
        capabilities: ['compute']
      };

      const result = await router.route(message, target);
      expect(result.success).toBe(true);
      expect(result.protocol).toBe('http2');
    });
  });

  describe('Statistics', () => {
    it('should track QUIC statistics', async () => {
      const config: TransportConfig = {
        protocol: 'quic',
        enableFallback: true,
        quicConfig: {
          host: 'localhost',
          port: 4433,
          maxConnections: 10
        }
      };

      router = new TransportRouter(config);
      await router.initialize();

      const stats = router.getStats('quic');
      expect(stats).toBeDefined();
      expect(stats.protocol).toBe('quic');
      expect(stats.messagesSent).toBe(0);
      expect(stats.messagesReceived).toBe(0);
    });

    it('should track HTTP/2 statistics', async () => {
      const config: TransportConfig = {
        protocol: 'http2',
        enableFallback: false,
        http2Config: {
          host: 'localhost',
          port: 8443,
          maxConnections: 10,
          secure: true
        }
      };

      router = new TransportRouter(config);
      await router.initialize();

      const stats = router.getStats('http2');
      expect(stats).toBeDefined();
      expect(stats.protocol).toBe('http2');
    });

    it('should track all protocol statistics', async () => {
      const config: TransportConfig = {
        protocol: 'auto',
        enableFallback: true,
        quicConfig: {
          host: 'localhost',
          port: 4433,
          maxConnections: 10
        },
        http2Config: {
          host: 'localhost',
          port: 8443,
          maxConnections: 10,
          secure: true
        }
      };

      router = new TransportRouter(config);
      await router.initialize();

      const allStats = router.getStats();
      expect(allStats instanceof Map).toBe(true);
      expect(allStats.size).toBe(2);
    });
  });

  describe('Swarm Integration', () => {
    it('should initialize swarm with QUIC coordinator', async () => {
      const config: TransportConfig = {
        protocol: 'quic',
        enableFallback: true,
        quicConfig: {
          host: 'localhost',
          port: 4433,
          maxConnections: 10
        }
      };

      router = new TransportRouter(config);
      await router.initialize();

      const coordinator = await router.initializeSwarm('test-swarm', 'mesh', 5);

      expect(coordinator).toBeDefined();
      expect(coordinator.getState().swarmId).toBe('test-swarm');
      expect(coordinator.getState().topology).toBe('mesh');
    });

    it('should get coordinator after initialization', async () => {
      const config: TransportConfig = {
        protocol: 'quic',
        enableFallback: true,
        quicConfig: {
          host: 'localhost',
          port: 4433,
          maxConnections: 10
        }
      };

      router = new TransportRouter(config);
      await router.initialize();

      await router.initializeSwarm('test-swarm', 'mesh', 5);

      const coordinator = router.getCoordinator();
      expect(coordinator).toBeDefined();
      expect(coordinator?.getState().swarmId).toBe('test-swarm');
    });
  });
});
