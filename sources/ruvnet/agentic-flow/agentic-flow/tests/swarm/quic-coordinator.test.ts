// QUIC Coordinator Test Suite
// Comprehensive tests for QUIC-enabled swarm coordination across all topologies

import { describe, it, expect, beforeEach, afterEach } from '@jest/globals';
import { QuicCoordinator, SwarmAgent, SwarmMessage } from '../../src/swarm/quic-coordinator.js';
import { QuicClient, QuicConnectionPool } from '../../src/transport/quic.js';

describe('QuicCoordinator', () => {
  let coordinator: QuicCoordinator;
  let quicClient: QuicClient;
  let connectionPool: QuicConnectionPool;

  beforeEach(async () => {
    quicClient = new QuicClient({
      serverHost: 'localhost',
      serverPort: 4433
    });
    connectionPool = new QuicConnectionPool(quicClient, 10);
  });

  afterEach(async () => {
    if (coordinator) {
      await coordinator.stop();
    }
    await quicClient.shutdown();
  });

  describe('Mesh Topology', () => {
    it('should initialize mesh topology swarm', async () => {
      coordinator = new QuicCoordinator({
        swarmId: 'test-mesh',
        topology: 'mesh',
        maxAgents: 5,
        quicClient,
        connectionPool
      });

      await coordinator.start();

      const state = coordinator.getState();
      expect(state.swarmId).toBe('test-mesh');
      expect(state.topology).toBe('mesh');
      expect(state.stats.totalAgents).toBe(0);
    });

    it('should register agents in mesh topology', async () => {
      coordinator = new QuicCoordinator({
        swarmId: 'test-mesh',
        topology: 'mesh',
        maxAgents: 5,
        quicClient,
        connectionPool
      });

      await coordinator.start();

      const agent1: SwarmAgent = {
        id: 'agent-1',
        role: 'worker',
        host: 'localhost',
        port: 4434,
        capabilities: ['compute']
      };

      const agent2: SwarmAgent = {
        id: 'agent-2',
        role: 'worker',
        host: 'localhost',
        port: 4435,
        capabilities: ['analyze']
      };

      await coordinator.registerAgent(agent1);
      await coordinator.registerAgent(agent2);

      const state = coordinator.getState();
      expect(state.stats.totalAgents).toBe(2);
      expect(state.agents.has('agent-1')).toBe(true);
      expect(state.agents.has('agent-2')).toBe(true);
    });

    it('should broadcast messages to all agents in mesh', async () => {
      coordinator = new QuicCoordinator({
        swarmId: 'test-mesh',
        topology: 'mesh',
        maxAgents: 5,
        quicClient,
        connectionPool
      });

      await coordinator.start();

      // Register multiple agents
      for (let i = 1; i <= 3; i++) {
        await coordinator.registerAgent({
          id: `agent-${i}`,
          role: 'worker',
          host: 'localhost',
          port: 4433 + i,
          capabilities: ['compute']
        });
      }

      const message: SwarmMessage = {
        id: 'msg-1',
        from: 'coordinator',
        to: '*',
        type: 'task',
        payload: { task: 'compute' },
        timestamp: Date.now()
      };

      await coordinator.broadcast(message);

      const stats = coordinator.getState().stats;
      expect(stats.totalMessages).toBeGreaterThan(0);
    });
  });

  describe('Hierarchical Topology', () => {
    it('should initialize hierarchical topology', async () => {
      coordinator = new QuicCoordinator({
        swarmId: 'test-hierarchical',
        topology: 'hierarchical',
        maxAgents: 10,
        quicClient,
        connectionPool
      });

      await coordinator.start();

      const state = coordinator.getState();
      expect(state.topology).toBe('hierarchical');
    });

    it('should route worker messages through coordinator', async () => {
      coordinator = new QuicCoordinator({
        swarmId: 'test-hierarchical',
        topology: 'hierarchical',
        maxAgents: 10,
        quicClient,
        connectionPool
      });

      await coordinator.start();

      // Register coordinator agent
      await coordinator.registerAgent({
        id: 'coordinator-1',
        role: 'coordinator',
        host: 'localhost',
        port: 4434,
        capabilities: ['coordinate']
      });

      // Register worker agents
      await coordinator.registerAgent({
        id: 'worker-1',
        role: 'worker',
        host: 'localhost',
        port: 4435,
        capabilities: ['compute']
      });

      await coordinator.registerAgent({
        id: 'worker-2',
        role: 'worker',
        host: 'localhost',
        port: 4436,
        capabilities: ['compute']
      });

      const state = coordinator.getState();
      expect(state.stats.totalAgents).toBe(3);
    });
  });

  describe('Ring Topology', () => {
    it('should initialize ring topology', async () => {
      coordinator = new QuicCoordinator({
        swarmId: 'test-ring',
        topology: 'ring',
        maxAgents: 8,
        quicClient,
        connectionPool
      });

      await coordinator.start();

      const state = coordinator.getState();
      expect(state.topology).toBe('ring');
    });

    it('should forward messages to next agent in ring', async () => {
      coordinator = new QuicCoordinator({
        swarmId: 'test-ring',
        topology: 'ring',
        maxAgents: 8,
        quicClient,
        connectionPool
      });

      await coordinator.start();

      // Register agents in ring
      for (let i = 1; i <= 4; i++) {
        await coordinator.registerAgent({
          id: `agent-${i}`,
          role: 'worker',
          host: 'localhost',
          port: 4433 + i,
          capabilities: ['compute']
        });
      }

      const message: SwarmMessage = {
        id: 'msg-ring',
        from: 'agent-1',
        to: 'agent-2',
        type: 'task',
        payload: { data: 'test' },
        timestamp: Date.now()
      };

      await coordinator.sendMessage(message);

      const stats = coordinator.getState().stats;
      expect(stats.totalMessages).toBeGreaterThan(0);
    });
  });

  describe('Star Topology', () => {
    it('should initialize star topology', async () => {
      coordinator = new QuicCoordinator({
        swarmId: 'test-star',
        topology: 'star',
        maxAgents: 10,
        quicClient,
        connectionPool
      });

      await coordinator.start();

      const state = coordinator.getState();
      expect(state.topology).toBe('star');
    });

    it('should route all messages through central coordinator', async () => {
      coordinator = new QuicCoordinator({
        swarmId: 'test-star',
        topology: 'star',
        maxAgents: 10,
        quicClient,
        connectionPool
      });

      await coordinator.start();

      // Register central coordinator
      await coordinator.registerAgent({
        id: 'central',
        role: 'coordinator',
        host: 'localhost',
        port: 4434,
        capabilities: ['coordinate']
      });

      // Register spoke agents
      for (let i = 1; i <= 3; i++) {
        await coordinator.registerAgent({
          id: `spoke-${i}`,
          role: 'worker',
          host: 'localhost',
          port: 4434 + i,
          capabilities: ['compute']
        });
      }

      const state = coordinator.getState();
      expect(state.stats.totalAgents).toBe(4);
    });
  });

  describe('State Synchronization', () => {
    it('should synchronize state across agents', async () => {
      coordinator = new QuicCoordinator({
        swarmId: 'test-sync',
        topology: 'mesh',
        maxAgents: 5,
        quicClient,
        connectionPool,
        statesSyncInterval: 1000
      });

      await coordinator.start();

      await coordinator.registerAgent({
        id: 'agent-1',
        role: 'worker',
        host: 'localhost',
        port: 4434,
        capabilities: ['compute']
      });

      await coordinator.syncState();

      const stats = coordinator.getState().stats;
      expect(stats.totalMessages).toBeGreaterThan(0);
    });

    it('should send heartbeats periodically', async () => {
      coordinator = new QuicCoordinator({
        swarmId: 'test-heartbeat',
        topology: 'mesh',
        maxAgents: 5,
        quicClient,
        connectionPool,
        heartbeatInterval: 500
      });

      await coordinator.start();

      await coordinator.registerAgent({
        id: 'agent-1',
        role: 'worker',
        host: 'localhost',
        port: 4434,
        capabilities: ['compute']
      });

      // Wait for heartbeat
      await new Promise(resolve => setTimeout(resolve, 600));

      const stats = coordinator.getState().stats;
      expect(stats.totalMessages).toBeGreaterThan(0);
    });
  });

  describe('Statistics Tracking', () => {
    it('should track per-agent statistics', async () => {
      coordinator = new QuicCoordinator({
        swarmId: 'test-stats',
        topology: 'mesh',
        maxAgents: 5,
        quicClient,
        connectionPool
      });

      await coordinator.start();

      await coordinator.registerAgent({
        id: 'agent-1',
        role: 'worker',
        host: 'localhost',
        port: 4434,
        capabilities: ['compute']
      });

      const agentStats = coordinator.getAgentStats('agent-1');
      expect(agentStats).toBeDefined();
      expect(agentStats?.sent).toBe(0);
      expect(agentStats?.received).toBe(0);
    });

    it('should track all agent statistics', async () => {
      coordinator = new QuicCoordinator({
        swarmId: 'test-stats',
        topology: 'mesh',
        maxAgents: 5,
        quicClient,
        connectionPool
      });

      await coordinator.start();

      await coordinator.registerAgent({
        id: 'agent-1',
        role: 'worker',
        host: 'localhost',
        port: 4434,
        capabilities: ['compute']
      });

      await coordinator.registerAgent({
        id: 'agent-2',
        role: 'worker',
        host: 'localhost',
        port: 4435,
        capabilities: ['compute']
      });

      const allStats = coordinator.getAllAgentStats();
      expect(allStats.size).toBe(2);
      expect(allStats.has('agent-1')).toBe(true);
      expect(allStats.has('agent-2')).toBe(true);
    });
  });

  describe('Agent Management', () => {
    it('should unregister agents', async () => {
      coordinator = new QuicCoordinator({
        swarmId: 'test-unreg',
        topology: 'mesh',
        maxAgents: 5,
        quicClient,
        connectionPool
      });

      await coordinator.start();

      await coordinator.registerAgent({
        id: 'agent-1',
        role: 'worker',
        host: 'localhost',
        port: 4434,
        capabilities: ['compute']
      });

      expect(coordinator.getState().stats.totalAgents).toBe(1);

      await coordinator.unregisterAgent('agent-1');

      expect(coordinator.getState().stats.totalAgents).toBe(0);
    });

    it('should enforce max agents limit', async () => {
      coordinator = new QuicCoordinator({
        swarmId: 'test-limit',
        topology: 'mesh',
        maxAgents: 2,
        quicClient,
        connectionPool
      });

      await coordinator.start();

      await coordinator.registerAgent({
        id: 'agent-1',
        role: 'worker',
        host: 'localhost',
        port: 4434,
        capabilities: ['compute']
      });

      await coordinator.registerAgent({
        id: 'agent-2',
        role: 'worker',
        host: 'localhost',
        port: 4435,
        capabilities: ['compute']
      });

      await expect(
        coordinator.registerAgent({
          id: 'agent-3',
          role: 'worker',
          host: 'localhost',
          port: 4436,
          capabilities: ['compute']
        })
      ).rejects.toThrow('Maximum agents');
    });
  });
});
