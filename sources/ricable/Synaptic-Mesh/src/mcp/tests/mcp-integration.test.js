/**
 * Comprehensive MCP Integration Test Suite
 * Tests all MCP functionality including tools, transport, auth, and WASM bridge
 */

import { describe, it, beforeEach, afterEach } from 'node:test';
import assert from 'node:assert';
import SynapticMeshMCP from '../index.js';
import { NeuralMeshTools } from '../neural-mesh/neural-mesh-tools.js';
import { TransportManager } from '../transport/transport-manager.js';
import { AuthManager } from '../auth/auth-manager.js';
import { EventStreamer } from '../events/event-streamer.js';
import { WasmBridge } from '../wasm-bridge/wasm-bridge.js';

describe('MCP Integration Tests', () => {
  let mcpServer;
  let config;

  beforeEach(async () => {
    config = {
      transport: 'stdio',
      enableAuth: false,
      enableEvents: true,
      wasmEnabled: false, // Disable for testing
      logLevel: 'error'
    };
    
    mcpServer = new SynapticMeshMCP(config);
  });

  afterEach(async () => {
    if (mcpServer) {
      await mcpServer.stop();
    }
  });

  describe('Server Initialization', () => {
    it('should initialize MCP server successfully', async () => {
      await mcpServer.initialize();
      const status = mcpServer.getStatus();
      
      assert.strictEqual(status.initialized, true);
      assert.strictEqual(status.running, false);
      assert.strictEqual(typeof status.toolsCount, 'number');
      assert(status.toolsCount > 0, 'Should have tools available');
    });

    it('should handle initialization errors gracefully', async () => {
      const badConfig = { ...config, transport: 'invalid' };
      const badServer = new SynapticMeshMCP(badConfig);
      
      await assert.rejects(
        async () => await badServer.initialize(),
        /Unsupported transport/
      );
    });
  });

  describe('Neural Mesh Tools', () => {
    let tools;

    beforeEach(async () => {
      await mcpServer.initialize();
      tools = mcpServer.tools;
    });

    it('should list all available tools', async () => {
      const toolList = await tools.listTools();
      
      assert(Array.isArray(toolList), 'Should return an array');
      assert(toolList.length > 20, 'Should have 20+ tools');
      
      // Check for required tools
      const toolNames = toolList.map(t => t.name);
      const requiredTools = [
        'neural_mesh_init',
        'neural_agent_spawn',
        'neural_consensus',
        'mesh_memory_store',
        'mesh_memory_retrieve',
        'neural_train',
        'mesh_performance'
      ];
      
      requiredTools.forEach(toolName => {
        assert(
          toolNames.includes(toolName),
          `Should include tool: ${toolName}`
        );
      });
    });

    it('should execute neural_mesh_init successfully', async () => {
      const result = await tools.executeTool('neural_mesh_init', {
        topology: 'mesh',
        maxAgents: 10,
        strategy: 'parallel',
        enableConsensus: true,
        cryptoLevel: 'quantum'
      });
      
      assert.strictEqual(result.success, true);
      assert(typeof result.meshId === 'string');
      assert(result.meshId.length > 0);
      assert.strictEqual(result.config.topology, 'mesh');
      assert.strictEqual(result.config.maxAgents, 10);
    });

    it('should spawn neural agents in mesh', async () => {
      // First create a mesh
      const meshResult = await tools.executeTool('neural_mesh_init', {
        topology: 'hierarchical',
        maxAgents: 5,
        strategy: 'balanced'
      });
      
      const meshId = meshResult.meshId;
      
      // Spawn different types of agents
      const agentTypes = ['coordinator', 'researcher', 'coder', 'analyst'];
      const agents = [];
      
      for (const type of agentTypes) {
        const agentResult = await tools.executeTool('neural_agent_spawn', {
          meshId,
          type,
          name: `test-${type}`,
          capabilities: [`${type}-capability`],
          neuralModel: 'test-model'
        });
        
        assert.strictEqual(agentResult.success, true);
        assert(typeof agentResult.agentId === 'string');
        assert.strictEqual(agentResult.agent.type, type);
        agents.push(agentResult.agentId);
      }
      
      assert.strictEqual(agents.length, 4);
    });

    it('should handle memory operations', async () => {
      const testKey = 'test-memory-key';
      const testValue = { data: 'test-data', number: 42 };
      const namespace = 'test-namespace';
      
      // Store data
      const storeResult = await tools.executeTool('mesh_memory_store', {
        key: testKey,
        value: testValue,
        namespace,
        ttl: 3600 // 1 hour
      });
      
      assert.strictEqual(storeResult.success, true);
      assert.strictEqual(storeResult.key, testKey);
      assert.strictEqual(storeResult.namespace, namespace);
      
      // Retrieve data
      const retrieveResult = await tools.executeTool('mesh_memory_retrieve', {
        key: testKey,
        namespace
      });
      
      assert.strictEqual(retrieveResult.success, true);
      assert.deepStrictEqual(retrieveResult.value, testValue);
    });

    it('should perform consensus operations', async () => {
      // Create mesh and agents first
      const meshResult = await tools.executeTool('neural_mesh_init', {
        topology: 'mesh',
        maxAgents: 5,
        strategy: 'parallel'
      });
      const meshId = meshResult.meshId;
      
      const agents = [];
      for (let i = 0; i < 3; i++) {
        const agentResult = await tools.executeTool('neural_agent_spawn', {
          meshId,
          type: 'coordinator',
          name: `consensus-agent-${i}`
        });
        agents.push(agentResult.agentId);
      }
      
      // Perform consensus
      const consensusResult = await tools.executeTool('neural_consensus', {
        meshId,
        proposal: { action: 'test-proposal', value: 123 },
        agents,
        consensusType: 'majority'
      });
      
      assert.strictEqual(consensusResult.success, true);
      assert(typeof consensusResult.consensusId === 'string');
      assert(['accepted', 'rejected'].includes(consensusResult.result));
      assert(typeof consensusResult.votes === 'object');
    });

    it('should get performance metrics', async () => {
      const metricsResult = await tools.executeTool('mesh_performance', {
        timeframe: '1h',
        metrics: ['cpu', 'memory']
      });
      
      assert(typeof metricsResult === 'object');
      assert(typeof metricsResult.timeframe === 'string');
      assert(typeof metricsResult.summary === 'object');
    });

    it('should handle tool validation errors', async () => {
      await assert.rejects(
        async () => await tools.executeTool('neural_mesh_init', {
          topology: 'invalid-topology', // Invalid value
          maxAgents: 10,
          strategy: 'parallel'
        }),
        /Invalid input/
      );
    });

    it('should handle unknown tool errors', async () => {
      await assert.rejects(
        async () => await tools.executeTool('unknown_tool', {}),
        /Unknown tool/
      );
    });
  });

  describe('Transport Manager', () => {
    let transport;

    beforeEach(() => {
      transport = new TransportManager({ transport: 'stdio' });
    });

    it('should initialize stdio transport', async () => {
      await transport.initialize();
      assert.strictEqual(transport.config.transport, 'stdio');
    });

    it('should track connection statistics', () => {
      const stats = transport.getStats();
      
      assert(typeof stats === 'object');
      assert(typeof stats.totalConnections === 'number');
      assert(typeof stats.activeConnections === 'number');
      assert(typeof stats.messagesProcessed === 'number');
      assert(typeof stats.errors === 'number');
    });

    it('should handle unsupported transport types', async () => {
      const badTransport = new TransportManager({ transport: 'unsupported' });
      
      await assert.rejects(
        async () => await badTransport.initialize(),
        /Unsupported transport/
      );
    });
  });

  describe('Authentication Manager', () => {
    let auth;

    beforeEach(() => {
      auth = new AuthManager({
        enableAuth: true,
        apiKeys: [
          {
            key: 'test-key-123',
            name: 'test-key',
            permissions: ['neural_*']
          }
        ]
      });
    });

    it('should validate API keys', async () => {
      const request = {
        headers: { authorization: 'Bearer test-key-123' },
        params: { name: 'neural_mesh_init' }
      };
      
      const result = await auth.authorize(request);
      
      assert.strictEqual(result.authorized, true);
      assert.strictEqual(result.user, 'test-key');
    });

    it('should reject invalid API keys', async () => {
      const request = {
        headers: { authorization: 'Bearer invalid-key' },
        params: { name: 'neural_mesh_init' }
      };
      
      await assert.rejects(
        async () => await auth.authorize(request),
        /Invalid API key/
      );
    });

    it('should enforce permissions', async () => {
      const request = {
        headers: { authorization: 'Bearer test-key-123' },
        params: { name: 'unauthorized_tool' }
      };
      
      await assert.rejects(
        async () => await auth.authorize(request),
        /Insufficient permissions/
      );
    });

    it('should handle rate limiting', async () => {
      const auth = new AuthManager({
        enableAuth: true,
        apiKeys: [
          {
            key: 'rate-test-key',
            name: 'rate-test',
            permissions: ['*'],
            rateLimits: { requests: 2, window: 1000, burst: 1 }
          }
        ]
      });

      const request = {
        headers: { authorization: 'Bearer rate-test-key' },
        params: { name: 'test_tool' }
      };

      // First request should pass
      await auth.authorize(request);
      
      // Second request should pass
      await auth.authorize(request);
      
      // Third request should fail due to rate limit
      await assert.rejects(
        async () => await auth.authorize(request),
        /Rate limit exceeded/
      );
    });
  });

  describe('Event Streamer', () => {
    let events;

    beforeEach(() => {
      events = new EventStreamer({
        bufferSize: 100,
        retentionTime: 60000
      });
    });

    it('should create and manage event streams', () => {
      const streamId = events.createStream('neural-mesh', {
        'data.meshId': 'test-mesh'
      });
      
      assert(typeof streamId === 'string');
      assert(streamId.length > 0);
      
      const streamInfo = events.getStreamInfo(streamId);
      assert.strictEqual(streamInfo.type, 'neural-mesh');
      assert.deepStrictEqual(streamInfo.filter, { 'data.meshId': 'test-mesh' });
    });

    it('should handle event subscriptions', () => {
      const streamId = events.createStream('agent', {});
      const receivedEvents = [];
      
      const subscriberId = events.subscribe(streamId, (event) => {
        receivedEvents.push(event);
      });
      
      assert(typeof subscriberId === 'string');
      
      // Emit test event
      events.emitAgentEvent('spawned', 'agent-1', 'mesh-1', {
        type: 'coordinator'
      });
      
      // Check that event was received
      assert.strictEqual(receivedEvents.length, 1);
      assert.strictEqual(receivedEvents[0].type, 'agent');
      assert.strictEqual(receivedEvents[0].data.eventType, 'spawned');
    });

    it('should apply event filters correctly', () => {
      const meshStreamId = events.createStream('neural-mesh', {
        'data.meshId': 'specific-mesh'
      });
      
      const allEventsStreamId = events.createStream('neural-mesh', {});
      
      const meshEvents = [];
      const allEvents = [];
      
      events.subscribe(meshStreamId, (event) => meshEvents.push(event));
      events.subscribe(allEventsStreamId, (event) => allEvents.push(event));
      
      // Emit events for different meshes
      events.emitNeuralMeshEvent('initialized', { meshId: 'specific-mesh' });
      events.emitNeuralMeshEvent('initialized', { meshId: 'other-mesh' });
      
      // Filtered stream should only get events for specific mesh
      assert.strictEqual(meshEvents.length, 1);
      assert.strictEqual(meshEvents[0].data.meshId, 'specific-mesh');
      
      // Unfiltered stream should get all events
      assert.strictEqual(allEvents.length, 2);
    });

    it('should track streaming metrics', () => {
      const metrics = events.getStreamingMetrics();
      
      assert(typeof metrics === 'object');
      assert(typeof metrics.totalEvents === 'number');
      assert(typeof metrics.totalStreams === 'number');
      assert(typeof metrics.totalSubscribers === 'number');
      assert(typeof metrics.eventsPerSecond === 'number');
    });
  });

  describe('WASM Bridge', () => {
    let wasmBridge;

    beforeEach(() => {
      wasmBridge = new WasmBridge({
        wasmEnabled: false // Skip actual WASM loading in tests
      });
    });

    it('should detect WASM capabilities', async () => {
      // Mock capability detection
      wasmBridge.supportsSIMD = true;
      wasmBridge.supportsThreads = true;
      
      const metrics = wasmBridge.getPerformanceMetrics();
      
      assert.strictEqual(metrics.capabilities.simd, true);
      assert.strictEqual(metrics.capabilities.threads, true);
    });

    it('should track module loading', () => {
      const modules = wasmBridge.getLoadedModules();
      assert(Array.isArray(modules));
    });

    it('should handle WASM function calls gracefully', async () => {
      // Mock a module for testing
      wasmBridge.loadedModules.set('test-module', {
        exports: {
          testFunction: () => 'test-result'
        },
        path: '/test/path',
        loadedAt: Date.now(),
        callCount: 0
      });
      
      const result = await wasmBridge.callWasmFunction('test-module', 'testFunction', {});
      assert.strictEqual(result, 'test-result');
    });
  });

  describe('End-to-End Integration', () => {
    it('should handle complete workflow', async () => {
      await mcpServer.initialize();
      
      // Create mesh
      const meshResult = await mcpServer.tools.executeTool('neural_mesh_init', {
        topology: 'mesh',
        maxAgents: 3,
        strategy: 'parallel'
      });
      
      const meshId = meshResult.meshId;
      
      // Spawn agents
      const agentResults = [];
      for (let i = 0; i < 3; i++) {
        const agentResult = await mcpServer.tools.executeTool('neural_agent_spawn', {
          meshId,
          type: 'coordinator',
          name: `workflow-agent-${i}`
        });
        agentResults.push(agentResult);
      }
      
      // Store some data
      await mcpServer.tools.executeTool('mesh_memory_store', {
        key: 'workflow-data',
        value: { step: 'completed', agents: agentResults.length },
        namespace: 'workflow'
      });
      
      // Retrieve data
      const dataResult = await mcpServer.tools.executeTool('mesh_memory_retrieve', {
        key: 'workflow-data',
        namespace: 'workflow'
      });
      
      // Verify complete workflow
      assert.strictEqual(meshResult.success, true);
      assert.strictEqual(agentResults.length, 3);
      assert(agentResults.every(r => r.success));
      assert.strictEqual(dataResult.success, true);
      assert.strictEqual(dataResult.value.step, 'completed');
      assert.strictEqual(dataResult.value.agents, 3);
    });

    it('should handle concurrent tool executions', async () => {
      await mcpServer.initialize();
      
      const promises = [];
      
      // Execute multiple mesh initializations concurrently
      for (let i = 0; i < 5; i++) {
        promises.push(
          mcpServer.tools.executeTool('neural_mesh_init', {
            topology: 'hierarchical',
            maxAgents: 5,
            strategy: 'balanced'
          })
        );
      }
      
      const results = await Promise.all(promises);
      
      // All should succeed
      assert.strictEqual(results.length, 5);
      assert(results.every(r => r.success));
      
      // All should have unique mesh IDs
      const meshIds = results.map(r => r.meshId);
      const uniqueIds = new Set(meshIds);
      assert.strictEqual(uniqueIds.size, 5);
    });
  });
});

// Run tests if this file is executed directly
if (import.meta.url === `file://${process.argv[1]}`) {
  console.log('Running MCP Integration Tests...');
}