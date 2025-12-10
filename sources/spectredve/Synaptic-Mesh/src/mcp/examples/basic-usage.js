#!/usr/bin/env node

/**
 * Basic Usage Example for Synaptic Neural Mesh MCP
 * Demonstrates core functionality and common usage patterns
 */

import SynapticMeshMCP from '../index.js';
import chalk from 'chalk';

class BasicUsageExample {
  constructor() {
    this.mcpServer = null;
  }

  async run() {
    console.log(chalk.blue('🧠 Synaptic Neural Mesh MCP - Basic Usage Example'));
    console.log();

    try {
      // Step 1: Initialize MCP server
      await this.initializeServer();

      // Step 2: Create a neural mesh
      const meshId = await this.createNeuralMesh();

      // Step 3: Spawn neural agents
      const agents = await this.spawnAgents(meshId);

      // Step 4: Demonstrate memory operations
      await this.demonstrateMemory();

      // Step 5: Perform consensus
      await this.performConsensus(meshId, agents);

      // Step 6: Monitor performance
      await this.monitorPerformance();

      // Step 7: Show event streaming
      await this.demonstrateEventStreaming(meshId);

      console.log();
      console.log(chalk.green('✅ Basic usage example completed successfully!'));

    } catch (error) {
      console.error(chalk.red('❌ Example failed:'), error.message);
      process.exit(1);
    } finally {
      if (this.mcpServer) {
        await this.mcpServer.stop();
      }
    }
  }

  async initializeServer() {
    console.log(chalk.cyan('📡 Initializing MCP Server...'));
    
    this.mcpServer = new SynapticMeshMCP({
      transport: 'stdio',
      enableAuth: false,
      enableEvents: true,
      wasmEnabled: false, // Disable for example
      logLevel: 'warn'
    });

    await this.mcpServer.initialize();
    
    const status = this.mcpServer.getStatus();
    console.log(chalk.green(`✅ Server initialized with ${status.toolsCount} tools available`));
  }

  async createNeuralMesh() {
    console.log(chalk.cyan('🕸️  Creating neural mesh...'));
    
    const result = await this.mcpServer.tools.executeTool('neural_mesh_init', {
      topology: 'mesh',
      maxAgents: 10,
      strategy: 'parallel',
      enableConsensus: true,
      cryptoLevel: 'quantum'
    });

    console.log(chalk.green(`✅ Neural mesh created: ${result.meshId}`));
    console.log(`   Topology: ${result.config.topology}`);
    console.log(`   Max Agents: ${result.config.maxAgents}`);
    console.log(`   Strategy: ${result.config.strategy}`);
    
    return result.meshId;
  }

  async spawnAgents(meshId) {
    console.log(chalk.cyan('🤖 Spawning neural agents...'));
    
    const agentTypes = [
      { type: 'coordinator', capabilities: ['coordination', 'planning'] },
      { type: 'researcher', capabilities: ['data-analysis', 'information-gathering'] },
      { type: 'coder', capabilities: ['code-generation', 'optimization'] },
      { type: 'analyst', capabilities: ['pattern-recognition', 'insights'] },
      { type: 'architect', capabilities: ['system-design', 'architecture'] }
    ];

    const agents = [];

    for (const config of agentTypes) {
      const result = await this.mcpServer.tools.executeTool('neural_agent_spawn', {
        meshId,
        type: config.type,
        name: `example-${config.type}`,
        capabilities: config.capabilities,
        neuralModel: 'adaptive-v1',
        resources: { cpu: 1, memory: 512 }
      });

      agents.push(result.agentId);
      console.log(chalk.green(`✅ Agent spawned: ${result.agent.name} (${result.agentId})`));
    }

    console.log(chalk.green(`✅ Total agents spawned: ${agents.length}`));
    return agents;
  }

  async demonstrateMemory() {
    console.log(chalk.cyan('💾 Demonstrating memory operations...'));
    
    // Store various types of data
    const dataItems = [
      { key: 'config', value: { version: '1.0', debug: true }, namespace: 'system' },
      { key: 'metrics', value: { cpu: 75, memory: 60, disk: 45 }, namespace: 'performance' },
      { key: 'agents-count', value: 5, namespace: 'stats' },
      { key: 'last-consensus', value: new Date().toISOString(), namespace: 'consensus' }
    ];

    // Store data
    for (const item of dataItems) {
      await this.mcpServer.tools.executeTool('mesh_memory_store', {
        key: item.key,
        value: item.value,
        namespace: item.namespace,
        ttl: 3600 // 1 hour
      });
      console.log(chalk.green(`✅ Stored: ${item.namespace}/${item.key}`));
    }

    // Retrieve and verify data
    for (const item of dataItems) {
      const result = await this.mcpServer.tools.executeTool('mesh_memory_retrieve', {
        key: item.key,
        namespace: item.namespace
      });
      
      console.log(chalk.green(`✅ Retrieved: ${item.namespace}/${item.key}`));
      console.log(`   Value: ${JSON.stringify(result.value)}`);
    }
  }

  async performConsensus(meshId, agents) {
    console.log(chalk.cyan('🤝 Performing consensus operations...'));
    
    const proposals = [
      { action: 'update-config', value: { newSetting: 'optimized' } },
      { action: 'scale-mesh', value: { targetAgents: 15 } },
      { action: 'change-strategy', value: { strategy: 'adaptive' } }
    ];

    for (const proposal of proposals) {
      const result = await this.mcpServer.tools.executeTool('neural_consensus', {
        meshId,
        proposal,
        agents: agents.slice(0, 3), // Use first 3 agents
        consensusType: 'majority'
      });

      console.log(chalk.green(`✅ Consensus ${result.result}: ${proposal.action}`));
      console.log(`   Votes: ${result.votes.accept} accept, ${result.votes.reject} reject`);
      console.log(`   Consensus ID: ${result.consensusId}`);
    }
  }

  async monitorPerformance() {
    console.log(chalk.cyan('📊 Monitoring performance...'));
    
    const result = await this.mcpServer.tools.executeTool('mesh_performance', {
      timeframe: '1h',
      metrics: ['cpu', 'memory', 'throughput', 'latency']
    });

    console.log(chalk.green('✅ Performance metrics retrieved'));
    console.log(`   Timeframe: ${result.timeframe}`);
    console.log(`   Total data points: ${result.summary.totalDataPoints}`);
    
    if (result.summary.cpu) {
      console.log(`   CPU - Avg: ${result.summary.cpu.avg?.toFixed(2)}%, Max: ${result.summary.cpu.max}%`);
    }
    if (result.summary.memory) {
      console.log(`   Memory - Avg: ${result.summary.memory.avg?.toFixed(2)}%, Max: ${result.summary.memory.max}%`);
    }
  }

  async demonstrateEventStreaming(meshId) {
    console.log(chalk.cyan('📡 Demonstrating event streaming...'));
    
    // Create event stream for mesh events
    const streamId = this.mcpServer.events.streamNeuralMeshEvents({
      'data.meshId': meshId
    });

    const receivedEvents = [];
    
    // Subscribe to events
    const subscriberId = this.mcpServer.events.subscribe(streamId, (event) => {
      receivedEvents.push(event);
      console.log(chalk.blue(`📨 Event received: ${event.type} - ${event.data.eventType}`));
    });

    // Generate some test events
    this.mcpServer.events.emitNeuralMeshEvent('optimization', { 
      meshId, 
      type: 'topology',
      improvement: 15.5 
    });

    this.mcpServer.events.emitNeuralMeshEvent('scaling', { 
      meshId, 
      action: 'add-agent',
      newCount: 6 
    });

    this.mcpServer.events.emitNeuralMeshEvent('consensus-reached', { 
      meshId, 
      proposal: 'config-update',
      result: 'accepted' 
    });

    // Wait a bit for events to be processed
    await new Promise(resolve => setTimeout(resolve, 100));

    console.log(chalk.green(`✅ Event streaming demo completed`));
    console.log(`   Events received: ${receivedEvents.length}`);

    // Clean up
    this.mcpServer.events.unsubscribe(subscriberId);
    this.mcpServer.events.destroyStream(streamId);
  }
}

// Advanced usage example
class AdvancedUsageExample {
  constructor() {
    this.mcpServer = null;
  }

  async run() {
    console.log(chalk.blue('🚀 Advanced Usage Example'));
    console.log();

    try {
      await this.initializeWithAuth();
      await this.demonstrateLoadBalancing();
      await this.demonstrateNeuralTraining();
      await this.demonstrateSecurityOperations();
      await this.demonstrateAnalytics();
      
      console.log();
      console.log(chalk.green('✅ Advanced usage example completed!'));
      
    } catch (error) {
      console.error(chalk.red('❌ Advanced example failed:'), error.message);
    } finally {
      if (this.mcpServer) {
        await this.mcpServer.stop();
      }
    }
  }

  async initializeWithAuth() {
    console.log(chalk.cyan('🔐 Initializing with authentication...'));
    
    this.mcpServer = new SynapticMeshMCP({
      transport: 'stdio',
      enableAuth: true,
      apiKeys: [
        {
          key: 'example-api-key-123',
          name: 'example-key',
          permissions: ['neural_*', 'mesh_*']
        }
      ],
      enableEvents: true,
      wasmEnabled: false
    });

    await this.mcpServer.initialize();
    console.log(chalk.green('✅ Server initialized with authentication'));
  }

  async demonstrateLoadBalancing() {
    console.log(chalk.cyan('⚖️  Demonstrating load balancing...'));
    
    // Create mesh first
    const meshResult = await this.mcpServer.tools.executeTool('neural_mesh_init', {
      topology: 'hierarchical',
      maxAgents: 8,
      strategy: 'balanced'
    });

    const tasks = [
      { id: 'task-1', type: 'computation', workload: 'heavy' },
      { id: 'task-2', type: 'analysis', workload: 'medium' },
      { id: 'task-3', type: 'optimization', workload: 'light' },
      { id: 'task-4', type: 'training', workload: 'heavy' }
    ];

    const result = await this.mcpServer.tools.executeTool('load_balance', {
      meshId: meshResult.meshId,
      tasks,
      strategy: 'adaptive'
    });

    console.log(chalk.green(`✅ Load balancing completed`));
    console.log(`   Tasks distributed: ${tasks.length}`);
    console.log(`   Strategy: adaptive`);
  }

  async demonstrateNeuralTraining() {
    console.log(chalk.cyan('🧠 Demonstrating neural training...'));
    
    const meshResult = await this.mcpServer.tools.executeTool('neural_mesh_init', {
      topology: 'mesh',
      maxAgents: 5,
      strategy: 'parallel'
    });

    const trainingData = {
      inputs: [[1, 0], [0, 1], [1, 1], [0, 0]],
      outputs: [[1], [1], [0], [0]], // XOR function
      labels: ['XOR training data']
    };

    const result = await this.mcpServer.tools.executeTool('neural_train', {
      meshId: meshResult.meshId,
      modelType: 'feedforward',
      trainingData,
      epochs: 50,
      distributionStrategy: 'data_parallel'
    });

    console.log(chalk.green(`✅ Neural training completed`));
    console.log(`   Model: ${result.modelType || 'feedforward'}`);
    console.log(`   Epochs: 50`);
    console.log(`   Distribution: data_parallel`);
  }

  async demonstrateSecurityOperations() {
    console.log(chalk.cyan('🔒 Demonstrating security operations...'));
    
    const sensitiveData = {
      userCredentials: 'secret-password-123',
      apiKeys: ['key1', 'key2', 'key3'],
      configuration: { debug: false, production: true }
    };

    // Encrypt data
    const encryptResult = await this.mcpServer.tools.executeTool('security_encrypt', {
      data: sensitiveData,
      algorithm: 'AES-256',
      keyId: 'primary-key'
    });

    console.log(chalk.green(`✅ Data encrypted`));
    console.log(`   Algorithm: AES-256`);
    console.log(`   Key ID: primary-key`);

    // Decrypt data
    const decryptResult = await this.mcpServer.tools.executeTool('security_decrypt', {
      encryptedData: encryptResult.encryptedData || 'mock-encrypted-data',
      keyId: 'primary-key'
    });

    console.log(chalk.green(`✅ Data decrypted successfully`));
  }

  async demonstrateAnalytics() {
    console.log(chalk.cyan('📈 Demonstrating analytics...'));
    
    const meshResult = await this.mcpServer.tools.executeTool('neural_mesh_init', {
      topology: 'star',
      maxAgents: 6,
      strategy: 'centralized'
    });

    const analyticsResult = await this.mcpServer.tools.executeTool('mesh_analytics', {
      meshId: meshResult.meshId,
      analysisType: 'performance',
      timeframe: '24h'
    });

    console.log(chalk.green(`✅ Analytics completed`));
    console.log(`   Analysis type: performance`);
    console.log(`   Timeframe: 24h`);
    console.log(`   Mesh ID: ${meshResult.meshId}`);
  }
}

// Main execution
async function main() {
  const args = process.argv.slice(2);
  const runAdvanced = args.includes('--advanced');

  if (runAdvanced) {
    const advancedExample = new AdvancedUsageExample();
    await advancedExample.run();
  } else {
    const basicExample = new BasicUsageExample();
    await basicExample.run();
  }
}

// Run example if this file is executed directly
if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch(error => {
    console.error(chalk.red('❌ Example execution failed:'), error);
    process.exit(1);
  });
}

export { BasicUsageExample, AdvancedUsageExample };