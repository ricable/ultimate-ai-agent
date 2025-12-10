/**
 * Global setup for Synaptic Neural Mesh test suite
 * Initializes test environment and resources
 */

import { existsSync, mkdirSync, writeFileSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

export default async function globalSetup() {
  console.log('🔧 Setting up global test environment...');
  
  // Create test directories
  const testDirs = [
    'reports',
    'coverage',
    'artifacts',
    'temp'
  ];
  
  testDirs.forEach(dir => {
    const dirPath = join(__dirname, dir);
    if (!existsSync(dirPath)) {
      mkdirSync(dirPath, { recursive: true });
      console.log(`   ✓ Created directory: ${dir}`);
    }
  });
  
  // Initialize test database (SQLite for testing)
  const testDbPath = join(__dirname, 'temp', 'test.db');
  
  // Create test configuration
  const testConfig = {
    environment: 'test',
    database: {
      path: testDbPath,
      type: 'sqlite'
    },
    neural: {
      maxNetworks: 100,
      defaultTimeout: 30000
    },
    swarm: {
      maxAgents: 1000,
      coordinationTimeout: 5000
    },
    dag: {
      maxNodes: 10000,
      validationTimeout: 1000
    },
    performance: {
      targets: {
        neuralInference: 100, // ms
        memoryPerAgent: 52428800, // 50MB in bytes
        concurrentAgents: 1000,
        swarmCoordination: 1000, // ms
        sweBenchScore: 84.8, // percentage
        systemThroughput: 10000 // ops/second
      }
    },
    stress: {
      duration: {
        short: 5000, // 5 seconds
        medium: 30000, // 30 seconds
        long: 300000 // 5 minutes
      },
      thresholds: {
        errorRate: 5, // percentage
        recoveryRate: 80, // percentage
        memoryGrowthRate: 10 // MB per minute
      }
    }
  };
  
  const configPath = join(__dirname, 'temp', 'test-config.json');
  writeFileSync(configPath, JSON.stringify(testConfig, null, 2));
  console.log('   ✓ Created test configuration');
  
  // Set environment variables
  process.env.NODE_ENV = 'test';
  process.env.TEST_CONFIG_PATH = configPath;
  process.env.TEST_DB_PATH = testDbPath;
  process.env.TEST_REPORTS_DIR = join(__dirname, 'reports');
  process.env.TEST_COVERAGE_DIR = join(__dirname, 'coverage');
  
  console.log('   ✓ Set environment variables');
  
  // Initialize mock services
  global.testServices = {
    startTime: Date.now(),
    
    // Mock neural service
    neural: {
      networks: new Map(),
      
      createNetwork: (config) => {
        const id = `net_${Date.now()}_${Math.random().toString(36).substr(2, 5)}`;
        const network = {
          id,
          config,
          status: 'ready',
          created: Date.now()
        };
        global.testServices.neural.networks.set(id, network);
        return network;
      },
      
      getNetwork: (id) => global.testServices.neural.networks.get(id),
      
      inference: async (networkId, input) => {
        const network = global.testServices.neural.networks.get(networkId);
        if (!network) throw new Error(`Network ${networkId} not found`);
        
        // Mock inference with random output
        await new Promise(resolve => setTimeout(resolve, Math.random() * 10));
        return Array(10).fill(0).map(() => Math.random());
      }
    },
    
    // Mock DAG service
    dag: {
      nodes: new Map(),
      
      addNode: (nodeData, parents = []) => {
        const id = `node_${Date.now()}_${Math.random().toString(36).substr(2, 5)}`;
        const node = {
          id,
          data: nodeData,
          parents,
          timestamp: Date.now(),
          validated: true
        };
        global.testServices.dag.nodes.set(id, node);
        return node;
      },
      
      getNode: (id) => global.testServices.dag.nodes.get(id),
      
      validateDAG: () => {
        // Mock DAG validation
        return { valid: true, cycleDetected: false };
      },
      
      getTips: () => {
        // Mock getting DAG tips
        const allNodes = Array.from(global.testServices.dag.nodes.values());
        return allNodes
          .filter(node => !allNodes.some(other => other.parents.includes(node.id)))
          .map(node => node.id);
      }
    },
    
    // Mock swarm service
    swarm: {
      agents: new Map(),
      
      spawnAgent: (type, capabilities = []) => {
        const id = `agent_${type}_${Date.now()}_${Math.random().toString(36).substr(2, 5)}`;
        const agent = {
          id,
          type,
          capabilities,
          status: 'active',
          tasks: [],
          spawned: Date.now()
        };
        global.testServices.swarm.agents.set(id, agent);
        return agent;
      },
      
      getAgent: (id) => global.testServices.swarm.agents.get(id),
      
      assignTask: (agentId, task) => {
        const agent = global.testServices.swarm.agents.get(agentId);
        if (!agent) throw new Error(`Agent ${agentId} not found`);
        
        agent.tasks.push({
          ...task,
          assigned: Date.now(),
          status: 'assigned'
        });
        
        return true;
      },
      
      getSwarmStatus: () => {
        const agents = Array.from(global.testServices.swarm.agents.values());
        return {
          totalAgents: agents.length,
          activeAgents: agents.filter(a => a.status === 'active').length,
          totalTasks: agents.reduce((sum, a) => sum + a.tasks.length, 0),
          avgTasksPerAgent: agents.length > 0 ? 
            agents.reduce((sum, a) => sum + a.tasks.length, 0) / agents.length : 0
        };
      }
    },
    
    // Mock memory service
    memory: {
      store: new Map(),
      
      set: (key, value, ttl = null) => {
        const entry = {
          value,
          timestamp: Date.now(),
          ttl
        };
        global.testServices.memory.store.set(key, entry);
        return true;
      },
      
      get: (key) => {
        const entry = global.testServices.memory.store.get(key);
        if (!entry) return null;
        
        if (entry.ttl && Date.now() - entry.timestamp > entry.ttl) {
          global.testServices.memory.store.delete(key);
          return null;
        }
        
        return entry.value;
      },
      
      delete: (key) => {
        return global.testServices.memory.store.delete(key);
      },
      
      clear: () => {
        global.testServices.memory.store.clear();
      },
      
      size: () => global.testServices.memory.store.size
    }
  };
  
  console.log('   ✓ Initialized mock services');
  
  // Create test manifest
  const manifest = {
    setupTime: Date.now(),
    environment: 'test',
    nodeVersion: process.version,
    platform: process.platform,
    architecture: process.arch,
    memoryAvailable: Math.round(process.memoryUsage().heapTotal / 1024 / 1024),
    directories: testDirs,
    services: Object.keys(global.testServices),
    config: testConfig
  };
  
  const manifestPath = join(__dirname, 'temp', 'test-manifest.json');
  writeFileSync(manifestPath, JSON.stringify(manifest, null, 2));
  console.log('   ✓ Created test manifest');
  
  // Verify WASM support
  try {
    const wasmSupported = typeof WebAssembly !== 'undefined' && 
                         typeof WebAssembly.Memory !== 'undefined';
    
    if (wasmSupported) {
      // Test basic WASM functionality
      const memory = new WebAssembly.Memory({ initial: 1 });
      console.log('   ✓ WASM support verified');
    } else {
      console.warn('   ⚠️ WASM support not available');
    }
  } catch (error) {
    console.warn('   ⚠️ WASM verification failed:', error.message);
  }
  
  // Setup performance monitoring
  global.testMetrics = {
    startTime: Date.now(),
    testCount: 0,
    passCount: 0,
    failCount: 0,
    skipCount: 0,
    totalDuration: 0,
    
    recordTest: (name, status, duration) => {
      global.testMetrics.testCount++;
      global.testMetrics.totalDuration += duration;
      
      switch (status) {
        case 'pass':
          global.testMetrics.passCount++;
          break;
        case 'fail':
          global.testMetrics.failCount++;
          break;
        case 'skip':
          global.testMetrics.skipCount++;
          break;
      }
    },
    
    getSummary: () => {
      const totalTime = Date.now() - global.testMetrics.startTime;
      return {
        totalTests: global.testMetrics.testCount,
        passed: global.testMetrics.passCount,
        failed: global.testMetrics.failCount,
        skipped: global.testMetrics.skipCount,
        totalExecutionTime: totalTime,
        avgTestDuration: global.testMetrics.testCount > 0 ? 
          global.testMetrics.totalDuration / global.testMetrics.testCount : 0,
        passRate: global.testMetrics.testCount > 0 ?
          (global.testMetrics.passCount / global.testMetrics.testCount) * 100 : 0
      };
    }
  };
  
  console.log('   ✓ Setup performance monitoring');
  
  console.log('✅ Global test environment setup complete');
  console.log('');
}