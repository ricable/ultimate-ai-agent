/**
 * Jest setup file for Synaptic Neural Mesh tests
 * Configures test environment and global utilities
 */

import { jest } from '@jest/globals';

// Global test configuration
global.TEST_TIMEOUT = 30000;
global.LONG_TEST_TIMEOUT = 60000;

// Performance monitoring
global.startTime = null;
global.measurePerformance = (name) => {
  if (!global.startTime) {
    global.startTime = performance.now();
    return `Started measuring: ${name}`;
  } else {
    const endTime = performance.now();
    const duration = endTime - global.startTime;
    global.startTime = null;
    return `${name} took ${duration.toFixed(2)}ms`;
  }
};

// Mock WASM module for testing
global.mockWasmModule = {
  memory: new WebAssembly.Memory({ initial: 1 }),
  exports: {
    neural_inference: jest.fn(() => 0.85),
    dag_validate: jest.fn(() => true),
    encrypt_data: jest.fn((data) => `encrypted_${data}`),
    decrypt_data: jest.fn((data) => data.replace('encrypted_', '')),
    init_network: jest.fn(() => 'network_id_123'),
    train_batch: jest.fn(() => ({ loss: 0.1, accuracy: 0.95 }))
  }
};

// Mock system metrics
global.mockSystemMetrics = {
  cpu: () => Math.random() * 50 + 25, // 25-75%
  memory: () => Math.random() * 40 + 30, // 30-70%
  network: () => Math.random() * 100 + 50, // 50-150 Mbps
  disk: () => Math.random() * 30 + 70 // 70-100% free
};

// Test utilities
global.testUtils = {
  // Generate test data
  generateRandomArray: (size, min = 0, max = 1) => {
    return Array(size).fill(0).map(() => Math.random() * (max - min) + min);
  },
  
  generateTestNeuralNetwork: (layers = [784, 128, 10]) => {
    return {
      id: `network_${Date.now()}`,
      layers,
      weights: layers.slice(0, -1).map((size, i) => 
        Array(layers[i + 1]).fill(0).map(() => 
          Array(size).fill(0).map(() => (Math.random() - 0.5) * 2)
        )
      ),
      biases: layers.slice(1).map(size => 
        Array(size).fill(0).map(() => (Math.random() - 0.5) * 2)
      )
    };
  },
  
  generateTestDAG: (nodeCount = 10) => {
    const nodes = [];
    
    // Genesis node
    nodes.push({
      id: 'genesis',
      data: 'genesis_data',
      parents: [],
      timestamp: Date.now() - nodeCount * 1000
    });
    
    // Additional nodes
    for (let i = 1; i < nodeCount; i++) {
      const possibleParents = nodes.map(n => n.id);
      const parentCount = Math.min(2, Math.floor(Math.random() * possibleParents.length) + 1);
      const parents = [];
      
      for (let j = 0; j < parentCount; j++) {
        const parentIndex = Math.floor(Math.random() * possibleParents.length);
        if (!parents.includes(possibleParents[parentIndex])) {
          parents.push(possibleParents[parentIndex]);
        }
      }
      
      nodes.push({
        id: `node_${i}`,
        data: `data_${i}`,
        parents,
        timestamp: Date.now() - (nodeCount - i) * 1000
      });
    }
    
    return nodes;
  },
  
  generateTestAgent: (type = 'coder', capabilities = []) => {
    return {
      id: `agent_${type}_${Date.now()}_${Math.random().toString(36).substr(2, 5)}`,
      type,
      capabilities: capabilities.length > 0 ? capabilities : ['coordination', 'communication'],
      status: 'spawned',
      tasks: [],
      memory: new Map(),
      metrics: {
        tasksCompleted: 0,
        avgExecutionTime: 0,
        successRate: 100
      },
      
      // Mock methods
      assignTask: jest.fn(),
      completeTask: jest.fn(),
      updateStatus: jest.fn(),
      getMetrics: jest.fn(() => ({
        tasksCompleted: Math.floor(Math.random() * 100),
        avgExecutionTime: Math.random() * 1000,
        successRate: Math.random() * 100
      }))
    };
  },
  
  // Async utilities
  delay: (ms) => new Promise(resolve => setTimeout(resolve, ms)),
  
  timeout: (promise, ms) => {
    return Promise.race([
      promise,
      new Promise((_, reject) => 
        setTimeout(() => reject(new Error(`Timeout after ${ms}ms`)), ms)
      )
    ]);
  },
  
  // Performance testing
  measureAsync: async (fn, name = 'operation') => {
    const start = performance.now();
    const result = await fn();
    const end = performance.now();
    
    return {
      result,
      duration: end - start,
      name
    };
  },
  
  // Memory testing
  estimateMemoryUsage: (obj) => {
    const seen = new WeakSet();
    
    function sizeOf(obj) {
      if (obj === null || typeof obj !== 'object') {
        return typeof obj === 'string' ? obj.length * 2 : 8;
      }
      
      if (seen.has(obj)) return 0;
      seen.add(obj);
      
      let size = 0;
      for (const key in obj) {
        if (obj.hasOwnProperty(key)) {
          size += sizeOf(key) + sizeOf(obj[key]);
        }
      }
      
      return size;
    }
    
    return sizeOf(obj);
  }
};

// Custom matchers
expect.extend({
  toBeWithinRange(received, floor, ceiling) {
    const pass = received >= floor && received <= ceiling;
    if (pass) {
      return {
        message: () => `expected ${received} not to be within range ${floor} - ${ceiling}`,
        pass: true,
      };
    } else {
      return {
        message: () => `expected ${received} to be within range ${floor} - ${ceiling}`,
        pass: false,
      };
    }
  },
  
  toBeValidDAGNode(received) {
    const pass = received && 
                 typeof received.id === 'string' &&
                 received.data !== undefined &&
                 Array.isArray(received.parents) &&
                 typeof received.timestamp === 'number';
    
    if (pass) {
      return {
        message: () => `expected ${JSON.stringify(received)} not to be a valid DAG node`,
        pass: true,
      };
    } else {
      return {
        message: () => `expected ${JSON.stringify(received)} to be a valid DAG node`,
        pass: false,
      };
    }
  },
  
  toBeValidAgent(received) {
    const pass = received &&
                 typeof received.id === 'string' &&
                 typeof received.type === 'string' &&
                 Array.isArray(received.capabilities) &&
                 typeof received.status === 'string';
    
    if (pass) {
      return {
        message: () => `expected agent not to be valid`,
        pass: true,
      };
    } else {
      return {
        message: () => `expected agent to be valid`,
        pass: false,
      };
    }
  },
  
  toHavePerformanceWithin(received, expectedMs) {
    const pass = received.duration <= expectedMs;
    
    if (pass) {
      return {
        message: () => `expected ${received.name} (${received.duration}ms) to exceed ${expectedMs}ms`,
        pass: true,
      };
    } else {
      return {
        message: () => `expected ${received.name} (${received.duration}ms) to complete within ${expectedMs}ms`,
        pass: false,
      };
    }
  }
});

// Global error handling
process.on('unhandledRejection', (reason, promise) => {
  console.error('Unhandled Rejection at:', promise, 'reason:', reason);
});

process.on('uncaughtException', (error) => {
  console.error('Uncaught Exception:', error);
});

// Console overrides for test output
const originalConsole = { ...console };

global.quietConsole = () => {
  console.log = jest.fn();
  console.info = jest.fn();
  console.warn = jest.fn();
  console.error = jest.fn();
};

global.restoreConsole = () => {
  Object.assign(console, originalConsole);
};

// Test environment info
console.log('🧪 Synaptic Neural Mesh Test Environment Initialized');
console.log(`   Node.js: ${process.version}`);
console.log(`   Platform: ${process.platform}`);
console.log(`   Architecture: ${process.arch}`);
console.log(`   Memory: ${Math.round(process.memoryUsage().heapTotal / 1024 / 1024)}MB available`);
console.log(`   Test timeout: ${global.TEST_TIMEOUT}ms`);
console.log('');