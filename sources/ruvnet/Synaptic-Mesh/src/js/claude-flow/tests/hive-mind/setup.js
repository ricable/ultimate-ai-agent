/**
 * Test setup for Hive Mind tests
 * Configures test environment and utilities
 */

import { jest } from '@jest/globals';
import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Set test environment variables
process.env.NODE_ENV = 'test';
process.env.HIVE_TEST_MODE = 'true';
process.env.HIVE_LOG_LEVEL = process.env.CI ? 'error' : 'warn';

// Global test utilities
global.testUtils = {
  // Create temporary test directory
  createTempDir: () => {
    const tempDir = path.join(__dirname, '../../tmp/test-' + Date.now());
    fs.mkdirSync(tempDir, { recursive: true });
    return tempDir;
  },
  
  // Clean up temporary directory
  cleanupTempDir: (dir) => {
    if (fs.existsSync(dir)) {
      fs.rmSync(dir, { recursive: true, force: true });
    }
  },
  
  // Mock SQLite database
  createMockDb: () => {
    return {
      prepare: jest.fn(() => ({
        run: jest.fn(),
        get: jest.fn(),
        all: jest.fn()
      })),
      exec: jest.fn(),
      close: jest.fn(),
      transaction: jest.fn((fn) => fn)
    };
  },
  
  // Wait for condition
  waitFor: async (condition, timeout = 5000, interval = 100) => {
    const start = Date.now();
    while (!condition()) {
      if (Date.now() - start > timeout) {
        throw new Error('Timeout waiting for condition');
      }
      await new Promise(resolve => setTimeout(resolve, interval));
    }
  },
  
  // Performance measurement
  measurePerformance: async (name, fn) => {
    const start = process.hrtime.bigint();
    const memStart = process.memoryUsage();
    
    const result = await fn();
    
    const end = process.hrtime.bigint();
    const memEnd = process.memoryUsage();
    
    return {
      name,
      result,
      duration: Number(end - start) / 1e6, // Convert to milliseconds
      memory: {
        heapUsed: (memEnd.heapUsed - memStart.heapUsed) / 1024 / 1024,
        external: (memEnd.external - memStart.external) / 1024 / 1024
      }
    };
  },
  
  // Generate test data
  generateAgents: (count, type = 'mixed') => {
    const types = ['researcher', 'coder', 'analyst', 'architect', 'tester', 'coordinator'];
    const agents = [];
    
    for (let i = 0; i < count; i++) {
      agents.push({
        id: `test-agent-${i}`,
        name: `TestAgent${i}`,
        type: type === 'mixed' ? types[i % types.length] : type,
        status: 'active',
        capabilities: [],
        metadata: {
          createdAt: Date.now(),
          testAgent: true
        }
      });
    }
    
    return agents;
  },
  
  generateTasks: (count, agents = []) => {
    const tasks = [];
    
    for (let i = 0; i < count; i++) {
      tasks.push({
        id: `test-task-${i}`,
        description: `Test task ${i}`,
        status: 'pending',
        priority: ['low', 'medium', 'high'][i % 3],
        assignedTo: agents.length > 0 ? agents[i % agents.length].id : null,
        createdAt: Date.now()
      });
    }
    
    return tasks;
  }
};

// Jest configuration
jest.setTimeout(30000); // 30 second default timeout

// Mock timers for specific tests
global.mockTimers = () => {
  jest.useFakeTimers();
  return {
    restore: () => jest.useRealTimers(),
    advance: (ms) => jest.advanceTimersByTime(ms),
    runAll: () => jest.runAllTimers()
  };
};

// Console spy for testing output
global.consoleSpies = {
  log: jest.spyOn(console, 'log').mockImplementation(),
  error: jest.spyOn(console, 'error').mockImplementation(),
  warn: jest.spyOn(console, 'warn').mockImplementation(),
  
  restore: () => {
    console.log.mockRestore();
    console.error.mockRestore();
    console.warn.mockRestore();
  },
  
  getOutput: (type = 'log') => {
    return console[type].mock.calls.map(call => call.join(' ')).join('\n');
  }
};

// Clean up after all tests
afterAll(() => {
  // Restore console
  global.consoleSpies.restore();
  
  // Clean up temp directories
  const tmpDir = path.join(__dirname, '../../tmp');
  if (fs.existsSync(tmpDir)) {
    const files = fs.readdirSync(tmpDir);
    files.forEach(file => {
      if (file.startsWith('test-')) {
        fs.rmSync(path.join(tmpDir, file), { recursive: true, force: true });
      }
    });
  }
});

// Performance tracking
if (process.env.TRACK_PERFORMANCE === 'true') {
  const performanceResults = [];
  
  afterEach((done) => {
    const testName = expect.getState().currentTestName;
    const duration = Date.now() - global.__testStartTime;
    
    performanceResults.push({
      test: testName,
      duration,
      memory: process.memoryUsage()
    });
    
    done();
  });
  
  beforeEach(() => {
    global.__testStartTime = Date.now();
  });
  
  afterAll(() => {
    const reportPath = path.join(__dirname, '../../test-results/performance.json');
    fs.writeFileSync(reportPath, JSON.stringify(performanceResults, null, 2));
    console.log(`Performance report saved to: ${reportPath}`);
  });
}

// Export for use in tests
export default {
  testUtils: global.testUtils,
  mockTimers: global.mockTimers,
  consoleSpies: global.consoleSpies
};