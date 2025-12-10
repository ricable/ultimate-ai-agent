/**
 * Jest Test Setup
 * Global setup and configuration for Kimi-K2 integration tests
 */

const fs = require('fs-extra');
const path = require('path');
const { v4: uuidv4 } = require('uuid');

// Global test configuration
global.TEST_CONFIG = {
  timeout: 30000,
  retries: 3,
  tempDir: '/tmp/synaptic-test',
  mockApiKey: 'test-kimi-k2-key-12345',
  testSession: uuidv4()
};

// Extended Jest matchers for Kimi-K2 testing
expect.extend({
  toBeValidKimiResponse(received) {
    const pass = received && 
                 typeof received.content === 'string' &&
                 received.content.length > 0 &&
                 received.usage &&
                 typeof received.usage.total_tokens === 'number';
    
    if (pass) {
      return {
        message: () => `expected ${received} not to be a valid Kimi response`,
        pass: true,
      };
    } else {
      return {
        message: () => `expected ${received} to be a valid Kimi response with content and usage`,
        pass: false,
      };
    }
  },

  toHaveExecutedTools(received, expectedCount) {
    const toolCalls = received.tool_calls || [];
    const pass = toolCalls.length >= expectedCount;
    
    if (pass) {
      return {
        message: () => `expected response not to have executed ${expectedCount} or more tools`,
        pass: true,
      };
    } else {
      return {
        message: () => `expected response to have executed at least ${expectedCount} tools, got ${toolCalls.length}`,
        pass: false,
      };
    }
  },

  toBeWithinContextLimit(received, contextLimit = 128000) {
    const tokenCount = received.usage?.prompt_tokens || 0;
    const pass = tokenCount <= contextLimit;
    
    if (pass) {
      return {
        message: () => `expected token count ${tokenCount} to exceed context limit ${contextLimit}`,
        pass: true,
      };
    } else {
      return {
        message: () => `expected token count ${tokenCount} to be within context limit ${contextLimit}`,
        pass: false,
      };
    }
  },

  toHaveReasonableLatency(received, maxLatencyMs = 5000) {
    const latency = received.latency || received.responseTime || 0;
    const pass = latency <= maxLatencyMs;
    
    if (pass) {
      return {
        message: () => `expected latency ${latency}ms to exceed maximum ${maxLatencyMs}ms`,
        pass: true,
      };
    } else {
      return {
        message: () => `expected latency ${latency}ms to be within ${maxLatencyMs}ms`,
        pass: false,
      };
    }
  },

  toContainValidDAGNode(received) {
    const pass = received &&
                 received.id &&
                 received.type &&
                 received.timestamp &&
                 received.signature;
    
    if (pass) {
      return {
        message: () => `expected ${received} not to contain a valid DAG node`,
        pass: true,
      };
    } else {
      return {
        message: () => `expected ${received} to contain a valid DAG node with id, type, timestamp, and signature`,
        pass: false,
      };
    }
  }
});

// Global setup before all tests
beforeAll(async () => {
  // Create temporary test directory
  await fs.ensureDir(global.TEST_CONFIG.tempDir);
  
  // Setup mock environment variables
  process.env.NODE_ENV = 'test';
  process.env.SYNAPTIC_TEST_MODE = 'true';
  process.env.KIMI_API_KEY = global.TEST_CONFIG.mockApiKey;
  
  // Initialize mock servers if needed
  console.log(`🧪 Starting Kimi-K2 integration tests - Session: ${global.TEST_CONFIG.testSession}`);
});

// Global cleanup after all tests
afterAll(async () => {
  // Cleanup temporary files
  try {
    await fs.remove(global.TEST_CONFIG.tempDir);
  } catch (error) {
    console.warn('Failed to cleanup test directory:', error.message);
  }
  
  // Reset environment
  delete process.env.SYNAPTIC_TEST_MODE;
  delete process.env.KIMI_API_KEY;
  
  console.log('🧹 Test cleanup completed');
});

// Setup before each test
beforeEach(async () => {
  // Create unique test workspace
  const testWorkspace = path.join(global.TEST_CONFIG.tempDir, `test-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`);
  await fs.ensureDir(testWorkspace);
  
  // Store workspace in global for test access
  global.currentTestWorkspace = testWorkspace;
  
  // Reset any global state
  jest.clearAllMocks();
});

// Cleanup after each test
afterEach(async () => {
  // Cleanup test workspace
  if (global.currentTestWorkspace) {
    try {
      await fs.remove(global.currentTestWorkspace);
    } catch (error) {
      // Ignore cleanup errors in tests
    }
    delete global.currentTestWorkspace;
  }
});

// Mock implementations for testing
global.mockKimiResponse = (content, options = {}) => {
  return {
    content: content || "This is a mock response from Kimi-K2",
    usage: {
      prompt_tokens: options.promptTokens || 100,
      completion_tokens: options.completionTokens || 50,
      total_tokens: (options.promptTokens || 100) + (options.completionTokens || 50)
    },
    model: "kimi-k2-instruct",
    finish_reason: options.finishReason || "stop",
    tool_calls: options.toolCalls || null,
    latency: options.latency || 1500
  };
};

global.mockToolCall = (toolName, args = {}) => {
  return {
    id: `call_${uuidv4()}`,
    type: "function",
    function: {
      name: toolName,
      arguments: JSON.stringify(args)
    }
  };
};

global.mockDAGNode = (type = "test_node", data = {}) => {
  return {
    id: `node_${uuidv4()}`,
    type: type,
    data: data,
    timestamp: Date.now(),
    signature: `sig_${uuidv4()}`,
    agent_id: `agent_${uuidv4()}`
  };
};

// Utility functions for tests
global.testUtils = {
  // Create test configuration
  createTestConfig: (overrides = {}) => {
    return {
      kimi: {
        provider: 'mocktest',
        api_key: global.TEST_CONFIG.mockApiKey,
        model: 'kimi-k2-instruct',
        temperature: 0.6,
        max_tokens: 4096,
        timeout: 120,
        context_window: 128000,
        ...overrides.kimi
      },
      mesh: {
        node_id: `test-node-${uuidv4()}`,
        port: 18080 + Math.floor(Math.random() * 1000),
        network_type: 'testnet',
        ...overrides.mesh
      },
      ...overrides
    };
  },

  // Generate large context for testing
  generateLargeContext: (tokenCount = 1000) => {
    return "token ".repeat(tokenCount);
  },

  // Wait for async operations
  waitFor: (ms) => {
    return new Promise(resolve => setTimeout(resolve, ms));
  },

  // Retry operation with backoff
  retryOperation: async (operation, maxRetries = 3, backoffMs = 1000) => {
    for (let i = 0; i < maxRetries; i++) {
      try {
        return await operation();
      } catch (error) {
        if (i === maxRetries - 1) throw error;
        await global.testUtils.waitFor(backoffMs * Math.pow(2, i));
      }
    }
  },

  // Validate JSON structure
  validateJsonStructure: (obj, requiredFields) => {
    for (const field of requiredFields) {
      if (!(field in obj)) {
        throw new Error(`Missing required field: ${field}`);
      }
    }
    return true;
  },

  // Generate test data
  generateTestData: {
    query: () => `Test query ${Date.now()}: What are the principles of neural networks?`,
    
    complexQuery: () => `
      Complex analysis task ${Date.now()}:
      
      Please analyze the following system architecture:
      - Microservices with REST APIs
      - Event-driven communication
      - Database per service pattern
      - API Gateway for routing
      
      Provide recommendations for:
      1. Performance optimization
      2. Security enhancements  
      3. Scalability improvements
      4. Monitoring and observability
    `,
    
    codeAnalysis: () => `
      Analyze this code snippet:
      
      \`\`\`javascript
      function calculateFibonacci(n) {
        if (n <= 1) return n;
        return calculateFibonacci(n - 1) + calculateFibonacci(n - 2);
      }
      \`\`\`
      
      Identify performance issues and suggest optimizations.
    `,
    
    toolTask: () => `Create a file called test-${Date.now()}.txt with the content "Hello Kimi-K2" and then read it back`,
    
    meshTask: () => `Coordinate with other agents to analyze distributed system patterns`
  }
};

// Error handling for async tests
process.on('unhandledRejection', (reason, promise) => {
  console.error('Unhandled Rejection at:', promise, 'reason:', reason);
  // Don't exit the process in tests, just log the error
});

// Timeout handling
jest.setTimeout(global.TEST_CONFIG.timeout);

// Memory usage monitoring
if (process.env.MONITOR_MEMORY === 'true') {
  let initialMemory = process.memoryUsage();
  
  afterEach(() => {
    const currentMemory = process.memoryUsage();
    const memoryIncrease = (currentMemory.heapUsed - initialMemory.heapUsed) / 1024 / 1024;
    
    if (memoryIncrease > 100) { // More than 100MB increase
      console.warn(`⚠️  High memory usage detected: +${memoryIncrease.toFixed(2)}MB`);
    }
  });
}

console.log('🔧 Kimi-K2 test setup completed');