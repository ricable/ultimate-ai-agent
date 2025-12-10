/**
 * Jest Setup File for Claude Skills Orchestration Tests
 * Configures global test environment and mocks for action execution and cognitive consciousness systems
 */

import 'jest';

// Set test environment variables
process.env.NODE_ENV = 'test';
process.env.LOG_LEVEL = 'error';

// Global test timeout
jest.setTimeout(30000);

// Mock console methods to reduce test noise
global.console = {
  ...console,
  log: jest.fn(),
  info: jest.fn(),
  warn: jest.fn(),
  error: jest.fn()
};

// Setup global mocks for Node.js modules that might not be available in test environment
jest.mock('fs', () => ({
  readFileSync: jest.fn(),
  writeFileSync: jest.fn(),
  existsSync: jest.fn(() => true),
  mkdirSync: jest.fn(),
  readdirSync: jest.fn(() => [])
}));

jest.mock('path', () => ({
  ...jest.requireActual('path'),
  join: jest.fn((...args) => args.join('/')),
  resolve: jest.fn((...args) => args.join('/'))
}));

// Global test utilities
declare global {
  namespace jest {
    interface Matchers<R> {
      toBeValidTimestamp(): R;
      toBeWithinRange(min: number, max: number): R;
      toBeValidSkill(): R;
      toBeValidActionExecution(): R;
      toBeValidOrchestration(): R;
      toHaveValidCognitiveIntegration(): R;
      toHaveValidTemporalAnalysis(): R;
      toHaveValidPerformanceMetrics(): R;
    }
  }
}

// Custom matchers
expect.extend({
  toBeValidTimestamp(received: number) {
    const pass = received > 0 && received <= Date.now();
    return {
      message: () => `expected ${received} to be a valid timestamp`,
      pass,
    };
  },

  toBeWithinRange(received: number, floor: number, ceiling: number) {
    const pass = received >= floor && received <= ceiling;
    return {
      message: () => `expected ${received} to be within range ${floor} - ${ceiling}`,
      pass,
    };
  },

  toBeValidSkill(received) {
    const isValid = received &&
      typeof received.id === 'string' &&
      typeof received.name === 'string' &&
      typeof received.type === 'string' &&
      Array.isArray(received.capabilities) &&
      ['low', 'medium', 'high', 'critical'].includes(received.priority);

    if (isValid) {
      return {
        message: () => `expected ${received} not to be a valid skill`,
        pass: true
      };
    } else {
      return {
        message: () => `expected ${received} to be a valid skill with id, name, type, capabilities, and priority`,
        pass: false
      };
    }
  },

  toBeValidActionExecution(received) {
    const isValid = received &&
      typeof received.id === 'string' &&
      typeof received.skillId === 'string' &&
      typeof received.action === 'string' &&
      typeof received.status === 'string' &&
      typeof received.startTime === 'number' &&
      ['pending', 'executing', 'completed', 'failed', 'recovered'].includes(received.status);

    if (isValid) {
      return {
        message: () => `expected ${received} not to be a valid action execution`,
        pass: true
      };
    } else {
      return {
        message: () => `expected ${received} to be a valid action execution with required fields`,
        pass: false
      };
    }
  },

  toBeValidOrchestration(received) {
    const isValid = received &&
      typeof received.id === 'string' &&
      typeof received.name === 'string' &&
      Array.isArray(received.skills) &&
      ['hierarchical', 'mesh', 'ring', 'star'].includes(received.coordination) &&
      typeof received.consensusRequired === 'boolean';

    if (isValid) {
      return {
        message: () => `expected ${received} not to be a valid orchestration`,
        pass: true
      };
    } else {
      return {
        message: () => `expected ${received} to be a valid orchestration with required fields`,
        pass: false
      };
    }
  },

  toHaveValidCognitiveIntegration(received) {
    const isValid = received &&
      received.cognitiveAnalysis &&
      typeof received.cognitiveAnalysis.consciousnessLevel === 'number' &&
      received.cognitiveAnalysis.temporalDepth &&
      received.performanceMetrics &&
      typeof received.performanceMetrics.solveRate === 'number' &&
      typeof received.performanceMetrics.speedImprovement === 'number';

    if (isValid) {
      return {
        message: () => `expected ${received} not to have valid cognitive integration`,
        pass: true
      };
    } else {
      return {
        message: () => `expected ${received} to have valid cognitive integration with consciousness analysis and performance metrics`,
        pass: false
      };
    }
  },

  toHaveValidTemporalAnalysis(received) {
    const isValid = received &&
      received.temporalAnalysis &&
      typeof received.temporalAnalysis.depth === 'number' &&
      Array.isArray(received.temporalAnalysis.insights) &&
      typeof received.temporalAnalysis.expansionFactor === 'number';

    if (isValid) {
      return {
        message: () => `expected ${received} not to have valid temporal analysis`,
        pass: true
      };
    } else {
      return {
        message: () => `expected ${received} to have valid temporal analysis with depth, insights, and expansion factor`,
        pass: false
      };
    }
  },

  toHaveValidPerformanceMetrics(received) {
    const isValid = received &&
      received.performanceMetrics &&
      typeof received.performanceMetrics.solveRate === 'number' &&
      typeof received.performanceMetrics.speedImprovement === 'number' &&
      typeof received.performanceMetrics.executionTime === 'number';

    if (isValid) {
      return {
        message: () => `expected ${received} not to have valid performance metrics`,
        pass: true
      };
    } else {
      return {
        message: () => `expected ${received} to have valid performance metrics with solve rate, speed improvement, and execution time`,
        pass: false
      };
    }
  }
});

// Cleanup after each test
afterEach(() => {
  jest.clearAllMocks();
});