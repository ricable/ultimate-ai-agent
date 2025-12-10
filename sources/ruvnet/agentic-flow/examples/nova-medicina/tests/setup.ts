/**
 * Jest Test Setup
 * Global configuration for all tests
 */

// Extend timeout for integration tests
jest.setTimeout(30000);

// Mock environment variables
process.env.NODE_ENV = 'test';
process.env.NOVA_MEDICINA_API_KEY = 'test-api-key';
process.env.AGENTDB_PATH = ':memory:';

// Global test utilities
global.console = {
  ...console,
  error: jest.fn(),
  warn: jest.fn(),
  log: jest.fn(),
};

// Clean up after tests
afterAll(() => {
  jest.clearAllMocks();
});
