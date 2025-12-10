/**
 * Jest Setup for Polyglot DevPod Functional Tests
 * 
 * Global test environment configuration and utilities
 */

import { jest } from '@jest/globals';

// Extend Jest timeout for long-running functional tests
jest.setTimeout(600000); // 10 minutes

// Global test configuration
global.FUNCTIONAL_TEST_CONFIG = {
  maxConcurrentWorkspaces: 15,
  defaultTimeout: 300000, // 5 minutes
  workspacePrefix: 'functional-test',
  enableCleanup: true,
  enablePerformanceTracking: true,
  enableResourceMonitoring: true
};

// Global utilities available to all tests
global.testUtils = {
  sleep: (ms) => new Promise(resolve => setTimeout(resolve, ms)),
  
  generateTestId: (prefix = 'test') => 
    `${prefix}-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
  
  formatDuration: (milliseconds) => {
    if (milliseconds < 1000) {
      return `${milliseconds}ms`;
    } else if (milliseconds < 60000) {
      return `${(milliseconds / 1000).toFixed(1)}s`;
    } else {
      return `${(milliseconds / 60000).toFixed(1)}m`;
    }
  },
  
  logTestProgress: (testName, progress, total) => {
    const percentage = Math.round((progress / total) * 100);
    console.log(`📊 ${testName}: ${progress}/${total} (${percentage}%)`);
  }
};

// Global error handling for unhandled promises
process.on('unhandledRejection', (reason, promise) => {
  console.error('Unhandled Rejection at:', promise, 'reason:', reason);
});

process.on('uncaughtException', (error) => {
  console.error('Uncaught Exception:', error);
  process.exit(1);
});

// Console formatting for better test output visibility
const originalLog = console.log;
const originalError = console.error;

console.log = (...args) => {
  const timestamp = new Date().toISOString();
  originalLog(`[${timestamp}]`, ...args);
};

console.error = (...args) => {
  const timestamp = new Date().toISOString();
  originalError(`[${timestamp}] ❌`, ...args);
};

// Test environment validation
console.log('🔧 Functional test environment initialized');
console.log(`⚙️ Configuration:`, global.FUNCTIONAL_TEST_CONFIG);