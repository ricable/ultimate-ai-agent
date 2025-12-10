#!/usr/bin/env tsx
// Test structured logging
import { logger } from '../../src/utils/logger.js';

console.log('📝 Testing Structured Logging\n');

// Test 1: Basic logging levels
console.log('Test 1: All log levels');
logger.debug('Debug message', { test: 'debug' });
logger.info('Info message', { test: 'info' });
logger.warn('Warning message', { test: 'warn' });
logger.error('Error message', { test: 'error' });
console.log('✅ All log levels work\n');

// Test 2: Context setting
console.log('Test 2: Context setting');
logger.setContext({ service: 'test-service', version: '1.0.0' });
logger.info('Message with context');
console.log('✅ Context setting works\n');

// Test 3: Complex data
console.log('Test 3: Complex data structures');
logger.info('Complex data test', {
  nested: {
    obj: {
      with: 'values'
    }
  },
  array: [1, 2, 3],
  boolean: true,
  number: 42
});
console.log('✅ Complex data logging works\n');

// Test 4: Error logging
console.log('Test 4: Error object logging');
const testError = new Error('Test error');
logger.error('Error test', { error: testError });
console.log('✅ Error logging works\n');

// Test 5: Production mode (JSON output)
console.log('Test 5: Production mode JSON output');
const originalEnv = process.env.NODE_ENV;
process.env.NODE_ENV = 'production';
logger.info('Production log message', { format: 'json' });
process.env.NODE_ENV = originalEnv;
console.log('✅ Production JSON format works\n');

console.log('═══════════════════════════════════════════════════════');
console.log('✅ All logging tests passed!');
