#!/usr/bin/env tsx
// Validation script for all quick wins
import { spawnSync } from 'child_process';
import { existsSync } from 'fs';

interface ValidationResult {
  name: string;
  passed: boolean;
  message: string;
  duration?: number;
}

const results: ValidationResult[] = [];

function validate(name: string, check: () => boolean, message: string): void {
  const startTime = Date.now();
  try {
    const passed = check();
    results.push({
      name,
      passed,
      message: passed ? `✓ ${message}` : `✗ ${message}`,
      duration: Date.now() - startTime
    });
  } catch (error) {
    results.push({
      name,
      passed: false,
      message: `✗ ${message} - Error: ${error}`,
      duration: Date.now() - startTime
    });
  }
}

console.log('🔍 Validating Quick Wins Implementation\n');

// Quick Win 1: Tool Integration
validate(
  'Tool Integration',
  () => existsSync('src/config/tools.ts'),
  'Tool configuration file exists'
);

validate(
  'Tool Config Content',
  () => {
    const config = require('../src/config/tools.ts');
    return config.toolConfig && config.toolConfig.enableAllTools === true;
  },
  'Tools are configured to be enabled'
);

// Quick Win 2: Streaming Support
validate(
  'Streaming in Agents',
  () => {
    const agent = require('../src/agents/webResearchAgent.ts');
    return agent.webResearchAgent.length === 2; // Has onStream parameter
  },
  'Agents support streaming callbacks'
);

// Quick Win 3: Error Handling & Retry
validate(
  'Retry Utility',
  () => existsSync('src/utils/retry.ts'),
  'Retry utility exists'
);

validate(
  'Retry Implementation',
  () => {
    const retry = require('../src/utils/retry.ts');
    return typeof retry.withRetry === 'function';
  },
  'withRetry function is exported'
);

// Quick Win 4: Structured Logging
validate(
  'Logger Utility',
  () => existsSync('src/utils/logger.ts'),
  'Logger utility exists'
);

validate(
  'Logger Methods',
  () => {
    const { logger } = require('../src/utils/logger.ts');
    return (
      typeof logger.info === 'function' &&
      typeof logger.error === 'function' &&
      typeof logger.warn === 'function' &&
      typeof logger.debug === 'function'
    );
  },
  'Logger has all required methods'
);

// Quick Win 5: Health Check Endpoint
validate(
  'Health Check',
  () => existsSync('src/health.ts'),
  'Health check module exists'
);

validate(
  'Health Check Functions',
  () => {
    const health = require('../src/health.ts');
    return (
      typeof health.getHealthStatus === 'function' &&
      typeof health.startHealthServer === 'function'
    );
  },
  'Health check functions are exported'
);

// Integration Validation
validate(
  'Main Index Updated',
  () => {
    const fs = require('fs');
    const indexContent = fs.readFileSync('src/index.ts', 'utf-8');
    return (
      indexContent.includes('logger') &&
      indexContent.includes('startHealthServer')
    );
  },
  'Main index file uses new utilities'
);

// Print Results
console.log('═══════════════════════════════════════════════════════\n');
console.log('VALIDATION RESULTS:\n');

let passed = 0;
let failed = 0;

results.forEach((result) => {
  console.log(result.message);
  if (result.duration) {
    console.log(`  (${result.duration}ms)`);
  }
  result.passed ? passed++ : failed++;
});

console.log('\n═══════════════════════════════════════════════════════');
console.log(`\nTotal: ${results.length} checks`);
console.log(`✓ Passed: ${passed}`);
console.log(`✗ Failed: ${failed}`);
console.log(`Success Rate: ${Math.round((passed / results.length) * 100)}%\n`);

if (failed > 0) {
  console.error('❌ Validation failed. Please fix the issues above.');
  process.exit(1);
} else {
  console.log('✅ All validations passed!');
  process.exit(0);
}
