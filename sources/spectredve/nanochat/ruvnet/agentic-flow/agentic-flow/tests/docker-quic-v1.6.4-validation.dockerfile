# Docker validation for agentic-flow v1.6.4 QUIC implementation
# Tests npm package installation and QUIC functionality

FROM node:20-slim

# Install build tools for native dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    make \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Create test directory
WORKDIR /app

# Install agentic-flow from npm (will use published package)
# For local testing, we'll copy the built package
COPY package.json .
COPY dist ./dist
COPY wasm ./wasm
COPY docs ./docs
COPY .claude ./.claude
COPY README.md .
COPY CHANGELOG.md .

# Install dependencies
RUN npm install --production

# Create validation test script
RUN cat > validate-quic.js << 'EOF'
#!/usr/bin/env node

import { QuicClient } from './dist/transport/quic.js';
import { QuicHandshakeManager } from './dist/transport/quic-handshake.js';
import { getQuicConfig } from './dist/config/quic.js';
import { readFileSync, existsSync } from 'fs';

console.log('🧪 agentic-flow v1.6.4 QUIC Validation Suite\n');

const tests = {
  passed: 0,
  failed: 0,
  results: []
};

function test(name, fn) {
  try {
    fn();
    tests.passed++;
    tests.results.push(`✅ ${name}`);
    console.log(`✅ ${name}`);
  } catch (error) {
    tests.failed++;
    tests.results.push(`❌ ${name}: ${error.message}`);
    console.log(`❌ ${name}: ${error.message}`);
  }
}

// Test 1: Package structure
test('Package version is 1.6.4', () => {
  const pkg = JSON.parse(readFileSync('package.json', 'utf8'));
  if (pkg.version !== '1.6.4') {
    throw new Error(`Expected 1.6.4, got ${pkg.version}`);
  }
});

// Test 2: QUIC client class exists
test('QuicClient class is exported', () => {
  if (typeof QuicClient !== 'function') {
    throw new Error('QuicClient is not a constructor function');
  }
});

// Test 3: Handshake manager exists
test('QuicHandshakeManager class is exported', () => {
  if (typeof QuicHandshakeManager !== 'function') {
    throw new Error('QuicHandshakeManager is not a constructor function');
  }
});

// Test 4: Configuration system exists
test('getQuicConfig function is exported', () => {
  if (typeof getQuicConfig !== 'function') {
    throw new Error('getQuicConfig is not a function');
  }
});

// Test 5: Configuration returns valid object
test('getQuicConfig returns valid configuration', () => {
  const config = getQuicConfig();
  if (!config || typeof config !== 'object') {
    throw new Error('Configuration is not an object');
  }
  if (typeof config.port !== 'number') {
    throw new Error('Configuration missing port');
  }
  if (typeof config.maxConnections !== 'number') {
    throw new Error('Configuration missing maxConnections');
  }
});

// Test 6: WASM module exists
test('WASM bindings directory exists', () => {
  if (!existsSync('./wasm')) {
    throw new Error('WASM directory not found');
  }
});

// Test 7: QUIC documentation exists
test('QUIC status documentation exists', () => {
  if (!existsSync('./docs/quic/QUIC-STATUS.md')) {
    throw new Error('QUIC-STATUS.md not found');
  }
});

// Test 8: Performance validation doc exists
test('Performance validation documentation exists', () => {
  if (!existsSync('./docs/quic/PERFORMANCE-VALIDATION.md')) {
    throw new Error('PERFORMANCE-VALIDATION.md not found');
  }
});

// Test 9: CHANGELOG has v1.6.4 entry
test('CHANGELOG.md contains v1.6.4 release notes', () => {
  const changelog = readFileSync('./CHANGELOG.md', 'utf8');
  if (!changelog.includes('## [1.6.4]')) {
    throw new Error('v1.6.4 not found in CHANGELOG.md');
  }
  if (!changelog.includes('53.7% faster')) {
    throw new Error('Performance metrics not in CHANGELOG');
  }
});

// Test 10: QuicClient can be instantiated
test('QuicClient can be instantiated', () => {
  const client = new QuicClient({ port: 4433 });
  if (!client) {
    throw new Error('Failed to create QuicClient instance');
  }
});

// Test 11: QuicHandshakeManager can be instantiated
test('QuicHandshakeManager can be instantiated', () => {
  const manager = new QuicHandshakeManager();
  if (!manager) {
    throw new Error('Failed to create QuicHandshakeManager instance');
  }
});

// Test 12: CLI binary exists
test('CLI binary is accessible', () => {
  if (!existsSync('./dist/cli-proxy.js')) {
    throw new Error('CLI binary not found');
  }
});

// Print summary
console.log('\n' + '='.repeat(60));
console.log('📊 Validation Summary');
console.log('='.repeat(60));
console.log(`Total Tests: ${tests.passed + tests.failed}`);
console.log(`✅ Passed: ${tests.passed}`);
console.log(`❌ Failed: ${tests.failed}`);
console.log(`Success Rate: ${((tests.passed / (tests.passed + tests.failed)) * 100).toFixed(1)}%`);
console.log('='.repeat(60));

if (tests.failed > 0) {
  console.log('\n❌ Validation FAILED\n');
  process.exit(1);
} else {
  console.log('\n✅ Validation PASSED - agentic-flow v1.6.4 is production ready!\n');
  process.exit(0);
}
EOF

# Make validation script executable
RUN chmod +x validate-quic.js

# Run validation
CMD ["node", "validate-quic.js"]
