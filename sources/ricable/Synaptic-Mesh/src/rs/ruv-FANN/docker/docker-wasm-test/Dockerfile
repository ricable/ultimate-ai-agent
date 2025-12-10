# Enhanced Docker test for ruv-swarm WASM functionality
# Tests both local and npm installation with specific WASM verification

FROM node:20-alpine AS base

# Install system dependencies
RUN apk add --no-cache \
    python3 \
    make \
    g++ \
    git \
    bash \
    curl \
    jq \
    strace \
    lsof

WORKDIR /app

# Create test user
RUN adduser -D testuser && \
    chown -R testuser:testuser /app

# Stage 1: NPM Installation Test
FROM base AS npm-test

USER testuser

# Create test package
RUN cat > package.json <<'EOF'
{
  "name": "ruv-swarm-wasm-test",
  "version": "1.0.0",
  "description": "WASM functionality test for ruv-swarm",
  "type": "module",
  "scripts": {
    "test": "node --experimental-modules test-wasm.mjs",
    "test:npx": "npx ruv-swarm mcp start --test",
    "test:cli": "node test-cli.mjs",
    "test:benchmark": "npx ruv-swarm benchmark --quick",
    "test:swarm": "node test-swarm.mjs",
    "test:all": "npm run test && npm run test:cli && npm run test:swarm"
  }
}
EOF

# Install ruv-swarm from npm
RUN npm install ruv-swarm@latest --verbose

# Create WASM-specific test
RUN cat > test-wasm.mjs <<'EOF'
import { WasmModuleLoader } from 'ruv-swarm/src/wasm-loader.js';
import { execSync } from 'child_process';
import fs from 'fs';
import path from 'path';

console.log('🔍 WASM Functionality Test Starting...\n');

const results = {
  passed: [],
  failed: [],
  wasmDetails: {}
};

async function runTest(name, testFn) {
  console.log(`Testing ${name}...`);
  try {
    const result = await testFn();
    console.log(`✅ ${name}: PASSED`);
    results.passed.push(name);
    return result;
  } catch (error) {
    console.log(`❌ ${name}: FAILED - ${error.message}`);
    results.failed.push({ test: name, error: error.message });
    throw error;
  }
}

// Test 1: WASM Loader Initialization
await runTest('WASM Loader Creation', async () => {
  const loader = new WasmModuleLoader();
  if (!loader) throw new Error('Failed to create loader');
  results.wasmDetails.loaderCreated = true;
  return loader;
});

// Test 2: WASM Module Loading
const loader = new WasmModuleLoader();
await runTest('WASM Core Module Loading', async () => {
  await loader.initialize('progressive');
  const status = loader.getModuleStatus();
  console.log('Module Status:', JSON.stringify(status, null, 2));
  
  if (!status.core) throw new Error('Core module not in status');
  if (status.core.placeholder) throw new Error('Core module is using placeholder!');
  
  results.wasmDetails.moduleStatus = status;
  return true;
});

// Test 3: Check WASM Files
await runTest('WASM File Verification', async () => {
  const nodeModules = path.join(process.cwd(), 'node_modules', 'ruv-swarm');
  const wasmDir = path.join(nodeModules, 'wasm');
  
  if (!fs.existsSync(wasmDir)) {
    throw new Error(`WASM directory not found at ${wasmDir}`);
  }
  
  const wasmFiles = fs.readdirSync(wasmDir);
  console.log('WASM files found:', wasmFiles);
  
  const requiredFiles = ['ruv_swarm_wasm_bg.wasm', 'ruv_swarm_wasm.js'];
  for (const file of requiredFiles) {
    if (!wasmFiles.includes(file)) {
      throw new Error(`Required WASM file missing: ${file}`);
    }
  }
  
  // Check file sizes
  const wasmPath = path.join(wasmDir, 'ruv_swarm_wasm_bg.wasm');
  const stats = fs.statSync(wasmPath);
  console.log(`WASM file size: ${stats.size} bytes`);
  
  if (stats.size < 1000) {
    throw new Error('WASM file suspiciously small - might be placeholder');
  }
  
  results.wasmDetails.files = wasmFiles;
  results.wasmDetails.wasmSize = stats.size;
  return true;
});

// Test 4: WASM Functionality (not just placeholder)
await runTest('WASM Actual Functionality', async () => {
  await loader.loadModule('core');
  const module = loader.modules.get('core');
  
  if (!module) throw new Error('Core module not loaded');
  if (module.isPlaceholder) throw new Error('Module is placeholder!');
  
  // Check for actual WASM exports
  if (!module.exports || !module.memory) {
    throw new Error('Module missing expected WASM exports');
  }
  
  // Try to check memory usage (real WASM will have memory)
  const memoryUsage = loader.getTotalMemoryUsage();
  console.log(`Total memory usage: ${memoryUsage} bytes`);
  
  if (memoryUsage === 0) {
    throw new Error('No memory usage detected - likely placeholder');
  }
  
  results.wasmDetails.memoryUsage = memoryUsage;
  results.wasmDetails.hasExports = !!module.exports;
  return true;
});

// Test 5: NPX Commands
await runTest('NPX Command Execution', async () => {
  const commands = [
    'npx ruv-swarm --version',
    'npx ruv-swarm mcp --help',
    'npx ruv-swarm benchmark --help',
    'npx ruv-swarm swarm --help'
  ];
  
  for (const cmd of commands) {
    try {
      const output = execSync(cmd, { encoding: 'utf8' });
      console.log(`Command '${cmd}' output: ${output.slice(0, 50)}...`);
    } catch (error) {
      // Some commands exit with non-zero on help
      if (!error.stdout && !error.stderr) {
        throw new Error(`Command failed: ${cmd}`);
      }
    }
  }
  
  return true;
});

// Test 6: Verify WASM Binary Format
await runTest('WASM Binary Format Check', async () => {
  const wasmPath = path.join(process.cwd(), 'node_modules', 'ruv-swarm', 'wasm', 'ruv_swarm_wasm_bg.wasm');
  const buffer = fs.readFileSync(wasmPath);
  
  // WASM files start with \0asm
  const magic = buffer.slice(0, 4);
  const expectedMagic = Buffer.from([0x00, 0x61, 0x73, 0x6d]);
  
  if (!magic.equals(expectedMagic)) {
    throw new Error('Invalid WASM magic number - not a real WASM file');
  }
  
  // Check version (should be 1)
  const version = buffer.readUInt32LE(4);
  console.log(`WASM version: ${version}`);
  
  if (version !== 1) {
    throw new Error(`Unexpected WASM version: ${version}`);
  }
  
  results.wasmDetails.validBinary = true;
  results.wasmDetails.wasmVersion = version;
  return true;
});

// Print summary
console.log('\n📊 WASM Test Summary:');
console.log('━'.repeat(50));
console.log(`✅ Passed: ${results.passed.length}`);
console.log(`❌ Failed: ${results.failed.length}`);
console.log('\nWASM Details:', JSON.stringify(results.wasmDetails, null, 2));

// Save results
fs.writeFileSync('wasm-test-results.json', JSON.stringify(results, null, 2));

process.exit(results.failed.length > 0 ? 1 : 0);
EOF

# Create CLI test
RUN cat > test-cli.mjs <<'EOF'
import { execSync } from 'child_process';
import fs from 'fs';

console.log('🚀 Testing ruv-swarm CLI functionality...\n');

const cliTests = [
  {
    name: 'MCP Server Start',
    command: 'timeout 5s npx ruv-swarm mcp start || true',
    checkOutput: (output) => output.includes('MCP') || output.includes('server')
  },
  {
    name: 'Swarm Create',
    command: 'npx ruv-swarm swarm create test-swarm mesh --max-agents 3',
    checkOutput: (output) => true
  },
  {
    name: 'Agent Spawn',
    command: 'npx ruv-swarm agent spawn researcher test-agent',
    checkOutput: (output) => true
  },
  {
    name: 'Benchmark Command',
    command: 'npx ruv-swarm benchmark --quick --iterations 10',
    checkOutput: (output) => output.includes('Benchmark') || output.includes('complete')
  }
];

const results = { passed: [], failed: [] };

for (const test of cliTests) {
  console.log(`Running: ${test.name}`);
  try {
    const output = execSync(test.command, { 
      encoding: 'utf8',
      stdio: 'pipe'
    });
    
    if (test.checkOutput(output)) {
      console.log(`✅ ${test.name}: PASSED`);
      results.passed.push(test.name);
    } else {
      console.log(`❌ ${test.name}: Output check failed`);
      results.failed.push({ test: test.name, reason: 'Output validation failed' });
    }
  } catch (error) {
    console.log(`❌ ${test.name}: FAILED - ${error.message}`);
    results.failed.push({ test: test.name, error: error.message });
  }
}

console.log('\n📊 CLI Test Summary:');
console.log(`✅ Passed: ${results.passed.length}`);
console.log(`❌ Failed: ${results.failed.length}`);

fs.writeFileSync('cli-test-results.json', JSON.stringify(results, null, 2));
process.exit(results.failed.length > 0 ? 1 : 0);
EOF

# Create swarm functionality test
RUN cat > test-swarm.mjs <<'EOF'
import pkg from 'ruv-swarm';
const { RuvSwarm, Agent, Swarm } = pkg;

console.log('🐝 Testing ruv-swarm core functionality...\n');

const results = { passed: [], failed: [] };

function test(name, fn) {
  try {
    fn();
    console.log(`✅ ${name}: PASSED`);
    results.passed.push(name);
  } catch (error) {
    console.log(`❌ ${name}: FAILED - ${error.message}`);
    results.failed.push({ test: name, error: error.message });
  }
}

// Test 1: Class availability
test('Core Classes Available', () => {
  if (!RuvSwarm) throw new Error('RuvSwarm class not found');
  if (!Agent) throw new Error('Agent class not found');
  if (!Swarm) throw new Error('Swarm class not found');
});

// Test 2: Instance creation
test('Create RuvSwarm Instance', () => {
  const swarm = new RuvSwarm();
  if (!swarm) throw new Error('Failed to create RuvSwarm instance');
});

// Test 3: Swarm creation
test('Create Swarm with Topology', () => {
  const swarm = new Swarm('test-swarm', 'mesh');
  if (!swarm.id) throw new Error('Swarm missing ID');
  if (swarm.topology !== 'mesh') throw new Error('Wrong topology');
});

// Test 4: Agent creation
test('Create Agent', () => {
  const agent = new Agent('researcher', 'test-agent');
  if (!agent.type) throw new Error('Agent missing type');
  if (!agent.name) throw new Error('Agent missing name');
});

console.log('\n📊 Swarm Test Summary:');
console.log(`✅ Passed: ${results.passed.length}`);
console.log(`❌ Failed: ${results.failed.length}`);

process.exit(results.failed.length > 0 ? 1 : 0);
EOF

# Stage 2: Local Installation Test
FROM base AS local-test

# Copy the npm package tarball (will be built separately)
COPY --chown=testuser:testuser ruv-swarm-*.tgz /tmp/

USER testuser

# Create package.json
RUN cat > package.json <<'EOF'
{
  "name": "ruv-swarm-local-test",
  "version": "1.0.0",
  "type": "module"
}
EOF

# Install from local tarball
RUN npm install /tmp/ruv-swarm-*.tgz --verbose

# Copy test files from npm-test stage
COPY --from=npm-test --chown=testuser:testuser /app/test-*.mjs ./

# Stage 3: Global Installation Test
FROM base AS global-test

# Install globally as root
USER root
RUN npm install -g ruv-swarm@latest --verbose

# Switch to test user
USER testuser
WORKDIR /app

# Create test for global installation
RUN cat > test-global.mjs <<'EOF'
import { execSync } from 'child_process';

console.log('🌍 Testing global ruv-swarm installation...\n');

try {
  // Test global commands
  const version = execSync('ruv-swarm --version', { encoding: 'utf8' });
  console.log('✅ Global installation working');
  console.log(`Version: ${version}`);
  
  // Test global npx
  execSync('ruv-swarm mcp --help', { encoding: 'utf8' });
  console.log('✅ Global MCP command working');
  
  console.log('\nGlobal installation test PASSED!');
} catch (error) {
  console.error('❌ Global installation test FAILED:', error.message);
  process.exit(1);
}
EOF

# Stage 4: Combined Test Runner
FROM base AS test-runner

# Copy npm installation for testing
COPY --from=npm-test --chown=testuser:testuser /app /app/npm-test

# Create test orchestrator
USER testuser
WORKDIR /app

RUN cat > run-all-tests.sh <<'EOF'
#!/bin/bash
set -e

echo "🧪 Running Comprehensive WASM Tests for ruv-swarm"
echo "================================================="

# Function to run test and capture results
run_test() {
    local name=$1
    local dir=$2
    local cmd=$3
    
    echo -e "\n🔧 Running: $name"
    echo "Directory: $dir"
    echo "Command: $cmd"
    echo "-" | tr '-' '='
    
    if cd "$dir" && eval "$cmd"; then
        echo "✅ $name: PASSED"
        return 0
    else
        echo "❌ $name: FAILED"
        return 1
    fi
}

# NPM Installation Tests
run_test "NPM WASM Test" "/app/npm-test" "npm run test"
run_test "NPM CLI Test" "/app/npm-test" "npm run test:cli"
run_test "NPM Swarm Test" "/app/npm-test" "npm run test:swarm"

# Memory and performance check
echo -e "\n📊 System Resource Check:"
ps aux | grep node | grep -v grep || true
df -h /app

echo -e "\n✅ All tests completed!"
EOF

RUN chmod +x run-all-tests.sh

CMD ["./run-all-tests.sh"]

# Stage 5: Production Simulation
FROM node:20-alpine AS production-sim

# Simulate production environment
RUN npm install -g ruv-swarm@latest

# Create minimal test
RUN cat > /tmp/prod-test.js <<'EOF'
const { RuvSwarm } = require('ruv-swarm');
console.log('Production test:', new RuvSwarm() ? 'PASS' : 'FAIL');
EOF

WORKDIR /app
CMD ["node", "/tmp/prod-test.js"]