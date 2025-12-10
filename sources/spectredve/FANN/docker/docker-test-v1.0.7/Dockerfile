# Multi-stage Docker build to test ruv-swarm v1.0.7 from npm
FROM node:18-alpine AS base

# Install essential packages for WASM and sqlite3
RUN apk add --no-cache \
    python3 \
    make \
    g++ \
    sqlite \
    bash

# Set working directory
WORKDIR /app

# Test stage - Install ruv-swarm from npm and test it
FROM base AS test

# Install the published package from npm
RUN npm install --no-cache ruv-swarm@1.0.7

# Create test script
COPY <<EOF test-npm-install.js
#!/usr/bin/env node

const { spawn } = require('child_process');
const fs = require('fs');

async function testRuvSwarmInstall() {
  console.log('🧪 Testing ruv-swarm v1.0.7 from npm...\n');
  
  const tests = [
    {
      name: 'Version Check',
      command: 'npx',
      args: ['ruv-swarm', '--version'],
      expectedPattern: /1\.0\.7/
    },
    {
      name: 'Help Command',
      command: 'npx', 
      args: ['ruv-swarm', '--help'],
      expectedPattern: /ruv-swarm/i
    },
    {
      name: 'MCP Server Startup',
      command: 'timeout',
      args: ['10s', 'npx', 'ruv-swarm', 'mcp', 'start', '--protocol=stdio'],
      expectedPattern: /ruv-swarm MCP server starting/i,
      timeout: 15000
    },
    {
      name: 'Init Command',
      command: 'npx',
      args: ['ruv-swarm', 'init', '--topology=mesh', '--max-agents=3'],
      expectedPattern: /swarm.*initialized|created/i
    },
    {
      name: 'Node.js Module Import',
      command: 'node',
      args: ['-e', 'const ruv = require("ruv-swarm"); console.log("✅ Module imported successfully");'],
      expectedPattern: /Module imported successfully/
    }
  ];

  let passed = 0;
  let failed = 0;

  for (const test of tests) {
    console.log(\`🔧 Running test: \${test.name}...\`);
    
    try {
      const result = await runCommand(test.command, test.args, test.timeout || 10000);
      
      if (test.expectedPattern.test(result.stdout + result.stderr)) {
        console.log(\`✅ \${test.name}: PASS\`);
        passed++;
      } else {
        console.log(\`❌ \${test.name}: FAIL - Pattern not found\`);
        console.log(\`   Expected: \${test.expectedPattern}\`);
        console.log(\`   Got: \${(result.stdout + result.stderr).substring(0, 200)}...\`);
        failed++;
      }
    } catch (error) {
      console.log(\`❌ \${test.name}: FAIL - \${error.message}\`);
      failed++;
    }
    
    console.log('');
  }

  console.log(\`📊 Test Results: \${passed} passed, \${failed} failed\`);
  
  if (failed === 0) {
    console.log('🎉 All tests passed! ruv-swarm v1.0.7 is working correctly in Docker.');
    process.exit(0);
  } else {
    console.log('⚠️ Some tests failed. Check the output above for details.');
    process.exit(1);
  }
}

function runCommand(command, args, timeout = 10000) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, { 
      stdio: ['pipe', 'pipe', 'pipe'],
      shell: true 
    });
    
    let stdout = '';
    let stderr = '';
    
    child.stdout.on('data', (data) => {
      stdout += data.toString();
    });
    
    child.stderr.on('data', (data) => {
      stderr += data.toString();
    });
    
    const timer = setTimeout(() => {
      child.kill('SIGTERM');
      resolve({ stdout, stderr, code: -1 });
    }, timeout);
    
    child.on('close', (code) => {
      clearTimeout(timer);
      resolve({ stdout, stderr, code });
    });
    
    child.on('error', (error) => {
      clearTimeout(timer);
      reject(error);
    });
  });
}

testRuvSwarmInstall().catch(console.error);
EOF

# Make test script executable
RUN chmod +x test-npm-install.js

# Add a second test to verify WASM functionality
COPY <<EOF test-wasm-functionality.js
#!/usr/bin/env node

async function testWasmFunctionality() {
  console.log('🧪 Testing WASM functionality...\n');
  
  try {
    const RuvSwarm = require('ruv-swarm');
    
    console.log('✅ Module import successful');
    
    // Test WASM initialization
    const swarm = await RuvSwarm.initialize({
      enableWasm: true,
      enableSIMD: false // Disable SIMD for Docker compatibility
    });
    
    console.log('✅ WASM initialization successful');
    
    // Test basic swarm creation
    const testSwarm = await swarm.createSwarm({
      topology: 'mesh',
      maxAgents: 2,
      strategy: 'balanced'
    });
    
    console.log('✅ Swarm creation successful');
    console.log(\`   Swarm ID: \${testSwarm.id}\`);
    console.log(\`   Topology: \${testSwarm.topology}\`);
    
    // Test agent spawning
    const agent = await testSwarm.spawn({
      type: 'researcher',
      name: 'test-agent'
    });
    
    console.log('✅ Agent spawning successful');
    console.log(\`   Agent ID: \${agent.id}\`);
    console.log(\`   Agent Type: \${agent.type}\`);
    
    // Test basic task execution
    const result = await agent.execute({
      task: 'simple test task',
      timeout: 5000
    });
    
    console.log('✅ Task execution successful');
    console.log(\`   Result: \${JSON.stringify(result).substring(0, 100)}...\`);
    
    console.log('\n🎉 All WASM functionality tests passed!');
    process.exit(0);
    
  } catch (error) {
    console.error('❌ WASM functionality test failed:', error.message);
    console.error('Stack:', error.stack);
    process.exit(1);
  }
}

testWasmFunctionality();
EOF

RUN chmod +x test-wasm-functionality.js

# Create comprehensive test script
COPY <<EOF run-all-tests.sh
#!/bin/bash

echo "🚀 Starting comprehensive ruv-swarm v1.0.7 Docker tests..."
echo "=================================================="

# Test 1: NPM Installation and CLI
echo ""
echo "📦 Test 1: NPM Installation and CLI Commands"
echo "----------------------------------------"
node test-npm-install.js
INSTALL_TEST_RESULT=\$?

# Test 2: WASM Functionality
echo ""
echo "🔧 Test 2: WASM Functionality"
echo "-------------------------"
node test-wasm-functionality.js
WASM_TEST_RESULT=\$?

# Test 3: MCP Protocol Test
echo ""
echo "🔌 Test 3: MCP Protocol Functionality"
echo "--------------------------------"
timeout 30s npx ruv-swarm mcp start --protocol=stdio > mcp_output.log 2>&1 &
MCP_PID=\$!
sleep 5

if ps -p \$MCP_PID > /dev/null; then
    echo "✅ MCP server started successfully"
    kill \$MCP_PID 2>/dev/null
    MCP_TEST_RESULT=0
else
    echo "❌ MCP server failed to start"
    MCP_TEST_RESULT=1
fi

# Test 4: Package Integrity
echo ""
echo "📋 Test 4: Package Integrity"  
echo "------------------------"
echo "Checking installed files..."
ls -la node_modules/ruv-swarm/ | head -10
echo ""
echo "Checking WASM files..."
ls -la node_modules/ruv-swarm/wasm/ 2>/dev/null || echo "WASM directory not found"
echo ""
echo "Checking package.json version..."
node -e "console.log('Version:', require('ruv-swarm/package.json').version)"

# Summary
echo ""
echo "📊 Test Summary"
echo "=============="
echo "NPM Installation: \$([ \$INSTALL_TEST_RESULT -eq 0 ] && echo '✅ PASS' || echo '❌ FAIL')"
echo "WASM Functionality: \$([ \$WASM_TEST_RESULT -eq 0 ] && echo '✅ PASS' || echo '❌ FAIL')"  
echo "MCP Protocol: \$([ \$MCP_TEST_RESULT -eq 0 ] && echo '✅ PASS' || echo '❌ FAIL')"

TOTAL_FAILURES=\$((INSTALL_TEST_RESULT + WASM_TEST_RESULT + MCP_TEST_RESULT))

if [ \$TOTAL_FAILURES -eq 0 ]; then
    echo ""
    echo "🎉 ALL TESTS PASSED! ruv-swarm v1.0.7 is working correctly in Docker."
    exit 0
else
    echo ""
    echo "⚠️ \$TOTAL_FAILURES test(s) failed. Check the output above for details."
    exit 1
fi
EOF

RUN chmod +x run-all-tests.sh

# Set the default command to run all tests
CMD ["./run-all-tests.sh"]