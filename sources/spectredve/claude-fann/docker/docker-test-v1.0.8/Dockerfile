# Fixed Docker test for ruv-swarm v1.0.8
FROM node:18-alpine

# Install essential packages
RUN apk add --no-cache \
    python3 \
    make \
    g++ \
    sqlite \
    bash \
    curl

WORKDIR /app

# Install ruv-swarm from npm
RUN npm install --no-cache ruv-swarm@1.0.8

# Create fixed test script that handles ES modules properly
COPY <<EOF test-npm-install.js
#!/usr/bin/env node

const { spawn } = require('child_process');

async function testRuvSwarmInstall() {
  console.log('🧪 Testing ruv-swarm v1.0.8 from npm...\n');
  
  const tests = [
    {
      name: 'Version Check (--version)',
      command: 'npx',
      args: ['ruv-swarm', '--version'],
      expectedPattern: /1\.0\.8/
    },
    {
      name: 'Version Check (version)',
      command: 'npx',
      args: ['ruv-swarm', 'version'],
      expectedPattern: /1\.0\.8/
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
    console.log('🎉 All tests passed! ruv-swarm v1.0.8 is working correctly in Docker.');
    return true;
  } else {
    console.log('⚠️ Some tests failed. Check the output above for details.');
    return false;
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

// Export for use in main test
if (require.main === module) {
  testRuvSwarmInstall().then(success => {
    process.exit(success ? 0 : 1);
  }).catch(error => {
    console.error('Test runner failed:', error);
    process.exit(1);
  });
} else {
  module.exports = { testRuvSwarmInstall };
}
EOF

# Create ES module test script  
COPY <<EOF test-wasm-functionality.mjs
#!/usr/bin/env node

async function testWasmFunctionality() {
  console.log('🧪 Testing WASM functionality with ES modules...\n');
  
  try {
    // Use dynamic import for ES modules
    const RuvSwarm = await import('ruv-swarm');
    
    console.log('✅ ES Module import successful');
    
    // Test WASM initialization
    const swarm = await RuvSwarm.default.initialize({
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
    
    console.log('\n🎉 WASM functionality tests passed!');
    return true;
    
  } catch (error) {
    console.error('❌ WASM functionality test failed:', error.message);
    if (error.stack) {
      console.error('Stack:', error.stack.split('\n').slice(0, 3).join('\n'));
    }
    return false;
  }
}

// Run if called directly
if (import.meta.url === \`file://\${process.argv[1]}\`) {
  testWasmFunctionality().then(success => {
    process.exit(success ? 0 : 1);
  }).catch(error => {
    console.error('Test runner failed:', error);
    process.exit(1);
  });
}

export { testWasmFunctionality };
EOF

COPY <<EOF run-all-tests.sh
#!/bin/bash

echo "🚀 Starting comprehensive ruv-swarm v1.0.8 Docker tests..."
echo "=================================================="

# Test 1: NPM Installation and CLI
echo ""
echo "📦 Test 1: NPM Installation and CLI Commands"
echo "----------------------------------------"
node test-npm-install.js
INSTALL_TEST_RESULT=\$?

# Test 2: WASM Functionality (ES modules)
echo ""
echo "🔧 Test 2: WASM Functionality (ES Modules)"
echo "------------------------------------------"
node test-wasm-functionality.mjs
WASM_TEST_RESULT=\$?

# Test 3: MCP Protocol Test (improved)
echo ""
echo "🔌 Test 3: MCP Protocol Functionality"
echo "--------------------------------"
timeout 15s npx ruv-swarm mcp start --protocol=stdio > mcp_output.log 2>&1 &
MCP_PID=\$!
sleep 8

# Check if MCP process is running using a more compatible method
if kill -0 \$MCP_PID 2>/dev/null; then
    echo "✅ MCP server started successfully"
    kill \$MCP_PID 2>/dev/null
    MCP_TEST_RESULT=0
else
    echo "❌ MCP server failed to start"
    echo "MCP output:"
    cat mcp_output.log 2>/dev/null || echo "No output captured"
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
    echo "🎉 ALL TESTS PASSED! ruv-swarm v1.0.8 is working correctly in Docker."
    exit 0
else
    echo ""
    echo "⚠️ \$TOTAL_FAILURES test(s) failed. Check the output above for details."
    exit 1
fi
EOF

RUN chmod +x test-npm-install.js test-wasm-functionality.mjs run-all-tests.sh

CMD ["./run-all-tests.sh"]