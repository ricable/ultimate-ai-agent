/**
 * Jest Global Setup for Polyglot DevPod Functional Tests
 * 
 * Prepares the test environment before any tests run
 */

import { spawn } from 'child_process';

export default async function globalSetup() {
  console.log('🚀 Starting Polyglot DevPod Functional Test Global Setup');
  
  // Verify required tools and dependencies
  const requiredTools = [
    { name: 'DevPod CLI', command: 'devpod --version' },
    { name: 'Nushell', command: 'nu --version' },
    { name: 'Docker', command: 'docker --version' },
    { name: 'Node.js', command: 'node --version' },
    { name: 'NPM', command: 'npm --version' }
  ];
  
  console.log('🔍 Verifying required tools...');
  
  for (const tool of requiredTools) {
    try {
      const result = await executeCommand(tool.command);
      if (result.success) {
        console.log(`  ✅ ${tool.name}: Available`);
      } else {
        console.log(`  ❌ ${tool.name}: Not available - ${result.error}`);
        throw new Error(`Required tool not available: ${tool.name}`);
      }
    } catch (error) {
      console.log(`  ❌ ${tool.name}: Error - ${error.message}`);
      throw new Error(`Failed to verify tool: ${tool.name}`);
    }
  }
  
  // Verify centralized DevPod management script
  console.log('🔍 Verifying centralized DevPod management...');
  const scriptPath = '../../host-tooling/devpod-management/manage-devpod.nu';
  const scriptCheck = await executeCommand(`test -f ${scriptPath} && echo "exists"`);
  
  if (!scriptCheck.success || !scriptCheck.output.includes('exists')) {
    throw new Error('Centralized DevPod management script not found');
  }
  console.log('  ✅ Centralized DevPod management: Available');
  
  // Check DevPod workspace capacity
  console.log('🔍 Checking DevPod workspace capacity...');
  const workspaceList = await executeCommand('devpod list --output json');
  
  if (workspaceList.success) {
    try {
      const workspaces = JSON.parse(workspaceList.output);
      const currentCount = workspaces.length;
      console.log(`  📊 Current workspaces: ${currentCount}`);
      
      if (currentCount > 10) {
        console.log('  ⚠️ Warning: High number of existing workspaces may affect tests');
      }
    } catch (error) {
      console.log('  ⚠️ Could not parse workspace list, continuing...');
    }
  }
  
  // Pre-clean any existing test workspaces
  console.log('🧹 Cleaning up any existing test workspaces...');
  const cleanupResult = await executeCommand(
    'devpod list --output json | jq -r \'.[] | select(.name | contains("test")) | .name\' | xargs -I {} devpod delete {} --force || true'
  );
  
  if (cleanupResult.success) {
    console.log('  ✅ Test workspace cleanup completed');
  } else {
    console.log('  ⚠️ Test workspace cleanup had warnings, continuing...');
  }
  
  // Initialize test directories
  console.log('📁 Creating test directories...');
  await executeCommand('mkdir -p ./test-reports ./coverage');
  console.log('  ✅ Test directories created');
  
  // Record setup completion
  const setupInfo = {
    timestamp: new Date().toISOString(),
    nodeVersion: process.version,
    platform: process.platform,
    architecture: process.arch,
    environment: process.env.NODE_ENV || 'test'
  };
  
  console.log('📋 Setup Information:');
  console.log(`  Node.js: ${setupInfo.nodeVersion}`);
  console.log(`  Platform: ${setupInfo.platform} (${setupInfo.architecture})`);
  console.log(`  Environment: ${setupInfo.environment}`);
  
  console.log('✅ Global setup completed successfully');
}

async function executeCommand(command, timeout = 30000) {
  return new Promise((resolve) => {
    const child = spawn('bash', ['-c', command], {
      stdio: ['pipe', 'pipe', 'pipe']
    });

    let output = '';
    let error = '';

    child.stdout?.on('data', (data) => {
      output += data.toString();
    });

    child.stderr?.on('data', (data) => {
      error += data.toString();
    });

    child.on('close', (code) => {
      resolve({
        success: code === 0,
        output: output.trim(),
        error: error.trim() || undefined
      });
    });

    // Timeout handling
    setTimeout(() => {
      child.kill();
      resolve({
        success: false,
        output: '',
        error: 'Command timeout'
      });
    }, timeout);
  });
}