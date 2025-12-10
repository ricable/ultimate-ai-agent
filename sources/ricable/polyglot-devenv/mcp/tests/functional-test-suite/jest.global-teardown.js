/**
 * Jest Global Teardown for Polyglot DevPod Functional Tests
 * 
 * Cleans up the test environment after all tests complete
 */

import { spawn } from 'child_process';

export default async function globalTeardown() {
  console.log('🧹 Starting Polyglot DevPod Functional Test Global Teardown');
  
  // Clean up all test workspaces
  console.log('🗑️ Cleaning up test workspaces...');
  
  try {
    // Get all test workspaces
    const workspaceListResult = await executeCommand('devpod list --output json');
    
    if (workspaceListResult.success) {
      const workspaces = JSON.parse(workspaceListResult.output);
      const testWorkspaces = workspaces.filter(ws => 
        ws.name.includes('test') || 
        ws.name.includes('functional') ||
        ws.name.includes('polyglot')
      );
      
      console.log(`  📊 Found ${testWorkspaces.length} test workspaces to clean up`);
      
      // Clean up workspaces in parallel (batches of 5)
      const batchSize = 5;
      for (let i = 0; i < testWorkspaces.length; i += batchSize) {
        const batch = testWorkspaces.slice(i, i + batchSize);
        
        const cleanupPromises = batch.map(ws => {
          console.log(`    🗑️ Deleting workspace: ${ws.name}`);
          return executeCommand(`devpod delete ${ws.name} --force`, 60000);
        });
        
        const results = await Promise.allSettled(cleanupPromises);
        const successful = results.filter(r => r.status === 'fulfilled' && r.value.success).length;
        
        console.log(`    ✅ Batch ${Math.floor(i / batchSize) + 1}: ${successful}/${batch.length} workspaces cleaned`);
      }
    } else {
      console.log('  ⚠️ Could not list workspaces, attempting general cleanup...');
    }
    
    // Force cleanup any remaining test containers
    console.log('🐳 Cleaning up test containers...');
    await executeCommand('docker ps -a --filter "name=test" --filter "name=functional" -q | xargs docker rm -f || true');
    
    // Clean up test volumes
    console.log('💾 Cleaning up test volumes...');
    await executeCommand('docker volume ls --filter "name=test" --filter "name=functional" -q | xargs docker volume rm || true');
    
  } catch (error) {
    console.log(`  ⚠️ Cleanup warning: ${error.message}`);
  }
  
  // Clean up temporary files
  console.log('📄 Cleaning up temporary files...');
  await executeCommand('rm -rf /tmp/test_* /tmp/functional_* || true');
  
  // Docker system cleanup (optional - removes unused resources)
  console.log('🧹 Docker system cleanup...');
  await executeCommand('docker system prune -f --volumes || true');
  
  // Generate cleanup summary
  console.log('📊 Generating cleanup summary...');
  
  const finalWorkspaceCheck = await executeCommand('devpod list --output json');
  let remainingWorkspaces = 0;
  
  if (finalWorkspaceCheck.success) {
    try {
      const workspaces = JSON.parse(finalWorkspaceCheck.output);
      remainingWorkspaces = workspaces.filter(ws => 
        ws.name.includes('test') || 
        ws.name.includes('functional') ||
        ws.name.includes('polyglot')
      ).length;
    } catch (error) {
      console.log('  ⚠️ Could not parse final workspace count');
    }
  }
  
  // Check remaining Docker resources
  const containerCheck = await executeCommand('docker ps -a --filter "name=test" --filter "name=functional" --format "table {{.Names}}" | wc -l');
  const containerCount = containerCheck.success ? parseInt(containerCheck.output) - 1 : 0; // Subtract header
  
  console.log('📋 Cleanup Summary:');
  console.log(`  Remaining test workspaces: ${remainingWorkspaces}`);
  console.log(`  Remaining test containers: ${Math.max(0, containerCount)}`);
  
  if (remainingWorkspaces > 0 || containerCount > 0) {
    console.log('  ⚠️ Some test resources may still exist - manual cleanup may be required');
    
    if (remainingWorkspaces > 0) {
      console.log('  💡 To manually clean workspaces: devpod list | grep test | xargs devpod delete --force');
    }
    
    if (containerCount > 0) {
      console.log('  💡 To manually clean containers: docker ps -a | grep test | awk \'{print $1}\' | xargs docker rm -f');
    }
  } else {
    console.log('  ✅ All test resources successfully cleaned up');
  }
  
  // Record teardown completion
  const teardownInfo = {
    timestamp: new Date().toISOString(),
    remainingWorkspaces,
    remainingContainers: Math.max(0, containerCount),
    cleanupSuccessful: remainingWorkspaces === 0 && containerCount === 0
  };
  
  console.log(`🕐 Teardown completed at: ${teardownInfo.timestamp}`);
  console.log(`✅ Cleanup status: ${teardownInfo.cleanupSuccessful ? 'SUCCESS' : 'PARTIAL'}`);
  
  console.log('✅ Global teardown completed');
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