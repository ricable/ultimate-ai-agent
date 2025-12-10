#!/usr/bin/env node

/**
 * SAFLA MCP Integration Test Script
 * 
 * This script tests the complete MCP integration functionality,
 * including meta-cognitive features, goal management, and performance benchmarking.
 */

const { SAFLAMCPClient } = require('./mcp_integration.js');

async function runComprehensiveTests() {
  console.log('🚀 Starting SAFLA MCP Integration Tests\n');
  
  const client = new SAFLAMCPClient();
  const testResults = {
    total: 0,
    passed: 0,
    failed: 0,
    errors: []
  };

  async function runTest(testName, testFunction) {
    testResults.total++;
    console.log(`🧪 Running: ${testName}`);
    
    try {
      const result = await testFunction();
      if (result) {
        console.log(`✅ PASSED: ${testName}`);
        testResults.passed++;
      } else {
        console.log(`❌ FAILED: ${testName}`);
        testResults.failed++;
        testResults.errors.push(`${testName}: Test returned false`);
      }
    } catch (error) {
      console.log(`❌ ERROR: ${testName} - ${error.message}`);
      testResults.failed++;
      testResults.errors.push(`${testName}: ${error.message}`);
    }
    console.log('');
  }

  try {
    // Test 1: Connection
    await runTest('MCP Server Connection', async () => {
      await client.connect();
      return true;
    });

    // Test 2: System Validation
    await runTest('Installation Validation', async () => {
      const result = await client.validateInstallation();
      return result.validation.valid === true;
    });

    // Test 3: Configuration Summary
    await runTest('Configuration Summary', async () => {
      const config = await client.getConfigSummary();
      return config.memory && config.safety && config.general;
    });

    // Test 4: System Awareness
    await runTest('System Awareness', async () => {
      const awareness = await client.getSystemAwareness();
      return awareness.awareness_level !== undefined && 
             awareness.focus_areas && 
             awareness.system_metrics;
    });

    // Test 5: Introspective Analysis
    await runTest('Introspective Analysis', async () => {
      const analysis = await client.analyzeIntrospection('performance', 1);
      return analysis.introspection && analysis.depth;
    });

    // Test 6: Goal Creation
    await runTest('Goal Creation', async () => {
      const goal = await client.createGoal(
        'Test Goal',
        'A test goal for MCP integration testing',
        'medium',
        { test_metric: 100 }
      );
      return goal.id && goal.status === 'active';
    });

    // Test 7: Goal Listing
    await runTest('Goal Listing', async () => {
      const goals = await client.listGoals();
      return Array.isArray(goals);
    });

    // Test 8: Memory Benchmarking
    await runTest('Memory Benchmarking', async () => {
      const benchmark = await client.benchmarkMemory(10, ['sequential']);
      return benchmark.results && benchmark.benchmark_id;
    });

    // Test 9: Vector Operations Benchmarking
    await runTest('Vector Operations Benchmarking', async () => {
      const benchmark = await client.benchmarkVectorOperations(100, 256, ['similarity']);
      return benchmark.results && benchmark.benchmark_id;
    });

    // Test 10: MCP Connectivity Test
    await runTest('MCP Connectivity Test', async () => {
      const connectivity = await client.testConnectivity('basic');
      return connectivity.overall_status === 'passed';
    });

    // Test 11: System Information
    await runTest('System Information', async () => {
      const sysInfo = await client.getSystemInfo();
      return sysInfo !== null;
    });

    // Test 12: GPU Status Check
    await runTest('GPU Status Check', async () => {
      const gpuStatus = await client.checkGPUStatus();
      return gpuStatus !== null;
    });

  } catch (error) {
    console.error('❌ Critical Error during testing:', error);
    testResults.errors.push(`Critical Error: ${error.message}`);
  } finally {
    client.disconnect();
  }

  // Print Results Summary
  console.log('📊 Test Results Summary');
  console.log('========================');
  console.log(`Total Tests: ${testResults.total}`);
  console.log(`Passed: ${testResults.passed}`);
  console.log(`Failed: ${testResults.failed}`);
  console.log(`Success Rate: ${((testResults.passed / testResults.total) * 100).toFixed(1)}%`);
  
  if (testResults.errors.length > 0) {
    console.log('\n❌ Errors:');
    testResults.errors.forEach(error => console.log(`  - ${error}`));
  }

  if (testResults.passed === testResults.total) {
    console.log('\n🎉 All tests passed! MCP integration is fully functional.');
  } else {
    console.log('\n⚠️  Some tests failed. Review the errors above.');
  }

  return testResults;
}

async function demonstrateMetaCognitiveFeatures() {
  console.log('\n🧠 Demonstrating Meta-Cognitive Features\n');
  
  const client = new SAFLAMCPClient();
  
  try {
    await client.connect();
    
    // Get current awareness state
    console.log('📊 Current System Awareness:');
    const awareness = await client.getSystemAwareness();
    console.log(`  - Awareness Level: ${(awareness.awareness_level * 100).toFixed(1)}%`);
    console.log(`  - Focus Areas: ${awareness.focus_areas.join(', ')}`);
    console.log(`  - Performance Score: ${(awareness.system_metrics.performance_score * 100).toFixed(1)}%`);
    console.log(`  - Memory Efficiency: ${(awareness.system_metrics.memory_efficiency * 100).toFixed(1)}%`);
    console.log(`  - Confidence: ${(awareness.self_assessment.confidence * 100).toFixed(1)}%`);
    
    // Update awareness
    console.log('\n🔄 Updating System Awareness...');
    await client.updateAwareness(0.8, ['performance', 'testing', 'integration'], 'deep');
    
    // Get updated awareness
    const updatedAwareness = await client.getSystemAwareness();
    console.log(`  - New Awareness Level: ${(updatedAwareness.awareness_level * 100).toFixed(1)}%`);
    console.log(`  - New Focus Areas: ${updatedAwareness.focus_areas.join(', ')}`);
    
    // Perform introspective analysis
    console.log('\n🔍 Performing Introspective Analysis...');
    const analysis = await client.analyzeIntrospection('comprehensive', 1);
    console.log(`  - Response Efficiency: ${(analysis.introspection.performance_analysis.response_efficiency * 100).toFixed(1)}%`);
    console.log(`  - Resource Utilization: ${(analysis.introspection.performance_analysis.resource_utilization * 100).toFixed(1)}%`);
    console.log(`  - Error Rate: ${(analysis.introspection.performance_analysis.error_rate * 100).toFixed(1)}%`);
    
  } catch (error) {
    console.error('❌ Error demonstrating meta-cognitive features:', error);
  } finally {
    client.disconnect();
  }
}

async function demonstrateGoalManagement() {
  console.log('\n🎯 Demonstrating Goal Management\n');
  
  const client = new SAFLAMCPClient();
  
  try {
    await client.connect();
    
    // Create multiple goals
    console.log('📝 Creating test goals...');
    const goals = [];
    
    const goal1 = await client.createGoal(
      'Performance Optimization',
      'Improve system response times and efficiency',
      'high',
      { target_response_time: 50, target_efficiency: 0.95 }
    );
    goals.push(goal1);
    console.log(`  ✅ Created: ${goal1.description} (ID: ${goal1.id})`);
    
    const goal2 = await client.createGoal(
      'Memory Management',
      'Optimize memory usage and reduce fragmentation',
      'medium',
      { target_memory_usage: 0.8, fragmentation_threshold: 0.1 }
    );
    goals.push(goal2);
    console.log(`  ✅ Created: ${goal2.description} (ID: ${goal2.id})`);
    
    // List all goals
    console.log('\n📋 Current Goals:');
    const allGoals = await client.listGoals();
    allGoals.forEach(goal => {
      console.log(`  - ${goal.description} [${goal.priority}] - ${goal.status}`);
    });
    
    // Update a goal
    console.log('\n🔄 Updating goal progress...');
    await client.updateGoal(goal1.id, {
      progress: 0.3,
      notes: 'Initial optimization phase completed'
    });
    console.log(`  ✅ Updated goal: ${goal1.id}`);
    
    // Evaluate goal progress
    console.log('\n📈 Evaluating goal progress...');
    const evaluation = await client.evaluateGoalProgress();
    console.log(`  - Evaluation completed for all goals`);
    
  } catch (error) {
    console.error('❌ Error demonstrating goal management:', error);
  } finally {
    client.disconnect();
  }
}

// Main execution
async function main() {
  console.log('🔧 SAFLA MCP Integration Test Suite');
  console.log('====================================\n');
  
  // Run comprehensive tests
  const testResults = await runComprehensiveTests();
  
  // If basic tests pass, demonstrate advanced features
  if (testResults.passed >= testResults.total * 0.8) {
    await demonstrateMetaCognitiveFeatures();
    await demonstrateGoalManagement();
  }
  
  console.log('\n✨ MCP Integration testing completed!');
  
  // Exit with appropriate code
  process.exit(testResults.failed > 0 ? 1 : 0);
}

// Run if called directly
if (require.main === module) {
  main().catch(error => {
    console.error('❌ Fatal error:', error);
    process.exit(1);
  });
}

module.exports = {
  runComprehensiveTests,
  demonstrateMetaCognitiveFeatures,
  demonstrateGoalManagement
};