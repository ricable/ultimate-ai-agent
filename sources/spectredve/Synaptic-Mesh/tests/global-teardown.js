/**
 * Global teardown for Synaptic Neural Mesh test suite
 * Cleans up resources and generates final reports
 */

import { writeFileSync, rmSync, existsSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

export default async function globalTeardown() {
  console.log('🧹 Cleaning up global test environment...');
  
  try {
    // Generate final test metrics report
    if (global.testMetrics) {
      const summary = global.testMetrics.getSummary();
      
      const finalReport = {
        executionSummary: summary,
        environment: {
          nodeVersion: process.version,
          platform: process.platform,
          architecture: process.arch,
          testEnvironment: process.env.NODE_ENV
        },
        testExecution: {
          startTime: global.testMetrics.startTime,
          endTime: Date.now(),
          totalDuration: Date.now() - global.testMetrics.startTime,
          avgTestDuration: summary.avgTestDuration
        },
        serviceUsage: global.testServices ? {
          neuralNetworks: global.testServices.neural.networks.size,
          dagNodes: global.testServices.dag.nodes.size,
          swarmAgents: global.testServices.swarm.agents.size,
          memoryEntries: global.testServices.memory.store.size
        } : {},
        performance: {
          passRate: summary.passRate,
          executionEfficiency: summary.totalTests > 0 ? 
            (summary.totalExecutionTime / summary.totalTests) : 0,
          resourceUtilization: process.memoryUsage()
        },
        recommendations: generateRecommendations(summary)
      };
      
      const reportPath = join(__dirname, 'reports', `final-test-report-${Date.now()}.json`);
      writeFileSync(reportPath, JSON.stringify(finalReport, null, 2));
      console.log(`   ✓ Generated final test report: ${reportPath}`);
      
      // Print summary to console
      console.log('\n📊 Test Execution Summary:');
      console.log(`   Total Tests: ${summary.totalTests}`);
      console.log(`   Passed: ${summary.passed} (${summary.passRate.toFixed(1)}%)`);
      console.log(`   Failed: ${summary.failed}`);
      console.log(`   Skipped: ${summary.skipped}`);
      console.log(`   Execution Time: ${(summary.totalExecutionTime / 1000).toFixed(2)}s`);
      console.log(`   Avg Test Duration: ${summary.avgTestDuration.toFixed(2)}ms`);
    }
    
    // Clean up mock services
    if (global.testServices) {
      global.testServices.neural.networks.clear();
      global.testServices.dag.nodes.clear();
      global.testServices.swarm.agents.clear();
      global.testServices.memory.store.clear();
      console.log('   ✓ Cleaned up mock services');
    }
    
    // Clean up temporary files
    const tempDir = join(__dirname, 'temp');
    if (existsSync(tempDir)) {
      try {
        rmSync(tempDir, { recursive: true, force: true });
        console.log('   ✓ Cleaned up temporary files');
      } catch (error) {
        console.warn(`   ⚠️ Failed to clean temp directory: ${error.message}`);
      }
    }
    
    // Generate coverage summary if available
    const coverageDir = join(__dirname, 'coverage');
    if (existsSync(coverageDir)) {
      try {
        const coverageSummaryPath = join(coverageDir, 'coverage-summary.json');
        if (existsSync(coverageSummaryPath)) {
          console.log('   ✓ Coverage reports available in coverage/');
        }
      } catch (error) {
        console.warn(`   ⚠️ Coverage summary error: ${error.message}`);
      }
    }
    
    // Memory cleanup
    if (global.mockWasmModule) {
      delete global.mockWasmModule;
    }
    
    if (global.mockSystemMetrics) {
      delete global.mockSystemMetrics;
    }
    
    if (global.testUtils) {
      delete global.testUtils;
    }
    
    console.log('   ✓ Cleaned up global objects');
    
    // Final memory check
    const memoryUsage = process.memoryUsage();
    const memoryMB = Math.round(memoryUsage.heapUsed / 1024 / 1024);
    
    if (memoryMB > 100) {
      console.warn(`   ⚠️ High memory usage detected: ${memoryMB}MB`);
    } else {
      console.log(`   ✓ Memory usage: ${memoryMB}MB`);
    }
    
    console.log('✅ Global test environment cleanup complete');
    
  } catch (error) {
    console.error('❌ Error during global teardown:', error.message);
    throw error;
  }
}

function generateRecommendations(summary) {
  const recommendations = [];
  
  // Performance recommendations
  if (summary.avgTestDuration > 1000) {
    recommendations.push({
      type: 'performance',
      severity: 'warning',
      message: 'Average test duration is high (>1s). Consider optimizing slow tests.',
      suggestion: 'Review test timeouts and mock heavy operations'
    });
  }
  
  // Pass rate recommendations
  if (summary.passRate < 95) {
    recommendations.push({
      type: 'quality',
      severity: 'error',
      message: `Pass rate is below 95% (${summary.passRate.toFixed(1)}%)`,
      suggestion: 'Investigate and fix failing tests before proceeding'
    });
  }
  
  // Test count recommendations
  if (summary.totalTests < 50) {
    recommendations.push({
      type: 'coverage',
      severity: 'info',
      message: 'Low test count detected. Consider adding more comprehensive tests.',
      suggestion: 'Add edge case tests and integration scenarios'
    });
  }
  
  // Success recommendations
  if (summary.passRate === 100 && summary.totalTests > 100) {
    recommendations.push({
      type: 'success',
      severity: 'info',
      message: 'Excellent test coverage and pass rate!',
      suggestion: 'Maintain current testing practices and consider adding performance tests'
    });
  }
  
  return recommendations;
}