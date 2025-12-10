/**
 * Jest Test Sequencer for Polyglot DevPod Functional Tests
 * 
 * Controls the order of test execution based on dependencies and priorities
 */

const Sequencer = require('@jest/test-sequencer').default;

class FunctionalTestSequencer extends Sequencer {
  /**
   * Sort tests based on priority and dependencies
   */
  sort(tests) {
    // Define test priorities and dependencies
    const testPriority = {
      'devpod-swarm-tests.ts': { priority: 1, dependencies: [] },
      'environment-specific-tests.ts': { priority: 2, dependencies: ['devpod-swarm-tests.ts'] },
      'mcp-tool-matrix-tests.ts': { priority: 2, dependencies: ['devpod-swarm-tests.ts'] },
      'ai-integration-tests.ts': { priority: 2, dependencies: ['devpod-swarm-tests.ts'] },
      'agentic-environment-tests.ts': { priority: 3, dependencies: ['devpod-swarm-tests.ts', 'ai-integration-tests.ts'] },
      'performance-load-tests.ts': { priority: 4, dependencies: ['devpod-swarm-tests.ts', 'environment-specific-tests.ts'] },
      'test-runner.ts': { priority: 5, dependencies: ['devpod-swarm-tests.ts'] }
    };
    
    // Helper function to get test file name from path
    const getTestFileName = (test) => {
      const pathParts = test.path.split('/');
      return pathParts[pathParts.length - 1];
    };
    
    // Helper function to check if dependencies are satisfied
    const areDependenciesSatisfied = (testFile, sortedTests) => {
      const config = testPriority[testFile];
      if (!config || !config.dependencies.length) {
        return true;
      }
      
      return config.dependencies.every(dep => 
        sortedTests.some(sortedTest => getTestFileName(sortedTest) === dep)
      );
    };
    
    const sortedTests = [];
    const remainingTests = [...tests];
    
    // Sort by priority and dependencies
    while (remainingTests.length > 0) {
      let addedInThisIteration = false;
      
      for (let i = remainingTests.length - 1; i >= 0; i--) {
        const test = remainingTests[i];
        const testFile = getTestFileName(test);
        const config = testPriority[testFile];
        
        // If no config exists, add to end with low priority
        if (!config) {
          continue;
        }
        
        // Check if dependencies are satisfied
        if (areDependenciesSatisfied(testFile, sortedTests)) {
          sortedTests.push(test);
          remainingTests.splice(i, 1);
          addedInThisIteration = true;
        }
      }
      
      // If we couldn't add any tests in this iteration, break dependency cycle
      if (!addedInThisIteration) {
        // Add the highest priority remaining test
        if (remainingTests.length > 0) {
          let highestPriority = Number.MAX_SAFE_INTEGER;
          let highestPriorityIndex = 0;
          
          for (let i = 0; i < remainingTests.length; i++) {
            const testFile = getTestFileName(remainingTests[i]);
            const config = testPriority[testFile];
            if (config && config.priority < highestPriority) {
              highestPriority = config.priority;
              highestPriorityIndex = i;
            }
          }
          
          console.warn(`Breaking dependency cycle, adding: ${getTestFileName(remainingTests[highestPriorityIndex])}`);
          sortedTests.push(remainingTests[highestPriorityIndex]);
          remainingTests.splice(highestPriorityIndex, 1);
        }
      }
    }
    
    // Add any remaining tests without config
    sortedTests.push(...remainingTests);
    
    // Log the execution order
    console.log('📅 Test execution order:');
    sortedTests.forEach((test, index) => {
      const testFile = getTestFileName(test);
      const config = testPriority[testFile];
      const priority = config ? config.priority : 'unknown';
      console.log(`  ${index + 1}. ${testFile} (priority: ${priority})`);
    });
    
    return sortedTests;
  }
  
  /**
   * Determine if tests should run in sequence (not in parallel)
   * For functional tests, we want sequential execution to avoid resource conflicts
   */
  shard(tests, { shardIndex, shardCount }) {
    // For functional tests, we'll run everything in sequence
    // But we can still use sharding for parallel test runs if needed
    
    const shardSize = Math.ceil(tests.length / shardCount);
    const startIndex = shardIndex * shardSize;
    const endIndex = Math.min(startIndex + shardSize, tests.length);
    
    return tests.slice(startIndex, endIndex);
  }
}

module.exports = FunctionalTestSequencer;