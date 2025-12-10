/**
 * Comprehensive Performance Metrics Test Suite
 * Tests and validates the Hive Mind performance metrics system end-to-end
 */

import { performance } from 'perf_hooks';
import { EventEmitter } from 'events';
import { CoordinationMetricsCollector } from '../src/coordination/metrics.js';

/**
 * Mock logger for testing
 */
class MockLogger {
  info(msg, data) { console.log(`[INFO] ${msg}`, data || ''); }
  debug(msg, data) { console.log(`[DEBUG] ${msg}`, data || ''); }
  warn(msg, data) { console.log(`[WARN] ${msg}`, data || ''); }
  error(msg, data) { console.log(`[ERROR] ${msg}`, data || ''); }
}

/**
 * Mock event bus for testing
 */
class MockEventBus extends EventEmitter {
  emit(event, data) {
    console.log(`📡 Event: ${event}`, data ? JSON.stringify(data, null, 2) : '');
    return super.emit(event, data);
  }
}

/**
 * Performance Metrics Test Suite
 */
export class PerformanceMetricsTestSuite {
  constructor() {
    this.logger = new MockLogger();
    this.eventBus = new MockEventBus();
    this.metricsCollector = new CoordinationMetricsCollector(this.logger, this.eventBus, 5000);
    this.testResults = [];
    this.swarmId = `test-swarm-${Date.now()}`;
  }

  /**
   * Run the complete test suite
   */
  async runCompleteTestSuite() {
    console.log('🔬 Starting Performance Metrics Test Suite...\n');
    console.log('=' .repeat(80));
    
    try {
      // Start metrics collection
      this.metricsCollector.start();
      
      const tests = [
        this.testBasicMetricsCollection,
        this.testTaskLifecycleMetrics,
        this.testAgentMetrics,
        this.testResourceMetrics,
        this.testCoordinationMetrics,
        this.testPerformanceCalculations,
        this.testRealisticSwarmScenario,
        this.testHighLoadScenario,
        this.testMetricsAggregation
      ];

      for (const test of tests) {
        console.log(`\n🧪 Running: ${test.name}...`);
        try {
          await test.call(this);
          console.log(`✅ ${test.name} PASSED`);
        } catch (error) {
          console.error(`❌ ${test.name} FAILED:`, error.message);
          this.testResults.push({
            test: test.name,
            status: 'FAILED',
            error: error.message
          });
        }
      }

      this.generateFinalReport();
      
    } finally {
      this.metricsCollector.stop();
    }
  }

  /**
   * Test 1: Basic metrics collection functionality
   */
  async testBasicMetricsCollection() {
    console.log('  📊 Testing basic metrics collection...');
    
    // Record some basic metrics
    this.metricsCollector.recordMetric('test.metric', 100, { type: 'basic' });
    this.metricsCollector.recordMetric('test.metric', 200, { type: 'basic' });
    this.metricsCollector.recordMetric('test.metric', 150, { type: 'advanced' });
    
    // Get metrics
    const metrics = this.metricsCollector.getCurrentMetrics();
    
    if (!metrics || !metrics.timestamp) {
      throw new Error('Failed to collect basic metrics');
    }
    
    console.log('    ✓ Basic metrics collection working');
    
    this.testResults.push({
      test: 'Basic Metrics Collection',
      status: 'PASSED',
      metricsCount: 3,
      timestamp: metrics.timestamp
    });
  }

  /**
   * Test 2: Task lifecycle metrics
   */
  async testTaskLifecycleMetrics() {
    console.log('  🔄 Testing task lifecycle metrics...');
    
    const taskIds = ['task-1', 'task-2', 'task-3', 'task-4', 'task-5'];
    
    // Simulate task lifecycle
    for (const taskId of taskIds) {
      // Task created
      this.eventBus.emit('system:task:created', { taskId, priority: 'high', type: 'coding' });
      await this.delay(50);
      
      // Task started
      this.eventBus.emit('system:task:started', { taskId });
      await this.delay(100);
      
      // Task completed (80% success rate)
      if (Math.random() > 0.2) {
        this.eventBus.emit('system:task:completed', { taskId });
      } else {
        this.eventBus.emit('system:task:failed', { taskId, reason: 'timeout' });
      }
      
      await this.delay(25);
    }
    
    // Wait for metrics to be processed
    await this.delay(200);
    
    const metrics = this.metricsCollector.getCurrentMetrics();
    
    if (metrics.taskMetrics.totalTasks < 5) {
      throw new Error(`Expected at least 5 tasks, got ${metrics.taskMetrics.totalTasks}`);
    }
    
    console.log(`    ✓ Task metrics: ${metrics.taskMetrics.totalTasks} total, ${metrics.taskMetrics.completedTasks} completed`);
    
    this.testResults.push({
      test: 'Task Lifecycle Metrics',
      status: 'PASSED',
      totalTasks: metrics.taskMetrics.totalTasks,
      completedTasks: metrics.taskMetrics.completedTasks,
      failedTasks: metrics.taskMetrics.failedTasks,
      successRate: ((metrics.taskMetrics.completedTasks / metrics.taskMetrics.totalTasks) * 100).toFixed(2) + '%'
    });
  }

  /**
   * Test 3: Agent metrics
   */
  async testAgentMetrics() {
    console.log('  🤖 Testing agent metrics...');
    
    const agentTypes = ['coordinator', 'coder', 'analyst', 'tester', 'researcher'];
    const agents = [];
    
    // Spawn agents
    for (let i = 0; i < agentTypes.length; i++) {
      const agentId = `agent-${i}`;
      const agentType = agentTypes[i % agentTypes.length];
      
      agents.push({ id: agentId, type: agentType });
      
      this.eventBus.emit('system:agent:spawned', { agentId, type: agentType });
      await this.delay(50);
    }
    
    // Simulate agent activity
    for (const agent of agents) {
      if (Math.random() > 0.3) {
        this.eventBus.emit('system:agent:active', { agentId: agent.id });
      } else {
        this.eventBus.emit('system:agent:idle', { agentId: agent.id });
      }
      await this.delay(25);
    }
    
    await this.delay(200);
    
    const metrics = this.metricsCollector.getCurrentMetrics();
    
    if (metrics.agentMetrics.totalAgents < 5) {
      throw new Error(`Expected at least 5 agents, got ${metrics.agentMetrics.totalAgents}`);
    }
    
    console.log(`    ✓ Agent metrics: ${metrics.agentMetrics.totalAgents} total, utilization: ${metrics.agentMetrics.agentUtilization.toFixed(1)}%`);
    
    this.testResults.push({
      test: 'Agent Metrics',
      status: 'PASSED',
      totalAgents: metrics.agentMetrics.totalAgents,
      activeAgents: metrics.agentMetrics.activeAgents,
      utilization: metrics.agentMetrics.agentUtilization.toFixed(2) + '%'
    });
  }

  /**
   * Test 4: Resource metrics
   */
  async testResourceMetrics() {
    console.log('  🔒 Testing resource metrics...');
    
    const resources = ['db-conn-1', 'db-conn-2', 'file-lock-1', 'memory-pool-1', 'cpu-core-1'];
    
    // Acquire resources
    for (const resourceId of resources) {
      this.eventBus.emit('system:resource:acquired', { resourceId });
      await this.delay(50);
    }
    
    // Release some resources
    for (let i = 0; i < 3; i++) {
      this.eventBus.emit('system:resource:released', { resourceId: resources[i] });
      await this.delay(75);
    }
    
    // Simulate deadlock
    this.eventBus.emit('system:deadlock:detected', { resources: ['db-conn-1', 'file-lock-1'] });
    
    await this.delay(200);
    
    const metrics = this.metricsCollector.getCurrentMetrics();
    
    if (metrics.resourceMetrics.totalResources < 2) {
      throw new Error(`Expected resource metrics, got ${metrics.resourceMetrics.totalResources}`);
    }
    
    console.log(`    ✓ Resource metrics: ${metrics.resourceMetrics.totalResources} total, ${metrics.resourceMetrics.lockedResources} locked, ${metrics.resourceMetrics.deadlockCount} deadlocks`);
    
    this.testResults.push({
      test: 'Resource Metrics',
      status: 'PASSED',
      totalResources: metrics.resourceMetrics.totalResources,
      lockedResources: metrics.resourceMetrics.lockedResources,
      deadlockCount: metrics.resourceMetrics.deadlockCount,
      utilization: metrics.resourceMetrics.resourceUtilization.toFixed(2) + '%'
    });
  }

  /**
   * Test 5: Coordination metrics
   */
  async testCoordinationMetrics() {
    console.log('  📡 Testing coordination metrics...');
    
    // Simulate message passing
    for (let i = 0; i < 20; i++) {
      const messageId = `msg-${i}`;
      this.eventBus.emit('system:message:sent', { 
        message: { id: messageId, from: 'agent-1', to: 'agent-2', content: `Message ${i}` }
      });
      
      await this.delay(25);
      
      this.eventBus.emit('system:message:received', { 
        message: { id: messageId, from: 'agent-1', to: 'agent-2', content: `Message ${i}` }
      });
      
      await this.delay(10);
    }
    
    // Simulate conflicts and resolutions
    for (let i = 0; i < 5; i++) {
      this.eventBus.emit('conflict:resource', { resourceId: `resource-${i}` });
      await this.delay(50);
      this.eventBus.emit('conflict:resolved', { resourceId: `resource-${i}` });
      await this.delay(25);
    }
    
    // Simulate work stealing
    for (let i = 0; i < 3; i++) {
      this.eventBus.emit('workstealing:request', { from: 'agent-1', to: 'agent-2' });
      await this.delay(30);
    }
    
    await this.delay(200);
    
    const metrics = this.metricsCollector.getCurrentMetrics();
    
    if (metrics.coordinationMetrics.messagesSent < 20) {
      throw new Error(`Expected 20 messages sent, got ${metrics.coordinationMetrics.messagesSent}`);
    }
    
    console.log(`    ✓ Coordination metrics: ${metrics.coordinationMetrics.messagesSent} sent, ${metrics.coordinationMetrics.messagesReceived} received, ${metrics.coordinationMetrics.conflictsResolved} conflicts resolved`);
    
    this.testResults.push({
      test: 'Coordination Metrics',
      status: 'PASSED',
      messagesSent: metrics.coordinationMetrics.messagesSent,
      messagesReceived: metrics.coordinationMetrics.messagesReceived,
      conflictsDetected: metrics.coordinationMetrics.conflictsDetected,
      conflictsResolved: metrics.coordinationMetrics.conflictsResolved,
      workStealingEvents: metrics.coordinationMetrics.workStealingEvents
    });
  }

  /**
   * Test 6: Performance calculations
   */
  async testPerformanceCalculations() {
    console.log('  📈 Testing performance calculations...');
    
    // Generate some errors
    for (let i = 0; i < 3; i++) {
      this.eventBus.emit('system:error', { type: 'timeout', message: `Error ${i}` });
      await this.delay(100);
    }
    
    await this.delay(200);
    
    const metrics = this.metricsCollector.getCurrentMetrics();
    
    // Verify performance metrics are calculated
    if (typeof metrics.performanceMetrics.memoryUsage !== 'number') {
      throw new Error('Memory usage calculation failed');
    }
    
    if (typeof metrics.performanceMetrics.errorRate !== 'number') {
      throw new Error('Error rate calculation failed');
    }
    
    console.log(`    ✓ Performance calculations: Memory: ${metrics.performanceMetrics.memoryUsage.toFixed(2)}MB, Error rate: ${metrics.performanceMetrics.errorRate}`);
    
    this.testResults.push({
      test: 'Performance Calculations',
      status: 'PASSED',
      memoryUsage: metrics.performanceMetrics.memoryUsage.toFixed(2) + 'MB',
      cpuUsage: metrics.performanceMetrics.cpuUsage.toFixed(2) + '%',
      errorRate: metrics.performanceMetrics.errorRate
    });
  }

  /**
   * Test 7: Realistic swarm scenario
   */
  async testRealisticSwarmScenario() {
    console.log('  🐝 Testing realistic swarm scenario...');
    
    const startTime = performance.now();
    
    // Simulate a realistic development task scenario
    const scenario = {
      name: 'Build REST API with Authentication',
      agents: [
        { type: 'coordinator', count: 1 },
        { type: 'coder', count: 3 },
        { type: 'tester', count: 2 },
        { type: 'analyst', count: 1 }
      ],
      tasks: [
        'Design API architecture',
        'Implement user authentication',
        'Create user CRUD operations',
        'Build JWT token system',
        'Write unit tests',
        'Perform integration testing',
        'Analyze performance metrics',
        'Optimize database queries'
      ]
    };
    
    // Spawn agents
    let agentCount = 0;
    for (const agentType of scenario.agents) {
      for (let i = 0; i < agentType.count; i++) {
        const agentId = `scenario-agent-${agentCount++}`;
        this.eventBus.emit('system:agent:spawned', { agentId, type: agentType.type });
        await this.delay(100);
      }
    }
    
    // Create and execute tasks
    for (let i = 0; i < scenario.tasks.length; i++) {
      const taskId = `scenario-task-${i}`;
      
      // Create task
      this.eventBus.emit('system:task:created', { 
        taskId, 
        description: scenario.tasks[i],
        priority: i < 2 ? 'high' : 'medium',
        type: 'development'
      });
      await this.delay(50);
      
      // Start task
      this.eventBus.emit('system:task:started', { taskId });
      
      // Simulate work with some coordination
      if (i % 3 === 0) {
        // Simulate resource contention
        this.eventBus.emit('system:resource:acquired', { resourceId: `shared-resource-${i}` });
        await this.delay(200);
        this.eventBus.emit('system:resource:released', { resourceId: `shared-resource-${i}` });
      }
      
      if (i % 4 === 0) {
        // Simulate agent communication
        this.eventBus.emit('system:message:sent', { 
          message: { id: `coord-msg-${i}`, from: 'coordinator', to: `agent-${i}` }
        });
        await this.delay(25);
        this.eventBus.emit('system:message:received', { 
          message: { id: `coord-msg-${i}`, from: 'coordinator', to: `agent-${i}` }
        });
      }
      
      // Complete task (90% success rate)
      await this.delay(300);
      if (Math.random() > 0.1) {
        this.eventBus.emit('system:task:completed', { taskId });
      } else {
        this.eventBus.emit('system:task:failed', { taskId, reason: 'complexity' });
        // Retry failed task
        await this.delay(100);
        this.eventBus.emit('system:task:created', { 
          taskId: `${taskId}-retry`, 
          description: `${scenario.tasks[i]} (Retry)`,
          priority: 'high',
          type: 'development'
        });
        this.eventBus.emit('system:task:started', { taskId: `${taskId}-retry` });
        await this.delay(200);
        this.eventBus.emit('system:task:completed', { taskId: `${taskId}-retry` });
      }
    }
    
    const executionTime = performance.now() - startTime;
    await this.delay(500); // Let metrics stabilize
    
    const metrics = this.metricsCollector.getCurrentMetrics();
    
    console.log(`    ✓ Realistic scenario completed in ${executionTime.toFixed(2)}ms`);
    console.log(`      Tasks: ${metrics.taskMetrics.totalTasks} total, ${metrics.taskMetrics.completedTasks} completed`);
    console.log(`      Agents: ${metrics.agentMetrics.totalAgents} total, utilization: ${metrics.agentMetrics.agentUtilization.toFixed(1)}%`);
    
    this.testResults.push({
      test: 'Realistic Swarm Scenario',
      status: 'PASSED',
      executionTime: executionTime.toFixed(2) + 'ms',
      totalTasks: metrics.taskMetrics.totalTasks,
      completedTasks: metrics.taskMetrics.completedTasks,
      totalAgents: metrics.agentMetrics.totalAgents,
      successRate: ((metrics.taskMetrics.completedTasks / metrics.taskMetrics.totalTasks) * 100).toFixed(2) + '%',
      agentUtilization: metrics.agentMetrics.agentUtilization.toFixed(2) + '%'
    });
  }

  /**
   * Test 8: High load scenario
   */
  async testHighLoadScenario() {
    console.log('  ⚡ Testing high load scenario...');
    
    const startTime = performance.now();
    
    // Simulate high load: 50 agents, 200 tasks, heavy coordination
    const agentPromises = [];
    const taskPromises = [];
    
    // Spawn many agents rapidly
    for (let i = 0; i < 50; i++) {
      agentPromises.push(
        this.simulateAgentSpawn(`load-agent-${i}`, ['coordinator', 'coder', 'analyst', 'tester'][i % 4])
      );
    }
    
    // Create many tasks rapidly
    for (let i = 0; i < 200; i++) {
      taskPromises.push(
        this.simulateTaskExecution(`load-task-${i}`, ['high', 'medium', 'low'][i % 3])
      );
    }
    
    // Wait for all operations
    await Promise.all([...agentPromises, ...taskPromises]);
    
    const loadTime = performance.now() - startTime;
    await this.delay(1000); // Let metrics stabilize
    
    const metrics = this.metricsCollector.getCurrentMetrics();
    
    console.log(`    ✓ High load scenario completed in ${loadTime.toFixed(2)}ms`);
    console.log(`      Peak load: ${metrics.taskMetrics.totalTasks} tasks, ${metrics.agentMetrics.totalAgents} agents`);
    
    this.testResults.push({
      test: 'High Load Scenario',
      status: 'PASSED',
      loadTime: loadTime.toFixed(2) + 'ms',
      peakTasks: metrics.taskMetrics.totalTasks,
      peakAgents: metrics.agentMetrics.totalAgents,
      taskThroughput: (metrics.taskMetrics.totalTasks / (loadTime / 1000)).toFixed(2) + ' tasks/sec',
      agentThroughput: (metrics.agentMetrics.totalAgents / (loadTime / 1000)).toFixed(2) + ' agents/sec'
    });
  }

  /**
   * Test 9: Metrics aggregation and history
   */
  async testMetricsAggregation() {
    console.log('  📊 Testing metrics aggregation and history...');
    
    // Get metric history
    const history = this.metricsCollector.getMetricHistory('task.completed');
    const topMetrics = this.metricsCollector.getTopMetrics(5);
    
    console.log(`    ✓ History: ${history.length} samples, Top metrics: ${topMetrics.length}`);
    
    // Test clearing metrics
    const beforeClear = this.metricsCollector.getCurrentMetrics();
    this.metricsCollector.clearMetrics();
    const afterClear = this.metricsCollector.getCurrentMetrics();
    
    if (afterClear.taskMetrics.totalTasks !== 0) {
      throw new Error('Metrics not properly cleared');
    }
    
    console.log(`    ✓ Metrics cleared successfully`);
    
    this.testResults.push({
      test: 'Metrics Aggregation',
      status: 'PASSED',
      historyCount: history.length,
      topMetricsCount: topMetrics.length,
      beforeClearTasks: beforeClear.taskMetrics.totalTasks,
      afterClearTasks: afterClear.taskMetrics.totalTasks
    });
  }

  /**
   * Helper method to simulate agent spawning
   */
  async simulateAgentSpawn(agentId, type) {
    this.eventBus.emit('system:agent:spawned', { agentId, type });
    await this.delay(Math.random() * 10 + 5);
    
    // Randomly activate or idle
    if (Math.random() > 0.3) {
      this.eventBus.emit('system:agent:active', { agentId });
    } else {
      this.eventBus.emit('system:agent:idle', { agentId });
    }
  }

  /**
   * Helper method to simulate task execution
   */
  async simulateTaskExecution(taskId, priority) {
    // Create task
    this.eventBus.emit('system:task:created', { taskId, priority, type: 'load-test' });
    await this.delay(Math.random() * 20 + 10);
    
    // Start task
    this.eventBus.emit('system:task:started', { taskId });
    await this.delay(Math.random() * 100 + 50);
    
    // Complete or fail task
    if (Math.random() > 0.15) {
      this.eventBus.emit('system:task:completed', { taskId });
    } else {
      this.eventBus.emit('system:task:failed', { taskId, reason: 'load-test-failure' });
    }
  }

  /**
   * Helper delay method
   */
  delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  /**
   * Generate final test report
   */
  generateFinalReport() {
    console.log('\n' + '='.repeat(80));
    console.log('📊 PERFORMANCE METRICS TEST RESULTS');
    console.log('='.repeat(80));
    
    const passedTests = this.testResults.filter(r => r.status === 'PASSED').length;
    const totalTests = this.testResults.length;
    
    console.log(`\n📈 Overall Results: ${passedTests}/${totalTests} tests passed`);
    
    // Display detailed results
    this.testResults.forEach((result, index) => {
      console.log(`\n${index + 1}. ${result.test}`);
      console.log('-'.repeat(50));
      
      Object.entries(result).forEach(([key, value]) => {
        if (key !== 'test' && key !== 'status') {
          console.log(`   ${key.charAt(0).toUpperCase() + key.slice(1)}: ${value}`);
        }
      });
      
      const statusIcon = result.status === 'PASSED' ? '✅' : '❌';
      console.log(`   Status: ${statusIcon} ${result.status}`);
    });
    
    // Final metrics snapshot
    const finalMetrics = this.metricsCollector.getCurrentMetrics();
    
    console.log('\n📊 FINAL METRICS SNAPSHOT');
    console.log('-'.repeat(50));
    console.log(`Tasks: ${finalMetrics.taskMetrics.totalTasks} total, ${finalMetrics.taskMetrics.completedTasks} completed`);
    console.log(`Agents: ${finalMetrics.agentMetrics.totalAgents} total, ${finalMetrics.agentMetrics.agentUtilization.toFixed(1)}% utilization`);
    console.log(`Resources: ${finalMetrics.resourceMetrics.totalResources} total, ${finalMetrics.resourceMetrics.resourceUtilization.toFixed(1)}% utilization`);
    console.log(`Messages: ${finalMetrics.coordinationMetrics.messagesSent} sent, ${finalMetrics.coordinationMetrics.messagesReceived} received`);
    console.log(`Performance: ${finalMetrics.performanceMetrics.memoryUsage.toFixed(2)}MB memory, ${finalMetrics.performanceMetrics.errorRate} errors`);
    
    console.log('\n🎉 Performance Metrics Test Suite Complete!');
    
    if (passedTests === totalTests) {
      console.log('✅ All performance metrics features are working correctly!');
    } else {
      console.log('⚠️  Some metrics features may need attention.');
    }
    
    return {
      totalTests,
      passedTests,
      successRate: (passedTests / totalTests) * 100,
      finalMetrics
    };
  }
}

// Export for use in other tests
export default PerformanceMetricsTestSuite;

// Run if executed directly
if (import.meta.url === `file://${process.argv[1]}`) {
  const testSuite = new PerformanceMetricsTestSuite();
  testSuite.runCompleteTestSuite().catch(console.error);
}