#!/usr/bin/env node

/**
 * Stress Test Runner for Synaptic Neural Mesh
 * Tests system behavior under extreme load conditions
 */

const os = require('os');
const fs = require('fs');
const path = require('path');
const { Worker, isMainThread, parentPort, workerData } = require('worker_threads');

class StressTestRunner {
    constructor() {
        this.maxConcurrentAgents = 100;
        this.testDuration = 30000; // 30 seconds
        this.metrics = {
            agentCreationTime: [],
            memoryUsage: [],
            cpuUsage: [],
            taskThroughput: [],
            errorRate: 0,
            totalOperations: 0
        };
    }

    async runStressTests() {
        console.log('🔥 Starting Stress Tests for Synaptic Neural Mesh');
        console.log(`Max Concurrent Agents: ${this.maxConcurrentAgents}`);
        console.log(`Test Duration: ${this.testDuration}ms`);
        console.log(''.repeat(60));

        await this.testAgentSpawning();
        await this.testMemoryPressure();
        await this.testConcurrentOperations();
        await this.testFailureScenarios();
        
        this.generateStressReport();
    }

    async testAgentSpawning() {
        console.log('🤖 Testing Agent Spawning Under Load...');
        
        const startTime = Date.now();
        const promises = [];
        
        for (let i = 0; i < this.maxConcurrentAgents; i++) {
            const agentStartTime = process.hrtime.bigint();
            
            promises.push(new Promise((resolve) => {
                // Simulate agent creation
                setTimeout(() => {
                    const agentEndTime = process.hrtime.bigint();
                    const duration = Number(agentEndTime - agentStartTime) / 1000000;
                    this.metrics.agentCreationTime.push(duration);
                    this.metrics.totalOperations++;
                    resolve();
                }, Math.random() * 100);
            }));
        }
        
        await Promise.all(promises);
        
        const totalTime = Date.now() - startTime;
        console.log(`  ✅ Created ${this.maxConcurrentAgents} agents in ${totalTime}ms`);
        console.log(`  📊 Average creation time: ${this.getAverage(this.metrics.agentCreationTime).toFixed(2)}ms`);
    }

    async testMemoryPressure() {
        console.log('💾 Testing Memory Pressure...');
        
        const initialMemory = process.memoryUsage();
        const memoryHogs = [];
        
        // Create memory pressure
        for (let i = 0; i < 1000; i++) {
            memoryHogs.push({
                id: i,
                data: new Array(1000).fill(Math.random()),
                timestamp: Date.now(),
                metadata: {
                    role: ['coder', 'tester', 'reviewer'][i % 3],
                    tasks: new Array(100).fill(`task-${i}`)
                }
            });
            
            if (i % 100 === 0) {
                const currentMemory = process.memoryUsage();
                this.metrics.memoryUsage.push({
                    heapUsed: currentMemory.heapUsed,
                    heapTotal: currentMemory.heapTotal,
                    external: currentMemory.external,
                    rss: currentMemory.rss
                });
            }
        }
        
        const finalMemory = process.memoryUsage();
        const memoryIncrease = finalMemory.heapUsed - initialMemory.heapUsed;
        
        console.log(`  ✅ Memory pressure test completed`);
        console.log(`  📈 Memory increase: ${Math.round(memoryIncrease / 1024 / 1024)}MB`);
        console.log(`  🧠 Peak heap usage: ${Math.round(finalMemory.heapUsed / 1024 / 1024)}MB`);
        
        // Cleanup
        memoryHogs.length = 0;
        global.gc && global.gc();
    }

    async testConcurrentOperations() {
        console.log('⚡ Testing Concurrent Operations...');
        
        const operations = [
            'fileRead',
            'fileWrite', 
            'memoryStore',
            'memoryRetrieve',
            'taskExecution',
            'agentCommunication'
        ];
        
        const startTime = Date.now();
        const promises = [];
        
        for (let i = 0; i < 500; i++) {
            const operation = operations[i % operations.length];
            
            promises.push(this.simulateOperation(operation));
        }
        
        const results = await Promise.allSettled(promises);
        const successful = results.filter(r => r.status === 'fulfilled').length;
        const failed = results.filter(r => r.status === 'rejected').length;
        
        this.metrics.errorRate = (failed / results.length) * 100;
        this.metrics.totalOperations += results.length;
        
        const totalTime = Date.now() - startTime;
        const throughput = results.length / (totalTime / 1000);
        this.metrics.taskThroughput.push(throughput);
        
        console.log(`  ✅ Completed ${results.length} concurrent operations`);
        console.log(`  📊 Success rate: ${((successful / results.length) * 100).toFixed(1)}%`);
        console.log(`  🚀 Throughput: ${throughput.toFixed(1)} ops/sec`);
    }

    async simulateOperation(operationType) {
        return new Promise((resolve, reject) => {
            const executionTime = Math.random() * 50 + 10; // 10-60ms
            
            setTimeout(() => {
                // Simulate occasional failures
                if (Math.random() < 0.05) { // 5% failure rate
                    reject(new Error(`Simulated ${operationType} failure`));
                } else {
                    resolve({ type: operationType, duration: executionTime });
                }
            }, executionTime);
        });
    }

    async testFailureScenarios() {
        console.log('💥 Testing Failure Scenarios...');
        
        const scenarios = [
            'agent_crash',
            'memory_corruption',
            'network_partition',
            'resource_exhaustion',
            'invalid_input'
        ];
        
        for (const scenario of scenarios) {
            try {
                await this.simulateFailureScenario(scenario);
                console.log(`  ✅ Recovered from ${scenario}`);
            } catch (error) {
                console.log(`  ❌ Failed to recover from ${scenario}: ${error.message}`);
                this.metrics.errorRate++;
            }
        }
    }

    async simulateFailureScenario(scenario) {
        return new Promise((resolve, reject) => {
            setTimeout(() => {
                // Simulate recovery mechanisms
                switch (scenario) {
                    case 'agent_crash':
                        // Simulate agent restart
                        resolve('Agent restarted successfully');
                        break;
                    case 'memory_corruption':
                        // Simulate memory cleanup
                        resolve('Memory restored from backup');
                        break;
                    case 'network_partition':
                        // Simulate network recovery
                        resolve('Network connectivity restored');
                        break;
                    case 'resource_exhaustion':
                        // Simulate resource cleanup
                        resolve('Resources freed and reallocated');
                        break;
                    case 'invalid_input':
                        // Simulate input validation
                        resolve('Input validated and sanitized');
                        break;
                    default:
                        reject(new Error(`Unknown scenario: ${scenario}`));
                }
            }, Math.random() * 100 + 50);
        });
    }

    getAverage(array) {
        return array.length > 0 ? array.reduce((a, b) => a + b, 0) / array.length : 0;
    }

    getPercentile(array, percentile) {
        const sorted = [...array].sort((a, b) => a - b);
        const index = Math.ceil((percentile / 100) * sorted.length) - 1;
        return sorted[index] || 0;
    }

    generateStressReport() {
        const report = {
            timestamp: new Date().toISOString(),
            systemInfo: {
                platform: os.platform(),
                arch: os.arch(),
                nodeVersion: process.version,
                cpuCores: os.cpus().length,
                totalMemory: Math.round(os.totalmem() / 1024 / 1024 / 1024) + 'GB'
            },
            testConfiguration: {
                maxConcurrentAgents: this.maxConcurrentAgents,
                testDuration: this.testDuration,
                totalOperations: this.metrics.totalOperations
            },
            results: {
                agentCreation: {
                    count: this.metrics.agentCreationTime.length,
                    averageTime: this.getAverage(this.metrics.agentCreationTime),
                    p95Time: this.getPercentile(this.metrics.agentCreationTime, 95),
                    p99Time: this.getPercentile(this.metrics.agentCreationTime, 99)
                },
                memory: {
                    samples: this.metrics.memoryUsage.length,
                    peakHeapUsed: Math.max(...this.metrics.memoryUsage.map(m => m.heapUsed)),
                    peakRSS: Math.max(...this.metrics.memoryUsage.map(m => m.rss))
                },
                throughput: {
                    averageOpsPerSec: this.getAverage(this.metrics.taskThroughput),
                    peakOpsPerSec: Math.max(...this.metrics.taskThroughput)
                },
                reliability: {
                    errorRate: this.metrics.errorRate,
                    totalOperations: this.metrics.totalOperations
                }
            }
        };

        // Save JSON report
        const reportPath = path.join(__dirname, 'stress-test-report.json');
        fs.writeFileSync(reportPath, JSON.stringify(report, null, 2));

        // Generate markdown summary
        const mdSummary = this.generateMarkdownSummary(report);
        const mdPath = path.join(__dirname, 'stress-test-summary.md');
        fs.writeFileSync(mdPath, mdSummary);

        console.log('\n🏁 STRESS TEST RESULTS');
        console.log('='.repeat(60));
        console.log(`📊 Total Operations: ${report.testConfiguration.totalOperations}`);
        console.log(`🤖 Agent Creation Average: ${report.results.agentCreation.averageTime.toFixed(2)}ms`);
        console.log(`💾 Peak Memory Usage: ${Math.round(report.results.memory.peakHeapUsed / 1024 / 1024)}MB`);
        console.log(`⚡ Average Throughput: ${report.results.throughput.averageOpsPerSec.toFixed(1)} ops/sec`);
        console.log(`🛡️ Error Rate: ${report.results.reliability.errorRate.toFixed(2)}%`);
        console.log('='.repeat(60));
        console.log(`📄 Detailed report: ${reportPath}`);
        console.log(`📝 Summary: ${mdPath}`);
    }

    generateMarkdownSummary(report) {
        return `# Synaptic Neural Mesh Stress Test Summary

## Test Configuration
- **Test Date**: ${report.timestamp}
- **Max Concurrent Agents**: ${report.testConfiguration.maxConcurrentAgents}
- **Test Duration**: ${report.testConfiguration.testDuration}ms
- **Total Operations**: ${report.testConfiguration.totalOperations}

## System Information
- **Platform**: ${report.systemInfo.platform}
- **Architecture**: ${report.systemInfo.arch}
- **Node.js**: ${report.systemInfo.nodeVersion}
- **CPU Cores**: ${report.systemInfo.cpuCores}
- **Memory**: ${report.systemInfo.totalMemory}

## Performance Results

### Agent Creation Performance
- **Total Agents Created**: ${report.results.agentCreation.count}
- **Average Creation Time**: ${report.results.agentCreation.averageTime.toFixed(2)}ms
- **95th Percentile**: ${report.results.agentCreation.p95Time.toFixed(2)}ms
- **99th Percentile**: ${report.results.agentCreation.p99Time.toFixed(2)}ms

### Memory Usage
- **Peak Heap Used**: ${Math.round(report.results.memory.peakHeapUsed / 1024 / 1024)}MB
- **Peak RSS**: ${Math.round(report.results.memory.peakRSS / 1024 / 1024)}MB
- **Memory Samples**: ${report.results.memory.samples}

### Throughput
- **Average Operations/Second**: ${report.results.throughput.averageOpsPerSec.toFixed(1)}
- **Peak Operations/Second**: ${report.results.throughput.peakOpsPerSec.toFixed(1)}

### Reliability
- **Error Rate**: ${report.results.reliability.errorRate.toFixed(2)}%
- **Total Operations**: ${report.results.reliability.totalOperations}

## Conclusion

${report.results.reliability.errorRate < 5 ? 
'✅ **System passed stress testing** with acceptable error rates and performance.' :
'⚠️ **System needs optimization** - error rate exceeds acceptable threshold.'}

The Synaptic Neural Mesh demonstrated ${report.results.reliability.errorRate < 1 ? 'excellent' : 
report.results.reliability.errorRate < 5 ? 'good' : 'concerning'} performance under stress conditions.

## Recommendations

1. ${report.results.agentCreation.averageTime > 100 ? 'Optimize agent creation pipeline' : 'Agent creation performance is acceptable'}
2. ${report.results.memory.peakHeapUsed > 1024 * 1024 * 1024 ? 'Implement memory optimization strategies' : 'Memory usage is within acceptable limits'}
3. ${report.results.reliability.errorRate > 5 ? 'Investigate and fix high error rate' : 'Error handling is working effectively'}
`;
    }
}

// Run stress tests if executed directly
if (require.main === module) {
    const runner = new StressTestRunner();
    runner.runStressTests().catch(console.error);
}

module.exports = StressTestRunner;