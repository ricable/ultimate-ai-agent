#!/usr/bin/env node
/**
 * ReasoningBank Benchmark Orchestrator
 *
 * Runs comprehensive benchmarks comparing baseline vs ReasoningBank agents
 */
import { BaselineAgent } from './agents/baseline-agent.js';
import { ReasoningBankAgent } from './agents/reasoningbank-agent.js';
import { MetricsCollector } from './lib/metrics.js';
import { generateReport } from './lib/report-generator.js';
import { codingTasksScenario } from './scenarios/coding-tasks.js';
import { debuggingTasksScenario } from './scenarios/debugging-tasks.js';
import { apiDesignTasksScenario } from './scenarios/api-design-tasks.js';
import { problemSolvingTasksScenario } from './scenarios/problem-solving-tasks.js';
import { db } from '../agentic-flow/src/reasoningbank/index.js';
// Default configuration
const DEFAULT_CONFIG = {
    iterations: 3,
    parallelAgents: 1,
    scenarios: ['coding-tasks', 'debugging-tasks', 'api-design-tasks', 'problem-solving-tasks'],
    enableWarmStart: false,
    memorySize: 1000,
    outputFormats: ['json', 'markdown'],
    anthropicApiKey: process.env.ANTHROPIC_API_KEY
};
class BenchmarkOrchestrator {
    config;
    scenarios;
    results = [];
    constructor(config = {}) {
        this.config = { ...DEFAULT_CONFIG, ...config };
        this.scenarios = new Map([
            ['coding-tasks', codingTasksScenario],
            ['debugging-tasks', debuggingTasksScenario],
            ['api-design-tasks', apiDesignTasksScenario],
            ['problem-solving-tasks', problemSolvingTasksScenario]
        ]);
    }
    async run() {
        console.log('\n🧪 ReasoningBank Benchmark Suite');
        console.log('═'.repeat(60));
        console.log(`Scenarios: ${this.config.scenarios.join(', ')}`);
        console.log(`Iterations: ${this.config.iterations}`);
        console.log(`Warm Start: ${this.config.enableWarmStart ? 'Yes' : 'No'}\n`);
        const startTime = Date.now();
        // Initialize database
        console.log('📦 Initializing database...');
        await db.runMigrations();
        // Run each scenario
        for (const scenarioName of this.config.scenarios) {
            const scenario = this.scenarios.get(scenarioName);
            if (!scenario) {
                console.error(`❌ Unknown scenario: ${scenarioName}`);
                continue;
            }
            const result = await this.runScenario(scenario);
            this.results.push(result);
        }
        const executionTime = Date.now() - startTime;
        // Generate report
        const report = this.generateBenchmarkReport(executionTime);
        // Save results
        await this.saveResults(report);
        // Print summary
        this.printSummary(report);
        return report;
    }
    async runScenario(scenario) {
        console.log(`\n🎯 Running Scenario: ${scenario.name}`);
        console.log('─'.repeat(60));
        const metrics = new MetricsCollector();
        // Create agents
        const baseline = new BaselineAgent(this.config.anthropicApiKey);
        const reasoningbank = new ReasoningBankAgent(this.config.anthropicApiKey);
        // Reset agents
        await baseline.reset();
        await reasoningbank.reset();
        // Run iterations
        for (let iteration = 0; iteration < this.config.iterations; iteration++) {
            console.log(`\n  Iteration ${iteration + 1}/${this.config.iterations}`);
            let baselineSuccesses = 0;
            let reasoningbankSuccesses = 0;
            let baselineTokens = 0;
            let reasoningbankTokens = 0;
            // Run all tasks in this iteration
            for (let i = 0; i < scenario.tasks.length; i++) {
                const task = scenario.tasks[i];
                const progress = ((i + 1) / scenario.tasks.length * 100).toFixed(0);
                process.stdout.write(`\r    Task ${i + 1}/${scenario.tasks.length} (${progress}%)`);
                // Run baseline
                const baselineResult = await baseline.executeTask(task);
                metrics.addResult(baselineResult);
                if (baselineResult.success)
                    baselineSuccesses++;
                baselineTokens += baselineResult.tokens;
                // Run ReasoningBank
                const rbResult = await reasoningbank.executeTask(task);
                metrics.addResult(rbResult);
                if (rbResult.success)
                    reasoningbankSuccesses++;
                reasoningbankTokens += rbResult.tokens;
            }
            // Record learning point
            const totalTasks = scenario.tasks.length;
            const memoryStats = await reasoningbank.getMemoryStats();
            metrics.addLearningPoint({
                iteration: iteration + 1,
                baselineSuccess: baselineSuccesses / totalTasks,
                reasoningbankSuccess: reasoningbankSuccesses / totalTasks,
                baselineTokens: baselineTokens / totalTasks,
                reasoningbankTokens: reasoningbankTokens / totalTasks,
                memoriesAvailable: memoryStats.total
            });
            console.log(`\n    ├─ Baseline:      ${baselineSuccesses}/${totalTasks} success (${baselineTokens.toLocaleString()} tokens)`);
            console.log(`    └─ ReasoningBank: ${reasoningbankSuccesses}/${totalTasks} success (${reasoningbankTokens.toLocaleString()} tokens, ${memoryStats.total} memories)`);
        }
        const results = metrics.generateScenarioResults(scenario.name);
        // Print scenario summary
        console.log('\n  📊 Scenario Results:');
        console.log(`    Success Rate: ${(results.baseline.successRate * 100).toFixed(1)}% → ${(results.reasoningbank.successRate * 100).toFixed(1)}% (${results.improvement.successRateDelta})`);
        console.log(`    Token Efficiency: ${results.improvement.tokenEfficiency}`);
        console.log(`    Latency Overhead: ${results.improvement.latencyOverhead}`);
        if (results.improvement.learningVelocity) {
            console.log(`    Learning Velocity: ${results.improvement.learningVelocity.toFixed(1)}x faster`);
        }
        const insights = metrics.generateInsights(results);
        if (insights.length > 0) {
            console.log('\n  💡 Insights:');
            insights.forEach(insight => console.log(`    ${insight}`));
        }
        return results;
    }
    generateBenchmarkReport(executionTime) {
        // Calculate overall metrics
        const totalTasks = this.results.reduce((sum, r) => sum + r.baseline.totalTasks + r.reasoningbank.totalTasks, 0);
        const overallBaseline = {
            successRate: 0,
            avgTokens: 0,
            avgLatency: 0
        };
        const overallRB = {
            successRate: 0,
            avgTokens: 0,
            avgLatency: 0
        };
        let totalBaselineTasks = 0;
        let totalRBTasks = 0;
        for (const result of this.results) {
            overallBaseline.successRate += result.baseline.successRate * result.baseline.totalTasks;
            overallBaseline.avgTokens += result.baseline.avgTokens * result.baseline.totalTasks;
            overallBaseline.avgLatency += result.baseline.avgLatency * result.baseline.totalTasks;
            totalBaselineTasks += result.baseline.totalTasks;
            overallRB.successRate += result.reasoningbank.successRate * result.reasoningbank.totalTasks;
            overallRB.avgTokens += result.reasoningbank.avgTokens * result.reasoningbank.totalTasks;
            overallRB.avgLatency += result.reasoningbank.avgLatency * result.reasoningbank.totalTasks;
            totalRBTasks += result.reasoningbank.totalTasks;
        }
        overallBaseline.successRate /= totalBaselineTasks;
        overallBaseline.avgTokens /= totalBaselineTasks;
        overallBaseline.avgLatency /= totalBaselineTasks;
        overallRB.successRate /= totalRBTasks;
        overallRB.avgTokens /= totalRBTasks;
        overallRB.avgLatency /= totalRBTasks;
        const overallImprovement = {
            successRateDelta: `+${((overallRB.successRate - overallBaseline.successRate) * 100).toFixed(1)}%`,
            tokenEfficiency: `+${((overallBaseline.avgTokens - overallRB.avgTokens) / overallBaseline.avgTokens * 100).toFixed(1)}%`,
            latencyOverhead: `+${((overallRB.avgLatency - overallBaseline.avgLatency) / overallBaseline.avgLatency * 100).toFixed(1)}%`
        };
        // Generate recommendations
        const recommendations = this.generateRecommendations();
        return {
            summary: {
                totalScenarios: this.results.length,
                totalTasks,
                overallImprovement,
                executionTime
            },
            scenarios: this.results,
            recommendations,
            generatedAt: new Date().toISOString()
        };
    }
    generateRecommendations() {
        const recommendations = [];
        for (const result of this.results) {
            if (result.improvement.successRatePercent < 0) {
                recommendations.push(`⚠️  ${result.scenarioName}: Baseline outperformed ReasoningBank - review memory quality`);
            }
            if (result.improvement.tokenSavings < 0) {
                recommendations.push(`💡 ${result.scenarioName}: Token overhead detected - optimize memory retrieval (k parameter)`);
            }
            if (result.improvement.latencyDelta > 1000) {
                recommendations.push(`⚡ ${result.scenarioName}: High latency - consider caching or reducing consolidation frequency`);
            }
        }
        if (recommendations.length === 0) {
            recommendations.push('✅ All metrics look good! ReasoningBank is performing optimally.');
        }
        return recommendations;
    }
    async saveResults(report) {
        const fs = await import('fs/promises');
        const path = await import('path');
        // Save JSON
        if (this.config.outputFormats.includes('json')) {
            const jsonPath = path.join(process.cwd(), 'bench', 'results', `benchmark-${Date.now()}.json`);
            await fs.writeFile(jsonPath, JSON.stringify(report, null, 2));
            console.log(`\n📄 JSON report saved: ${jsonPath}`);
        }
        // Save Markdown
        if (this.config.outputFormats.includes('markdown')) {
            const mdPath = path.join(process.cwd(), 'bench', 'reports', `benchmark-${Date.now()}.md`);
            const markdown = generateReport(report);
            await fs.writeFile(mdPath, markdown);
            console.log(`📄 Markdown report saved: ${mdPath}`);
        }
    }
    printSummary(report) {
        console.log('\n\n' + '═'.repeat(60));
        console.log('📊 BENCHMARK SUMMARY');
        console.log('═'.repeat(60));
        console.log(`Total Scenarios: ${report.summary.totalScenarios}`);
        console.log(`Total Tasks: ${report.summary.totalTasks}`);
        console.log(`Execution Time: ${(report.summary.executionTime / 1000).toFixed(1)}s`);
        console.log('\nOverall Improvement:');
        console.log(`  Success Rate: ${report.summary.overallImprovement.successRateDelta}`);
        console.log(`  Token Efficiency: ${report.summary.overallImprovement.tokenEfficiency}`);
        console.log(`  Latency Overhead: ${report.summary.overallImprovement.latencyOverhead}`);
        if (report.recommendations.length > 0) {
            console.log('\n💡 Recommendations:');
            report.recommendations.forEach(rec => console.log(`  ${rec}`));
        }
        console.log('\n' + '═'.repeat(60) + '\n');
    }
}
// CLI Entry Point
if (import.meta.url === `file://${process.argv[1]}`) {
    const orchestrator = new BenchmarkOrchestrator();
    orchestrator.run().catch(error => {
        console.error('\n❌ Benchmark failed:', error);
        process.exit(1);
    });
}
export { BenchmarkOrchestrator };
