"use strict";
/**
 * Basic RAN Optimization Example
 *
 * This example demonstrates the core functionality of the Ericsson RAN
 * Optimization SDK with a simple optimization scenario.
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.basicRANMetrics = exports.demonstrateMemoryCoordination = exports.demonstrateSkillDiscovery = exports.runBasicOptimization = void 0;
const ran_optimization_sdk_1 = require("../src/sdk/ran-optimization-sdk");
/**
 * Basic RAN metrics for optimization
 */
const basicRANMetrics = {
    energy_efficiency: 0.75,
    mobility_performance: 0.80,
    coverage_quality: 0.85,
    capacity_utilization: 0.70,
    user_experience: 0.78,
    // Additional scenario-specific metrics
    time_of_day: 'peak_hour',
    traffic_load: 'high',
    weather_conditions: 'clear',
    event_type: 'normal_operation'
};
exports.basicRANMetrics = basicRANMetrics;
/**
 * Main optimization function
 */
async function runBasicOptimization() {
    console.log('🚀 Starting Basic RAN Optimization...');
    console.log('=====================================');
    try {
        // Initialize SDK with development configuration
        const config = {
            ...ran_optimization_sdk_1.DEFAULT_CONFIG,
            environment: 'development',
            agentDB: {
                ...ran_optimization_sdk_1.DEFAULT_CONFIG.agentDB,
                dbPath: '.agentdb/basic-example.db',
                enableQUICSync: false // Disable for basic example
            }
        };
        const sdk = new ran_optimization_sdk_1.RANOptimizationSDK(config);
        console.log('✅ SDK initialized with development configuration');
        // Initialize all components
        console.log('🔧 Initializing SDK components...');
        await sdk.initialize();
        console.log('✅ SDK components initialized successfully');
        // Execute RAN optimization
        console.log('⚡ Executing RAN performance optimization...');
        console.log('Input metrics:', JSON.stringify(basicRANMetrics, null, 2));
        const startTime = Date.now();
        const result = await sdk.optimizeRANPerformance(basicRANMetrics);
        const executionTime = Date.now() - startTime;
        console.log('📊 Optimization Results:');
        console.log('=====================');
        console.log(`✅ Success: ${result.success}`);
        console.log(`⏱️  Execution Time: ${result.executionTime}ms`);
        console.log(`🚀 Performance Gain: ${(result.performanceGain * 100).toFixed(1)}%`);
        console.log(`🤖 Agents Used: ${result.agentsUsed}`);
        if (result.success) {
            console.log('\\n🎯 Optimization Strategies Applied:');
            if (result.optimizations) {
                console.log(result.optimizations);
            }
            console.log('\\n💡 Performance Improvements:');
            console.log(`  • Energy Efficiency: +${(basicRANMetrics.energy_efficiency * result.performanceGain * 100).toFixed(1)}%`);
            console.log(`  • Mobility Performance: +${(basicRANMetrics.mobility_performance * result.performanceGain * 100).toFixed(1)}%`);
            console.log(`  • Coverage Quality: +${(basicRANMetrics.coverage_quality * result.performanceGain * 100).toFixed(1)}%`);
            console.log(`  • Capacity Utilization: +${(basicRANMetrics.capacity_utilization * result.performanceGain * 100).toFixed(1)}%`);
            console.log(`  • User Experience: +${(basicRANMetrics.user_experience * result.performanceGain * 100).toFixed(1)}%`);
        }
        // Run performance benchmark
        console.log('\\n📈 Running Performance Benchmark...');
        const benchmark = await sdk.runPerformanceBenchmark();
        console.log('Benchmark Results:');
        console.log('=================');
        console.log(`🎯 Overall Score: ${(benchmark.overall.score * 100).toFixed(1)}%`);
        console.log(`⏱️  Total Time: ${benchmark.overall.totalTime}ms`);
        console.log(`✅ Target Met: ${benchmark.overall.targetMet ? 'Yes' : 'No'}`);
        console.log('\\nVector Search Performance:');
        console.log(`  • Average Latency: ${benchmark.vectorSearch.avgLatency}ms`);
        console.log(`  • Target Met: ${benchmark.vectorSearch.target ? '✅' : '❌'}`);
        console.log(`  • Throughput: ${benchmark.vectorSearch.throughput.toFixed(0)} queries/sec`);
        console.log('\\nRecommendations:');
        benchmark.recommendations.forEach(rec => {
            console.log(`  • ${rec}`);
        });
        console.log('\\n🎉 Basic optimization completed successfully!');
    }
    catch (error) {
        console.error('❌ Optimization failed:', error.message);
        console.error('Stack trace:', error.stack);
        process.exit(1);
    }
}
exports.runBasicOptimization = runBasicOptimization;
/**
 * Demonstrate progressive skill discovery
 */
async function demonstrateSkillDiscovery() {
    console.log('\\n🧠 Demonstrating Progressive Skill Discovery...');
    console.log('==============================================');
    try {
        const sdk = new ran_optimization_sdk_1.RANOptimizationSDK({
            ...ran_optimization_sdk_1.DEFAULT_CONFIG,
            environment: 'development'
        });
        await sdk.initialize();
        const skillDiscovery = sdk['skillDiscovery'];
        // Level 1: Load metadata for all skills
        console.log('📚 Loading skill metadata (Level 1)...');
        const allSkills = await skillDiscovery.loadSkillMetadata();
        console.log(`✅ Loaded ${allSkills.length} skill metadata in minimal context`);
        // Display skill categories
        const categories = {};
        allSkills.forEach(skill => {
            categories[skill.category] = (categories[skill.category] || 0) + 1;
        });
        console.log('\\n📊 Skill Categories:');
        Object.entries(categories).forEach(([category, count]) => {
            console.log(`  • ${category}: ${count} skills`);
        });
        // Level 2: Load content for relevant skills
        console.log('\\n🔍 Finding relevant skills for current scenario...');
        const relevantSkills = await skillDiscovery.findRelevantSkills({
            metrics: basicRANMetrics,
            optimization_type: 'energy-efficiency',
            scenario: 'peak-hour-optimization'
        });
        console.log(`✅ Found ${relevantSkills.length} relevant skills:`);
        relevantSkills.forEach(skill => {
            console.log(`  • ${skill.name}: ${skill.description}`);
        });
        // Level 3: Load specific resources
        if (relevantSkills.length > 0) {
            console.log('\\n📄 Loading skill resources (Level 3)...');
            const firstSkill = relevantSkills[0];
            try {
                const resources = await skillDiscovery.loadSkillResource(firstSkill.name, 'examples/usage-patterns.md');
                console.log(`✅ Loaded resources for ${firstSkill.name}`);
            }
            catch (error) {
                console.log(`ℹ️  No additional resources found for ${firstSkill.name}`);
            }
        }
        console.log('\\n✨ Progressive skill discovery demonstration completed!');
    }
    catch (error) {
        console.error('❌ Skill discovery demo failed:', error.message);
    }
}
exports.demonstrateSkillDiscovery = demonstrateSkillDiscovery;
/**
 * Demonstrate memory coordination
 */
async function demonstrateMemoryCoordination() {
    console.log('\\n💾 Demonstrating Memory Coordination...');
    console.log('=====================================');
    try {
        const sdk = new ran_optimization_sdk_1.RANOptimizationSDK({
            ...ran_optimization_sdk_1.DEFAULT_CONFIG,
            environment: 'development'
        });
        await sdk.initialize();
        const memoryCoordinator = sdk['memoryCoordinator'];
        // Store architectural decision
        console.log('📝 Storing architectural decision...');
        await memoryCoordinator.storeDecision({
            id: 'demo-energy-strategy',
            title: 'Energy Optimization Strategy',
            context: 'Peak hour traffic optimization example',
            decision: 'Use adaptive energy saving based on traffic patterns',
            alternatives: [
                'Fixed energy saving schedule',
                'No energy optimization',
                'Reactive energy management'
            ],
            consequences: [
                '15% energy efficiency improvement',
                'Maintains service quality',
                'Adapts to changing conditions'
            ],
            confidence: 0.92,
            timestamp: Date.now()
        });
        console.log('✅ Architectural decision stored');
        // Share memory between agents
        console.log('\\n🔄 Sharing memory between agents...');
        await memoryCoordinator.shareMemory('energy-optimizer', 'mobility-manager', {
            optimization_patterns: [
                'peak_hour_energy_saving',
                'traffic_adaptive_scaling',
                'quality_preservation_strategy'
            ],
            performance_metrics: {
                energy_saved: '15%',
                quality_maintained: '98%',
                user_impact: 'minimal'
            },
            learned_strategies: [
                'gradual_power_reduction',
                'anticipatory_resource_allocation'
            ]
        }, 'high');
        console.log('✅ Memory shared between energy-optimizer and mobility-manager');
        // Retrieve agent context
        console.log('\\n📖 Retrieving agent context...');
        const agentContext = await memoryCoordinator.getContext('mobility-manager');
        console.log('✅ Agent context retrieved:');
        console.log(`  • Agent Type: ${agentContext.agentType}`);
        console.log(`  • Initialized: ${new Date(agentContext.initialized).toISOString()}`);
        console.log(`  • Memory Items: ${agentContext.memory.length}`);
        console.log('\\n💭 Memory coordination demonstration completed!');
    }
    catch (error) {
        console.error('❌ Memory coordination demo failed:', error.message);
    }
}
exports.demonstrateMemoryCoordination = demonstrateMemoryCoordination;
/**
 * Main execution function
 */
async function main() {
    console.log('🌟 Ericsson RAN Optimization SDK - Basic Example');
    console.log('================================================');
    console.log('This example demonstrates core SDK functionality including:');
    console.log('• Basic RAN optimization execution');
    console.log('• Progressive skill discovery');
    console.log('• Memory coordination patterns');
    console.log('• Performance benchmarking');
    console.log('');
    // Run basic optimization
    await runBasicOptimization();
    // Demonstrate advanced features
    await demonstrateSkillDiscovery();
    await demonstrateMemoryCoordination();
    console.log('\\n🎊 All examples completed successfully!');
    console.log('');
    console.log('Next steps:');
    console.log('1. Explore the integration guide: docs/SDK-Integration-Guide.md');
    console.log('2. Try advanced examples: examples/advanced-optimization.ts');
    console.log('3. Run integration tests: npm run test:integration');
    console.log('4. Deploy to production: npm run deploy:production');
}
// Execute if run directly
if (require.main === module) {
    main().catch(error => {
        console.error('💥 Example execution failed:', error);
        process.exit(1);
    });
}
//# sourceMappingURL=basic-optimization.js.map