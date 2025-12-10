"use strict";
/**
 * ReasoningBank Adaptive Learning Demo
 * Demonstrates the complete adaptive learning workflow with AgentDB integration
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.demonstrateReasoningBankIntegration = void 0;
const ReasoningBankAgentDBIntegration_1 = require("../src/reasoningbank/core/ReasoningBankAgentDBIntegration");
async function demonstrateReasoningBankIntegration() {
    console.log('🧠 ReasoningBank Adaptive Learning Demo');
    console.log('=====================================\n');
    // 1. Configure ReasoningBank for RAN optimization
    const config = {
        agentDB: {
            swarmId: 'demo-ran-optimization-swarm',
            syncProtocol: 'QUIC',
            persistenceEnabled: false,
            crossAgentLearning: true,
            vectorDimension: 512,
            indexingStrategy: 'HNSW',
            quantization: { enabled: true, bits: 8 }
        },
        adaptiveLearning: {
            learningRate: 0.01,
            adaptationThreshold: 0.7,
            trajectoryLength: 1000,
            patternExtractionEnabled: true,
            crossDomainTransfer: true
        },
        temporalReasoning: {
            subjectiveTimeExpansion: 1000,
            temporalPatternWindow: 300000,
            causalInferenceEnabled: true,
            predictionHorizon: 600000 // 10 minutes
        },
        performance: {
            cacheEnabled: true,
            quantizationEnabled: true,
            parallelProcessingEnabled: true,
            memoryCompressionEnabled: true
        }
    };
    // 2. Initialize ReasoningBank
    console.log('🚀 Initializing ReasoningBank...');
    const reasoningBank = new ReasoningBankAgentDBIntegration_1.ReasoningBankAgentDBIntegration(config);
    await reasoningBank.initialize();
    console.log('✅ ReasoningBank initialized successfully\n');
    // 3. Execute adaptive RL training
    console.log('🔄 Executing Adaptive RL Training...');
    console.log('----------------------------------------');
    const startTime = performance.now();
    const adaptivePolicy = await reasoningBank.adaptiveRLTraining();
    const endTime = performance.now();
    // 4. Display results
    console.log('📊 Adaptive Training Results:');
    console.log(`   Policy ID: ${adaptivePolicy.id}`);
    console.log(`   Domain: ${adaptivePolicy.domain}`);
    console.log(`   Version: ${adaptivePolicy.version}`);
    console.log(`   Execution Time: ${(endTime - startTime).toFixed(2)}ms`);
    console.log('\n📈 Performance Metrics:');
    console.log(`   Overall Score: ${(adaptivePolicy.performance_metrics.overall_score * 100).toFixed(1)}%`);
    console.log(`   Accuracy: ${(adaptivePolicy.performance_metrics.accuracy * 100).toFixed(1)}%`);
    console.log(`   Efficiency: ${(adaptivePolicy.performance_metrics.efficiency * 100).toFixed(1)}%`);
    console.log(`   Robustness: ${(adaptivePolicy.performance_metrics.robustness * 100).toFixed(1)}%`);
    console.log('\n🤝 Cross-Agent Learning:');
    console.log(`   Applicability: ${(adaptivePolicy.cross_agent_applicability * 100).toFixed(1)}%`);
    console.log(`   Transfer Success: ${(adaptivePolicy.performance_metrics.cross_agent_transfer_success * 100).toFixed(1)}%`);
    console.log('\n⏰ Temporal Analysis:');
    console.log(`   Prediction Accuracy: ${(adaptivePolicy.performance_metrics.temporal_prediction_accuracy * 100).toFixed(1)}%`);
    console.log(`   Temporal Patterns: ${adaptivePolicy.temporal_patterns.length}`);
    // 5. Demonstrate policy optimization
    console.log('\n⚡ Policy Optimization');
    console.log('---------------------');
    const optimizationResult = await reasoningBank.optimizePolicyStorage(adaptivePolicy);
    console.log('📊 Optimization Results:');
    console.log(`   Type: ${optimizationResult.optimization_type}`);
    console.log(`   Success: ${optimizationResult.success ? 'YES' : 'NO'}`);
    console.log(`   Improvement: ${optimizationResult.improvement_percentage.toFixed(1)}%`);
    console.log(`   Processing Time: ${optimizationResult.optimization_time.toFixed(2)}ms`);
    if (optimizationResult.success) {
        console.log(`   Memory Saved: ${optimizationResult.memory_savings.toFixed(2)}MB`);
        console.log(`   Performance Gain: ${optimizationResult.performance_after.overall_performance.search_performance.queries_per_second.toFixed(0)} QPS`);
    }
    // 6. Demonstrate knowledge distillation
    console.log('\n🗜️ Knowledge Distillation');
    console.log('-------------------------');
    // Create sample patterns for distillation
    const samplePatterns = [
        {
            pattern_id: 'pattern_1',
            type: 'energy_optimization',
            confidence: 0.85,
            performance_impact: 0.12,
            temporal_signature: [0.1, 0.3, 0.2, 0.4, 0.3]
        },
        {
            pattern_id: 'pattern_2',
            type: 'mobility_optimization',
            confidence: 0.78,
            performance_impact: 0.08,
            temporal_signature: [0.2, 0.4, 0.3, 0.5, 0.4]
        },
        {
            pattern_id: 'pattern_3',
            type: 'coverage_optimization',
            confidence: 0.92,
            performance_impact: 0.15,
            temporal_signature: [0.3, 0.5, 0.4, 0.6, 0.5]
        }
    ];
    const distillationResult = await reasoningBank.distillPatterns(samplePatterns, {
        compression_method: 'knowledge_compression',
        compression_ratio: 0.3,
        quality_preservation: 0.9
    });
    console.log('📊 Distillation Results:');
    console.log(`   Distilled Data ID: ${distillationResult.distilled_data.data_id}`);
    console.log(`   Data Type: ${distillationResult.distilled_data.data_type}`);
    console.log(`   Compression Achieved: ${distillationResult.compression_achieved.toFixed(2)}x`);
    console.log(`   Quality Preserved: ${(distillationResult.quality_preserved * 100).toFixed(1)}%`);
    console.log(`   Knowledge Retention: ${(distillationResult.knowledge_retention * 100).toFixed(1)}%`);
    console.log(`   Cross-Agent Applicability: ${(distillationResult.cross_agent_applicability * 100).toFixed(1)}%`);
    console.log(`   Memory Savings: ${distillationResult.memory_savings.toFixed(2)}MB`);
    console.log(`   Processing Time: ${distillationResult.distillation_time.toFixed(2)}ms`);
    // 7. Demonstrate search optimization
    console.log('\n🔍 Search Optimization');
    console.log('--------------------');
    const searchQueries = [
        'energy efficiency optimization strategies',
        'mobility handover patterns',
        'coverage hole detection',
        'capacity management techniques',
        'performance optimization approaches'
    ];
    console.log('📊 Search Results:');
    for (const query of searchQueries) {
        const searchStart = performance.now();
        const searchResult = await reasoningBank.optimizeSearchQuery(query, {
            context: 'ran_optimization',
            agent_types: ['ml_researcher', 'optimizer', 'analyst']
        });
        const searchEnd = performance.now();
        console.log(`   Query: "${query}"`);
        console.log(`     Search Time: ${(searchEnd - searchStart).toFixed(2)}ms`);
        console.log(`     Results: ${searchResult.results?.length || 0}`);
        console.log(`     Cache Hit: ${searchResult.cache_hit ? 'YES' : 'NO'}`);
        console.log(`     Search Method: ${searchResult.search_method}`);
    }
    // 8. Display comprehensive statistics
    console.log('\n📊 System Statistics');
    console.log('==================');
    const stats = await reasoningBank.getStatistics();
    console.log('🧠 ReasoningBank:');
    console.log(`   Active Policies: ${stats.reasoningbank.active_policies}`);
    console.log(`   Learning Patterns: ${stats.reasoningbank.learning_patterns}`);
    console.log(`   Cross-Agent Memories: ${stats.reasoningbank.cross_agent_memories}`);
    console.log(`   Initialized: ${stats.reasoningbank.is_initialized}`);
    console.log('\n💾 AgentDB Performance:');
    console.log(`   Total Memories: ${stats.agentdb.totalMemories}`);
    console.log(`   Shared Memories: ${stats.agentdb.sharedMemories}`);
    console.log(`   Learning Patterns: ${stats.agentdb.learningPatterns}`);
    console.log(`   Sync Status: ${stats.agentdb.syncStatus}`);
    console.log(`   Search Speed: ${stats.agentdb.performance.searchSpeed.toFixed(0)} queries/sec`);
    console.log(`   Sync Latency: ${stats.agentdb.performance.syncLatency.toFixed(2)}ms`);
    console.log('\n⚡ Performance Optimization:');
    console.log(`   Cache Enabled: ${stats.performance_optimization.cache_enabled}`);
    console.log(`   Quantization Enabled: ${stats.performance_optimization.quantization_enabled}`);
    console.log(`   Parallel Processing: ${stats.performance_optimization.parallel_processing_enabled}`);
    console.log(`   Memory Compression: ${stats.performance_optimization.memory_compression_enabled}`);
    console.log(`   HNSW Indexing: ${stats.performance_optimization.hnsw_indexing_enabled}`);
    console.log('\n🔍 Search Performance:');
    console.log(`   Average Search Time: ${stats.performance_optimization.search_performance.average_search_time.toFixed(2)}ms`);
    console.log(`   Queries Per Second: ${stats.performance_optimization.search_performance.queries_per_second.toFixed(0)}`);
    console.log(`   Search Accuracy: ${(stats.performance_optimization.search_performance.search_accuracy * 100).toFixed(1)}%`);
    console.log(`   Cache Hit Rate: ${(stats.performance_optimization.search_performance.cache_hit_rate * 100).toFixed(1)}%`);
    console.log('\n💾 Memory Performance:');
    console.log(`   Total Memory Usage: ${stats.performance_optimization.memory_performance.total_memory_usage.toFixed(1)}MB`);
    console.log(`   Memory Efficiency: ${(stats.performance_optimization.memory_performance.memory_efficiency * 100).toFixed(1)}%`);
    console.log(`   Memory Savings: ${stats.performance_optimization.memory_performance.quantization_memory_savings.toFixed(1)}MB`);
    console.log('\n📊 Optimization History:');
    console.log(`   Total Optimizations: ${stats.performance_optimization.optimization_history.total_optimizations}`);
    console.log(`   Successful Optimizations: ${stats.performance_optimization.optimization_history.successful_optimizations}`);
    console.log(`   Average Improvement: ${stats.performance_optimization.optimization_history.average_improvement.toFixed(1)}%`);
    console.log(`   Total Memory Saved: ${stats.performance_optimization.optimization_history.total_memory_savings.toFixed(1)}MB`);
    // 9. Demonstrate cross-agent knowledge transfer
    console.log('\n🤝 Cross-Agent Knowledge Transfer');
    console.log('===============================');
    const agentTypes = ['ml_researcher', 'coder', 'tester', 'optimizer', 'analyst'];
    console.log('📊 Cross-Agent Mappings:');
    for (let i = 0; i < agentTypes.length - 1; i++) {
        for (let j = i + 1; j < agentTypes.length; j++) {
            const mappings = reasoningBank.getCrossAgentMappings(agentTypes[i], agentTypes[j]);
            if (mappings.length > 0) {
                console.log(`   ${agentTypes[i]} → ${agentTypes[j]}: ${mappings.length} mappings`);
                mappings.forEach(mapping => {
                    console.log(`     Confidence: ${(mapping.mapping_confidence * 100).toFixed(1)}%, ` +
                        `Transfer Success: ${(mapping.transfer_success_rate * 100).toFixed(1)}%, ` +
                        `Adaptation Overhead: ${(mapping.adaptation_overhead * 100).toFixed(1)}%`);
                });
            }
        }
    }
    // 10. Performance benchmark
    console.log('\n🏃 Performance Benchmark');
    console.log('=====================');
    console.log('Running performance benchmarks...');
    const benchmarkIterations = 10;
    const benchmarkTimes = [];
    for (let i = 0; i < benchmarkIterations; i++) {
        const benchmarkStart = performance.now();
        await reasoningBank.adaptiveRLTraining();
        const benchmarkEnd = performance.now();
        benchmarkTimes.push(benchmarkEnd - benchmarkStart);
    }
    const averageTime = benchmarkTimes.reduce((sum, time) => sum + time, 0) / benchmarkTimes.length;
    const minTime = Math.min(...benchmarkTimes);
    const maxTime = Math.max(...benchmarkTimes);
    const standardDeviation = Math.sqrt(benchmarkTimes.reduce((sum, time) => sum + Math.pow(time - averageTime, 2), 0) / benchmarkIterations);
    console.log(`📊 Benchmark Results (${benchmarkIterations} iterations):`);
    console.log(`   Average Time: ${averageTime.toFixed(2)}ms`);
    console.log(`   Min Time: ${minTime.toFixed(2)}ms`);
    console.log(`   Max Time: ${maxTime.toFixed(2)}ms`);
    console.log(`   Standard Deviation: ${standardDeviation.toFixed(2)}ms`);
    console.log(`   Throughput: ${(1000 / averageTime).toFixed(2)} operations/second`);
    // 11. Cleanup
    console.log('\n🧹 Cleanup');
    console.log('=========');
    await reasoningBank.shutdown();
    console.log('✅ ReasoningBank shutdown complete');
    console.log('\n🎉 Demo Completed Successfully!');
    console.log('=====================================');
    console.log('Key Achievements:');
    console.log(`✅ Executed adaptive RL training with ${(adaptivePolicy.performance_metrics.overall_score * 100).toFixed(1)}% performance`);
    console.log(`✅ Achieved ${optimizationResult.improvement_percentage.toFixed(1)}% policy optimization`);
    console.log(`✅ Compressed knowledge with ${distillationResult.compression_achieved.toFixed(2)}x ratio`);
    console.log(`✅ Enabled ${(stats.performance_optimization.search_performance.queries_per_second).toFixed(0)} QPS search performance`);
    console.log(`✅ Saved ${stats.performance_optimization.optimization_history.total_memory_savings.toFixed(1)}MB through optimization`);
}
exports.demonstrateReasoningBankIntegration = demonstrateReasoningBankIntegration;
// Run the demo
if (require.main === module) {
    demonstrateReasoningBankIntegration()
        .then(() => {
        console.log('\n🎯 Demo execution completed successfully!');
        process.exit(0);
    })
        .catch((error) => {
        console.error('\n❌ Demo execution failed:', error);
        process.exit(1);
    });
}
//# sourceMappingURL=reasoningbank-adaptive-learning-demo.js.map