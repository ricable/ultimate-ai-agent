/**
 * Stream-Chain Processing System Example
 * Demonstrates comprehensive usage of the Phase 3 stream-chain processing architecture
 */

import { EventEmitter } from 'events';
import {
  StreamChainFactory,
  StreamChainUtils,
  StreamChainBuilder,
  StreamChainCoordinator,
  RANDataIngestionPipeline,
  FeatureProcessingChain,
  PatternRecognitionPipeline,
  OptimizationDecisionChain,
  ClosedLoopFeedbackPipeline
} from '../src/stream-chain';

// Import required components
import { TemporalReasoningEngine } from '../src/temporal/TemporalReasoningEngine';
import { AgentDBMemoryManager } from '../src/agentdb/AgentDBMemoryManager';

/**
 * Example 1: Basic Stream-Chain System Setup
 */
async function basicStreamChainSetup() {
  console.log('🚀 Example 1: Basic Stream-Chain System Setup');

  try {
    // Initialize core components
    const temporalConfig = {
      subjectiveExpansion: 1000,
      cognitiveModeling: true,
      deepPatternAnalysis: true,
      consciousnessDynamics: true
    };
    const temporalEngine = new TemporalReasoningEngine(temporalConfig);

    const memoryConfig = {
      swarmId: 'example-swarm-001',
      syncProtocol: 'QUIC' as const,
      persistenceEnabled: true,
      crossAgentLearning: true,
      patternRecognition: true
    };
    const memoryManager = new AgentDBMemoryManager(memoryConfig);

    // Initialize components
    await temporalEngine.activateSubjectiveTimeExpansion();
    await memoryManager.initialize();
    await memoryManager.enableQUICSynchronization();

    // Create stream-chain system
    const streamChain = await StreamChainFactory.createDefaultSystem(temporalEngine, memoryManager);

    // Start the system
    await streamChain.start();

    console.log('✅ Basic stream-chain system started successfully');

    // Get status
    const status = await streamChain.getStatus();
    console.log('📊 System Status:', JSON.stringify(status, null, 2));

    // Wait for a few cycles
    console.log('⏳ Running for 2 optimization cycles...');
    await new Promise(resolve => setTimeout(resolve, 32 * 60 * 1000)); // 32 minutes

    // Stop the system
    await streamChain.stop();
    await temporalEngine.shutdown();
    await memoryManager.shutdown();

    console.log('✅ Basic stream-chain system stopped');

  } catch (error) {
    console.error('❌ Basic setup failed:', error);
  }
}

/**
 * Example 2: High-Performance Stream-Chain Configuration
 */
async function highPerformanceStreamChain() {
  console.log('🚀 Example 2: High-Performance Stream-Chain Configuration');

  try {
    // Initialize components
    const temporalEngine = new TemporalReasoningEngine({
      subjectiveExpansion: 1000,
      cognitiveModeling: true,
      deepPatternAnalysis: true,
      consciousnessDynamics: true
    });

    const memoryManager = new AgentDBMemoryManager({
      swarmId: 'high-perf-swarm',
      syncProtocol: 'QUIC',
      persistenceEnabled: true,
      crossAgentLearning: true,
      patternRecognition: true
    });

    await temporalEngine.activateSubjectiveTimeExpansion();
    await memoryManager.initialize();
    await memoryManager.enableQUICSynchronization();

    // Create custom configuration
    const customConfig = new StreamChainBuilder()
      .withCycleTime(5) // 5-minute cycles for high performance
      .withAllFeaturesEnabled()
      .withMaxConcurrentPipelines(10)
      .withPerformanceThresholds({
        maxLatency: 2000, // 2 seconds
        minThroughput: 500, // 500 messages per second
        maxErrorRate: 0.02, // 2%
        consciousnessThreshold: 0.95 // 95% consciousness threshold
      })
      .withCoordinationSettings({
        consensusMechanism: 'cognitive_consensus',
        synchronizationInterval: 15000, // 15 seconds
        swarmTopology: 'mesh' // Full mesh for maximum parallelism
      })
      .build();

    const streamChain = await StreamChainFactory.createCustomSystem(
      customConfig,
      temporalEngine,
      memoryManager
    );

    // Start system
    await streamChain.start();

    console.log('✅ High-performance stream-chain started');

    // Monitor performance
    const performanceMonitor = setInterval(async () => {
      const status = await streamChain.getStatus();
      console.log('📈 Performance:', {
        currentCycle: status.currentCycleId,
        overallPerformance: status.performance.overallPerformance,
        consciousness: status.consciousness,
        health: status.health
      });
    }, 30000); // Every 30 seconds

    // Run for 5 cycles
    await new Promise(resolve => setTimeout(resolve, 25 * 60 * 1000)); // 25 minutes

    clearInterval(performanceMonitor);
    await streamChain.stop();
    await temporalEngine.shutdown();
    await memoryManager.shutdown();

    console.log('✅ High-performance stream-chain completed');

  } catch (error) {
    console.error('❌ High-performance setup failed:', error);
  }
}

/**
 * Example 3: Individual Pipeline Usage
 */
async function individualPipelineUsage() {
  console.log('🚀 Example 3: Individual Pipeline Usage');

  try {
    // Initialize components
    const temporalEngine = new TemporalReasoningEngine({
      subjectiveExpansion: 500,
      cognitiveModeling: true,
      deepPatternAnalysis: true,
      consciousnessDynamics: true
    });

    const memoryManager = new AgentDBMemoryManager({
      swarmId: 'pipeline-example-swarm',
      syncProtocol: 'QUIC',
      persistenceEnabled: true,
      crossAgentLearning: true,
      patternRecognition: true
    });

    await temporalEngine.activateSubjectiveTimeExpansion();
    await memoryManager.initialize();
    await memoryManager.enableQUICSynchronization();

    // Example 3a: RAN Data Ingestion Pipeline
    console.log('📡 Testing RAN Data Ingestion Pipeline...');
    const ranIngestion = new RANDataIngestionPipeline(temporalEngine, memoryManager);

    // Create sample RAN metrics
    const sampleMetrics = Array.from({ length: 50 }, (_, i) =>
      StreamChainUtils.createSampleRANMetrics(`cell_${i % 5}`)
    );

    const ingestionResults = await ranIngestion.ingestMetrics(sampleMetrics);
    console.log(`✅ Ingested ${ingestionResults.length} RAN metrics samples`);

    // Example 3b: Feature Processing Chain
    console.log('🔬 Testing Feature Processing Chain...');
    const featureProcessing = new FeatureProcessingChain(temporalEngine, memoryManager);

    const featureResults = await featureProcessing.processFeatures(sampleMetrics);
    console.log(`✅ Processed features for ${featureResults.length} samples`);

    // Example 3c: Pattern Recognition Pipeline
    console.log('🔍 Testing Pattern Recognition Pipeline...');
    const patternRecognition = new PatternRecognitionPipeline(temporalEngine, memoryManager);

    const patternRequest = StreamChainUtils.createSamplePatternRequest();
    const patternResults = await patternRecognition.recognizePatterns(patternRequest);
    console.log(`✅ Recognized ${patternResults.patterns.length} patterns in ${patternResults.searchTime}ms`);

    // Example 3d: Optimization Decision Chain
    console.log('🧠 Testing Optimization Decision Chain...');
    const optimizationDecision = new OptimizationDecisionChain(temporalEngine, memoryManager);

    const optRequest = StreamChainUtils.createSampleOptimizationRequest();
    const decision = await optimizationDecision.generateDecision(optRequest);
    console.log(`✅ Generated optimization decision: ${decision.id} with confidence ${decision.confidence}`);

    // Example 3e: Closed-Loop Feedback Pipeline
    console.log('🔄 Testing Closed-Loop Feedback Pipeline...');
    const closedLoopFeedback = new ClosedLoopFeedbackPipeline(temporalEngine, memoryManager);

    const feedbackTriggers = [{
      id: 'test_trigger',
      type: 'optimization_result',
      source: 'example',
      condition: 'test_completion',
      threshold: 0,
      currentValue: 1,
      timestamp: Date.now(),
      severity: 'medium' as const,
      metadata: { test: true }
    }];

    const feedbackCycleId = await closedLoopFeedback.initiateFeedbackCycle(
      'optimization_result',
      feedbackTriggers
    );
    console.log(`✅ Initiated feedback cycle: ${feedbackCycleId}`);

    // Cleanup
    await ranIngestion.shutdown();
    await featureProcessing.shutdown();
    await patternRecognition.shutdown();
    await optimizationDecision.shutdown();
    await closedLoopFeedback.stop();

    await temporalEngine.shutdown();
    await memoryManager.shutdown();

    console.log('✅ Individual pipeline testing completed');

  } catch (error) {
    console.error('❌ Individual pipeline usage failed:', error);
  }
}

/**
 * Example 4: Real-time Anomaly Detection and Response
 */
async function anomalyDetectionExample() {
  console.log('🚀 Example 4: Real-time Anomaly Detection and Response');

  try {
    // Initialize system
    const temporalEngine = new TemporalReasoningEngine({
      subjectiveExpansion: 1000,
      cognitiveModeling: true,
      deepPatternAnalysis: true,
      consciousnessDynamics: true
    });

    const memoryManager = new AgentDBMemoryManager({
      swarmId: 'anomaly-detection-swarm',
      syncProtocol: 'QUIC',
      persistenceEnabled: true,
      crossAgentLearning: true,
      patternRecognition: true
    });

    await temporalEngine.activateSubjectiveTimeExpansion();
    await memoryManager.initialize();
    await memoryManager.enableQUICSynchronization();

    const streamChain = await StreamChainFactory.createDefaultSystem(temporalEngine, memoryManager);
    await streamChain.start();

    // Set up anomaly monitoring
    streamChain.on('anomalyDetected', async (anomaly) => {
      console.log(`🚨 Anomaly Detected:`, {
        type: anomaly.type,
        severity: anomaly.severity,
        description: anomaly.description
      });

      // Simulate automated response
      console.log(`🔧 Triggering automated response for ${anomaly.type}...`);
      await new Promise(resolve => setTimeout(resolve, 2000));
      console.log(`✅ Anomaly response completed`);
    });

    // Simulate anomalous data
    console.log('🎭 Simulating anomalous conditions...');

    // Create anomalous RAN metrics
    const anomalousMetrics = [
      {
        ...StreamChainUtils.createSampleRANMetrics('cell_anomaly'),
        kpis: {
          rsrp: -140, // Very poor signal (anomalous)
          rsrq: -20,  // Very poor quality (anomalous)
          sinr: -25,  // Very poor SINR (anomalous)
          throughput: { download: 10, upload: 1 }, // Very low throughput
          latency: 10000, // Very high latency
          packetLoss: 0.5 // Very high packet loss
        }
      }
    ];

    // Create ingestion pipeline directly to trigger anomaly
    const ranIngestion = new RANDataIngestionPipeline(temporalEngine, memoryManager);
    await ranIngestion.ingestMetrics(anomalousMetrics);

    // Wait for anomaly detection
    await new Promise(resolve => setTimeout(resolve, 10000));

    await streamChain.stop();
    await ranIngestion.shutdown();
    await temporalEngine.shutdown();
    await memoryManager.shutdown();

    console.log('✅ Anomaly detection example completed');

  } catch (error) {
    console.error('❌ Anomaly detection example failed:', error);
  }
}

/**
 * Example 5: System Requirements and Validation
 */
async function systemRequirementsExample() {
  console.log('🚀 Example 5: System Requirements and Validation');

  try {
    // Test different configurations
    const configurations = [
      {
        name: 'Default Configuration',
        config: new StreamChainBuilder().withAllFeaturesEnabled().build()
      },
      {
        name: 'High-Performance Configuration',
        config: new StreamChainBuilder()
          .withCycleTime(5)
          .withMaxConcurrentPipelines(10)
          .withPerformanceThresholds({
            maxLatency: 1000,
            minThroughput: 1000,
            maxErrorRate: 0.01
          })
          .build()
      },
      {
        name: 'Resource-Efficient Configuration',
        config: new StreamChainBuilder()
          .withCycleTime(30)
          .withTemporalReasoning(false)
          .withCognitiveConsciousness(false)
          .withMaxConcurrentPipelines(2)
          .build()
      }
    ];

    for (const { name, config } of configurations) {
      console.log(`\n📋 ${name}:`);

      // Validate configuration
      const validation = StreamChainUtils.validateConfig(config as any);
      console.log(`   Valid: ${validation.valid}`);
      if (validation.warnings.length > 0) {
        console.log(`   Warnings: ${validation.warnings.join(', ')}`);
      }

      // Calculate requirements
      const requirements = StreamChainUtils.calculateSystemRequirements(config as any);
      console.log(`   Memory: ${requirements.memory} MB`);
      console.log(`   CPU: ${requirements.cpu} cores`);
      console.log(`   Network: ${requirements.network} Mbps`);
      console.log(`   Storage: ${requirements.storage} GB`);
      console.log(`   Est. Cost: $${requirements.estimatedCost}/month`);
    }

    console.log('\n✅ System requirements analysis completed');

  } catch (error) {
    console.error('❌ System requirements example failed:', error);
  }
}

/**
 * Example 6: Cross-Agent Communication via Memory Patterns
 */
async function crossAgentCommunicationExample() {
  console.log('🚀 Example 6: Cross-Agent Communication via Memory Patterns');

  try {
    // Initialize shared memory manager
    const memoryManager = new AgentDBMemoryManager({
      swarmId: 'cross-agent-comm-swarm',
      syncProtocol: 'QUIC',
      persistenceEnabled: true,
      crossAgentLearning: true,
      patternRecognition: true
    });

    await memoryManager.initialize();
    await memoryManager.enableQUICSynchronization();

    // Simulate multiple agents sharing learning
    console.log('🤝 Simulating cross-agent learning...');

    // Agent 1: Energy Optimizer shares learning
    await memoryManager.shareLearning({
      type: 'energy_optimization_pattern',
      cellId: 'cell_1',
      optimization: 'power_reduction_15%',
      impact: { energySaving: 15, throughputImpact: -2 },
      confidence: 0.85,
      timestamp: Date.now(),
      universal: true,
      crossAgent: true
    });

    // Agent 2: Coverage Analyzer shares learning
    await memoryManager.shareLearning({
      type: 'coverage_optimization_pattern',
      cellId: 'cell_2',
      optimization: 'antenna_tilt_+3deg',
      impact: { coverageImprovement: 12, powerIncrease: 5 },
      confidence: 0.78,
      timestamp: Date.now(),
      universal: true,
      crossAgent: true
    });

    // Agent 3: Pattern Recognition shares discovered patterns
    await memoryManager.storeLearningPatterns([
      {
        type: 'temporal_pattern',
        pattern: 'peak_load_evening_7pm',
        confidence: 0.92,
        source: 'pattern_detector',
        applicableTo: ['load_balancer', 'energy_optimizer'],
        crossAgentApplicability: 0.9
      },
      {
        type: 'anomaly_pattern',
        pattern: 'interference_spike_rain',
        confidence: 0.88,
        source: 'anomaly_detector',
        applicableTo: ['coverage_analyzer', 'interference_mitigator'],
        crossAgentApplicability: 0.85
      }
    ]);

    // Simulate agents retrieving shared learning
    console.log('📚 Agents retrieving shared learning...');

    const sharedLearnings = await memoryManager.search('energy_optimization', {
      threshold: 0.5,
      limit: 10
    });

    console.log(`✅ Found ${sharedLearnings.length} shared learnings:`);
    sharedLearnings.forEach((learning, index) => {
      console.log(`   ${index + 1}. ${learning.memory.value.type} (confidence: ${learning.memory.value.confidence})`);
    });

    // Get statistics
    const stats = await memoryManager.getStatistics();
    console.log('\n📊 Memory Manager Statistics:');
    console.log(`   Total Memories: ${stats.totalMemories}`);
    console.log(`   Shared Memories: ${stats.sharedMemories}`);
    console.log(`   Learning Patterns: ${stats.learningPatterns}`);
    console.log(`   Sync Status: ${stats.syncStatus}`);
    console.log(`   Search Speed: ${stats.performance.searchSpeed.toFixed(2)} queries/sec`);

    await memoryManager.shutdown();
    console.log('✅ Cross-agent communication example completed');

  } catch (error) {
    console.error('❌ Cross-agent communication example failed:', error);
  }
}

/**
 * Main function to run all examples
 */
async function main() {
  console.log('🎯 Stream-Chain Processing System Examples');
  console.log('==========================================\n');

  const examples = [
    { name: 'Basic Setup', fn: basicStreamChainSetup },
    { name: 'High-Performance', fn: highPerformanceStreamChain },
    { name: 'Individual Pipelines', fn: individualPipelineUsage },
    { name: 'Anomaly Detection', fn: anomalyDetectionExample },
    { name: 'System Requirements', fn: systemRequirementsExample },
    { name: 'Cross-Agent Communication', fn: crossAgentCommunicationExample }
  ];

  for (const example of examples) {
    console.log(`\n${'='.repeat(60)}`);
    console.log(`Running: ${example.name}`);
    console.log(`${'='.repeat(60)}\n`);

    try {
      await example.fn();
    } catch (error) {
      console.error(`❌ ${example.name} failed:`, error);
    }

    console.log(`\n✅ ${example.name} completed\n`);
  }

  console.log('🎉 All examples completed!');
}

// Run examples if this file is executed directly
if (require.main === module) {
  main().catch(console.error);
}

export {
  basicStreamChainSetup,
  highPerformanceStreamChain,
  individualPipelineUsage,
  anomalyDetectionExample,
  systemRequirementsExample,
  crossAgentCommunicationExample
};