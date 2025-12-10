/**
 * ENM CLI Batch Operations Framework - Usage Examples
 *
 * Comprehensive examples demonstrating how to use the batch operations framework
 * for Ericsson RAN configuration management with cognitive optimization.
 */

import {
  BatchOperationsManager,
  BatchOperationConfig,
  BatchExecutionContext,
  NodeCollection,
  NodePattern,
  ScopeFilter,
  RTBTemplate
} from '../core/types';

import { BatchOperationsManager as Manager } from '../core/BatchOperationsManager';

/**
 * Example 1: Basic Batch Operation with Simple Pattern Matching
 */
export async function example1_BasicBatchOperation(): Promise<void> {
  console.log('\n=== Example 1: Basic Batch Operation ===');

  // Create batch operations manager
  const batchManager = new Manager();

  // Define batch configuration
  const config: BatchOperationConfig = {
    name: 'Basic 4G Cell Configuration',
    description: 'Apply basic configuration to all 4G cells in Paris region',
    collection: {
      id: 'paris_4g_cells',
      name: 'Paris 4G Cells Collection',
      nodePatterns: [
        {
          id: 'paris_enb_pattern',
          type: 'wildcard',
          pattern: 'ERBS*',
          priority: 1,
          inclusions: ['PARIS'],
          exclusions: ['*TEST*']
        }
      ],
      metadata: { region: 'Paris', technology: '4G' },
      type: 'dynamic'
    },
    scopeFilters: [
      {
        id: 'active_nodes_only',
        type: 'sync_status',
        condition: {
          attribute: 'syncStatus',
          operator: 'eq',
          value: 'synchronized'
        },
        action: 'include',
        priority: 10
      },
      {
        id: 'exclude_maintenance',
        type: 'sync_status',
        condition: {
          attribute: 'status',
          operator: 'ne',
          value: 'maintenance'
        },
        action: 'exclude',
        priority: 9
      }
    ],
    template: {
      meta: {
        version: '1.0.0',
        author: ['RAN Optimization Team'],
        description: 'Basic 4G cell configuration template',
        priority: 50,
        source: 'RTB System'
      },
      custom: [],
      configuration: {
        EUtranCellFDD: {
          qRxLevMin: -130,
          qQualMin: -32,
          cellIndividualOffset: 3
        }
      }
    },
    options: {
      mode: 'parallel',
      maxConcurrency: 5,
      commandTimeout: 30,
      batchTimeout: 300,
      preview: false,
      force: false,
      dryRun: false,
      continueOnError: true,
      verbose: true,
      saveRollback: true
    },
    cognitiveSettings: {
      enabled: true,
      temporalDepth: 100,
      strangeLoopLevel: 1,
      enableLearning: true,
      patternRecognitionLevel: 2,
      autonomousThreshold: 0.8,
      persistCognitiveState: true
    },
    errorHandling: {
      retry: {
        maxAttempts: 3,
        baseDelay: 1000,
        backoffMultiplier: 2,
        maxDelay: 30000,
        jitter: true,
        retryablePatterns: ['timeout', 'connection', 'network'],
        nonRetryablePatterns: ['authentication', 'authorization', 'not found']
      },
      fallback: [
        {
          id: 'alternative_command',
          type: 'alternative_command',
          triggerConditions: ['timeout'],
          config: { alternativeCommand: 'cmedit set --timeout 60' },
          priority: 8
        },
        {
          id: 'manual_intervention',
          type: 'manual_intervention',
          triggerConditions: ['authentication_error'],
          config: { escalateToAdmin: true },
          priority: 10
        }
      ],
      classification: {
        critical: ['authentication', 'authorization', 'security'],
        warning: ['timeout', 'connection'],
        informational: ['completed', 'success'],
        temporary: ['timeout', 'network', 'connection'],
        permanent: ['not found', 'authentication']
      },
      recovery: [
        {
          id: 'network_recovery',
          type: 'restart',
          triggerConditions: ['network_error'],
          config: { networkReset: true },
          estimatedTime: 5000
        }
      ],
      notifications: {
        enabled: true,
        channels: [
          {
            type: 'email',
            config: { recipients: ['admin@ericsson.com'] },
            levels: ['critical', 'warning']
          }
        ],
        thresholds: [
          {
            metric: 'error_rate',
            threshold: 0.1,
            operator: 'gt',
            level: 'critical'
          }
        ]
      }
    },
    monitoring: {
      enabled: true,
      metrics: [
        {
          id: 'execution_time',
          name: 'Command Execution Time',
          type: 'timer',
          unit: 'milliseconds',
          interval: 5000,
          tags: { component: 'batch_operations' }
        },
        {
          id: 'success_rate',
          name: 'Operation Success Rate',
          type: 'gauge',
          unit: 'percentage',
          interval: 10000,
          tags: { component: 'batch_operations' }
        }
      ],
      alerts: [
        {
          id: 'high_error_rate',
          name: 'High Error Rate Alert',
          condition: {
            metric: 'error_rate',
            operator: 'gt',
            threshold: 0.05,
            window: 60000,
            minDataPoints: 5
          },
          actions: [
            {
              type: 'notification',
              config: { message: 'High error rate detected in batch operations' },
              delay: 0
            }
          ],
          severity: 'high'
        }
      ],
      thresholds: [
        {
          metric: 'duration',
          warning: 10000,
          critical: 30000,
          type: 'duration'
        }
      ],
      retention: {
        rawData: 7,
        aggregatedData: 30,
        alertData: 90,
        auditLog: 365
      }
    }
  };

  // Create execution context
  const context: BatchExecutionContext = {
    batchId: `batch_${Date.now()}`,
    timestamp: new Date(),
    userId: 'ran_operator_001',
    sessionId: 'session_12345',
    consciousnessLevel: 'enhanced',
    environment: 'staging',
    region: 'Paris',
    networkSlice: 'default'
  };

  try {
    // Execute batch operation
    console.log('Starting batch operation...');
    const result = await batchManager.executeBatchOperation(config, context);

    // Display results
    console.log('\nBatch Operation Results:');
    console.log(`Status: ${result.status}`);
    console.log(`Total Nodes: ${result.statistics.totalNodes}`);
    console.log(`Successful Nodes: ${result.statistics.successfulNodes}`);
    console.log(`Failed Nodes: ${result.statistics.failedNodes}`);
    console.log(`Success Rate: ${(result.statistics.successRate * 100).toFixed(2)}%`);
    console.log(`Total Duration: ${result.statistics.totalDuration}ms`);

    // Display performance metrics
    console.log('\nPerformance Metrics:');
    console.log(`Commands/Second: ${result.metrics.throughput.commandsPerSecond.toFixed(2)}`);
    console.log(`Average Command Latency: ${result.metrics.latency.averageCommandLatency.toFixed(2)}ms`);
    console.log(`Cognitive Optimizations Applied: ${result.metrics.cognitive.strangeLoopOptimizations}`);

    // Display errors (if any)
    if (result.errorSummary.totalErrors > 0) {
      console.log('\nError Summary:');
      console.log(`Total Errors: ${result.errorSummary.totalErrors}`);
      result.errorSummary.mostFrequentErrors.slice(0, 3).forEach(error => {
        console.log(`  - ${error.message} (${error.count} occurrences)`);
      });
    }

  } catch (error) {
    console.error('Batch operation failed:', error);
  }
}

/**
 * Example 2: Advanced Multi-Technology Configuration
 */
export async function example2_AdvancedMultiTechnology(): Promise<void> {
  console.log('\n=== Example 2: Advanced Multi-Technology Configuration ===');

  const batchManager = new Manager();

  // Complex collection with multiple patterns and cognitive selection
  const config: BatchOperationConfig = {
    name: '5G-4G Interoperability Configuration',
    description: 'Configure EN-DC parameters for 5G-4G interoperability',
    collection: {
      id: 'multi_tech_nodes',
      name: 'Multi-Technology Nodes',
      nodePatterns: [
        {
          id: '5g_gnb_pattern',
          type: 'wildcard',
          pattern: 'GNB*',
          priority: 10
        },
        {
          id: '4g_enb_5g_capable',
          type: 'cognitive',
          pattern: 'high capacity 5g capable 4g nodes',
          priority: 8
        },
        {
          id: 'specific_locations',
          type: 'list',
          pattern: 'ERBS001,ERBS002,ERBS003,GNB001,GNB002',
          priority: 9
        }
      ],
      metadata: {
        technologies: ['4G', '5G'],
        feature: 'EN-DC',
        complexity: 'high'
      },
      type: 'computed'
    },
    scopeFilters: [
      {
        id: 'high_capacity_only',
        type: 'custom',
        condition: {
          attribute: 'capacity',
          operator: 'eq',
          value: 'high'
        },
        action: 'prioritize',
        priority: 10
      },
      {
        id: 'recent_versions',
        type: 'version',
        condition: {
          attribute: 'version',
          operator: 'gte',
          value: '21B'
        },
        action: 'include',
        priority: 8
      },
      {
        id: 'low_latency_nodes',
        type: 'performance',
        condition: {
          attribute: 'latency',
          operator: 'lt',
          value: '10'
        },
        action: 'prioritize',
        priority: 9
      }
    ],
    template: {
      meta: {
        version: '2.0.0',
        author: ['5G Optimization Team'],
        description: 'EN-DC configuration template for 5G-4G interoperability',
        priority: 75,
        source: 'Advanced RTB System'
      },
      custom: [
        {
          name: 'calculateENBParameters',
          args: ['nodeType', 'capacity'],
          body: [
            'if (nodeType === "ENB" && capacity === "high") {',
            '  return {',
            '    nrEventB1Threshold: -110,',
            '    nrEventB1Hysteresis: 2,',
            '    endcEnabled: true',
            '    splitBearerSupport: true',
            '  };',
            '}',
            'return { endcEnabled: false };'
          ]
        }
      ],
      configuration: {
        ENBFunction: {
          $eval: 'calculateENBParameters(nodeType, capacity)'
        },
        NRCellCU: {
          nrdcEnabled: true,
          nrdcCapacitySharing: true,
          maxAggregatedBandwidth: 400
        }
      }
    },
    options: {
      mode: 'adaptive',
      maxConcurrency: 3, // Lower for complex operations
      commandTimeout: 60,
      batchTimeout: 600,
      preview: false,
      force: false,
      dryRun: false,
      continueOnError: true,
      verbose: true,
      saveRollback: true
    },
    cognitiveSettings: {
      enabled: true,
      temporalDepth: 500, // Higher temporal reasoning for complex scenarios
      strangeLoopLevel: 3,
      enableLearning: true,
      patternRecognitionLevel: 5,
      autonomousThreshold: 0.9,
      persistCognitiveState: true
    },
    errorHandling: {
      retry: {
        maxAttempts: 5,
        baseDelay: 2000,
        backoffMultiplier: 1.5,
        maxDelay: 60000,
        jitter: true,
        retryablePatterns: ['timeout', 'connection', 'synchronization', 'resource'],
        nonRetryablePatterns: ['authentication', 'security_violation']
      },
      fallback: [
        {
          id: 'stepwise_configuration',
          type: 'different_template',
          triggerConditions: ['configuration_error'],
          config: { alternativeTemplate: 'basic_endc_template' },
          priority: 9
        },
        {
          id: 'rollback_and_retry',
          type: 'rollback',
          triggerConditions: ['synchronization_error'],
          config: { fullRollback: true },
          priority: 7
        }
      ],
      classification: {
        critical: ['security', 'authentication', 'system_failure'],
        warning: ['timeout', 'synchronization', 'resource_unavailable'],
        informational: ['success', 'completed'],
        temporary: ['timeout', 'connection', 'synchronization'],
        permanent: ['authentication', 'security', 'not_found']
      },
      recovery: [
        {
          id: 'advanced_recovery',
          type: 'compensate',
          triggerConditions: ['partial_failure'],
          config: { compensationStrategy: 'graceful_degradation' },
          estimatedTime: 10000
        }
      ],
      notifications: {
        enabled: true,
        channels: [
          {
            type: 'slack',
            config: { webhook: 'https://hooks.slack.com/...', channel: '#ran-alerts' },
            levels: ['critical', 'high', 'warning']
          },
          {
            type: 'email',
            config: { recipients: ['5g-team@ericsson.com'] },
            levels: ['critical']
          }
        ],
        thresholds: [
          {
            metric: 'failure_rate',
            threshold: 0.15,
            operator: 'gt',
            level: 'critical'
          }
        ]
      }
    },
    monitoring: {
      enabled: true,
      metrics: [
        {
          id: 'cognitive_performance',
          name: 'Cognitive Optimization Performance',
          type: 'gauge',
          unit: 'score',
          interval: 15000,
          tags: { type: 'cognitive' }
        },
        {
          id: 'temporal_analysis_depth',
          name: 'Temporal Analysis Depth',
          type: 'gauge',
          unit: 'multiplier',
          interval: 30000,
          tags: { type: 'temporal' }
        }
      ],
      alerts: [
        {
          id: 'cognitive_performance_degradation',
          name: 'Cognitive Performance Degradation',
          condition: {
            metric: 'cognitive_performance',
            operator: 'lt',
            threshold: 0.7,
            window: 120000,
            minDataPoints: 8
          },
          actions: [
            {
              type: 'webhook',
              config: { url: 'https://api.monitoring.com/alerts' },
              delay: 1000
            }
          ],
          severity: 'medium'
        }
      ],
      thresholds: [
        {
          metric: 'cognitive_optimization_time',
          warning: 5000,
          critical: 15000,
          type: 'duration'
        }
      ],
      retention: {
        rawData: 14,
        aggregatedData: 90,
        alertData: 180,
        auditLog: 730
      }
    }
  };

  const context: BatchExecutionContext = {
    batchId: `multi_tech_batch_${Date.now()}`,
    timestamp: new Date(),
    userId: '5g_specialist_002',
    sessionId: 'session_5g_67890',
    consciousnessLevel: 'maximum',
    environment: 'production',
    region: 'Paris-London',
    networkSlice: 'enterprise'
  };

  try {
    console.log('Starting advanced multi-technology batch operation...');
    const result = await batchManager.executeBatchOperation(config, context);

    console.log('\nAdvanced Multi-Technology Results:');
    console.log(`Status: ${result.status}`);
    console.log(`Total Nodes: ${result.statistics.totalNodes}`);
    console.log(`Success Rate: ${(result.statistics.successRate * 100).toFixed(2)}%`);
    console.log(`Cognitive Insights: ${JSON.stringify(batchManager.getCognitiveInsights(context.batchId), null, 2)}`);

  } catch (error) {
    console.error('Advanced batch operation failed:', error);
  }
}

/**
 * Example 3: Preview and Validation Mode
 */
export async function example3_PreviewValidationMode(): Promise<void> {
  console.log('\n=== Example 3: Preview and Validation Mode ===');

  const batchManager = new Manager();

  const config: BatchOperationConfig = {
    name: 'Configuration Preview - Urban Cells',
    description: 'Preview configuration changes for urban cells before actual deployment',
    collection: {
      id: 'urban_cells_preview',
      name: 'Urban Cells Preview Collection',
      nodePatterns: [
        {
          id: 'urban_pattern',
          type: 'regex',
          pattern: '.*(URBAN|CITY|DOWNTOWN).*',
          priority: 10
        }
      ],
      metadata: { deploymentMode: 'preview', cellType: 'urban' },
      type: 'static'
    },
    scopeFilters: [
      {
        id: 'active_only',
        type: 'sync_status',
        condition: {
          attribute: 'status',
          operator: 'eq',
          value: 'active'
        },
        action: 'include',
        priority: 10
      }
    ],
    template: {
      meta: {
        version: '1.5.0',
        author: ['Urban Optimization Team'],
        description: 'Urban cell optimization template',
        priority: 60,
        source: 'Urban RTB System'
      },
      custom: [],
      configuration: {
        EUtranCellFDD: {
          qRxLevMin: -128,
          qQualMin: -30,
          cellIndividualOffset: 2,
          ul256qamEnabled: true
        }
      }
    },
    options: {
      mode: 'sequential',
      maxConcurrency: 1,
      commandTimeout: 30,
      batchTimeout: 180,
      preview: true, // Enable preview mode
      force: false,
      dryRun: true, // Enable dry run mode
      continueOnError: true,
      verbose: true,
      saveRollback: true
    },
    cognitiveSettings: {
      enabled: false, // Disable cognitive for preview
      temporalDepth: 1,
      strangeLoopLevel: 0,
      enableLearning: false,
      patternRecognitionLevel: 1,
      autonomousThreshold: 0.5,
      persistCognitiveState: false
    },
    errorHandling: {
      retry: {
        maxAttempts: 1,
        baseDelay: 500,
        backoffMultiplier: 1,
        maxDelay: 2000,
        jitter: false,
        retryablePatterns: [],
        nonRetryablePatterns: ['*']
      },
      fallback: [],
      classification: {
        critical: [],
        warning: [],
        informational: ['*'],
        temporary: [],
        permanent: ['*']
      },
      recovery: [],
      notifications: {
        enabled: false,
        channels: [],
        thresholds: []
      }
    },
    monitoring: {
      enabled: false,
      metrics: [],
      alerts: [],
      thresholds: [],
      retention: {
        rawData: 1,
        aggregatedData: 1,
        alertData: 1,
        auditLog: 1
      }
    }
  };

  const context: BatchExecutionContext = {
    batchId: `preview_batch_${Date.now()}`,
    timestamp: new Date(),
    userId: 'validation_operator_003',
    sessionId: 'session_preview_11111',
    consciousnessLevel: 'basic',
    environment: 'staging',
    region: 'Urban'
  };

  try {
    console.log('Starting preview batch operation...');
    const result = await batchManager.executeBatchOperation(config, context);

    console.log('\nPreview Results:');
    console.log(`Status: ${result.status}`);
    console.log(`Total Nodes: ${result.statistics.totalNodes}`);
    console.log(`Preview Commands Generated: ${result.statistics.totalCommands}`);

    // Display audit information
    console.log('\nAudit Trail:');
    result.audit.entries.slice(0, 5).forEach(entry => {
      console.log(`  [${entry.timestamp.toISOString()}] ${entry.eventType}: ${entry.description}`);
    });

  } catch (error) {
    console.error('Preview batch operation failed:', error);
  }
}

/**
 * Example 4: Real-time Monitoring and Progress Tracking
 */
export async function example4_RealTimeMonitoring(): Promise<void> {
  console.log('\n=== Example 4: Real-time Monitoring and Progress Tracking ===');

  const batchManager = new Manager();

  const config: BatchOperationConfig = {
    name: 'Large Scale Deployment with Monitoring',
    description: 'Deploy configuration to large number of nodes with real-time monitoring',
    collection: {
      id: 'large_scale_deployment',
      name: 'Large Scale Deployment Collection',
      nodePatterns: [
        {
          id: 'massive_pattern',
          type: 'wildcard',
          pattern: '*',
          priority: 1
        }
      ],
      metadata: { scale: 'large', monitoring: 'real-time' },
      type: 'dynamic'
    },
    scopeFilters: [
      {
        id: 'filter_active',
        type: 'sync_status',
        condition: {
          attribute: 'status',
          operator: 'eq',
          value: 'active'
        },
        action: 'include',
        priority: 10
      }
    ],
    template: {
      meta: {
        version: '1.0.0',
        author: ['Operations Team'],
        description: 'Large scale deployment template',
        priority: 40,
        source: 'RTB System'
      },
      custom: [],
      configuration: {
        basicParameters: {
          optimizationEnabled: true
        }
      }
    },
    options: {
      mode: 'adaptive',
      maxConcurrency: 10,
      commandTimeout: 45,
      batchTimeout: 900,
      preview: false,
      force: false,
      dryRun: false,
      continueOnError: true,
      verbose: true,
      saveRollback: true
    },
    cognitiveSettings: {
      enabled: true,
      temporalDepth: 200,
      strangeLoopLevel: 2,
      enableLearning: true,
      patternRecognitionLevel: 3,
      autonomousThreshold: 0.85,
      persistCognitiveState: true
    },
    errorHandling: {
      retry: {
        maxAttempts: 3,
        baseDelay: 1500,
        backoffMultiplier: 2,
        maxDelay: 45000,
        jitter: true,
        retryablePatterns: ['timeout', 'connection', 'temporary'],
        nonRetryablePatterns: ['authentication', 'permanent']
      },
      fallback: [
        {
          id: 'skip_and_continue',
          type: 'skip',
          triggerConditions: ['non_critical_error'],
          config: {},
          priority: 5
        }
      ],
      classification: {
        critical: ['system_failure', 'security'],
        warning: ['timeout', 'connection'],
        informational: ['success', 'completed'],
        temporary: ['timeout', 'connection'],
        permanent: ['authentication', 'not_found']
      },
      recovery: [
        {
          id: 'basic_recovery',
          type: 'restart',
          triggerConditions: ['temporary_error'],
          config: {},
          estimatedTime: 3000
        }
      ],
      notifications: {
        enabled: true,
        channels: [
          {
            type: 'webhook',
            config: { url: 'https://monitoring.example.com/webhook' },
            levels: ['critical', 'warning']
          }
        ],
        thresholds: [
          {
            metric: 'progress_stall',
            threshold: 0.05,
            operator: 'lt',
            level: 'warning'
          }
        ]
      }
    },
    monitoring: {
      enabled: true,
      metrics: [
        {
          id: 'progress_percentage',
          name: 'Batch Progress',
          type: 'gauge',
          unit: 'percentage',
          interval: 5000,
          tags: { phase: 'execution' }
        },
        {
          id: 'nodes_per_second',
          name: 'Nodes Processing Rate',
          type: 'counter',
          unit: 'nodes/sec',
          interval: 10000,
          tags: { performance: 'throughput' }
        },
        {
          id: 'error_rate',
          name: 'Error Rate',
          type: 'gauge',
          unit: 'percentage',
          interval: 15000,
          tags: { quality: 'errors' }
        }
      ],
      alerts: [
        {
          id: 'progress_stalled',
          name: 'Progress Stalled Alert',
          condition: {
            metric: 'progress_percentage',
            operator: 'lt',
            threshold: 5,
            window: 300000,
            minDataPoints: 60
          },
          actions: [
            {
              type: 'notification',
              config: { message: 'Batch operation progress has stalled' },
              delay: 0
            }
          ],
          severity: 'warning'
        }
      ],
      thresholds: [
        {
          metric: 'nodes_per_second',
          warning: 0.1,
          critical: 0.01,
          type: 'rate'
        }
      ],
      retention: {
        rawData: 7,
        aggregatedData: 30,
        alertData: 90,
        auditLog: 365
      }
    }
  };

  const context: BatchExecutionContext = {
    batchId: `monitoring_batch_${Date.now()}`,
    timestamp: new Date(),
    userId: 'operations_team_004',
    sessionId: 'session_monitoring_22222',
    consciousnessLevel: 'enhanced',
    environment: 'production',
    region: 'National'
  };

  try {
    console.log('Starting large scale deployment with monitoring...');

    // Start the batch operation in background
    const batchPromise = batchManager.executeBatchOperation(config, context);

    // Monitor progress in real-time
    const monitoringInterval = setInterval(() => {
      const progress = batchManager.getBatchProgress(context.batchId);
      if (progress) {
        console.log(`\rProgress: ${progress.overallProgress.toFixed(1)}% | Phase: ${progress.currentPhase} | Activity: ${progress.currentActivity}`);

        if (progress.recentErrors.length > 0) {
          console.log(`Recent Errors: ${progress.recentErrors.slice(-3).join(', ')}`);
        }

        if (progress.overallProgress >= 100 || ['completed', 'failed', 'cancelled'].includes(progress.currentPhase)) {
          clearInterval(monitoringInterval);
        }
      }
    }, 2000);

    // Wait for completion
    const result = await batchPromise;
    clearInterval(monitoringInterval);

    console.log('\n\nLarge Scale Deployment Results:');
    console.log(`Final Status: ${result.status}`);
    console.log(`Success Rate: ${(result.statistics.successRate * 100).toFixed(2)}%`);
    console.log(`Total Duration: ${(result.statistics.totalDuration / 1000).toFixed(2)} seconds`);
    console.log(`Parallelism Efficiency: ${(result.statistics.parallelismEfficiency * 100).toFixed(2)}%`);

    // Display performance metrics
    console.log('\nPerformance Summary:');
    console.log(`Throughput: ${result.metrics.throughput.commandsPerSecond.toFixed(2)} commands/sec`);
    console.log(`Average Latency: ${result.metrics.latency.averageCommandLatency.toFixed(2)}ms`);
    console.log(`P99 Latency: ${result.metrics.latency.p99CommandLatency.toFixed(2)}ms`);

  } catch (error) {
    console.error('Large scale deployment failed:', error);
  }
}

/**
 * Main function to run all examples
 */
export async function runAllExamples(): Promise<void> {
  console.log('ENM CLI Batch Operations Framework - Usage Examples');
  console.log('=====================================================');

  await example1_BasicBatchOperation();
  await example2_AdvancedMultiTechnology();
  await example3_PreviewValidationMode();
  await example4_RealTimeMonitoring();

  console.log('\nAll examples completed!');
}

// Export the main runner function
export default runAllExamples;