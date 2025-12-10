/**
 * Performance Thresholds and Targets
 *
 * Defines critical, warning, and optimal thresholds for all performance metrics
 */

import { MLPerformanceMetrics, SwarmPerformanceMetrics, PerformanceTargets } from './MLPerformanceMetrics';

export class PerformanceThresholds {
  public static readonly PERFORMANCE_TARGETS: PerformanceTargets = {
    reinforcementLearning: {
      trainingSpeed: 1.0, // <1ms sync
      convergenceRate: 0.95, // >95%
      policyAccuracy: 0.90 // >90%
    },
    causalInference: {
      discoverySpeed: 150, // 150x faster than baseline
      causalAccuracy: 0.85, // >85%
      predictionPrecision: 0.90 // >90%
    },
    dspyOptimization: {
      mobilityImprovement: 0.15, // 15% improvement
      handoverSuccess: 0.95, // >95%
      coverageOptimization: 0.80 // >80%
    },
    agentdbIntegration: {
      vectorSearchSpeed: 1.0, // <1ms
      memoryEfficiency: 0.97, // 32x reduction = 97% efficiency
      synchronizationLatency: 1.0 // <1ms QUIC sync
    },
    cognitiveConsciousness: {
      temporalExpansionRatio: 1000, // 1000x subjective time
      autonomousHealingEfficiency: 0.90, // >90%
      consciousnessEvolutionScore: 0.80 // >80%
    }
  };

  public static readonly CRITICAL_THRESHOLDS: Partial<MLPerformanceMetrics> = {
    reinforcementLearning: {
      trainingSpeed: 5.0, // >5ms is critical
      convergenceRate: 0.70, // <70% is critical
      policyAccuracy: 0.60, // <60% is critical
      rewardOptimization: 0.50 // <50% is critical
    },
    causalInference: {
      discoverySpeed: 50, // <50x faster is critical
      causalAccuracy: 0.60, // <60% is critical
      predictionPrecision: 0.70 // <70% is critical
    },
    dspyOptimization: {
      mobilityImprovement: 0.05, // <5% improvement is critical
      handoverSuccess: 0.80, // <80% is critical
      coverageOptimization: 0.60 // <60% is critical
    },
    agentdbIntegration: {
      vectorSearchSpeed: 10.0, // >10ms is critical
      memoryEfficiency: 0.50, // <50% efficiency is critical
      synchronizationLatency: 10.0 // >10ms is critical
    },
    cognitiveConsciousness: {
      temporalExpansionRatio: 100, // <100x is critical
      autonomousHealingEfficiency: 0.50, // <50% is critical
      consciousnessEvolutionScore: 0.40 // <40% is critical
    }
  };

  public static readonly WARNING_THRESHOLDS: Partial<MLPerformanceMetrics> = {
    reinforcementLearning: {
      trainingSpeed: 2.0, // >2ms is warning
      convergenceRate: 0.85, // <85% is warning
      policyAccuracy: 0.75, // <75% is warning
      rewardOptimization: 0.65 // <65% is warning
    },
    causalInference: {
      discoverySpeed: 100, // <100x faster is warning
      causalAccuracy: 0.75, // <75% is warning
      predictionPrecision: 0.80 // <80% is warning
    },
    dspyOptimization: {
      mobilityImprovement: 0.10, // <10% improvement is warning
      handoverSuccess: 0.90, // <90% is warning
      coverageOptimization: 0.70 // <70% is warning
    },
    agentdbIntegration: {
      vectorSearchSpeed: 2.0, // >2ms is warning
      memoryEfficiency: 0.75, // <75% efficiency is warning
      synchronizationLatency: 2.0 // >2ms is warning
    },
    cognitiveConsciousness: {
      temporalExpansionRatio: 500, // <500x is warning
      autonomousHealingEfficiency: 0.70, // <70% is warning
      consciousnessEvolutionScore: 0.60 // <60% is warning
    }
  };

  public static readonly OPTIMAL_THRESHOLDS: Partial<MLPerformanceMetrics> = {
    reinforcementLearning: {
      trainingSpeed: 0.5, // <0.5ms is optimal
      convergenceRate: 0.98, // >98% is optimal
      policyAccuracy: 0.95, // >95% is optimal
      rewardOptimization: 0.90 // >90% is optimal
    },
    causalInference: {
      discoverySpeed: 200, // >200x faster is optimal
      causalAccuracy: 0.92, // >92% is optimal
      predictionPrecision: 0.95 // >95% is optimal
    },
    dspyOptimization: {
      mobilityImprovement: 0.20, // >20% improvement is optimal
      handoverSuccess: 0.98, // >98% is optimal
      coverageOptimization: 0.90 // >90% is optimal
    },
    agentdbIntegration: {
      vectorSearchSpeed: 0.5, // <0.5ms is optimal
      memoryEfficiency: 0.99, // >99% efficiency is optimal
      synchronizationLatency: 0.5 // <0.5ms is optimal
    },
    cognitiveConsciousness: {
      temporalExpansionRatio: 1500, // >1500x is optimal
      autonomousHealingEfficiency: 0.95, // >95% is optimal
      consciousnessEvolutionScore: 0.90 // >90% is optimal
    }
  };

  public static evaluateMetric(
    metricName: string,
    currentValue: number,
    category: keyof MLPerformanceMetrics
  ): 'critical' | 'warning' | 'optimal' | 'normal' {
    const criticalThreshold = this.CRITICAL_THRESHOLDS[category]?.[metricName as keyof any];
    const warningThreshold = this.WARNING_THRESHOLDS[category]?.[metricName as keyof any];
    const optimalThreshold = this.OPTIMAL_THRESHOLDS[category]?.[metricName as keyof any];

    if (criticalThreshold !== undefined) {
      // For metrics where lower is better (like latency)
      if (metricName.includes('Speed') || metricName.includes('Latency')) {
        if (currentValue > criticalThreshold) return 'critical';
        if (currentValue > warningThreshold) return 'warning';
        if (currentValue < optimalThreshold) return 'optimal';
      }
      // For metrics where higher is better (like accuracy, efficiency)
      else {
        if (currentValue < criticalThreshold) return 'critical';
        if (currentValue < warningThreshold) return 'warning';
        if (currentValue > optimalThreshold) return 'optimal';
      }
    }

    return 'normal';
  }

  public static getThresholdValue(
    metricName: string,
    severity: 'critical' | 'warning' | 'optimal',
    category: keyof MLPerformanceMetrics
  ): number | undefined {
    const thresholds = {
      critical: this.CRITICAL_THRESHOLDS,
      warning: this.WARNING_THRESHOLDS,
      optimal: this.OPTIMAL_THRESHOLDS
    };

    return thresholds[severity][category]?.[metricName as keyof any] as number;
  }

  public static calculatePerformanceScore(metrics: MLPerformanceMetrics): number {
    let totalScore = 0;
    let metricCount = 0;

    for (const category of Object.keys(metrics) as Array<keyof MLPerformanceMetrics>) {
      const categoryMetrics = metrics[category] as any;

      for (const metricName of Object.keys(categoryMetrics)) {
        const currentValue = categoryMetrics[metricName];
        const target = this.PERFORMANCE_TARGETS[category]?.[metricName as keyof any];

        if (target !== undefined && typeof currentValue === 'number') {
          // Normalize to 0-1 scale
          let normalizedScore: number;

          if (metricName.includes('Speed') || metricName.includes('Latency')) {
            // Lower is better
            normalizedScore = Math.min(1, target / currentValue);
          } else {
            // Higher is better
            normalizedScore = Math.min(1, currentValue / target);
          }

          totalScore += normalizedScore;
          metricCount++;
        }
      }
    }

    return metricCount > 0 ? totalScore / metricCount : 0;
  }

  public static getRecommendations(
    metricName: string,
    currentValue: number,
    severity: 'critical' | 'warning',
    category: keyof MLPerformanceMetrics
  ): string[] {
    const recommendations: string[] = [];

    switch (category) {
      case 'reinforcementLearning':
        if (metricName === 'trainingSpeed') {
          recommendations.push(
            'Optimize neural network architecture for faster training',
            'Implement gradient accumulation and mixed precision training',
            'Increase computational resources or use GPU acceleration'
          );
        } else if (metricName === 'convergenceRate') {
          recommendations.push(
            'Adjust learning rate and optimization algorithm',
            'Implement curriculum learning techniques',
            'Increase training data quality and diversity'
          );
        }
        break;

      case 'causalInference':
        if (metricName === 'discoverySpeed') {
          recommendations.push(
            'Optimize causal graph algorithms with parallel processing',
            'Implement incremental causal discovery methods',
            'Use approximate causal inference for faster results'
          );
        } else if (metricName === 'causalAccuracy') {
          recommendations.push(
            'Increase sample size and data quality',
            'Implement ensemble causal discovery methods',
            'Use domain knowledge to constrain causal search space'
          );
        }
        break;

      case 'dspyOptimization':
        if (metricName === 'mobilityImprovement') {
          recommendations.push(
            'Enhance mobility prediction models with more features',
            'Implement adaptive handover algorithms',
            'Optimize cell selection and load balancing strategies'
          );
        } else if (metricName === 'handoverSuccess') {
          recommendations.push(
            'Improve signal quality prediction accuracy',
            'Implement predictive handover decision making',
            'Optimize handover timing and parameter thresholds'
          );
        }
        break;

      case 'agentdbIntegration':
        if (metricName === 'vectorSearchSpeed') {
          recommendations.push(
            'Optimize vector indexing with HNSW or IVF algorithms',
            'Implement query caching and result memoization',
            'Scale AgentDB cluster for better parallel processing'
          );
        } else if (metricName === 'synchronizationLatency') {
          recommendations.push(
            'Optimize QUIC protocol configuration for lower latency',
            'Implement delta synchronization to reduce data transfer',
            'Use compression for synchronization payloads'
          );
        }
        break;

      case 'cognitiveConsciousness':
        if (metricName === 'temporalExpansionRatio') {
          recommendations.push(
            'Optimize temporal reasoning algorithms for efficiency',
            'Implement hierarchical temporal processing',
            'Use WASM acceleration for temporal computations'
          );
        } else if (metricName === 'autonomousHealingEfficiency') {
          recommendations.push(
            'Enhance anomaly detection and root cause analysis',
            'Implement automated remediation workflows',
            'Improve predictive failure detection models'
          );
        }
        break;
    }

    return recommendations;
  }
}